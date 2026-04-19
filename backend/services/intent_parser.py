"""
Intent parser – classifies natural-language questions into one of four intents.

Uses a Google Gemini API call to determine the user's intent and extract the
target metric, dimensions, aggregation, and optional time range.

Column schema (names + types + sample values) is injected so the model can
ground its extraction in the actual dataset. A deterministic post-processing
layer corrects common model mistakes before results reach the data engine.
"""

import json
import logging
import re

from google import genai

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Aggregation keyword map ───────────────────────────────────────────────────

_AGG_KEYWORD_MAP: dict[str, str] = {
    # longest phrases first so "number of" beats "number"
    "number of": "count", "how many": "count",
    "average": "mean", "avg": "mean", "mean": "mean",
    "total": "sum", "sum": "sum",
    "count": "count",
    "unique": "nunique", "distinct": "nunique",
    "minimum": "min", "smallest": "min", "lowest": "min",
    "maximum": "max", "largest": "max",
    "median": "median",
    # "highest" / "lowest" without explicit field → aggregation context, not sort
    # Keep "highest"/"lowest" out of this map; they're handled in post-process
    # as sort-direction signals, not as pandas agg functions.
}

# ── Intent classification prompt ──────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are a data-analysis intent classifier for a "Talk to Data" application.

Given a user question and a dataset schema, classify the question into ONE intent
and extract structured fields. Read the schema carefully — use EXACT column names.

INTENT DECISION RULES
=====================

1. COMPARE
   Use when the user wants a metric computed SEPARATELY FOR EACH GROUP of a categorical column,
   OR when the user wants to compare two time periods.

   STRONG signals (any of these → COMPARE):
   - "X per Y"              → "average age per species"
   - "X for each Y"         → "revenue for each product"
   - "X for different Y"    → "average age for different sex"
   - "X by Y"               → "revenue by region"
   - "X across Y"           → "performance across teams"
   - "X vs Y" (groups)      → "male vs female age"
   - "which Y has highest X"→ "which species has highest petal length"
   - "compare X by Y"       → "compare sales by region"
   - "X grouped by Y"       → "purchases grouped by category"
   - "distribution of X by Y" → "distribution of salary by department"
   - "this week vs last week" → time period comparison
   - "month over month"     → time-based comparison
   - "Region A vs Region B" → specific value comparison
   - "Product X performance vs Product Y" → specific group comparison

   target_metrics = [the NUMERIC column being measured]
   dimensions = [the CATEGORICAL column doing the grouping]

2. BREAKDOWN
   Use when the user wants proportions/shares/composition/decomposition.
   Signals: "what makes up", "breakdown of", "composition of", "proportion of",
            "what percentage", "share of total", "how is X divided",
            "show the breakdown", "decompose", "contribute to",
            "biggest contributors", "what drives"

3. CHANGE
   Use when the user wants to understand WHY or HOW something changed over time,
   or identify drivers behind increases/decreases.
   Signals: "why did X drop/rise", "what caused", "how did X change over time",
            "trend in X", "increase/decrease in X", "what happened to X",
            "X went up/down", "what drove the change", "reasons for",
            "show me the trend", "over time", "time series",
            "monthly/weekly/daily progression", "growth of"

4. SUMMARY
   Use when the user wants an overview, stats, insights, or a general understanding.
   Signals: "tell me about", "describe", "overview of", "summarise", "stats on X",
            "what is X like", "give me insights", "what's important",
            "key findings", "highlight anomalies", "weekly summary",
            "what should I know", "scan the data", "any outliers",
            "interesting patterns", "overall performance", "executive summary"

EXTRACTION RULES
================

target_metrics: The NUMERIC column(s) the user is measuring or computing.
  - Strip aggregation words: "average age" → ["Age"], "total sales" → ["Sales"]
  - Use EXACT column names from the schema. Capitalisation must match exactly.
  - If truly unspecified, output ["unknown"].
  - NEVER put categorical columns here.

dimensions: The CATEGORICAL column(s) used to GROUP the metric.
  - "per sex" → ["Sex"], "by region" → ["Region"]
  - NEVER put numeric columns here.
  - Output [] if no grouping is requested.

aggregation: The aggregation explicitly stated.
  - "average" / "mean" → "mean"
  - "total" / "sum" → "sum"
  - "count" / "how many" → "count"
  - "highest which group" → "mean"  (ranking by mean, not pandas max)
  - "lowest which group" → "mean"
  - "median" → "median"
  - Not stated → null

time_range: Any time period mentioned, or null.
  - "last month" → "last month"
  - "this week vs last week" → "this week vs last week"
  - "January" → "January"
  - "Q1 2025" → "Q1 2025"

suggested_chart:
  COMPARE → "bar"
  BREAKDOWN (≤8 groups) → "pie", else "bar"
  CHANGE → "bar" (or "line" for trend over many periods)
  SUMMARY (single numeric) → "bar", (time series) → "line"

EXAMPLES
========

Q: "what is average age for people of different sex"
Schema: Age (numeric), Sex (categorical: male/female)
→ {"intent":"COMPARE","target_metrics":["Age"],"dimensions":["Sex"],"aggregation":"mean","time_range":null,"suggested_chart":"bar"}

Q: "show me revenue by region"
Schema: Revenue (numeric), Region (categorical: North/South/East/West)
→ {"intent":"COMPARE","target_metrics":["Revenue"],"dimensions":["Region"],"aggregation":null,"time_range":null,"suggested_chart":"bar"}

Q: "which species has the highest average petal length"
Schema: PetalLengthCm (numeric), Species (categorical: setosa/versicolor/virginica)
→ {"intent":"COMPARE","target_metrics":["PetalLengthCm"],"dimensions":["Species"],"aggregation":"mean","time_range":null,"suggested_chart":"bar"}

Q: "total sales per product category"
Schema: Sales (numeric), Category (categorical)
→ {"intent":"COMPARE","target_metrics":["Sales"],"dimensions":["Category"],"aggregation":"sum","time_range":null,"suggested_chart":"bar"}

Q: "what is the survival rate by passenger class"
Schema: Survived (numeric 0/1), Pclass (numeric 1/2/3)
→ {"intent":"COMPARE","target_metrics":["Survived"],"dimensions":["Pclass"],"aggregation":"mean","time_range":null,"suggested_chart":"bar"}

Q: "what makes up total revenue?"
Schema: Revenue (numeric), Category (categorical)
→ {"intent":"BREAKDOWN","target_metrics":["Revenue"],"dimensions":["Category"],"aggregation":"sum","time_range":null,"suggested_chart":"pie"}

Q: "why did revenue drop last month?"
→ {"intent":"CHANGE","target_metrics":["Revenue"],"dimensions":[],"aggregation":null,"time_range":"last month","suggested_chart":"bar"}

Q: "show me the trend of revenue over time"
→ {"intent":"CHANGE","target_metrics":["Revenue"],"dimensions":[],"aggregation":null,"time_range":null,"suggested_chart":"line"}

Q: "this week vs last week performance"
→ {"intent":"COMPARE","target_metrics":["unknown"],"dimensions":[],"aggregation":null,"time_range":"this week vs last week","suggested_chart":"bar"}

Q: "tell me about the species column"
Schema: Species (categorical: setosa/versicolor/virginica)
→ {"intent":"SUMMARY","target_metrics":["Species"],"dimensions":[],"aggregation":"count","time_range":null,"suggested_chart":"bar"}

Q: "give me a summary of sepal length"
Schema: SepalLengthCm (numeric)
→ {"intent":"SUMMARY","target_metrics":["SepalLengthCm"],"dimensions":[],"aggregation":null,"time_range":null,"suggested_chart":"bar"}

Q: "give me a weekly summary of customer metrics"
→ {"intent":"SUMMARY","target_metrics":["unknown"],"dimensions":[],"aggregation":null,"time_range":"weekly","suggested_chart":"bar"}

Q: "any anomalies or interesting patterns in the data?"
→ {"intent":"SUMMARY","target_metrics":["unknown"],"dimensions":[],"aggregation":null,"time_range":null,"suggested_chart":"bar"}

Respond with ONLY a valid JSON object. No markdown, no explanation, no extra text:
{
  "intent": "CHANGE|COMPARE|BREAKDOWN|SUMMARY",
  "target_metrics": ["exact_column_name"],
  "dimensions": ["exact_column_name"],
  "aggregation": "mean|sum|count|max|min|median|null",
  "time_range": "string or null",
  "suggested_chart": "bar|pie|line|table"
}"""


# ── Schema context builder ────────────────────────────────────────────────────

def _build_schema_context(
    columns: list[str],
    col_types: dict[str, str],
    col_samples: dict[str, list],
) -> str:
    lines = ["Dataset schema (use EXACT column names below):"]
    for col in columns:
        dtype = col_types.get(col, "unknown")
        samples = col_samples.get(col, [])
        sample_str = ", ".join(str(s) for s in samples[:4]) if samples else "—"
        if "int" in dtype or "float" in dtype:
            kind = "numeric"
        elif dtype in ("object", "category", "string") or dtype.startswith("category"):
            kind = "categorical"
        elif "datetime" in dtype or "date" in dtype:
            kind = "datetime"
        else:
            kind = dtype
        lines.append(f"  • {col} ({kind}) — e.g. {sample_str}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_intent(
    question: str,
    available_columns: list[str],
    col_types: dict[str, str] | None = None,
    col_samples: dict[str, list] | None = None,
) -> dict:
    """Classify a natural-language question into a structured intent.

    Args:
        question: The user's plain-English question.
        available_columns: Column names in the dataset.
        col_types: column → dtype string (e.g. "float64", "object").
        col_samples: column → list of representative sample values.

    Returns:
        Dict with: intent, target_metrics, dimensions, aggregation,
                   time_range, suggested_chart.
    """
    col_types = col_types or {}
    col_samples = col_samples or {}

    schema_context = _build_schema_context(available_columns, col_types, col_samples)
    user_prompt = (
        f"{schema_context}\n\n"
        f'User question: "{question}"\n\n'
        "Classify this question. Use EXACT column names from the schema above."
    )

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=INTENT_SYSTEM_PROMPT,
            max_output_tokens=500,
        ),
    )

    raw_text = response.text.strip()
    logger.info("Intent parser raw response: %s", raw_text)

    parsed = _safe_parse_json(raw_text)
    parsed = _normalise(parsed, available_columns)
    parsed = _post_process(parsed, question, available_columns, col_types)

    logger.info("Final parsed intent: %s", parsed)
    return parsed


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(parsed: dict, available_columns: list[str]) -> dict:
    """Validate types, coerce legacy fields, and ground column names."""
    valid_intents = {"CHANGE", "COMPARE", "BREAKDOWN", "SUMMARY"}
    parsed["intent"] = parsed.get("intent", "SUMMARY").upper()
    if parsed["intent"] not in valid_intents:
        parsed["intent"] = "SUMMARY"

    # Coerce legacy scalar field name
    if "target_metric" in parsed and "target_metrics" not in parsed:
        parsed["target_metrics"] = [parsed.pop("target_metric")]
    parsed.setdefault("target_metrics", ["unknown"])
    if not isinstance(parsed["target_metrics"], list) or not parsed["target_metrics"]:
        parsed["target_metrics"] = ["unknown"]

    parsed.setdefault("dimensions", [])
    if not isinstance(parsed["dimensions"], list):
        parsed["dimensions"] = []

    parsed.setdefault("aggregation", None)
    if parsed.get("aggregation") in ("null", "none", ""):
        parsed["aggregation"] = None

    parsed.setdefault("time_range", None)
    parsed.setdefault("suggested_chart", _default_chart(parsed["intent"]))

    # Ground all names to real columns
    parsed["target_metrics"] = _ground_to_columns(parsed["target_metrics"], available_columns)
    parsed["dimensions"] = _ground_to_columns(parsed["dimensions"], available_columns)
    # Remove any dimension that duplicates a metric
    parsed["dimensions"] = [d for d in parsed["dimensions"] if d not in parsed["target_metrics"]]

    return parsed


# ── Post-processing (deterministic corrections) ───────────────────────────────

def _post_process(
    parsed: dict,
    question: str,
    available_columns: list[str],
    col_types: dict[str, str],
) -> dict:
    """Rule-based corrections that run after LLM classification.

    Fixes:
    1. Explicit aggregation extraction from raw question text
    2. Grouping phrases (per/by/for each/for different/across) → COMPARE
    3. Superlative phrases (which X has highest/lowest Y) → COMPARE with sort hint
    4. Numeric↔categorical swap between target_metrics and dimensions
    5. "highest/lowest" as aggregation context → use mean + set sort_desc hint
    6. Missing numeric metric in COMPARE → infer from question tokens
    7. Strip "unknown" placeholders when real columns were found
    """
    q_lower = question.lower()

    numeric_cols = {
        c for c in available_columns
        if col_types.get(c, "").startswith(("int", "float"))
    }
    categorical_cols = {
        c for c in available_columns
        if col_types.get(c, "") in ("object", "category", "string")
        or col_types.get(c, "").startswith("category")
    }

    # ── 1. Extract explicit aggregation from question ──────────────────────
    if not parsed.get("aggregation"):
        parsed["aggregation"] = _extract_agg_keyword(q_lower)

    # ── 2. Detect grouping phrases → COMPARE ──────────────────────────────
    # Patterns that signal "compute metric separately for each group"
    grouping_patterns = [
        r"\bper\s+([\w][\w\s]{1,28}?)(?=\s*$|\s*\?|\s+and\b|\s+or\b)",
        r"\bby\s+([\w][\w\s]{1,28}?)(?=\s*$|\s*\?|\s+and\b|\s+or\b)",
        r"\bfor\s+(?:each|different|every|various)\s+([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
        r"\bacross\s+(?:different\s+)?([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
        r"\bamong\s+(?:different\s+)?([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
        r"\bgrouped?\s+by\s+([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
        r"\bcompare\s+[\w\s]+\s+by\s+([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
        r"\bdistribution\s+of\s+[\w\s]+\s+by\s+([\w][\w\s]{1,24}?)(?=\s*$|\s*\?)",
    ]
    found_dim: str | None = None
    for pattern in grouping_patterns:
        m = re.search(pattern, q_lower)
        if m:
            phrase = m.group(1).strip().rstrip("?").strip()
            # Require phrase is at least 2 chars to avoid single-letter false matches
            if len(phrase) < 2:
                continue
            col = _match_column_phrase(phrase, available_columns)
            if col and col in categorical_cols:
                found_dim = col
                break

    if found_dim:
        parsed["intent"] = "COMPARE"
        parsed["suggested_chart"] = "bar"
        if found_dim not in parsed["dimensions"]:
            parsed["dimensions"] = [found_dim] + [d for d in parsed["dimensions"] if d != found_dim]
        if found_dim in parsed["target_metrics"]:
            parsed["target_metrics"] = [m for m in parsed["target_metrics"] if m != found_dim]

    # ── 3. Superlative phrases → COMPARE with descending/ascending sort ────
    # "which species has highest petal length" → COMPARE, sort desc
    # "which region has lowest churn" → COMPARE, sort asc
    superlative_re = re.search(
        r"\bwhich\s+([\w][\w\s]{1,24}?)\s+(?:has|have|is|are)\s+"
        r"(?:the\s+)?(?P<dir>highest|largest|most|best|lowest|smallest|least|worst|maximum|minimum)\b",
        q_lower,
    )
    if superlative_re:
        dim_phrase = superlative_re.group(1).strip()
        direction = superlative_re.group("dir")
        dim_col = _match_column_phrase(dim_phrase, available_columns)
        if dim_col and dim_col in categorical_cols:
            parsed["intent"] = "COMPARE"
            parsed["suggested_chart"] = "bar"
            if dim_col not in parsed["dimensions"]:
                parsed["dimensions"] = [dim_col] + [d for d in parsed["dimensions"] if d != dim_col]
            if dim_col in parsed["target_metrics"]:
                parsed["target_metrics"] = [m for m in parsed["target_metrics"] if m != dim_col]
            # "highest/most" → sort descending; "lowest/least" → ascending
            parsed["sort_desc"] = direction in ("highest", "largest", "most", "best", "maximum")
            # Force mean aggregation for ranking questions (not raw pandas max/min)
            if not parsed.get("aggregation"):
                parsed["aggregation"] = "mean"

    # ── 4. Fix numeric↔categorical swap ───────────────────────────────────
    # Categorical col in target_metrics → move to dimensions (unless SUMMARY)
    # Numeric col in dimensions → move to target_metrics
    corrected_metrics: list[str] = []
    corrected_dims: list[str] = list(parsed["dimensions"])

    for m in parsed["target_metrics"]:
        if m in categorical_cols and m not in corrected_dims:
            corrected_dims.insert(0, m)
            # Only force COMPARE if intent wasn't deliberately SUMMARY
            if parsed["intent"] != "SUMMARY":
                parsed["intent"] = "COMPARE"
        else:
            corrected_metrics.append(m)

    for d in list(corrected_dims):
        if d in numeric_cols and d not in corrected_metrics:
            corrected_metrics.insert(0, d)
            corrected_dims.remove(d)

    if corrected_metrics:
        parsed["target_metrics"] = corrected_metrics
    parsed["dimensions"] = list(dict.fromkeys(corrected_dims))
    parsed["dimensions"] = [d for d in parsed["dimensions"] if d not in parsed["target_metrics"]]

    # ── 5. COMPARE with no numeric metric → infer from question tokens ─────
    if parsed["intent"] == "COMPARE":
        if not _has_numeric_metric(parsed["target_metrics"], numeric_cols):
            best = _infer_numeric_metric(q_lower, available_columns, numeric_cols)
            if best:
                parsed["target_metrics"] = [best]

    # ── 6. Strip "unknown" placeholders if real columns were found ─────────
    real_metrics = [m for m in parsed["target_metrics"] if m != "unknown"]
    if real_metrics:
        parsed["target_metrics"] = real_metrics

    return parsed


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_parse_json(raw: str) -> dict:
    """Best-effort JSON extraction from a Gemini response string."""
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse intent JSON, defaulting to SUMMARY.")
    return {"intent": "SUMMARY", "target_metrics": ["unknown"], "dimensions": [], "time_range": None}


def _default_chart(intent: str) -> str:
    return {"CHANGE": "bar", "COMPARE": "bar", "BREAKDOWN": "pie", "SUMMARY": "bar"}.get(intent, "bar")


def _normalize_str(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _ground_to_columns(names: list[str], available_columns: list[str]) -> list[str]:
    """Map AI-extracted names to closest actual column names."""
    col_exact = {c: c for c in available_columns}
    col_lower = {c.lower(): c for c in available_columns}
    col_norm = {_normalize_str(c): c for c in available_columns}
    grounded, seen = [], set()
    for name in names:
        if name == "unknown":
            if name not in seen:
                grounded.append(name)
                seen.add(name)
            continue
        resolved = (
            col_exact.get(name)
            or col_lower.get(name.lower())
            or col_norm.get(_normalize_str(name))
            or name  # keep for downstream fuzzy matching
        )
        if resolved not in seen:
            grounded.append(resolved)
            seen.add(resolved)
    return grounded


def _match_column_phrase(phrase: str, available_columns: list[str]) -> str | None:
    """Match a free-text phrase to the closest column name.

    Uses exact → substring (with minimum length guard) → token overlap.
    """
    phrase_norm = _normalize_str(phrase)
    if not phrase_norm:
        return None
    col_norm_map = {_normalize_str(c): c for c in available_columns}

    # 1. Exact normalized match
    if phrase_norm in col_norm_map:
        return col_norm_map[phrase_norm]

    # 2. Substring — only when the shorter string is ≥3 chars to avoid noise
    for cn, col in col_norm_map.items():
        shorter = phrase_norm if len(phrase_norm) <= len(cn) else cn
        if len(shorter) >= 3 and (phrase_norm in cn or cn in phrase_norm):
            return col

    # 3. Token overlap — require at least 1 meaningful token match
    phrase_tokens = set(re.findall(r"[a-z]{2,}", phrase_norm))  # ≥2 chars per token
    best_col, best_score = None, 0
    for cn, col in col_norm_map.items():
        col_tokens = set(re.findall(r"[a-z]{2,}", cn))
        score = len(phrase_tokens & col_tokens)
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score > 0 else None


def _has_numeric_metric(metrics: list[str], numeric_cols: set[str]) -> bool:
    return any(m in numeric_cols for m in metrics)


def _infer_numeric_metric(
    q_lower: str, available_columns: list[str], numeric_cols: set[str]
) -> str | None:
    """Guess the most likely numeric metric by token overlap with question."""
    q_tokens = set(re.findall(r"[a-z]{2,}", q_lower))  # ≥2 chars
    best_col, best_score = None, 0
    for col in available_columns:
        if col not in numeric_cols:
            continue
        col_tokens = set(re.findall(r"[a-z]{2,}", _normalize_str(col)))
        score = len(q_tokens & col_tokens)
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score > 0 else None  # explicit score guard


def _extract_agg_keyword(q_lower: str) -> str | None:
    """Extract an explicit aggregation keyword from the question (longest match first)."""
    for phrase in sorted(_AGG_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q_lower:
            return _AGG_KEYWORD_MAP[phrase]
    return None
