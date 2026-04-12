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
    "average": "mean", "avg": "mean", "mean": "mean",
    "total": "sum", "sum": "sum",
    "count": "count", "number of": "count", "how many": "count",
    "unique": "nunique", "distinct": "nunique",
    "minimum": "min", "min": "min", "lowest": "min", "smallest": "min",
    "maximum": "max", "max": "max", "highest": "max", "largest": "max",
    "median": "median",
}

# ── Intent classification prompt ──────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are a data-analysis intent classifier for a "Talk to Data" application.

Given a user question and a dataset schema, classify the question into ONE intent
and extract structured fields. Read the schema carefully — use EXACT column names.

INTENT DECISION RULES
=====================

1. COMPARE
   Use when the user wants a metric computed SEPARATELY FOR EACH GROUP of a categorical column.

   STRONG signals:
   - "X per Y"           → "average age per species", "sales per region"
   - "X for each Y"      → "revenue for each product"
   - "X for different Y" → "average age for different sex"
   - "X by Y"            → "revenue by region", "score by grade"
   - "X across Y"        → "performance across teams"
   - "X vs Y" (groups)   → "male vs female age"
   - "which Y has highest/lowest X" → "which species has highest petal length"

   target_metrics = [the NUMERIC column being measured]
   dimensions = [the CATEGORICAL column doing the grouping]

2. BREAKDOWN
   Use when the user wants proportions/shares/composition of a total.
   Signals: "what makes up", "breakdown of", "composition", "proportion", "share", "percentage"

3. CHANGE
   Use when the user wants to know WHY or HOW something changed over time.
   Signals: "why did X drop/rise", "what caused", "trend in X over time"

4. SUMMARY
   Use when none of the above apply — user wants general stats about a column.
   Signals: "tell me about", "describe", "overview", "stats on", "summarise"

EXTRACTION RULES
================

target_metrics: NUMERIC columns being measured.
  - REMOVE aggregation words (average/total/sum/count/mean/max/min) — keep only the base column name.
  - "average age" → ["Age"]   NOT ["average Age"]
  - "total sales" → ["Sales"] NOT ["total Sales"]
  - Use EXACT column names from schema.
  - Output ["unknown"] only if truly unspecified.

dimensions: CATEGORICAL columns used to GROUP or SPLIT the metric.
  - "per sex" → ["Sex"]
  - "by region" → ["Region"]
  - "for different departments" → ["Department"]
  - NEVER include numeric columns here.
  - Output [] if no grouping.

aggregation: The function explicitly stated or strongly implied.
  - "average" / "mean" → "mean"
  - "total" / "sum" → "sum"
  - "count" / "how many" → "count"
  - "highest" / "maximum" → "max"
  - "lowest" / "minimum" → "min"
  - "median" → "median"
  - If not stated → null

time_range: Any time period mentioned, or null.

suggested_chart: "bar" for COMPARE/CHANGE, "pie" for BREAKDOWN ≤8 groups, "bar" for SUMMARY.

EXAMPLES
========

Q: "what is average age for people of different sex"
Schema: Age (numeric), Sex (categorical: male/female)
→ {"intent":"COMPARE","target_metrics":["Age"],"dimensions":["Sex"],"aggregation":"mean","time_range":null,"suggested_chart":"bar"}

Q: "show me revenue by region"
Schema: Revenue (numeric), Region (categorical)
→ {"intent":"COMPARE","target_metrics":["Revenue"],"dimensions":["Region"],"aggregation":null,"time_range":null,"suggested_chart":"bar"}

Q: "which species has the highest average petal length"
Schema: PetalLengthCm (numeric), Species (categorical)
→ {"intent":"COMPARE","target_metrics":["PetalLengthCm"],"dimensions":["Species"],"aggregation":"mean","time_range":null,"suggested_chart":"bar"}

Q: "total sales per product category"
Schema: Sales (numeric), Category (categorical)
→ {"intent":"COMPARE","target_metrics":["Sales"],"dimensions":["Category"],"aggregation":"sum","time_range":null,"suggested_chart":"bar"}

Q: "what makes up total revenue?"
Schema: Revenue (numeric), Category (categorical)
→ {"intent":"BREAKDOWN","target_metrics":["Revenue"],"dimensions":["Category"],"aggregation":"sum","time_range":null,"suggested_chart":"pie"}

Q: "why did revenue drop last month?"
→ {"intent":"CHANGE","target_metrics":["Revenue"],"dimensions":[],"aggregation":null,"time_range":"last month","suggested_chart":"bar"}

Q: "tell me about the species column"
Schema: Species (categorical: setosa/versicolor/virginica)
→ {"intent":"SUMMARY","target_metrics":["Species"],"dimensions":[],"aggregation":"count","time_range":null,"suggested_chart":"bar"}

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
        elif dtype in ("object", "category", "string"):
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
        col_types: column → dtype string.
        col_samples: column → representative sample values.

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
            max_output_tokens=300,
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
    valid_intents = {"CHANGE", "COMPARE", "BREAKDOWN", "SUMMARY"}
    parsed["intent"] = parsed.get("intent", "SUMMARY").upper()
    if parsed["intent"] not in valid_intents:
        parsed["intent"] = "SUMMARY"

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

    parsed["target_metrics"] = _ground_to_columns(parsed["target_metrics"], available_columns)
    parsed["dimensions"] = _ground_to_columns(parsed["dimensions"], available_columns)
    parsed["dimensions"] = [d for d in parsed["dimensions"] if d not in parsed["target_metrics"]]

    return parsed


# ── Post-processing ───────────────────────────────────────────────────────────

def _post_process(
    parsed: dict,
    question: str,
    available_columns: list[str],
    col_types: dict[str, str],
) -> dict:
    """Deterministic corrections applied after LLM classification.

    Fixes the most common failure modes:
    1. Grouping phrases ("per/by/for each/for different") → force COMPARE
    2. Superlative phrases ("which X has highest Y") → force COMPARE
    3. Numeric/categorical swap between metrics and dimensions
    4. Aggregation keyword extraction from raw question text
    5. Missing numeric metric in COMPARE queries
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
    grouping_patterns = [
        r"\bper\s+([\w\s]{2,30}?)(?:\s*$|\s*\?|\s+and\b|\s+or\b)",
        r"\bby\s+([\w\s]{2,30}?)(?:\s*$|\s*\?|\s+and\b|\s+or\b)",
        r"\bfor\s+(?:each|different|every|various)\s+([\w\s]{2,25}?)(?:\s*$|\s*\?)",
        r"\bacross\s+(?:different\s+)?([\w\s]{2,25}?)(?:\s*$|\s*\?)",
        r"\bamong\s+(?:different\s+)?([\w\s]{2,25}?)(?:\s*$|\s*\?)",
        r"\bgrouped?\s+by\s+([\w\s]{2,25}?)(?:\s*$|\s*\?)",
        r"\bfor\s+([\w\s]{2,25}?)\s+(?:groups?|categories|segments?|types?)",
    ]
    found_dim: str | None = None
    for pattern in grouping_patterns:
        m = re.search(pattern, q_lower)
        if m:
            phrase = m.group(1).strip().rstrip("?").strip()
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

    # ── 3. Superlative phrases → COMPARE ──────────────────────────────────
    superlative_re = re.search(
        r"\bwhich\s+([\w\s]{2,25}?)\s+(?:has|have|is|are)\s+(?:the\s+)?(?:highest|lowest|most|least|best|worst|largest|smallest|maximum|minimum)\b",
        q_lower,
    )
    if superlative_re:
        dim_phrase = superlative_re.group(1).strip()
        dim_col = _match_column_phrase(dim_phrase, available_columns)
        if dim_col and dim_col in categorical_cols:
            parsed["intent"] = "COMPARE"
            parsed["suggested_chart"] = "bar"
            if dim_col not in parsed["dimensions"]:
                parsed["dimensions"] = [dim_col] + [d for d in parsed["dimensions"] if d != dim_col]
            if dim_col in parsed["target_metrics"]:
                parsed["target_metrics"] = [m for m in parsed["target_metrics"] if m != dim_col]

    # ── 4. Fix numeric↔categorical swap ───────────────────────────────────
    # If a categorical col ended up in target_metrics, move it to dimensions
    # If a numeric col ended up in dimensions, move it to target_metrics
    corrected_metrics = []
    corrected_dims = list(parsed["dimensions"])

    for m in parsed["target_metrics"]:
        if m in categorical_cols and m not in corrected_dims:
            corrected_dims.insert(0, m)
            if parsed["intent"] == "SUMMARY":
                pass  # keep SUMMARY for "tell me about Species"
            else:
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

    # ── 5. COMPARE with no numeric metric → infer from question ───────────
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
            or name
        )
        if resolved not in seen:
            grounded.append(resolved)
            seen.add(resolved)
    return grounded


def _match_column_phrase(phrase: str, available_columns: list[str]) -> str | None:
    phrase_norm = _normalize_str(phrase)
    col_norm_map = {_normalize_str(c): c for c in available_columns}
    if phrase_norm in col_norm_map:
        return col_norm_map[phrase_norm]
    for cn, col in col_norm_map.items():
        if phrase_norm in cn or cn in phrase_norm:
            return col
    phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase_norm))
    best_col, best_score = None, 0
    for cn, col in col_norm_map.items():
        score = len(phrase_tokens & set(re.findall(r"[a-z0-9]+", cn)))
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score > 0 else None


def _has_numeric_metric(metrics: list[str], numeric_cols: set[str]) -> bool:
    return any(m in numeric_cols for m in metrics)


def _infer_numeric_metric(q_lower: str, available_columns: list[str], numeric_cols: set[str]) -> str | None:
    q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))
    best_col, best_score = None, 0
    for col in available_columns:
        if col not in numeric_cols:
            continue
        col_tokens = set(re.findall(r"[a-z0-9]+", _normalize_str(col)))
        score = len(q_tokens & col_tokens)
        if score > best_score:
            best_score, best_col = score, col
    return best_col


def _extract_agg_keyword(q_lower: str) -> str | None:
    for phrase in sorted(_AGG_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q_lower:
            return _AGG_KEYWORD_MAP[phrase]
    return None