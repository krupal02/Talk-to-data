"""
Gemini service – builds intent-specific prompts and calls the Google Gemini API.

This is the only module that invokes Gemini for answer generation.
It receives pre-aggregated data from the data engine and produces a
plain-English explanation grounded strictly in the numbers provided.
Raw row data is NEVER included in any prompt.
"""

import json
import logging
from typing import Any

from google import genai

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Base system prompt ────────────────────────────────────────────────────────

_SYSTEM_BASE = """You are a data analyst assistant inside a "Talk to Data" application.
You receive pre-computed, aggregated numbers from a dataset and your job is to explain
them clearly to a non-technical business user.

STRICT RULES — follow all of these:
1. Write 3–6 complete sentences. No bullet points. No headers. Plain prose only.
2. Lead with the single most important finding (biggest number, largest change, dominant group).
3. Always state specific numbers from the data. Never say "significantly higher" without a number.
4. Use plain English. If you use a statistical word (e.g. "median"), define it in parentheses.
5. Do not invent anything not present in the aggregated data provided to you.
6. Refer to columns by their human meaning, not their technical name
   (e.g. say "age" not "the Age column", say "sales" not "the Sales_Amount field").
7. Do not reveal raw individual records or any personally identifiable information.
8. End with one actionable sentence: what the user could explore next.
9. CRITICAL: You MUST finish every sentence completely. Never stop mid-sentence.
   Every response must end with a complete sentence ending in a period, exclamation mark, or question mark.
10. Stick to what the data shows. Do not use external knowledge to infer context.
11. If trend data is present, always mention whether things are improving or declining.
12. If anomalies or outliers are mentioned in the data, highlight them."""


# ── Intent-specific prompt builders ──────────────────────────────────────────

def _build_change_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "group_ranking")
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "group")

    if analysis_type == "period_over_period":
        overall = result.get("overall_change", {})
        top_movers = result.get("top_movers", [])
        secondary_drivers = result.get("secondary_drivers", {})

        overall_text = ""
        if overall:
            overall_text = (
                f"\nOverall change: {overall.get('direction', 'unknown')} by "
                f"{overall.get('delta', 0)} ({overall.get('pct_change', 'N/A')}%). "
                f"Period 1 total: {overall.get('period_1_total', 'N/A')}, "
                f"Period 2 total: {overall.get('period_2_total', 'N/A')}."
            )

        movers_text = ""
        if top_movers:
            movers_lines = []
            for m in top_movers:
                movers_lines.append(
                    f"  - {m['group']}: {m['direction']} by {m['delta']} "
                    f"({m.get('pct_change', 'N/A')}%)"
                )
            movers_text = "\nTop movers (biggest changes):\n" + "\n".join(movers_lines)

        secondary_text = ""
        if secondary_drivers:
            for dim_name, dim_data in secondary_drivers.items():
                secondary_text += f"\nSecondary driver ({dim_name}):\n{json.dumps(dim_data, indent=2)}"

        instruction = (
            f"The data shows period-over-period changes grouped by {group_col}. "
            "'period_1' is the earlier half of the data, 'period_2' is the later half. "
            "'delta' is the absolute change (positive = increase, negative = decrease). "
            "'pct_change' is the percentage change. "
            "Start by stating the overall change direction and magnitude. "
            "Then identify the group with the largest absolute change and explain if it increased or decreased. "
            "Mention the top 2-3 drivers by name with specific numbers. "
            "If secondary drivers from other dimensions are provided, briefly mention the most notable one. "
            "Conclude with what the user should investigate next."
        )

        return (
            f'User question: "{question}"\n'
            f"Metric: {result['metric_used']}\n"
            f"Grouping column: {group_col}\n"
            f"Time range: {intent.get('time_range') or 'not specified'}\n"
            f"Analysis type: {analysis_type}\n"
            f"{overall_text}\n"
            f"{movers_text}\n\n"
            f"Detailed data by {group_col}:\n{data_json}\n"
            f"{secondary_text}\n\n"
            f"Instructions: {instruction}"
        )
    else:
        instruction = (
            f"The data shows {result['metric_used']} for each {group_col}. "
            "'value' is the metric value; 'share_pct' is its percentage of the total. "
            "Identify the dominant group and its value, then note how the others compare. "
            "Since no time comparison is available, focus on the current ranking."
        )

    return (
        f'User question: "{question}"\n'
        f"Metric: {result['metric_used']}\n"
        f"Grouping column: {group_col}\n"
        f"Time range: {intent.get('time_range') or 'not specified'}\n"
        f"Analysis type: {analysis_type}\n\n"
        f"Data:\n{data_json}\n\n"
        f"Instructions: {instruction}"
    )


def _build_compare_prompt(question: str, intent: dict, result: dict) -> str:
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "group")
    agg_func = result.get("agg_func", "")
    agg_label = {
        "mean": "average", "sum": "total", "count": "count",
        "max": "maximum", "min": "minimum", "median": "median",
    }.get(agg_func, agg_func)

    return (
        f'User question: "{question}"\n'
        f"Metric: {result['metric_used']}\n"
        f"Grouping column: {group_col}\n"
        f"Aggregation: {agg_label} per group\n\n"
        f"Comparison data (sorted highest to lowest):\n{data_json}\n\n"
        "Instructions: "
        f"State which {group_col} has the highest {agg_label} and give the exact value. "
        "Then compare it to the next group(s). If 'diff_from_leader' is present, use it "
        "to state the gap (e.g. 'Group B is 3.2 lower than Group A'). "
        "Cover all groups shown in the data — do not skip any. "
        "If the differences are small, note that the groups are similar. "
        "If one group clearly dominates, highlight that. "
        "Conclude with what this comparison suggests and what to explore next."
    )


def _build_breakdown_prompt(question: str, intent: dict, result: dict) -> str:
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "category")
    total_groups = result.get("total_groups", len(result["aggregated_data"]))
    show_pct = result.get("show_pct", True)
    secondary = result.get("secondary_breakdown")
    secondary_col = result.get("secondary_group_col")
    concentration = result.get("concentration", {})

    pct_note = (
        "Each entry has 'value' and 'percentage' (share of total)."
        if show_pct
        else "Each entry has 'value' only (percentages are not shown because the metric is an average)."
    )

    concentration_text = ""
    if concentration:
        pattern = concentration.get("pattern", "unknown")
        top_1 = concentration.get("top_1_share", "N/A")
        top_3 = concentration.get("top_3_share", "N/A")
        concentration_text = (
            f"\nConcentration: {pattern}. "
            f"Top 1 group accounts for {top_1}% of total, top 3 account for {top_3}%."
        )
        outliers = concentration.get("outlier_groups", [])
        if outliers:
            concentration_text += f" Outlier groups (unusually high/low): {', '.join(outliers)}."

    pct_note = (
        "Each entry has 'value' and 'percentage' (share of total)."
        if show_pct
        else "Each entry has 'value' only (percentages are not shown because the metric is an average)."
    )

    extra = ""
    if secondary:
        extra = (
            f"\n\nSecondary breakdown by {secondary_col}:\n"
            f"{json.dumps(secondary, indent=2)}\n"
            "After the primary breakdown, add one sentence about the secondary dimension."
        )

    return (
        f'User question: "{question}"\n'
        f"Metric: {result['metric_used']}\n"
        f"Grouped by: {group_col} ({total_groups} total groups, showing top 10)\n"
        f"{pct_note}\n"
        f"{concentration_text}\n\n"
        f"Breakdown data:\n{data_json}"
        f"{extra}\n\n"
        "Instructions: Start with the largest segment and its value"
        + (" and percentage share" if show_pct else "")
        + ". Then cover the second and third largest. "
        "If an 'Other' bucket exists, mention it. "
        "If concentration info is provided, mention whether the distribution is concentrated or spread. "
        "Highlight any outlier groups if mentioned. "
        "Conclude with whether one group dominates or values are spread evenly, "
        "and suggest what the user could explore next."
    )


def _build_summary_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "summary")

    if analysis_type == "categorical_summary":
        data_json = json.dumps(result["aggregated_data"], indent=2)
        group_col = result.get("group_col", "column")
        total_groups = result.get("total_groups", len(result["aggregated_data"]))
        concentration = result.get("concentration", "")
        top_1_pct = result.get("top_1_pct", 0)

        concentration_note = ""
        if concentration:
            concentration_note = (
                f"\nDistribution pattern: {concentration} "
                f"(most frequent value represents {top_1_pct}% of all entries)."
            )

        return (
            f'User question: "{question}"\n'
            f"Column: {group_col} ({total_groups} unique values)\n"
            f"{concentration_note}\n\n"
            f"Value distribution (count + percentage of total):\n{data_json}\n\n"
            "Instructions: State the total number of unique values. "
            "Name the most frequent value and its percentage. "
            "Describe how evenly or unevenly distributed the values are. "
            "Cover all values shown in the data. "
            "End with what this distribution suggests and what else to explore."
        )

    data_json = json.dumps(result["aggregated_data"], indent=2)
    return (
        f'User question: "{question}"\n'
        f"Metric(s): {result['metric_used']}\n\n"
        f"Statistical summary (one entry per metric column):\n{data_json}\n\n"
        "Instructions: For each metric, state the mean (average) and the range (min to max). "
        "If 'percentile_25' and 'percentile_75' are present, mention the interquartile range "
        "(the middle 50% of values fall between these). "
        "If 'distribution' or 'skewness' is present, explain in simple terms what it means "
        "(e.g. 'most values are clustered at the lower end with a few high outliers'). "
        "If 'trend' is present, mention it and the 'trend_pct_change' — state whether "
        "things are improving or declining. "
        "If 'anomaly_note' or 'outlier_count' is non-zero, highlight them. "
        "If 'correlations' is present, mention the strongest positive or negative correlation. "
        "Keep the answer focused and useful for a non-technical user. "
        "End with what the user should investigate next."
    )


_PROMPT_BUILDERS = {
    "CHANGE": _build_change_prompt,
    "COMPARE": _build_compare_prompt,
    "BREAKDOWN": _build_breakdown_prompt,
    "SUMMARY": _build_summary_prompt,
}


# ── Chart data builder ────────────────────────────────────────────────────────

def _build_chart_data(aggregated_data: dict, intent_type: str, analysis_type: str) -> list[dict]:
    """Convert aggregated engine data into chart-ready data points."""
    chart_data: list[dict] = []

    # ── SUMMARY ───────────────────────────────────────────────────────────
    if intent_type == "SUMMARY":
        if analysis_type == "categorical_summary":
            for cat_val, stats in aggregated_data.items():
                if isinstance(stats, dict):
                    chart_data.append({
                        "name": str(cat_val),
                        "value": stats.get("count", 0),
                        "percentage": stats.get("percentage", 0),
                    })
            return chart_data

        # Numeric summary → one row per metric, with mean/min/max keys
        for metric_name, stats in aggregated_data.items():
            if not isinstance(stats, dict):
                continue
            if "mean" in stats:
                chart_data.append({
                    "name": metric_name,
                    "mean": stats.get("mean"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "median": stats.get("median"),
                    # "value" alias for plain BarChart fallback
                    "value": stats.get("mean"),
                })
            elif "unique_values" in stats:
                for val, cnt in stats.get("top_values", {}).items():
                    chart_data.append({"name": str(val), "value": cnt})
        return chart_data

    # ── CHANGE period_over_period ─────────────────────────────────────────
    if intent_type == "CHANGE" and analysis_type == "period_over_period":
        for group, vals in aggregated_data.items():
            if isinstance(vals, dict) and "period_1" in vals:
                chart_data.append({
                    "name": str(group),
                    "period_1": vals["period_1"],
                    "period_2": vals["period_2"],
                    "delta": vals.get("delta", 0),
                })
        return chart_data

    # ── COMPARE, CHANGE (group_ranking), BREAKDOWN ────────────────────────
    for key, value in aggregated_data.items():
        if isinstance(value, dict):
            entry: dict[str, Any] = {"name": str(key)}
            if "value" in value:
                entry["value"] = value["value"]
            else:
                # Multi-metric COMPARE: include every numeric sub-field
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)) and sub_key != "diff_from_leader":
                        entry[sub_key] = round(float(sub_val), 2)
                # Still need a "value" fallback for single-key bar charts
                if "value" not in entry and entry:
                    first_numeric = next(
                        (v for k, v in entry.items() if k != "name" and isinstance(v, (int, float))),
                        None,
                    )
                    if first_numeric is not None:
                        entry["value"] = first_numeric
            if "percentage" in value:
                entry["percentage"] = value["percentage"]
            if "share_pct" in value:
                entry["percentage"] = value["share_pct"]
            chart_data.append(entry)
        elif isinstance(value, (int, float)):
            chart_data.append({"name": str(key), "value": round(float(value), 2)})

    return chart_data


# ── Main entry point ──────────────────────────────────────────────────────────

_MAX_OUTPUT_TOKENS = 4096
_RETRY_MAX_TOKENS = 8192


def generate_answer(
    question: str,
    intent: dict,
    engine_result: dict,
    row_count: int,
    filename: str,
) -> dict:
    """Call Gemini to produce a plain-language answer from aggregated data.

    Args:
        question: The original user question.
        intent: Parsed intent dict from the intent parser.
        engine_result: Output from the data engine.
        row_count: Total rows in the session dataset.
        filename: Name of the uploaded file.

    Returns:
        Dict with answer, chart_type, chart_data, source_ref, metric_used,
        confidence_note.
    """
    intent_type = intent.get("intent", "SUMMARY")
    analysis_type = engine_result.get("analysis_type", "unknown")

    prompt_builder = _PROMPT_BUILDERS.get(intent_type, _build_summary_prompt)
    user_prompt = prompt_builder(question, intent, engine_result)

    logger.debug("Gemini user prompt:\n%s", user_prompt)

    # First attempt
    answer_text = _call_gemini(user_prompt, _MAX_OUTPUT_TOKENS)

    # If truncated, retry with higher token limit
    if answer_text is not None and _looks_truncated(answer_text):
        logger.warning("Response appears truncated, retrying with higher token limit.")
        retry_text = _call_gemini(user_prompt, _RETRY_MAX_TOKENS)
        if retry_text:
            answer_text = retry_text

    if not answer_text:
        answer_text = "Unable to generate an answer. Please try rephrasing your question."

    # Ensure answer ends with proper punctuation
    answer_text = _ensure_complete(answer_text)

    # Chart type: engine hint takes precedence
    chart_type = engine_result.get("chart_hint", _intent_default_chart(intent_type))
    if analysis_type == "period_over_period":
        chart_type = "grouped_bar"

    chart_data = _build_chart_data(engine_result["aggregated_data"], intent_type, analysis_type)

    confidence_note = (
        f"Based on {row_count:,} rows in {filename} · "
        f"Metric: {engine_result['metric_used']}"
    )

    return {
        "answer": answer_text,
        "chart_type": chart_type,
        "chart_data": chart_data,
        "source_ref": engine_result["source_ref"],
        "metric_used": engine_result["metric_used"],
        "confidence_note": confidence_note,
    }


def _call_gemini(user_prompt: str, max_tokens: int) -> str | None:
    """Make a single Gemini API call and return the text."""
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_BASE,
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        text = response.text.strip() if response.text else ""

        # Check finish reason
        try:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) in ("MAX_TOKENS", "2"):
                logger.warning("Gemini response truncated (MAX_TOKENS) at %d tokens.", max_tokens)
        except (AttributeError, IndexError):
            pass

        return text
    except Exception:
        logger.exception("Gemini API call failed.")
        return None


def _looks_truncated(text: str) -> bool:
    """Heuristic: does the text look like it was cut off mid-sentence?"""
    if not text:
        return True
    text = text.strip()
    # Doesn't end with sentence-ending punctuation
    if not text.endswith((".", "!", "?", '"', "'")):
        return True
    # Ends with common truncation patterns
    if text.endswith(("...", " the", " and", " or", " to", " a", " an")):
        return True
    return False


def _ensure_complete(text: str) -> str:
    """Ensure the answer text ends with a complete sentence."""
    text = text.strip()
    if not text:
        return text

    # If it ends properly, return as-is
    if text.endswith((".", "!", "?")):
        return text

    # Try to find the last complete sentence
    for i in range(len(text) - 1, max(0, len(text) - 100), -1):
        if text[i] in ".!?":
            return text[:i + 1]

    # Last resort: just add a period
    return text + "."


def _intent_default_chart(intent: str) -> str:
    return {"CHANGE": "bar", "COMPARE": "bar", "BREAKDOWN": "pie", "SUMMARY": "bar"}.get(intent, "bar")
