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
1. Write 3–5 complete sentences. No bullet points. No headers. Plain prose only.
2. Lead with the single most important finding (biggest number, largest change, dominant group).
3. Always state specific numbers from the data. Never say "significantly higher" without a number.
4. Use plain English. If you use a statistical word (e.g. "median"), define it in parentheses.
5. Do not invent anything not present in the aggregated data provided to you.
6. Refer to columns by their human meaning, not their technical name
   (e.g. say "age" not "the Age column", say "sales" not "the Sales_Amount field").
7. Do not reveal raw individual records or any personally identifiable information.
8. End with one actionable sentence: what the user could explore next.
9. CRITICAL: You MUST finish every sentence. Never stop mid-sentence.
10. Stick to what the data shows. Do not use external knowledge to infer context."""


# ── Intent-specific prompt builders ──────────────────────────────────────────

def _build_change_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "group_ranking")
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "group")

    if analysis_type == "period_over_period":
        instruction = (
            f"The data shows period-over-period changes grouped by {group_col}. "
            "'period_1' is the earlier half of the data, 'period_2' is the later half. "
            "'delta' is the absolute change (positive = increase, negative = decrease). "
            "'pct_change' is the percentage change. "
            "Identify the group with the largest absolute change and state whether it increased or decreased. "
            "If multiple groups moved in the same direction, note that pattern."
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
        "Conclude with what this comparison suggests."
    )


def _build_breakdown_prompt(question: str, intent: dict, result: dict) -> str:
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "category")
    total_groups = result.get("total_groups", len(result["aggregated_data"]))
    show_pct = result.get("show_pct", True)
    secondary = result.get("secondary_breakdown")
    secondary_col = result.get("secondary_group_col")

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
        f"{pct_note}\n\n"
        f"Breakdown data:\n{data_json}"
        f"{extra}\n\n"
        "Instructions: Start with the largest segment and its value"
        + (" and percentage share" if show_pct else "")
        + ". Then cover the second and third largest. "
        "If an 'Other' bucket exists, mention it. "
        "Conclude with whether one group dominates or values are spread evenly."
    )


def _build_summary_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "summary")

    if analysis_type == "categorical_summary":
        data_json = json.dumps(result["aggregated_data"], indent=2)
        group_col = result.get("group_col", "column")
        total_groups = result.get("total_groups", len(result["aggregated_data"]))
        return (
            f'User question: "{question}"\n'
            f"Column: {group_col} ({total_groups} unique values)\n\n"
            f"Value distribution (count + percentage of total):\n{data_json}\n\n"
            "Instructions: State the total number of unique values. "
            "Name the most frequent value and its percentage. "
            "Describe how evenly or unevenly distributed the values are. "
            "Cover all values shown in the data."
        )

    data_json = json.dumps(result["aggregated_data"], indent=2)
    return (
        f'User question: "{question}"\n'
        f"Metric(s): {result['metric_used']}\n\n"
        f"Statistical summary (one entry per metric column):\n{data_json}\n\n"
        "Instructions: For each metric, state the mean (average) and the range (min to max). "
        "If 'trend' is present, mention it and the 'trend_pct_change'. "
        "If 'outlier_count' is non-zero, note how many outliers exist. "
        "Keep the answer focused and useful for a non-technical user."
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
        # Frontend renders as grouped bar (mean bar + error range, or separate bars)
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

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_BASE,
            max_output_tokens=2048,
            temperature=0.2,      # lower = more factual, less hallucination
        ),
    )

    # Check finish reason to detect truncation
    answer_text = response.text.strip() if response.text else ""
    try:
        finish_reason = response.candidates[0].finish_reason
        if str(finish_reason) in ("MAX_TOKENS", "2"):  # 2 is the proto enum value
            logger.warning("Gemini response was truncated (MAX_TOKENS). Consider raising limit.")
            # Append a safe notice rather than returning a broken sentence
            if answer_text and not answer_text.endswith((".", "!", "?")):
                answer_text = answer_text.rsplit(" ", 1)[0] + "."
    except (AttributeError, IndexError):
        pass

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


def _intent_default_chart(intent: str) -> str:
    return {"CHANGE": "bar", "COMPARE": "bar", "BREAKDOWN": "pie", "SUMMARY": "bar"}.get(intent, "bar")
