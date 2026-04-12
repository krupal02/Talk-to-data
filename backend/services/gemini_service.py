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

STRICT RULES — follow every one of these:
1. Answer in 3–5 sentences. No bullet points. No headers. Plain prose only.
2. Lead with the single most important finding (biggest number, largest change, dominant segment).
3. Mention specific numbers from the data (exact values or percentages). Vague language like
   "significantly higher" without a number is not acceptable.
4. Use plain English. If you must use a statistical word (e.g. "median"), define it in parentheses.
5. Do not invent or infer anything not present in the aggregated data provided.
6. Do not reference column names in a database/technical sense (e.g. do not say "the 'Sales_Amount'
   column"). Instead, refer to it by its human-readable meaning (e.g. "sales").
7. Do not reveal raw individual records or any personally identifiable information.
8. End with one sentence that tells the user what to explore next or what question to ask.
9. Stick strictly to what the numbers show. Do not use external knowledge to infer context
   (e.g. do not assume "Iris" means flowers unless the data says so explicitly)."""


# ── Intent-specific prompt builders ──────────────────────────────────────────

def _build_change_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "group_ranking")
    data_json = json.dumps(result["aggregated_data"], indent=2)

    if analysis_type == "period_over_period":
        instruction = (
            "The data shows period-over-period changes for each group. "
            "'period_1' is the earlier half of the data, 'period_2' is the later half. "
            "'delta' is the absolute change (positive = increase, negative = decrease). "
            "'pct_change' is the percentage change. "
            "Identify the group with the largest absolute change, explain whether it increased or decreased, "
            "and name the top contributor. If most groups moved in the same direction, note that pattern."
        )
    else:
        instruction = (
            "The data shows the current value and percentage share for each group. "
            "Identify the dominant group and its share, then note how the others compare. "
            "Since no direct time comparison is available, focus on the current distribution "
            "and which groups are leading vs lagging."
        )

    return (
        f'User question: "{question}"\n'
        f"Metric: {result['metric_used']}\n"
        f"Time range: {intent.get('time_range') or 'not specified'}\n"
        f"Analysis type: {analysis_type}\n\n"
        f"Data:\n{data_json}\n\n"
        f"Instructions: {instruction}"
    )


def _build_compare_prompt(question: str, intent: dict, result: dict) -> str:
    data_json = json.dumps(result["aggregated_data"], indent=2)
    return (
        f'User question: "{question}"\n'
        f"Metric(s): {result['metric_used']}\n\n"
        f"Comparison data:\n{data_json}\n\n"
        "Instructions: Identify which group or metric performs best. State the winner clearly "
        "and by how much (use the 'diff_from_leader' field if present). Then briefly note "
        "any interesting patterns among the remaining groups. If multiple metrics are compared, "
        "mention which group leads on each metric."
    )


def _build_breakdown_prompt(question: str, intent: dict, result: dict) -> str:
    data_json = json.dumps(result["aggregated_data"], indent=2)
    group_col = result.get("group_col", "category")
    total_groups = result.get("total_groups", len(result["aggregated_data"]))
    secondary = result.get("secondary_breakdown")
    secondary_col = result.get("secondary_group_col")

    extra = ""
    if secondary:
        extra = (
            f"\n\nSecondary breakdown by {secondary_col}:\n"
            f"{json.dumps(secondary, indent=2)}\n"
            "After discussing the primary breakdown, add one sentence about the secondary dimension."
        )

    return (
        f'User question: "{question}"\n'
        f"Metric: {result['metric_used']}\n"
        f"Grouped by: {group_col} ({total_groups} total groups, showing top 10)\n\n"
        f"Breakdown data (value + percentage of total):\n{data_json}"
        f"{extra}\n\n"
        "Instructions: Start with the largest segment and its percentage share. "
        "Then mention the second and third largest. If an 'Other' bucket exists, note it. "
        "Conclude with one observation about concentration (e.g. whether one group dominates)."
    )


def _build_summary_prompt(question: str, intent: dict, result: dict) -> str:
    analysis_type = result.get("analysis_type", "summary")

    if analysis_type == "categorical_summary":
        data_json = json.dumps(result["aggregated_data"], indent=2)
        group_col = result.get("group_col", "category")
        total_groups = result.get("total_groups", len(result["aggregated_data"]))
        return (
            f'User question: "{question}"\n'
            f"Column analysed: {group_col} ({total_groups} unique values)\n\n"
            f"Value distribution (count + % of total rows):\n{data_json}\n\n"
            "Instructions: Tell the user what unique values this column contains, "
            "which value appears most frequently and its percentage, and how evenly or "
            "unevenly distributed the values are. Mention the total number of unique values."
        )

    data_json = json.dumps(result["aggregated_data"], indent=2)
    return (
        f'User question: "{question}"\n'
        f"Metric(s): {result['metric_used']}\n\n"
        f"Statistical summary:\n{data_json}\n\n"
        "Instructions: For each metric, lead with its mean (explain: average) and range (min to max). "
        "If a 'trend' field is present, mention it ('increasing', 'decreasing', or 'stable') and "
        "the 'trend_pct_change' if available. If 'outlier_count' is non-zero, note it. "
        "For categorical fields, mention the most frequent value and how many unique values exist. "
        "Keep the answer focused on what is most useful for a business user."
    )


_PROMPT_BUILDERS = {
    "CHANGE": _build_change_prompt,
    "COMPARE": _build_compare_prompt,
    "BREAKDOWN": _build_breakdown_prompt,
    "SUMMARY": _build_summary_prompt,
}


# ── Chart data builder ────────────────────────────────────────────────────────

def _build_chart_data(aggregated_data: dict, intent_type: str, analysis_type: str) -> list[dict]:
    """Convert aggregated data into a flat list of chart-ready data points."""
    chart_data: list[dict] = []

    if intent_type == "SUMMARY":
        if analysis_type == "categorical_summary":
            # Frequency chart: one bar per category value
            for cat_val, stats in aggregated_data.items():
                if isinstance(stats, dict):
                    chart_data.append({
                        "name": str(cat_val),
                        "value": stats.get("count", 0),
                        "percentage": stats.get("percentage", 0),
                    })
            return chart_data

        # Numeric summary → map primary mean to "value" so regular BarChart can draw it.
        # Ensure we always output {"name": str, "value": number}
        for metric_name, stats in aggregated_data.items():
            if not isinstance(stats, dict):
                continue
            if "mean" in stats:
                chart_data.append({
                    "name": metric_name,
                    "value": stats.get("mean"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                })
            elif "unique_values" in stats:
                # Categorical column inside a numeric summary – show top values
                for val, cnt in stats.get("top_values", {}).items():
                    chart_data.append({"name": f"{metric_name}:{val}", "value": cnt})
        return chart_data

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

    # CHANGE (group_ranking), COMPARE, BREAKDOWN
    for key, value in aggregated_data.items():
        if isinstance(value, dict):
            entry: dict[str, Any] = {"name": str(key)}
            if "value" in value:
                entry["value"] = value["value"]
            else:
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        entry[sub_key] = round(float(sub_val), 2)
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

    # Build the intent-specific user prompt
    prompt_builder = _PROMPT_BUILDERS.get(intent_type, _build_summary_prompt)
    user_prompt = prompt_builder(question, intent, engine_result)

    logger.debug("Gemini user prompt:\n%s", user_prompt)

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_BASE + "\nCRITICAL INSTRUCTION: You MUST completely finish your final sentence. NEVER terminate your response abruptly.",
            max_output_tokens=2048,
            temperature=0.3,
        ),
    )

    answer_text = response.text.strip()

    # Determine chart type from engine hint first, then intent default
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