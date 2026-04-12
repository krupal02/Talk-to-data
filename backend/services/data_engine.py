"""
Data engine – executes pandas operations based on the classified intent.

Each intent handler produces aggregated results, a human-readable source
reference, a metric label, and structured metadata that the Gemini service
uses to generate a grounded, accurate answer.

Raw row data is NEVER returned; only aggregates reach Gemini.
"""

import logging
import re
from typing import Any

import pandas as pd

from backend.services.semantic_layer import resolve_metric

logger = logging.getLogger(__name__)


# ── String normalisation helpers ──────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _tokenize(s: str) -> set[str]:
    """Split on non-alphanumeric and CamelCase boundaries."""
    spaced = re.sub(r"([A-Z])", r" \1", str(s))
    return set(re.findall(r"[a-z0-9]+", spaced.lower()))


_ID_LIKE = {"id", "index", "idx", "uid", "unnamed0", "rowid", "row"}

def _is_id_column(col_name: str) -> bool:
    norm = _normalize(col_name)
    if norm in _ID_LIKE:
        return True
    tokens = _tokenize(col_name)
    if "id" in tokens or "idx" in tokens or "uid" in tokens or "index" in tokens:
        return True
    return False


# ── Column resolution ─────────────────────────────────────────────────────────

def _find_best_column(df: pd.DataFrame, metric_name: str) -> str | None:
    """Resolve a metric name to a DataFrame column using a priority chain."""
    if not metric_name or metric_name == "unknown":
        return None

    norm_target = _normalize(metric_name)
    col_norm_map = {_normalize(c): c for c in df.columns}

    # 1. Exact normalized match
    if norm_target in col_norm_map:
        return col_norm_map[norm_target]

    # 2. Semantic-layer canonical name
    resolved = resolve_metric(metric_name)
    if resolved:
        canon_norm = _normalize(resolved["canonical_name"])
        if canon_norm in col_norm_map:
            return col_norm_map[canon_norm]
        # Also try aliases
        for alias in resolved.get("aliases", []):
            alias_norm = _normalize(alias)
            if alias_norm in col_norm_map:
                return col_norm_map[alias_norm]

    # 3. Substring match (target contained in column or vice-versa)
    for col_norm, col_original in col_norm_map.items():
        if norm_target in col_norm or col_norm in norm_target:
            return col_original

    # 4. Token-intersection fallback
    metric_tokens = _tokenize(metric_name)
    best_col, best_score = None, 0
    for col in df.columns:
        score = len(metric_tokens & _tokenize(col))
        if score > best_score:
            best_score, best_col = score, col

    return best_col if best_score > 0 else None


def _find_dimension_columns(df: pd.DataFrame, dimensions: list[str]) -> list[str]:
    """Map requested dimension names to actual DataFrame column names."""
    matched: list[str] = []
    col_norm_map = {_normalize(c): c for c in df.columns}
    for dim in dimensions:
        dim_norm = _normalize(dim)
        if not dim_norm:
            continue
        if dim_norm in col_norm_map:
            matched.append(col_norm_map[dim_norm])
        else:
            for col_norm, col_original in col_norm_map.items():
                if dim_norm in col_norm or col_norm in dim_norm:
                    matched.append(col_original)
                    break
    # Deduplicate
    seen, result = set(), []
    for c in matched:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _best_numeric_column(df: pd.DataFrame, exclude: list[str] | None = None) -> str | None:
    """Return the most metric-like numeric column, ignoring ID-like columns."""
    exclude = set(exclude or [])
    candidates = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude and not _is_id_column(c)
    ]
    return candidates[0] if candidates else None


def _best_category_columns(df: pd.DataFrame, exclude: list[str] | None = None, max_cols: int = 2) -> list[str]:
    """Return non-numeric columns suitable for grouping (≤50 unique values)."""
    exclude = set(exclude or [])
    cats = [
        c for c in df.columns
        if c not in exclude
        and not _is_id_column(c)
        and (df[c].dtype == "object" or str(df[c].dtype) == "category")
        and df[c].nunique() <= 50
    ]
    return cats[:max_cols]


def _detect_time_column(df: pd.DataFrame) -> str | None:
    """Heuristically identify a date/time column."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        norm = _normalize(col)
        if any(kw in norm for kw in ("date", "time", "month", "year", "week", "day", "period")):
            return col
    return None


_AGG_KEYWORD_MAP = {
    # mean / average
    "average": "mean", "avg": "mean", "mean": "mean",
    # sum / total
    "total": "sum", "sum": "sum",
    # count
    "count": "count", "number of": "count", "how many": "count",
    # unique
    "unique": "nunique", "distinct": "nunique",
    # min / max
    "minimum": "min", "min": "min", "lowest": "min", "smallest": "min",
    "maximum": "max", "max": "max", "highest": "max", "largest": "max",
    # median
    "median": "median", "middle": "median",
}


def _extract_agg_from_question(question: str) -> str | None:
    """Return an explicit aggregation function if the question states one clearly."""
    q = question.lower()
    # Longest-match first to catch "number of" before "number"
    for phrase in sorted(_AGG_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q:
            return _AGG_KEYWORD_MAP[phrase]
    return None


def _resolve_agg(df: pd.DataFrame, metric_col: str, metric_name: str, question: str = "", intent_agg: str | None = None) -> str:
    """Choose the correct aggregation function for a column.

    Priority:
      1. Explicit aggregation extracted by the intent parser (intent_agg field)
      2. Explicit aggregation word found in the raw question text
      3. Semantic-layer definition for the metric
      4. Default: "sum" for numeric, "nunique" for categorical
    """
    def _guard(agg: str) -> str:
        """Ensure aggregation is compatible with column dtype."""
        if not pd.api.types.is_numeric_dtype(df[metric_col].dropna()):
            return "nunique"
        return agg

    # 1. Intent parser already extracted the aggregation — highest trust
    if intent_agg and intent_agg not in (None, "null", ""):
        return _guard(intent_agg)

    # 2. Keyword scan of raw question
    if question:
        explicit = _extract_agg_from_question(question)
        if explicit:
            return _guard(explicit)

    # 3. Semantic-layer definition
    resolved = resolve_metric(metric_name)
    agg_func = resolved["aggregation"] if resolved else "sum"

    # 4. Dtype guard
    if not pd.api.types.is_numeric_dtype(df[metric_col].dropna()):
        agg_func = "nunique" if agg_func == "sum" else "count"

    return agg_func


def _source_ref(df: pd.DataFrame, cols: list[str]) -> str:
    col_str = ", ".join(cols) if cols else "all columns"
    return f"rows 1–{len(df):,}, column(s): {col_str}"


# ── Intent handlers ───────────────────────────────────────────────────────────

def handle_change(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """CHANGE – show how a metric is distributed across groups, with delta context.

    When a time column exists, computes an actual period-over-period delta.
    Otherwise ranks groups by the metric so the biggest contributors are clear.
    """
    primary_metric = intent.get("target_metrics", ["unknown"])[0]
    metric_col = _find_best_column(df, primary_metric) or _best_numeric_column(df)
    if metric_col is None:
        return _fallback_summary(df)

    dim_cols = _find_dimension_columns(df, intent.get("dimensions", []))
    if not dim_cols:
        dim_cols = _best_category_columns(df, exclude=[metric_col])

    question = intent.get("question", "")
    intent_agg = intent.get("aggregation") or None
    agg_func = _resolve_agg(df, metric_col, primary_metric, question, intent_agg)
    metric_label = f"{metric_col} ({agg_func})"

    time_col = _detect_time_column(df)

    # ── Period-over-period delta when a time column exists ────────────────
    if time_col and time_col != metric_col:
        try:
            df2 = df.copy()
            df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
            df2 = df2.dropna(subset=[time_col])

            midpoint = df2[time_col].median()
            period_a = df2[df2[time_col] <= midpoint]
            period_b = df2[df2[time_col] > midpoint]

            if dim_cols:
                group_col = dim_cols[0]
                agg_a = period_a.groupby(group_col)[metric_col].agg(agg_func)
                agg_b = period_b.groupby(group_col)[metric_col].agg(agg_func)
                combined = pd.DataFrame({"period_1": agg_a, "period_2": agg_b}).fillna(0)
                combined["delta"] = combined["period_2"] - combined["period_1"]
                combined["pct_change"] = (
                    (combined["delta"] / combined["period_1"].replace(0, float("nan"))) * 100
                ).round(1)
                combined = combined.sort_values("delta", key=abs, ascending=False)

                aggregated_data = {
                    str(k): {
                        "period_1": round(float(v["period_1"]), 2),
                        "period_2": round(float(v["period_2"]), 2),
                        "delta": round(float(v["delta"]), 2),
                        "pct_change": round(float(v["pct_change"]), 1) if pd.notna(v["pct_change"]) else None,
                    }
                    for k, v in combined.head(10).iterrows()
                }
                return {
                    "aggregated_data": aggregated_data,
                    "source_ref": _source_ref(df, [metric_col, time_col, group_col]),
                    "metric_used": metric_label,
                    "chart_hint": "bar",
                    "analysis_type": "period_over_period",
                }
        except Exception:
            logger.debug("Period-over-period failed, falling back to group ranking.", exc_info=True)

    # ── Fallback: rank groups by metric value ────────────────────────────
    if dim_cols:
        group_col = dim_cols[0]
        grouped = df.groupby(group_col)[metric_col].agg(agg_func).sort_values(ascending=False)
        total = grouped.sum() if agg_func in ("sum", "count") else grouped.mean()
        aggregated_data = {}
        for k, v in grouped.head(10).items():
            pct = round((v / total) * 100, 1) if total else 0
            aggregated_data[str(k)] = {"value": round(float(v), 2), "share_pct": pct}
    else:
        total = df[metric_col].agg(agg_func)
        aggregated_data = {"total": {"value": round(float(total), 2), "share_pct": 100.0}}

    return {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, [metric_col] + (dim_cols[:1] if dim_cols else [])),
        "metric_used": metric_label,
        "chart_hint": "bar",
        "analysis_type": "group_ranking",
    }


def handle_compare(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """COMPARE – side-by-side comparison of groups across one or more metrics.

    If multiple target metrics are specified, each is computed for every group.
    The result includes absolute values and relative difference from the leader.
    """
    target_metrics = intent.get("target_metrics", ["unknown"])
    dim_cols = _find_dimension_columns(df, intent.get("dimensions", []))

    # Resolve all requested metric columns
    metric_cols: list[str] = []
    for m in target_metrics:
        col = _find_best_column(df, m)
        if col and col not in metric_cols:
            metric_cols.append(col)
    if not metric_cols:
        col = _best_numeric_column(df)
        if col:
            metric_cols = [col]
    if not metric_cols:
        return _fallback_summary(df)

    if not dim_cols:
        dim_cols = _best_category_columns(df, exclude=metric_cols)

    primary_metric_col = metric_cols[0]
    question = intent.get("question", "")
    intent_agg = intent.get("aggregation") or None
    agg_func = _resolve_agg(df, primary_metric_col, target_metrics[0], question, intent_agg)

    if dim_cols:
        group_col = dim_cols[0]
        frames = {}
        for mc in metric_cols:
            af = _resolve_agg(df, mc, mc, question, intent_agg)
            frames[mc] = df.groupby(group_col)[mc].agg(af)

        result_df = pd.DataFrame(frames).fillna(0).sort_values(primary_metric_col, ascending=False)
        top = result_df.head(10)
        leader_val = top[primary_metric_col].iloc[0] if len(top) else 1

        aggregated_data = {}
        for group_val, row in top.iterrows():
            entry = {}
            for mc in metric_cols:
                entry[mc] = round(float(row[mc]), 2)
            if leader_val and primary_metric_col in entry:
                diff_from_leader = round(entry[primary_metric_col] - float(leader_val), 2)
                entry["diff_from_leader"] = diff_from_leader
            aggregated_data[str(group_val)] = entry
    else:
        # No grouping dimension – compare the metrics against each other
        aggregated_data = {}
        for mc in metric_cols:
            af = _resolve_agg(df, mc, mc, question, intent_agg)
            val = df[mc].agg(af)
            aggregated_data[mc] = {"value": round(float(val), 2)}

    return {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, metric_cols + dim_cols[:1]),
        "metric_used": ", ".join(f"{c} ({_resolve_agg(df, c, c, question, intent_agg)})" for c in metric_cols),
        "chart_hint": "bar",
        "analysis_type": "comparison",
    }


def handle_breakdown(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """BREAKDOWN – decompose a metric by one or more dimensions with percentages."""
    primary_metric = intent.get("target_metrics", ["unknown"])[0]
    metric_col = _find_best_column(df, primary_metric) or _best_numeric_column(df)
    if metric_col is None:
        return _fallback_summary(df)

    dim_cols = _find_dimension_columns(df, intent.get("dimensions", []))
    if not dim_cols:
        dim_cols = _best_category_columns(df, exclude=[metric_col])

    agg_func = _resolve_agg(df, metric_col, primary_metric, intent.get("question", ""), intent.get("aggregation") or None)
    metric_label = f"{metric_col} ({agg_func})"

    if dim_cols:
        group_col = dim_cols[0]
        grouped = df.groupby(group_col)[metric_col].agg(agg_func).sort_values(ascending=False)
        total = grouped.sum()
        n_groups = len(grouped)

        aggregated_data = {}
        for k, v in grouped.head(10).items():
            pct = round((v / total) * 100, 1) if total else 0
            aggregated_data[str(k)] = {"value": round(float(v), 2), "percentage": pct}

        # Append "Other" bucket if more than 10 groups
        if n_groups > 10:
            other_sum = float(grouped.iloc[10:].sum())
            other_pct = round((other_sum / total) * 100, 1) if total else 0
            aggregated_data["Other"] = {"value": round(other_sum, 2), "percentage": other_pct}

        # Include secondary dimension if present
        secondary_data: dict | None = None
        if len(dim_cols) >= 2:
            group_col_2 = dim_cols[1]
            grouped2 = df.groupby(group_col_2)[metric_col].agg(agg_func).sort_values(ascending=False)
            total2 = grouped2.sum()
            secondary_data = {
                str(k): {"value": round(float(v), 2), "percentage": round((v / total2) * 100, 1) if total2 else 0}
                for k, v in grouped2.head(8).items()
            }
    else:
        total = df[metric_col].agg(agg_func)
        aggregated_data = {"total": {"value": round(float(total), 2), "percentage": 100.0}}
        secondary_data = None

    result = {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, [metric_col] + dim_cols),
        "metric_used": metric_label,
        "chart_hint": "pie" if (dim_cols and len(aggregated_data) <= 8) else "bar",
        "analysis_type": "breakdown",
        "group_col": dim_cols[0] if dim_cols else None,
        "total_groups": n_groups if dim_cols else 1,
    }
    if secondary_data:
        result["secondary_breakdown"] = secondary_data
        result["secondary_group_col"] = dim_cols[1]
    return result


def handle_summary(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """SUMMARY – rich descriptive statistics for all requested metrics.

    Always returns a consistent dict shape: one key per metric column,
    each containing a stats sub-dict. This prevents the shape-switching
    that confused the Gemini prompts in the original implementation.

    Special case: if the user mentions a categorical column (e.g. "tell me about
    the species column"), redirect to a frequency breakdown of that column so the
    answer is actually useful instead of falling back to unrelated numeric stats.
    """
    target_metrics = intent.get("target_metrics", ["unknown"])

    # Resolve all requested metric columns
    metric_cols: list[str] = []
    cat_cols: list[str] = []          # categorical columns mentioned by the user
    for m in target_metrics:
        col = _find_best_column(df, m)
        if col and col not in metric_cols:
            if pd.api.types.is_numeric_dtype(df[col].dropna()):
                metric_cols.append(col)
            else:
                cat_cols.append(col)  # user asked about a categorical column

    # If the user specifically asked about a categorical column, give a
    # frequency breakdown of it rather than dumping unrelated numeric stats.
    if cat_cols and not metric_cols:
        cat_col = cat_cols[0]
        series = df[cat_col].dropna()
        top_vals = series.value_counts().head(10)
        total = len(series)
        aggregated_data = {
            str(k): {"count": int(v), "percentage": round((v / total) * 100, 1)}
            for k, v in top_vals.items()
        }
        n_groups = series.nunique()
        other_count = total - int(top_vals.sum())
        if other_count > 0:
            aggregated_data["Other"] = {
                "count": other_count,
                "percentage": round((other_count / total) * 100, 1),
            }
        return {
            "aggregated_data": aggregated_data,
            "source_ref": _source_ref(df, [cat_col]),
            "metric_used": f"{cat_col} (frequency)",
            "chart_hint": "pie" if n_groups <= 8 else "bar",
            "analysis_type": "categorical_summary",
            "group_col": cat_col,
            "total_groups": n_groups,
        }

    if not metric_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if not _is_id_column(c)
        ]
        metric_cols = numeric_cols[:4] if numeric_cols else []

    if not metric_cols:
        return _fallback_summary(df)

    aggregated_data: dict[str, dict] = {}
    for col in metric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            stats: dict[str, Any] = {
                "count": int(len(series)),
                "sum": round(float(series.sum()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "std_dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            }
            # Outlier count: values beyond 2 std devs
            mean, std = series.mean(), series.std()
            if std > 0:
                stats["outlier_count"] = int(((series - mean).abs() > 2 * std).sum())

            # Trend direction from first vs second half
            if len(series) >= 6:
                mid = len(series) // 2
                first_half_mean = float(series.iloc[:mid].mean())
                second_half_mean = float(series.iloc[mid:].mean())
                pct_change = (
                    ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100
                    if first_half_mean != 0 else 0.0
                )
                stats["trend"] = (
                    "increasing" if pct_change > 2
                    else "decreasing" if pct_change < -2
                    else "stable"
                )
                stats["trend_pct_change"] = round(pct_change, 1)
        else:
            # Categorical column
            top_vals = series.value_counts().head(5)
            stats = {
                "count": int(len(series)),
                "unique_values": int(series.nunique()),
                "most_frequent": str(series.mode().iloc[0]) if not series.empty else "N/A",
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
            }

        aggregated_data[col] = stats

    if not aggregated_data:
        return _fallback_summary(df)

    # Use grouped_bar when multiple numeric metrics are present so the frontend
    # can render one group per metric (mean/min/max bars side-by-side), keeping
    # the x-axis clean. Single-metric summaries stay as plain bar.
    chart_hint = "grouped_bar" if len([c for c in metric_cols if c in aggregated_data and "mean" in aggregated_data[c]]) > 1 else "bar"

    return {
        "aggregated_data": aggregated_data,  # Always keyed by column name
        "source_ref": _source_ref(df, metric_cols),
        "metric_used": ", ".join(f"{c} (summary)" for c in metric_cols),
        "chart_hint": chart_hint,
        "analysis_type": "summary",
    }


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generic summary when no matching column can be found."""
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not _is_id_column(c)
    ]
    summary: dict[str, Any] = {}
    for col in numeric_cols[:5]:
        summary[col] = {
            "mean": round(float(df[col].mean()), 2),
            "sum": round(float(df[col].sum()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
        }
    return {
        "aggregated_data": summary or {"info": f"Dataset has {len(df):,} rows and {len(df.columns)} columns"},
        "source_ref": f"rows 1–{len(df):,}, all numeric columns",
        "metric_used": "general summary",
        "chart_hint": "bar",
        "analysis_type": "fallback",
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_HANDLERS = {
    "CHANGE": handle_change,
    "COMPARE": handle_compare,
    "BREAKDOWN": handle_breakdown,
    "SUMMARY": handle_summary,
}


def run_query(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """Dispatch the query to the appropriate intent handler.

    Args:
        df: The session DataFrame.
        intent: Parsed intent dict from the intent parser.

    Returns:
        Dict with aggregated_data, source_ref, metric_used, chart_hint, analysis_type.
    """
    handler = _HANDLERS.get(intent.get("intent", "SUMMARY"), handle_summary)
    try:
        return handler(df, intent)
    except Exception:
        logger.exception("Handler %s failed, using fallback.", intent.get("intent"))
        return _fallback_summary(df)