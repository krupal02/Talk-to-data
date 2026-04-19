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


_ID_LIKE = {"id", "index", "idx", "uid", "unnamed0", "rowid", "row", "no", "num", "number", "sr", "sl"}


def _is_id_column(col_name: str) -> bool:
    """Return True if the column looks like a surrogate key / row index."""
    norm = _normalize(col_name)
    if norm in _ID_LIKE:
        return True
    tokens = _tokenize(col_name)
    # Must be more than just an ID token — e.g. "passenger_id" but not "student_grade"
    id_tokens = {"id", "idx", "uid", "index"}
    if tokens and tokens.issubset(id_tokens | {"unnamed", "row", "no", "num", "number", "sr", "sl"}):
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

    # 2. Semantic-layer canonical name + aliases
    resolved = resolve_metric(metric_name)
    if resolved:
        for candidate in [resolved["canonical_name"]] + resolved.get("aliases", []):
            cn = _normalize(candidate)
            if cn in col_norm_map:
                return col_norm_map[cn]

    # 3. Substring match — require the shorter string to be ≥3 chars
    for col_norm, col_original in col_norm_map.items():
        shorter = norm_target if len(norm_target) <= len(col_norm) else col_norm
        if len(shorter) >= 3 and (norm_target in col_norm or col_norm in norm_target):
            return col_original

    # 4. Token-intersection fallback (≥1 shared token of ≥2 chars)
    metric_tokens = {t for t in _tokenize(metric_name) if len(t) >= 2}
    best_col, best_score = None, 0
    for col in df.columns:
        col_tokens = {t for t in _tokenize(col) if len(t) >= 2}
        score = len(metric_tokens & col_tokens)
        if score > best_score:
            best_score, best_col = score, col

    return best_col if best_score > 0 else None


def _find_dimension_columns(df: pd.DataFrame, dimensions: list[str]) -> list[str]:
    """Map requested dimension names to actual DataFrame column names."""
    matched: list[str] = []
    col_norm_map = {_normalize(c): c for c in df.columns}
    for dim in dimensions:
        dim_norm = _normalize(dim)
        if not dim_norm or len(dim_norm) < 2:
            continue
        if dim_norm in col_norm_map:
            matched.append(col_norm_map[dim_norm])
        else:
            for col_norm, col_original in col_norm_map.items():
                shorter = dim_norm if len(dim_norm) <= len(col_norm) else col_norm
                if len(shorter) >= 3 and (dim_norm in col_norm or col_norm in dim_norm):
                    matched.append(col_original)
                    break
    # Deduplicate preserving order
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


def _best_category_columns(
    df: pd.DataFrame, exclude: list[str] | None = None, max_cols: int = 2
) -> list[str]:
    """Return non-numeric columns suitable for grouping (≤50 unique values)."""
    exclude = set(exclude or [])
    cats = [
        c for c in df.columns
        if c not in exclude
        and not _is_id_column(c)
        and (
            df[c].dtype == "object" 
            or str(df[c].dtype).startswith("category") 
            or "string" in str(df[c].dtype).lower()
        )
        and df[c].nunique() <= 50
    ]
    return cats[:max_cols]


def _detect_time_column(df: pd.DataFrame) -> str | None:
    """Identify a date/time column, avoiding false positives.

    Only matches datetime-typed columns or columns whose name contains
    a time keyword as a whole token (not just a substring).
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # Check by name — require the keyword to be a standalone token
    time_keywords = {"date", "time", "month", "year", "week", "period", "timestamp"}
    for col in df.columns:
        tokens = set(_tokenize(col))
        if tokens & time_keywords:
            return col
    return None


# ── Aggregation resolution ────────────────────────────────────────────────────

_AGG_KEYWORD_MAP = {
    "number of": "count", "how many": "count",
    "average": "mean", "avg": "mean", "mean": "mean",
    "total": "sum", "sum": "sum",
    "count": "count",
    "unique": "nunique", "distinct": "nunique",
    "minimum": "min", "smallest": "min", "lowest": "min",
    "maximum": "max", "largest": "max",
    "median": "median",
}


def _extract_agg_from_question(question: str) -> str | None:
    """Return an explicit aggregation function from the question (longest-match first)."""
    q = question.lower()
    for phrase in sorted(_AGG_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q:
            return _AGG_KEYWORD_MAP[phrase]
    return None


def _resolve_agg(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    question: str = "",
    intent_agg: str | None = None,
) -> str:
    """Choose the correct aggregation function for a column.

    Priority:
      1. intent_agg — explicit aggregation extracted by the intent parser
      2. Question keyword scan
      3. Semantic-layer definition
      4. Default: "sum" for numeric, "nunique" for categorical
    """
    def _guard(agg: str) -> str:
        if not pd.api.types.is_numeric_dtype(df[metric_col].dropna()):
            return "nunique"
        # pandas doesn't have a "median" groupby agg by name for some versions
        # — it does since pandas 1.1, so this is safe
        return agg

    if intent_agg and intent_agg not in (None, "null", "none", ""):
        return _guard(intent_agg)

    if question:
        explicit = _extract_agg_from_question(question)
        if explicit:
            return _guard(explicit)

    resolved = resolve_metric(metric_name)
    agg_func = resolved["aggregation"] if resolved else "sum"

    if not pd.api.types.is_numeric_dtype(df[metric_col].dropna()):
        agg_func = "nunique" if agg_func == "sum" else "count"

    return agg_func


def _source_ref(df: pd.DataFrame, cols: list[str]) -> str:
    col_str = ", ".join(c for c in cols if c) or "all columns"
    return f"rows 1–{len(df):,}, column(s): {col_str}"


# ── Intent handlers ───────────────────────────────────────────────────────────

def handle_change(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """CHANGE – identify drivers behind increases or decreases.

    Matches hackathon requirements:
    - Identify the drivers behind increases or decreases
    - Highlight the most influential categories (region, product, channel)
    - Provide clear, concise explanations in everyday language
    - Reference data sources used for the insight
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
            df2 = df2.dropna(subset=[time_col]).sort_values(time_col)

            # Split by row position (not median timestamp) for correctness
            mid = len(df2) // 2
            period_a = df2.iloc[:mid]
            period_b = df2.iloc[mid:]

            # Overall change
            total_a = float(period_a[metric_col].agg(agg_func))
            total_b = float(period_b[metric_col].agg(agg_func))
            overall_delta = total_b - total_a
            overall_pct = round((overall_delta / total_a * 100), 1) if total_a != 0 else None

            # Analyze drivers across ALL categorical dimensions
            all_drivers: dict[str, dict] = {}
            best_driver_col = None
            best_driver_impact = 0.0

            analysis_dims = dim_cols if dim_cols else _best_category_columns(df2, exclude=[metric_col, time_col], max_cols=3)

            for group_col in analysis_dims:
                agg_a = period_a.groupby(group_col)[metric_col].agg(agg_func)
                agg_b = period_b.groupby(group_col)[metric_col].agg(agg_func)
                combined = pd.DataFrame({"period_1": agg_a, "period_2": agg_b}).fillna(0)
                combined["delta"] = combined["period_2"] - combined["period_1"]
                combined["pct_change"] = (
                    (combined["delta"] / combined["period_1"].replace(0, float("nan"))) * 100
                ).round(1)
                combined = combined.sort_values("delta", key=abs, ascending=False)

                driver_data = {
                    str(k): {
                        "period_1": round(float(v["period_1"]), 2),
                        "period_2": round(float(v["period_2"]), 2),
                        "delta": round(float(v["delta"]), 2),
                        "pct_change": round(float(v["pct_change"]), 1) if pd.notna(v["pct_change"]) else None,
                    }
                    for k, v in combined.head(10).iterrows()
                }
                all_drivers[group_col] = driver_data

                # Track which dimension explains the most variance
                max_impact = float(combined["delta"].abs().max()) if len(combined) > 0 else 0
                if max_impact > best_driver_impact:
                    best_driver_impact = max_impact
                    best_driver_col = group_col

            # Build time-series data points for trend visualization
            time_series_data: list[dict] = []
            try:
                ts_grouped = df2.groupby(df2[time_col].dt.to_period("W"))[metric_col].agg(agg_func)
                if len(ts_grouped) > 20:
                    ts_grouped = df2.groupby(df2[time_col].dt.to_period("M"))[metric_col].agg(agg_func)
                for period, val in ts_grouped.items():
                    time_series_data.append({
                        "name": str(period),
                        "value": round(float(val), 2),
                    })
            except Exception:
                pass

            # Primary driver data (most impactful dimension)
            primary_driver_col = best_driver_col or (analysis_dims[0] if analysis_dims else None)
            aggregated_data = all_drivers.get(primary_driver_col, {}) if primary_driver_col else {}

            # Identify top movers (biggest absolute deltas)
            top_movers: list[dict] = []
            if primary_driver_col and primary_driver_col in all_drivers:
                for name, vals in list(all_drivers[primary_driver_col].items())[:3]:
                    direction = "increased" if vals["delta"] > 0 else "decreased"
                    top_movers.append({
                        "group": name,
                        "direction": direction,
                        "delta": vals["delta"],
                        "pct_change": vals["pct_change"],
                    })

            result: dict[str, Any] = {
                "aggregated_data": aggregated_data,
                "source_ref": _source_ref(df, [metric_col, time_col] + analysis_dims),
                "metric_used": metric_label,
                "chart_hint": "bar",
                "analysis_type": "period_over_period",
                "group_col": primary_driver_col,
                "overall_change": {
                    "period_1_total": round(total_a, 2),
                    "period_2_total": round(total_b, 2),
                    "delta": round(overall_delta, 2),
                    "pct_change": overall_pct,
                    "direction": "increased" if overall_delta > 0 else "decreased" if overall_delta < 0 else "unchanged",
                },
                "top_movers": top_movers,
            }

            # Include other driver dimensions for context
            if len(all_drivers) > 1 and primary_driver_col:
                result["secondary_drivers"] = {
                    k: v for k, v in all_drivers.items() if k != primary_driver_col
                }

            # Include time series if available for line chart
            if time_series_data:
                result["time_series"] = time_series_data

            return result

        except Exception:
            logger.debug("Period-over-period failed, falling back to group ranking.", exc_info=True)

    # ── Fallback: rank groups by metric value ─────────────────────────────
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
        "group_col": dim_cols[0] if dim_cols else None,
    }


def handle_compare(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """COMPARE – compute a metric for each group of a categorical dimension.

    The result shows each group's value; the leader row is omitted from
    diff_from_leader to avoid a redundant zero entry.
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

    question = intent.get("question", "")
    intent_agg = intent.get("aggregation") or None
    sort_desc = intent.get("sort_desc", True)

    primary_metric_col = metric_cols[0]
    agg_func = _resolve_agg(df, primary_metric_col, target_metrics[0], question, intent_agg)

    if dim_cols:
        group_col = dim_cols[0]
        frames: dict[str, pd.Series] = {}
        for mc in metric_cols:
            af = _resolve_agg(df, mc, mc, question, intent_agg)
            frames[mc] = df.groupby(group_col)[mc].agg(af)

        result_df = (
            pd.DataFrame(frames)
            .fillna(0)
            .sort_values(primary_metric_col, ascending=not sort_desc)
        )
        top = result_df.head(10)
        leader_val = float(top[primary_metric_col].iloc[0]) if len(top) else None

        aggregated_data: dict[str, Any] = {}
        for i, (group_val, row) in enumerate(top.iterrows()):
            entry: dict[str, Any] = {}
            for mc in metric_cols:
                entry[mc] = round(float(row[mc]), 2)
            # diff_from_leader: skip the leader itself (diff = 0 is noise)
            if i > 0 and leader_val is not None:
                entry["diff_from_leader"] = round(entry[primary_metric_col] - leader_val, 2)
            aggregated_data[str(group_val)] = entry

        metric_label = ", ".join(
            f"{c} ({_resolve_agg(df, c, c, question, intent_agg)})" for c in metric_cols
        )
    else:
        # No grouping dimension — compare metrics against each other
        aggregated_data = {}
        for mc in metric_cols:
            af = _resolve_agg(df, mc, mc, question, intent_agg)
            val = df[mc].agg(af)
            aggregated_data[mc] = {"value": round(float(val), 2)}
        metric_label = ", ".join(
            f"{c} ({_resolve_agg(df, c, c, question, intent_agg)})" for c in metric_cols
        )

    return {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, metric_cols + dim_cols[:1]),
        "metric_used": metric_label,
        "chart_hint": "bar",
        "analysis_type": "comparison",
        "group_col": dim_cols[0] if dim_cols else None,
        "agg_func": agg_func,
    }


def handle_breakdown(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """BREAKDOWN – decompose a metric by one dimension with percentage shares.

    Matches hackathon requirements:
    - Decompose a number into its components (by region/category/product/channel)
    - Surface patterns (e.g., concentration, outliers)
    - Highlight the biggest contributors
    - Provide both table and narrative explanation
    """
    primary_metric = intent.get("target_metrics", ["unknown"])[0]
    metric_col = _find_best_column(df, primary_metric) or _best_numeric_column(df)
    if metric_col is None:
        return _fallback_summary(df)

    dim_cols = _find_dimension_columns(df, intent.get("dimensions", []))
    if not dim_cols:
        dim_cols = _best_category_columns(df, exclude=[metric_col])

    agg_func = _resolve_agg(
        df, metric_col, primary_metric,
        intent.get("question", ""), intent.get("aggregation") or None
    )
    metric_label = f"{metric_col} ({agg_func})"

    # Percentage shares are only sensible for additive aggregations
    show_pct = agg_func in ("sum", "count", "nunique")

    if dim_cols:
        group_col = dim_cols[0]
        grouped = df.groupby(group_col)[metric_col].agg(agg_func).sort_values(ascending=False)
        n_groups = len(grouped)
        total = grouped.sum() if show_pct else None

        aggregated_data: dict[str, Any] = {}
        for k, v in grouped.head(10).items():
            entry: dict[str, Any] = {"value": round(float(v), 2)}
            if show_pct and total:
                entry["percentage"] = round((v / total) * 100, 1)
            aggregated_data[str(k)] = entry

        # "Other" bucket when there are more than 10 groups
        if n_groups > 10 and show_pct and total:
            other_sum = float(grouped.iloc[10:].sum())
            aggregated_data["Other"] = {
                "value": round(other_sum, 2),
                "percentage": round((other_sum / total) * 100, 1),
            }

        # ── Concentration analysis ────────────────────────────────────────
        concentration_info: dict[str, Any] = {}
        if show_pct and total and total > 0:
            top_1_share = round((float(grouped.iloc[0]) / float(total)) * 100, 1) if len(grouped) > 0 else 0
            top_3_share = round((float(grouped.head(3).sum()) / float(total)) * 100, 1) if len(grouped) >= 3 else top_1_share

            concentration_info = {
                "top_1_share": top_1_share,
                "top_3_share": top_3_share,
                "pattern": (
                    "highly concentrated" if top_1_share > 50
                    else "moderately concentrated" if top_3_share > 70
                    else "evenly distributed"
                ),
            }

            # Detect outlier groups (unusually small contributors)
            if len(grouped) > 3:
                group_mean = float(grouped.mean())
                group_std = float(grouped.std())
                if group_std > 0:
                    outlier_groups = [
                        str(k) for k, v in grouped.items()
                        if abs(float(v) - group_mean) > 2 * group_std
                    ]
                    if outlier_groups:
                        concentration_info["outlier_groups"] = outlier_groups[:3]

        # Optional secondary dimension
        secondary_data: dict | None = None
        if len(dim_cols) >= 2:
            group_col_2 = dim_cols[1]
            grouped2 = df.groupby(group_col_2)[metric_col].agg(agg_func).sort_values(ascending=False)
            total2 = grouped2.sum() if show_pct else None
            secondary_data = {}
            for k, v in grouped2.head(8).items():
                entry2: dict[str, Any] = {"value": round(float(v), 2)}
                if show_pct and total2:
                    entry2["percentage"] = round((v / total2) * 100, 1)
                secondary_data[str(k)] = entry2
    else:
        total_val = df[metric_col].agg(agg_func)
        aggregated_data = {"total": {"value": round(float(total_val), 2), "percentage": 100.0}}
        secondary_data = None
        n_groups = 1
        concentration_info = {}

    result: dict[str, Any] = {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, [metric_col] + dim_cols),
        "metric_used": metric_label,
        "chart_hint": "pie" if (dim_cols and len(aggregated_data) <= 8 and show_pct) else "bar",
        "analysis_type": "breakdown",
        "group_col": dim_cols[0] if dim_cols else None,
        "total_groups": n_groups,
        "show_pct": show_pct,
    }
    if concentration_info:
        result["concentration"] = concentration_info
    if secondary_data:
        result["secondary_breakdown"] = secondary_data
        result["secondary_group_col"] = dim_cols[1]
    return result


def handle_summary(df: pd.DataFrame, intent: dict) -> dict[str, Any]:
    """SUMMARY – descriptive statistics with trends, anomalies, and insights.

    Matches hackathon requirements:
    - Scan datasets for trends, anomalies, and important shifts
    - Produce a concise update focused on what truly matters
    - Provide source references
    - Make summaries easy for non-technical users

    Special case: if the user asks about a categorical column (e.g. "tell me
    about the species column"), return a frequency breakdown of that column.
    """
    target_metrics = intent.get("target_metrics", ["unknown"])

    metric_cols: list[str] = []
    cat_cols: list[str] = []

    for m in target_metrics:
        col = _find_best_column(df, m)
        if col:
            if pd.api.types.is_numeric_dtype(df[col].dropna()):
                if col not in metric_cols:
                    metric_cols.append(col)
            else:
                if col not in cat_cols:
                    cat_cols.append(col)

    # Redirect categorical-column questions to a frequency breakdown
    if cat_cols and not metric_cols:
        cat_col = cat_cols[0]
        series = df[cat_col].dropna()
        top_vals = series.value_counts().head(10)
        total = len(series)
        n_groups = series.nunique()
        aggregated_data = {
            str(k): {"count": int(v), "percentage": round((v / total) * 100, 1)}
            for k, v in top_vals.items()
        }
        remaining = total - int(top_vals.sum())
        if remaining > 0:
            aggregated_data["Other"] = {
                "count": remaining,
                "percentage": round((remaining / total) * 100, 1),
            }

        # Detect concentration for categorical columns
        top_1_pct = round((top_vals.iloc[0] / total) * 100, 1) if len(top_vals) > 0 else 0
        concentration = (
            "highly concentrated" if top_1_pct > 60
            else "moderately concentrated" if top_1_pct > 35
            else "evenly distributed"
        )

        return {
            "aggregated_data": aggregated_data,
            "source_ref": _source_ref(df, [cat_col]),
            "metric_used": f"{cat_col} (frequency)",
            "chart_hint": "pie" if n_groups <= 8 else "bar",
            "analysis_type": "categorical_summary",
            "group_col": cat_col,
            "total_groups": n_groups,
            "concentration": concentration,
            "top_1_pct": top_1_pct,
        }

    # Fallback to all non-ID numeric columns when target is unspecified
    if not metric_cols:
        metric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if not _is_id_column(c)
        ][:4]

    if not metric_cols:
        return _fallback_summary(df)

    # Detect time column once for trend analysis
    time_col = _detect_time_column(df)

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
                "percentile_25": round(float(series.quantile(0.25)), 2),
                "percentile_75": round(float(series.quantile(0.75)), 2),
            }

            # Distribution shape — helps identify skewed data
            if len(series) > 3:
                try:
                    skew_val = float(series.skew())
                    stats["skewness"] = round(skew_val, 2)
                    if skew_val > 1:
                        stats["distribution"] = "right-skewed (many low values, few high outliers)"
                    elif skew_val < -1:
                        stats["distribution"] = "left-skewed (many high values, few low outliers)"
                    else:
                        stats["distribution"] = "approximately normal"
                except Exception:
                    pass

            # Outliers: values beyond 2 standard deviations
            mean, std = series.mean(), series.std()
            if std > 0:
                outlier_mask = (series - mean).abs() > 2 * std
                outlier_count = int(outlier_mask.sum())
                stats["outlier_count"] = outlier_count
                if outlier_count > 0:
                    outlier_vals = series[outlier_mask]
                    stats["outlier_examples"] = [
                        round(float(v), 2) for v in outlier_vals.head(3).tolist()
                    ]
                    stats["anomaly_note"] = (
                        f"{outlier_count} anomalous values detected "
                        f"(beyond 2 standard deviations from the mean of {stats['mean']})"
                    )

            # Trend: only meaningful when a time column exists and data is sortable
            if time_col and time_col in df.columns and len(series) >= 6:
                try:
                    sorted_df = (
                        df[[time_col, col]]
                        .dropna()
                        .sort_values(time_col)
                    )
                    sorted_series = sorted_df[col]
                    mid = len(sorted_series) // 2
                    first_half_mean = float(sorted_series.iloc[:mid].mean())
                    second_half_mean = float(sorted_series.iloc[mid:].mean())
                    if first_half_mean != 0:
                        pct_change = ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100
                        stats["trend"] = (
                            "increasing" if pct_change > 2
                            else "decreasing" if pct_change < -2
                            else "stable"
                        )
                        stats["trend_pct_change"] = round(pct_change, 1)
                        stats["first_half_avg"] = round(first_half_mean, 2)
                        stats["second_half_avg"] = round(second_half_mean, 2)
                except Exception:
                    pass  # non-critical; skip trend for this column
        else:
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

    # Cross-column correlations for the primary metric
    if len(metric_cols) >= 2:
        primary = metric_cols[0]
        correlations: dict[str, float] = {}
        for other in metric_cols[1:]:
            try:
                corr = float(df[[primary, other]].corr().iloc[0, 1])
                if not pd.isna(corr):
                    correlations[other] = round(corr, 3)
            except Exception:
                pass
        if correlations:
            aggregated_data[primary]["correlations"] = correlations

    numeric_with_mean = [c for c in metric_cols if c in aggregated_data and "mean" in aggregated_data[c]]
    chart_hint = "grouped_bar" if len(numeric_with_mean) > 1 else "bar"

    return {
        "aggregated_data": aggregated_data,
        "source_ref": _source_ref(df, metric_cols),
        "metric_used": ", ".join(f"{c} (summary)" for c in metric_cols),
        "chart_hint": chart_hint,
        "analysis_type": "summary",
    }


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic stats on all non-ID numeric columns."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _is_id_column(c)]
    summary: dict[str, Any] = {}
    for col in numeric_cols[:5]:
        summary[col] = {
            "mean": round(float(df[col].mean()), 2),
            "sum": round(float(df[col].sum()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
        }
    return {
        "aggregated_data": summary or {
            "info": f"Dataset has {len(df):,} rows and {len(df.columns)} columns"
        },
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
    """Dispatch the query to the appropriate intent handler."""
    handler = _HANDLERS.get(intent.get("intent", "SUMMARY"), handle_summary)
    try:
        return handler(df, intent)
    except Exception:
        logger.exception("Handler %s failed, using fallback.", intent.get("intent"))
        return _fallback_summary(df)