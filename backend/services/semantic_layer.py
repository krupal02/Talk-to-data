"""
Semantic layer – provides a shared metric dictionary and column aliasing.

Business teams often refer to the same concept by different names
(e.g. "revenue", "sales", "income"). The semantic layer normalises
these into a single canonical metric with a defined aggregation so
that all queries produce consistent results regardless of phrasing.
"""


import re

METRICS: dict[str, dict] = {
    "revenue": {
        "aliases": ["sales", "income", "turnover", "total_sales", "net_sales"],
        "aggregation": "sum",
        "description": "Total monetary value of sales transactions",
    },
    "orders": {
        "aliases": ["transactions", "purchases", "order_count", "num_orders"],
        "aggregation": "count",
        "description": "Number of individual orders or transactions",
    },
    "active_users": {
        "aliases": ["dau", "mau", "users", "unique_users", "user_count"],
        "aggregation": "nunique",
        "description": "Count of distinct active users",
    },
    "churn": {
        "aliases": ["attrition", "cancellations", "lost_customers", "churn_rate"],
        "aggregation": "count",
        "description": "Number of customers who stopped using the service",
    },
    "profit": {
        "aliases": ["net_profit", "earnings", "margin", "net_income"],
        "aggregation": "sum",
        "description": "Net profit after costs are subtracted from revenue",
    },
    "quantity": {
        "aliases": ["units", "items_sold", "volume", "qty"],
        "aggregation": "sum",
        "description": "Total number of units or items sold",
    },
}

def _normalize_string(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Pre-build a reverse lookup from every alias to its canonical metric
_ALIAS_MAP: dict[str, str] = {}
for metric_name, meta in METRICS.items():
    _ALIAS_MAP[_normalize_string(metric_name)] = metric_name
    for alias in meta["aliases"]:
        _ALIAS_MAP[_normalize_string(alias)] = metric_name


def resolve_metric(user_term: str) -> dict | None:
    """Resolve a user-supplied term to its canonical metric definition.

    Args:
        user_term: The word or phrase the user typed (e.g. "sales").

    Returns:
        A dictionary with keys ``canonical_name``, ``aggregation``, and
        ``description``, or ``None`` if no match is found.
    """
    canonical = _ALIAS_MAP.get(_normalize_string(user_term))
    if canonical is None:
        return None
    meta = METRICS[canonical]
    return {
        "canonical_name": canonical,
        "aggregation": meta["aggregation"],
        "description": meta["description"],
    }


def get_metric_definition(metric_name: str) -> str:
    """Return a human-readable definition string for a metric."""
    meta = METRICS.get(metric_name)
    if meta is None:
        return "No definition available."
    return f"{metric_name} ({meta['aggregation']}): {meta['description']}"


def list_all_metrics() -> list[dict]:
    """Return a list of all known metrics with their aliases."""
    result = []
    for name, meta in METRICS.items():
        result.append(
            {
                "name": name,
                "aliases": meta["aliases"],
                "aggregation": meta["aggregation"],
                "description": meta["description"],
            }
        )
    return result
