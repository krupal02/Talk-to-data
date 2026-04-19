"""
Semantic layer – provides a shared metric dictionary and column aliasing.

Business teams often refer to the same concept by different names
(e.g. "revenue", "sales", "income"). The semantic layer normalises
these into a single canonical metric with a defined aggregation so
that all queries produce consistent results regardless of phrasing.
"""


import re

METRICS: dict[str, dict] = {
    # ── Finance ───────────────────────────────────────────────────────────
    "revenue": {
        "aliases": ["sales", "income", "turnover", "total_sales", "net_sales",
                     "gross_sales", "sales_amount", "revenue_amount"],
        "aggregation": "sum",
        "description": "Total monetary value of sales transactions",
    },
    "profit": {
        "aliases": ["net_profit", "earnings", "margin", "net_income",
                     "gross_profit", "operating_profit", "ebitda"],
        "aggregation": "sum",
        "description": "Net profit after costs are subtracted from revenue",
    },
    "cost": {
        "aliases": ["expense", "expenses", "expenditure", "spend", "spending",
                     "total_cost", "operating_cost", "cogs"],
        "aggregation": "sum",
        "description": "Total cost or expenditure",
    },
    "price": {
        "aliases": ["unit_price", "selling_price", "cost_price", "mrp",
                     "list_price", "avg_price"],
        "aggregation": "mean",
        "description": "Price per unit",
    },
    "discount": {
        "aliases": ["rebate", "markdown", "discount_amount", "discount_pct",
                     "discount_rate", "savings"],
        "aggregation": "mean",
        "description": "Discount applied to transactions",
    },
    "budget": {
        "aliases": ["allocated_budget", "planned_budget", "budget_amount",
                     "approved_budget", "forecast"],
        "aggregation": "sum",
        "description": "Allocated or planned budget amount",
    },

    # ── Orders / Transactions ─────────────────────────────────────────────
    "orders": {
        "aliases": ["transactions", "purchases", "order_count", "num_orders",
                     "bookings", "deals", "trade_count"],
        "aggregation": "count",
        "description": "Number of individual orders or transactions",
    },
    "quantity": {
        "aliases": ["units", "items_sold", "volume", "qty", "units_sold",
                     "items", "count_sold", "pieces"],
        "aggregation": "sum",
        "description": "Total number of units or items sold",
    },

    # ── Users / Customers ─────────────────────────────────────────────────
    "active_users": {
        "aliases": ["dau", "mau", "users", "unique_users", "user_count",
                     "visitors", "unique_visitors", "signups"],
        "aggregation": "nunique",
        "description": "Count of distinct active users",
    },
    "churn": {
        "aliases": ["attrition", "cancellations", "lost_customers",
                     "churn_rate", "customer_loss", "unsubscribes"],
        "aggregation": "count",
        "description": "Number of customers who stopped using the service",
    },
    "retention": {
        "aliases": ["retention_rate", "renewal", "renewal_rate",
                     "repeat_customers", "returning_customers"],
        "aggregation": "mean",
        "description": "Rate or count of customers retained over a period",
    },

    # ── HR / People ───────────────────────────────────────────────────────
    "salary": {
        "aliases": ["wage", "wages", "compensation", "pay", "income",
                     "annual_salary", "monthly_salary", "ctc", "package"],
        "aggregation": "mean",
        "description": "Employee salary or compensation",
    },
    "age": {
        "aliases": ["employee_age", "customer_age", "user_age", "years_old"],
        "aggregation": "mean",
        "description": "Age of individuals",
    },
    "experience": {
        "aliases": ["years_experience", "tenure", "years_of_service",
                     "work_experience", "seniority"],
        "aggregation": "mean",
        "description": "Years of professional experience",
    },

    # ── Ratings / Scores ──────────────────────────────────────────────────
    "rating": {
        "aliases": ["score", "review_score", "stars", "customer_rating",
                     "satisfaction", "nps", "feedback_score", "grade"],
        "aggregation": "mean",
        "description": "Rating or score given by users",
    },

    # ── Operations / Metrics ──────────────────────────────────────────────
    "duration": {
        "aliases": ["time_taken", "response_time", "handle_time",
                     "processing_time", "lead_time", "cycle_time",
                     "wait_time", "resolution_time", "turnaround_time"],
        "aggregation": "mean",
        "description": "Duration or time taken for a process",
    },
    "weight": {
        "aliases": ["mass", "net_weight", "gross_weight", "kg", "lbs"],
        "aggregation": "mean",
        "description": "Weight or mass measurement",
    },
    "distance": {
        "aliases": ["length", "height", "width", "depth", "km", "miles",
                     "meters", "area", "size"],
        "aggregation": "mean",
        "description": "Distance or dimensional measurement",
    },
    "temperature": {
        "aliases": ["temp", "celsius", "fahrenheit", "heat"],
        "aggregation": "mean",
        "description": "Temperature measurement",
    },

    # ── Support / Service ─────────────────────────────────────────────────
    "tickets": {
        "aliases": ["issues", "complaints", "cases", "incidents",
                     "support_tickets", "requests", "bugs"],
        "aggregation": "count",
        "description": "Number of support tickets or issues",
    },
    "conversion": {
        "aliases": ["conversion_rate", "close_rate", "win_rate",
                     "success_rate", "hit_rate"],
        "aggregation": "mean",
        "description": "Rate of successful conversions",
    },

    # ── General ───────────────────────────────────────────────────────────
    "count": {
        "aliases": ["total_count", "number", "frequency", "occurrences",
                     "records", "entries", "rows"],
        "aggregation": "count",
        "description": "Count of items or occurrences",
    },
    "percentage": {
        "aliases": ["pct", "percent", "rate", "ratio", "proportion", "share"],
        "aggregation": "mean",
        "description": "Percentage or rate value",
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
