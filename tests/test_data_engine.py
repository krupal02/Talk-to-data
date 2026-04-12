"""
Tests for the data engine and sanitiser modules.

Verifies that pandas operations produce correct aggregated results
and that the sanitiser correctly strips PII columns.
"""

import pandas as pd
import pytest

from backend.services.data_engine import (
    handle_breakdown,
    handle_change,
    handle_compare,
    handle_summary,
    run_query,
)
from backend.utils.sanitiser import sanitise_dataframe


# ─── Sample data fixture ─────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small DataFrame mimicking sales data."""
    return pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01",
                 "2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"],
        "region": ["North", "South", "North", "South",
                    "East", "West", "East", "West"],
        "product": ["Widget A", "Widget A", "Widget A", "Widget A",
                     "Widget B", "Widget B", "Widget B", "Widget B"],
        "revenue": [12000, 9000, 10000, 7000, 11000, 8000, 9500, 6500],
        "orders": [100, 80, 90, 60, 95, 70, 85, 55],
    })


# ─── Data engine tests ───────────────────────────────────────────

def test_change_computes_delta(sample_df):
    """CHANGE handler should compute aggregated values by dimension."""
    intent = {
        "intent": "CHANGE",
        "target_metric": "revenue",
        "dimensions": ["region"],
        "time_range": "last month",
    }
    result = handle_change(sample_df, intent)

    assert "aggregated_data" in result
    assert "source_ref" in result
    assert "metric_used" in result
    assert len(result["aggregated_data"]) > 0
    # North region should have highest total revenue (12000 + 10000 = 22000)
    assert "North" in result["aggregated_data"]


def test_breakdown_sorts_descending(sample_df):
    """BREAKDOWN handler should return groups sorted by value descending."""
    intent = {
        "intent": "BREAKDOWN",
        "target_metric": "revenue",
        "dimensions": ["region"],
        "time_range": None,
    }
    result = handle_breakdown(sample_df, intent)

    values = [v["value"] for v in result["aggregated_data"].values()]
    assert values == sorted(values, reverse=True), "Groups should be sorted descending by value"


def test_breakdown_has_percentages(sample_df):
    """BREAKDOWN handler should include percentage contributions."""
    intent = {
        "intent": "BREAKDOWN",
        "target_metric": "revenue",
        "dimensions": ["region"],
        "time_range": None,
    }
    result = handle_breakdown(sample_df, intent)

    for entry in result["aggregated_data"].values():
        assert "percentage" in entry
        assert 0 <= entry["percentage"] <= 100


def test_compare_groups_by_dimension(sample_df):
    """COMPARE handler should group metric values by the dimension."""
    intent = {
        "intent": "COMPARE",
        "target_metric": "revenue",
        "dimensions": ["product"],
        "time_range": None,
    }
    result = handle_compare(sample_df, intent)

    assert "Widget A" in result["aggregated_data"]
    assert "Widget B" in result["aggregated_data"]


def test_summary_contains_key_stats(sample_df):
    """SUMMARY handler should return count, sum, mean, min, max."""
    intent = {
        "intent": "SUMMARY",
        "target_metric": "revenue",
        "dimensions": [],
        "time_range": None,
    }
    result = handle_summary(sample_df, intent)

    data = result["aggregated_data"]
    assert "count" in data
    assert "sum" in data
    assert "mean" in data
    assert "min" in data
    assert "max" in data
    assert data["count"] == 8
    assert data["sum"] == 73000.0


def test_run_query_dispatches_correctly(sample_df):
    """run_query should dispatch to the correct handler based on intent."""
    intent = {"intent": "SUMMARY", "target_metric": "orders", "dimensions": [], "time_range": None}
    result = run_query(sample_df, intent)
    assert "aggregated_data" in result
    assert result["aggregated_data"]["count"] == 8


def test_fallback_when_metric_not_found(sample_df):
    """When the target metric doesn't exist, the engine should still return data."""
    intent = {
        "intent": "SUMMARY",
        "target_metric": "nonexistent_metric",
        "dimensions": [],
        "time_range": None,
    }
    result = run_query(sample_df, intent)
    assert "aggregated_data" in result
    assert "source_ref" in result


# ─── Sanitiser tests ─────────────────────────────────────────────

def test_sanitiser_removes_pii():
    """Columns with PII-indicating names should be removed."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "email": ["a@test.com", "b@test.com"],
        "phone": ["123", "456"],
        "revenue": [100, 200],
        "SSN": ["111-22-3333", "444-55-6666"],
    })
    cleaned = sanitise_dataframe(df)

    assert "email" not in cleaned.columns
    assert "phone" not in cleaned.columns
    assert "SSN" not in cleaned.columns
    assert "name" in cleaned.columns
    assert "revenue" in cleaned.columns


def test_sanitiser_handles_no_pii():
    """If no PII columns exist, the DataFrame should be unchanged."""
    df = pd.DataFrame({
        "region": ["North", "South"],
        "revenue": [100, 200],
    })
    cleaned = sanitise_dataframe(df)
    assert list(cleaned.columns) == ["region", "revenue"]
    assert len(cleaned) == 2


def test_sanitiser_case_insensitive():
    """PII detection should be case-insensitive."""
    df = pd.DataFrame({
        "Email_Address": ["a@test.com"],
        "PHONE_NUMBER": ["123"],
        "revenue": [100],
    })
    cleaned = sanitise_dataframe(df)

    assert "Email_Address" not in cleaned.columns
    assert "PHONE_NUMBER" not in cleaned.columns
    assert "revenue" in cleaned.columns
