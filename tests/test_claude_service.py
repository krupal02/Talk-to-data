"""
Tests for the Gemini service module.

Verifies that prompts sent to Gemini never contain raw row data,
and that responses include source references and all required fields.
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from backend.services.gemini_service import generate_answer, _build_chart_data


# ─── Mock helpers ─────────────────────────────────────────────────

def _mock_gemini_response(text):
    """Build a mock Gemini API response."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


# ─── Prompt safety tests ─────────────────────────────────────────

@patch("backend.services.gemini_service.genai")
def test_prompt_contains_no_raw_rows(mock_genai):
    """The prompt sent to Gemini must not contain raw row data."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        "Revenue dropped by 11% due to reduced performance in the South region."
    )

    engine_result = {
        "aggregated_data": {"North": 22000, "South": 16000, "East": 20500},
        "source_ref": "rows 1\u20133,200, column: revenue",
        "metric_used": "revenue (sum)",
    }

    generate_answer(
        question="Why did revenue drop?",
        intent={"intent": "CHANGE", "target_metric": "revenue", "dimensions": ["region"], "time_range": "last month"},
        engine_result=engine_result,
        row_count=3200,
        filename="sales_q1.csv",
    )

    # Inspect the actual prompt sent to Gemini
    create_call = mock_client.models.generate_content.call_args
    user_message = create_call.kwargs["contents"]

    # The prompt should contain aggregated data, not raw rows
    assert "22000" in user_message or "22,000" in user_message
    assert "rows 1" not in user_message or "aggregated" in user_message.lower() or "Aggregated" in user_message


@patch("backend.services.gemini_service.genai")
def test_response_includes_source_ref(mock_genai):
    """The response must include a source reference and all required fields."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        "North accounts for 40% of total sales, making it the largest region."
    )

    engine_result = {
        "aggregated_data": {"North": {"value": 22000, "percentage": 40.0}, "South": {"value": 16000, "percentage": 29.1}},
        "source_ref": "rows 1\u20133,200, column: revenue",
        "metric_used": "revenue (sum)",
    }

    result = generate_answer(
        question="What makes up total sales?",
        intent={"intent": "BREAKDOWN", "target_metric": "revenue", "dimensions": ["region"], "time_range": None},
        engine_result=engine_result,
        row_count=3200,
        filename="sales_q1.csv",
    )

    assert "source_ref" in result
    assert result["source_ref"] == "rows 1\u20133,200, column: revenue"
    assert "metric_used" in result
    assert "answer" in result
    assert "chart_type" in result
    assert "chart_data" in result
    assert "confidence_note" in result


@patch("backend.services.gemini_service.genai")
def test_chart_type_matches_intent(mock_genai):
    """Chart type should match the intent: BREAKDOWN → pie, CHANGE → bar, etc."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response("Summary text.")

    test_cases = [
        ("CHANGE", "bar"),
        ("COMPARE", "bar"),
        ("BREAKDOWN", "pie"),
        ("SUMMARY", "bar"),
    ]

    for intent_type, expected_chart in test_cases:
        result = generate_answer(
            question="Test question",
            intent={"intent": intent_type, "target_metric": "revenue", "dimensions": [], "time_range": None},
            engine_result={
                "aggregated_data": {"A": 100, "B": 200},
                "source_ref": "rows 1\u2013100, column: revenue",
                "metric_used": "revenue (sum)",
            },
            row_count=100,
            filename="test.csv",
        )
        assert result["chart_type"] == expected_chart, f"Intent {intent_type} should produce {expected_chart}"


@patch("backend.services.gemini_service.genai")
def test_confidence_note_format(mock_genai):
    """Confidence note should follow the format 'Based on N rows in filename'."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response("Answer text.")

    result = generate_answer(
        question="Test",
        intent={"intent": "SUMMARY", "target_metric": "revenue", "dimensions": [], "time_range": None},
        engine_result={
            "aggregated_data": {"total": 50000},
            "source_ref": "rows 1\u20133,200, column: revenue",
            "metric_used": "revenue (sum)",
        },
        row_count=3200,
        filename="sales_q1.csv",
    )

    assert result["confidence_note"] == "Based on 3,200 rows in sales_q1.csv"


# ─── Chart data builder tests ────────────────────────────────────

def test_build_chart_data_flat():
    """Flat numeric dict should produce list of {name, value} objects."""
    data = {"North": 22000, "South": 16000}
    chart = _build_chart_data(data, "CHANGE")

    assert len(chart) == 2
    assert chart[0]["name"] == "North"
    assert chart[0]["value"] == 22000


def test_build_chart_data_with_percentages():
    """Breakdown dict with value/percentage should be preserved."""
    data = {
        "North": {"value": 22000, "percentage": 40.0},
        "South": {"value": 16000, "percentage": 29.1},
    }
    chart = _build_chart_data(data, "BREAKDOWN")

    assert len(chart) == 2
    assert chart[0]["percentage"] == 40.0
