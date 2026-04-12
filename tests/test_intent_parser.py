"""
Tests for the intent parser module.

These tests mock the Google Gemini API to verify that the parser correctly
classifies questions into the four intent categories.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.services.intent_parser import parse_intent


def _mock_gemini_response(content_text):
    """Build a mock Gemini API response."""
    mock_response = MagicMock()
    mock_response.text = content_text
    return mock_response


@patch("backend.services.intent_parser.genai")
def test_change_intent_detected(mock_genai):
    """'Why did sales drop?' should be classified as CHANGE intent."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        '{"intent": "CHANGE", "target_metric": "sales", "dimensions": ["region"], "time_range": "last month"}'
    )

    result = parse_intent(
        "Why did sales drop last month?",
        ["date", "region", "product", "sales", "orders"],
    )

    assert result["intent"] == "CHANGE"
    assert result["target_metric"] == "sales"
    assert "region" in result["dimensions"]


@patch("backend.services.intent_parser.genai")
def test_compare_intent_detected(mock_genai):
    """'region A vs region B' should be classified as COMPARE intent."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        '{"intent": "COMPARE", "target_metric": "revenue", "dimensions": ["region"], "time_range": null}'
    )

    result = parse_intent(
        "Compare region A vs region B revenue",
        ["date", "region", "product", "revenue", "orders"],
    )

    assert result["intent"] == "COMPARE"
    assert result["target_metric"] == "revenue"


@patch("backend.services.intent_parser.genai")
def test_breakdown_intent_detected(mock_genai):
    """'breakdown by product' should be classified as BREAKDOWN intent."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        '{"intent": "BREAKDOWN", "target_metric": "revenue", "dimensions": ["product"], "time_range": null}'
    )

    result = parse_intent(
        "Show me a breakdown of revenue by product",
        ["date", "region", "product", "revenue", "orders"],
    )

    assert result["intent"] == "BREAKDOWN"
    assert "product" in result["dimensions"]


@patch("backend.services.intent_parser.genai")
def test_summary_intent_detected(mock_genai):
    """'weekly summary' should be classified as SUMMARY intent."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        '{"intent": "SUMMARY", "target_metric": "orders", "dimensions": [], "time_range": "this week"}'
    )

    result = parse_intent(
        "Give me a weekly summary of orders",
        ["date", "region", "product", "revenue", "orders"],
    )

    assert result["intent"] == "SUMMARY"
    assert result["target_metric"] == "orders"


@patch("backend.services.intent_parser.genai")
def test_invalid_intent_defaults_to_summary(mock_genai):
    """An unrecognised intent should default to SUMMARY."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        '{"intent": "UNKNOWN", "target_metric": "revenue", "dimensions": [], "time_range": null}'
    )

    result = parse_intent(
        "What is the revenue?",
        ["date", "region", "product", "revenue"],
    )

    assert result["intent"] == "SUMMARY"


@patch("backend.services.intent_parser.genai")
def test_malformed_json_defaults_gracefully(mock_genai):
    """If Gemini returns non-JSON, the parser should not crash."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_response(
        "I think this is a summary question."
    )

    result = parse_intent(
        "Tell me about revenue",
        ["date", "region", "revenue"],
    )

    assert result["intent"] == "SUMMARY"
    assert "target_metric" in result
