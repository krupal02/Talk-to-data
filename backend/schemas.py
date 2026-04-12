# Copyright 2026 Talk-to-Data Contributors
# Licensed under the Apache License, Version 2.0

"""
Pydantic request and response models for API endpoints.

These schemas enforce type safety and produce clear validation errors
for every inbound request and outbound response.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    session_id: str = Field(
        ..., description="Unique session identifier for subsequent queries"
    )
    columns: list[str] = Field(
        ..., description="List of column names in the uploaded dataset"
    )
    row_count: int = Field(
        ..., description="Number of rows in the cleaned dataset"
    )
    preview: list[dict] = Field(
        ..., description="First five rows of data for preview"
    )


class QueryRequest(BaseModel):
    """Payload for a natural-language query against an uploaded dataset."""

    session_id: str = Field(
        ..., description="Session ID returned by the upload endpoint"
    )
    question: str = Field(
        ..., description="Natural-language question about the data"
    )


class QueryResponse(BaseModel):
    """Structured response containing the answer, chart, and provenance."""

    answer: str = Field(
        ..., description="Plain-language answer to the user's question"
    )
    chart_type: str = Field(
        ..., description="Recommended chart type: bar, pie, or line"
    )
    chart_data: list[dict] = Field(
        ..., description="Data points for the chart visualisation"
    )
    source_ref: str = Field(
        ..., description="Row/column reference used to derive the answer"
    )
    metric_used: str = Field(
        ..., description="Name and aggregation of the metric applied"
    )
    confidence_note: str = Field(
        ..., description="One-line note about data coverage and confidence"
    )
