"""
Query route – accepts a natural-language question and returns an answer.

POST /query orchestrates the full pipeline:
    intent parsing → data engine → Gemini narration

Column schema metadata (types + sample values) is extracted from the
session DataFrame and injected into the intent parser so the model can
ground metric and dimension extraction in the actual dataset structure.
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from google.genai import errors

from backend.routes.upload import session_store
from backend.schemas import QueryRequest, QueryResponse
from backend.services.data_engine import run_query
from backend.services.gemini_service import generate_answer
from backend.services.intent_parser import parse_intent

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of sample values to pass per column to the intent parser
_MAX_SAMPLES_PER_COL = 4
# Maximum number of columns to send sample values for (keeps prompt size reasonable)
_MAX_COLS_WITH_SAMPLES = 30


def _extract_schema_metadata(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list]]:
    """Extract column dtypes and representative sample values from a DataFrame.

    Args:
        df: The session DataFrame.

    Returns:
        col_types:   {column_name: dtype_string}
        col_samples: {column_name: [up to 4 representative values]}
    """
    col_types: dict[str, str] = {}
    col_samples: dict[str, list] = {}

    cols = df.columns.tolist()[:_MAX_COLS_WITH_SAMPLES]

    for col in cols:
        dtype = str(df[col].dtype)
        col_types[col] = dtype

        series = df[col].dropna()
        if len(series) == 0:
            col_samples[col] = []
            continue

        if pd.api.types.is_numeric_dtype(series):
            # For numeric columns: show min, median, max, and one random mid-value
            vals = [series.min(), series.median(), series.max()]
            col_samples[col] = [round(float(v), 4) for v in vals[:_MAX_SAMPLES_PER_COL]]
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_samples[col] = [str(series.min()), str(series.max())]
        else:
            # For categorical/object columns: show most common unique values
            unique_vals = series.value_counts().head(_MAX_SAMPLES_PER_COL).index.tolist()
            col_samples[col] = [str(v) for v in unique_vals]

    return col_types, col_samples


@router.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question about an uploaded dataset.

    Pipeline:
        1. Validate request and look up session DataFrame.
        2. Extract column schema metadata (types + samples).
        3. Parse the user's intent via Gemini (lightweight call, schema-grounded).
        4. Run the data engine to produce aggregated results.
        5. Pass aggregated results + intent to Gemini for plain-language narration.
        6. Return structured QueryResponse.

    Raises:
        HTTPException 400: Empty question.
        HTTPException 404: Session not found.
        HTTPException 429: Gemini quota exhausted.
        HTTPException 502: Gemini client error.
        HTTPException 503: Gemini server/overload error.
        HTTPException 500: Unexpected pipeline failure.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a dataset first.",
        )

    df: pd.DataFrame
    filename: str
    df, filename = session

    try:
        # ── Step 1: Extract schema metadata ──────────────────────────────
        col_types, col_samples = _extract_schema_metadata(df)
        logger.info(
            "Schema extracted: %d columns, types sample: %s",
            len(col_types),
            {k: v for k, v in list(col_types.items())[:5]},
        )

        # ── Step 2: Parse intent (schema-grounded) ────────────────────────
        intent = parse_intent(
            question=request.question,
            available_columns=df.columns.tolist(),
            col_types=col_types,
            col_samples=col_samples,
        )
        # Attach the raw question so the data engine can detect aggregation keywords
        intent["question"] = request.question
        logger.info("Intent parsed: %s", intent)

        # ── Step 3: Run data engine ───────────────────────────────────────
        engine_result = run_query(df, intent)
        logger.info(
            "Engine result: analysis_type=%s, metric=%s, groups=%d",
            engine_result.get("analysis_type"),
            engine_result.get("metric_used"),
            len(engine_result.get("aggregated_data", {})),
        )

        # ── Step 4: Generate answer via Gemini ────────────────────────────
        result = generate_answer(
            question=request.question,
            intent=intent,
            engine_result=engine_result,
            row_count=len(df),
            filename=filename,
        )

        return QueryResponse(**result)

    except HTTPException:
        raise

    except errors.ClientError as exc:
        logger.exception("Gemini API client error: %s", exc)
        if getattr(exc, "code", None) == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Google Gemini API daily quota exhausted. "
                    "Please try again tomorrow or use a different API key."
                ),
            )
        raise HTTPException(status_code=502, detail="AI service error. Please try again.")

    except errors.APIError as exc:
        logger.exception("Gemini API server error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Google Gemini is currently unavailable. Please try again in a few moments.",
        )

    except Exception as exc:
        logger.exception("Query pipeline failed for session %s: %s", request.session_id, exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question. Please try again.",
        ) from exc