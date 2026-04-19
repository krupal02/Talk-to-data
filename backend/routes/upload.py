"""
Upload route – handles file uploads, sanitisation, and session creation.

POST /upload accepts a CSV or SQLite file via multipart/form-data,
validates and sanitises it, stores the cleaned DataFrame in the
server-side session store, and returns column info with a preview.
"""

import io
import sqlite3
import tempfile
import uuid
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas import UploadResponse
from backend.utils.sanitiser import (
    sanitise_dataframe,
    truncate_if_needed,
    validate_extension,
    validate_file_size,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Server-side session store – maps session_id → (DataFrame, filename)
session_store: dict[str, tuple[pd.DataFrame, str]] = {}


def _auto_detect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns that look like dates into datetime64.

    Tries pd.to_datetime on each object column with a sample of values.
    Only converts if >50% of non-null values parse successfully — this avoids
    false positives on columns like 'name' or 'category'.

    Supports both pandas 2.x (object dtype) and pandas 3.x (StringDtype).
    """
    # Detect string-like columns (both 'object' and pandas 3.x 'string'/'str')
    string_cols = []
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if dtype_str in ("object", "string", "str") or "string" in dtype_str:
            string_cols.append(col)

    for col in string_cols:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        # Quick check: sample up to 20 values for performance
        test_sample = sample.head(20)
        try:
            converted = pd.to_datetime(test_sample, errors="coerce", format="mixed")
            success_rate = converted.notna().sum() / len(test_sample)
            if success_rate > 0.5:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                logger.info("Auto-detected date column: %s (%.0f%% parse rate)", col, success_rate * 100)
        except Exception:
            pass  # Not a date column, skip silently
    return df


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a CSV or SQLite file upload (max 50 MB).

    The file is validated, PII columns are stripped, and a cleaned
    DataFrame is stored server-side keyed by a unique session ID.

    Returns:
        UploadResponse with session_id, columns, row_count, and a 5-row preview.

    Raises:
        HTTPException 400: Invalid file type, empty file, or oversized file.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a CSV or SQLite file.",
        )

    # Read file content
    content = await file.read()

    if not validate_file_size(len(content)):
        raise HTTPException(
            status_code=400,
            detail="File exceeds the 50 MB size limit.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Parse into DataFrame
    extension = file.filename.rsplit(".", 1)[-1].lower()

    try:
        if extension == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif extension in ("db", "sqlite"):
            df = _read_sqlite(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to parse uploaded file.")
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the file. Please check the format. Error: {str(exc)}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=400, detail="The uploaded file contains no data."
        )

    # Sanitise PII columns
    df = sanitise_dataframe(df)

    # Auto-detect and convert date columns stored as strings
    df = _auto_detect_dates(df)

    # Truncate if over the row limit
    df, was_truncated = truncate_if_needed(df)

    # Store in session
    session_id = str(uuid.uuid4())
    session_store[session_id] = (df, file.filename)

    # Build preview (first 5 rows)
    preview = df.head(5).fillna("").to_dict(orient="records")

    truncation_note = ""
    if was_truncated:
        truncation_note = " (dataset was truncated to the maximum allowed rows)"

    logger.info(
        "Session %s created: %d rows, %d columns%s",
        session_id,
        len(df),
        len(df.columns),
        truncation_note,
    )

    return UploadResponse(
        session_id=session_id,
        columns=df.columns.tolist(),
        row_count=len(df),
        preview=preview,
    )


def _read_sqlite(content: bytes) -> pd.DataFrame:
    """Read the first user table from an in-memory SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    conn = sqlite3.connect(tmp_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            raise HTTPException(
                status_code=400,
                detail="SQLite database contains no tables.",
            )
        df = pd.read_sql_query(f"SELECT * FROM [{tables[0]}]", conn)
    finally:
        conn.close()

    return df
