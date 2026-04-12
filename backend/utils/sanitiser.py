"""
Upload sanitiser – validates files and strips PII columns before storage.

Enforces file-size limits, allowed extensions, and automatically removes
columns whose names suggest personally identifiable information.
"""

import logging
import re

import pandas as pd

from backend.config import settings

logger = logging.getLogger(__name__)

# Column-name patterns that indicate PII – matched case-insensitively
_PII_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ssn",
        r"social.?security",
        r"password",
        r"passwd",
        r"\bdob\b",
        r"date.?of.?birth",
        r"national.?id",
        r"credit.?card",
        r"\bphone\b",
        r"phone.?number",
        r"\bemail\b",
        r"e.?mail",
    ]
]

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_extension(filename: str) -> bool:
    """Return True if the file extension is in the allow-list."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions


def sanitise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that match PII patterns and return a cleaned copy.

    Logs which columns were removed (by count only — never logs names
    externally) and returns a new DataFrame without those columns.
    """
    pii_columns: list[str] = []
    for col in df.columns:
        for pattern in _PII_PATTERNS:
            if pattern.search(str(col)):
                pii_columns.append(col)
                break

    if pii_columns:
        logger.info(
            "Sanitiser removed %d PII column(s) from the uploaded dataset.",
            len(pii_columns),
        )
        df = df.drop(columns=pii_columns)

    return df


def validate_file_size(size_bytes: int) -> bool:
    """Return True if the file is within the 50 MB limit."""
    return size_bytes <= MAX_FILE_SIZE_BYTES


def truncate_if_needed(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Truncate the DataFrame to MAX_UPLOAD_ROWS if it exceeds the limit.

    Returns a tuple of (possibly truncated DataFrame, was_truncated).
    """
    max_rows = settings.max_upload_rows
    if len(df) > max_rows:
        logger.info(
            "Dataset truncated from %d to %d rows.", len(df), max_rows
        )
        return df.head(max_rows), True
    return df, False
