# Copyright 2026 Talk-to-Data Contributors
# Licensed under the Apache License, Version 2.0

"""
Application configuration module.

Reads environment variables from a .env file and exposes a single
Settings object used across the backend.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    """Centralised application settings loaded from environment variables."""

    def __init__(self):
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.max_upload_rows: int = int(os.getenv("MAX_UPLOAD_ROWS", "100000"))
        self.allowed_extensions: list[str] = os.getenv(
            "ALLOWED_EXTENSIONS", "csv,db,sqlite"
        ).split(",")

    def validate(self) -> None:
        """Raise an error if critical settings are missing."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


settings = Settings()


