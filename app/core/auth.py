"""API key authentication middleware."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> None:
    """Validate the API key from the request header.

    If API_KEY is not configured (None or empty), authentication is
    disabled — all requests pass through. This allows local development
    without needing to set up a key.
    """
    settings = get_settings()
    if not settings.API_KEY:
        return  # Auth disabled in dev mode
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
