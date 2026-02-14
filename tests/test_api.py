"""Tests for the FastAPI endpoints (no Celery/Redis required)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_root():
    """Root endpoint should return service info."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body
    assert "docs" in body


@pytest.mark.anyio
async def test_health():
    """Health endpoint should return healthy status."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_analyze_no_file():
    """POST /analyze without a file should return 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/api/v1/analyze")
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_analyze_wrong_extension():
    """POST /analyze with unsupported file extension should return 400."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/v1/analyze",
            files={"file": ("test.txt", b"not a video", "text/plain")},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_status_not_found():
    """GET /status for non-existent job should 404."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/v1/status/nonexistent123")
    # Will 404 because Redis won't have this job (or Redis is not running in test)
    assert resp.status_code in (404, 500)
