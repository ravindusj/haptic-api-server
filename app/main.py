"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

# ── Logging ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App lifespan ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Upload dir : %s", settings.UPLOAD_DIR)
    logger.info("Results dir: %s", settings.RESULTS_DIR)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


# ── Create app ───────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Intelligent haptic pattern generator for videos. "
        "Upload a video → AI + DSP analysis → AHAP haptic file. "
        "Detects bass, impacts, explosions, beats and suppresses "
        "dialogue and ambient noise for immersive vibration feedback."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow mobile app access) ──────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────

app.include_router(router)


# ── Root redirect ────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
    }
