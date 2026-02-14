"""Celery application and async task definitions.

Pipeline: extract_audio → analyze_dsp → classify_ai → fuse_scores → generate_ahap
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Celery app ───────────────────────────────────────────

celery_app = Celery(
    "haptic_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                 # re-deliver on worker crash
    worker_prefetch_multiplier=1,        # one task at a time (heavy)
    result_expires=86400,                # results expire after 24h
)


# ── Job status helpers (stored in Redis) ─────────────────

def _redis():
    """Return a Redis client from the Celery broker connection."""
    import redis
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _set_job_status(
    job_id: str,
    status: str,
    progress: float = 0.0,
    **extra,
) -> None:
    """Persist job status to Redis."""
    r = _redis()
    key = f"haptic:job:{job_id}"
    data = {
        "status": status,
        "progress": progress,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **{k: str(v) if not isinstance(v, (str, int, float)) else v for k, v in extra.items()},
    }
    r.hset(key, mapping=data)
    r.expire(key, 86400)  # 24h TTL


def get_job_status(job_id: str) -> dict | None:
    """Read job status from Redis."""
    r = _redis()
    key = f"haptic:job:{job_id}"
    data = r.hgetall(key)
    return data if data else None


# ── Main pipeline task ───────────────────────────────────

@celery_app.task(bind=True, name="haptic.analyze_video", max_retries=1)
def analyze_video_task(
    self,
    job_id: str,
    video_path: str,
    sensitivity: float = 0.5,
    style: str = "auto",
    bass_boost: float = 1.0,
) -> dict:
    """
    Full haptic analysis pipeline.

    This is the main Celery task that runs the entire pipeline:
    1. Extract audio from video (FFmpeg)
    2. DSP feature extraction (librosa)
    3. AI sound event classification (PANNs CNN14)
    4. Haptic score fusion (DSP + AI)
    5. AHAP file generation

    Parameters
    ----------
    job_id : str
        Unique job identifier.
    video_path : str
        Path to the uploaded video file.
    sensitivity : float
        0-1 haptic sensitivity.
    style : str
        "auto", "cinematic", or "music".
    bass_boost : float
        0.5-2.0 bass energy multiplier.
    """
    start_time = time.time()

    try:
        # ── Step 1: Extract audio ────────────────────────
        _set_job_status(job_id, "extracting_audio", progress=5.0)
        logger.info("[%s] Step 1/5: Extracting audio…", job_id)

        from app.services.audio_extractor import extract_audio
        audio_result = extract_audio(video_path, job_id)
        duration = audio_result["duration"]

        _set_job_status(
            job_id, "extracting_audio", progress=20.0,
            duration=duration,
        )

        # ── Step 2: DSP Analysis ─────────────────────────
        _set_job_status(job_id, "analyzing_dsp", progress=25.0)
        logger.info("[%s] Step 2/5: DSP analysis…", job_id)

        from app.services.dsp_analyzer import analyze_dsp
        dsp_features = analyze_dsp(audio_result["librosa_wav"])

        _set_job_status(job_id, "analyzing_dsp", progress=45.0)

        # ── Step 3: AI Classification ────────────────────
        _set_job_status(job_id, "classifying_ai", progress=50.0)
        logger.info("[%s] Step 3/5: AI classification…", job_id)

        from app.services.ai_classifier import classify_audio
        ai_result = classify_audio(audio_result["panns_wav"])

        _set_job_status(job_id, "classifying_ai", progress=70.0)

        # ── Step 4: Score Fusion ─────────────────────────
        _set_job_status(job_id, "scoring", progress=75.0)
        logger.info("[%s] Step 4/5: Score fusion…", job_id)

        from app.services.haptic_scorer import fuse_scores
        timeline = fuse_scores(
            dsp=dsp_features,
            ai=ai_result,
            sensitivity=sensitivity,
            bass_boost=bass_boost,
        )

        _set_job_status(job_id, "scoring", progress=85.0)

        # ── Step 5: AHAP Generation ─────────────────────
        _set_job_status(job_id, "generating_ahap", progress=90.0)
        logger.info("[%s] Step 5/5: Generating AHAP…", job_id)

        from app.services.ahap_generator import generate_ahap, save_ahap
        ahap = generate_ahap(timeline, job_id)
        ahap_path = save_ahap(ahap, job_id)

        # ── Done ─────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        file_size = os.path.getsize(ahap_path)

        _set_job_status(
            job_id,
            "completed",
            progress=100.0,
            ahap_path=ahap_path,
            duration=duration,
            total_events=ahap.total_events,
            total_chunks=len(ahap.chunks),
            file_size=file_size,
            processing_time=elapsed,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "[%s] Pipeline complete in %.1fs: %d events, %d chunks, %d bytes",
            job_id, elapsed, ahap.total_events, len(ahap.chunks), file_size,
        )

        # Cleanup temp audio files
        from app.services.audio_extractor import cleanup_job_files
        cleanup_job_files(job_id)

        return {
            "job_id": job_id,
            "status": "completed",
            "ahap_path": ahap_path,
            "duration": duration,
            "total_events": ahap.total_events,
            "total_chunks": len(ahap.chunks),
            "processing_time": elapsed,
        }

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        error_msg = str(exc)[:500]
        logger.exception("[%s] Pipeline failed after %.1fs", job_id, elapsed)

        _set_job_status(
            job_id,
            "failed",
            progress=0.0,
            error=error_msg,
            processing_time=elapsed,
        )

        # Retry once on transient errors
        raise self.retry(exc=exc, countdown=10, max_retries=1)
