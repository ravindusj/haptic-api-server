"""Celery application and async task definitions.

Pipeline: extract_audio → [analyze_dsp + analyze_video] → classify_ai → fuse_scores → generate_ahap
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


# ── Analysis data persistence for visualization ─────────


def _save_analysis_data(
    job_id: str,
    dsp,
    ai,
    video,
    timeline,
) -> None:
    """Persist intermediate analysis data for the /visualize endpoint."""
    results_dir = Path(settings.RESULTS_DIR) / job_id
    results_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = results_dir / f"{job_id}_analysis.json"

    data = {
        "dsp": {
            "sample_rate": dsp.sample_rate,
            "hop_length": dsp.hop_length,
            "duration_seconds": dsp.duration_seconds,
            "rms_energy": dsp.rms_energy,
            "percussive_rms": dsp.percussive_rms,
            "harmonic_rms": dsp.harmonic_rms,
            "spectral_centroid": dsp.spectral_centroid,
            "sub_bass_energy": dsp.sub_bass_energy,
            "bass_energy": dsp.bass_energy,
            "low_mid_energy": dsp.low_mid_energy,
            "mid_energy": dsp.mid_energy,
            "presence_energy": dsp.presence_energy,
            "brilliance_energy": dsp.brilliance_energy,
            "beat_times": dsp.beat_times,
            "beat_strengths": dsp.beat_strengths,
        },
        "ai": {
            "haptic_scores": ai.haptic_scores,
            "speech_scores": ai.speech_scores,
            "dominant_classes": ai.dominant_classes,
            "speech_segments": [
                {"start": s.start, "end": s.end, "confidence": s.confidence}
                for s in ai.speech_segments
            ],
        },
        "video": None,
        "timeline": {
            "intensity_envelope": timeline.intensity_envelope,
            "sharpness_envelope": timeline.sharpness_envelope,
            "envelope_fps": timeline.envelope_fps,
            "duration_seconds": timeline.duration_seconds,
            "events": [
                {
                    "time": e.time,
                    "event_type": e.event_type,
                    "intensity": e.intensity,
                    "sharpness": e.sharpness,
                }
                for e in timeline.events
            ],
        },
    }

    if video is not None:
        data["video"] = {
            "motion_intensity": video.motion_intensity,
            "scene_changes": [
                {"time": sc.time, "magnitude": sc.magnitude}
                for sc in video.scene_changes
            ],
            "dominant_actions": video.dominant_actions,
            "action_scores": video.action_scores,
            "action_window_duration_s": video.action_window_duration_s,
        }

    with open(analysis_path, "w") as f:
        json.dump(data, f)

    logger.info("[%s] Analysis data saved: %s", job_id, analysis_path)


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
    3. AI sound event classification (YAMNet + Whisper)
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
        logger.info("[%s] Step 1/6: Extracting audio…", job_id)

        from app.services.audio_extractor import extract_audio
        audio_result = extract_audio(video_path, job_id)
        duration = audio_result["duration"]

        _set_job_status(
            job_id, "extracting_audio", progress=20.0,
            duration=duration,
        )

        # ── Step 2: DSP + Video Analysis (parallel) ──────
        _set_job_status(job_id, "analyzing_dsp", progress=25.0)
        logger.info("[%s] Step 2/6: DSP + Video analysis (parallel)…", job_id)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from app.services.dsp_analyzer import analyze_dsp
        from app.services.video_analyzer import analyze_video

        video_features = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            dsp_future = pool.submit(analyze_dsp, audio_result["librosa_wav"])
            vid_future = pool.submit(analyze_video, video_path)

            for future in as_completed([dsp_future, vid_future]):
                if future is dsp_future:
                    dsp_features = future.result()
                    _set_job_status(job_id, "analyzing_dsp", progress=40.0)
                    logger.info("[%s] DSP analysis done", job_id)
                else:
                    try:
                        video_features = future.result()
                        _set_job_status(
                            job_id, "analyzing_video", progress=45.0,
                        )
                        logger.info(
                            "[%s] Video analysis done – dominant: %s",
                            job_id,
                            video_features.dominant_actions[:3]
                            if video_features else "none",
                        )
                    except Exception as ve:
                        logger.warning(
                            "[%s] Video analysis failed (non-fatal): %s",
                            job_id, ve,
                        )
                        video_features = None

        _set_job_status(job_id, "analyzing_dsp", progress=48.0)

        # ── Step 3: AI Classification ────────────────────
        _set_job_status(job_id, "classifying_ai", progress=50.0)
        logger.info("[%s] Step 3/6: AI classification…", job_id)

        from app.services.ai_classifier import classify_audio
        ai_result = classify_audio(audio_result["classifier_wav"])

        _set_job_status(job_id, "classifying_ai", progress=70.0)

        # ── Step 4: Score Fusion ─────────────────────────
        _set_job_status(job_id, "scoring", progress=75.0)
        logger.info("[%s] Step 4/6: Score fusion…", job_id)

        from app.services.haptic_scorer import fuse_scores
        timeline = fuse_scores(
            dsp=dsp_features,
            ai=ai_result,
            sensitivity=sensitivity,
            bass_boost=bass_boost,
            video=video_features,
        )

        _set_job_status(job_id, "scoring", progress=85.0)

        # ── Persist analysis data for visualization ──────
        try:
            _save_analysis_data(
                job_id, dsp_features, ai_result, video_features, timeline,
            )
        except Exception as save_err:
            logger.warning("[%s] Analysis data save failed (non-fatal): %s", job_id, save_err)

        # ── Step 5: AHAP Generation ─────────────────────
        _set_job_status(job_id, "generating_ahap", progress=90.0)
        logger.info("[%s] Step 5/6: Generating AHAP…", job_id)

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
