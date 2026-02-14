"""API routes for the Haptic Video Analyzer."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.celery_app import analyze_video_task, get_job_status, _set_job_status
from app.core.config import get_settings
from app.models.schemas import (
    AHAPDownloadInfo,
    AnalysisStyle,
    AnalyzeRequest,
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["haptic"])


# ── POST /analyze ────────────────────────────────────────


@router.post(
    "/analyze",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Upload a video for haptic analysis",
    description=(
        "Upload a video file (MP4, MOV, MKV, etc.) and receive a job_id. "
        "The server will extract audio, run DSP + AI analysis, and generate "
        "an AHAP haptic pattern file. Poll /status/{job_id} for progress."
    ),
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    sensitivity: float = Query(
        default=0.5, ge=0.0, le=1.0,
        description="Haptic sensitivity: 0 = fewer/softer, 1 = more/stronger",
    ),
    style: AnalysisStyle = Query(
        default=AnalysisStyle.AUTO,
        description="Analysis style: auto, cinematic (impacts), music (beats)",
    ),
    bass_boost: float = Query(
        default=1.0, ge=0.5, le=2.0,
        description="Bass energy multiplier: >1 = more rumble",
    ),
) -> JobCreatedResponse:
    """Accept a video upload and queue it for haptic analysis."""

    # ── Validate file ────────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # ── Generate job ID & save upload ────────────────────
    job_id = uuid.uuid4().hex[:12]
    upload_dir = Path(settings.UPLOAD_DIR) / job_id
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = f"{job_id}{ext}"
    video_path = str(upload_dir / safe_name)

    # Stream file to disk (handles large files)
    file_size = 0
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            file_size += len(chunk)
            if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                os.remove(video_path)
                raise HTTPException(
                    413,
                    f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
                )
            f.write(chunk)

    logger.info(
        "Upload saved: %s (%.1f MB) → job %s",
        file.filename,
        file_size / 1024 / 1024,
        job_id,
    )

    # ── Initialise job status ────────────────────────────
    _set_job_status(
        job_id, "queued", progress=0.0,
        file_name=file.filename,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # ── Dispatch Celery task ─────────────────────────────
    analyze_video_task.delay(
        job_id=job_id,
        video_path=video_path,
        sensitivity=sensitivity,
        style=style.value,
        bass_boost=bass_boost,
    )

    return JobCreatedResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message=f"Job '{job_id}' queued. Poll GET /status/{job_id} for progress.",
    )


# ── GET /status/{job_id} ─────────────────────────────────


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check job processing status",
)
async def get_status(job_id: str) -> JobStatusResponse:
    """Return the current processing status of a job."""
    data = get_job_status(job_id)
    if not data:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id=job_id,
        status=data.get("status", "queued"),
        progress=float(data.get("progress", 0)),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        error=data.get("error"),
        file_name=data.get("file_name"),
        duration_seconds=float(data["duration"]) if data.get("duration") else None,
    )


# ── GET /result/{job_id} ─────────────────────────────────


@router.get(
    "/result/{job_id}",
    summary="Download the generated AHAP file",
    description="Returns the .ahap file once processing is complete.",
)
async def download_result(job_id: str):
    """Download the generated AHAP haptic pattern file."""
    data = get_job_status(job_id)
    if not data:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    if data.get("status") != "completed":
        raise HTTPException(
            409,
            f"Job is not complete yet. Status: {data.get('status')}",
        )

    ahap_path = data.get("ahap_path")
    if not ahap_path or not os.path.exists(ahap_path):
        raise HTTPException(500, "AHAP file not found on server.")

    return FileResponse(
        ahap_path,
        media_type="application/json",
        filename=f"{job_id}.ahap",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.ahap"'},
    )


# ── GET /result/{job_id}/info ────────────────────────────


@router.get(
    "/result/{job_id}/info",
    response_model=AHAPDownloadInfo,
    summary="Get AHAP file metadata without downloading",
)
async def result_info(job_id: str) -> AHAPDownloadInfo:
    """Return metadata about the generated AHAP file."""
    data = get_job_status(job_id)
    if not data:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    if data.get("status") != "completed":
        raise HTTPException(409, f"Job not complete. Status: {data.get('status')}")

    return AHAPDownloadInfo(
        job_id=job_id,
        download_url=f"{settings.API_V1_PREFIX}/result/{job_id}",
        file_size_bytes=int(data.get("file_size", 0)),
        duration_seconds=float(data.get("duration", 0)),
        total_events=int(data.get("total_events", 0)),
        total_chunks=int(data.get("total_chunks", 1)),
    )


# ── GET /preview/{job_id} ────────────────────────────────


@router.get(
    "/preview/{job_id}",
    summary="Preview haptic timeline as JSON",
    description=(
        "Returns a lightweight timeline visualization showing "
        "event positions, intensities, and types. Useful for debugging."
    ),
)
async def preview_timeline(job_id: str):
    """Return a JSON timeline preview of haptic events."""
    import json

    data = get_job_status(job_id)
    if not data:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    if data.get("status") != "completed":
        raise HTTPException(409, f"Job not complete. Status: {data.get('status')}")

    ahap_path = data.get("ahap_path")
    if not ahap_path or not os.path.exists(ahap_path):
        raise HTTPException(500, "AHAP file not found on server.")

    with open(ahap_path) as f:
        ahap_data = json.load(f)

    # Build a simplified preview
    preview_events = []
    intensity_curve: list[dict] = []
    sharpness_curve: list[dict] = []

    for entry in ahap_data.get("Pattern", []):
        if "Event" in entry:
            evt = entry["Event"]
            params = {
                p["ParameterID"]: p["ParameterValue"]
                for p in evt.get("EventParameters", [])
            }
            preview_events.append({
                "time": evt.get("Time", 0),
                "type": evt.get("EventType", ""),
                "duration": evt.get("EventDuration", 0),
                "intensity": params.get("HapticIntensity", 0),
                "sharpness": params.get("HapticSharpness", 0),
            })
        elif "ParameterCurve" in entry:
            curve = entry["ParameterCurve"]
            param_id = curve.get("ParameterID", "")
            points = curve.get("ParameterCurveControlPoints", [])
            if param_id == "HapticIntensityControl":
                intensity_curve = points
            elif param_id == "HapticSharpnessControl":
                sharpness_curve = points

    return JSONResponse({
        "job_id": job_id,
        "duration": float(data.get("duration", 0)),
        "total_events": len(preview_events),
        "events": preview_events,
        "intensity_curve_points": len(intensity_curve),
        "intensity_curve": intensity_curve,
        "sharpness_curve_points": len(sharpness_curve),
        "sharpness_curve": sharpness_curve,
    })


# ── GET /health ──────────────────────────────────────────


@router.get("/health", tags=["system"], summary="Health check")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
