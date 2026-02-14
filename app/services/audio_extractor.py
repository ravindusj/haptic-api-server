"""Extract audio track from uploaded video files using FFmpeg."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def extract_audio(
    video_path: str | Path,
    job_id: str,
) -> dict[str, str]:
    """
    Extract audio from a video file, producing two WAV files:

    1. ``{job_id}_22050.wav`` – 22 050 Hz mono (for librosa DSP analysis)
    2. ``{job_id}_32000.wav`` – 32 000 Hz mono (for PANNs AI inference)

    Parameters
    ----------
    video_path : str | Path
        Absolute path to the uploaded video.
    job_id : str
        Unique job identifier (used for output filenames).

    Returns
    -------
    dict with keys ``"librosa_wav"`` and ``"panns_wav"`` pointing to
    the output file paths, plus ``"duration"`` in seconds.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = Path(settings.UPLOAD_DIR) / job_id
    os.makedirs(work_dir, exist_ok=True)

    librosa_wav = str(work_dir / f"{job_id}_22050.wav")
    panns_wav = str(work_dir / f"{job_id}_32000.wav")

    # ── Get video duration ───────────────────────────────
    duration = _get_duration(str(video_path))
    logger.info("Video duration: %.2f seconds", duration)

    # ── Extract at 22050 Hz (librosa) ────────────────────
    _ffmpeg_extract(str(video_path), librosa_wav, sample_rate=settings.AUDIO_SAMPLE_RATE)

    # ── Extract at 32000 Hz (PANNs) ─────────────────────
    _ffmpeg_extract(str(video_path), panns_wav, sample_rate=settings.PANNS_SAMPLE_RATE)

    return {
        "librosa_wav": librosa_wav,
        "panns_wav": panns_wav,
        "duration": duration,
    }


def _ffmpeg_extract(
    input_path: str,
    output_path: str,
    sample_rate: int,
) -> None:
    """Run FFmpeg to convert video → mono WAV at the given sample rate."""
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite
        "-i", input_path,         # input video
        "-vn",                    # drop video stream
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(sample_rate),  # target sample rate
        "-ac", "1",               # mono
        output_path,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,  # 10 min max
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        logger.error("FFmpeg failed:\n%s", err)
        raise RuntimeError(f"FFmpeg audio extraction failed: {err[:500]}")
    logger.info("Extracted: %s", output_path)


def _get_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed – is FFmpeg installed?")
    return float(result.stdout.decode().strip())


def cleanup_job_files(job_id: str) -> None:
    """Remove temporary audio files for a completed job."""
    import shutil

    work_dir = Path(settings.UPLOAD_DIR) / job_id
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Cleaned up: %s", work_dir)
