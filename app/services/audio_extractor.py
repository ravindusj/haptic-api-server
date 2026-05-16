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
    """Extract audio from video, producing 22050 Hz (librosa) and 16000 Hz (YAMNet/Whisper) WAVs."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = Path(settings.UPLOAD_DIR) / job_id
    os.makedirs(work_dir, exist_ok=True)

    librosa_wav = str(work_dir / f"{job_id}_22050.wav")
    classifier_wav = str(work_dir / f"{job_id}_16000.wav")

    duration = _get_duration(str(video_path))
    logger.info("Video duration: %.2f seconds", duration)

    _ffmpeg_extract(str(video_path), librosa_wav, sample_rate=settings.AUDIO_SAMPLE_RATE)
    _ffmpeg_extract(str(video_path), classifier_wav, sample_rate=settings.CLASSIFIER_SAMPLE_RATE)

    return {
        "librosa_wav": librosa_wav,
        "classifier_wav": classifier_wav,
        "duration": duration,
    }


def _ffmpeg_extract(
    input_path: str,
    output_path: str,
    sample_rate: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        logger.error("FFmpeg failed:\n%s", err)
        raise RuntimeError(f"FFmpeg audio extraction failed: {err[:500]}")
    logger.info("Extracted: %s", output_path)


def _get_duration(video_path: str) -> float:
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
    import shutil

    work_dir = Path(settings.UPLOAD_DIR) / job_id
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Cleaned up: %s", work_dir)
