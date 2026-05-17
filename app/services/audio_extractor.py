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
) -> dict[str, str | None | float]:
    """Extract audio from video.

    Produces up to three WAV files:
      1. ``{job_id}_22050.wav``     – 22 050 Hz mono (librosa DSP)
      2. ``{job_id}_16000.wav``     – 16 000 Hz mono (YAMNet + Whisper)
      3. ``{job_id}_lfe_22050.wav`` – 22 050 Hz mono LFE only (A1)
         when the source has a 5.1+ layout with an LFE channel.

    The LFE (sub-woofer) channel is the haptic ground truth for cinema
    mixes — explosions, sub-bass drops and earthquake rumbles are
    placed there explicitly.  Extracting it separately lets the DSP
    analyser drive sub-bass haptics from the producer's intent rather
    than inferring it from a mono downmix.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir = Path(settings.UPLOAD_DIR) / job_id
    os.makedirs(work_dir, exist_ok=True)

    librosa_wav = str(work_dir / f"{job_id}_22050.wav")
    classifier_wav = str(work_dir / f"{job_id}_16000.wav")
    lfe_wav: str | None = str(work_dir / f"{job_id}_lfe_22050.wav")

    duration = _get_duration(str(video_path))
    logger.info("Video duration: %.2f seconds", duration)

    _ffmpeg_extract(str(video_path), librosa_wav, sample_rate=settings.AUDIO_SAMPLE_RATE)
    _ffmpeg_extract(str(video_path), classifier_wav, sample_rate=settings.CLASSIFIER_SAMPLE_RATE)

    # ── A1: optional LFE extraction (5.1+ sources only) ──
    if _probe_has_lfe(str(video_path)):
        try:
            _ffmpeg_extract_lfe(
                str(video_path), lfe_wav,
                sample_rate=settings.AUDIO_SAMPLE_RATE,
            )
            logger.info("LFE channel extracted: %s", lfe_wav)
        except Exception as e:
            logger.warning("LFE extraction failed (will skip LFE haptics): %s", e)
            lfe_wav = None
    else:
        logger.info("No LFE channel detected — skipping LFE extraction")
        lfe_wav = None

    return {
        "librosa_wav": librosa_wav,
        "classifier_wav": classifier_wav,
        "lfe_wav": lfe_wav,
        "duration": duration,
    }


def _probe_has_lfe(video_path: str) -> bool:
    """Return True when the first audio stream carries an LFE channel.

    Reads ``channel_layout`` (e.g. ``5.1``, ``7.1``, ``5.1(side)``)
    via ffprobe.  Treats any layout containing ``lfe`` or a ``.1``
    suffix as carrying an LFE channel.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channel_layout,channels",
        "-of", "default=noprint_wrappers=1:nokey=0",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    text = result.stdout.decode(errors="replace").lower()
    if "lfe" in text:
        return True
    return any(tok in text for tok in ("5.1", "6.1", "7.1"))


def _ffmpeg_extract_lfe(
    input_path: str,
    output_path: str,
    sample_rate: int,
) -> None:
    """Extract the LFE channel as a mono WAV.

    Uses FFmpeg's ``pan`` filter so the routing is explicit and works
    regardless of the source's exact channel order.
    ``pan=mono|c0=LFE`` picks the channel named LFE in the source
    layout and routes it to a single mono output channel.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-af", "pan=mono|c0=LFE",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    logger.info("Running LFE extract: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg LFE extraction failed: {err[:500]}")


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
