"""DSP-based audio feature extraction using librosa.

Extracts frame-level features that map to haptic characteristics:
- RMS energy → overall loudness / intensity
- Onset strength → percussive transients (impacts, hits)
- Low-frequency energy (20-200 Hz) → bass rumble
- Spectral centroid → brightness → haptic sharpness
- Spectral flux → timbral change rate
- Beat positions → rhythmic pulse points
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from app.core.config import get_settings
from app.models.schemas import DSPFeatures

logger = logging.getLogger(__name__)
settings = get_settings()


def analyze_dsp(wav_path: str) -> DSPFeatures:
    """
    Run full DSP feature extraction on a WAV file.

    Parameters
    ----------
    wav_path : str
        Path to mono WAV at 22050 Hz.

    Returns
    -------
    DSPFeatures
        Frame-level arrays + beat positions.
    """
    sr = settings.AUDIO_SAMPLE_RATE
    hop = settings.HOP_LENGTH

    logger.info("Loading audio: %s", wav_path)
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info("Audio loaded: %.2fs, %d samples", duration, len(y))

    # ── RMS Energy ───────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_norm = _normalise(rms)

    # ── Onset Strength ───────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_norm = _normalise(onset_env)

    # ── Low-Frequency Energy (20-200 Hz band) ───────────
    low_freq = _bandpass_energy(y, sr, low=20, high=200, hop_length=hop)
    low_freq_norm = _normalise(low_freq)

    # ── Spectral Centroid → Sharpness mapping ───────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    # Normalise relative to Nyquist, but use a practical max of ~8 kHz
    # (most haptic-relevant energy is below that)
    practical_max = min(8000.0, sr / 2.0)
    centroid_norm = np.clip(centroid / practical_max, 0.0, 1.0)

    # ── Spectral Flux ───────────────────────────────────
    spec = np.abs(librosa.stft(y, hop_length=hop))
    flux = np.sqrt(np.mean(np.diff(spec, axis=1) ** 2, axis=0))
    # Pad to match frame count
    flux = np.concatenate([[0.0], flux])
    flux_norm = _normalise(flux)

    # ── Beat Tracking ────────────────────────────────────
    beat_times, beat_strengths = _detect_beats(y, sr, hop)

    # Ensure all arrays have the same length
    n_frames = len(rms_norm)
    onset_norm = _pad_or_trim(onset_norm, n_frames)
    low_freq_norm = _pad_or_trim(low_freq_norm, n_frames)
    centroid_norm = _pad_or_trim(centroid_norm, n_frames)
    flux_norm = _pad_or_trim(flux_norm, n_frames)

    logger.info(
        "DSP analysis complete: %d frames, %d beats detected",
        n_frames,
        len(beat_times),
    )

    return DSPFeatures(
        sample_rate=sr,
        hop_length=hop,
        total_frames=n_frames,
        duration_seconds=round(duration, 4),
        rms_energy=rms_norm.tolist(),
        onset_strength=onset_norm.tolist(),
        low_freq_energy=low_freq_norm.tolist(),
        spectral_centroid=centroid_norm.tolist(),
        spectral_flux=flux_norm.tolist(),
        beat_times=[round(float(t), 4) for t in beat_times],
        beat_strengths=[round(float(s), 4) for s in beat_strengths],
    )


# ── Helpers ──────────────────────────────────────────────


def _normalise(arr: np.ndarray, percentile: float = 98) -> np.ndarray:
    """Percentile-based normalisation to [0, 1].

    Uses the given percentile as the effective max rather than the
    absolute max, which prevents a single extreme outlier from
    squashing the entire signal to near-zero.
    """
    mn = float(np.min(arr))
    mx = float(np.percentile(arr, percentile))
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    normalised = (arr - mn) / (mx - mn)
    return np.clip(normalised, 0.0, 1.0)


def _pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or trim to exact length."""
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.zeros(length - len(arr))])


def _bandpass_energy(
    y: np.ndarray,
    sr: int,
    low: float,
    high: float,
    hop_length: int,
    order: int = 4,
) -> np.ndarray:
    """Compute RMS energy of a band-pass filtered signal."""
    nyq = sr / 2.0
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    y_filtered = sosfilt(sos, y)
    rms = librosa.feature.rms(y=y_filtered, hop_length=hop_length)[0]
    return rms


def _detect_beats(
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect beat positions and their strengths.

    Uses librosa's beat tracker. For higher accuracy, madmom can
    be swapped in here (RNNBeatProcessor + DBNBeatTrackingProcessor).
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Compute strength at each beat position from onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    beat_strengths = np.array([
        onset_env[min(f, len(onset_env) - 1)] for f in beat_frames
    ])
    if len(beat_strengths) > 0:
        beat_strengths = _normalise(beat_strengths)

    return beat_times, beat_strengths
