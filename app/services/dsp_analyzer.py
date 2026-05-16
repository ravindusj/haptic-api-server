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
    """Run full DSP feature extraction: HPSS, multi-band energies, beat tracking."""
    sr = settings.AUDIO_SAMPLE_RATE
    hop = settings.HOP_LENGTH

    logger.info("Loading audio: %s", wav_path)
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info("Audio loaded: %.2fs, %d samples", duration, len(y))

    D = librosa.stft(y, hop_length=hop)
    H, P = librosa.decompose.hpss(D, margin=3.0)
    y_harmonic = librosa.istft(H, hop_length=hop, length=len(y))
    y_percussive = librosa.istft(P, hop_length=hop, length=len(y))

    harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=hop)[0]
    percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=hop)[0]

    # Onset from percussive only — avoids false triggers on speech or sustained notes.
    percussive_onset = librosa.onset.onset_strength(
        y=y_percussive, sr=sr, hop_length=hop,
    )

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    raw_rms_mean = float(np.mean(rms))
    raw_rms_peak = float(np.percentile(rms, 98)) if len(rms) > 0 else 0.0

    bands = settings.FREQ_BANDS
    band_energies: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        band_energies[name] = _bandpass_energy(y, sr, lo, hi, hop)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    practical_max = min(8000.0, sr / 2.0)
    centroid_norm = np.clip(centroid / practical_max, 0.0, 1.0)

    spec = np.abs(D)
    flux = np.sqrt(np.mean(np.diff(spec, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])

    beat_times, beat_strengths = _detect_beats(y_percussive, sr, hop)

    n_frames = len(rms)
    frame_dur = hop / sr
    rms_norm = _normalise(rms, frame_dur=frame_dur)
    harmonic_norm = _normalise(harmonic_rms, frame_dur=frame_dur)
    percussive_norm = _normalise(percussive_rms, frame_dur=frame_dur)
    onset_norm = _normalise(percussive_onset, frame_dur=frame_dur)
    centroid_norm = _pad_or_trim(centroid_norm, n_frames)
    flux_norm = _normalise(_pad_or_trim(flux, n_frames), frame_dur=frame_dur)

    band_norms: dict[str, np.ndarray] = {}
    for name, raw in band_energies.items():
        band_norms[name] = _normalise(_pad_or_trim(raw, n_frames), frame_dur=frame_dur)

    harmonic_norm = _pad_or_trim(harmonic_norm, n_frames)
    percussive_norm = _pad_or_trim(percussive_norm, n_frames)
    onset_norm = _pad_or_trim(onset_norm, n_frames)

    logger.info(
        "DSP analysis complete: %d frames, %d beats, HPSS + 6-band",
        n_frames,
        len(beat_times),
    )

    return DSPFeatures(
        sample_rate=sr,
        hop_length=hop,
        total_frames=n_frames,
        duration_seconds=round(duration, 4),
        harmonic_rms=harmonic_norm.tolist(),
        percussive_rms=percussive_norm.tolist(),
        percussive_onset=onset_norm.tolist(),
        rms_energy=rms_norm.tolist(),
        spectral_centroid=centroid_norm.tolist(),
        spectral_flux=flux_norm.tolist(),
        sub_bass_energy=band_norms["sub_bass"].tolist(),
        bass_energy=band_norms["bass"].tolist(),
        low_mid_energy=band_norms["low_mid"].tolist(),
        mid_energy=band_norms["mid"].tolist(),
        presence_energy=band_norms["presence"].tolist(),
        brilliance_energy=band_norms["brilliance"].tolist(),
        raw_rms_mean=round(raw_rms_mean, 6),
        raw_rms_peak=round(raw_rms_peak, 6),
        raw_rms_array=rms.tolist(),
        beat_times=[round(float(t), 4) for t in beat_times],
        beat_strengths=[round(float(s), 4) for s in beat_strengths],
    )


def _normalise(
    arr: np.ndarray,
    percentile: float = 98,
    window_sec: float = 30.0,
    frame_dur: float | None = None,
) -> np.ndarray:
    """Sliding-window percentile normalisation to [0, 1] preserving local dynamics."""
    n = len(arr)
    if n == 0:
        return arr.copy()

    if frame_dur is not None and frame_dur > 0:
        win_frames = max(1, int(window_sec / frame_dur))
    else:
        win_frames = max(1, int(window_sec * 43))

    if n <= win_frames:
        mn = float(np.min(arr))
        mx = float(np.percentile(arr, percentile))
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

    step = max(1, win_frames // 2)
    out = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)

    for start in range(0, n, step):
        end = min(start + win_frames, n)
        segment = arr[start:end]
        mn = float(np.min(segment))
        mx = float(np.percentile(segment, percentile))
        if mx - mn < 1e-8:
            normed = np.zeros_like(segment)
        else:
            normed = np.clip((segment - mn) / (mx - mn), 0.0, 1.0)

        w = np.hanning(len(segment) + 2)[1:-1]
        out[start:end] += normed * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-12)
    return np.clip(out / weight, 0.0, 1.0)


def _pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
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
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    beat_strengths = np.array([
        onset_env[min(f, len(onset_env) - 1)] for f in beat_frames
    ])
    if len(beat_strengths) > 0:
        beat_strengths = _normalise(beat_strengths)

    return beat_times, beat_strengths
