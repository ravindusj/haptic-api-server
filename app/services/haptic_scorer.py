"""Haptic Score Fusion – combines DSP features with AI classification.

This is the core intelligence layer that:
  1. Builds a **continuous intensity envelope** that maps every moment
     of audio energy to a proportional vibration level (Sony DVS-style
     sound-to-vibration).
  2. Extracts **transient tap events** at onset spikes and beat
     positions, overlaid on the continuous rumble for impact emphasis.
  3. Applies intelligent speech suppression and silence gating while
     preserving the continuous feel.

The intensity envelope is converted into Apple ParameterCurve control
points by the AHAP generator, achieving frame-accurate (~50 ms)
intensity modulation throughout the entire audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.config import get_settings
from app.models.schemas import (
    AIClassification,
    DSPFeatures,
    HapticEvent,
    HapticTimeline,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Envelope sample rate – one control point every 50 ms.
# 20 fps × 30 s-chunk = 600 control points, well within Apple limits.
ENVELOPE_FPS: float = 20.0


@dataclass
class ScoringWeights:
    """Relative importance of each signal source.

    RMS is dominant (0.45) so the base vibration directly tracks
    audio loudness — this is what Sony DVS does.
    """

    rms: float = 0.45
    onset: float = 0.20
    bass: float = 0.20
    ai: float = 0.15


# ── Public API ───────────────────────────────────────────


def fuse_scores(
    dsp: DSPFeatures,
    ai: AIClassification,
    sensitivity: float = 0.5,
    bass_boost: float = 1.0,
) -> HapticTimeline:
    """
    Combine DSP + AI signals into a continuous haptic envelope
    plus transient accent events.

    Parameters
    ----------
    dsp : DSPFeatures
        Frame-level librosa features (~43 fps).
    ai : AIClassification
        Frame-level PANNs classification.
    sensitivity : float
        0-1: higher = more reactive.
    bass_boost : float
        0.5-2.0: multiplier for low-frequency energy.

    Returns
    -------
    HapticTimeline
        Contains a continuous intensity/sharpness envelope *and*
        transient tap events.
    """
    weights = ScoringWeights()

    # ── Convert DSP arrays to numpy ──────────────────────
    rms = np.array(dsp.rms_energy, dtype=np.float64)
    onset = np.array(dsp.onset_strength, dtype=np.float64)
    bass = np.clip(np.array(dsp.low_freq_energy, dtype=np.float64) * bass_boost, 0.0, 1.0)
    centroid = np.array(dsp.spectral_centroid, dtype=np.float64)
    n_frames = dsp.total_frames
    frame_dur = dsp.hop_length / dsp.sample_rate  # ~0.023 s

    # ── Resample AI scores to DSP frame rate ─────────────
    ai_haptic = _resample_to_length(np.array(ai.haptic_scores), n_frames)
    ai_speech = _resample_to_length(np.array(ai.speech_scores), n_frames)

    # ── Compute combined haptic score per frame ──────────
    combined = (
        weights.rms * rms
        + weights.onset * onset
        + weights.bass * bass
        + weights.ai * ai_haptic
    )

    # ── Apply gentle power curve ─────────────────────────
    # Expand mid-range values upward so quiet-but-audible audio
    # still produces a perceptible vibration.  Using 0.7 instead
    # of 0.6 preserves more dynamic range.
    combined = np.power(np.clip(combined, 0.0, None), 0.7)

    # ── Speech suppression (softer than before: ×0.3) ────
    speech_gate = np.where(ai_speech > 0.4, 0.30, 1.0)
    combined *= speech_gate

    # ── Silence gate with soft fade ──────────────────────
    silence_mask = rms < settings.SILENCE_RMS_THRESHOLD
    # Apply a 3-frame (~70 ms) fade at silence boundaries
    # instead of hard zero to avoid jarring on/off.
    fade_frames = 3
    combined = _apply_silence_fade(combined, silence_mask, fade_frames)

    # ── Light smoothing (50 ms) ──────────────────────────
    smooth_win = max(1, int(0.05 / frame_dur))
    combined = _smooth(combined, smooth_win)

    # ── Build sharpness signal (bass → low, treble → high) ─
    sharpness = centroid.copy()
    # Where bass energy is strong, push sharpness down for a
    # deeper "rumble" feel.
    bass_heavy = bass > 0.3
    sharpness[bass_heavy] = np.clip(sharpness[bass_heavy] * 0.4, 0.05, 0.95)
    sharpness = np.clip(sharpness, 0.05, 0.95)
    sharpness = _smooth(sharpness, smooth_win)

    # ── Boost combined into perceptible range [0.25, 1.0] ─
    # Apple Taptic Engine barely feels values under ~0.3.
    boosted = _boost_array(combined)

    # ── Downsample to envelope rate (~20 fps) ────────────
    intensity_env = _downsample_max(boosted, frame_dur, ENVELOPE_FPS)
    sharpness_env = _downsample_mean(sharpness, frame_dur, ENVELOPE_FPS)

    # ── Extract transient tap events ─────────────────────
    # Threshold only affects which onset spikes get a tap —
    # the continuous envelope is always emitted for non-silent audio.
    threshold = 0.45 - (sensitivity * 0.40)
    threshold = max(0.05, threshold)

    events = _extract_transient_events(
        combined=boosted,
        onset=onset,
        centroid=centroid,
        bass=bass,
        threshold=threshold,
        sr=dsp.sample_rate,
        hop=dsp.hop_length,
        beat_times=dsp.beat_times,
        beat_strengths=dsp.beat_strengths,
    )

    logger.info(
        "Score fusion: %d frames → %d transient events, "
        "%d envelope points (%.0f fps), threshold=%.2f",
        n_frames,
        len(events),
        len(intensity_env),
        ENVELOPE_FPS,
        threshold,
    )

    return HapticTimeline(
        duration_seconds=dsp.duration_seconds,
        events=events,
        intensity_envelope=[round(float(v), 4) for v in intensity_env],
        sharpness_envelope=[round(float(v), 4) for v in sharpness_env],
        envelope_fps=ENVELOPE_FPS,
        metadata={
            "sensitivity": sensitivity,
            "bass_boost": bass_boost,
            "threshold": threshold,
            "total_frames": n_frames,
            "envelope_points": len(intensity_env),
            "speech_suppressed_pct": round(
                float(np.mean(ai_speech > 0.4)) * 100, 1
            ),
        },
    )


# ── Transient event extraction ───────────────────────────


def _extract_transient_events(
    combined: np.ndarray,
    onset: np.ndarray,
    centroid: np.ndarray,
    bass: np.ndarray,
    threshold: float,
    sr: int,
    hop: int,
    beat_times: list[float],
    beat_strengths: list[float],
) -> list[HapticEvent]:
    """Extract transient (tap) events only.

    The continuous vibration is handled entirely by the intensity
    envelope + ParameterCurve.  Transients are **accent taps** that
    sit on top and provide punchy impact for onsets/beats.
    """
    events: list[HapticEvent] = []
    frame_dur = hop / sr
    min_interval = settings.MIN_TRANSIENT_INTERVAL_MS / 1000.0
    last_t = -1.0
    n_frames = len(combined)

    # ── Onset-spike transients ───────────────────────────
    onset_gate = 0.20  # low gate – most audible onsets pass
    for fi in range(n_frames):
        if combined[fi] < threshold * 0.5:
            continue
        if onset[fi] < onset_gate:
            continue
        t = fi * frame_dur
        if (t - last_t) < min_interval:
            continue

        intensity = float(combined[fi])
        sharpness = float(np.clip(centroid[fi], 0.05, 0.95))
        if bass[fi] > 0.4:
            sharpness = max(0.1, sharpness * 0.5)

        events.append(
            HapticEvent(
                time=round(t, 4),
                event_type="transient",
                duration=0.0,
                intensity=round(np.clip(intensity, 0.0, 1.0), 4),
                sharpness=round(sharpness, 4),
            )
        )
        last_t = t

    # ── Beat-aligned transients ──────────────────────────
    for bt, bs in zip(beat_times, beat_strengths):
        if bs < 0.12:
            continue
        too_close = any(
            abs(e.time - bt) < min_interval
            for e in events
        )
        if too_close:
            continue

        intensity = float(np.clip(0.25 + bs * 0.75, 0.0, 1.0))
        events.append(
            HapticEvent(
                time=round(bt, 4),
                event_type="transient",
                duration=0.0,
                intensity=round(intensity, 4),
                sharpness=0.5,
            )
        )

    events.sort(key=lambda e: e.time)
    return events


# ── Envelope helpers ─────────────────────────────────────


def _boost_array(arr: np.ndarray) -> np.ndarray:
    """Remap values into the perceptible range [0.25, 1.0].

    Silent frames (value ≈ 0) stay at 0.  Everything else is
    lifted so the Taptic Engine can actually be felt.
    """
    out = np.where(arr > 0.01, 0.25 + arr * 0.75, 0.0)
    return np.clip(out, 0.0, 1.0)


def _downsample_max(arr: np.ndarray, frame_dur: float, target_fps: float) -> np.ndarray:
    """Downsample using windowed max-pooling to preserve transient peaks."""
    step = max(1, int(round(1.0 / (target_fps * frame_dur))))
    n = len(arr)
    out = []
    for i in range(0, n, step):
        window = arr[i : i + step]
        out.append(float(np.max(window)))
    return np.array(out)


def _downsample_mean(arr: np.ndarray, frame_dur: float, target_fps: float) -> np.ndarray:
    """Downsample using windowed mean for smoother signals."""
    step = max(1, int(round(1.0 / (target_fps * frame_dur))))
    n = len(arr)
    out = []
    for i in range(0, n, step):
        window = arr[i : i + step]
        out.append(float(np.mean(window)))
    return np.array(out)


def _apply_silence_fade(
    combined: np.ndarray,
    silence_mask: np.ndarray,
    fade_frames: int,
) -> np.ndarray:
    """Apply silence gate with a soft fade instead of hard zero."""
    result = combined.copy()
    n = len(result)

    # Find silence-boundary transitions and apply fade
    for i in range(n):
        if silence_mask[i]:
            result[i] = 0.0
        else:
            # Check proximity to silence boundary
            dist_to_silence = fade_frames
            for d in range(1, fade_frames + 1):
                if i - d >= 0 and silence_mask[i - d]:
                    dist_to_silence = min(dist_to_silence, d)
                    break
                if i + d < n and silence_mask[i + d]:
                    dist_to_silence = min(dist_to_silence, d)
                    break
            if dist_to_silence < fade_frames:
                fade = dist_to_silence / fade_frames
                result[i] *= fade

    return result


# ── General utilities ────────────────────────────────────


def _resample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate an array to a new length."""
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving average."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")
