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

    Onset is the dominant action signal (0.30) — it
    represents percussive impacts, hits, and musical beats.
    Bass rumble and RMS track overall energy.
    """

    rms: float = 0.30
    onset: float = 0.30
    bass: float = 0.25
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

    # ── DSP-based speech detection ───────────────────────
    # PANNs may be unavailable (fallback mode → speech=0.0).
    # Use spectral features to detect dialogue:
    #   • speech centroid ≈ 0.04-0.40 (300-3200 Hz / 8 kHz)
    #   • low bass (< 0.25)  — explosions have high bass
    #   • low onset (< 0.25) — impacts are percussive
    #   • moderate RMS (> 0.05) — not silence
    # When all true → likely dialogue → suppress.
    dsp_speech = _detect_speech_dsp(rms, onset, bass, centroid)

    # Merge: take the stronger of AI and DSP speech scores
    speech_score = np.maximum(ai_speech, dsp_speech)

    # ── Compute combined haptic score per frame ──────────
    # Novelty modulation: RMS and bass are suppressed when
    # constant (steady drone, background hum → fatiguing).
    # Onset passes through unmodified — it inherently measures
    # *change* in the audio so it's already a novelty signal.
    novelty_window = max(3, int(0.5 / frame_dur))   # ~22 frames

    # RMS novelty
    rms_nov = _rolling_std(rms, novelty_window)
    rms_nmax = np.percentile(rms_nov, 98) if len(rms_nov) > 0 else 1.0
    if rms_nmax > 1e-6:
        rms_nov = rms_nov / rms_nmax
    rms_nov = np.clip(rms_nov, 0.0, 1.0)
    rms_gate = 0.15 + 0.85 * rms_nov   # constant → 0.15, dynamic → 1.0
    rms_modulated = rms * rms_gate

    combined = (
        weights.rms * rms_modulated
        + weights.onset * onset       # onset is inherently transient
        + weights.bass * bass         # bass rumble passes through
        + weights.ai * ai_haptic
    )

    # ── Impact amplification ─────────────────────────────
    # When bass AND onset are BOTH strong, this is a genuine
    # impact (explosion, drum hit, crash).  Amplify so these
    # moments feel physically powerful.  Requires high onset
    # to avoid boosting constant bass scenes.
    impact_factor = np.where(
        (bass > 0.30) & (onset > 0.30),
        1.0 + 0.7 * bass * onset,   # up to ~1.7× for heavy hits
        1.0,
    )
    combined = combined * impact_factor

    # ── Speech suppression (hard: ×0.0 for dialogue) ─────
    # During speech, completely zero out the vibration.
    # Guard frames extend the zone by ~150 ms on each side
    # so boundary taps don't fire at dialogue edges.
    guard_frames = max(1, int(0.15 / frame_dur))  # ~7 frames
    speech_binary = (speech_score > 0.5).astype(np.float64)
    # Dilate: extend speech regions by guard_frames in each direction
    dilated = speech_binary.copy()
    for offset in range(1, guard_frames + 1):
        dilated[offset:] = np.maximum(dilated[offset:], speech_binary[:-offset])
        dilated[:-offset] = np.maximum(dilated[:-offset], speech_binary[offset:])
    speech_gate = np.where(dilated > 0.5, 0.0, 1.0)
    # Smooth the gate over ~80 ms for a natural crossfade
    gate_smooth = max(1, int(0.08 / frame_dur))
    speech_gate = _smooth(speech_gate, gate_smooth)
    # Hard floor: ensure near-zero values are fully zeroed
    speech_gate = np.where(speech_gate < 0.15, 0.0, speech_gate)
    combined *= speech_gate

    # ── Silence gate with soft fade ──────────────────────
    silence_mask = rms < settings.SILENCE_RMS_THRESHOLD
    fade_frames = 3
    combined = _apply_silence_fade(combined, silence_mask, fade_frames)

    # ── Re-apply speech & silence gates ──────────────────
    # The silence fade affects boundary frames.  Re-apply
    # speech gate to keep dialogue zones perfectly clean.
    combined *= speech_gate
    combined = _apply_silence_fade(combined, silence_mask, fade_frames)

    # ── Clamp to [0, 1] ─────────────────────────────────
    combined = np.clip(combined, 0.0, 1.0)

    # ── Direct mapping (no power curve) ───────────────────
    # The novelty modulation and speech gating already provide
    # dynamic range.  Use the combined signal directly so
    # action scenes retain full punch.
    envelope_signal = combined.copy()

    # ── Comfort ceiling ──────────────────────────────────
    # Cap at 0.85 so the loudest moments feel strong but not
    # jarring.  Transient taps can still hit 1.0.
    envelope_signal = np.clip(envelope_signal, 0.0, 0.85)

    # ── Rest gate ────────────────────────────────────────
    # Zero out frames below a minimum energy so the user gets
    # intentional rest periods.  After x^1.2, raw 0.10 → 0.06.
    rest_threshold = 0.05
    envelope_signal[envelope_signal < rest_threshold] = 0.0

    # ── Build sharpness signal (bass → low, treble → high) ─
    sharpness = centroid.copy()
    bass_heavy = bass > 0.3
    sharpness[bass_heavy] = np.clip(sharpness[bass_heavy] * 0.4, 0.05, 0.95)
    sharpness = np.clip(sharpness, 0.05, 0.95)
    sharpness_smooth_win = max(1, int(0.05 / frame_dur))
    sharpness = _smooth(sharpness, sharpness_smooth_win)

    # ── Downsample to envelope rate (~20 fps) ────────────
    intensity_env = _downsample_max(envelope_signal, frame_dur, ENVELOPE_FPS)
    sharpness_env = _downsample_mean(sharpness, frame_dur, ENVELOPE_FPS)

    # ── Boost only for transient event detection ─────────
    # Transient taps need to punch through, so we boost them.
    boosted_for_taps = _boost_array(combined)

    # ── Suppress transients during speech too ────────────
    # Apply the same speech gate so no taps fire during dialogue.
    boosted_for_taps *= speech_gate

    # ── Extract transient tap events ─────────────────────
    threshold = 0.45 - (sensitivity * 0.40)
    threshold = max(0.05, threshold)

    events = _extract_transient_events(
        combined=boosted_for_taps,
        onset=onset,
        centroid=centroid,
        bass=bass,
        threshold=threshold,
        sr=dsp.sample_rate,
        hop=dsp.hop_length,
        beat_times=dsp.beat_times,
        beat_strengths=dsp.beat_strengths,
        speech_gate=speech_gate,
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
                float(np.mean(speech_score > 0.5)) * 100, 1
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
    speech_gate: np.ndarray | None = None,
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
        # Skip if in a speech-suppressed region
        if speech_gate is not None and fi < len(speech_gate) and speech_gate[fi] < 0.1:
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
        # Skip beats that land during speech/dialogue
        if speech_gate is not None:
            beat_frame = int(bt / frame_dur)
            if beat_frame < len(speech_gate) and speech_gate[beat_frame] < 0.1:
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


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation (measures local variation)."""
    if window <= 1 or len(arr) < 2:
        return np.ones_like(arr)
    n = len(arr)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.std(arr[lo:hi])
    return out


def _detect_speech_dsp(
    rms: np.ndarray,
    onset: np.ndarray,
    bass: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Heuristic speech detection from DSP features alone.

    Speech has a recognisable spectral fingerprint compared to
    action/effects/music:
      • Centroid in the vocal range (~300–3200 Hz → 0.04–0.40
        normalised against 8 kHz practical max)
      • Low bass energy (voice lacks sub-bass unlike explosions)
      • Low onset strength (speech is not percussive)
      • Moderate-to-high RMS (not silence)

    Returns a per-frame speech probability in [0, 1].
    """
    n = len(rms)
    speech = np.zeros(n, dtype=np.float64)

    # Conditions (all must be loosely true)
    in_vocal_range = (centroid >= 0.03) & (centroid <= 0.45)
    low_bass = bass < 0.25
    low_onset = onset < 0.25
    has_energy = rms > 0.05

    # Score: fraction of conditions met → soft probability
    cond_sum = (
        in_vocal_range.astype(np.float64)
        + low_bass.astype(np.float64)
        + low_onset.astype(np.float64)
        + has_energy.astype(np.float64)
    )
    # All 4 met → 1.0;  3 met → 0.6;  ≤2 → 0
    speech = np.where(cond_sum >= 4, 1.0, np.where(cond_sum >= 3, 0.6, 0.0))

    # Temporal smoothing (~300 ms) to avoid flickering
    if n > 1:
        kernel_size = max(1, int(0.3 * 43))  # ~13 frames at 43fps
        speech = _smooth(speech, kernel_size)
        speech = np.clip(speech, 0.0, 1.0)

    return speech
