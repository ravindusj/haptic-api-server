"""Haptic Score Fusion – combines HPSS-separated DSP with AI classification.

Architecture
------------
The new pipeline uses three complementary signal sources:

  1. **Percussive RMS + onset** (from HPSS) → drives transient tap
     events.  Since HPSS strips out harmonics, onset detection no
     longer false-triggers on speech or sustained instruments.

  2. **Multi-band frequency energies** (6 bands) → drive the continuous
     intensity envelope.  Each band contributes a weighted share:
     sub-bass/bass → deep rumble, mid/presence → medium-sharp texture.

  3. **YAMNet haptic scores + Whisper speech segments** → semantic
     awareness.  YAMNet amplifies genuine impacts (explosions, drums).
     Whisper gives pixel-accurate speech timestamps for suppression.

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
ENVELOPE_FPS: float = 20.0


@dataclass
class ScoringWeights:
    """Relative importance of each signal source.

    Percussive captures transient energy (impacts, drums).
    Bass bands drive the rumble.  AI amplifies semantically
    meaningful sounds identified by YAMNet.
    """

    percussive: float = 0.25   # HPSS percussive RMS
    sub_bass: float = 0.20     # 20-60 Hz deep rumble
    bass: float = 0.20         # 60-250 Hz punch
    low_mid: float = 0.05      # 250-500 Hz body
    mid: float = 0.05          # 500-2000 Hz texture
    presence: float = 0.05     # 2000-4000 Hz detail
    ai: float = 0.20           # YAMNet haptic score


# ── Public API ───────────────────────────────────────────


def fuse_scores(
    dsp: DSPFeatures,
    ai: AIClassification,
    sensitivity: float = 0.5,
    bass_boost: float = 1.0,
) -> HapticTimeline:
    """
    Combine HPSS-separated DSP + YAMNet/Whisper AI signals into a
    continuous haptic envelope plus transient accent events.
    """
    weights = ScoringWeights()
    n_frames = dsp.total_frames
    frame_dur = dsp.hop_length / dsp.sample_rate  # ~0.023 s

    # ── Load DSP arrays ──────────────────────────────────
    perc_rms = np.array(dsp.percussive_rms, dtype=np.float64)
    perc_onset = np.array(dsp.percussive_onset, dtype=np.float64)
    harm_rms = np.array(dsp.harmonic_rms, dtype=np.float64)
    rms = np.array(dsp.rms_energy, dtype=np.float64)
    centroid = np.array(dsp.spectral_centroid, dtype=np.float64)
    raw_rms = np.array(dsp.raw_rms_array, dtype=np.float64) if dsp.raw_rms_array else rms.copy()

    # Multi-band arrays
    sub_bass = np.array(dsp.sub_bass_energy, dtype=np.float64)
    bass = np.clip(np.array(dsp.bass_energy, dtype=np.float64) * bass_boost, 0.0, 1.0)
    low_mid = np.array(dsp.low_mid_energy, dtype=np.float64)
    mid = np.array(dsp.mid_energy, dtype=np.float64)
    presence = np.array(dsp.presence_energy, dtype=np.float64)

    # ── Resample AI scores to DSP frame rate ─────────────
    ai_haptic = _resample_to_length(np.array(ai.haptic_scores), n_frames)

    # ── Resample AI dominant classes to DSP frame rate ───
    # Nearest-neighbor mapping for semantic sharpness & crash bursts.
    _ai_n = len(ai.dominant_classes)
    if _ai_n > 0:
        _ai_idx = np.clip(
            (np.arange(n_frames) * frame_dur / ai.frame_duration_s).astype(int),
            0, _ai_n - 1,
        )
        dsp_dominant: list[str] = [ai.dominant_classes[i] for i in _ai_idx]
    else:
        _ai_idx = np.zeros(n_frames, dtype=int)
        dsp_dominant = ["unknown"] * n_frames

    # ── Build speech mask from Whisper segments ──────────
    # Whisper gives precise [start, end] timestamps for speech.
    # Convert to per-frame gate: 0 = speech, 1 = non-speech.
    speech_gate = _build_whisper_speech_gate(
        speech_segments=ai.speech_segments,
        n_frames=n_frames,
        frame_dur=frame_dur,
    )

    # Merge with YAMNet speech scores as a backup
    ai_speech = _resample_to_length(np.array(ai.speech_scores), n_frames)
    yamnet_gate = 1.0 - np.clip((ai_speech - 0.5) / 0.4, 0.0, 1.0)
    # Take the more suppressive of the two gates
    speech_gate = np.minimum(speech_gate, yamnet_gate)

    # Smooth gate edges (~80 ms crossfade)
    gate_smooth = max(1, int(0.08 / frame_dur))
    speech_gate = _smooth(speech_gate, gate_smooth)
    speech_gate = np.where(speech_gate < 0.05, 0.0, speech_gate)

    # ── Haptic-content override ──────────────────────────
    # If YAMNet detects strong haptic content (crash, music, etc.),
    # do NOT let the speech gate kill it — guarantee at least 70%.
    haptic_override = ai_haptic > 0.12
    speech_gate = np.where(haptic_override, np.maximum(speech_gate, 0.70), speech_gate)

    # ── Percussive novelty gating ────────────────────────
    # Constant percussive energy (machine hum, engine) → suppress.
    # Floor raised to 0.50 so steady rhythms (drums, bass lines)
    # retain at least half their energy instead of being killed.
    novelty_win = max(3, int(0.5 / frame_dur))
    perc_nov = _rolling_std(perc_rms, novelty_win)
    perc_nmax = np.percentile(perc_nov, 98) if len(perc_nov) > 0 else 1.0
    if perc_nmax > 1e-6:
        perc_nov /= perc_nmax
    perc_nov = np.clip(perc_nov, 0.0, 1.0)
    perc_gate = 0.50 + 0.50 * perc_nov
    perc_modulated = perc_rms * perc_gate

    # ── Combine weighted signals ─────────────────────────
    # Include harmonic RMS so sustained music (strings, pads,
    # singing) drives continuous vibration — not just transients.
    harmonic_contribution = 0.15 * harm_rms

    combined = (
        weights.percussive * perc_modulated
        + weights.sub_bass * sub_bass
        + weights.bass * bass
        + weights.low_mid * low_mid
        + weights.mid * mid
        + weights.presence * presence
        + weights.ai * ai_haptic
        + harmonic_contribution
    )

    # ── Impact amplification ─────────────────────────────
    # When percussive + bass are both strong → genuine impact.
    # Boost by up to 2.5× so explosions/hits feel powerful.
    impact_factor = np.where(
        (perc_rms > 0.06) & (bass > 0.06),
        1.0 + 1.5 * perc_rms * bass,
        1.0,
    )
    # Percussive-only boost for crashes that lack strong bass
    perc_only_boost = np.where(perc_rms > 0.20, 1.0 + 0.8 * perc_rms, 1.0)
    combined *= impact_factor * perc_only_boost

    # ── Apply speech gate (once only) ─────────────────────
    combined *= speech_gate

    # ── Silence gate (raw RMS) ───────────────────────────
    raw_rms_trimmed = _pad_or_trim_np(raw_rms, n_frames)
    silence_mask = raw_rms_trimmed < settings.SILENCE_RMS_THRESHOLD
    combined = _apply_silence_fade(combined, silence_mask, fade_frames=3)

    # ── Clamp & comfort ceiling ──────────────────────────
    combined = np.clip(combined, 0.0, 1.0)
    envelope_signal = combined.copy()

    # ── Adaptive rest gate — zero out faint frames ───────
    # Use a low fixed floor plus a fraction of the local median
    # so quiet-but-real content isn't zeroed after normalisation.
    local_median = np.median(envelope_signal[envelope_signal > 0]) if np.any(envelope_signal > 0) else 0.0
    rest_threshold = min(0.02, max(0.010, 0.05 * local_median))
    envelope_signal[envelope_signal < rest_threshold] = 0.0

    # ── Perceptual floor boost ───────────────────────────
    # Remap non-silent frames from [0, 1] to [0.20, 1.0] so
    # any audible content produces at least a perceptible
    # vibration on the Taptic Engine (values < 0.2 are barely felt).
    envelope_signal = np.where(
        envelope_signal > 0.01,
        0.20 + envelope_signal * 0.80,
        0.0,
    )
    envelope_signal = np.clip(envelope_signal, 0.0, 1.0)

    # ── Build sharpness from band balance ────────────────
    # Sub-bass heavy → low sharpness, presence/brilliance → high.
    brilliance = np.array(dsp.brilliance_energy, dtype=np.float64)
    low_energy = sub_bass + bass + 1e-8
    high_energy = presence + brilliance + 1e-8
    band_ratio = high_energy / (low_energy + high_energy)
    sharpness = np.clip(0.1 + 0.8 * band_ratio, 0.05, 0.95)
    sharpness_smooth_win = max(1, int(0.05 / frame_dur))
    sharpness = _smooth(sharpness, sharpness_smooth_win)

    # ── AI-driven sharpness modulation ───────────────────
    # Crash/explosion → high sharpness (metallic crunch),
    # bass/engine → low sharpness (deep rumble),
    # drums → medium-high (punchy).  Blended 50/50 with spectral.
    _CRASH_LABELS = {
        "Explosion", "Smash, crash", "Bang", "Thunder",
        "Thunderstorm", "Thump, thud", "Whack, thwack",
        "Slap, smack", "Artillery fire",
    }
    _GUNSHOT_LABELS = {"Gunshot, gunfire", "Machine gun", "Fusillade"}
    _DRUM_LABELS = {
        "Drum", "Snare drum", "Bass drum", "Rimshot",
        "Drum roll", "Cymbal", "Hi-hat", "Drum kit",
    }
    _DEEP_LABELS = {
        "Bass guitar", "Double bass", "Engine",
        "Motor vehicle (road)", "Truck",
    }
    _TICK_LABELS = {"Clock", "Tick", "Tick-tock", "Alarm clock"}
    if _ai_n > 0:
        _sem_sharp = np.full(_ai_n, 0.5)
        for _si, _lbl in enumerate(ai.dominant_classes):
            if _lbl in _CRASH_LABELS:
                _sem_sharp[_si] = 0.85
            elif _lbl in _GUNSHOT_LABELS:
                _sem_sharp[_si] = 0.95
            elif _lbl in _TICK_LABELS:
                _sem_sharp[_si] = 0.90
            elif _lbl in _DRUM_LABELS:
                _sem_sharp[_si] = 0.60
            elif _lbl in _DEEP_LABELS:
                _sem_sharp[_si] = 0.15
        semantic_sharpness = _sem_sharp[_ai_idx]
        sharpness = 0.50 * semantic_sharpness + 0.50 * sharpness
        sharpness = np.clip(sharpness, 0.05, 0.95)

    # ── Downsample to envelope rate (~20 fps) ────────────
    intensity_env, actual_env_fps = _downsample_max(envelope_signal, frame_dur, ENVELOPE_FPS)
    sharpness_env, _ = _downsample_mean(sharpness, frame_dur, ENVELOPE_FPS)

    # ── Extract transient tap events ─────────────────────
    # Boost percussive signal for tap detection
    boosted_for_taps = _boost_array(combined) * speech_gate

    threshold = 0.45 - (sensitivity * 0.40)
    threshold = max(0.05, threshold)

    events = _extract_transient_events(
        combined=boosted_for_taps,
        onset=perc_onset,
        centroid=centroid,
        bass=bass,
        sub_bass=sub_bass,
        threshold=threshold,
        sr=dsp.sample_rate,
        hop=dsp.hop_length,
        beat_times=dsp.beat_times,
        beat_strengths=dsp.beat_strengths,
        speech_gate=speech_gate,
        ai_haptic=ai_haptic,
        dsp_dominant=dsp_dominant,
    )

    # Compute speech suppression percentage
    whisper_pct = 0.0
    if ai.speech_segments:
        total_speech = sum(s.end - s.start for s in ai.speech_segments)
        whisper_pct = round(total_speech / max(dsp.duration_seconds, 0.01) * 100, 1)

    logger.info(
        "Score fusion: %d frames → %d transient events, "
        "%d envelope points (%.0f fps), threshold=%.2f, "
        "speech=%.1f%%",
        n_frames,
        len(events),
        len(intensity_env),
        ENVELOPE_FPS,
        threshold,
        whisper_pct,
    )

    return HapticTimeline(
        duration_seconds=dsp.duration_seconds,
        events=events,
        intensity_envelope=[round(float(v), 4) for v in intensity_env],
        sharpness_envelope=[round(float(v), 4) for v in sharpness_env],
        envelope_fps=actual_env_fps,
        metadata={
            "sensitivity": sensitivity,
            "bass_boost": bass_boost,
            "threshold": threshold,
            "total_frames": n_frames,
            "envelope_points": len(intensity_env),
            "speech_suppressed_pct": whisper_pct,
            "whisper_segments": len(ai.speech_segments),
        },
    )


# ── Speech gate from Whisper timestamps ──────────────────


def _build_whisper_speech_gate(
    speech_segments: list,
    n_frames: int,
    frame_dur: float,
) -> np.ndarray:
    """Convert Whisper speech segments to a per-frame gate.

    Returns an array where 1.0 = non-speech (pass through) and
    0.0 = speech (suppress).  Guard frames (~100 ms) provide
    smooth edges around speech boundaries.
    """
    gate = np.ones(n_frames, dtype=np.float64)

    if not speech_segments:
        return gate

    guard_dur = 0.10  # 100 ms guard on each side
    for seg in speech_segments:
        start_s = max(0.0, seg.start - guard_dur)
        end_s = seg.end + guard_dur
        start_f = int(start_s / frame_dur)
        end_f = min(n_frames, int(end_s / frame_dur) + 1)

        # Proportional suppression based on confidence
        suppression = float(seg.confidence)
        gate[start_f:end_f] = np.minimum(
            gate[start_f:end_f],
            1.0 - suppression,
        )

    return gate


# ── Transient event extraction ───────────────────────────


def _extract_transient_events(
    combined: np.ndarray,
    onset: np.ndarray,
    centroid: np.ndarray,
    bass: np.ndarray,
    sub_bass: np.ndarray,
    threshold: float,
    sr: int,
    hop: int,
    beat_times: list[float],
    beat_strengths: list[float],
    speech_gate: np.ndarray | None = None,
    ai_haptic: np.ndarray | None = None,
    dsp_dominant: list[str] | None = None,
) -> list[HapticEvent]:
    """Extract transient (tap) events from percussive signal.

    Transients are accent taps overlaid on the continuous ParameterCurve
    envelope for punchy impact emphasis at onsets and beats.
    Crash/explosion frames get burst transients for a "shattering" feel.
    """
    events: list[HapticEvent] = []
    frame_dur = hop / sr
    min_interval = settings.MIN_TRANSIENT_INTERVAL_MS / 1000.0
    last_t = -1.0
    n_frames = len(combined)

    # ── Onset-spike transients ───────────────────────────
    onset_gate = 0.20
    for fi in range(n_frames):
        if combined[fi] < threshold * 0.5:
            continue
        if onset[fi] < onset_gate:
            continue
        if speech_gate is not None and fi < len(speech_gate) and speech_gate[fi] < 0.1:
            continue
        t = fi * frame_dur
        if (t - last_t) < min_interval:
            continue

        intensity = float(combined[fi])
        # Sharpness from band balance at this frame
        lo = float(sub_bass[fi] + bass[fi]) + 1e-8
        hi = float(centroid[fi])
        sharpness = float(np.clip(0.1 + 0.8 * hi, 0.05, 0.95))
        if lo > 0.4:
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
        if speech_gate is not None:
            beat_frame = int(bt / frame_dur)
            if beat_frame < len(speech_gate) and speech_gate[beat_frame] < 0.1:
                continue
        too_close = any(abs(e.time - bt) < min_interval for e in events)
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

    # ── Crash/impact burst transients ────────────────────
    # When AI detects a crash/explosion with strong confidence,
    # inject a burst of 3 rapid transients with varied sharpness
    # (sharp→medium→deep) for a visceral "shattering" feel.
    _BURST_CLASSES = {
        "Explosion", "Smash, crash", "Bang", "Thunder",
        "Thunderstorm", "Gunshot, gunfire", "Machine gun",
        "Fusillade", "Artillery fire", "Thump, thud",
        "Whack, thwack", "Slap, smack",
    }
    if ai_haptic is not None and dsp_dominant is not None:
        burst_spacing = 0.030     # 30 ms between burst taps
        burst_cooldown = 0.50     # max one burst per 500 ms
        last_burst_t = -1.0
        burst_sharpness = [0.95, 0.60, 0.25]  # sharp→medium→deep
        for fi in range(n_frames):
            if fi >= len(ai_haptic) or fi >= len(dsp_dominant):
                break
            if ai_haptic[fi] < 0.15:
                continue
            if dsp_dominant[fi] not in _BURST_CLASSES:
                continue
            t = fi * frame_dur
            if (t - last_burst_t) < burst_cooldown:
                continue
            for b in range(3):
                bt = t + b * burst_spacing
                overlap = any(abs(e.time - bt) < 0.015 for e in events)
                if overlap:
                    continue
                bi = float(np.clip(0.85 + 0.15 * ai_haptic[fi], 0.85, 1.0))
                events.append(HapticEvent(
                    time=round(bt, 4),
                    event_type="transient",
                    duration=0.0,
                    intensity=round(bi, 4),
                    sharpness=burst_sharpness[b],
                ))
            last_burst_t = t

    events.sort(key=lambda e: e.time)
    return events


# ── Envelope helpers ─────────────────────────────────────


def _boost_array(arr: np.ndarray) -> np.ndarray:
    """Remap values into the perceptible range [0.25, 1.0].

    Silent frames (value ≈ 0) stay at 0.
    """
    out = np.where(arr > 0.01, 0.25 + arr * 0.75, 0.0)
    return np.clip(out, 0.0, 1.0)


def _downsample_max(
    arr: np.ndarray, frame_dur: float, target_fps: float,
) -> tuple[np.ndarray, float]:
    """Downsample using windowed max-pooling to preserve transient peaks.

    Returns (downsampled_array, actual_fps) so downstream consumers
    (AHAP generator) use the *real* effective FPS instead of the
    target, eliminating progressive time-drift in later chunks.
    """
    step = max(1, int(round(1.0 / (target_fps * frame_dur))))
    actual_fps = 1.0 / (step * frame_dur)
    n = len(arr)
    out = []
    for i in range(0, n, step):
        window = arr[i : i + step]
        out.append(float(np.max(window)))
    return np.array(out), round(actual_fps, 4)


def _downsample_mean(
    arr: np.ndarray, frame_dur: float, target_fps: float,
) -> tuple[np.ndarray, float]:
    """Downsample using windowed mean for smoother signals."""
    step = max(1, int(round(1.0 / (target_fps * frame_dur))))
    actual_fps = 1.0 / (step * frame_dur)
    n = len(arr)
    out = []
    for i in range(0, n, step):
        window = arr[i : i + step]
        out.append(float(np.mean(window)))
    return np.array(out), round(actual_fps, 4)


def _apply_silence_fade(
    combined: np.ndarray,
    silence_mask: np.ndarray,
    fade_frames: int,
) -> np.ndarray:
    """Apply silence gate with a soft fade instead of hard zero."""
    result = combined.copy()
    n = len(result)
    for i in range(n):
        if silence_mask[i]:
            result[i] = 0.0
        else:
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


def _pad_or_trim_np(arr: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or trim to exact length."""
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.zeros(length - len(arr))])


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
