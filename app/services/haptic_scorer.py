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
    SceneChange,
    VideoFeatures,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Envelope sample rate – one control point every 50 ms.
ENVELOPE_FPS: float = 20.0

# Acoustic drum labels — suppressed because drums self-punch through audio.
_DRUM_LABELS_SET: set[str] = {
    "Drum", "Snare drum", "Bass drum", "Rimshot",
    "Drum roll", "Cymbal", "Hi-hat", "Drum kit",
}

# Impact labels — these always override speech suppression during dialogue.
_IMPACT_LABELS_SET: set[str] = {
    "Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade",
    "Artillery fire", "Smash, crash", "Thump, thud", "Bang",
    "Whack, thwack", "Slap, smack", "Thunder", "Thunderstorm",
}

# Per-class impact authoring template. Each template is a dict with:
#   "taps":   list[tuple[offset_ms, intensity_scale, sharpness]]
#   "pre":    optional tuple(offset_ms, intensity_scale, sharpness) — emitted BEFORE t
#   "tail":   optional tuple(duration_s, peak_intensity, decay_tau_s, sharpness)
# offset_ms is relative to the impact frame time t. intensity_scale is
# multiplied by the per-frame base intensity. sharpness is absolute.
_IMPACT_TEMPLATES: dict[str, dict] = {
    "Explosion": {
        "taps":  [(0, 1.00, 0.95), (30, 0.75, 0.80),
                  (65, 0.50, 0.55), (110, 0.30, 0.30), (160, 0.15, 0.15)],
        "pre":   (180, 0.30, 0.40),
        "tail":  (1.20, 0.55, 0.45, 0.15),
    },
    "Thunder": {
        "taps":  [(0, 1.00, 0.70), (45, 0.60, 0.45), (105, 0.35, 0.20)],
        "tail":  (1.80, 0.50, 0.65, 0.10),
    },
    "Thunderstorm": {
        "taps":  [(0, 1.00, 0.70), (45, 0.60, 0.45), (105, 0.35, 0.20)],
        "tail":  (1.80, 0.50, 0.65, 0.10),
    },
    "Gunshot, gunfire": {
        "taps":  [(0, 1.00, 1.00), (15, 0.55, 0.85)],
    },
    "Machine gun":   {"taps": [(0, 1.00, 0.95)]},
    "Fusillade":     {"taps": [(0, 1.00, 0.95)]},
    "Artillery fire": {
        "taps":  [(0, 1.00, 0.85), (35, 0.70, 0.65), (80, 0.45, 0.40)],
        "tail":  (1.00, 0.55, 0.40, 0.15),
    },
    "Smash, crash": {
        "taps":  [(0, 1.00, 0.95), (28, 0.75, 0.90), (60, 0.55, 0.85),
                  (100, 0.40, 0.75), (150, 0.25, 0.55), (210, 0.15, 0.35),
                  (280, 0.08, 0.20)],
        "pre":   (120, 0.20, 0.50),
    },
    "Bang": {
        "taps":  [(0, 1.00, 0.85), (40, 0.50, 0.55), (95, 0.20, 0.25)],
        "tail":  (0.45, 0.35, 0.25, 0.20),
    },
    "Thump, thud": {
        "taps":  [(0, 1.00, 0.55), (50, 0.45, 0.30)],
        "tail":  (0.55, 0.40, 0.25, 0.10),
    },
    "Whack, thwack": {"taps": [(0, 1.00, 0.85), (28, 0.45, 0.65)]},
    "Slap, smack":   {"taps": [(0, 1.00, 0.80), (25, 0.40, 0.60)]},
}

_IMPACT_TEMPLATE_DEFAULT: dict = {
    "taps": [(0, 1.00, 0.95), (30, 0.70, 0.60), (60, 0.40, 0.25)],
}


@dataclass
class ScoringWeights:
    """Relative importance of each signal source.

    Percussive captures transient energy (impacts, drums).
    Bass bands drive the rumble.  AI amplifies semantically
    meaningful sounds identified by YAMNet.
    """

    percussive: float = 0.20   # HPSS percussive RMS
    sub_bass: float = 0.18     # 20-60 Hz deep rumble
    bass: float = 0.18         # 60-250 Hz punch
    low_mid: float = 0.05      # 250-500 Hz body
    mid: float = 0.05          # 500-2000 Hz texture
    presence: float = 0.04     # 2000-4000 Hz detail
    ai: float = 0.17           # YAMNet haptic score
    video: float = 0.13        # Video motion intensity


# ── Style-specific weight presets ────────────────────────

_STYLE_WEIGHTS: dict[str, ScoringWeights] = {
    "music": ScoringWeights(
        percussive=0.28, sub_bass=0.22, bass=0.22,
        low_mid=0.06, mid=0.06, presence=0.04,
        ai=0.12, video=0.00,          # all-audio for music
    ),
    "cinematic": ScoringWeights(
        percussive=0.15, sub_bass=0.12, bass=0.12,
        low_mid=0.04, mid=0.04, presence=0.03,
        ai=0.25, video=0.25,          # heavy AI + video for film
    ),
    "auto": ScoringWeights(),         # balanced default
}


def _detect_style(ai: AIClassification) -> str:
    """Auto-detect content style from YAMNet classification.

    Computes the ratio of music-dominant frames to speech-dominant
    frames.  If music clearly dominates → "music"; if speech
    dominates → "cinematic"; otherwise → "auto" (balanced).
    """
    if not ai.dominant_classes:
        return "auto"

    _MUSIC_LABELS = {
        "Music", "Rock music", "Pop music", "Hip hop music",
        "Heavy metal", "Punk rock", "Disco", "Electronic music",
        "Techno", "Drum", "Snare drum", "Bass drum", "Drum kit",
        "Guitar", "Electric guitar", "Acoustic guitar", "Piano",
        "Bass guitar", "Singing", "Cymbal", "Hi-hat", "Drum roll",
        "Synthesizer", "Organ", "Keyboard (musical)",
    }
    _SPEECH_LABELS = {
        "Speech", "Male speech, man speaking",
        "Female speech, woman speaking", "Child speech, kid speaking",
        "Conversation", "Narration, monologue",
    }

    music_count = sum(1 for c in ai.dominant_classes if c in _MUSIC_LABELS)
    speech_count = sum(1 for c in ai.dominant_classes if c in _SPEECH_LABELS)
    total = len(ai.dominant_classes)

    music_ratio = music_count / total if total > 0 else 0
    speech_ratio = speech_count / total if total > 0 else 0

    if music_ratio > 0.40 and music_ratio > speech_ratio * 2:
        return "music"
    if speech_ratio > 0.35 and speech_ratio > music_ratio * 1.5:
        return "cinematic"
    return "auto"


# ── Public API ───────────────────────────────────────────


def fuse_scores(
    dsp: DSPFeatures,
    ai: AIClassification,
    sensitivity: float = 0.5,
    bass_boost: float = 1.0,
    video: VideoFeatures | None = None,
    style: str = "auto",
) -> HapticTimeline:
    """
    Combine HPSS-separated DSP + YAMNet/Whisper AI signals + video
    motion/action recognition into a continuous haptic envelope plus
    transient accent events.

    Parameters
    ----------
    style : str
        "auto" (detect from content), "music", or "cinematic".
        Controls the relative weight of each signal source.
    """
    # ── Resolve style → weights ──────────────────────────
    effective_style = style.lower()
    if effective_style == "auto":
        effective_style = _detect_style(ai)
    weights = _STYLE_WEIGHTS.get(effective_style, ScoringWeights())
    logger.info("Fusion style: requested=%s, effective=%s", style, effective_style)
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

    # ── Per-band novelty gating ──────────────────────────
    # Suppress constant energy in each frequency band so ambient
    # noise (HVAC, room tone, wind, traffic) doesn't produce
    # continuous vibration.  Only varying (interesting) content
    # passes through.  Floor of 0.10 keeps faint musical
    # sustains alive while killing flat drone.
    _band_nov_win = max(3, int(1.0 / frame_dur))  # ~1 s window
    for _band in (sub_bass, bass, low_mid, mid, presence):
        _bnov = _rolling_std(_band, _band_nov_win)
        _bnov_max = np.percentile(_bnov, 98) if len(_bnov) > 0 else 1.0
        if _bnov_max > 1e-6:
            _bnov /= _bnov_max
        _bnov = np.clip(_bnov, 0.0, 1.0)
        _floor = settings.NOVELTY_FLOOR_PER_BAND
        _bgate = _floor + (1.0 - _floor) * _bnov  # range [floor, 1.0]
        _band[:] = _band * _bgate            # in-place modulation

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
    # Only use YAMNet backup where Whisper found NO speech (gate == 1.0).
    # Where Whisper already set a suppression value, trust its precise timing
    # over YAMNet's coarse 0.48s frames which bleed across boundaries.
    _whisper_has_speech = speech_gate < 0.95
    speech_gate = np.where(
        _whisper_has_speech,
        speech_gate,                          # Whisper active -> trust it
        np.minimum(speech_gate, yamnet_gate), # Whisper silent -> use YAMNet backup
    )

    # Smooth gate edges (crossfade)
    gate_smooth = max(
        1, int((settings.SPEECH_GATE_SMOOTH_MS / 1000.0) / frame_dur)
    )
    speech_gate = _smooth(speech_gate, gate_smooth)
    speech_gate = np.where(speech_gate < 0.02, 0.0, speech_gate)

    # ── Haptic-content override ──────────────────────────
    # Impact classes (explosions, gunshots, crashes) always override
    # speech suppression so they vibrate through dialogue.  Score-based
    # threshold is a fallback for unlabeled but strong impacts.
    _is_impact_frame = np.array([lbl in _IMPACT_LABELS_SET for lbl in dsp_dominant])
    haptic_override = _is_impact_frame | (ai_haptic > settings.HAPTIC_OVERRIDE_THRESHOLD)
    speech_gate = np.where(
        haptic_override,
        np.maximum(speech_gate, settings.HAPTIC_OVERRIDE_PASS_THROUGH),
        speech_gate,
    )

    # ── Percussive novelty gating ────────────────────────
    # Constant percussive energy (machine hum, engine) → suppress.
    # Floor of 0.15: steady rhythms (drums, bass lines) keep some
    # energy, but flat drones (A/C, engine idle) are cut hard.
    novelty_win = max(3, int(0.5 / frame_dur))
    perc_nov = _rolling_std(perc_rms, novelty_win)
    perc_nmax = np.percentile(perc_nov, 98) if len(perc_nov) > 0 else 1.0
    if perc_nmax > 1e-6:
        perc_nov /= perc_nmax
    perc_nov = np.clip(perc_nov, 0.0, 1.0)
    _pf = settings.NOVELTY_FLOOR_PERCUSSIVE
    perc_gate = _pf + (1.0 - _pf) * perc_nov        # [floor, 1.0]
    perc_modulated = perc_rms * perc_gate

    # ── Harmonic novelty gating ───────────────────────────
    # harm_rms carries sustained tones (HPSS harmonic component).
    # Ambient hum/drone is sustained → lands here.  Gate it so
    # only *varying* harmonic content (melody, chord changes) passes.
    _harm_nov = _rolling_std(harm_rms, _band_nov_win)
    _harm_nov_max = np.percentile(_harm_nov, 98) if len(_harm_nov) > 0 else 1.0
    if _harm_nov_max > 1e-6:
        _harm_nov /= _harm_nov_max
    _harm_nov = np.clip(_harm_nov, 0.0, 1.0)
    _hf = settings.NOVELTY_FLOOR_HARMONIC
    _harm_gate = _hf + (1.0 - _hf) * _harm_nov
    harm_rms_gated = harm_rms * _harm_gate

    # ── Combine weighted signals ─────────────────────────
    # Include harmonic RMS so sustained music (strings, pads,
    # singing) drives continuous vibration — not just transients.
    harmonic_contribution = 0.15 * harm_rms_gated

    # Voice F0 (~85-255 Hz) overlaps sub_bass+bass+low_mid; even
    # after speech_gate=0.05 those bands carry voice prosody as low
    # rumble. Hard-mute them on ANY detected-speech frame (gate < 0.95)
    # so borderline-confidence speech is also handled. mid/presence
    # still carry impact override energy for explosions in dialogue.
    _voice_mute_active = speech_gate < 0.95
    voice_band_mute = np.where(_voice_mute_active, 0.0, 1.0).astype(np.float64)
    voice_band_mute = _smooth(voice_band_mute, gate_smooth)
    voice_band_mute = np.where(haptic_override, 1.0, voice_band_mute)

    sub_bass_v = sub_bass * voice_band_mute
    bass_v = bass * voice_band_mute
    low_mid_v = low_mid * voice_band_mute

    combined = (
        weights.percussive * perc_modulated
        + weights.sub_bass * sub_bass_v
        + weights.bass * bass_v
        + weights.low_mid * low_mid_v
        + weights.mid * mid
        + weights.presence * presence
        + weights.ai * ai_haptic
        + harmonic_contribution
    )

    # ── Video motion contribution ────────────────────────
    # Resample video motion_intensity to DSP frame rate and add
    # as a weighted signal.  When no video data, weight is zero.
    video_motion = np.zeros(n_frames, dtype=np.float64)
    video_actions: list[str] = []
    video_action_scores: dict[str, np.ndarray] = {}
    if video is not None and video.motion_intensity:
        video_motion = _resample_to_length(
            np.array(video.motion_intensity, dtype=np.float64), n_frames
        )
        combined += weights.video * video_motion
        logger.info(
            "Video motion fused: avg=%.3f, peak=%.3f",
            float(np.mean(video_motion)),
            float(np.max(video_motion)),
        )
        # Resample action labels to DSP frame rate
        if video.dominant_actions:
            _va_n = len(video.dominant_actions)
            _va_idx = np.clip(
                (np.arange(n_frames) * frame_dur / video.action_window_duration_s).astype(int),
                0, _va_n - 1,
            )
            video_actions = [video.dominant_actions[i] for i in _va_idx]
            # Resample per-category scores
            for cat, scores in video.action_scores.items():
                if scores:
                    video_action_scores[cat] = _resample_to_length(
                        np.array(scores, dtype=np.float64), n_frames
                    )
        else:
            video_actions = ["none"] * n_frames

    # ── Impact amplification ─────────────────────────────
    # When percussive + bass are both strong → genuine impact.
    # Boost by up to 2.5× so explosions/hits feel powerful.
    # Skip for acoustic drums — they self-punch through audio.
    # Detect drums via YAMNet per-class probabilities (not just dominant label).
    # Drum-heavy music often has dominant class "Music" or "Rock music",
    # but drum class probabilities are still high.
    _drum_label_match = np.array([lbl in _DRUM_LABELS_SET for lbl in dsp_dominant])
    if ai.drum_scores:
        _drum_prob = _resample_to_length(np.array(ai.drum_scores), n_frames)
        is_drum_frame = _drum_label_match | (_drum_prob > 0.15)
    else:
        is_drum_frame = _drum_label_match
    impact_factor = np.where(
        (perc_rms > 0.06) & (bass > 0.06) & (~is_drum_frame),
        1.0 + 1.5 * perc_rms * bass,
        1.0,
    )
    # Percussive-only boost for crashes that lack strong bass
    perc_only_boost = np.where(
        (perc_rms > 0.20) & (~is_drum_frame),
        1.0 + 0.8 * perc_rms,
        1.0,
    )
    combined *= impact_factor * perc_only_boost

    # ── Gunshot-specific amplification ──────────────────
    # When YAMNet detects gunshot/weapon classes, apply extra boost
    # on top of the generic impact amplification for punchier feel.
    _GUNSHOT_LABELS_SET = {"Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire"}
    _is_gunshot = np.array([lbl in _GUNSHOT_LABELS_SET for lbl in dsp_dominant])
    gunshot_boost = np.where(_is_gunshot, 1.5, 1.0)
    combined *= gunshot_boost

    # ── Machine gun cadence modulation ──────────────────
    # Overlay rapid 9 Hz pulsing during sustained automatic fire
    # for a distinctive rapid-fire vibration feel.
    _is_machinegun = np.array([lbl == "Machine gun" for lbl in dsp_dominant])
    if np.any(_is_machinegun):
        _mg_rate = 9.0  # Hz (simulates rate of fire)
        _t_arr = np.arange(n_frames) * frame_dur
        _mg_pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * _mg_rate * _t_arr)
        _mg_mod = np.where(_is_machinegun, 0.7 + 0.3 * _mg_pulse, 1.0)
        combined *= _mg_mod

    # ── Scenario-specific modulation from video actions ──
    # Each detected action scenario applies a distinct pattern
    # to the haptic signal for differentiated feel.
    scenario_transients: list[HapticEvent] = []
    if video_actions:
        combined, sharpness_mod, scenario_transients = _apply_scenario_modulation(
            combined=combined,
            video_actions=video_actions,
            video_motion=video_motion,
            video_action_scores=video_action_scores,
            frame_dur=frame_dur,
            n_frames=n_frames,
        )
    else:
        sharpness_mod = np.zeros(n_frames, dtype=np.float64)

    # ── Global ambient / variance gate ────────────────────
    # Even after per-band novelty gating, the sum of many small
    # residuals can produce a non-trivial constant signal.  Measure
    # the rolling variance of the *combined* signal: where it is
    # flat (no dynamics), suppress toward zero.
    _ambient_win = max(3, int(0.5 / frame_dur))     # ~0.5 s window (faster reaction)
    combined_var = _rolling_std(combined, _ambient_win)
    _cvar_max = np.percentile(combined_var, 98) if len(combined_var) > 0 else 1.0
    if _cvar_max > 1e-6:
        combined_var /= _cvar_max
    combined_var = np.clip(combined_var, 0.0, 1.0)
    # Gate: ramp from 0.05 (flat signal) to 1.0 (dynamic signal)
    ambient_gate = np.clip((combined_var - 0.02) / 0.10, settings.NOVELTY_FLOOR_GLOBAL_AMBIENT, 1.0)
    combined *= ambient_gate

    # ── AI activity gate ──────────────────────────────────
    # When YAMNet classifies nothing haptic-worthy (wind, silence,
    # white noise, room tone), suppress the DSP-derived continuous
    # signal.  Only semantically meaningful content passes fully.
    _ai_activity = _pad_or_trim_np(ai_haptic, n_frames)
    _ai_active = _ai_activity > settings.AI_ACTIVITY_GATE_THRESHOLD
    _ai_active_smooth = _smooth(
        _ai_active.astype(np.float64), max(1, int(0.1 / frame_dur))
    )
    _ai_gate = (
        settings.AI_ACTIVITY_GATE_FLOOR
        + (1.0 - settings.AI_ACTIVITY_GATE_FLOOR) * _ai_active_smooth
    )
    # Bypass AI activity gate for impact frames — gunshots/explosions
    # should not be suppressed just because YAMNet's overall activity is low.
    _impact_bypass = _pad_or_trim_np(
        haptic_override.astype(np.float64), n_frames
    )
    _ai_gate_with_bypass = np.maximum(_ai_gate, _impact_bypass)
    combined *= _ai_gate_with_bypass

    # ── Apply speech gate (once only) ─────────────────────
    combined *= speech_gate

    # ── Drum suppression gate ─────────────────────────────
    # Acoustic drums already provide natural "kick" through audio.
    # Suppress haptic signal when YAMNet detects drums as dominant.
    drum_mask = is_drum_frame.astype(np.float64)
    drum_gate = 1.0 - drum_mask * (1.0 - settings.DRUM_SUPPRESSION_FACTOR)
    drum_gate = _smooth(drum_gate, max(1, int(0.08 / frame_dur)))
    combined *= drum_gate

    # ── Music genre suppression gate ─────────────────────
    # Drum-heavy music often gets dominant class "Music" / "Rock music",
    # bypassing the drum gate.  Apply moderate suppression to all
    # music-genre frames so they produce noticeably less vibration
    # than action/impacts but still some feedback.
    _MUSIC_LABELS_SET = {
        "Music", "Rock music", "Pop music", "Hip hop music",
        "Heavy metal", "Punk rock", "Disco", "Electronic music", "Techno",
    }
    is_music_frame = np.array([lbl in _MUSIC_LABELS_SET for lbl in dsp_dominant])
    music_suppression = 0.40  # 40% pass-through for music
    music_gate = 1.0 - is_music_frame.astype(np.float64) * (1.0 - music_suppression)
    music_gate = _smooth(music_gate, max(1, int(0.08 / frame_dur)))
    combined *= music_gate

    # ── Silence gate (raw RMS) ───────────────────────────
    raw_rms_trimmed = _pad_or_trim_np(raw_rms, n_frames)
    silence_mask = raw_rms_trimmed < settings.SILENCE_RMS_THRESHOLD
    combined = _apply_silence_fade(combined, silence_mask, fade_frames=3)

    # ── Clamp & comfort ceiling ──────────────────────────
    combined = np.clip(combined, 0.0, 1.0)
    envelope_signal = combined.copy()

    # ── Adaptive rest gate — zero out faint frames ───────
    # Variance-aware: use a higher ceiling so ambient residual
    # that survived earlier gates gets caught here.
    local_median = np.median(envelope_signal[envelope_signal > 0]) if np.any(envelope_signal > 0) else 0.0
    rest_threshold = min(0.06, max(0.015, 0.10 * local_median))
    envelope_signal[envelope_signal < rest_threshold] = 0.0

    # ── Perceptual floor boost (variance-conditional) ────
    # Remap non-silent frames from [0, 1] to [0.20, 1.0] so
    # audible *dynamic* content produces a perceptible vibration.
    # BUT: only apply the 0.20 floor where the *post-speech* signal
    # has real variance.  Using pre-speech ambient_gate would
    # incorrectly boost speech-suppressed frames (speech is dynamic
    # audio, so ambient_gate > 0.30 during dialogue).
    _post_speech_var = _rolling_std(envelope_signal, max(3, int(1.0 / frame_dur)))
    _psv_max = np.percentile(_post_speech_var, 98) if len(_post_speech_var) > 0 else 1.0
    if _psv_max > 1e-6:
        _post_speech_var /= _psv_max
    _env_var = np.clip(_post_speech_var, 0.0, 1.0)
    # Exclude speech-suppressed frames from the perceptual floor boost.
    # Use < 0.95 so even low-confidence Whisper segments (whose gate
    # only drops to ~0.5) are excluded — otherwise borderline speech
    # gets re-boosted to 0.20+ and defeats dialogue silencing.
    _speech_suppressed = _pad_or_trim_np(speech_gate, len(envelope_signal)) < 0.95
    _dynamic_mask = (envelope_signal > 0.01) & (_env_var > 0.30) & (~_speech_suppressed)
    _ambient_mask = (envelope_signal > 0.01) & (~_dynamic_mask)
    envelope_signal = np.where(
        _dynamic_mask,
        0.20 + envelope_signal * 0.80,   # full boost for dynamic
        np.where(
            _ambient_mask,
            envelope_signal,              # no boost for ambient
            0.0,                          # silence stays silent
        ),
    )
    envelope_signal = np.clip(envelope_signal, 0.0, 1.0)

    # ── Post-boost rest gate ──────────────────────────────
    # Non-dynamic frames that survived the boost stage with low
    # residual intensity are ambient artifacts — zero them out.
    _non_dynamic_low = (
        (~_dynamic_mask)
        & (envelope_signal > 0.0)
        & (envelope_signal < settings.POST_BOOST_REST_THRESHOLD)
    )
    envelope_signal[_non_dynamic_low] = 0.0

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
    _DRUM_LABELS = _DRUM_LABELS_SET
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

    # ── Blend video scenario sharpness modifier ──────────
    # The scenario modulation function returns a sharpness offset
    # that is blended in additively (clamped to 0-1).
    if np.any(sharpness_mod != 0):
        sharpness = np.clip(sharpness + 0.55 * sharpness_mod, 0.05, 0.95)

    # Compute transient threshold here — needed by both the rumble-tail
    # gate below and the transient extractor a few lines down.
    threshold = 0.45 - (sensitivity * 0.40)
    threshold = max(0.05, threshold)

    # ── Per-impact rumble tail + sharpness peak ──────────
    # For each detected impact frame, add an exponential decay onto
    # envelope_signal (per-class tail duration / decay tau) and punch
    # sharpness toward IMPACT_SHARPNESS_PEAK with its own decay.
    # Uses max(...) so existing louder content is never lowered.
    if (ai_haptic is not None
            and len(dsp_dominant) == n_frames
            and len(envelope_signal) == n_frames):
        _rumble_max = settings.IMPACT_RUMBLE_TAIL_S
        _sharp_peak = settings.IMPACT_SHARPNESS_PEAK
        _sharp_tau = settings.IMPACT_SHARPNESS_DECAY_S
        _sharp_n = max(1, int((_sharp_tau * 5.0) / frame_dur))
        _last_tail_t = -1.0
        _BURST_CLASSES_LOCAL = {
            "Explosion", "Smash, crash", "Bang", "Thunder",
            "Thunderstorm", "Gunshot, gunfire", "Machine gun",
            "Fusillade", "Artillery fire", "Thump, thud",
            "Whack, thwack", "Slap, smack",
        }
        for _fi in range(n_frames):
            if ai_haptic[_fi] < 0.15:
                continue
            if combined[_fi] < threshold * 0.5:
                continue
            _lbl = dsp_dominant[_fi]
            if _lbl not in _BURST_CLASSES_LOCAL:
                continue
            _t_now = _fi * frame_dur
            if (_t_now - _last_tail_t) < 0.40:
                continue
            _last_tail_t = _t_now
            _tmpl = _IMPACT_TEMPLATES.get(_lbl, _IMPACT_TEMPLATE_DEFAULT)

            # Sharpness peak (always — even templates without tails crisp up)
            _end_sf = min(n_frames, _fi + _sharp_n)
            for _sf in range(_fi, _end_sf):
                _decay_s = float(np.exp(-(_sf - _fi) * frame_dur / _sharp_tau))
                _target_s = _sharp_peak * _decay_s + sharpness[_sf] * (1.0 - _decay_s)
                if _target_s > sharpness[_sf]:
                    sharpness[_sf] = _target_s

            # Rumble tail (only for templates that specify one)
            _tail = _tmpl.get("tail")
            if _tail is None:
                continue
            _tail_dur, _tail_peak, _tail_tau, _ = _tail
            _tail_dur = min(_tail_dur, _rumble_max)
            _end_tf = min(n_frames, _fi + int(_tail_dur / frame_dur))
            _base_int = float(np.clip(0.85 + 0.15 * ai_haptic[_fi], 0.85, 1.0))
            for _tf in range(_fi, _end_tf):
                _decay_t = float(np.exp(-(_tf - _fi) * frame_dur / _tail_tau))
                _v = _base_int * _tail_peak * _decay_t
                if _v > envelope_signal[_tf]:
                    envelope_signal[_tf] = _v
        sharpness = np.clip(sharpness, 0.05, 0.95)
        envelope_signal = np.clip(envelope_signal, 0.0, 1.0)

    # ── Downsample to envelope rate (~20 fps) ────────────
    intensity_env, actual_env_fps = _downsample_max(envelope_signal, frame_dur, ENVELOPE_FPS)
    sharpness_env, _ = _downsample_mean(sharpness, frame_dur, ENVELOPE_FPS)

    # ── Extract transient tap events ─────────────────────
    # Use the raw post-gate combined signal (NOT _boost_array which
    # remaps everything > 0.01 up to 0.25 — that floor would fire
    # 25%-intensity taps on every suppressed-speech residual frame).

    events = _extract_transient_events(
        combined=combined,
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

    # ── Scene-cut accent transients from video ───────────
    # Each detected scene change gets a tap scaled by cut magnitude,
    # but only if the audio at that moment isn't fully silenced — a
    # cut over silent dialogue or black frames should not click.
    if video is not None and video.scene_changes:
        min_interval_sc = settings.MIN_TRANSIENT_INTERVAL_MS / 1000.0
        added_sc = 0
        for sc in video.scene_changes:
            sc_frame = int(sc.time / frame_dur)
            if sc_frame >= n_frames or combined[sc_frame] < threshold * 0.5:
                continue
            too_close = any(abs(e.time - sc.time) < min_interval_sc for e in events)
            if not too_close:
                norm_mag = min(sc.magnitude / 5.0, 1.0)
                events.append(HapticEvent(
                    time=round(sc.time, 4),
                    event_type="transient",
                    duration=0.0,
                    intensity=round(0.50 + 0.50 * norm_mag, 4),
                    sharpness=round(0.40 + 0.50 * norm_mag, 4),
                ))
                added_sc += 1
        events.sort(key=lambda e: e.time)
        logger.info("Added %d/%d scene-cut transients (audio-gated)", added_sc, len(video.scene_changes))

    # ── Video scenario transients ────────────────────
    # Per-scenario transient events from action recognition. These are
    # video-driven, so audio-gate them: a "punch" action with no impact
    # sound should not click the wrist.
    if scenario_transients:
        min_interval_vt = settings.MIN_TRANSIENT_INTERVAL_MS / 1000.0
        added_vt = 0
        for vt in scenario_transients:
            vt_frame = int(vt.time / frame_dur)
            if vt_frame >= n_frames or combined[vt_frame] < threshold * 0.5:
                continue
            too_close = any(abs(e.time - vt.time) < min_interval_vt for e in events)
            if not too_close:
                events.append(vt)
                added_vt += 1
        events.sort(key=lambda e: e.time)
        logger.info("Added %d/%d scenario-specific transients (audio-gated)", added_vt, len(scenario_transients))

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
            "style": effective_style,
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

    guard_pre = settings.SPEECH_GUARD_PRE_MS / 1000.0
    guard_post = settings.SPEECH_GUARD_POST_MS / 1000.0
    floor = settings.SPEECH_SUPPRESSION_FACTOR
    for seg in speech_segments:
        start_s = max(0.0, seg.start - guard_pre)
        end_s = seg.end + guard_post
        start_f = int(start_s / frame_dur)
        end_f = min(n_frames, int(end_s / frame_dur) + 1)

        # Quadratic ramp toward SPEECH_SUPPRESSION_FACTOR floor —
        # borderline confidence still suppresses substantially, and
        # confirmed speech bottoms out at the configured floor.
        conf = float(seg.confidence)
        gate_val = floor + (1.0 - floor) * (1.0 - conf) ** 2
        gate[start_f:end_f] = np.minimum(gate[start_f:end_f], gate_val)

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
            if onset[fi] < 0.60:  # allow very strong onsets at speech boundary
                continue
        # Skip drum-dominated frames — drums self-punch through audio
        if dsp_dominant is not None and fi < len(dsp_dominant) and dsp_dominant[fi] in _DRUM_LABELS_SET:
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
        if bs < 0.25:
            continue
        beat_frame = int(bt / frame_dur)
        # Underlying combined signal must be above the same gate used for
        # onset transients — prevents steady "tik tik" beats firing in
        # silent or fully-gated frames where there's nothing to vibrate to.
        if beat_frame >= n_frames or combined[beat_frame] < threshold * 0.5:
            continue
        if speech_gate is not None:
            if beat_frame < len(speech_gate) and speech_gate[beat_frame] < 0.1:
                if bs < 0.50:  # allow very strong beats at speech boundary
                    continue
        # Skip beats on drum-dominant frames
        if dsp_dominant is not None and beat_frame < len(dsp_dominant) and dsp_dominant[beat_frame] in _DRUM_LABELS_SET:
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

    # ── Crash/impact burst transients (per-class templates) ──
    # Each impact class has its own multi-tap signature in
    # _IMPACT_TEMPLATES (offset, intensity scale, sharpness per tap)
    # plus optional pre-impact whoosh and rumble tail.
    _BURST_CLASSES = {
        "Explosion", "Smash, crash", "Bang", "Thunder",
        "Thunderstorm", "Gunshot, gunfire", "Machine gun",
        "Fusillade", "Artillery fire", "Thump, thud",
        "Whack, thwack", "Slap, smack",
    }
    _RAPID_FIRE_CLASSES = {"Machine gun", "Fusillade", "Artillery fire"}
    if ai_haptic is not None and dsp_dominant is not None:
        last_burst_t = -1.0
        for fi in range(n_frames):
            if fi >= len(ai_haptic) or fi >= len(dsp_dominant):
                break
            if ai_haptic[fi] < 0.15:
                continue
            lbl = dsp_dominant[fi]
            if lbl not in _BURST_CLASSES:
                continue
            # YAMNet false positives can label silent frames as "Explosion".
            if combined[fi] < threshold * 0.5:
                continue
            t = fi * frame_dur
            _cooldown = 0.12 if lbl in _RAPID_FIRE_CLASSES else 0.50
            if (t - last_burst_t) < _cooldown:
                continue

            tmpl = _IMPACT_TEMPLATES.get(lbl, _IMPACT_TEMPLATE_DEFAULT)
            base_int = float(np.clip(0.85 + 0.15 * ai_haptic[fi], 0.85, 1.0))

            # Pre-impact anticipation tap (audio-gated against silence/dialogue)
            pre = tmpl.get("pre")
            if pre is not None:
                p_off_ms, p_iscale, p_sharp = pre
                pt = t - (p_off_ms / 1000.0)
                pf = int(pt / frame_dur)
                if (pt >= 0.0 and 0 <= pf < n_frames
                        and combined[pf] >= threshold * 0.25
                        and not any(abs(e.time - pt) < 0.015 for e in events)):
                    events.append(HapticEvent(
                        time=round(pt, 4),
                        event_type="transient",
                        duration=0.0,
                        intensity=round(base_int * p_iscale, 4),
                        sharpness=round(p_sharp, 4),
                    ))

            # Main multi-tap burst from template
            for off_ms, iscale, sharp in tmpl["taps"]:
                bt = t + (off_ms / 1000.0)
                if any(abs(e.time - bt) < 0.012 for e in events):
                    continue
                events.append(HapticEvent(
                    time=round(bt, 4),
                    event_type="transient",
                    duration=0.0,
                    intensity=round(base_int * iscale, 4),
                    sharpness=round(sharp, 4),
                ))
            last_burst_t = t

    events.sort(key=lambda e: e.time)
    return events


# ── Scenario-specific haptic modulation ──────────────────


def _apply_scenario_modulation(
    combined: np.ndarray,
    video_actions: list[str],
    video_motion: np.ndarray,
    video_action_scores: dict[str, np.ndarray],
    frame_dur: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[HapticEvent]]:
    """Apply scenario-specific intensity/sharpness patterns with temporal waveforms.

    Each action category produces a distinct haptic feel with unique
    temporal dynamics and per-scenario transient events:

    - **impact** (fighting): exponential decay after peak, double-tap transients
    - **chase** (running):   adaptive-cadence pulsing (2-6 Hz) + accent taps
    - **crash** (collisions): sustain plateau → damped oscillation aftershock,
                              5-tap decaying burst transients
    - **fall** (jumping):    rising ramp → landing spike, landing thud transient
    - **driving** (racing):  engine RPM oscillation, deep sharpness
    - **sports_hit**:        accent boost with sharp crack transient

    Returns
    -------
    combined : np.ndarray
        Modified intensity signal.
    sharpness_mod : np.ndarray
        Additive sharpness modifier (-1 to +1) to blend with base sharpness.
    scenario_transients : list[HapticEvent]
        Per-scenario transient (tap) events.
    """
    result = combined.copy()
    sharpness_mod = np.zeros(n_frames, dtype=np.float64)
    transients: list[HapticEvent] = []

    # ── Temporal state tracking ──────────────────────────
    chase_phase = 0.0              # phase accumulator for adaptive cadence
    impact_peak_t = -10.0          # time of last impact peak
    impact_peak_val = 0.0          # boost value at peak
    crash_peak_t = -10.0           # time of last crash peak
    crash_peak_val = 0.0           # crash boost at peak
    fall_start_t = -10.0           # start of current fall sequence
    fall_landed = False            # whether landing thud has fired
    prev_action = "none"

    # ── Transient cooldowns ──────────────────────────────
    last_impact_tap_t = -10.0
    last_chase_tap_t = -10.0
    last_crash_burst_t = -10.0
    last_sports_tap_t = -10.0

    IMPACT_TAP_CD = 0.30           # seconds between impact double-taps
    CHASE_TAP_CD = 0.12            # seconds between chase accent taps
    CRASH_BURST_CD = 0.60          # seconds between crash burst salvos
    SPORTS_TAP_CD = 0.40           # seconds between sports crack taps

    for fi in range(n_frames):
        if fi >= len(video_actions):
            break
        action = video_actions[fi]
        t = fi * frame_dur
        motion = float(video_motion[fi]) if fi < len(video_motion) else 0.0

        # Reset state on scenario transitions
        if action != prev_action:
            if action == "fall":
                fall_start_t = t
                fall_landed = False
            if action == "chase":
                chase_phase = 0.0

        # ── IMPACT ───────────────────────────────────────
        if action == "impact":
            score = video_action_scores.get("impact", np.zeros(1))
            s = float(score[min(fi, len(score) - 1)]) if len(score) > 0 else 0.0
            raw_boost = max(s, motion)

            # Track peak for decay envelope
            if raw_boost > 0.25 and raw_boost >= impact_peak_val * 0.8:
                impact_peak_t = t
                impact_peak_val = raw_boost

            # Exponential decay after peak (tau=150ms)
            dt = t - impact_peak_t
            decay = np.exp(-dt / 0.15) if 0 < dt < 2.0 else 1.0

            result[fi] *= 1.0 + 0.5 * raw_boost * decay
            sharpness_mod[fi] = 0.50  # metallic punch

            # Double-tap transient when motion spikes
            if raw_boost > 0.20 and (t - last_impact_tap_t) > IMPACT_TAP_CD:
                prev_m = float(video_motion[fi - 1]) if fi > 0 and fi - 1 < len(video_motion) else 0.0
                motion_deriv = motion - prev_m
                if motion_deriv > 0.12 or s > 0.15:
                    tap_int = min(1.0, 0.85 * raw_boost + 0.15)
                    transients.append(HapticEvent(
                        time=round(t, 4),
                        event_type="transient",
                        intensity=round(tap_int, 4),
                        sharpness=0.90,
                    ))
                    transients.append(HapticEvent(
                        time=round(t + 0.040, 4),
                        event_type="transient",
                        intensity=round(tap_int * 0.70, 4),
                        sharpness=0.60,
                    ))
                    last_impact_tap_t = t

        # ── CHASE ────────────────────────────────────────
        elif action == "chase":
            # Adaptive cadence: 2 Hz jog → 6 Hz sprint
            freq = 2.0 + 4.0 * motion
            chase_phase += 2.0 * np.pi * freq * frame_dur
            pulse = 0.5 + 0.5 * np.sin(chase_phase)

            result[fi] *= 0.65 + 0.35 * pulse * max(0.3, motion)
            sharpness_mod[fi] = 0.15  # footstep crispness

            # Accent tap at pulse peaks
            if (pulse > 0.92
                    and motion > 0.15
                    and (t - last_chase_tap_t) > CHASE_TAP_CD):
                tap_int = min(1.0, 0.35 + 0.40 * motion)
                transients.append(HapticEvent(
                    time=round(t, 4),
                    event_type="transient",
                    intensity=round(tap_int, 4),
                    sharpness=0.55,
                ))
                last_chase_tap_t = t

        # ── CRASH ────────────────────────────────────────
        elif action == "crash":
            score = video_action_scores.get("crash", np.zeros(1))
            s = float(score[min(fi, len(score) - 1)]) if len(score) > 0 else 0.0
            raw_crash = max(s, motion)

            # Track crash peak
            if raw_crash > 0.25 and raw_crash >= crash_peak_val * 0.7:
                crash_peak_t = t
                crash_peak_val = raw_crash

            dt = t - crash_peak_t
            if 0 <= dt < 0.20:
                # Sustain plateau: high intensity during initial crash
                result[fi] *= 1.0 + 0.7 * raw_crash
                sharpness_mod[fi] = 0.55 * max(0.2, motion)
            elif 0.20 <= dt < 1.5:
                # Damped oscillation aftershock
                decay_phase = dt - 0.20
                aftershock = crash_peak_val * np.exp(-4.0 * decay_phase) * (
                    0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * decay_phase)
                )
                result[fi] *= 1.0 + 0.5 * max(0.0, aftershock)
                sharpness_mod[fi] = 0.55 * max(0.1, motion) * np.exp(-3.0 * decay_phase)
            else:
                result[fi] *= 1.0 + 0.3 * raw_crash
                sharpness_mod[fi] = 0.20 * motion

            # 5-tap decaying burst at crash peak
            if (raw_crash > 0.25
                    and dt < 0.05
                    and (t - last_crash_burst_t) > CRASH_BURST_CD):
                burst_offsets = [0.0, 0.025, 0.055, 0.095, 0.145]
                burst_int_scale = [1.0, 0.75, 0.50, 0.30, 0.15]
                burst_sharp = [0.95, 0.80, 0.60, 0.40, 0.20]
                base_int = min(1.0, 0.80 + 0.20 * raw_crash)
                for bi in range(5):
                    transients.append(HapticEvent(
                        time=round(t + burst_offsets[bi], 4),
                        event_type="transient",
                        intensity=round(base_int * burst_int_scale[bi], 4),
                        sharpness=burst_sharp[bi],
                    ))
                last_crash_burst_t = t

        # ── FALL ─────────────────────────────────────────
        elif action == "fall":
            elapsed = t - fall_start_t
            # Rising ramp over ~2 seconds
            ramp = min(1.0, elapsed / 2.0)

            if motion > 0.5 and not fall_landed:
                # Landing spike: high-intensity impact
                result[fi] *= 1.0 + 0.8 * motion
                sharpness_mod[fi] = 0.35 * motion

                # Landing thud transient — deep, heavy tap
                transients.append(HapticEvent(
                    time=round(t, 4),
                    event_type="transient",
                    intensity=0.95,
                    sharpness=0.15,  # deep thud
                ))
                fall_landed = True
            else:
                # During fall: gradual intensity build
                result[fi] *= 1.0 + 0.55 * ramp * motion
                sharpness_mod[fi] = 0.15 * ramp  # gradually sharpens

        # ── DRIVING ──────────────────────────────────────
        elif action == "driving":
            # Engine RPM oscillation: 1.5 Hz idle → 4.5 Hz high RPM
            f_rpm = 1.5 + 3.0 * motion
            engine_osc = 0.15 * np.sin(2.0 * np.pi * f_rpm * t)

            # Steady vibration floor + engine oscillation
            result[fi] = max(result[fi], 0.30 + 0.30 * motion + engine_osc)
            sharpness_mod[fi] = -0.50  # deep engine rumble

        # ── SPORTS_HIT ───────────────────────────────────
        elif action == "sports_hit":
            score = video_action_scores.get("sports_hit", np.zeros(1))
            s = float(score[min(fi, len(score) - 1)]) if len(score) > 0 else 0.0
            result[fi] *= 1.0 + 0.4 * max(s, motion * 0.5)
            sharpness_mod[fi] = 0.45  # crisp crack

            # Sharp crack transient
            if s > 0.12 and (t - last_sports_tap_t) > SPORTS_TAP_CD:
                tap_int = min(1.0, 0.70 + 0.30 * s)
                transients.append(HapticEvent(
                    time=round(t, 4),
                    event_type="transient",
                    intensity=round(tap_int, 4),
                    sharpness=0.95,  # maximum crispness
                ))
                last_sports_tap_t = t

        prev_action = action

    # Smooth with reduced window (~50 ms) to preserve waveform detail
    smooth_win = max(1, int(0.05 / frame_dur))
    result = _smooth(result, smooth_win)
    sharpness_mod = _smooth(sharpness_mod, smooth_win)

    transients.sort(key=lambda e: e.time)
    return result, sharpness_mod, transients


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
