"""AI-based sound event classification using PANNs (CNN14).

Uses pre-trained CNN14 from the PANNs (Pretrained Audio Neural Networks)
project to classify audio into 527 AudioSet classes at frame level.

This lets us semantically distinguish haptic-worthy sounds (explosions,
drums, bass, impacts) from non-worthy ones (speech, silence, ambient).
"""

from __future__ import annotations

import logging
from typing import Any

import librosa
import numpy as np
import torch

from app.core.config import get_settings
from app.models.schemas import AIClassification

logger = logging.getLogger(__name__)
settings = get_settings()

# ── AudioSet class indices for haptic-worthy / non-worthy events ──
# Full list: https://github.com/audioset/ontology

HAPTIC_WORTHY_CLASSES: dict[int, str] = {
    # Explosions & impacts
    426: "Explosion",
    427: "Burst, pop",
    428: "Eruption",
    429: "Gunshot, gunfire",
    430: "Machine gun",
    431: "Fusillade",
    432: "Artillery fire",
    450: "Cap gun",
    # Crashes & breaking
    469: "Crash",
    470: "Breaking",
    471: "Bouncing",
    472: "Whip",
    474: "Slam",
    # Thunder & weather
    399: "Thunder",
    400: "Thunderstorm",
    # Drums & percussion
    339: "Drum",
    340: "Snare drum",
    341: "Bass drum",
    342: "Rimshot",
    343: "Drum roll",
    344: "Cymbal",
    345: "Hi-hat",
    346: "Drum kit",
    347: "Tabla",
    # Bass & low-freq instruments
    326: "Bass guitar",
    327: "Electric bass guitar",
    310: "Bass drum",
    # Engine & machinery
    300: "Engine",
    301: "Motor vehicle (road)",
    302: "Car",
    308: "Motorcycle",
    310: "Truck",
    # Music categories (impactful)
    488: "Heavy metal",
    489: "Punk rock",
    137: "Music",
    141: "Musical instrument",
    # Other impacts
    473: "Slap, smack",
    476: "Thump, thud",
    477: "Knock",
    478: "Tap",
}

SPEECH_CLASSES: dict[int, str] = {
    0: "Speech",
    1: "Male speech, man speaking",
    2: "Female speech, woman speaking",
    3: "Child speech, kid speaking",
    4: "Conversation",
    5: "Narration, monologue",
    6: "Babbling",
    7: "Speech synthesizer",
    10: "Whispering",
    13: "Singing",
}

NON_WORTHY_CLASSES: dict[int, str] = {
    **SPEECH_CLASSES,
    # Silence & ambient
    494: "Silence",
    495: "White noise",
    # Gentle nature sounds
    396: "Rain",
    397: "Raindrop",
    398: "Rain on surface",
    401: "Wind",
    402: "Rustling leaves",
    # Crowd / murmur
    16: "Crowd",
    17: "Hubbub, speech noise, speech babble",
}


# Lazy-loaded model singleton
_model: Any | None = None
_labels: list[str] | None = None


def _load_model() -> Any:
    """Load PANNs CNN14 model (lazy singleton)."""
    global _model, _labels

    if _model is not None:
        return _model

    try:
        from panns_inference import AudioTagging

        logger.info("Loading PANNs CNN14 model…")
        _model = AudioTagging(
            checkpoint_path=settings.PANNS_CHECKPOINT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(
            "PANNs model loaded on %s",
            "CUDA" if torch.cuda.is_available() else "CPU",
        )

        # Load AudioSet labels
        try:
            import panns_inference

            _labels = list(panns_inference.labels) if hasattr(panns_inference, "labels") else None
        except Exception:
            _labels = None

        return _model

    except Exception as e:
        logger.warning(
            "PANNs model failed to load: %s – AI classification disabled. "
            "Install with: pip install panns-inference",
            str(e),
        )
        return None


def classify_audio(wav_path: str) -> AIClassification:
    """
    Run PANNs frame-level sound event detection.

    Parameters
    ----------
    wav_path : str
        Path to mono WAV at 32 000 Hz.

    Returns
    -------
    AIClassification
        Per-frame haptic scores, speech scores, and dominant class labels.
    """
    model = _load_model()

    # Load audio at PANNs sample rate (32 kHz)
    sr = settings.PANNS_SAMPLE_RATE
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = len(y) / sr
    logger.info("AI classification: %.2fs audio loaded", duration)

    if model is None:
        # Fallback: no AI model available → return neutral scores
        logger.warning("No PANNs model – returning neutral AI scores")
        return _fallback_classification(duration)

    # ── Segment audio into overlapping windows ───────────
    # PANNs expects ~10s clips; we use 1s windows with 0.5s hop
    # for finer temporal resolution
    window_s = 1.0
    hop_s = 0.5
    window_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)

    haptic_scores: list[float] = []
    speech_scores: list[float] = []
    dominant_classes: list[str] = []

    n_windows = max(1, int((len(y) - window_samples) / hop_samples) + 1)

    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples
        segment = y[start:end]

        # Pad if shorter than window
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))

        # PANNs inference
        audio_input = segment[np.newaxis, :]  # (1, samples)
        clipwise_output, _ = model.inference(audio_input)

        probs = clipwise_output[0]  # shape (527,)

        # Compute haptic-worthiness score
        h_score = _compute_haptic_score(probs)
        s_score = _compute_speech_score(probs)
        top_class = _get_dominant_class(probs)

        haptic_scores.append(round(float(h_score), 4))
        speech_scores.append(round(float(s_score), 4))
        dominant_classes.append(top_class)

    logger.info(
        "AI classification complete: %d frames, avg haptic=%.3f, avg speech=%.3f",
        len(haptic_scores),
        np.mean(haptic_scores),
        np.mean(speech_scores),
    )

    return AIClassification(
        frame_duration_s=hop_s,
        total_frames=len(haptic_scores),
        haptic_scores=haptic_scores,
        speech_scores=speech_scores,
        dominant_classes=dominant_classes,
    )


# ── Score computation helpers ────────────────────────────


def _compute_haptic_score(probs: np.ndarray) -> float:
    """Aggregate probability of haptic-worthy classes."""
    worthy_indices = list(HAPTIC_WORTHY_CLASSES.keys())
    valid_idx = [i for i in worthy_indices if i < len(probs)]
    if not valid_idx:
        return 0.0
    # Use top-3 max rather than sum to avoid inflated scores
    worthy_probs = probs[valid_idx]
    top_k = min(3, len(worthy_probs))
    return float(np.sort(worthy_probs)[-top_k:].mean())


def _compute_speech_score(probs: np.ndarray) -> float:
    """Aggregate probability of speech/dialogue classes."""
    speech_indices = list(SPEECH_CLASSES.keys())
    valid_idx = [i for i in speech_indices if i < len(probs)]
    if not valid_idx:
        return 0.0
    return float(np.max(probs[valid_idx]))


def _get_dominant_class(probs: np.ndarray) -> str:
    """Return the name of the highest-probability class."""
    global _labels
    top_idx = int(np.argmax(probs))

    # Check our known dictionaries first
    if top_idx in HAPTIC_WORTHY_CLASSES:
        return HAPTIC_WORTHY_CLASSES[top_idx]
    if top_idx in NON_WORTHY_CLASSES:
        return NON_WORTHY_CLASSES[top_idx]
    if _labels and top_idx < len(_labels):
        return _labels[top_idx]
    return f"class_{top_idx}"


# ── Fallback when no model is available ──────────────────


def _fallback_classification(duration: float) -> AIClassification:
    """
    Return neutral classification when PANNs is not available.
    All frames get a neutral haptic score of 0.5 (let DSP drive).
    """
    hop_s = 0.5
    n_frames = max(1, int(duration / hop_s))
    return AIClassification(
        frame_duration_s=hop_s,
        total_frames=n_frames,
        haptic_scores=[0.5] * n_frames,
        speech_scores=[0.0] * n_frames,
        dominant_classes=["unknown"] * n_frames,
    )
