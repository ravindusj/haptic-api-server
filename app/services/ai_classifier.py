"""AI-based sound event classification using YAMNet + faster-whisper.

Two complementary models work together:

1. **YAMNet** (Google, TensorFlow Hub) — classifies audio into 521
   AudioSet classes at ~0.48 s hop.  Tiny MobileNet-v1 backbone (18 MB),
   runs reliably on CPU, auto-cached.  Provides per-frame haptic-worthy
   and speech scores.

2. **faster-whisper** (CTranslate2) — state-of-the-art speech/dialogue
   detection via the Whisper model.  Returns precise speech segment
   timestamps.  Uses the ``tiny`` model (~75 MB, int8 quantised) on CPU.
   No PyTorch dependency.

Together they replace the old PANNs CNN14 (300 MB, frequently failed to
load) and the unreliable DSP speech heuristic.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

import librosa
import numpy as np

from app.core.config import get_settings
from app.models.schemas import AIClassification, SpeechSegment

logger = logging.getLogger(__name__)
settings = get_settings()

# ── YAMNet AudioSet class indices (521 classes) ──────────

HAPTIC_WORTHY_CLASSES: dict[int, str] = {
    # Explosions & impacts
    420: "Explosion",
    421: "Gunshot, gunfire",
    422: "Machine gun",
    423: "Fusillade",
    424: "Artillery fire",
    # Crashes & breaking
    463: "Smash, crash",
    454: "Thump, thud",
    460: "Bang",
    462: "Whack, thwack",
    464: "Slap, smack",
    # Thunder
    281: "Thunder",
    282: "Thunderstorm",
    # Drums & percussion
    159: "Drum",
    160: "Snare drum",
    163: "Bass drum",
    161: "Rimshot",
    162: "Drum roll",
    164: "Cymbal",
    165: "Hi-hat",
    166: "Drum kit",
    # Bass & low-freq instruments
    153: "Bass guitar",
    # Engine & machinery
    337: "Engine",
    338: "Motor vehicle (road)",
    340: "Car",
    346: "Motorcycle",
    348: "Truck",
    # Music categories (impactful)
    489: "Heavy metal",
    490: "Punk rock",
    132: "Music",
    135: "Guitar",
    # Other impacts
    455: "Knock",
    456: "Tap",
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
    24: "Singing",
}

NON_WORTHY_CLASSES: dict[int, str] = {
    **SPEECH_CLASSES,
    494: "Silence",
    495: "White noise",
    # Gentle nature sounds
    278: "Rain",
    279: "Raindrop",
    283: "Wind",
    # Crowd / murmur
    16: "Crowd",
    17: "Hubbub, speech noise",
}

# ── Lazy-loaded model singletons ─────────────────────────

_yamnet_model: Any | None = None
_yamnet_classes: list[str] | None = None
_whisper_model: Any | None = None


def _load_yamnet():
    """Load YAMNet model from TensorFlow Hub (lazy, ~18 MB)."""
    global _yamnet_model, _yamnet_classes

    if _yamnet_model is not None:
        return _yamnet_model

    try:
        import tensorflow_hub as hub
        import tensorflow as tf

        logger.info("Loading YAMNet model from TF Hub…")
        _yamnet_model = hub.load(settings.YAMNET_MODEL_HANDLE)

        # Load class names from the model's bundled CSV
        class_map_path = _yamnet_model.class_map_path().numpy()
        class_map_csv = tf.io.read_file(class_map_path).numpy().decode("utf-8")
        reader = csv.reader(io.StringIO(class_map_csv))
        next(reader)  # skip header
        _yamnet_classes = [row[2] for row in reader]

        logger.info("YAMNet loaded: %d classes", len(_yamnet_classes))
        return _yamnet_model

    except Exception as e:
        logger.warning("YAMNet failed to load: %s", str(e))
        return None


def _load_whisper():
    """Load faster-whisper tiny model (lazy, ~75 MB, int8 CPU)."""
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper '%s' model…", settings.WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("faster-whisper loaded (compute_type=%s)", settings.WHISPER_COMPUTE_TYPE)
        return _whisper_model

    except Exception as e:
        logger.warning("faster-whisper failed to load: %s", str(e))
        return None


# ── Public API ───────────────────────────────────────────


def classify_audio(wav_path: str) -> AIClassification:
    """
    Run YAMNet sound event detection + Whisper speech detection.

    Parameters
    ----------
    wav_path : str
        Path to mono WAV at 16 kHz.

    Returns
    -------
    AIClassification
        Per-frame haptic scores, speech scores, dominant labels,
        and Whisper speech segment timestamps.
    """
    # Load audio at 16 kHz for both YAMNet and Whisper
    sr = settings.CLASSIFIER_SAMPLE_RATE
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = float(len(y) / sr)
    logger.info("AI classification: %.2fs audio loaded at %d Hz", duration, sr)

    # ── YAMNet classification ────────────────────────────
    yamnet = _load_yamnet()
    if yamnet is not None:
        haptic_scores, speech_scores, dominant_classes = _run_yamnet(
            yamnet, y, sr, duration,
        )
    else:
        logger.warning("No YAMNet model – using neutral AI scores")
        n_frames = max(1, int(duration / 0.48))
        haptic_scores = [0.0] * n_frames
        speech_scores = [0.0] * n_frames
        dominant_classes = ["unknown"] * n_frames

    # ── Whisper speech detection ─────────────────────────
    speech_segments = _run_whisper(wav_path, duration)

    # YAMNet frame hop is 0.48 s
    frame_dur = 0.48

    logger.info(
        "AI classification complete: %d YAMNet frames, %d Whisper speech segments, "
        "avg haptic=%.3f, avg speech=%.3f",
        len(haptic_scores),
        len(speech_segments),
        np.mean(haptic_scores) if haptic_scores else 0.0,
        np.mean(speech_scores) if speech_scores else 0.0,
    )

    return AIClassification(
        frame_duration_s=frame_dur,
        total_frames=len(haptic_scores),
        haptic_scores=haptic_scores,
        speech_scores=speech_scores,
        dominant_classes=dominant_classes,
        speech_segments=speech_segments,
    )


# ── YAMNet inference ─────────────────────────────────────


def _run_yamnet(
    model: Any,
    waveform: np.ndarray,
    sr: int,
    duration: float,
) -> tuple[list[float], list[float], list[str]]:
    """Run YAMNet on the full waveform.

    YAMNet internally windows at 0.96 s with 0.48 s hop,
    producing ~2 frames/second.
    """
    import tensorflow as tf

    # YAMNet expects float32 in [-1, 1]
    waveform_f32 = waveform.astype(np.float32)
    # Ensure values are in [-1, 1]
    peak = np.max(np.abs(waveform_f32))
    if peak > 1.0:
        waveform_f32 = waveform_f32 / peak

    scores, embeddings, log_mel = model(waveform_f32)
    scores_np = scores.numpy()  # shape (N, 521)

    haptic_scores: list[float] = []
    speech_scores: list[float] = []
    dominant_classes: list[str] = []

    for frame_scores in scores_np:
        h = _compute_haptic_score(frame_scores)
        s = _compute_speech_score(frame_scores)
        top = _get_dominant_class(frame_scores)

        haptic_scores.append(round(float(h), 4))
        speech_scores.append(round(float(s), 4))
        dominant_classes.append(top)

    return haptic_scores, speech_scores, dominant_classes


# ── Whisper speech detection ─────────────────────────────


def _run_whisper(wav_path: str, duration: float) -> list[SpeechSegment]:
    """Detect speech segments using faster-whisper."""
    whisper = _load_whisper()
    if whisper is None:
        return []

    try:
        segments, info = whisper.transcribe(
            wav_path,
            beam_size=1,           # fastest decoding
            vad_filter=True,       # use Silero VAD for filtering
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
        )

        speech_segments: list[SpeechSegment] = []
        for seg in segments:
            # no_speech_prob: probability that this segment has no speech
            confidence = 1.0 - getattr(seg, "no_speech_prob", 0.0)
            if confidence < 0.3:
                continue  # skip very low confidence segments

            speech_segments.append(SpeechSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                confidence=round(confidence, 3),
            ))

        logger.info(
            "Whisper detected %d speech segments in %.1fs audio",
            len(speech_segments),
            duration,
        )
        return speech_segments

    except Exception as e:
        logger.warning("Whisper transcription failed: %s", str(e))
        return []


# ── Score computation helpers ────────────────────────────


def _compute_haptic_score(probs: np.ndarray) -> float:
    """Aggregate probability of haptic-worthy classes."""
    worthy_indices = [i for i in HAPTIC_WORTHY_CLASSES.keys() if i < len(probs)]
    if not worthy_indices:
        return 0.0
    worthy_probs = probs[worthy_indices]
    top_k = min(3, len(worthy_probs))
    return float(np.sort(worthy_probs)[-top_k:].mean())


def _compute_speech_score(probs: np.ndarray) -> float:
    """Aggregate probability of speech/dialogue classes."""
    speech_indices = [i for i in SPEECH_CLASSES.keys() if i < len(probs)]
    if not speech_indices:
        return 0.0
    return float(np.max(probs[speech_indices]))


def _get_dominant_class(probs: np.ndarray) -> str:
    """Return the name of the highest-probability class."""
    global _yamnet_classes
    top_idx = int(np.argmax(probs))

    if top_idx in HAPTIC_WORTHY_CLASSES:
        return HAPTIC_WORTHY_CLASSES[top_idx]
    if top_idx in NON_WORTHY_CLASSES:
        return NON_WORTHY_CLASSES[top_idx]
    if _yamnet_classes and top_idx < len(_yamnet_classes):
        return _yamnet_classes[top_idx]
    return f"class_{top_idx}"
