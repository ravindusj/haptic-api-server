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

# ── YAMNet AudioSet class mappings (B1) ────────────────────
#
# IMPORTANT: indices verified against YAMNet's bundled
# yamnet_class_map.csv (521 classes).  The previous mapping contained
# multiple wrong indices (e.g. idx 153 was labelled "Bass guitar" but
# YAMNet's idx 153 is actually "Synthesizer"; idx 278 was "Rain" but
# is "Rustling leaves"; idx 486 was "Rock music" but is "Clickety-clack").
# All entries below are now ground-truthed.
#
# Each entry maps idx → (display_name, haptic_weight).  Weight scale:
#   3.0 = very strong impact (explosion, crash, thunder, gunshot)
#   2.5 = strong impact (bang, slam, hammer, glass shatter)
#   2.0 = clear impact-worthy (knock, tap, click, alarm)
#   1.5 = bass-rich / sustained mechanical (engine, bass instruments,
#         liquids, vehicles, cheers, heart, tools, music genres)
#   1.0 = melodic / harmonic music content
#   0.3 = drums (suppressed — drums self-punch through audio)
HAPTIC_WORTHY_CLASSES: dict[int, tuple[str, float]] = {
    # ── Vocal non-speech (shouts / cheers / cries) ─────
    6:   ("Shout", 1.5),
    7:   ("Bellow", 1.5),
    8:   ("Whoop", 1.2),
    9:   ("Yell", 1.5),
    10:  ("Children shouting", 1.2),
    11:  ("Screaming", 1.5),
    24:  ("Singing", 1.0),
    25:  ("Choir", 1.0),
    26:  ("Yodeling", 1.0),
    27:  ("Chant", 1.0),
    31:  ("Rapping", 1.2),

    # ── Body / hands / heart ──────────────────────────
    56:  ("Hands", 1.5),
    57:  ("Finger snapping", 2.0),
    58:  ("Clapping", 2.0),
    59:  ("Heart sounds, heartbeat", 1.5),
    60:  ("Heart murmur", 1.0),
    61:  ("Cheering", 1.8),
    62:  ("Applause", 1.8),
    66:  ("Children playing", 1.0),

    # ── Animals (selected — barks/roars produce taps) ──
    70:  ("Bark", 1.5),
    71:  ("Yip", 1.0),
    72:  ("Howl", 1.2),
    73:  ("Bow-wow", 1.2),
    74:  ("Growling", 1.5),
    83:  ("Clip-clop", 2.0),
    86:  ("Moo", 1.0),
    87:  ("Cowbell", 1.5),
    102: ("Honk", 1.2),
    104: ("Roaring cats (lions, tigers)", 1.8),
    105: ("Roar", 1.8),
    115: ("Hoot", 1.0),
    125: ("Buzz", 1.0),

    # ── Music (general) ────────────────────────────────
    132: ("Music", 1.0),
    133: ("Musical instrument", 1.0),

    # ── String instruments ────────────────────────────
    134: ("Plucked string instrument", 1.0),
    135: ("Guitar", 1.0),
    136: ("Electric guitar", 1.2),
    137: ("Bass guitar", 1.5),
    138: ("Acoustic guitar", 1.0),
    140: ("Tapping (guitar technique)", 1.2),
    141: ("Strum", 1.0),
    142: ("Banjo", 1.0),
    143: ("Sitar", 1.0),
    144: ("Mandolin", 1.0),
    146: ("Ukulele", 1.0),

    # ── Keyboard instruments ──────────────────────────
    147: ("Keyboard (musical)", 1.0),
    148: ("Piano", 1.0),
    149: ("Electric piano", 1.0),
    150: ("Organ", 1.0),
    153: ("Synthesizer", 1.2),

    # ── Drums (suppressed — self-punch through audio) ─
    156: ("Percussion", 0.3),
    157: ("Drum kit", 0.3),
    158: ("Drum machine", 0.3),
    159: ("Drum", 0.3),
    160: ("Snare drum", 0.3),
    161: ("Rimshot", 0.3),
    162: ("Drum roll", 0.3),
    163: ("Bass drum", 0.3),
    164: ("Timpani", 0.5),
    165: ("Tabla", 0.5),
    166: ("Cymbal", 0.3),
    167: ("Hi-hat", 0.3),
    168: ("Wood block", 0.8),
    169: ("Tambourine", 0.5),
    170: ("Rattle (instrument)", 0.6),
    172: ("Gong", 1.5),

    # ── Brass / strings / wind ────────────────────────
    180: ("Brass instrument", 1.0),
    182: ("Trumpet", 1.0),
    183: ("Trombone", 1.0),
    184: ("Bowed string instrument", 1.0),
    186: ("Violin, fiddle", 1.0),
    188: ("Cello", 1.0),
    189: ("Double bass", 1.5),
    191: ("Flute", 1.0),
    192: ("Saxophone", 1.0),
    193: ("Clarinet", 1.0),
    194: ("Harp", 1.0),

    # ── Bells / chimes ────────────────────────────────
    195: ("Bell", 1.5),
    196: ("Church bell", 1.5),
    197: ("Jingle bell", 1.2),
    200: ("Chime", 1.2),
    201: ("Wind chime", 1.0),

    # ── Music genres ──────────────────────────────────
    211: ("Pop music", 1.0),
    212: ("Hip hop music", 1.2),
    213: ("Beatboxing", 1.5),
    214: ("Rock music", 1.0),
    215: ("Heavy metal", 1.5),
    216: ("Punk rock", 1.5),
    219: ("Rock and roll", 1.2),
    227: ("Funk", 1.2),
    230: ("Jazz", 1.0),
    231: ("Disco", 1.0),
    234: ("Electronic music", 1.2),
    235: ("House music", 1.2),
    236: ("Techno", 1.2),
    237: ("Dubstep", 1.5),
    238: ("Drum and bass", 1.5),
    240: ("Electronic dance music", 1.2),
    269: ("Dance music", 1.2),
    274: ("Exciting music", 1.3),

    # ── Weather / nature ──────────────────────────────
    280: ("Thunderstorm", 3.0),
    281: ("Thunder", 3.0),
    282: ("Water", 1.5),
    286: ("Stream", 1.0),
    287: ("Waterfall", 1.5),
    288: ("Ocean", 1.2),
    289: ("Waves, surf", 1.5),
    291: ("Gurgling", 1.0),
    292: ("Fire", 1.5),
    293: ("Crackle", 1.8),

    # ── Vehicles ──────────────────────────────────────
    294: ("Vehicle", 1.5),
    298: ("Motorboat, speedboat", 1.5),
    299: ("Ship", 1.5),
    300: ("Motor vehicle (road)", 1.5),
    301: ("Car", 1.5),
    302: ("Vehicle horn, car horn, honking", 1.8),
    304: ("Car alarm", 2.0),
    306: ("Skidding", 2.0),
    307: ("Tire squeal", 2.0),
    308: ("Car passing by", 1.2),
    309: ("Race car, auto racing", 1.8),
    310: ("Truck", 1.5),
    311: ("Air brake", 1.5),
    315: ("Bus", 1.2),
    316: ("Emergency vehicle", 1.8),
    317: ("Police car (siren)", 2.0),
    318: ("Ambulance (siren)", 2.0),
    320: ("Motorcycle", 1.5),
    322: ("Rail transport", 1.5),
    323: ("Train", 1.5),
    324: ("Train whistle", 1.8),
    325: ("Train horn", 1.8),
    327: ("Train wheels squealing", 2.0),
    328: ("Subway, metro, underground", 1.5),
    329: ("Aircraft", 1.5),
    330: ("Aircraft engine", 1.8),
    331: ("Jet engine", 2.0),
    332: ("Propeller, airscrew", 1.5),
    333: ("Helicopter", 1.8),

    # ── Engines / power ───────────────────────────────
    337: ("Engine", 1.5),
    339: ("Dental drill, dentist's drill", 1.5),
    340: ("Lawn mower", 1.5),
    341: ("Chainsaw", 1.8),
    342: ("Medium engine (mid frequency)", 1.5),
    343: ("Heavy engine (low frequency)", 1.8),
    344: ("Engine knocking", 1.8),
    345: ("Engine starting", 1.8),
    347: ("Accelerating, revving, vroom", 1.8),

    # ── Doors / surface impacts ───────────────────────
    348: ("Door", 2.0),
    349: ("Doorbell", 1.5),
    350: ("Ding-dong", 1.5),
    351: ("Sliding door", 1.2),
    352: ("Slam", 2.5),
    353: ("Knock", 2.0),
    354: ("Tap", 1.8),
    355: ("Squeak", 1.0),
    356: ("Cupboard open or close", 1.2),
    357: ("Drawer open or close", 1.2),

    # ── Kitchen / cooking ─────────────────────────────
    358: ("Dishes, pots, and pans", 1.5),
    359: ("Cutlery, silverware", 1.5),
    360: ("Chopping (food)", 1.8),
    361: ("Frying (food)", 1.0),

    # ── Alarms / signalling ───────────────────────────
    382: ("Alarm", 2.0),
    383: ("Telephone", 1.5),
    384: ("Telephone bell ringing", 1.8),
    385: ("Ringtone", 1.5),
    389: ("Alarm clock", 2.0),
    390: ("Siren", 2.0),
    391: ("Civil defense siren", 2.0),
    392: ("Buzzer", 1.5),
    394: ("Fire alarm", 2.0),
    396: ("Whistle", 1.5),
    397: ("Steam whistle", 1.5),

    # ── Mechanism / clocks ────────────────────────────
    398: ("Mechanisms", 1.0),
    399: ("Ratchet, pawl", 1.5),
    400: ("Clock", 2.0),
    401: ("Tick", 2.0),
    402: ("Tick-tock", 2.0),
    403: ("Gears", 1.2),
    405: ("Sewing machine", 1.0),
    406: ("Mechanical fan", 0.8),

    # ── Tools ─────────────────────────────────────────
    412: ("Tools", 1.5),
    413: ("Hammer", 2.5),
    414: ("Jackhammer", 2.0),
    415: ("Sawing", 1.5),
    416: ("Filing (rasp)", 1.2),
    417: ("Sanding", 1.0),
    418: ("Power tool", 1.5),
    419: ("Drill", 1.5),

    # ── Weapons / explosives ──────────────────────────
    420: ("Explosion", 3.0),
    421: ("Gunshot, gunfire", 3.0),
    422: ("Machine gun", 3.0),
    423: ("Fusillade", 3.0),
    424: ("Artillery fire", 3.0),
    425: ("Cap gun", 2.0),
    426: ("Fireworks", 2.5),
    427: ("Firecracker", 2.5),
    428: ("Burst, pop", 2.5),
    429: ("Eruption", 3.0),
    430: ("Boom", 3.0),

    # ── Wood / glass / breaking ───────────────────────
    431: ("Wood", 1.8),
    432: ("Chop", 2.0),
    433: ("Splinter", 2.0),
    434: ("Crack", 2.0),
    435: ("Glass", 2.0),
    436: ("Chink, clink", 1.8),
    437: ("Shatter", 2.8),

    # ── Liquids ──────────────────────────────────────
    438: ("Liquid", 1.2),
    439: ("Splash, splatter", 2.0),
    440: ("Slosh", 1.5),
    441: ("Squish", 1.2),
    442: ("Drip", 1.5),
    443: ("Pour", 1.5),
    444: ("Trickle, dribble", 1.0),
    445: ("Gush", 1.5),
    447: ("Spray", 1.0),
    450: ("Boiling", 1.0),

    # ── Whoosh / arrow / thump ────────────────────────
    451: ("Sonar", 1.5),
    452: ("Arrow", 2.0),
    453: ("Whoosh, swoosh, swish", 1.5),
    454: ("Thump, thud", 2.5),
    455: ("Thunk", 2.0),

    # ── Bounces / impacts ─────────────────────────────
    459: ("Basketball bounce", 1.8),
    460: ("Bang", 3.0),
    461: ("Slap, smack", 2.0),
    462: ("Whack, thwack", 2.5),
    463: ("Smash, crash", 3.0),
    464: ("Breaking", 2.5),
    465: ("Bouncing", 1.5),
    466: ("Whip", 2.5),
    467: ("Flap", 1.0),
    468: ("Scratch", 1.0),
    472: ("Crushing", 1.8),
    473: ("Crumpling, crinkling", 1.0),
    474: ("Tearing", 1.5),

    # ── Misc bells / dings / clatter ──────────────────
    475: ("Beep, bleep", 1.5),
    476: ("Ping", 1.5),
    477: ("Ding", 1.5),
    478: ("Clang", 2.0),
    479: ("Squeal", 1.5),
    480: ("Creak", 1.0),
    483: ("Clatter", 1.5),
    484: ("Sizzle", 1.0),
    485: ("Clicking", 1.2),
    486: ("Clickety-clack", 1.5),
    487: ("Rumble", 2.0),
    488: ("Plop", 1.2),
    489: ("Jingle, tinkle", 1.0),
    491: ("Zing", 1.5),
    492: ("Boing", 1.2),
    493: ("Crunch", 1.8),
}

# ── Speech classes (ground-truthed) ──────────────────
# YAMNet's 521-class set does NOT include gender-tagged speech labels
# (the old "Male speech / Female speech" entries did not exist in
# the model and were dead lookups).  Real speech-bearing indices are
# 0-5, 10-12 plus shout/yell/cry — but shouts/cries are kept OUT of
# this set because they're vocally expressive in action scenes and
# should drive haptics, not be muted.
SPEECH_CLASSES: dict[int, str] = {
    0:  "Speech",
    1:  "Child speech, kid speaking",
    2:  "Conversation",
    3:  "Narration, monologue",
    4:  "Babbling",
    5:  "Speech synthesizer",
    12: "Whispering",
}

# Non-haptic noise / ambience — used purely for display-name lookup
# when assigning the dominant_class label.  Membership here does NOT
# affect scoring; it only stops the fallback to generic class names.
NON_WORTHY_CLASSES: dict[int, str] = {
    **SPEECH_CLASSES,
    13: "Laughter",
    14: "Baby laughter",
    15: "Giggle",
    16: "Snicker",
    17: "Belly laugh",
    18: "Chuckle, chortle",
    19: "Crying, sobbing",
    20: "Baby cry, infant cry",
    23: "Sigh",
    36: "Breathing",
    37: "Wheeze",
    38: "Snoring",
    42: "Cough",
    44: "Sneeze",
    63: "Chatter",
    64: "Crowd",
    65: "Hubbub, speech noise, speech babble",
    277: "Wind",
    278: "Rustling leaves",
    279: "Wind noise (microphone)",
    283: "Rain",
    284: "Raindrop",
    285: "Rain on surface",
    494: "Silence",
    507: "Noise",
    508: "Environmental noise",
    509: "Static",
    510: "Mains hum",
    514: "White noise",
    515: "Pink noise",
}

_yamnet_model: Any | None = None
_yamnet_classes: list[str] | None = None
_whisper_model: Any | None = None


def _load_yamnet():
    """Load YAMNet from TensorFlow Hub (lazy, ~18 MB)."""
    global _yamnet_model, _yamnet_classes

    if _yamnet_model is not None:
        return _yamnet_model

    try:
        import tensorflow_hub as hub
        import tensorflow as tf

        logger.info("Loading YAMNet model from TF Hub…")
        _yamnet_model = hub.load(settings.YAMNET_MODEL_HANDLE)

        class_map_path = _yamnet_model.class_map_path().numpy()
        class_map_csv = tf.io.read_file(class_map_path).numpy().decode("utf-8")
        reader = csv.reader(io.StringIO(class_map_csv))
        next(reader)
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


def classify_audio(wav_path: str) -> AIClassification:
    """Run YAMNet sound event detection + Whisper speech detection on a 16 kHz WAV."""
    sr = settings.CLASSIFIER_SAMPLE_RATE
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = float(len(y) / sr)
    logger.info("AI classification: %.2fs audio loaded at %d Hz", duration, sr)

    yamnet = _load_yamnet()
    if yamnet is not None:
        haptic_scores, speech_scores, drum_scores, dominant_classes = _run_yamnet(
            yamnet, y, sr, duration,
        )
    else:
        logger.warning("No YAMNet model – using neutral AI scores")
        n_frames = max(1, int(duration / 0.48))
        haptic_scores = [0.0] * n_frames
        speech_scores = [0.0] * n_frames
        drum_scores = [0.0] * n_frames
        dominant_classes = ["unknown"] * n_frames

    speech_segments = _run_whisper(wav_path, duration)

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
        drum_scores=drum_scores,
        dominant_classes=dominant_classes,
        speech_segments=speech_segments,
    )


def _run_yamnet(
    model: Any,
    waveform: np.ndarray,
    sr: int,
    duration: float,
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Run YAMNet on the full waveform (0.96 s window, 0.48 s hop)."""
    import tensorflow as tf

    waveform_f32 = waveform.astype(np.float32)
    peak = np.max(np.abs(waveform_f32))
    if peak > 1.0:
        waveform_f32 = waveform_f32 / peak

    scores, embeddings, log_mel = model(waveform_f32)
    scores_np = scores.numpy()

    haptic_scores: list[float] = []
    speech_scores: list[float] = []
    drum_scores: list[float] = []
    dominant_classes: list[str] = []

    for frame_scores in scores_np:
        h = _compute_haptic_score(frame_scores)
        s = _compute_speech_score(frame_scores)
        d = _compute_drum_score(frame_scores)
        top = _get_dominant_class(frame_scores)

        haptic_scores.append(round(float(h), 4))
        speech_scores.append(round(float(s), 4))
        drum_scores.append(round(float(d), 4))
        dominant_classes.append(top)

    haptic_scores = _ema_smooth(haptic_scores, alpha=0.3)
    speech_scores = _ema_smooth(speech_scores, alpha=0.3)
    drum_scores = _ema_smooth(drum_scores, alpha=0.3)
    dominant_classes = _median_filter_labels(dominant_classes, window=3)

    return haptic_scores, speech_scores, drum_scores, dominant_classes


def _run_whisper(wav_path: str, duration: float) -> list[SpeechSegment]:
    """Detect speech segments using faster-whisper."""
    whisper = _load_whisper()
    if whisper is None:
        return []

    try:
        segments, info = whisper.transcribe(
            wav_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
        )

        speech_segments: list[SpeechSegment] = []
        for seg in segments:
            confidence = 1.0 - getattr(seg, "no_speech_prob", 0.0)
            if confidence < settings.WHISPER_MIN_CONFIDENCE:
                continue

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


def _compute_haptic_score(probs: np.ndarray) -> float:
    """Weighted sum of haptic-worthy class probabilities (top-5 classes)."""
    worthy_indices = [i for i in HAPTIC_WORTHY_CLASSES.keys() if i < len(probs)]
    if not worthy_indices:
        return 0.0
    weighted_probs = np.array([
        probs[i] * HAPTIC_WORTHY_CLASSES[i][1] for i in worthy_indices
    ])
    top_k = min(5, len(weighted_probs))
    return float(np.clip(np.sort(weighted_probs)[-top_k:].sum(), 0.0, 1.0))


# Verified drum/percussion indices (was previously {159..166} which
# wrongly included 164 Timpani / 165 Tabla as "drum kit" members).
# Real drum-kit elements per YAMNet's class CSV:
#   156 Percussion, 157 Drum kit, 158 Drum machine,
#   159 Drum, 160 Snare drum, 161 Rimshot, 162 Drum roll,
#   163 Bass drum, 166 Cymbal, 167 Hi-hat
_DRUM_INDICES: set[int] = {156, 157, 158, 159, 160, 161, 162, 163, 166, 167}


def _compute_drum_score(probs: np.ndarray) -> float:
    drum_indices = [i for i in _DRUM_INDICES if i < len(probs)]
    if not drum_indices:
        return 0.0
    return float(np.max(probs[drum_indices]))


def _compute_speech_score(probs: np.ndarray) -> float:
    speech_indices = [i for i in SPEECH_CLASSES.keys() if i < len(probs)]
    if not speech_indices:
        return 0.0
    return float(np.max(probs[speech_indices]))


def _get_dominant_class(probs: np.ndarray) -> str:
    global _yamnet_classes
    top_idx = int(np.argmax(probs))

    if top_idx in HAPTIC_WORTHY_CLASSES:
        return HAPTIC_WORTHY_CLASSES[top_idx][0]
    if top_idx in NON_WORTHY_CLASSES:
        return NON_WORTHY_CLASSES[top_idx]
    if _yamnet_classes and top_idx < len(_yamnet_classes):
        return _yamnet_classes[top_idx]
    return f"class_{top_idx}"


def _ema_smooth(values: list[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average smoothing (alpha=0.3 preserves transient spikes)."""
    if len(values) <= 1:
        return values

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(round(alpha * v + (1.0 - alpha) * smoothed[-1], 4))
    return smoothed


def _median_filter_labels(labels: list[str], window: int = 3) -> list[str]:
    """Majority-vote filter to eliminate single-frame class flips."""
    if len(labels) <= window:
        return labels

    half = window // 2
    result: list[str] = []

    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        neighborhood = labels[lo:hi]

        counts: dict[str, int] = {}
        for lbl in neighborhood:
            counts[lbl] = counts.get(lbl, 0) + 1

        max_count = max(counts.values())
        if counts.get(labels[i], 0) == max_count:
            result.append(labels[i])
        else:
            result.append(max(counts, key=counts.get))  # type: ignore[arg-type]

    return result
