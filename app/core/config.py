"""Application configuration using pydantic-settings."""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App-wide settings loaded from environment / .env file."""

    # ── API ──────────────────────────────────────────────
    APP_NAME: str = "Haptic Video Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── File Storage ─────────────────────────────────────
    UPLOAD_DIR: str = "/tmp/haptic-jobs/uploads"
    RESULTS_DIR: str = "/tmp/haptic-jobs/results"
    MAX_UPLOAD_SIZE_MB: int = 500  # max video file size
    ALLOWED_EXTENSIONS: list[str] = [
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv",
    ]

    # ── Celery / Redis ───────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ── Audio Analysis ───────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 22050        # librosa default
    CLASSIFIER_SAMPLE_RATE: int = 16000   # YAMNet & Whisper expect 16 kHz
    HOP_LENGTH: int = 512                 # ~23 ms at 22050 Hz
    FRAME_DURATION_MS: float = 23.2       # hop_length / sr * 1000

    # ── Haptic Generation ────────────────────────────────
    DEFAULT_SENSITIVITY: float = 0.5      # 0-1, controls threshold
    MIN_TRANSIENT_INTERVAL_MS: float = 50 # debounce between taps
    SILENCE_RMS_THRESHOLD: float = 0.003  # below = silence (raw RMS)
    SPEECH_SUPPRESSION_FACTOR: float = 0.05  # near-zero for dialogue
    DRUM_SUPPRESSION_FACTOR: float = 0.15   # residual when drums dominant (0=mute, 1=pass)
    HAPTIC_OVERRIDE_THRESHOLD: float = 0.40    # ai_haptic must exceed this to override speech gate
    HAPTIC_OVERRIDE_PASS_THROUGH: float = 0.50 # max speech_gate value during override

    # ── Novelty Gate Floors ──────────────────────────────
    NOVELTY_FLOOR_PER_BAND: float = 0.03        # min pass-through for band gates
    NOVELTY_FLOOR_PERCUSSIVE: float = 0.05      # min pass-through for percussive gate
    NOVELTY_FLOOR_HARMONIC: float = 0.03        # min pass-through for harmonic gate
    NOVELTY_FLOOR_GLOBAL_AMBIENT: float = 0.02  # min pass-through for global ambient gate

    # ── AI Activity Gate ─────────────────────────────────
    AI_ACTIVITY_GATE_FLOOR: float = 0.08        # min pass-through when AI detects no haptic content
    AI_ACTIVITY_GATE_THRESHOLD: float = 0.05    # haptic_score below this = "inactive"

    # ── Post-Boost Rest Gate ─────────────────────────────
    POST_BOOST_REST_THRESHOLD: float = 0.18     # non-dynamic frames below this after boost → zero

    MAX_AHAP_EVENTS_PER_CHUNK: int = 128  # Apple limit per pattern
    AHAP_CHUNK_DURATION_S: float = 30.0   # Apple limit per pattern

    # ── AHAP Segmentation ────────────────────────────────
    # Short HapticContinuous segments prevent Apple Core Haptics
    # from auto-reducing intensity on long carriers.
    HAPTIC_SEGMENT_DURATION_S: float = 2.0       # carrier segment length
    HAPTIC_SEGMENT_OVERLAP_S: float = 0.05       # 50 ms overlap between segments
    HAPTIC_CURVE_VARIANCE_THRESHOLD: float = 0.03  # std-dev below → static params
    HAPTIC_REST_INTENSITY_THRESHOLD: float = 0.02   # max intensity below → skip segment

    # ── YAMNet Model ─────────────────────────────────────
    YAMNET_MODEL_HANDLE: str = "https://tfhub.dev/google/yamnet/1"

    # ── Whisper Model ────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "tiny"      # tiny | base | small
    WHISPER_COMPUTE_TYPE: str = "int8"    # int8 for CPU efficiency

    # ── Frequency Bands (Hz) ─────────────────────────────
    FREQ_BANDS: dict[str, tuple[int, int]] = {
        "sub_bass":   (20, 60),
        "bass":       (60, 250),
        "low_mid":    (250, 500),
        "mid":        (500, 2000),
        "presence":   (2000, 4000),
        "brilliance": (4000, 8000),
    }

    # ── AWS (optional, for S3 storage) ───────────────────
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    settings = Settings()
    # Ensure directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    return settings
