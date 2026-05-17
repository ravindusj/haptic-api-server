import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Haptic Video Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    UPLOAD_DIR: str = "/tmp/haptic-jobs/uploads"
    RESULTS_DIR: str = "/tmp/haptic-jobs/results"
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: list[str] = [
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv",
    ]

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    AUDIO_SAMPLE_RATE: int = 22050
    CLASSIFIER_SAMPLE_RATE: int = 16000
    HOP_LENGTH: int = 512
    FRAME_DURATION_MS: float = 23.2

    DEFAULT_SENSITIVITY: float = 0.5
    MIN_TRANSIENT_INTERVAL_MS: float = 25
    SILENCE_RMS_THRESHOLD: float = 0.003
    SPEECH_SUPPRESSION_FACTOR: float = 0.05
    DRUM_SUPPRESSION_FACTOR: float = 0.15
    HAPTIC_OVERRIDE_THRESHOLD: float = 0.30
    HAPTIC_OVERRIDE_PASS_THROUGH: float = 0.75

    WHISPER_MIN_CONFIDENCE: float = 0.30
    SPEECH_GUARD_PRE_MS: float = 100.0
    SPEECH_GUARD_POST_MS: float = 180.0
    SPEECH_GATE_SMOOTH_MS: float = 80.0
    SPEECH_HIGH_CONFIDENCE: float = 0.55

    NOVELTY_FLOOR_PER_BAND: float = 0.03
    NOVELTY_FLOOR_PERCUSSIVE: float = 0.05
    NOVELTY_FLOOR_HARMONIC: float = 0.03
    NOVELTY_FLOOR_GLOBAL_AMBIENT: float = 0.10

    AI_ACTIVITY_GATE_FLOOR: float = 0.15
    AI_ACTIVITY_GATE_THRESHOLD: float = 0.05

    POST_BOOST_REST_THRESHOLD: float = 0.18

    IMPACT_PRE_OFFSET_MS: float = 180.0
    IMPACT_PRE_INTENSITY_SCALE: float = 0.30
    IMPACT_PRE_SHARPNESS: float = 0.40
    IMPACT_RUMBLE_TAIL_S: float = 1.20
    IMPACT_RUMBLE_DECAY_S: float = 0.45
    IMPACT_SHARPNESS_PEAK: float = 0.95
    IMPACT_SHARPNESS_DECAY_S: float = 0.30

    VISUAL_FLASH_THRESHOLD: float = 0.30
    CAMERA_SHAKE_THRESHOLD: float = 0.30
    CROSS_MODAL_WINDOW_MS: float = 150.0
    CROSS_MODAL_BOOST: float = 1.15
    VISUAL_ONLY_FLASH_MIN: float = 0.50
    VISUAL_ONLY_SHAKE_MIN: float = 0.30

    MAX_AHAP_EVENTS_PER_CHUNK: int = 128
    AHAP_CHUNK_DURATION_S: float = 30.0

    HAPTIC_SEGMENT_DURATION_S: float = 2.0
    HAPTIC_SEGMENT_OVERLAP_S: float = 0.05
    HAPTIC_CURVE_VARIANCE_THRESHOLD: float = 0.03
    HAPTIC_REST_INTENSITY_THRESHOLD: float = 0.02

    # ── C2: MoViNet variant ──────────────────────────────
    # "a0" (172², ~3M params, fast) or "a2" (224², ~5M params,
    # ~5× more accurate per Google's reported numbers).  The
    # video_analyzer loads the variant declared here and falls back
    # to "a0" if the larger model fails to download or load.
    MOVINET_VARIANT: str = "a2"

    # ── A1: LFE-channel haptic weight ────────────────────
    # When the source is 5.1+ and a dedicated LFE channel is
    # extracted, the LFE RMS envelope is folded directly into the
    # haptic intensity signal with this weight.  Cinema LFE peaks
    # are intentionally near-clipping during explosions, so a weight
    # around 0.30-0.45 keeps it dominant on impact frames while
    # leaving headroom for the multi-band mix on quieter content.
    LFE_WEIGHT: float = 0.35

    # ── A4: Attack-time sharpness blending ───────────────
    # Weight of the per-frame attack envelope (rising onset rate)
    # applied additively to the heuristic sharpness signal.
    # 0.0 disables A4; 0.30 means a max-attack frame can raise
    # sharpness by up to 0.30 above its band-balance value.
    ATTACK_SHARPNESS_WEIGHT: float = 0.30

    # ── Transient "thud" duration (E2) ───────────────────
    # HapticTransient events are zero-duration impulses on iOS, which can
    # feel clinical for low-sharpness ("thud") taps.  When a transient's
    # sharpness is below TRANSIENT_THUD_SHARPNESS_THRESHOLD it is emitted
    # instead as a short HapticContinuous with EventDuration linearly
    # mapped from sharpness:
    #   sharpness = 0.0       → THUD_DURATION_MAX_S (deepest thud)
    #   sharpness = threshold → THUD_DURATION_MIN_S (borderline)
    # Sharper transients stay as zero-duration HapticTransient impulses.
    TRANSIENT_THUD_SHARPNESS_THRESHOLD: float = 0.40
    TRANSIENT_THUD_DURATION_MIN_S: float = 0.020
    TRANSIENT_THUD_DURATION_MAX_S: float = 0.060

    YAMNET_MODEL_HANDLE: str = "https://tfhub.dev/google/yamnet/1"

    WHISPER_MODEL_SIZE: str = "tiny"
    WHISPER_COMPUTE_TYPE: str = "int8"

    FREQ_BANDS: dict[str, tuple[int, int]] = {
        "sub_bass":   (20, 60),
        "bass":       (60, 250),
        "low_mid":    (250, 500),
        "mid":        (500, 2000),
        "presence":   (2000, 4000),
        "brilliance": (4000, 8000),
    }

    API_KEY: str | None = None

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
    settings = Settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    return settings
