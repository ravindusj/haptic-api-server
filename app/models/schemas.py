"""Pydantic schemas for API request/response models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────


class AnalysisStyle(str, enum.Enum):
    """Controls how the haptic pattern is optimised."""

    AUTO = "auto"           # auto-detect music vs cinematic
    CINEMATIC = "cinematic" # prioritise impacts, explosions, suppress dialogue
    MUSIC = "music"         # prioritise beat, bass, rhythm


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    EXTRACTING_AUDIO = "extracting_audio"
    ANALYZING_VIDEO = "analyzing_video"
    ANALYZING_DSP = "analyzing_dsp"
    CLASSIFYING_AI = "classifying_ai"
    SCORING = "scoring"
    GENERATING_AHAP = "generating_ahap"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Request ──────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Query-string parameters for the /analyze endpoint."""

    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall sensitivity: 0 = fewer haptics, 1 = more haptics.",
    )
    style: AnalysisStyle = Field(
        default=AnalysisStyle.AUTO,
        description="Analysis style to optimise for.",
    )
    bass_boost: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Multiplier for low-frequency energy. >1 = more bass rumble.",
    )


# ── Response ─────────────────────────────────────────────


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job queued for processing."


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Approximate progress percentage.",
    )
    created_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    file_name: str | None = None
    duration_seconds: float | None = None


class AHAPDownloadInfo(BaseModel):
    job_id: str
    download_url: str
    file_size_bytes: int
    duration_seconds: float
    total_events: int
    total_chunks: int


# ── Internal data models (used between services) ────────


class DSPFeatures(BaseModel):
    """Frame-level DSP features extracted by librosa."""

    sample_rate: int
    hop_length: int
    total_frames: int
    duration_seconds: float

    # ── HPSS-separated signals ───────────────────────────
    harmonic_rms: list[float]          # RMS of harmonic component
    percussive_rms: list[float]        # RMS of percussive component
    percussive_onset: list[float]      # onset strength of percussive only

    # ── Full-mix features ────────────────────────────────
    rms_energy: list[float]            # overall RMS (normalised)
    spectral_centroid: list[float]     # normalised 0-1
    spectral_flux: list[float]

    # ── Multi-band frequency energies (normalised) ───────
    sub_bass_energy: list[float]       # 20-60 Hz
    bass_energy: list[float]           # 60-250 Hz
    low_mid_energy: list[float]        # 250-500 Hz
    mid_energy: list[float]            # 500-2000 Hz
    presence_energy: list[float]       # 2000-4000 Hz
    brilliance_energy: list[float]     # 4000-8000 Hz

    # Raw (pre-normalisation) RMS for absolute-loudness awareness
    raw_rms_mean: float = 0.0
    raw_rms_peak: float = 0.0
    raw_rms_array: list[float] = []

    # Beat positions (in seconds)
    beat_times: list[float]
    beat_strengths: list[float]

    class Config:
        arbitrary_types_allowed = True


class SpeechSegment(BaseModel):
    """A contiguous speech segment detected by Whisper."""

    start: float    # seconds
    end: float      # seconds
    confidence: float = 1.0  # 0-1


class AIClassification(BaseModel):
    """Frame-level AI sound event detection results."""

    frame_duration_s: float            # duration of each AI frame
    total_frames: int

    haptic_scores: list[float]         # 0-1, how haptic-worthy
    speech_scores: list[float]         # 0-1, speech probability
    drum_scores: list[float] = []     # 0-1, drum/percussion probability
    dominant_classes: list[str]        # top class label per frame

    # Whisper-detected speech segments (precise timestamps)
    speech_segments: list[SpeechSegment] = []


class HapticEvent(BaseModel):
    """A single haptic event on the timeline."""

    time: float                        # seconds from start
    event_type: str                    # "transient" | "continuous"
    duration: float = 0.0             # only for continuous events
    intensity: float = 0.0            # 0-1
    sharpness: float = 0.5            # 0-1


class SceneChange(BaseModel):
    """A detected scene cut with timestamp and magnitude."""

    time: float                        # seconds from start
    magnitude: float                   # d/threshold ratio (1.0=barely, 5.0+=hard cut)


class VideoFeatures(BaseModel):
    """Frame-level video motion and action recognition features.

    Produced by the video_analyzer service.  Motion intensity
    comes from optical flow; action scores from MoViNet-A0.
    """

    fps: float                         # analysis sample rate
    total_frames: int
    duration_seconds: float

    # Tier 1: Optical flow
    motion_intensity: list[float]      # 0-1 per analysis frame
    scene_changes: list[SceneChange]   # scene cuts with time + magnitude

    # Tier 1.5: Per-frame visual impact signals
    visual_flash: list[float] = []     # 0-1, brightness-spike vs rolling baseline
    camera_shake: list[float] = []     # 0-1, normalised global-translation magnitude

    # Tier 2: MoViNet action recognition
    action_scores: dict[str, list[float]]  # category → per-window scores
    dominant_actions: list[str]        # dominant category per window
    action_window_duration_s: float    # seconds per classification window


class HapticTimeline(BaseModel):
    """Full timeline of haptic events for a video.

    The ``intensity_envelope`` and ``sharpness_envelope`` carry a
    continuous per-frame intensity/sharpness signal (sampled at
    ``envelope_fps``) that the AHAP generator converts into
    ParameterCurve control points.  This is what enables Sony-DVS-style
    *continuous* sound-to-vibration mapping rather than sparse events.
    """

    duration_seconds: float
    events: list[HapticEvent]
    metadata: dict[str, Any] = {}

    # Continuous envelope (added for DVS-style mapping)
    intensity_envelope: list[float] = []
    sharpness_envelope: list[float] = []
    envelope_fps: float = 20.0  # samples per second


class AHAPPattern(BaseModel):
    """One AHAP pattern chunk (≤30s, ≤128 events)."""

    version: float = 1.0
    pattern: list[dict[str, Any]]
    chunk_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


class AHAPFile(BaseModel):
    """Complete AHAP output (may contain multiple chunks)."""

    chunks: list[AHAPPattern]
    total_duration: float
    total_events: int
    metadata: dict[str, Any] = {}
