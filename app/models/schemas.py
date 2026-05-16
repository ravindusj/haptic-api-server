from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisStyle(str, enum.Enum):
    AUTO = "auto"
    CINEMATIC = "cinematic"
    MUSIC = "music"


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


class AnalyzeRequest(BaseModel):
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    style: AnalysisStyle = Field(default=AnalysisStyle.AUTO)
    bass_boost: float = Field(default=1.0, ge=0.5, le=2.0)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job queued for processing."


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
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


class DSPFeatures(BaseModel):
    sample_rate: int
    hop_length: int
    total_frames: int
    duration_seconds: float

    harmonic_rms: list[float]
    percussive_rms: list[float]
    percussive_onset: list[float]

    rms_energy: list[float]
    spectral_centroid: list[float]
    spectral_flux: list[float]

    sub_bass_energy: list[float]
    bass_energy: list[float]
    low_mid_energy: list[float]
    mid_energy: list[float]
    presence_energy: list[float]
    brilliance_energy: list[float]

    raw_rms_mean: float = 0.0
    raw_rms_peak: float = 0.0
    raw_rms_array: list[float] = []

    beat_times: list[float]
    beat_strengths: list[float]

    class Config:
        arbitrary_types_allowed = True


class SpeechSegment(BaseModel):
    start: float
    end: float
    confidence: float = 1.0


class AIClassification(BaseModel):
    frame_duration_s: float
    total_frames: int

    haptic_scores: list[float]
    speech_scores: list[float]
    drum_scores: list[float] = []
    dominant_classes: list[str]

    speech_segments: list[SpeechSegment] = []


class HapticEvent(BaseModel):
    time: float
    event_type: str
    duration: float = 0.0
    intensity: float = 0.0
    sharpness: float = 0.5


class SceneChange(BaseModel):
    time: float
    magnitude: float


class VideoFeatures(BaseModel):
    fps: float
    total_frames: int
    duration_seconds: float

    motion_intensity: list[float]
    scene_changes: list[SceneChange]

    visual_flash: list[float] = []
    camera_shake: list[float] = []

    action_scores: dict[str, list[float]]
    dominant_actions: list[str]
    action_window_duration_s: float


class HapticTimeline(BaseModel):
    duration_seconds: float
    events: list[HapticEvent]
    metadata: dict[str, Any] = {}

    intensity_envelope: list[float] = []
    sharpness_envelope: list[float] = []
    envelope_fps: float = 20.0


class AHAPPattern(BaseModel):
    version: float = 1.0
    pattern: list[dict[str, Any]]
    chunk_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


class AHAPFile(BaseModel):
    chunks: list[AHAPPattern]
    total_duration: float
    total_events: int
    metadata: dict[str, Any] = {}
