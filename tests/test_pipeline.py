"""Tests for AHAP generator and haptic scorer services."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from app.models.schemas import (
    AIClassification,
    DSPFeatures,
    HapticEvent,
    HapticTimeline,
)
from app.services.ahap_generator import generate_ahap, _chunk_to_ahap_dict
from app.services.haptic_scorer import fuse_scores


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def sample_dsp_features() -> DSPFeatures:
    """Create minimal DSP features for testing."""
    n_frames = 200  # ~4.6 seconds at hop_length=512, sr=22050
    return DSPFeatures(
        sample_rate=22050,
        hop_length=512,
        total_frames=n_frames,
        duration_seconds=4.6,
        # Simulate: quiet→loud→quiet pattern
        rms_energy=[0.1] * 50 + [0.8] * 50 + [0.9] * 50 + [0.1] * 50,
        onset_strength=[0.1] * 50 + [0.7] * 50 + [0.9] * 50 + [0.1] * 50,
        low_freq_energy=[0.05] * 50 + [0.9] * 50 + [0.3] * 50 + [0.05] * 50,
        spectral_centroid=[0.5] * 200,
        spectral_flux=[0.1] * 50 + [0.6] * 50 + [0.8] * 50 + [0.1] * 50,
        beat_times=[1.0, 2.0, 3.0],
        beat_strengths=[0.5, 0.8, 0.6],
    )


@pytest.fixture
def sample_ai_classification() -> AIClassification:
    """Create minimal AI classification for testing."""
    return AIClassification(
        frame_duration_s=0.5,
        total_frames=10,  # 5 seconds at 0.5s hops
        # Speech in first 2 frames, haptic-worthy in middle, quiet at end
        haptic_scores=[0.1, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.1, 0.1, 0.1],
        speech_scores=[0.8, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.5],
        dominant_classes=[
            "Speech", "Speech", "Explosion", "Bass drum", "Drum",
            "Music", "Music", "Silence", "Speech", "Speech",
        ],
    )


@pytest.fixture
def sample_timeline() -> HapticTimeline:
    """Create a timeline with known events."""
    return HapticTimeline(
        duration_seconds=10.0,
        events=[
            HapticEvent(time=1.0, event_type="transient", intensity=0.8, sharpness=0.3),
            HapticEvent(time=2.5, event_type="continuous", duration=1.5, intensity=0.6, sharpness=0.15),
            HapticEvent(time=5.0, event_type="transient", intensity=0.9, sharpness=0.1),
            HapticEvent(time=7.0, event_type="transient", intensity=0.4, sharpness=0.7),
        ],
    )


# ── Score Fusion Tests ───────────────────────────────────


class TestHapticScorer:
    def test_fuse_produces_events(
        self,
        sample_dsp_features: DSPFeatures,
        sample_ai_classification: AIClassification,
    ):
        """Fusion should produce at least some haptic events."""
        timeline = fuse_scores(
            sample_dsp_features,
            sample_ai_classification,
            sensitivity=0.5,
        )
        assert timeline.duration_seconds == pytest.approx(4.6)
        assert len(timeline.events) > 0

    def test_high_sensitivity_more_events(
        self,
        sample_dsp_features: DSPFeatures,
        sample_ai_classification: AIClassification,
    ):
        """Higher sensitivity should produce more events."""
        low = fuse_scores(sample_dsp_features, sample_ai_classification, sensitivity=0.1)
        high = fuse_scores(sample_dsp_features, sample_ai_classification, sensitivity=0.9)
        assert len(high.events) >= len(low.events)

    def test_events_sorted_by_time(
        self,
        sample_dsp_features: DSPFeatures,
        sample_ai_classification: AIClassification,
    ):
        """Events should be sorted chronologically."""
        timeline = fuse_scores(sample_dsp_features, sample_ai_classification)
        times = [e.time for e in timeline.events]
        assert times == sorted(times)

    def test_intensities_in_range(
        self,
        sample_dsp_features: DSPFeatures,
        sample_ai_classification: AIClassification,
    ):
        """All intensities should be between 0 and 1."""
        timeline = fuse_scores(sample_dsp_features, sample_ai_classification)
        for event in timeline.events:
            assert 0.0 <= event.intensity <= 1.0
            assert 0.0 <= event.sharpness <= 1.0

    def test_speech_suppression_metadata(
        self,
        sample_dsp_features: DSPFeatures,
        sample_ai_classification: AIClassification,
    ):
        """Metadata should report speech suppression percentage."""
        timeline = fuse_scores(sample_dsp_features, sample_ai_classification)
        assert "speech_suppressed_pct" in timeline.metadata


# ── AHAP Generator Tests ────────────────────────────────


class TestAHAPGenerator:
    def test_generate_produces_chunks(self, sample_timeline: HapticTimeline):
        """Should produce at least one AHAP chunk."""
        ahap = generate_ahap(sample_timeline, job_id="test123")
        assert len(ahap.chunks) >= 1
        assert ahap.total_duration == 10.0
        assert ahap.total_events > 0

    def test_ahap_json_structure(self, sample_timeline: HapticTimeline):
        """AHAP JSON should have correct top-level keys."""
        ahap = generate_ahap(sample_timeline, job_id="test456")
        chunk = ahap.chunks[0]
        ahap_dict = _chunk_to_ahap_dict(chunk)

        assert "Version" in ahap_dict
        assert ahap_dict["Version"] == 1.0
        assert "Pattern" in ahap_dict
        assert isinstance(ahap_dict["Pattern"], list)

    def test_transient_events_format(self, sample_timeline: HapticTimeline):
        """HapticTransient events should have correct structure."""
        ahap = generate_ahap(sample_timeline, job_id="test789")
        chunk = ahap.chunks[0]
        ahap_dict = _chunk_to_ahap_dict(chunk)

        transients = [
            e for e in ahap_dict["Pattern"]
            if "Event" in e and e["Event"]["EventType"] == "HapticTransient"
        ]

        for t in transients:
            evt = t["Event"]
            assert "Time" in evt
            assert evt["Time"] >= 0.0
            params = {p["ParameterID"]: p["ParameterValue"] for p in evt["EventParameters"]}
            assert "HapticIntensity" in params
            assert "HapticSharpness" in params
            assert 0.0 <= params["HapticIntensity"] <= 1.0
            assert 0.0 <= params["HapticSharpness"] <= 1.0

    def test_continuous_events_have_duration(self, sample_timeline: HapticTimeline):
        """HapticContinuous events should have EventDuration."""
        ahap = generate_ahap(sample_timeline, job_id="test_cont")
        chunk = ahap.chunks[0]
        ahap_dict = _chunk_to_ahap_dict(chunk)

        continuous = [
            e for e in ahap_dict["Pattern"]
            if "Event" in e and e["Event"]["EventType"] == "HapticContinuous"
        ]

        for c in continuous:
            evt = c["Event"]
            assert "EventDuration" in evt
            assert evt["EventDuration"] > 0

    def test_empty_timeline(self):
        """Empty timeline should produce valid (empty) AHAP."""
        empty = HapticTimeline(duration_seconds=5.0, events=[])
        ahap = generate_ahap(empty, job_id="empty_test")
        assert len(ahap.chunks) == 1
        assert ahap.total_events == 0

    def test_ahap_serialises_to_json(self, sample_timeline: HapticTimeline):
        """Full AHAP output should be JSON-serialisable."""
        ahap = generate_ahap(sample_timeline, job_id="json_test")
        for chunk in ahap.chunks:
            ahap_dict = _chunk_to_ahap_dict(chunk)
            json_str = json.dumps(ahap_dict)
            parsed = json.loads(json_str)
            assert parsed["Version"] == 1.0


# ── API Schema Tests ─────────────────────────────────────


class TestSchemas:
    def test_haptic_event_defaults(self):
        """HapticEvent should have sensible defaults."""
        e = HapticEvent(time=1.0, event_type="transient")
        assert e.intensity == 0.0
        assert e.sharpness == 0.5
        assert e.duration == 0.0

    def test_dsp_features_roundtrip(self, sample_dsp_features: DSPFeatures):
        """DSPFeatures should survive JSON serialisation."""
        data = sample_dsp_features.model_dump()
        restored = DSPFeatures(**data)
        assert restored.total_frames == sample_dsp_features.total_frames
        assert len(restored.rms_energy) == len(sample_dsp_features.rms_energy)
