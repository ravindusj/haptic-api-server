"""AHAP (Apple Haptic and Audio Pattern) file generator.

Converts a HapticTimeline into a valid AHAP JSON file that can be
played by Apple's Core Haptics framework on iPhones (iPhone 8+).

Architecture
------------
The generator uses Apple's ParameterCurve mechanism for Sony-DVS-style
continuous sound-to-vibration mapping:

  1. **One HapticContinuous event** per chunk spans the full chunk
     duration at intensity=1.0 (the "carrier signal").
  2. **A HapticIntensityControl ParameterCurve** with one control point
     every ~50 ms modulates the carrier multiplicatively, so the
     vibration intensity tracks the audio energy envelope frame-by-frame.
  3. **A HapticSharpnessControl ParameterCurve** tracks tonal content
     (bass → dull, treble → sharp).
  4. **HapticTransient events** are overlaid at onset / beat positions
     for punchy impact accents on top of the continuous rumble.

AHAP spec constraints:
  - Version 1.0
  - Max ~128 Event entries per CHHapticPattern (ParameterCurves are
    separate and don't count toward this limit)
  - Max ~30 seconds per pattern
  - EventTypes: HapticTransient, HapticContinuous
  - Parameters: HapticIntensity (0-1), HapticSharpness (0-1)
  - ParameterCurves: HapticIntensityControl, HapticSharpnessControl

For long videos we chunk into multiple sequential patterns and embed
chunk metadata so the iOS player can chain them.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.models.schemas import (
    AHAPFile,
    AHAPPattern,
    HapticEvent,
    HapticTimeline,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def generate_ahap(
    timeline: HapticTimeline,
    job_id: str,
) -> AHAPFile:
    """
    Convert a HapticTimeline into a full AHAP file.

    Parameters
    ----------
    timeline : HapticTimeline
        Haptic events + continuous envelope.
    job_id : str
        Used for file naming.

    Returns
    -------
    AHAPFile
        Contains one or more AHAP pattern chunks.
    """
    events = sorted(timeline.events, key=lambda e: e.time)
    duration = timeline.duration_seconds
    chunk_dur = settings.AHAP_CHUNK_DURATION_S

    intensity_env = timeline.intensity_envelope
    sharpness_env = timeline.sharpness_envelope
    env_fps = timeline.envelope_fps or 20.0

    # ── Split into time-based chunks ─────────────────────
    chunks: list[AHAPPattern] = []
    n_chunks = max(1, int(duration / chunk_dur) + (1 if duration % chunk_dur > 0.01 else 0))

    for ci in range(n_chunks):
        chunk_start = ci * chunk_dur
        chunk_end = min((ci + 1) * chunk_dur, duration)
        chunk_actual_dur = chunk_end - chunk_start

        if chunk_actual_dur < 0.01:
            continue

        # ── Gather transient events in this chunk window ─
        chunk_events = [
            e for e in events
            if chunk_start <= e.time < chunk_end
        ]

        # ── Slice envelope for this chunk ────────────────
        env_start_idx = int(chunk_start * env_fps)
        env_end_idx = int(chunk_end * env_fps)
        chunk_intensity = intensity_env[env_start_idx:env_end_idx] if intensity_env else []
        chunk_sharpness = sharpness_env[env_start_idx:env_end_idx] if sharpness_env else []

        pattern = _build_pattern(
            transient_events=chunk_events,
            intensity_envelope=chunk_intensity,
            sharpness_envelope=chunk_sharpness,
            envelope_fps=env_fps,
            chunk_start=chunk_start,
            chunk_duration=chunk_actual_dur,
            chunk_index=ci,
        )
        chunks.append(pattern)

    # If no chunks at all, create a minimal silent pattern
    if not chunks:
        chunks.append(
            AHAPPattern(
                version=1.0,
                pattern=[],
                chunk_index=0,
                start_time=0.0,
                end_time=duration,
            )
        )

    total_events = sum(len(c.pattern) for c in chunks)

    ahap = AHAPFile(
        chunks=chunks,
        total_duration=duration,
        total_events=total_events,
        metadata={
            **timeline.metadata,
            "total_chunks": len(chunks),
        },
    )

    logger.info(
        "AHAP generated: %d chunks, %d events, %.1fs duration",
        len(chunks),
        total_events,
        duration,
    )

    return ahap


def save_ahap(ahap: AHAPFile, job_id: str) -> str:
    """
    Save AHAP to disk. Returns the file path.

    If there's only one chunk, save as a single `.ahap` file.
    If multiple chunks, save a combined JSON with chunk array + individual files.
    """
    results_dir = Path(settings.RESULTS_DIR) / job_id
    os.makedirs(results_dir, exist_ok=True)

    if len(ahap.chunks) == 1:
        # Single file
        ahap_data = _chunk_to_ahap_dict(ahap.chunks[0])
        file_path = str(results_dir / f"{job_id}.ahap")
        with open(file_path, "w") as f:
            json.dump(ahap_data, f, indent=2)
        logger.info("Saved single AHAP: %s", file_path)
        return file_path

    # Multiple chunks → save individual files + manifest
    manifest: dict[str, Any] = {
        "version": 1.0,
        "total_duration": ahap.total_duration,
        "total_events": ahap.total_events,
        "total_chunks": len(ahap.chunks),
        "chunks": [],
    }

    for chunk in ahap.chunks:
        chunk_file = f"{job_id}_chunk_{chunk.chunk_index:04d}.ahap"
        chunk_path = str(results_dir / chunk_file)
        chunk_data = _chunk_to_ahap_dict(chunk)
        with open(chunk_path, "w") as f:
            json.dump(chunk_data, f, indent=2)

        manifest["chunks"].append({
            "file": chunk_file,
            "index": chunk.chunk_index,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "event_count": len(chunk.pattern),
        })

    # Save manifest
    manifest_path = str(results_dir / f"{job_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save a combined single-file version
    combined_path = str(results_dir / f"{job_id}.ahap")
    combined = _build_combined_ahap(ahap)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(
        "Saved %d AHAP chunks + manifest + combined to %s",
        len(ahap.chunks),
        results_dir,
    )
    return combined_path


# ── Internal builders ────────────────────────────────────


def _build_pattern(
    transient_events: list[HapticEvent],
    intensity_envelope: list[float],
    sharpness_envelope: list[float],
    envelope_fps: float,
    chunk_start: float,
    chunk_duration: float,
    chunk_index: int,
) -> AHAPPattern:
    """Build one AHAP pattern chunk.

    Emits:
      1. One HapticContinuous event spanning the chunk at intensity=1.0
      2. A HapticIntensityControl ParameterCurve from the envelope
      3. A HapticSharpnessControl ParameterCurve from the envelope
      4. HapticTransient events for accent taps
    """
    pattern_entries: list[dict[str, Any]] = []

    # ── 1. Continuous carrier event ──────────────────────
    # Spans the full chunk duration; the ParameterCurve modulates
    # its intensity multiplicatively.
    pattern_entries.append({
        "Event": {
            "Time": 0.0,
            "EventType": "HapticContinuous",
            "EventDuration": round(chunk_duration, 4),
            "EventParameters": [
                {"ParameterID": "HapticIntensity", "ParameterValue": 1.0},
                {"ParameterID": "HapticSharpness", "ParameterValue": 0.5},
            ],
        }
    })

    # ── 2. Intensity ParameterCurve ──────────────────────
    if intensity_envelope:
        intensity_curve = _build_parameter_curve(
            envelope=intensity_envelope,
            fps=envelope_fps,
            parameter_id="HapticIntensityControl",
        )
        if intensity_curve:
            pattern_entries.append(intensity_curve)

    # ── 3. Sharpness ParameterCurve ──────────────────────
    if sharpness_envelope:
        sharpness_curve = _build_parameter_curve(
            envelope=sharpness_envelope,
            fps=envelope_fps,
            parameter_id="HapticSharpnessControl",
        )
        if sharpness_curve:
            pattern_entries.append(sharpness_curve)

    # ── 4. Transient accent taps ─────────────────────────
    for event in transient_events:
        rel_time = round(max(0.0, event.time - chunk_start), 4)
        pattern_entries.append(_make_transient(
            rel_time, event.intensity, event.sharpness,
        ))

    chunk_end = chunk_start + chunk_duration

    return AHAPPattern(
        version=1.0,
        pattern=pattern_entries,
        chunk_index=chunk_index,
        start_time=round(chunk_start, 4),
        end_time=round(chunk_end, 4),
    )


def _build_parameter_curve(
    envelope: list[float],
    fps: float,
    parameter_id: str,
) -> dict[str, Any] | None:
    """Build a ParameterCurve from an envelope array.

    Each element becomes a control point spaced at 1/fps seconds.
    The haptic engine linearly interpolates between points, giving
    smooth frame-accurate intensity/sharpness modulation.
    """
    if not envelope:
        return None

    interval = 1.0 / fps
    control_points: list[dict[str, float]] = []

    for i, value in enumerate(envelope):
        control_points.append({
            "Time": round(i * interval, 4),
            "ParameterValue": round(max(0.0, min(1.0, value)), 4),
        })

    if not control_points:
        return None

    return {
        "ParameterCurve": {
            "ParameterID": parameter_id,
            "Time": 0.0,
            "ParameterCurveControlPoints": control_points,
        }
    }


def _make_transient(
    time: float,
    intensity: float,
    sharpness: float,
) -> dict[str, Any]:
    """Create a HapticTransient event dict."""
    return {
        "Event": {
            "Time": time,
            "EventType": "HapticTransient",
            "EventParameters": [
                {
                    "ParameterID": "HapticIntensity",
                    "ParameterValue": round(intensity, 3),
                },
                {
                    "ParameterID": "HapticSharpness",
                    "ParameterValue": round(sharpness, 3),
                },
            ],
        }
    }


def _chunk_to_ahap_dict(chunk: AHAPPattern) -> dict[str, Any]:
    """Convert an AHAPPattern to the standard AHAP JSON structure."""
    return {
        "Version": chunk.version,
        "Metadata": {
            "ChunkIndex": chunk.chunk_index,
            "StartTime": chunk.start_time,
            "EndTime": chunk.end_time,
        },
        "Pattern": chunk.pattern,
    }


def _build_combined_ahap(ahap: AHAPFile) -> dict[str, Any]:
    """
    Build a single combined AHAP with all events (absolute times).
    Useful for short-to-medium content or players that handle large patterns.
    """
    all_events: list[dict[str, Any]] = []

    for chunk in ahap.chunks:
        for entry in chunk.pattern:
            # ParameterCurve entries: shift times to absolute
            if "ParameterCurve" in entry:
                curve = entry["ParameterCurve"].copy()
                curve["Time"] = round(curve["Time"] + chunk.start_time, 4)
                adjusted_points = []
                for pt in curve.get("ParameterCurveControlPoints", []):
                    adjusted_points.append({
                        "Time": round(pt["Time"] + chunk.start_time, 4),
                        "ParameterValue": pt["ParameterValue"],
                    })
                curve["ParameterCurveControlPoints"] = adjusted_points
                all_events.append({"ParameterCurve": curve})
            # Event entries: shift time to absolute
            elif "Event" in entry:
                evt = entry["Event"].copy()
                evt["Time"] = round(evt["Time"] + chunk.start_time, 4)
                all_events.append({"Event": evt})

    return {
        "Version": 1.0,
        "Metadata": {
            "TotalDuration": ahap.total_duration,
            "TotalEvents": ahap.total_events,
            "TotalChunks": len(ahap.chunks),
        },
        "Pattern": all_events,
    }
