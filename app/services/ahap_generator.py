"""AHAP (Apple Haptic and Audio Pattern) file generator.

Converts a HapticTimeline into a valid AHAP JSON file that can be
played by Apple's Core Haptics framework on iPhones (iPhone 8+).

Architecture — Segmented Carriers
----------------------------------
Apple Core Haptics auto-reduces intensity and may desync or kill
long HapticContinuous events (>3-5 s).  This generator therefore
splits every 30 s chunk into **many short 2 s HapticContinuous
carriers** ("segments"), each with its own ParameterCurves.

  1. Each 30 s chunk is subdivided into ~15 × 2 s **segments**.
  2. Segments where max intensity < 0.02 are **skipped** ("rest").
  3. If the envelope inside a segment has std-dev < 0.03 the carrier
     uses **static** HapticIntensity / HapticSharpness (no curve).
  4. Otherwise a short ParameterCurve modulates the segment carrier.
  5. Adjacent segments overlap by 50 ms for seamless transitions.
  6. HapticTransient events are **debounced** at 50 ms minimum interval.
  7. Per-chunk event limit (128) counts only Event entries, not
     ParameterCurve entries.

AHAP spec constraints:
  - Version 1.0
  - Max ~128 Event entries per CHHapticPattern
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
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.models.schemas import (
    AHAPFile,
    AHAPPattern,
    HapticEvent,
    HapticTimeline,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────


def generate_ahap(
    timeline: HapticTimeline,
    job_id: str,
) -> AHAPFile:
    """Convert a HapticTimeline into a segmented AHAP file.

    Parameters
    ----------
    timeline : HapticTimeline
        Haptic events + continuous envelope.
    job_id : str
        Used for file naming.

    Returns
    -------
    AHAPFile
        Contains one or more AHAP pattern chunks, each built from
        many short 2 s HapticContinuous segments.
    """
    events = sorted(timeline.events, key=lambda e: e.time)
    duration = timeline.duration_seconds
    chunk_dur = settings.AHAP_CHUNK_DURATION_S

    intensity_env = timeline.intensity_envelope
    sharpness_env = timeline.sharpness_envelope
    env_fps = timeline.envelope_fps or 20.0

    # ── Split into time-based chunks (≤30 s each) ────────
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
        env_start_idx = round(chunk_start * env_fps)
        env_end_idx = round(chunk_end * env_fps)
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

    # Count only real events (HapticContinuous + HapticTransient),
    # not ParameterCurve entries.
    total_events = _count_events(chunks)

    ahap = AHAPFile(
        chunks=chunks,
        total_duration=duration,
        total_events=total_events,
        metadata={
            **timeline.metadata,
            "total_chunks": len(chunks),
            "segment_duration_s": settings.HAPTIC_SEGMENT_DURATION_S,
        },
    )

    logger.info(
        "AHAP generated: %d chunks, %d events, %.1fs duration "
        "(segment=%.1fs, hybrid curves)",
        len(chunks),
        total_events,
        duration,
        settings.HAPTIC_SEGMENT_DURATION_S,
    )

    return ahap


def save_ahap(ahap: AHAPFile, job_id: str) -> str:
    """Save AHAP to disk.  Returns the file path.

    If there's only one chunk, save as a single ``.ahap`` file.
    If multiple chunks, save individual files + manifest + combined.
    """
    results_dir = Path(settings.RESULTS_DIR) / job_id
    os.makedirs(results_dir, exist_ok=True)

    if len(ahap.chunks) == 1:
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
        "segment_duration_s": settings.HAPTIC_SEGMENT_DURATION_S,
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
            "event_count": _count_events([chunk]),
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


# ─────────────────────────────────────────────────────────
# Internal builders
# ─────────────────────────────────────────────────────────


def _build_pattern(
    transient_events: list[HapticEvent],
    intensity_envelope: list[float],
    sharpness_envelope: list[float],
    envelope_fps: float,
    chunk_start: float,
    chunk_duration: float,
    chunk_index: int,
) -> AHAPPattern:
    """Build one AHAP pattern chunk using short 2 s carrier segments.

    Instead of a single HapticContinuous spanning the full 30 s chunk,
    emits many short carriers.  Each segment gets either static params
    (flat envelope) or a short ParameterCurve (varying envelope).
    Rest segments (intensity < threshold) are skipped entirely.
    """
    seg_dur = settings.HAPTIC_SEGMENT_DURATION_S
    overlap = settings.HAPTIC_SEGMENT_OVERLAP_S
    var_threshold = settings.HAPTIC_CURVE_VARIANCE_THRESHOLD
    rest_threshold = settings.HAPTIC_REST_INTENSITY_THRESHOLD

    pattern_entries: list[dict[str, Any]] = []

    int_arr = np.array(intensity_envelope, dtype=np.float64) if intensity_envelope else np.array([], dtype=np.float64)
    shp_arr = np.array(sharpness_envelope, dtype=np.float64) if sharpness_envelope else np.array([], dtype=np.float64)

    n_segments = max(1, math.ceil(chunk_duration / seg_dur))
    carrier_count = 0

    for si in range(n_segments):
        seg_start_rel = si * seg_dur                      # relative to chunk
        seg_end_rel = min(seg_start_rel + seg_dur + overlap, chunk_duration)
        seg_actual_dur = seg_end_rel - seg_start_rel

        if seg_actual_dur < 0.01:
            continue

        # ── Slice envelope samples for this segment ──────
        env_start = round(seg_start_rel * envelope_fps)
        env_end = round(seg_end_rel * envelope_fps)
        seg_int = int_arr[env_start:env_end] if len(int_arr) > 0 else np.array([])
        seg_shp = shp_arr[env_start:env_end] if len(shp_arr) > 0 else np.array([])

        # ── Rest detection: skip silent segments ─────────
        if len(seg_int) > 0 and float(np.max(seg_int)) < rest_threshold:
            continue

        # ── Decide: static params vs ParameterCurve ──────
        int_std = float(np.std(seg_int)) if len(seg_int) > 1 else 0.0
        shp_std = float(np.std(seg_shp)) if len(seg_shp) > 1 else 0.0
        use_intensity_curve = int_std >= var_threshold and len(seg_int) > 1
        use_sharpness_curve = shp_std >= var_threshold and len(seg_shp) > 1

        # Compute static values (mean, clamped 0-1)
        static_intensity = float(np.clip(np.mean(seg_int), 0.0, 1.0)) if len(seg_int) > 0 else 0.5
        static_sharpness = float(np.clip(np.mean(seg_shp), 0.0, 1.0)) if len(seg_shp) > 0 else 0.5

        # ── Emit HapticContinuous carrier for segment ────
        # If using curves, set carrier to 1.0 so curve controls value.
        # If static, set carrier to the computed mean directly.
        carrier_intensity = 1.0 if use_intensity_curve else round(static_intensity, 4)
        carrier_sharpness = 1.0 if use_sharpness_curve else round(static_sharpness, 4)

        pattern_entries.append({
            "Event": {
                "Time": round(seg_start_rel, 4),
                "EventType": "HapticContinuous",
                "EventDuration": round(seg_actual_dur, 4),
                "EventParameters": [
                    {"ParameterID": "HapticIntensity", "ParameterValue": carrier_intensity},
                    {"ParameterID": "HapticSharpness", "ParameterValue": carrier_sharpness},
                ],
            }
        })
        carrier_count += 1

        # ── Intensity ParameterCurve (if needed) ─────────
        if use_intensity_curve:
            curve = _build_parameter_curve(
                envelope=seg_int.tolist(),
                fps=envelope_fps,
                parameter_id="HapticIntensityControl",
                time_offset=seg_start_rel,
            )
            if curve:
                pattern_entries.append(curve)

        # ── Sharpness ParameterCurve (if needed) ─────────
        if use_sharpness_curve:
            curve = _build_parameter_curve(
                envelope=seg_shp.tolist(),
                fps=envelope_fps,
                parameter_id="HapticSharpnessControl",
                time_offset=seg_start_rel,
            )
            if curve:
                pattern_entries.append(curve)

    # ── Transient accent taps (with debounce) ────────────
    debounced = _debounce_transients(transient_events)

    # Enforce per-chunk event limit: carriers + transients ≤ 128
    max_transients = settings.MAX_AHAP_EVENTS_PER_CHUNK - carrier_count
    max_transients = max(0, max_transients)
    if len(debounced) > max_transients:
        debounced = sorted(debounced, key=lambda e: e.intensity, reverse=True)[:max_transients]
        debounced = sorted(debounced, key=lambda e: e.time)

    for event in debounced:
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
    time_offset: float = 0.0,
    rdp_epsilon: float = 0.015,
) -> dict[str, Any] | None:
    """Build a ParameterCurve from an envelope with RDP simplification.

    Applies Ramer-Douglas-Peucker to reduce control points where the
    envelope is near-linear, keeping density high only at transients.
    Typically reduces point count 40-60% with no perceptual loss.

    Parameters
    ----------
    time_offset : float
        Absolute time within the chunk where this curve starts.
    rdp_epsilon : float
        Maximum allowed deviation from the simplified curve (0-1 scale).
        Smaller = more points kept.  0.015 ≈ 1.5% intensity deviation.
    """
    if not envelope:
        return None

    interval = 1.0 / fps

    # Build full-resolution point array: [(time, value), ...]
    points = np.array([
        (i * interval, max(0.0, min(1.0, v)))
        for i, v in enumerate(envelope)
    ], dtype=np.float64)

    if len(points) < 3:
        # Too few points to simplify — emit all
        control_points = [
            {"Time": round(float(p[0]), 4), "ParameterValue": round(float(p[1]), 4)}
            for p in points
        ]
    else:
        # Always keep first and last; simplify middle via RDP
        simplified = _rdp_simplify(points, rdp_epsilon)
        control_points = [
            {"Time": round(float(p[0]), 4), "ParameterValue": round(float(p[1]), 4)}
            for p in simplified
        ]

    if not control_points:
        return None

    return {
        "ParameterCurve": {
            "ParameterID": parameter_id,
            "Time": round(time_offset, 4),
            "ParameterCurveControlPoints": control_points,
        }
    }


def _rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker line simplification.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) — column 0 = time, column 1 = value.
    epsilon : float
        Maximum perpendicular distance threshold.

    Returns
    -------
    np.ndarray
        Simplified point array (subset of input rows).
    """
    if len(points) <= 2:
        return points

    # Line from first to last point
    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_len = np.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2)

    if line_len < 1e-12:
        # Start ≈ end — keep point with max deviation
        dists = np.abs(points[:, 1] - start[1])
        max_idx = int(np.argmax(dists))
        if dists[max_idx] > epsilon:
            return np.array([start, points[max_idx], end])
        return np.array([start, end])

    # Perpendicular distance via cross-product
    point_vecs = points - start
    cross = np.abs(
        line_vec[0] * point_vecs[:, 1] - line_vec[1] * point_vecs[:, 0]
    )
    dists = cross / line_len

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist <= epsilon:
        # All points close enough — just keep endpoints
        return np.array([start, end])

    # Recurse on each half
    left = _rdp_simplify(points[: max_idx + 1], epsilon)
    right = _rdp_simplify(points[max_idx:], epsilon)

    # Concatenate (skip duplicate midpoint)
    return np.vstack([left[:-1], right])


def _debounce_transients(
    events: list[HapticEvent],
) -> list[HapticEvent]:
    """Remove transients that fire within MIN_TRANSIENT_INTERVAL_MS of
    the previous accepted transient.  Keeps the first occurrence and
    drops subsequent ones inside the debounce window.
    """
    if not events:
        return []

    min_gap = settings.MIN_TRANSIENT_INTERVAL_MS / 1000.0  # convert to seconds
    sorted_evts = sorted(events, key=lambda e: e.time)
    accepted: list[HapticEvent] = [sorted_evts[0]]

    for evt in sorted_evts[1:]:
        if (evt.time - accepted[-1].time) >= min_gap:
            accepted.append(evt)

    return accepted


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


def _count_events(chunks: list[AHAPPattern]) -> int:
    """Count only Event entries (HapticContinuous + HapticTransient),
    excluding ParameterCurve entries."""
    total = 0
    for chunk in chunks:
        for entry in chunk.pattern:
            if "Event" in entry:
                total += 1
    return total


def _build_combined_ahap(ahap: AHAPFile) -> dict[str, Any]:
    """Build a single combined AHAP with all events (absolute times).

    Unlike the old approach that merged everything into **one** long
    HapticContinuous carrier, the combined file now preserves the
    segmented structure: many short 2 s carriers with their per-segment
    ParameterCurves, all offset to absolute time.
    """
    combined_pattern: list[dict[str, Any]] = []

    for chunk in ahap.chunks:
        for entry in chunk.pattern:
            if "ParameterCurve" in entry:
                curve = entry["ParameterCurve"]
                # Offset the curve's Time to absolute
                combined_pattern.append({
                    "ParameterCurve": {
                        "ParameterID": curve["ParameterID"],
                        "Time": round(curve["Time"] + chunk.start_time, 4),
                        "ParameterCurveControlPoints": curve["ParameterCurveControlPoints"],
                    }
                })
            elif "Event" in entry:
                evt = entry["Event"].copy()
                evt["Time"] = round(evt["Time"] + chunk.start_time, 4)
                combined_pattern.append({"Event": evt})

    total_events = sum(1 for e in combined_pattern if "Event" in e)

    return {
        "Version": 1.0,
        "Metadata": {
            "TotalDuration": ahap.total_duration,
            "TotalEvents": total_events,
            "TotalChunks": len(ahap.chunks),
            "SegmentDuration": settings.HAPTIC_SEGMENT_DURATION_S,
        },
        "Pattern": combined_pattern,
    }
