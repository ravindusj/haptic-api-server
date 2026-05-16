from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import get_settings
from app.models.schemas import SceneChange, VideoFeatures

logger = logging.getLogger(__name__)
settings = get_settings()

_movinet_model: Any | None = None

VIDEO_ANALYSIS_FPS: float = 5.0

MOVINET_SIZE: int = 172
MOVINET_WINDOW: int = 16
MOVINET_HUB_URL: str = (
    "https://www.kaggle.com/models/google/movinet/"
    "TensorFlow2/a0-base-kinetics-600-classification/3"
)

_kinetics_labels: list[str] | None = None

IMPACT_INDICES: set[int] = {
    4, 11, 58, 177, 209, 211, 319, 390, 391, 458, 472, 475, 511, 513, 594,
}

CHASE_INDICES: set[int] = {236, 307, 427}

CRASH_INDICES: set[int] = {59, 67, 475}

FALL_INDICES: set[int] = {
    17, 22, 60, 71, 139, 171, 172, 173, 204, 210, 240,
    241, 263, 381, 464, 470, 486, 490, 545,
}

DRIVING_INDICES: set[int] = {149, 150, 235, 253, 292, 409, 415}

SPORTS_HIT_INDICES: set[int] = {
    76, 197, 198, 199, 205, 213, 224, 336, 342, 349, 372, 450, 453, 509,
}

_CATEGORY_MAP: dict[int, str] = {}
for _idx in IMPACT_INDICES:
    _CATEGORY_MAP[_idx] = "impact"
for _idx in CHASE_INDICES:
    _CATEGORY_MAP[_idx] = "chase"
for _idx in CRASH_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "crash")
for _idx in FALL_INDICES:
    _CATEGORY_MAP[_idx] = "fall"
for _idx in DRIVING_INDICES:
    _CATEGORY_MAP[_idx] = "driving"
for _idx in SPORTS_HIT_INDICES:
    _CATEGORY_MAP[_idx] = "sports_hit"

ALL_ACTION_CATEGORIES = ["impact", "chase", "crash", "fall", "driving", "sports_hit"]


def analyze_video(video_path: str) -> VideoFeatures:
    """Run optical flow motion detection + MoViNet action recognition on a video."""
    video_path = str(video_path)
    duration = _get_duration(video_path)
    logger.info("Video analysis: %.2fs video at %s", duration, video_path)

    (motion_intensity, scene_changes, visual_flash, camera_shake,
     movinet_frames) = _decode_and_compute_motion(video_path)

    logger.info(
        "Motion analysis: %d frames, %d scene changes, "
        "avg_motion=%.3f, peak_motion=%.3f",
        len(motion_intensity),
        len(scene_changes),
        float(np.mean(motion_intensity)) if motion_intensity else 0.0,
        float(np.max(motion_intensity)) if motion_intensity else 0.0,
    )

    action_scores, dominant_actions, window_dur = _classify_actions(
        movinet_frames, duration,
    )
    logger.info(
        "Action classification: %d windows (%.2fs each), categories=%s",
        len(dominant_actions),
        window_dur,
        {c: sum(1 for a in dominant_actions if a == c) for c in ALL_ACTION_CATEGORIES if any(a == c for a in dominant_actions)},
    )

    return VideoFeatures(
        fps=VIDEO_ANALYSIS_FPS,
        total_frames=len(motion_intensity),
        duration_seconds=duration,
        motion_intensity=motion_intensity,
        scene_changes=scene_changes,
        visual_flash=visual_flash,
        camera_shake=camera_shake,
        action_scores=action_scores,
        dominant_actions=dominant_actions,
        action_window_duration_s=window_dur,
    )


def _decode_and_compute_motion(
    video_path: str,
) -> tuple[list[float], list[SceneChange], list[float], list[float], list[np.ndarray]]:
    """Single decode pass producing optical flow features and MoViNet input frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return [], [], [], [], []

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_step = max(1, int(round(vid_fps / VIDEO_ANALYSIS_FPS)))

    prev_gray: np.ndarray | None = None
    raw_magnitudes: list[float] = []
    frame_diffs: list[float] = []
    frame_times: list[float] = []
    luma_vals: list[float] = []
    global_trans: list[float] = []
    movinet_frames: list[np.ndarray] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            mv = cv2.resize(frame, (MOVINET_SIZE, MOVINET_SIZE))
            mv = cv2.cvtColor(mv, cv2.COLOR_BGR2RGB)
            movinet_frames.append(mv)

            h, w = frame.shape[:2]
            scale = 320.0 / max(w, 1)
            small = cv2.resize(frame, (320, max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            t = frame_idx / vid_fps

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                raw_magnitudes.append(float(np.mean(mag)))

                luma_vals.append(float(np.mean(gray)))

                gx = float(np.mean(flow[..., 0]))
                gy = float(np.mean(flow[..., 1]))
                global_trans.append(float(np.sqrt(gx * gx + gy * gy)))

                diff = float(np.mean(np.abs(
                    gray.astype(np.float32) - prev_gray.astype(np.float32)
                )))
                frame_diffs.append(diff)
                frame_times.append(t)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not raw_magnitudes:
        return [], [], [], [], movinet_frames

    raw_arr = np.array(raw_magnitudes, dtype=np.float64)
    peak = np.percentile(raw_arr, 99) if len(raw_arr) > 0 else 1.0
    if peak > 1e-6:
        raw_arr /= peak
    motion_intensity = [round(float(np.clip(v, 0.0, 1.0)), 4) for v in raw_arr]

    scene_changes: list[SceneChange] = []
    if frame_diffs:
        diff_arr = np.array(frame_diffs)
        median_diff = np.median(diff_arr)
        threshold = max(30.0, median_diff * 3.0)
        for i, d in enumerate(diff_arr):
            if d > threshold and i < len(frame_times):
                scene_changes.append(SceneChange(
                    time=round(frame_times[i], 3),
                    magnitude=round(float(d / threshold), 3),
                ))

    # Visual flash: brightness spike relative to rolling median baseline.
    visual_flash: list[float] = []
    if luma_vals:
        luma_arr = np.array(luma_vals, dtype=np.float64)
        win = max(3, int(round(VIDEO_ANALYSIS_FPS * 0.6)))
        pad = win // 2
        padded = np.pad(luma_arr, pad, mode="edge")
        baseline = np.array([
            np.median(padded[i:i + win]) for i in range(len(luma_arr))
        ])
        eps = 1.0
        rel = (luma_arr - baseline) / np.maximum(baseline, eps)
        visual_flash = [round(float(np.clip(v, 0.0, 1.0)), 4) for v in rel]

    # Camera shake: normalised global translation magnitude.
    camera_shake: list[float] = []
    if global_trans:
        trans_arr = np.array(global_trans, dtype=np.float64)
        peak = np.percentile(trans_arr, 99) if len(trans_arr) > 0 else 1.0
        if peak > 1e-6:
            trans_arr = trans_arr / peak
        camera_shake = [round(float(np.clip(v, 0.0, 1.0)), 4) for v in trans_arr]

    logger.info(
        "Optical flow: %d frames analysed (step=%d), "
        "scene_cuts=%d, flash_max=%.3f, shake_max=%.3f",
        len(motion_intensity), frame_step, len(scene_changes),
        max(visual_flash) if visual_flash else 0.0,
        max(camera_shake) if camera_shake else 0.0,
    )

    return motion_intensity, scene_changes, visual_flash, camera_shake, movinet_frames


def _load_movinet():
    """Load MoViNet-A0 from TF Hub (lazy, ~20 MB)."""
    global _movinet_model

    if _movinet_model is not None:
        return _movinet_model

    try:
        import tensorflow_hub as hub
        import tensorflow as tf

        logger.info("Loading MoViNet-A0 from TF Hub…")
        _movinet_model = hub.load(MOVINET_HUB_URL)
        logger.info("MoViNet-A0 loaded successfully")
        return _movinet_model
    except Exception as e:
        logger.warning("MoViNet failed to load: %s", str(e))
        return None


def _load_kinetics_labels() -> list[str]:
    """Load Kinetics-600 class labels (cached locally after first download)."""
    global _kinetics_labels

    if _kinetics_labels is not None:
        return _kinetics_labels

    label_path = Path(__file__).parent.parent / "data" / "kinetics_600_labels.txt"
    if label_path.exists():
        _kinetics_labels = label_path.read_text().strip().split("\n")
        return _kinetics_labels

    try:
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/tensorflow/models/"
            "f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/"
            "official/projects/movinet/files/kinetics_600_labels.txt"
        )
        response = urllib.request.urlopen(url, timeout=30)
        text = response.read().decode("utf-8")
        _kinetics_labels = text.strip().split("\n")

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(text)
        logger.info("Kinetics-600 labels downloaded and cached (%d labels)", len(_kinetics_labels))
        return _kinetics_labels
    except Exception as e:
        logger.warning("Failed to load Kinetics labels: %s", str(e))
        return [f"class_{i}" for i in range(600)]


def _classify_actions(
    frames: list[np.ndarray],
    duration: float,
) -> tuple[dict[str, list[float]], list[str], float]:
    """Classify video actions using MoViNet-A0 (sliding 16-frame windows, 50% overlap)."""
    model = _load_movinet()
    labels = _load_kinetics_labels()
    window_dur = MOVINET_WINDOW / VIDEO_ANALYSIS_FPS

    if model is None:
        n_windows = max(1, int(duration / window_dur))
        empty_scores = {cat: [0.0] * n_windows for cat in ALL_ACTION_CATEGORIES}
        return empty_scores, ["none"] * n_windows, window_dur

    if not frames:
        empty_scores = {cat: [0.0] for cat in ALL_ACTION_CATEGORIES}
        return empty_scores, ["none"], window_dur

    import tensorflow as tf

    step = max(1, MOVINET_WINDOW // 2)

    action_scores: dict[str, list[float]] = {cat: [] for cat in ALL_ACTION_CATEGORIES}
    dominant_actions: list[str] = []

    for start in range(0, len(frames), step):
        window = frames[start : start + MOVINET_WINDOW]
        if len(window) < 4:
            break

        while len(window) < MOVINET_WINDOW:
            window.append(window[-1])

        batch = np.expand_dims(
            np.stack(window).astype(np.float32) / 255.0, axis=0,
        )
        input_tensor = tf.constant(batch, dtype=tf.float32)

        try:
            logits = model(dict(image=input_tensor))
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

            window_scores: dict[str, float] = {}
            for cat in ALL_ACTION_CATEGORIES:
                cat_indices = [i for i, c in _CATEGORY_MAP.items() if c == cat and i < len(probs)]
                if cat_indices:
                    window_scores[cat] = float(np.max(probs[cat_indices]))
                else:
                    window_scores[cat] = 0.0
                action_scores[cat].append(round(window_scores[cat], 4))

            best_cat = max(window_scores, key=window_scores.get)  # type: ignore
            if window_scores[best_cat] >= 0.05:
                dominant_actions.append(best_cat)
            else:
                dominant_actions.append("none")

        except Exception as e:
            logger.warning("MoViNet inference failed for window %d: %s", start, str(e))
            for cat in ALL_ACTION_CATEGORIES:
                action_scores[cat].append(0.0)
            dominant_actions.append("none")

    if not dominant_actions:
        dominant_actions = ["none"]
        for cat in ALL_ACTION_CATEGORIES:
            action_scores[cat] = [0.0]

    logger.info(
        "MoViNet: %d windows classified, dominant breakdown: %s",
        len(dominant_actions),
        {c: dominant_actions.count(c) for c in set(dominant_actions) if c != "none"},
    )

    return action_scores, dominant_actions, window_dur


def _get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for video analysis")
    return float(result.stdout.decode().strip())
