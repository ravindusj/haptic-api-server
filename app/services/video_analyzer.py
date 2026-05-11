"""Video-based action recognition and motion detection for haptic generation.

Two complementary analysis tiers:

1. **Optical Flow (OpenCV)** — lightweight frame-differencing and dense
   optical flow to produce a per-frame motion intensity signal.  Catches
   all fast action (chases, crashes, falls) regardless of semantic label.
   Also detects scene cuts via sudden frame-difference spikes.

2. **MoViNet-A0 (TensorFlow)** — Google's Mobile Video Network for
   action classification on Kinetics-600 classes.  Runs on CPU using
   the existing TensorFlow installation (~3 M params, 20 MB).
   Provides semantic action labels mapped to six haptic scenario
   categories: IMPACT, CHASE, CRASH, FALL, DRIVING, SPORTS_HIT.
"""

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

# ── MoViNet singleton ────────────────────────────────────

_movinet_model: Any | None = None

# ── Analysis FPS (frames sampled per second of video) ────
VIDEO_ANALYSIS_FPS: float = 5.0

# ── MoViNet input resolution ────────────────────────────
MOVINET_SIZE: int = 172
MOVINET_WINDOW: int = 16       # frames per classification window
MOVINET_HUB_URL: str = (
    "https://www.kaggle.com/models/google/movinet/"
    "TensorFlow2/a0-base-kinetics-600-classification/3"
)

# ── Kinetics-600 label file ──────────────────────────────
_kinetics_labels: list[str] | None = None

# ── Haptic scenario category mappings (Kinetics-600 indices) ─

IMPACT_INDICES: set[int] = {
    4,    # alligator wrestling
    11,   # arm wrestling
    58,   # bull fighting
    177,  # fencing (sport)
    209,  # headbutting
    211,  # high kick
    319,  # pillow fight
    390,  # punching bag
    391,  # punching person (boxing)
    458,  # side kick
    472,  # slapping
    475,  # smashing
    511,  # sword fighting
    513,  # tackling
    594,  # wrestling
}

CHASE_INDICES: set[int] = {
    236,  # jogging
    307,  # parkour
    427,  # running on treadmill
}

CRASH_INDICES: set[int] = {
    59,   # bulldozing
    67,   # capsizing
    475,  # smashing (also IMPACT — dual-mapped)
}

FALL_INDICES: set[int] = {
    17,   # backflip (human)
    22,   # base jumping
    60,   # bungee jumping
    71,   # cartwheeling
    139,  # diving cliff
    171,  # faceplanting
    172,  # falling off bike
    173,  # falling off chair
    204,  # gymnastics tumbling
    210,  # high jump
    240,  # jumping bicycle
    241,  # jumping into pool
    263,  # long jump
    381,  # pole vault
    464,  # ski jumping
    470,  # skydiving
    486,  # somersaulting
    490,  # springboard diving
    545,  # triple jump
}

DRIVING_INDICES: set[int] = {
    149,  # driving car
    150,  # driving tractor
    235,  # jetskiing
    253,  # lawn mower racing
    292,  # motorcycling
    409,  # riding a bike
    415,  # riding scooter
}

SPORTS_HIT_INDICES: set[int] = {
    76,   # catching or throwing baseball
    197,  # golf chipping
    198,  # golf driving
    199,  # golf putting
    205,  # hammer throw
    213,  # hitting baseball
    224,  # hurling (sport)
    336,  # playing cricket
    342,  # playing field hockey
    349,  # playing ice hockey
    372,  # playing tennis
    450,  # shooting basketball
    453,  # shot put
    509,  # swinging baseball bat
}

# All category mappings as {index → category_name}
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


# ── Public API ───────────────────────────────────────────


def analyze_video(video_path: str) -> VideoFeatures:
    """Run full video analysis: motion detection + action recognition.

    Parameters
    ----------
    video_path : str
        Path to the original video file.

    Returns
    -------
    VideoFeatures
        Per-window motion intensity, scene changes, and action scores.
    """
    video_path = str(video_path)
    duration = _get_duration(video_path)
    logger.info("Video analysis: %.2fs video at %s", duration, video_path)

    # ── Tier 1: Motion intensity via optical flow ────────
    motion_intensity, scene_changes, visual_flash, camera_shake = _compute_motion_features(
        video_path, duration,
    )
    logger.info(
        "Motion analysis: %d frames, %d scene changes, "
        "avg_motion=%.3f, peak_motion=%.3f",
        len(motion_intensity),
        len(scene_changes),
        float(np.mean(motion_intensity)) if motion_intensity else 0.0,
        float(np.max(motion_intensity)) if motion_intensity else 0.0,
    )

    # ── Tier 2: MoViNet action recognition ───────────────
    action_scores, dominant_actions, window_dur = _classify_actions(
        video_path, duration,
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


# ── Tier 1: Optical flow motion detection ───────────────


def _compute_motion_features(
    video_path: str,
    duration: float,
) -> tuple[list[float], list[SceneChange]]:
    """Compute per-frame motion intensity via dense optical flow.

    Samples the video at VIDEO_ANALYSIS_FPS (5 fps) and computes
    Farneback optical flow between consecutive frames.

    Returns
    -------
    motion_intensity : list[float]
        0-1 normalised motion magnitude per frame.
    scene_changes : list[float]
        Timestamps (seconds) of detected scene cuts.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return [], [], [], []

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Compute frame step to achieve target analysis FPS
    frame_step = max(1, int(round(vid_fps / VIDEO_ANALYSIS_FPS)))
    analysis_interval = frame_step / vid_fps  # seconds between samples

    prev_gray: np.ndarray | None = None
    raw_magnitudes: list[float] = []
    frame_diffs: list[float] = []
    frame_times: list[float] = []
    # Per-frame luma + per-frame global translation (V1 + V2)
    luma_vals: list[float] = []
    global_trans: list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Resize for speed (320px wide)
            h, w = frame.shape[:2]
            scale = 320.0 / max(w, 1)
            small = cv2.resize(frame, (320, max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            t = frame_idx / vid_fps

            if prev_gray is not None:
                # Dense optical flow (Farneback)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                mean_mag = float(np.mean(mag))
                raw_magnitudes.append(mean_mag)

                # V1: per-frame luma (aligned with motion_intensity)
                luma_vals.append(float(np.mean(gray)))

                # V2: global translation magnitude (mean flow vector)
                gx = float(np.mean(flow[..., 0]))
                gy = float(np.mean(flow[..., 1]))
                global_trans.append(float(np.sqrt(gx * gx + gy * gy)))

                # Frame difference for scene cut detection
                diff = float(np.mean(np.abs(
                    gray.astype(np.float32) - prev_gray.astype(np.float32)
                )))
                frame_diffs.append(diff)
                frame_times.append(t)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not raw_magnitudes:
        return [], [], [], []

    # Normalise motion to 0-1
    raw_arr = np.array(raw_magnitudes, dtype=np.float64)
    peak = np.percentile(raw_arr, 99) if len(raw_arr) > 0 else 1.0
    if peak > 1e-6:
        raw_arr /= peak
    motion_intensity = [round(float(np.clip(v, 0.0, 1.0)), 4) for v in raw_arr]

    # Detect scene changes: frame diff > 3× rolling median
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

    # ── V1: Visual flash via brightness-spike vs rolling baseline ──
    # Frame-relative measure: (luma - rolling_median) / rolling_median.
    # Robust to lighting; spikes only on sudden brightenings.
    visual_flash: list[float] = []
    if luma_vals:
        luma_arr = np.array(luma_vals, dtype=np.float64)
        win = max(3, int(round(VIDEO_ANALYSIS_FPS * 0.6)))  # ~600ms baseline
        pad = win // 2
        padded = np.pad(luma_arr, pad, mode="edge")
        baseline = np.array([
            np.median(padded[i:i + win]) for i in range(len(luma_arr))
        ])
        eps = 1.0  # avoid div-by-zero on very dark frames
        rel = (luma_arr - baseline) / np.maximum(baseline, eps)
        visual_flash = [round(float(np.clip(v, 0.0, 1.0)), 4) for v in rel]

    # ── V2: Camera shake from global translation magnitude ─────────
    # Normalised by 99th percentile so a few violent shakes don't
    # compress the rest of the signal to near-zero.
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

    return motion_intensity, scene_changes, visual_flash, camera_shake


# ── Tier 2: MoViNet action classification ───────────────


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
    """Load Kinetics-600 class labels (cached)."""
    global _kinetics_labels

    if _kinetics_labels is not None:
        return _kinetics_labels

    # Try loading from bundled file first
    label_path = Path(__file__).parent.parent / "data" / "kinetics_600_labels.txt"
    if label_path.exists():
        _kinetics_labels = label_path.read_text().strip().split("\n")
        return _kinetics_labels

    # Fallback: download
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

        # Cache locally
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(text)
        logger.info("Kinetics-600 labels downloaded and cached (%d labels)", len(_kinetics_labels))
        return _kinetics_labels
    except Exception as e:
        logger.warning("Failed to load Kinetics labels: %s", str(e))
        return [f"class_{i}" for i in range(600)]


def _classify_actions(
    video_path: str,
    duration: float,
) -> tuple[dict[str, list[float]], list[str], float]:
    """Classify video actions using MoViNet-A0.

    Processes the video in sliding windows of MOVINET_WINDOW frames,
    each resized to MOVINET_SIZE × MOVINET_SIZE.

    Returns
    -------
    action_scores : dict[str, list[float]]
        Per-window scores for each haptic category (0-1).
    dominant_actions : list[str]
        The dominant haptic category per window.
    window_duration_s : float
        Duration of each classification window in seconds.
    """
    model = _load_movinet()
    labels = _load_kinetics_labels()

    # If model failed to load, return motion-only fallback
    if model is None:
        n_windows = max(1, int(duration / (MOVINET_WINDOW / VIDEO_ANALYSIS_FPS)))
        empty_scores = {cat: [0.0] * n_windows for cat in ALL_ACTION_CATEGORIES}
        empty_dominant = ["none"] * n_windows
        window_dur = MOVINET_WINDOW / VIDEO_ANALYSIS_FPS
        return empty_scores, empty_dominant, window_dur

    import tensorflow as tf

    # ── Extract frames at analysis FPS ───────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video for action classification: %s", video_path)
        n_windows = max(1, int(duration / (MOVINET_WINDOW / VIDEO_ANALYSIS_FPS)))
        empty_scores = {cat: [0.0] * n_windows for cat in ALL_ACTION_CATEGORIES}
        return empty_scores, ["none"] * n_windows, MOVINET_WINDOW / VIDEO_ANALYSIS_FPS

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(vid_fps / VIDEO_ANALYSIS_FPS)))

    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            # Resize to MoViNet input size and normalise to [0, 1]
            resized = cv2.resize(frame, (MOVINET_SIZE, MOVINET_SIZE))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(resized.astype(np.float32) / 255.0)
        frame_idx += 1
    cap.release()

    if not frames:
        n_windows = 1
        empty_scores = {cat: [0.0] for cat in ALL_ACTION_CATEGORIES}
        return empty_scores, ["none"], MOVINET_WINDOW / VIDEO_ANALYSIS_FPS

    # ── Classify in sliding windows ──────────────────────
    window_dur = MOVINET_WINDOW / VIDEO_ANALYSIS_FPS
    step = max(1, MOVINET_WINDOW // 2)  # 50% overlap

    action_scores: dict[str, list[float]] = {cat: [] for cat in ALL_ACTION_CATEGORIES}
    dominant_actions: list[str] = []

    for start in range(0, len(frames), step):
        window = frames[start : start + MOVINET_WINDOW]
        if len(window) < 4:  # too short for meaningful classification
            break

        # Pad short windows by repeating last frame
        while len(window) < MOVINET_WINDOW:
            window.append(window[-1])

        # [1, N, H, W, 3]
        batch = np.expand_dims(np.stack(window), axis=0)
        input_tensor = tf.constant(batch, dtype=tf.float32)

        try:
            # MoViNet forward pass
            logits = model(dict(image=input_tensor))
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  # (600,)

            # Aggregate per-category scores
            window_scores: dict[str, float] = {}
            for cat in ALL_ACTION_CATEGORIES:
                cat_indices = [i for i, c in _CATEGORY_MAP.items() if c == cat and i < len(probs)]
                if cat_indices:
                    window_scores[cat] = float(np.max(probs[cat_indices]))
                else:
                    window_scores[cat] = 0.0
                action_scores[cat].append(round(window_scores[cat], 4))

            # Dominant action = category with highest score (min threshold 0.05)
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


# ── Utilities ────────────────────────────────────────────


def _get_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
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
