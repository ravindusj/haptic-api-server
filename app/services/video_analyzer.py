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
_active_movinet_variant: str | None = None
_active_movinet_size: int | None = None

VIDEO_ANALYSIS_FPS: float = 5.0
MOVINET_WINDOW: int = 16

# ── C2: per-variant MoViNet config ──────────────────────
# Each variant is published on Kaggle as a TF2 SavedModel.  Input
# resolution is variant-specific; the spec sheet:
#   A0: 172×172 (~3M params)
#   A1: 172×172 (~4M params)
#   A2: 224×224 (~5M params)  ← default (C2)
#   A3: 256×256 (~7M params)
_MOVINET_VARIANTS: dict[str, tuple[str, int]] = {
    "a0": (
        "https://www.kaggle.com/models/google/movinet/"
        "TensorFlow2/a0-base-kinetics-600-classification/3",
        172,
    ),
    "a1": (
        "https://www.kaggle.com/models/google/movinet/"
        "TensorFlow2/a1-base-kinetics-600-classification/3",
        172,
    ),
    "a2": (
        "https://www.kaggle.com/models/google/movinet/"
        "TensorFlow2/a2-base-kinetics-600-classification/3",
        224,
    ),
    "a3": (
        "https://www.kaggle.com/models/google/movinet/"
        "TensorFlow2/a3-base-kinetics-600-classification/3",
        256,
    ),
}


def _movinet_resolution() -> int:
    """Return the input resolution of the currently-loaded MoViNet variant."""
    if _active_movinet_size is not None:
        return _active_movinet_size
    return _MOVINET_VARIANTS.get(
        settings.MOVINET_VARIANT, _MOVINET_VARIANTS["a0"],
    )[1]


_kinetics_labels: list[str] | None = None

# ── Kinetics-600 → haptic category mapping (C1) ─────────
# Comprehensive coverage: ~340 of 600 classes mapped to 11 haptic scenarios.
# Classes that are inherently haptic-neutral (eating, sleeping, talking,
# fine-motor crafts) remain unmapped and resolve to "none" at runtime.
# When a class fits multiple categories (e.g. "smashing" is both impact
# and crash), the *first* assigned category wins via setdefault() below.

IMPACT_INDICES: set[int] = {
    # Wrestling / fighting / strikes
    4, 11, 30, 49, 58, 66, 152, 177, 209, 211,
    245, 246, 319, 388, 390, 391, 458, 472, 475,
    495, 511, 513, 527, 530, 594,
    # 4 alligator wrestling, 11 arm wrestling, 30 bending metal,
    # 49 breaking boards, 58 bull fighting, 66 capoeira,
    # 152 drop kicking, 177 fencing, 209 headbutting, 211 high kick,
    # 245 kicking field goal, 246 kicking soccer ball, 319 pillow fight,
    # 388 pumping fist, 390 punching bag, 391 punching person (boxing),
    # 458 side kick, 472 slapping, 475 smashing, 495 steer roping,
    # 511 sword fighting, 513 tackling, 527 throwing axe,
    # 530 throwing knife, 594 wrestling
}

CHASE_INDICES: set[int] = {
    223, 236, 281, 307, 427, 457, 537,
    # 223 hurdling, 236 jogging, 281 marching, 307 parkour,
    # 427 running on treadmill, 457 shuffling feet, 537 tiptoeing
}

CRASH_INDICES: set[int] = {
    59, 67, 393, 475, 552,
    # 59 bulldozing, 67 capsizing, 393 pushing car, 475 smashing,
    # 552 unloading truck
}

FALL_INDICES: set[int] = {
    17, 22, 29, 31, 39, 44, 60, 71, 139, 171, 172, 173,
    204, 207, 210, 217, 218, 225, 240, 241, 242, 263,
    266, 293, 305, 306, 381, 422, 424, 463, 464, 465, 466,
    467, 468, 470, 483, 485, 486, 487, 490, 493, 510,
    536, 538, 541, 545,
    # 17 backflip, 22 base jumping, 29 bending back, 31 biking through snow,
    # 39 bobsledding, 44 bouncing on trampoline, 60 bungee jumping,
    # 71 cartwheeling, 139 diving cliff, 171 faceplanting, 172 falling off bike,
    # 173 falling off chair, 204 gymnastics tumbling, 207 head stand,
    # 210 high jump, 217 hopscotch, 218 hoverboarding, 225 ice climbing,
    # 240 jumping bicycle, 241 jumping into pool, 242 jumping jacks,
    # 263 long jump, 266 luge, 293 mountain climber, 305 paragliding,
    # 306 parasailing, 381 pole vault, 422 rock climbing, 424 roller skating,
    # 463 skateboarding, 464 ski jumping, 465-467 skiing variants,
    # 468 skipping rope, 470 skydiving, 483 snowboarding, 485 snowmobiling,
    # 486 somersaulting, 487 spelunking, 490 springboard diving,
    # 493 standing on hands, 510 swinging on something, 536 tightrope walking,
    # 538 tobogganing, 541 trapezing, 545 triple jump
}

DRIVING_INDICES: set[int] = {
    80, 149, 150, 235, 251, 253, 292, 389, 408,
    409, 412, 415, 416, 417, 564,
    # 80 changing gear in car, 149 driving car, 150 driving tractor,
    # 235 jetskiing, 251 land sailing, 253 lawn mower racing,
    # 292 motorcycling, 389 pumping gas, 408 repairing puncture,
    # 409 riding a bike, 412 riding mechanical bull, 415 riding scooter,
    # 416 riding snow blower, 417 riding unicycle, 564 using segway
}

SPORTS_HIT_INDICES: set[int] = {
    9, 45, 76, 77, 78, 94, 133, 138, 141, 147, 155,
    189, 197, 198, 199, 205, 213, 214, 224, 233, 239,
    308, 309, 310, 326, 328, 336, 338, 342, 349, 351,
    357, 360, 364, 366, 371, 372, 377, 426, 450, 451,
    453, 480, 492, 509, 528, 529, 531, 533,
    # 9 archery, 45 bowling, 76-78 catching/throwing ball variants,
    # 94 clean and jerk, 133 deadlifting, 138 disc golfing, 141 dodgeball,
    # 147 dribbling basketball, 155 dunking basketball, 189 front raises,
    # 197-199 golf, 205 hammer throw, 213 hitting baseball, 214 hockey stop,
    # 224 hurling, 233 javelin throw, 239 juggling soccer ball,
    # 308-310 passing variants, 326 playing badminton, 328 playing basketball,
    # 336 playing cricket, 338 playing darts, 342 playing field hockey,
    # 349 playing ice hockey, 351 playing kickball, 357 playing netball,
    # 360 playing paintball, 364 playing ping pong, 366 playing polo,
    # 371 playing squash, 372 playing tennis, 377 playing volleyball,
    # 426 rope pushdown, 450 shooting basketball, 451 shooting goal,
    # 453 shot put, 480 snatch weight lifting, 492 squat,
    # 509 swinging baseball bat, 528-529 throwing variants,
    # 531 throwing snowballs, 533 throwing water balloon
}

# ── NEW: Dance / rhythmic body movement ──
DANCE_INDICES: set[int] = {
    27, 48, 84, 113, 121, 129, 130, 131, 132, 142,
    208, 222, 243, 250, 289, 291, 321, 421, 429, 488,
    491, 501, 508, 515, 517, 518, 599,
    # 27 belly dancing, 48 breakdancing, 84 cheerleading,
    # 113 country line dancing, 121 cumbia, 129-132 dancing variants,
    # 142 doing aerobics, 208 headbanging, 222 hula hooping,
    # 243 jumpstyle dancing, 250 krumping, 289 moon walking,
    # 291 mosh pit dancing, 321 pirouetting, 421 robot dancing,
    # 429 salsa dancing, 488 spinning poi, 491 square dancing,
    # 501 surfing crowd, 508 swing dancing, 515 tai chi,
    # 517 tango dancing, 518 tap dancing, 599 zumba
}

# ── NEW: Music performance / instruments / singing ──
MUSIC_PERFORMANCE_INDICES: set[int] = {
    3, 25, 62, 153, 179, 200, 244, 325, 327, 329,
    332, 334, 337, 339, 341, 343, 344, 345, 346, 347,
    348, 350, 353, 354, 358, 359, 361, 362, 367, 369,
    373, 374, 375, 376, 379, 407, 460, 519, 520,
    # 3 air drumming, 25 beatboxing, 62 busking, 153 drumming fingers,
    # 179 finger snapping, 200 gospel singing, 244 karaoke,
    # 325 accordion, 327 bagpipes, 329 bass guitar, 332 cello,
    # 334 clarinet, 337 cymbals, 339 didgeridoo, 341 drums,
    # 343 flute, 344 gong, 345 guitar, 346 hand clapping games,
    # 347 harmonica, 348 harp, 350 keyboard, 353 lute, 354 maracas,
    # 358 ocarina, 359 organ, 361 pan pipes, 362 piano,
    # 367 recorder, 369 saxophone, 373 trombone, 374 trumpet,
    # 375 ukulele, 376 violin, 379 xylophone, 407 recording music,
    # 460 singing, 519 tapping guitar, 520 tapping pen
}

# ── NEW: Water action ──
WATER_ACTION_INDICES: set[int] = {
    40, 65, 91, 119, 140, 228, 248, 428, 436, 469,
    482, 484, 502, 504, 505, 506, 507, 567, 568, 578,
    579, 590,
    # 40 bodysurfing, 65 canoeing/kayaking, 91 clam digging,
    # 119 crossing river, 140 docking boat, 228 ice swimming,
    # 248 kitesurfing, 428 sailing, 436 scuba diving,
    # 469 skipping stone, 482 snorkeling, 484 snowkiting,
    # 502 surfing water, 504-507 swimming strokes,
    # 567 wading through mud, 568 wading through water,
    # 578 water skiing, 579 water sliding, 590 windsurfing
}

# ── NEW: Construction / tools / repetitive mechanical work ──
CONSTRUCTION_INDICES: set[int] = {
    13, 32, 35, 54, 55, 56, 57, 86, 87, 90,
    180, 230, 254, 255, 256, 257, 273, 295, 322, 324,
    430, 432, 444, 554, 555, 556, 557, 560, 588,
    # 13 assembling bicycle, 32 blasting sand, 35 blowing glass,
    # 54-57 building variants, 86-87 chiseling, 90 chopping wood,
    # 180 fixing bicycle, 230 installing carpet, 254-257 laying variants,
    # 273 making horseshoes, 295 mowing lawn, 322 planing wood,
    # 324 plastering, 430 sanding floor, 432 sawing wood,
    # 444 sharpening pencil, 554 using paint roller,
    # 555 using power drill, 556 using sledge hammer,
    # 557 using wrench, 560 using circular saw, 588 welding
}

# ── NEW: Cooking / food prep ──
COOKING_INDICES: set[int] = {
    18, 20, 47, 88, 89, 107, 108, 109, 110, 124,
    126, 127, 128, 183, 190, 201, 268, 269, 272, 276,
    278, 279, 311, 312, 385, 420, 425, 433, 437, 442,
    443, 455, 497,
    # 18 baking cookies, 20 barbequing, 47 breading,
    # 88-89 chopping meat/vegetables, 107-110 cooking variants,
    # 124-128 cutting fruit, 183 flipping pancake, 190 frying vegetables,
    # 201 grinding meat, 268-279 making variants, 311-312 peeling,
    # 385 preparing salad, 420 roasting pig, 425 rolling pastry,
    # 433 scrambling eggs, 437 separating eggs, 442 shaping bread dough,
    # 443 sharpening knives, 455 shucking oysters, 497 stomping grapes
}

# All category mappings as {index → category_name}.
# setdefault() preserves first-assigned category so dual-mapped indices
# (e.g. 475 smashing — IMPACT primary, CRASH secondary) stay stable.
_CATEGORY_MAP: dict[int, str] = {}
for _idx in IMPACT_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "impact")
for _idx in CHASE_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "chase")
for _idx in CRASH_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "crash")
for _idx in FALL_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "fall")
for _idx in DRIVING_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "driving")
for _idx in SPORTS_HIT_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "sports_hit")
for _idx in DANCE_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "dance")
for _idx in MUSIC_PERFORMANCE_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "music_performance")
for _idx in WATER_ACTION_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "water_action")
for _idx in CONSTRUCTION_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "construction")
for _idx in COOKING_INDICES:
    _CATEGORY_MAP.setdefault(_idx, "cooking")

ALL_ACTION_CATEGORIES = [
    "impact", "chase", "crash", "fall", "driving", "sports_hit",
    "dance", "music_performance", "water_action",
    "construction", "cooking",
]


def analyze_video(video_path: str) -> VideoFeatures:
    """Run optical flow motion detection + MoViNet action recognition on a video."""
    video_path = str(video_path)
    duration = _get_duration(video_path)
    logger.info("Video analysis: %.2fs video at %s", duration, video_path)

    # C2: pre-load MoViNet *before* the decode pass so the active
    # variant's input resolution (172² or 224²) is locked in for the
    # MoViNet frame buffer.  A failed A2 load that silently falls
    # back to A0 would otherwise leave 224²-sized frames in a buffer
    # the model can't consume.
    _load_movinet()

    # Decode pass returns scene_changes from a legacy frame-diff
    # heuristic — we discard it and use PySceneDetect (C3) instead.
    (motion_intensity, _heuristic_scenes, visual_flash, camera_shake,
     movinet_frames) = _decode_and_compute_motion(video_path)

    # C3: AdaptiveDetector catches fades, dissolves, crossfades that
    # the old frame-diff > 3× median heuristic misses.  Falls back to
    # the heuristic if PySceneDetect import/run fails.
    scene_changes = _detect_scenes_pyscenedetect(video_path)
    if scene_changes is None:
        scene_changes = _heuristic_scenes
        logger.info("Using heuristic scene cuts (%d)", len(scene_changes))
    else:
        logger.info(
            "PySceneDetect: %d scene cuts (heuristic would have found %d)",
            len(scene_changes), len(_heuristic_scenes),
        )

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
            # C2: resolution is variant-specific (A0=172, A2=224, …)
            _mv_size = _movinet_resolution()
            mv = cv2.resize(frame, (_mv_size, _mv_size))
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


def _detect_scenes_pyscenedetect(video_path: str) -> list[SceneChange] | None:
    """Detect scene cuts using PySceneDetect's AdaptiveDetector (C3).

    Returns ``None`` if PySceneDetect is unavailable or fails — caller
    then falls back to the legacy frame-diff heuristic.

    AdaptiveDetector compares each frame against a rolling baseline
    of neighbours, which makes it robust to:
      * gradual fades and dissolves (heuristic misses entirely)
      * crossfades / soft cuts (heuristic underweights)
      * varying overall scene brightness (heuristic gets false
        positives from a single bright frame)

    Magnitude is derived from the cut's StatsManager content_val
    relative to the detector threshold, capped at 5× for parity with
    the heuristic's normalised range (the scorer scales by 5.0).
    """
    try:
        from scenedetect import (
            SceneManager, StatsManager, AdaptiveDetector, open_video,
        )
    except Exception as e:
        logger.warning("scenedetect import failed (%s) — heuristic fallback", e)
        return None

    try:
        video = open_video(video_path)
        stats = StatsManager()
        sm = SceneManager(stats_manager=stats)
        detector = AdaptiveDetector()
        sm.add_detector(detector)
        sm.detect_scenes(video=video, show_progress=False)
        scene_list = sm.get_scene_list()
    except Exception as e:
        logger.warning("PySceneDetect run failed (%s) — heuristic fallback", e)
        return None

    cuts: list[SceneChange] = []
    # First scene starts at t=0 which is not a cut — skip it.
    # PySceneDetect's AdaptiveDetector / ContentDetector metric scales
    # don't map 1:1 to the legacy heuristic's "frame-diff / 3× median"
    # magnitude, so we use a flat magnitude=3.0 (≈ medium-strong cut)
    # which the scorer maps to intensity≈0.80.  Better magnitude
    # estimation would require per-detector calibration.
    for i, (start, _end) in enumerate(scene_list):
        if i == 0:
            continue
        cuts.append(SceneChange(
            time=round(float(start.get_seconds()), 3),
            magnitude=3.0,
        ))
    return cuts


def _load_movinet():
    """Load the configured MoViNet variant lazily (C2).

    Default is A2-base (224², ~5× more accurate than A0).  On any
    failure (download, OOM, TF Hub outage) we fall back automatically
    to A0-base so analysis still runs — just at the old accuracy.
    """
    global _movinet_model, _active_movinet_variant, _active_movinet_size

    if _movinet_model is not None:
        return _movinet_model

    try:
        import tensorflow_hub as hub
    except Exception as e:
        logger.warning("tensorflow_hub import failed: %s", e)
        return None

    requested = settings.MOVINET_VARIANT.lower()
    if requested not in _MOVINET_VARIANTS:
        logger.warning(
            "Unknown MOVINET_VARIANT=%r — defaulting to a2", requested,
        )
        requested = "a2"

    # Try requested variant first, then A0 as fallback.  De-dupe so we
    # don't double-try A0 when A0 itself is requested.
    candidates: list[str] = [requested]
    if "a0" not in candidates:
        candidates.append("a0")

    for variant in candidates:
        url, size = _MOVINET_VARIANTS[variant]
        try:
            logger.info(
                "Loading MoViNet-%s (%d²) from TF Hub…",
                variant.upper(), size,
            )
            _movinet_model = hub.load(url)
            _active_movinet_variant = variant
            _active_movinet_size = size
            logger.info(
                "MoViNet-%s loaded successfully (input=%d²)",
                variant.upper(), size,
            )
            return _movinet_model
        except Exception as e:
            logger.warning(
                "MoViNet-%s load failed: %s — trying next variant",
                variant.upper(), str(e)[:200],
            )

    logger.warning("All MoViNet variants failed to load")
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
