from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

C_AUDIO = "#2196F3"
C_HAPTIC = "#FF5722"
C_SHARP_A = "#4CAF50"
C_SHARP_H = "#E91E63"
C_MOTION = "#9C27B0"
C_SCENE = "#F44336"
C_BEAT = "#607D8B"
C_SPEECH = "#FFEB3B"

BAND_COLORS = [
    "#1A237E", "#283593", "#43A047", "#FDD835", "#FF8F00", "#E53935",
]
BAND_LABELS = [
    "Sub-bass (20–60 Hz)", "Bass (60–250 Hz)", "Low-mid (250–500 Hz)",
    "Mid (500–2 kHz)", "Presence (2–4 kHz)", "Brilliance (4–8 kHz)",
]

ACTION_COLORS = {
    "impact":     "#F44336",
    "chase":      "#FF9800",
    "crash":      "#9C27B0",
    "fall":       "#2196F3",
    "driving":    "#4CAF50",
    "sports_hit": "#FFEB3B",
    "none":       "#EEEEEE",
}


def generate_visualization(job_id: str) -> bytes:
    """Generate a multi-panel comparison PNG for a completed job."""
    analysis_path = Path(settings.RESULTS_DIR) / job_id / f"{job_id}_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis data not found for job {job_id}")

    with open(analysis_path) as f:
        data = json.load(f)

    dsp = data["dsp"]
    ai = data["ai"]
    video = data.get("video")
    tl = data["timeline"]

    duration = dsp["duration_seconds"]
    has_video = video is not None and bool(video.get("motion_intensity"))

    n_dsp = len(dsp["rms_energy"])
    t_dsp = np.linspace(0, duration, n_dsp)

    n_env = len(tl["intensity_envelope"])
    t_env = np.linspace(0, duration, n_env)

    n_panels = 5 if has_video else 4
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(18, 3.4 * n_panels),
        sharex=True,
        gridspec_kw={"hspace": 0.38},
    )
    fig.suptitle(
        f"Haptic Analysis Visualization — {job_id}  ({duration:.1f}s)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    ax1 = axes[0]
    ax1.fill_between(t_dsp, dsp["rms_energy"], alpha=0.22, color=C_AUDIO)
    ax1.plot(t_dsp, dsp["rms_energy"], color=C_AUDIO, linewidth=0.7, label="Audio RMS Energy")
    ax1.plot(t_env, tl["intensity_envelope"], color=C_HAPTIC, linewidth=1.3, label="Haptic Intensity")
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_title("Audio RMS Energy  vs  Haptic Intensity Envelope", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.15)

    ax2 = axes[1]
    bands = [
        np.array(dsp["sub_bass_energy"]),
        np.array(dsp["bass_energy"]),
        np.array(dsp["low_mid_energy"]),
        np.array(dsp["mid_energy"]),
        np.array(dsp["presence_energy"]),
        np.array(dsp["brilliance_energy"]),
    ]
    ax2.stackplot(t_dsp, *bands, colors=BAND_COLORS, labels=BAND_LABELS, alpha=0.80)
    ax2.set_ylabel("Energy", fontsize=9)
    ax2.set_title("6-Band Frequency Decomposition", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.15)

    ax3 = axes[2]
    ax3.plot(t_dsp, dsp["spectral_centroid"], color=C_SHARP_A, linewidth=0.7, alpha=0.65, label="Spectral Centroid (brightness)")
    ax3.plot(t_env, tl["sharpness_envelope"], color=C_SHARP_H, linewidth=1.3, label="Haptic Sharpness")
    ax3.set_ylabel("Value (0–1)", fontsize=9)
    ax3.set_title("Audio Brightness  vs  Haptic Sharpness", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_ylim(0, 1.08)
    ax3.grid(True, alpha=0.15)

    ax4 = axes[3]

    for seg in ai.get("speech_segments", []):
        ax4.axvspan(seg["start"], seg["end"], alpha=0.18, color=C_SPEECH, zorder=1)

    for bt in dsp.get("beat_times", []):
        ax4.axvline(bt, color=C_BEAT, alpha=0.22, linewidth=0.5, zorder=2)

    ai_n = len(ai["haptic_scores"])
    if ai_n > 0:
        t_ai = np.linspace(0, duration, ai_n)
        ax4.fill_between(t_ai, ai["haptic_scores"], alpha=0.12, color=C_AUDIO)
        ax4.plot(t_ai, ai["haptic_scores"], color=C_AUDIO, linewidth=0.5, alpha=0.45)

    trans_events = [e for e in tl["events"] if e["event_type"] == "transient"]
    if trans_events:
        trans_t = [e["time"] for e in trans_events]
        trans_i = [e["intensity"] for e in trans_events]
        markerline, stemlines, baseline = ax4.stem(trans_t, trans_i, linefmt="-", markerfmt="o", basefmt="")
        plt.setp(stemlines, linewidth=0.5, alpha=0.55, color=C_HAPTIC)
        plt.setp(markerline, markersize=2.5, color=C_HAPTIC)

    patches_4 = [
        mpatches.Patch(color=C_HAPTIC, alpha=0.7, label=f"Transient Events ({len(trans_events)})"),
        mpatches.Patch(color=C_BEAT, alpha=0.3, label=f"Beats ({len(dsp.get('beat_times', []))})"),
        mpatches.Patch(color=C_SPEECH, alpha=0.25, label=f"Speech Regions ({len(ai.get('speech_segments', []))})"),
        mpatches.Patch(color=C_AUDIO, alpha=0.25, label="AI Haptic Score"),
    ]
    ax4.legend(handles=patches_4, loc="upper right", fontsize=7)
    ax4.set_ylabel("Intensity", fontsize=9)
    ax4.set_title("Transient Events  /  Beats  /  Speech Regions  /  AI Score", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 1.18)
    ax4.grid(True, alpha=0.15)

    if has_video:
        ax5 = axes[4]
        n_motion = len(video["motion_intensity"])
        t_motion = np.linspace(0, duration, n_motion)

        ax5.fill_between(t_motion, video["motion_intensity"], alpha=0.22, color=C_MOTION)
        ax5.plot(t_motion, video["motion_intensity"], color=C_MOTION, linewidth=1.0, label="Motion Intensity")

        if video.get("dominant_actions"):
            win_dur = video.get("action_window_duration_s", 3.2)
            for wi, act in enumerate(video["dominant_actions"]):
                if act != "none":
                    t_start = wi * win_dur
                    t_end = min(t_start + win_dur, duration)
                    ax5.axvspan(t_start, t_end, alpha=0.15, color=ACTION_COLORS.get(act, "#CCCCCC"), zorder=1)

        for sc in video.get("scene_changes", []):
            sc_time = sc["time"] if isinstance(sc, dict) else sc
            sc_mag = sc.get("magnitude", 1.0) if isinstance(sc, dict) else 1.0
            ax5.axvline(sc_time, color=C_SCENE, alpha=min(0.9, 0.3 + 0.12 * sc_mag), linewidth=1.0 + sc_mag * 0.35, linestyle="--", zorder=3)

        action_patches = [mpatches.Patch(color=C_MOTION, alpha=0.4, label="Motion Intensity")]
        seen = set(video.get("dominant_actions", []))
        for act in ["impact", "chase", "crash", "fall", "driving", "sports_hit"]:
            if act in seen:
                action_patches.append(mpatches.Patch(color=ACTION_COLORS[act], alpha=0.3, label=act.replace("_", " ").title()))
        if video.get("scene_changes"):
            action_patches.append(mpatches.Patch(color=C_SCENE, alpha=0.5, label=f"Scene Cuts ({len(video['scene_changes'])})"))
        ax5.legend(handles=action_patches, loc="upper right", fontsize=7, ncol=2)
        ax5.set_ylabel("Intensity", fontsize=9)
        ax5.set_title("Video Motion  /  Action Recognition  /  Scene Cuts", fontsize=11, fontweight="bold")
        ax5.set_ylim(0, 1.18)
        ax5.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    axes[-1].set_xlim(0, duration)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    png_bytes = buf.read()
    logger.info("[%s] Visualization generated: %d bytes", job_id, len(png_bytes))
    return png_bytes
