"""preview.py — Generate sped-up captioned MP4 preview and scene ranges from color trajectory.

Usage:
    python -m lab.video.preview \
        --color-npz work/video/traj/run_seed_100_color.npz \
        --out-mp4 work/video/preview_s100_color.mp4 \
        --out-scenes work/video/scenes.json \
        --stride 4 \
        --fps 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from lab.dwm.eval_rollouts import get_map_name


def extract_scenes(
    map_ids: np.ndarray,
    goals: np.ndarray,
    legs: np.ndarray | None = None,
    bys: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """Compute contiguous segments by (map_id, goal)."""
    n = len(map_ids)
    if n == 0:
        return []

    scenes: List[Dict[str, Any]] = []
    start_step = 0
    cur_map = int(map_ids[0])
    cur_goal = str(goals[0])
    seen_legs = set()
    if legs is not None and len(legs) > 0 and legs[0]:
        seen_legs.add(str(legs[0]))

    for i in range(1, n):
        m = int(map_ids[i])
        g = str(goals[i])
        if m != cur_map or g != cur_goal:
            # Finish previous segment
            end_step = i - 1
            scenes.append({
                "scene_id": len(scenes) + 1,
                "map_id": cur_map,
                "map_name": get_map_name(cur_map),
                "goal": cur_goal,
                "start_step": start_step,
                "end_step": end_step,
                "length": end_step - start_step + 1,
                "legs": sorted(list(seen_legs)),
            })
            start_step = i
            cur_map = m
            cur_goal = g
            seen_legs = set()
        if legs is not None and i < len(legs) and legs[i]:
            seen_legs.add(str(legs[i]))

    # Final segment
    end_step = n - 1
    scenes.append({
        "scene_id": len(scenes) + 1,
        "map_id": cur_map,
        "map_name": get_map_name(cur_map),
        "goal": cur_goal,
        "start_step": start_step,
        "end_step": end_step,
        "length": end_step - start_step + 1,
        "legs": sorted(list(seen_legs)),
    })

    return scenes


def render_preview(
    color_npz_path: str | Path,
    out_mp4_path: str | Path,
    out_scenes_path: str | Path | None = None,
    stride: int = 4,
    fps: float = 30.0,
    bar_height: int = 64,
) -> Dict[str, Any]:
    """Read color npz, write 4x nearest scaled video with caption bar, and scenes.json."""
    color_npz_path = Path(color_npz_path)
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(color_npz_path, allow_pickle=True) as data:
        colors = data["color"]  # (N, 144, 160, 3) uint8 RGB
        map_ids = data["map_ids"]
        goals = data["goals"]
        in_battles = data["in_battles"] if "in_battles" in data else np.zeros(len(colors), dtype=bool)
        legs = data["leg"] if "leg" in data else data.get("legs", np.array([""] * len(colors)))
        bys = data["by"] if "by" in data else data.get("bys", np.array([""] * len(colors)))

    n_total = len(colors)
    if n_total == 0:
        raise ValueError(f"Color array is empty in {color_npz_path}")

    # Generate scenes
    scenes = extract_scenes(map_ids, goals, legs, bys)
    if out_scenes_path is not None:
        out_scenes_path = Path(out_scenes_path)
        out_scenes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_scenes_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2)
        print(f"[Preview] Saved {len(scenes)} scenes -> {out_scenes_path}")

    # Video parameters
    game_w = 160 * 4  # 640
    game_h = 144 * 4  # 576
    total_w = game_w
    total_h = game_h + bar_height  # 640

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_mp4_path}")

    frame_indices = list(range(0, n_total, stride))
    n_written = 0

    try:
        for idx in frame_indices:
            raw_rgb = colors[idx]
            # 4x nearest upscale
            scaled_game = cv2.resize(raw_rgb, (game_w, game_h), interpolation=cv2.INTER_NEAREST)

            # Build full canvas with caption bar below
            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            canvas[:game_h, :game_w] = scaled_game
            canvas[game_h:, :] = (24, 24, 28)  # dark bar background

            # Divider line
            cv2.line(canvas, (0, game_h), (total_w, game_h), (60, 60, 70), 1)

            # Metadata for captions
            m_id = int(map_ids[idx])
            m_name = get_map_name(m_id)
            g_str = str(goals[idx])
            in_b = bool(in_battles[idx])
            leg_name = str(legs[idx]) if idx < len(legs) else ""
            by_name = str(bys[idx]) if idx < len(bys) else ""

            # Line 1: Step, Map, Goal
            line1 = f"Step {idx:05d}/{n_total} | {m_name} (Map {m_id}) | Goal: {g_str}"
            # Line 2: Active leg & navigation agent / battle state
            if in_b:
                leg_info = f"[BATTLE] Leg: {leg_name or 'none'}"
            elif leg_name:
                by_label = f"({by_name})" if by_name else ""
                leg_info = f"Leg: {leg_name} {by_label}".strip()
            else:
                leg_info = "Leg: none"

            line2 = leg_info

            # Render text (OpenCV uses BGR, but canvas is RGB here; we'll convert once at end)
            cv2.putText(
                canvas,
                line1,
                (14, game_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line2,
                (14, game_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 210, 255) if leg_name else (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

            # Convert RGB to BGR for VideoWriter
            bgr_frame = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
            n_written += 1
    finally:
        writer.release()

    video_duration = n_written / fps
    print(f"[Preview] Wrote {n_written} frames ({video_duration:.2f}s @ {fps} fps) -> {out_mp4_path}")

    return {
        "output_mp4": str(out_mp4_path),
        "output_scenes": str(out_scenes_path) if out_scenes_path else None,
        "frames_written": n_written,
        "duration_seconds": round(video_duration, 2),
        "num_scenes": len(scenes),
        "total_steps": n_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sped-up preview video and scene segmentation.")
    parser.add_argument("--color-npz", type=str, default="work/video/traj/run_seed_100_color.npz", help="Path to color trajectory npz")
    parser.add_argument("--out-mp4", type=str, default="work/video/preview_s100_color.mp4", help="Path to output preview MP4")
    parser.add_argument("--out-scenes", type=str, default="work/video/scenes.json", help="Path to output scenes.json")
    parser.add_argument("--stride", type=int, default=4, help="Frame stride (show every Nth frame)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frames per second")
    args = parser.parse_args()

    render_preview(
        color_npz_path=args.color_npz,
        out_mp4_path=args.out_mp4,
        out_scenes_path=args.out_scenes,
        stride=args.stride,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
