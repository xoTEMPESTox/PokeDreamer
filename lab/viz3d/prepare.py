"""Prepare scene.npz for 3D diorama renders."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from lab.color.rules import PLAYER_BOX
from lab.viz3d.classes import frame_classes, load_palette, sprite_regions
from lab.viz3d.mapgrid import full_map
from lab.viz3d.shots import (
    SHOTS,
    compute_gym_closer_tracks,
    compute_opener_tracks,
    compute_twin_tracks,
)


def prepare_shot(
    shot_name: str,
    traj_color_path: str | Path = "work/video/traj/run_seed_100_color.npz",
    traj_run_path: str | Path = "work/video/traj/run_seed_100.npz",
    out_dir: str | Path = "work/viz3d/shots",
) -> Path:
    """Build and save scene.npz for a given shot name."""
    if shot_name not in SHOTS:
        # Check alias
        if shot_name == "route1_opener":
            print("Note: route1_opener requested; using opener_8s configuration per amendment.")
            shot_name = "opener_8s"
        else:
            raise KeyError(f"Unknown shot '{shot_name}'. Available: {list(SHOTS.keys())}")

    cfg = SHOTS[shot_name]
    map_id = cfg["map_id"]
    state_path = cfg["state_path"]
    total_frames = cfg["total_frames"]
    fps = cfg["fps"]

    print(f"Preparing shot '{shot_name}' (map {map_id}, {total_frames} frames)...")

    # 1. Load full map layout and tile classes
    tile_ids, cidx_img, class_per_tile = full_map(state_path)
    H_tiles, W_tiles = tile_ids.shape
    H_px, W_px = cidx_img.shape

    names, _ = load_palette()
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Map class_per_tile (H_tiles, W_tiles) to per-pixel class (H_px, W_px) int8
    cls_px = np.empty((H_px, W_px), dtype=np.int8)
    for r in range(H_tiles):
        for c in range(W_tiles):
            cname = class_per_tile[r, c]
            cidx = name_to_idx.get(cname, name_to_idx.get("other", 21))
            cls_px[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = cidx

    shade_px = cidx_img.astype(np.uint8)
    origin = np.array([0, 0], dtype=np.int32)

    # 2. Load trajectory data
    r_data = np.load(traj_run_path)
    c_data = np.load(traj_color_path)
    color_frames = c_data["color"]
    xs = r_data["xs"]
    ys = r_data["ys"]

    # 3. Compute camera & player tracks
    dream_cls = None
    dream_shade = None
    reanchors = None
    brock_cls = None
    brock_shade = None

    if shot_name == "opener_8s":
        player_track, camera_aim, camera_pos, steps_per_frame = compute_opener_tracks(
            xs, ys, total_frames=total_frames
        )
    elif shot_name == "route2_twin":
        player_track, camera_aim, camera_pos, steps_per_frame = compute_twin_tracks(
            xs, ys, total_frames=total_frames, step_start=cfg["step_start"], step_end=cfg["step_end"]
        )
        print("Colorizing dream frames for Route 2 twin...")
        from lab.video.compose import get_dream_colorizer

        dream_data = np.load("work/video/dream_4_8718_k10.npz")
        color_fn, _ = get_dream_colorizer(device="cpu")
        start_step = cfg["step_start"]
        end_step = cfg["step_end"]
        d_start = int(dream_data["start"])
        d_indices = [s - d_start for s in range(start_step, end_step + 1)]
        d_raw = dream_data["dream"][d_indices]  # (N, 72, 80)
        d_color = color_fn(d_raw)  # (N, 144, 160, 3)

        d_cls_list = []
        d_shade_list = []
        for df in d_color:
            cf, shf = frame_classes(df)
            d_cls_list.append(cf)
            d_shade_list.append(shf)
        dream_cls = np.array(d_cls_list, dtype=np.int8)
        dream_shade = np.array(d_shade_list, dtype=np.uint8)
        reanchors = np.array(cfg.get("reanchors", [6874, 6884]), dtype=np.int32)
        print(f"Prepared {len(dream_cls)} dream frames. Re-anchors: {reanchors}")

    elif shot_name == "gym_closer":
        player_track, camera_aim, camera_pos, steps_per_frame = compute_gym_closer_tracks(
            total_frames=total_frames, orbit_deg=cfg.get("orbit_deg", 40.0)
        )
        # Extract Brock sprite from frame 8262 (nearest NPC in gym)
        st = 8262
        f_rgb = color_frames[st]
        # In frame 8262, Brock is at rows 44..60, cols 64..80
        brock_crop = f_rgb[44:60, 64:80]
        brock_cls, brock_shade = frame_classes(brock_crop)
        print(f"Extracted Brock NPC sprite: shape={brock_cls.shape}")

    else:
        # Fallback generic tracking for other shots
        step_start = cfg["step_start"]
        step_end = cfg["step_end"]
        walk_steps = list(range(step_start, step_end + 1))
        steps_per_frame = np.linspace(
            step_start, step_end, total_frames, dtype=np.int32
        )
        player_track = np.zeros((total_frames, 3), dtype=np.float32)
        camera_aim = np.zeros((total_frames, 3), dtype=np.float32)
        camera_pos = np.zeros((total_frames, 3), dtype=np.float32)
        for f in range(total_frames):
            st = steps_per_frame[f]
            px_x = xs[st] * 16.0 + 8.0
            px_y = ys[st] * 16.0 + 8.0
            player_track[f] = (px_x, px_y, st)
            camera_aim[f] = (px_x, px_y, 0.0)
            camera_pos[f] = (px_x - 35.0, px_y + 80.0, 75.0)

    # 4. Extract player sprite crops for distinct steps
    distinct_steps = np.unique(steps_per_frame)
    (r0, r1), (c0, c1) = PLAYER_BOX
    player_sprites_cls = np.empty((len(distinct_steps), 16, 16), dtype=np.int8)
    player_sprites_shade = np.empty((len(distinct_steps), 16, 16), dtype=np.uint8)

    for i, st in enumerate(distinct_steps):
        crop = color_frames[st, r0:r1, c0:c1]
        c_crop, sh_crop = frame_classes(crop)
        player_sprites_cls[i] = c_crop
        player_sprites_shade[i] = sh_crop

    # 5. Save scene.npz
    shot_dir = Path(out_dir) / shot_name
    shot_dir.mkdir(parents=True, exist_ok=True)
    out_file = shot_dir / "scene.npz"

    save_dict = {
        "cls": cls_px,
        "shade": shade_px,
        "origin": origin,
        "player_track": player_track,
        "camera_track": camera_aim,
        "camera_pos": camera_pos,
        "steps": steps_per_frame,
        "fps": fps,
        "player_sprites_cls": player_sprites_cls,
        "player_sprites_shade": player_sprites_shade,
        "distinct_steps": distinct_steps,
        "shot_name": shot_name,
    }
    if dream_cls is not None:
        save_dict["dream_cls"] = dream_cls
        save_dict["dream_shade"] = dream_shade
        save_dict["reanchors"] = reanchors
    if brock_cls is not None:
        save_dict["brock_cls"] = brock_cls
        save_dict["brock_shade"] = brock_shade

    np.savez_compressed(out_file, **save_dict)

    print(f"Wrote {out_file} (cls: {cls_px.shape}, frames: {total_frames})")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 3D scene data for a shot")
    parser.add_argument("--shot", default="opener_8s", help="Shot name (e.g. opener_8s, route1_opener)")
    parser.add_argument("--out-dir", default="work/viz3d/shots", help="Output directory")
    args = parser.parse_args()

    prepare_shot(args.shot, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
