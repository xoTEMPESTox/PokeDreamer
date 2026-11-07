"""CPU hook timeline and camera/player tracking for the 591-frame hook diorama."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from lab.color.rules import PLAYER_BOX
from lab.viz3d.classes import frame_classes
from lab.viz3d.shots import project_point_to_frame, smooth_exponential

S = 0.01  # 1 cm per GB pixel


def assert_camera_building_clearance_world(
    cam_pos: np.ndarray,
    world_npz_path: str | Path = "work/viz3d/v2/world/world.npz",
    min_clearance_m: float = 0.048,  # 0.3 tile = 4.8 cm
) -> float:
    """Assert camera never comes within min_clearance_m of any building box in the combined world."""
    data = np.load(world_npz_path)
    wall_tiles = data["wall_tiles"]
    roof_tiles = data["roof_tiles"]

    boxes = []
    # 8x8 tile boxes in meters
    for r, c in wall_tiles:
        x0 = c * 0.08
        x1 = (c + 1) * 0.08
        y0 = -(r + 1) * 0.08
        y1 = -r * 0.08
        boxes.append([x0, x1, y0, y1, 0.0, 0.28])

    for r, c in roof_tiles:
        x0 = c * 0.08
        x1 = (c + 1) * 0.08
        y0 = -(r + 1) * 0.08
        y1 = -r * 0.08
        boxes.append([x0, x1, y0, y1, 0.0, 0.42])

    if not boxes:
        return 999.0

    boxes = np.array(boxes, dtype=np.float32)  # (N, 6)

    min_dist_overall = 999.0
    for cam in cam_pos:
        cx, cy, cz = cam
        dx = np.maximum(0.0, np.maximum(boxes[:, 0] - cx, cx - boxes[:, 1]))
        dy = np.maximum(0.0, np.maximum(boxes[:, 2] - cy, cy - boxes[:, 3]))
        dz = np.maximum(0.0, np.maximum(boxes[:, 4] - cz, cz - boxes[:, 5]))
        dists = np.sqrt(dx * dx + dy * dy + dz * dz)
        d_min = float(np.min(dists))
        if d_min < min_dist_overall:
            min_dist_overall = d_min

    assert min_dist_overall >= min_clearance_m, (
        f"Camera building clearance violated: min dist {min_dist_overall:.4f} m < {min_clearance_m:.4f} m"
    )
    return min_dist_overall


def compute_hook_tracks(
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    fps: int = 30,
    speed: float = 4.0,
    traj_path: str | Path = "work/video/traj/run_seed_100.npz",
    world_npz_path: str | Path = "work/viz3d/v2/world/world.npz",
) -> Dict[str, np.ndarray]:
    """Compute the 591-frame hook timeline tracks for camera, player, focus, reveal edges, sun pos, and titles."""
    if xs is None or ys is None:
        traj_data = np.load(traj_path)
        xs = traj_data["xs"]
        ys = traj_data["ys"]
        map_ids = traj_data["map_ids"]
    else:
        traj_data = np.load(traj_path)
        map_ids = traj_data["map_ids"]

    total_frames = 591
    F = total_frames

    # Key scene landmarks (Blender meters)
    pond_center = np.array([1.76, -4.16, 0.0], dtype=np.float32)
    pc_door = np.array([3.76, -4.24, 0.0], dtype=np.float32)  # Tile (23, 26)

    # Initialize per-frame arrays
    cam_pos = np.zeros((F, 3), dtype=np.float32)
    cam_aim = np.zeros((F, 3), dtype=np.float32)
    cam_focus = np.zeros((F, 3), dtype=np.float32)
    cam_fstop = np.full((F,), 2.8, dtype=np.float32)
    player_track = np.zeros((F, 3), dtype=np.float32)  # [x_m, y_m, step_idx]
    shot_id = np.zeros((F,), dtype=object)
    edge_a = np.full((F,), 999.0, dtype=np.float32)
    edge_b = np.full((F,), 999.0, dtype=np.float32)
    title_alpha = np.zeros((F,), dtype=np.float32)
    title_rise = np.zeros((F,), dtype=np.float32)
    sun_pos = np.zeros((F, 2), dtype=np.float32)
    focus_object_id = np.zeros((F,), dtype=np.int32)  # 0: Title, 1: PC Door, 2: Player

    # Sun direction from North-West (elevation 14 deg, azimuth 38 deg NW)
    # Light comes from NW (-X, +Y, +Z)
    sun_elev_rad = math.radians(14.0)
    sun_azim_rad = math.radians(38.0)
    # Sun unit vector in sky
    sun_vec = np.array([
        -math.cos(sun_elev_rad) * math.sin(sun_azim_rad),
        math.cos(sun_elev_rad) * math.cos(sun_azim_rad),
        math.sin(sun_elev_rad),
    ], dtype=np.float32)

    # -------------------------------------------------------------
    # 1. Player positions across all 591 frames
    # Steps 75-110
    # Step 75-77: standing at PC door
    # Step 78-88: walking Viridian (map 1)
    # Step 89-110: walking Route 1 (map 12, origin at (+160 px, +576 px))
    # -------------------------------------------------------------
    # In frames 0..290 (shots 1 and 2), player stands on PC door at step 75
    door_x = 3.76
    door_y = -4.24

    for f in range(291):
        player_track[f] = [door_x, door_y, 75.0]

    # Walking timeline for shot 3 (frames 291..590 = 300 frames)
    # Steps:
    # 291..300: pause / turn (step 75)
    # 301..305: step 76 pause (0.15s)
    # 306..310: step 77 pause (0.15s)
    # 311..404: steps 78..88 (11 moves -> 94 frames, ~8.5 frames/move = 3.5-4.0 tiles/s)
    # 405..569: steps 89..110 (22 moves -> 165 frames, 7.5 frames/move = 4.0 tiles/s)
    # 570..590: hold at step 110 (tall grass)
    walk_frame_map = {}
    for f in range(291, 301):
        walk_frame_map[f] = 75
    for f in range(301, 306):
        walk_frame_map[f] = 76
    for f in range(306, 311):
        walk_frame_map[f] = 77

    viridian_steps = np.linspace(78, 88, 405 - 311, dtype=np.int32)
    for i, f in enumerate(range(311, 405)):
        walk_frame_map[f] = int(viridian_steps[i])

    route1_steps = np.linspace(89, 110, 570 - 405, dtype=np.int32)
    for i, f in enumerate(range(405, 570)):
        walk_frame_map[f] = int(route1_steps[i])

    for f in range(570, 591):
        walk_frame_map[f] = 110

    for f in range(291, 591):
        st = walk_frame_map[f]
        m = map_ids[st]
        if m == 1:
            px = xs[st] * 16.0 + 8.0
            py = ys[st] * 16.0 + 8.0
        else:
            # Route 1: offset +160 px, +576 px
            px = 160.0 + xs[st] * 16.0 + 8.0
            py = 576.0 + ys[st] * 16.0 + 8.0
        player_track[f] = [px * S, -py * S, float(st)]

    # -------------------------------------------------------------
    # 2. Camera, Aim, Focus, Titles across Storyboard shots
    # -------------------------------------------------------------
    def get_cam_pos(aim, dist, elev_deg, azim_deg):
        elev = math.radians(elev_deg)
        azim = math.radians(azim_deg)
        dx = dist * math.cos(elev) * math.sin(azim)
        dy = -dist * math.cos(elev) * math.cos(azim)
        dz = dist * math.sin(elev)
        return aim + np.array([dx, dy, dz], dtype=np.float32)

    # Shot 1a: frames 0..23 (24 frames)
    # Start: aim pond centre (1.76, -4.16, 0.05), distance 5.8 m, elevation 13.5 deg, azimuth +40 deg
    # Pointers to Northwest into the low sun, revealing rolling hills, tree line, and hazy sky in top 20-25%
    for f in range(0, 24):
        shot_id[f] = "1a"
        aim = pond_center + np.array([0, 0, 0.05], dtype=np.float32)
        cam_aim[f] = aim
        cam_pos[f] = get_cam_pos(aim, 5.8, 13.5, 40.0)
        cam_focus[f] = pond_center + np.array([0, 0, 0.14], dtype=np.float32)
        cam_fstop[f] = 2.8
        focus_object_id[f] = 0  # Title

    # Shot 1b: frames 24..128 (105 frames)
    # Crane-down and push-in: distance 5.8 -> 1.5 m, elevation 13.5 -> 8.0 deg, azimuth +40 -> +22 deg
    # Title rises (0.8-1.4 s = frames 24..42), fades in, reflected in water
    for f in range(24, 129):
        shot_id[f] = "1b"
        t = (f - 24) / (128 - 24)
        s = 3.0 * t * t - 2.0 * t * t * t
        dist = 5.8 + s * (1.5 - 5.8)
        elev = 13.5 + s * (8.0 - 13.5)
        azim = 40.0 + s * (22.0 - 40.0)
        aim = pond_center + np.array([0, 0, 0.05], dtype=np.float32)
        cam_aim[f] = aim
        cam_pos[f] = get_cam_pos(aim, dist, elev, azim)
        cam_focus[f] = pond_center + np.array([0, 0, 0.14], dtype=np.float32)
        cam_fstop[f] = 2.8
        focus_object_id[f] = 0  # Title

        # Title rise and alpha ramp (rises 6 cm, fades in over frames 24..42)
        if f <= 42:
            tr = (f - 24) / 18.0
            title_rise[f] = tr * 0.06
            title_alpha[f] = tr
        elif f <= 115:
            title_rise[f] = 0.06
            title_alpha[f] = 1.0
        else:
            # Fade out over frames 116..128
            tout = (128 - f) / 13.0
            title_rise[f] = 0.06
            title_alpha[f] = max(0.0, tout)

    # Shot 1c: frames 129..149 (21 frames)
    # Glide east along pond's south shore, rising towards PC door
    for f in range(129, 150):
        shot_id[f] = "1c"
        t = (f - 129) / (149 - 129)
        s = 3.0 * t * t - 2.0 * t * t * t
        dist = 1.5 + s * (2.2 - 1.5)
        elev = 15.0 + s * (38.0 - 15.0)
        azim = 22.0 + s * (25.0 - 22.0)
        aim = pond_center + s * (pc_door - pond_center)
        cam_aim[f] = aim
        cam_pos[f] = get_cam_pos(aim, dist, elev, azim)
        cam_focus[f] = pc_door
        cam_fstop[f] = 2.8
        focus_object_id[f] = 1  # PC Door

    # Shot 2a: frames 150..173 (24 frames)
    # Reveal framing: aim PC door, distance 2.2 -> 2.0 m, elevation 38, azimuth 25
    for f in range(150, 174):
        shot_id[f] = "2a"
        t = (f - 150) / (173 - 150)
        dist = 2.2 + t * (2.0 - 2.2)
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, dist, 38.0, 25.0)
        cam_focus[f] = pc_door
        cam_fstop[f] = 2.8
        focus_object_id[f] = 1

    # Visible ground X-range at reveal framing (frame 174):
    # PC door is at X=3.76, Y=-4.24, camera at distance 2.0 m, elevation 38 deg, azimuth 25 deg.
    # Visible range on ground spans roughly X in [2.6 m, 4.8 m].
    # Margin 0.2 m -> edges sweep from 2.4 m to 5.0 m.
    x_min_reveal = 2.40
    x_max_reveal = 5.00

    # Shot 2b: frames 174..196 (23 frames)
    # Snap into raw: right-to-left wipe in 0.25s (8 frames = 174..181), hold 0.5s (182..196)
    # Focus pulls 10 cm short (f/1.4)
    for f in range(174, 197):
        shot_id[f] = "2b"
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, 2.0, 38.0, 25.0)
        # Pull focus 10 cm short
        cam_focus[f] = pc_door + np.array([-0.05, -0.08, 0.0], dtype=np.float32)
        cam_fstop[f] = 1.4
        focus_object_id[f] = 1

        if f <= 181:
            # Wipe from right to left (x_max -> x_min)
            t = (f - 174) / 7.0
            edge_a[f] = x_max_reveal - t * (x_max_reveal - (x_min_reveal - 0.40))
            edge_b[f] = edge_a[f]
        else:
            # Held in raw (state 0 everywhere in frame)
            edge_a[f] = x_min_reveal - 0.40
            edge_b[f] = x_min_reveal - 0.40

    # Shot 2c: frames 197..254 (58 frames)
    # Reveal: Edge A sweeps left -> right over 1.5 s (45 frames, 197..241)
    # Edge B follows 0.4 s (12 frames) behind (209..253)
    # Focus returns to door at f/2.8 as A crosses centre (frame 219)
    for f in range(197, 255):
        shot_id[f] = "2c"
        t_cam = (f - 197) / (254 - 197)
        dist = 2.0 - t_cam * 0.05
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, dist, 38.0, 25.0)
        cam_focus[f] = pc_door
        focus_object_id[f] = 1

        # Edge A sweep: sweeps across visible ground [2.65, 4.85] in frames 197..232
        if f <= 232:
            t_a = (f - 197) / 35.0
            edge_a[f] = 2.20 + t_a * 3.40  # At f=215: 2.20 + (18/35)*3.40 = 3.95 m
        else:
            edge_a[f] = 999.0

        # Edge B sweep: follows 6 frames behind (203..243)
        if f < 203:
            edge_b[f] = 2.00
        elif f <= 243:
            t_b = (f - 203) / 40.0
            edge_b[f] = 2.20 + t_b * 3.40  # At f=215: 2.20 + (12/40)*3.40 = 3.22 m
        else:
            edge_b[f] = 999.0

        # F-stop recovery to 2.8 as A crosses center (frame 219)
        if f < 219:
            cam_fstop[f] = 1.4 + ((f - 197) / 22.0) * (2.8 - 1.4)
        else:
            cam_fstop[f] = 2.8

    # Shot 2d: frames 255..290 (36 frames)
    # 2D insert (real model output)
    for f in range(255, 291):
        shot_id[f] = "2d"
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, 1.95, 38.0, 25.0)
        cam_focus[f] = pc_door
        cam_fstop[f] = 2.8
        focus_object_id[f] = 1
        edge_a[f] = 999.0
        edge_b[f] = 999.0

    # Shot 3a: frames 291..320 (30 frames)
    # Push to follow distance: distance 2.0 -> 1.25 m, elevation 38 -> 48 deg, azimuth +25 -> -25 deg
    for f in range(291, 321):
        shot_id[f] = "3a"
        t = (f - 291) / (320 - 291)
        s = 3.0 * t * t - 2.0 * t * t * t
        dist = 1.95 + s * (1.25 - 1.95)
        elev = 38.0 + s * (48.0 - 38.0)
        azim = 25.0 + s * (-25.0 - 25.0)
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, dist, elev, azim)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2  # Player

    # Shot 3b: frames 321..404 (84 frames)
    # Follow player through Viridian (steps 78-88)
    for f in range(321, 405):
        shot_id[f] = "3b"
        t = (f - 321) / (404 - 321)
        azim = -25.0 + t * (-19.0 - (-25.0))
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, azim)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Shot 3c: frames 405..569 (165 frames)
    # Follow crosses into Route 1 (steps 89-110)
    for f in range(405, 570):
        shot_id[f] = "3c"
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, -19.0)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Shot 3d: frames 570..590 (21 frames)
    # Hold in Route 1 tall grass at step 110
    for f in range(570, 591):
        shot_id[f] = "3d"
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, -19.0)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Smooth aim track during follow (shots 3a-3d)
    cam_aim[291:] = smooth_exponential(cam_aim[291:], alpha=0.20)

    # Calculate sun projected screen position per frame
    for f in range(F):
        sun_3d = cam_pos[f] + 1000.0 * sun_vec
        u, v = project_point_to_frame(sun_3d, cam_pos[f], cam_aim[f], lens=40.0)
        sun_pos[f] = [u, v]

    # Verify building clearance
    if Path(world_npz_path).exists():
        assert_camera_building_clearance_world(cam_pos, world_npz_path=world_npz_path, min_clearance_m=0.048)

    # Caption timings per COPY_REVIEW.md §4 and §1 (two-tier: headline + sub-line)
    captions = {
        "title": {
            "start_s": 0.8,
            "end_s": 4.8,
            "headline": "PokeDreamer",
            "sub": "",
            "text": "PokeDreamer",
        },
        "subtitle": {
            "start_s": 1.6,
            "end_s": 4.8,
            "headline": "Dreams the game. Learns to walk in the dream. Plays for real.",
            "sub": "diffusion world model · navigation policies trained only in imagination · neural upscaler & colorizer",
            "text": "Dreams the game. Learns to walk in the dream. Plays for real.",
        },
        "third_line": {
            "start_s": 2.4,
            "end_s": 4.8,
            "headline": "",
            "sub": "Real run, seed 100 · 3D scene built from the game's own map data",
            "text": "Real run, seed 100 · 3D scene built from the game's own map data",
        },
        "card_raw": {
            "start_s": 6.05,
            "end_s": 6.55,
            "headline": "This is all the AI sees",
            "sub": "72 x 80 pixels, greyscale",
            "text": "This is all the AI sees (72 x 80 pixels, greyscale)",
        },
        "reveal_a": {
            "start_s": 6.55,
            "end_s": 8.5,
            "headline": "Detail restored",
            "sub": "neural upscaler: 144 x 160, the Game Boy's 4 shades",
            "text": "Detail restored · neural upscaler: 144 x 160",
        },
        "reveal_b": {
            "start_s": 6.95,
            "end_s": 8.5,
            "headline": "Colour restored",
            "sub": "neural colorizer: a colour per pixel",
            "text": "Colour restored · neural colorizer: a colour per pixel",
        },
        "reveal_note": {
            "start_s": 6.55,
            "end_s": 8.5,
            "headline": "",
            "sub": "this 3D view is an illustration; the real screen is next",
            "text": "this 3D view is an illustration; the real screen is next",
        },
        "agent_walk": {
            "start_s": 11.0,
            "end_s": 13.5,
            "headline": "Now it walks. It learned this in the dream, never in the real game.",
            "sub": "navigation policy trained only inside the world model, from frames",
            "text": "Now it walks. It learned this in the dream, never in the real game.",
        },
        "route1_label": {
            "start_s": 14.0,
            "end_s": 16.5,
            "headline": "Route 1",
            "sub": "",
            "text": "Route 1",
        },
    }

    # Extract player sprite crops for distinct steps
    color_npz_path = Path("work/video/traj/run_seed_100_color.npz")
    if color_npz_path.exists():
        c_data = np.load(color_npz_path)
        color_frames = c_data["color"]
        distinct_steps = np.unique(player_track[:, 2].astype(np.int32))
        p_cls_list = []
        p_sh_list = []
        for st in distinct_steps:
            st_idx = min(st, len(color_frames) - 1)
            cf = color_frames[st_idx]
            crop = cf[PLAYER_BOX[0][0] : PLAYER_BOX[0][1], PLAYER_BOX[1][0] : PLAYER_BOX[1][1]]
            c_cls, c_sh = frame_classes(crop)
            p_cls_list.append(c_cls)
            p_sh_list.append(c_sh)
        player_sprites_cls = np.array(p_cls_list, dtype=np.int8)
        player_sprites_shade = np.array(p_sh_list, dtype=np.uint8)
    else:
        distinct_steps = np.array([75], dtype=np.int32)
        player_sprites_cls = np.zeros((1, 16, 16), dtype=np.int8)
        player_sprites_shade = np.zeros((1, 16, 16), dtype=np.uint8)

    return {
        "cam_pos": cam_pos,
        "cam_aim": cam_aim,
        "cam_focus": cam_focus,
        "cam_fstop": cam_fstop,
        "player_track": player_track,
        "shot_id": shot_id,
        "edge_a": edge_a,
        "edge_b": edge_b,
        "title_alpha": title_alpha,
        "title_rise": title_rise,
        "sun_pos": sun_pos,
        "focus_object_id": focus_object_id,
        "player_sprites_cls": player_sprites_cls,
        "player_sprites_shade": player_sprites_shade,
        "distinct_steps": distinct_steps,
        "captions": captions,
    }


def save_hook_tracks(
    out_dir: str | Path = "work/viz3d/v2",
    world_npz_path: str | Path = "work/viz3d/v2/world/world.npz",
) -> Path:
    """Save computed tracks to hook_tracks.npz and captions.json."""
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    tracks = compute_hook_tracks(world_npz_path=world_npz_path)

    npz_path = out_path / "hook_tracks.npz"
    np.savez_compressed(
        npz_path,
        cam_pos=tracks["cam_pos"],
        cam_aim=tracks["cam_aim"],
        cam_focus=tracks["cam_focus"],
        cam_fstop=tracks["cam_fstop"],
        player_track=tracks["player_track"],
        shot_id=tracks["shot_id"],
        edge_a=tracks["edge_a"],
        edge_b=tracks["edge_b"],
        title_alpha=tracks["title_alpha"],
        title_rise=tracks["title_rise"],
        sun_pos=tracks["sun_pos"],
        focus_object_id=tracks["focus_object_id"],
        player_sprites_cls=tracks["player_sprites_cls"],
        player_sprites_shade=tracks["player_sprites_shade"],
        distinct_steps=tracks["distinct_steps"],
    )

    caption_json_path = out_path / "hook_captions.json"
    caption_json_path.write_text(json.dumps(tracks["captions"], indent=2), encoding="utf-8")

    print(f"Saved hook tracks ({len(tracks['cam_pos'])} frames) to {npz_path}")
    print(f"Saved captions to {caption_json_path}")
    return npz_path


if __name__ == "__main__":
    save_hook_tracks()
