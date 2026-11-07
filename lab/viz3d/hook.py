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

    total_frames = 741
    F = total_frames

    # Key scene landmarks (Blender meters)
    pond_center = np.array([1.76, -4.16, 0.0], dtype=np.float32)
    pc_door = np.array([3.76, -4.24, 0.0], dtype=np.float32)  # Tile (23, 26)

    # Initialize per-frame arrays
    cam_pos = np.zeros((F, 3), dtype=np.float32)
    cam_aim = np.zeros((F, 3), dtype=np.float32)
    cam_focus = np.zeros((F, 3), dtype=np.float32)
    cam_fstop = np.full((F,), 2.8, dtype=np.float32)
    player_track = np.zeros((F, 4), dtype=np.float32)  # [x_m, y_m, facing_step, progress]
    player_hop = np.zeros(F, dtype=np.float32)
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
    # 1. Player positions and continuous walk across all 741 frames
    # -------------------------------------------------------------
    def get_world_pos(st: int) -> np.ndarray:
        m = map_ids[st]
        if m == 1:
            px = xs[st] * 16.0 + 8.0
            py = ys[st] * 16.0 + 8.0
        else:
            # Route 1: offset +160 px, +576 px
            px = 160.0 + xs[st] * 16.0 + 8.0
            py = 576.0 + ys[st] * 16.0 + 8.0
        return np.array([px * S, -py * S], dtype=np.float32)

    def smoothstep_blend(p: float | np.ndarray, alpha: float = 0.75) -> float | np.ndarray:
        s = 3.0 * p * p - 2.0 * p * p * p
        return (1.0 - alpha) * p + alpha * s

    door_pos = get_world_pos(75)

    # In frames 0..440 (shots 1 and 2), player stands at PC door at step 75
    for f in range(441):
        player_track[f] = [door_pos[0], door_pos[1], 75.0, 75.0]
        player_hop[f] = 0.0

    # Shot 3 intro: pauses at door before walking
    for f in range(441, 451):
        player_track[f] = [door_pos[0], door_pos[1], 75.0, 75.0]
        player_hop[f] = 0.0
    for f in range(451, 456):
        player_track[f] = [door_pos[0], door_pos[1], 76.0, 76.0]
        player_hop[f] = 0.0
    for f in range(456, 461):
        player_track[f] = [door_pos[0], door_pos[1], 77.0, 77.0]
        player_hop[f] = 0.0

    # Viridian continuous walk: 11 moves (steps 77..88) over frames 461..555
    vir_bounds = np.round(np.linspace(461, 555, 12)).astype(int)
    steps_vir = list(range(77, 89))
    for i in range(11):
        f_start = vir_bounds[i]
        f_end = vir_bounds[i + 1]
        s0 = steps_vir[i]
        s1 = steps_vir[i + 1]
        p0 = get_world_pos(s0)
        p1 = get_world_pos(s1)
        dur = float(f_end - f_start)
        for f in range(f_start, f_end):
            p = (f - f_start) / dur
            t = smoothstep_blend(p, alpha=0.75)
            pos = (1.0 - t) * p0 + t * p1
            hop = 0.004 * math.sin(math.pi * t)
            player_track[f] = [pos[0], pos[1], float(s1), float(s0) + p]
            player_hop[f] = hop

    # Route 1 continuous walk: 22 moves (steps 88..110) over frames 555..720
    r1_bounds = np.round(np.linspace(555, 720, 23)).astype(int)
    steps_r1 = list(range(88, 111))
    for i in range(22):
        f_start = r1_bounds[i]
        f_end = r1_bounds[i + 1]
        s0 = steps_r1[i]
        s1 = steps_r1[i + 1]
        p0 = get_world_pos(s0)
        p1 = get_world_pos(s1)
        dur = float(f_end - f_start)
        for f in range(f_start, f_end):
            p = (f - f_start) / dur
            t = smoothstep_blend(p, alpha=0.75)
            pos = (1.0 - t) * p0 + t * p1
            hop = 0.004 * math.sin(math.pi * t)
            player_track[f] = [pos[0], pos[1], float(s1), float(s0) + p]
            player_hop[f] = hop

    # Frames 720..740: hold at step 110 in tall grass
    p_end = get_world_pos(110)
    for f in range(720, 741):
        player_track[f] = [p_end[0], p_end[1], 110.0, 110.0]
        player_hop[f] = 0.0

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
    # Title rises twice as slowly (1.2 s = frames 24..60), fades in, reflected in water
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

        # Title rise and alpha ramp (rises 6 cm, fades in over frames 24..60)
        if f <= 60:
            tr = (f - 24) / 36.0
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
    x_min_reveal = 2.40
    x_max_reveal = 5.00

    # Shot 2b: frames 174..227 (54 frames)
    # Snap into raw: right-to-left wipe in 0.25s (8 frames = 174..181), hold ~1.5s (182..227)
    # Focus pulls 10 cm short (f/1.4)
    for f in range(174, 228):
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

    # Shot 2c: frames 228..350 (123 frames)
    # Reveal: Edge A sweeps left -> right over 3.0 s (90 frames, 228..317)
    # Edge B follows 12 frames behind (240..329)
    # Hold on full colour for frames 330..350 (21 frames)
    for f in range(228, 351):
        shot_id[f] = "2c"
        t_cam = (f - 228) / (350 - 228)
        dist = 2.0 - t_cam * 0.05
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, dist, 38.0, 25.0)
        cam_focus[f] = pc_door
        focus_object_id[f] = 1

        # Edge A sweep: sweeps across visible ground [2.20, 5.60] in frames 228..317 (90 frames)
        if f <= 317:
            t_a = (f - 228) / 89.0
            edge_a[f] = 2.20 + t_a * 3.40
        else:
            edge_a[f] = 999.0

        # Edge B sweep: follows 12 frames behind (240..329, 90 frames)
        if f < 240:
            edge_b[f] = 2.00
        elif f <= 329:
            t_b = (f - 240) / 89.0
            edge_b[f] = 2.20 + t_b * 3.40
        else:
            edge_b[f] = 999.0

        # F-stop recovery to 2.8 as A crosses center (frame 272)
        if f < 272:
            cam_fstop[f] = 1.4 + ((f - 228) / 44.0) * (2.8 - 1.4)
        else:
            cam_fstop[f] = 2.8

    # Shot 2d: frames 351..440 (90 frames = 3.0 s)
    # 2D insert (real model output)
    for f in range(351, 441):
        shot_id[f] = "2d"
        cam_aim[f] = pc_door
        cam_pos[f] = get_cam_pos(pc_door, 1.95, 38.0, 25.0)
        cam_focus[f] = pc_door
        cam_fstop[f] = 2.8
        focus_object_id[f] = 1
        edge_a[f] = 999.0
        edge_b[f] = 999.0

    # Shot 3a: frames 441..470 (30 frames)
    # Push to follow distance: distance 1.95 -> 1.25 m, elevation 38 -> 48 deg, azimuth +25 -> -25 deg
    for f in range(441, 471):
        shot_id[f] = "3a"
        t = (f - 441) / (470 - 441)
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

    # Shot 3b: frames 471..554 (84 frames)
    # Follow player through Viridian (steps 78-88)
    for f in range(471, 555):
        shot_id[f] = "3b"
        t = (f - 471) / (554 - 471)
        azim = -25.0 + t * (-19.0 - (-25.0))
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, azim)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Shot 3c: frames 555..719 (165 frames)
    # Follow crosses into Route 1 (steps 89-110)
    for f in range(555, 720):
        shot_id[f] = "3c"
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, -19.0)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Shot 3d: frames 720..740 (21 frames)
    # Hold in Route 1 tall grass at step 110
    for f in range(720, 741):
        shot_id[f] = "3d"
        aim = player_track[f][:2]
        aim_3d = np.array([aim[0], aim[1], 0.0], dtype=np.float32)
        cam_aim[f] = aim_3d
        cam_pos[f] = get_cam_pos(aim_3d, 1.25, 48.0, -19.0)
        cam_focus[f] = np.array([player_track[f, 0], player_track[f, 1], 0.06], dtype=np.float32)
        cam_fstop[f] = 4.0
        focus_object_id[f] = 2

    # Smooth aim track during follow (shots 3a-3d)
    cam_aim[441:] = smooth_exponential(cam_aim[441:], alpha=0.20)

    # Calculate sun projected screen position per frame
    for f in range(F):
        sun_3d = cam_pos[f] + 1000.0 * sun_vec
        u, v = project_point_to_frame(sun_3d, cam_pos[f], cam_aim[f], lens=40.0)
        sun_pos[f] = [u, v]

    # Verify building clearance
    if Path(world_npz_path).exists():
        assert_camera_building_clearance_world(cam_pos, world_npz_path=world_npz_path, min_clearance_m=0.048)

    # Caption definitions with layout fields for post.py
    captions = {
        "title": {
            "start_s": 0.8,
            "end_s": 4.27,
            "headline": "PokeDreamer",
            "sub": "",
            "anchor": "center",
            "dx": 0,
            "dy": 0,
            "size": 60,
        },
        "subtitle": {
            "start_s": 2.1,
            "end_s": 4.27,
            "headline": "World Model · Dream-Trained Agent · Game Harness · Upscaler & Colorizer",
            "sub": "",
            "anchor": "bottom_center",
            "dx": 0,
            "dy": -40,
            "size": 22,
        },
        "card_raw": {
            "start_s": 5.80,
            "end_s": 7.57,
            "headline": "This is all the AI sees",
            "sub": "72 x 80 pixels, greyscale",
            "anchor": "top_center",
            "dx": 0,
            "dy": 40,
            "size": 26,
        },
        "reveal_a": {
            "start_s": 7.60,
            "end_s": 11.67,
            "headline": "Detail restored",
            "sub": "neural upscaler: 144 x 160, the Game Boy's 4 shades",
            "anchor": "top_center",
            "dx": 0,
            "dy": 40,
            "size": 26,
        },
        "reveal_b": {
            "start_s": 8.00,
            "end_s": 11.67,
            "headline": "Colour restored",
            "sub": "neural colorizer: a colour per pixel",
            "anchor": "top_center",
            "dx": 0,
            "dy": 40,
            "size": 26,
        },
        "reveal_note": {
            "start_s": 7.60,
            "end_s": 11.67,
            "headline": "",
            "sub": "this 3D view is an illustration; the real screen is next",
            "anchor": "bottom_center",
            "dx": 0,
            "dy": -40,
            "size": 18,
        },
        "swipe2d": {
            "start_s": 11.70,
            "end_s": 14.67,
            "headline": "The real screen, restored by two small networks",
            "sub": "72x80 input · detail: 91% of held-out frames pixel-exact · colour: 99.27% of pixels right",
            "anchor": "bottom_center",
            "dx": 0,
            "dy": -40,
            "size": 24,
        },
        "agent_walk": {
            "start_s": 15.37,
            "end_s": 18.50,
            "headline": "Now it walks. It learned this in the dream, never in the real game.",
            "sub": "navigation policy trained only inside the world model, from frames",
            "anchor": "bottom_center",
            "dx": 0,
            "dy": -40,
            "size": 24,
        },
        "route1_label": {
            "start_s": 18.50,
            "end_s": 21.50,
            "headline": "Route 1",
            "sub": "",
            "anchor": "top_center",
            "dx": 0,
            "dy": 40,
            "size": 28,
        },
    }

    # Extract player sprite crops with validation (replaces disconnected step 78 sprite)
    color_npz_path = Path("work/video/traj/run_seed_100_color.npz")
    if color_npz_path.exists():
        c_data = np.load(color_npz_path)
        color_frames = c_data["color"]
        distinct_steps = np.unique(player_track[:, 2].astype(np.int32))
        p_cls_list = []
        p_sh_list = []
        valid_cls = None
        valid_sh = None
        for st in distinct_steps:
            st_idx = min(st, len(color_frames) - 1)
            cf = color_frames[st_idx]
            crop = cf[PLAYER_BOX[0][0] : PLAYER_BOX[0][1], PLAYER_BOX[1][0] : PLAYER_BOX[1][1]]
            c_cls, c_sh = frame_classes(crop)

            # Check for vertical gap (split sprite bug, e.g. step 78)
            has_p = (c_cls >= 17) & (c_cls <= 20)
            rc = np.sum(has_p, axis=1)
            ar = np.where(rc > 0)[0]
            is_invalid = False
            if len(ar) >= 2:
                gap = int(np.sum(rc[ar[0] : ar[-1] + 1] == 0))
                if gap >= 2:
                    is_invalid = True
                    print(f"Detected invalid player sprite at step {st} (vertical gap of {gap} rows).")

            if is_invalid and valid_cls is not None:
                print(f"Replacing invalid step {st} sprite with step {st - 1} valid sprite.")
                c_cls = valid_cls.copy()
                c_sh = valid_sh.copy()
            else:
                valid_cls = c_cls.copy()
                valid_sh = c_sh.copy()

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
        "player_hop": player_hop,
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
        player_hop=tracks["player_hop"],
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
