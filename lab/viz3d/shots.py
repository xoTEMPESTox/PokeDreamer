"""Shot definitions, camera paths, and frame-to-step motion mapping."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

SHOTS: Dict[str, Dict[str, Any]] = {
    "opener_8s": {
        "name": "opener_8s",
        "map_id": 1,
        "state_path": "work/route_states/seed100_map1.state",
        "total_frames": 240,
        "fps": 30,
        "speed": 3.12,  # tiles/s
        "step_start": 74,
        "step_end": 88,
        # Pond center in world px (cols 8-13, rows 24-27 -> x ~ 176, y ~ 416)
        "pond_aim": (176.0, 416.0, 0.0),
        # Pokemon center door in world px (tile 23, 26 -> x = 376, y = 416)
        "pc_door": (376.0, 416.0, 0.0),
        "engine_defaults": {
            "samples": 64,
            "dof_fstop": 2.0,
            "sun_strength": 4.0,
            "sun_angle": 1.5,
        },
    },
    "route1_opener": {
        "name": "route1_opener",
        "map_id": 12,
        "state_path": "work/route_states/seed100_map12.state",
        "total_frames": 150,
        "fps": 30,
        "speed": 3.2,
        "step_start": 89,
        "step_end": 110,
        "engine_defaults": {
            "samples": 64,
            "dof_fstop": 2.0,
            "sun_strength": 4.0,
            "sun_angle": 1.5,
        },
    },
    "route2_twin": {
        "name": "route2_twin",
        "map_id": 13,
        "state_path": "work/route_states/seed100_map13.state",
        "total_frames": 210,
        "fps": 30,
        "speed": 3.2,
        "step_start": 6871,
        "step_end": 6892,
        "reanchors": [6874, 6884],
    },
    "gym_closer": {
        "name": "gym_closer",
        "map_id": 54,
        "state_path": "work/route_states/seed100_map54.state",
        "total_frames": 150,
        "fps": 30,
        "step_start": 8234,
        "step_end": 8262,
        "orbit_deg": 40.0,
    },
}


def smooth_exponential(points: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply exponential smoothing to a sequence of points (N, D)."""
    out = np.empty_like(points)
    out[0] = points[0]
    for i in range(1, len(points)):
        out[i] = alpha * points[i] + (1.0 - alpha) * out[i - 1]
    return out


def project_point_to_frame(
    point: np.ndarray,
    cam_pos: np.ndarray,
    cam_aim: np.ndarray,
    lens: float = 50.0,
    sensor_w: float = 36.0,
    sensor_h: float = 20.25,
) -> Tuple[float, float]:
    """Calculate normalized screen coordinates (u, v) in [0, 1] for a 3D world point."""
    forward = cam_aim - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.vstack([right, up, -forward])
    p_cam = R @ (point - cam_pos)

    fov_x = 2.0 * np.arctan(sensor_w / (2.0 * lens))
    fov_y = 2.0 * np.arctan(sensor_h / (2.0 * lens))

    ndc_x = p_cam[0] / (-p_cam[2] * np.tan(fov_x / 2.0))
    ndc_y = p_cam[1] / (-p_cam[2] * np.tan(fov_y / 2.0))

    u = float(0.5 * (ndc_x + 1.0))
    v = float(0.5 * (ndc_y + 1.0))
    return u, v


def assert_camera_building_clearance(
    camera_pos: np.ndarray,
    state_path: str = "work/route_states/seed100_map1.state",
    min_clearance_tiles: float = 0.3,
) -> float:
    """Assert camera never comes within min_clearance_tiles (default 0.3 tile = 0.048 m) of any building box."""
    from lab.viz3d.mapgrid import full_map

    tile_ids, cidx, c_per_tile = full_map(state_path)
    wall_names = {"building_wall", "door", "sign", "roof_red", "roof_blue"}
    is_bldg = np.isin(c_per_tile, list(wall_names))
    r_bldg, c_bldg = np.where(is_bldg)

    bldg_boxes = []
    for r, c in zip(r_bldg, c_bldg):
        cname = c_per_tile[r, c]
        h = 0.42 if "roof" in cname else 0.28
        bldg_boxes.append((c * 0.08, (c + 1) * 0.08, -(r + 1) * 0.08, -r * 0.08, 0.0, h))
    bldg_boxes = np.array(bldg_boxes, dtype=np.float32)

    min_dist_m = min_clearance_tiles * 0.16  # 0.3 tile = 0.048 m
    cam_blender = np.column_stack([
        camera_pos[:, 0] * 0.01,
        -camera_pos[:, 1] * 0.01,
        camera_pos[:, 2] * 0.01,
    ])

    min_dist_overall = 999.0
    for f in range(len(cam_blender)):
        cx, cy, cz = cam_blender[f]
        dx = np.maximum(0.0, np.maximum(bldg_boxes[:, 0] - cx, cx - bldg_boxes[:, 1]))
        dy = np.maximum(0.0, np.maximum(bldg_boxes[:, 2] - cy, cy - bldg_boxes[:, 3]))
        dz = np.maximum(0.0, np.maximum(bldg_boxes[:, 4] - cz, cz - bldg_boxes[:, 5]))
        dists = np.sqrt(dx * dx + dy * dy + dz * dz)
        min_dist_overall = min(min_dist_overall, float(dists.min()))

    print(f"[Clearance Check] Minimum camera-to-building clearance: {min_dist_overall:.4f} m ({min_dist_overall / 0.16:.2f} tiles)")
    assert min_dist_overall >= min_dist_m, (
        f"Camera too close to building! Got {min_dist_overall:.4f} m < required {min_dist_m:.4f} m"
    )
    return min_dist_overall


def compute_opener_tracks(
    xs: np.ndarray,
    ys: np.ndarray,
    total_frames: int = 240,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute player track, camera aim track, camera position, and step IDs for opener_8s.

    Returns:
        player_track: (F, 3) -> (x_px, y_px, facing_step)
        camera_aim: (F, 3) -> (aim_x, aim_y, aim_z)
        camera_pos: (F, 3) -> (cam_x, cam_y, cam_z)
        steps_per_frame: (F,) int32
    """
    F = total_frames
    player_track = np.zeros((F, 3), dtype=np.float32)
    camera_aim = np.zeros((F, 3), dtype=np.float32)
    camera_pos = np.zeros((F, 3), dtype=np.float32)
    steps_per_frame = np.zeros(F, dtype=np.int32)

    # Pond center in world px: cols 8..13, rows 24..27 -> center x=176, y=416, z=0
    pond_center = np.array([176.0, 416.0, 0.0], dtype=np.float32)

    # Pokemon Center door (step 74: x=23, y=25; step 75: x=23, y=26)
    pc_pos = np.array([23 * 16 + 8.0, 26 * 16 + 8.0, 0.0], dtype=np.float32)

    # Steps to walk: 75..88
    walk_steps = list(range(75, 89))
    num_walk_steps = len(walk_steps)  # 14 steps
    walk_frames = 220 - 105  # 115 frames

    # Frame-by-frame population
    for f in range(F):
        if f < 75:
            # Establishing shot: framed on water center (176, 416), camera south-east of it, high & wide, then cranes down
            t = f / 74.0
            t_ease = t * t * (3.0 - 2.0 * t)

            aim = pond_center.copy()
            camera_aim[f] = aim

            # South-east azimuth (+30 deg to +26 deg), high & wide closing from 320 px to 120 px
            el = math.radians(28.0 - 8.0 * t_ease)  # 28 deg down to 20 deg
            az = math.radians(30.0 - 4.0 * t_ease)  # South-east: azimuth +30 deg
            dist = 320.0 - 200.0 * t_ease           # 320 px (3.2m) down to 120 px (1.2m)

            cam_x = aim[0] + dist * math.cos(el) * math.sin(az)
            cam_y = aim[1] + dist * math.cos(el) * math.cos(az)
            cam_z = aim[2] + dist * math.sin(el)
            camera_pos[f] = (cam_x, cam_y, cam_z)

            # Player stays stationary at PC door during pond shot
            player_track[f] = (pc_pos[0], pc_pos[1] - 16.0, 74)
            steps_per_frame[f] = 74

        elif f < 105:
            # Glide from pond to PC door (2.5s to 3.5s)
            t = (f - 75) / 29.0
            t_ease = t * t * (3.0 - 2.0 * t)

            aim = (1.0 - t_ease) * pond_center + t_ease * pc_pos
            camera_aim[f] = aim

            # Transition from pond SE angle to standard diorama angle (48 deg tilt)
            # Maintain minimum altitude Z >= 90 px (0.90 m) and clearance >= 0.3 tile
            el = math.radians(35.0 + 13.0 * t_ease)  # 35 deg up to 48 deg
            az = math.radians(26.0 - 51.0 * t_ease)  # Swings from +26 deg to -25 deg
            dist = 155.0 - 30.0 * t_ease            # 155 px down to 125 px
            cam_x = aim[0] + dist * math.cos(el) * math.sin(az)
            cam_y = aim[1] + dist * math.cos(el) * math.cos(az)
            cam_z = aim[2] + dist * math.sin(el)
            camera_pos[f] = (cam_x, cam_y, cam_z)

            # Player steps out of door
            player_track[f] = (pc_pos[0], pc_pos[1] - 16.0 * (1.0 - t_ease), 74 if t < 0.5 else 75)
            steps_per_frame[f] = 74 if t < 0.5 else 75

        elif f < 220:
            # Walking south (steps 75..88) at medium distance (~1/8 frame height, 48 deg tilt)
            t_walk = (f - 105) / float(walk_frames)
            step_idx_float = t_walk * num_walk_steps
            step_idx = min(int(step_idx_float), num_walk_steps - 1)
            step_frac = step_idx_float - step_idx

            cur_step = walk_steps[step_idx]
            next_step = walk_steps[min(step_idx + 1, num_walk_steps - 1)]

            x0, y0 = int(xs[cur_step]), int(ys[cur_step])
            x1, y1 = int(xs[next_step]), int(ys[next_step])

            px_x = (1.0 - step_frac) * (x0 * 16 + 8) + step_frac * (x1 * 16 + 8)
            px_y = (1.0 - step_frac) * (y0 * 16 + 8) + step_frac * (y1 * 16 + 8)

            bob = 0.3 * math.sin(math.pi * step_frac) if (x0 != x1 or y0 != y1) else 0.0

            facing_step = cur_step if step_frac < 0.5 else next_step
            player_track[f] = (px_x, px_y, facing_step)
            steps_per_frame[f] = facing_step

            camera_aim[f] = (px_x, px_y, bob)

            # 48 deg tilt, follow distance 125 px (Z ~ 93 px = 0.93m), smooth azimuth drift
            az_drift = -25.0 + 6.0 * (f - 105) / 115.0
            el = math.radians(48.0)
            az = math.radians(az_drift)
            dist = 125.0
            cam_x = px_x + dist * math.cos(el) * math.sin(az)
            cam_y = px_y + dist * math.cos(el) * math.cos(az)
            cam_z = bob + dist * math.sin(el)
            camera_pos[f] = (cam_x, cam_y, cam_z)

        else:
            # Hold at Viridian exit (step 88)
            final_step = 88
            final_x = int(xs[final_step]) * 16 + 8
            final_y = int(ys[final_step]) * 16 + 8

            player_track[f] = (final_x, final_y, final_step)
            steps_per_frame[f] = final_step
            camera_aim[f] = (final_x, final_y, 0.0)

            az_drift = -19.0 + 1.0 * (f - 220) / 19.0
            el = math.radians(48.0)
            az = math.radians(az_drift)
            dist = 125.0
            cam_x = final_x + dist * math.cos(el) * math.sin(az)
            cam_y = final_y + dist * math.cos(el) * math.cos(az)
            cam_z = dist * math.sin(el)
            camera_pos[f] = (cam_x, cam_y, cam_z)

    # Smooth camera aim and pos to avoid jerky transitions
    camera_aim = smooth_exponential(camera_aim, alpha=0.20)
    camera_pos = smooth_exponential(camera_pos, alpha=0.20)

    # Verify camera building clearance (must never come within 0.3 tile = 0.048 m)
    assert_camera_building_clearance(camera_pos, min_clearance_tiles=0.3)

    # Verify that the pond center projects inside the middle third of frame 0
    # In Blender coordinate space: X = px_x * 0.01, Y = -px_y * 0.01, Z = px_z * 0.01
    pond_world = np.array([pond_center[0] * 0.01, -pond_center[1] * 0.01, pond_center[2] * 0.01], dtype=np.float32)
    cam_pos_f0 = np.array([camera_pos[0, 0] * 0.01, -camera_pos[0, 1] * 0.01, camera_pos[0, 2] * 0.01], dtype=np.float32)
    cam_aim_f0 = np.array([camera_aim[0, 0] * 0.01, -camera_aim[0, 1] * 0.01, camera_aim[0, 2] * 0.01], dtype=np.float32)

    u, v = project_point_to_frame(pond_world, cam_pos_f0, cam_aim_f0, lens=50.0)
    print(f"[Projection Check] Frame 0 pond center projected screen coords: (u={u:.4f}, v={v:.4f})")
    assert 1.0 / 3.0 <= u <= 2.0 / 3.0 and 1.0 / 3.0 <= v <= 2.0 / 3.0, (
        f"Pond center projection out of middle third! Got u={u:.4f}, v={v:.4f}"
    )

    return player_track, camera_aim, camera_pos, steps_per_frame


def compute_twin_tracks(
    xs: np.ndarray,
    ys: np.ndarray,
    total_frames: int = 210,
    step_start: int = 6871,
    step_end: int = 6892,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute tracks for route2_twin: 210 frames, 22 steps at 3.2 tiles/s."""
    F = total_frames
    steps_list = list(range(step_start, step_end + 1))
    num_steps = len(steps_list)

    player_track = np.zeros((F, 3), dtype=np.float32)
    camera_aim = np.zeros((F, 3), dtype=np.float32)
    camera_pos = np.zeros((F, 3), dtype=np.float32)
    steps_per_frame = np.zeros(F, dtype=np.int32)

    # 22 steps over ~200 frames, final 10 frames hold pose
    active_frames = 195
    for f in range(F):
        if f < active_frames:
            t = f / float(active_frames)
            s_float = t * (num_steps - 1)
            s_idx = min(int(s_float), num_steps - 2)
            frac = s_float - s_idx

            s0 = steps_list[s_idx]
            s1 = steps_list[s_idx + 1]

            x0, y0 = float(xs[s0]), float(ys[s0])
            x1, y1 = float(xs[s1]), float(ys[s1])
            px_x = (1.0 - frac) * (x0 * 16.0 + 8.0) + frac * (x1 * 16.0 + 8.0)
            px_y = (1.0 - frac) * (y0 * 16.0 + 8.0) + frac * (y1 * 16.0 + 8.0)
            facing_step = s0 if frac < 0.5 else s1
            bob = 0.3 * math.sin(math.pi * frac) if (x0 != x1 or y0 != y1) else 0.0
        else:
            final_step = steps_list[-1]
            px_x = float(xs[final_step]) * 16.0 + 8.0
            px_y = float(ys[final_step]) * 16.0 + 8.0
            facing_step = final_step
            bob = 0.0

        player_track[f] = (px_x, px_y, facing_step)
        steps_per_frame[f] = facing_step
        camera_aim[f] = (px_x, px_y, bob)

        # Diroama angle: 46 deg tilt, 120 px distance, subtle azimuth drift
        az_drift = -22.0 + 8.0 * (f / float(F))
        el = math.radians(46.0)
        az = math.radians(az_drift)
        dist = 120.0
        cam_x = px_x + dist * math.cos(el) * math.sin(az)
        cam_y = px_y + dist * math.cos(el) * math.cos(az)
        cam_z = bob + dist * math.sin(el)
        camera_pos[f] = (cam_x, cam_y, cam_z)

    camera_aim = smooth_exponential(camera_aim, alpha=0.20)
    camera_pos = smooth_exponential(camera_pos, alpha=0.20)
    return player_track, camera_aim, camera_pos, steps_per_frame


def compute_gym_closer_tracks(
    total_frames: int = 150,
    orbit_deg: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute tracks for gym_closer: static character at (4,2) facing Brock at (4,1), 40 deg orbit."""
    F = total_frames
    player_track = np.zeros((F, 3), dtype=np.float32)
    camera_aim = np.zeros((F, 3), dtype=np.float32)
    camera_pos = np.zeros((F, 3), dtype=np.float32)
    steps_per_frame = np.full(F, 8262, dtype=np.int32)

    # Red is at tile (4, 2) -> (4*16+8, 2*16+8) = (72, 40)
    red_pos = np.array([72.0, 40.0, 0.0], dtype=np.float32)
    # Brock is at tile (4, 1) -> (4*16+8, 1*16+8) = (72, 24)
    # Aim between Red and Brock at eye level (Z ~ 8 px = 8 cm)
    aim_pt = np.array([72.0, 32.0, 8.0], dtype=np.float32)

    for f in range(F):
        player_track[f] = (red_pos[0], red_pos[1], 8262)
        camera_aim[f] = aim_pt

        # 40 deg orbit from -20 deg to +20 deg
        t = f / float(F - 1)
        az_deg = -20.0 + orbit_deg * t
        el = math.radians(30.0)
        az = math.radians(az_deg)
        dist = 85.0  # 85 px = 0.85m close-in diorama

        cam_x = aim_pt[0] + dist * math.cos(el) * math.sin(az)
        cam_y = aim_pt[1] + dist * math.cos(el) * math.cos(az)
        cam_z = aim_pt[2] + dist * math.sin(el)
        camera_pos[f] = (cam_x, cam_y, cam_z)

    camera_aim = smooth_exponential(camera_aim, alpha=0.15)
    camera_pos = smooth_exponential(camera_pos, alpha=0.15)
    return player_track, camera_aim, camera_pos, steps_per_frame
