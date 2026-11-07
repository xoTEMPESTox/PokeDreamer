"""Tests for lab.viz3d.hook (CPU only)."""
from pathlib import Path
import numpy as np
import pytest

from lab.viz3d.hook import compute_hook_tracks, assert_camera_building_clearance_world


def test_hook_tracks_structure_and_counts():
    """Verify frame counts, shot counts, and storyboard alignment."""
    tracks = compute_hook_tracks()

    total_frames = len(tracks["cam_pos"])
    assert total_frames == 723

    shot_id = tracks["shot_id"]
    # Check storyboard frame counts per shot (723-frame timeline)
    expected_counts = {
        "1a": 24,
        "1b": 105,
        "1c": 21,
        "2a": 24,
        "2b": 46,
        "2c": 131,
        "2d": 90,
        "3a": 30,
        "3b": 84,
        "3c": 80,
        "3d": 88,
    }

    for sid, exp_count in expected_counts.items():
        act_count = int(np.sum(shot_id == sid))
        assert abs(act_count - exp_count) <= 1, f"Shot {sid}: expected ~{exp_count}, got {act_count}"


def test_player_positions_and_door():
    """Verify player starts at PC door tile (23, 26) and ends at step 99."""
    tracks = compute_hook_tracks()
    player = tracks["player_track"]

    # Frame 0: PC door tile (3.76, -4.24, step 75)
    np.testing.assert_allclose(player[0, :2], [3.76, -4.24], atol=1e-3)
    assert player[0, 2] == 75.0

    # Frame 722: Route 1 tall grass at step 99
    assert player[722, 2] == 99.0


def test_camera_clearance_and_boundaries():
    """Verify camera clearance from building boxes, bounds in 400m square, and height < 20m."""
    tracks = compute_hook_tracks()
    cam_pos = tracks["cam_pos"]

    # Check bounds: inside 400m terrain square ([-200, 200] x [-200, 200]) and Z < 20m
    assert np.all(cam_pos[:, 0] >= -200.0) and np.all(cam_pos[:, 0] <= 200.0)
    assert np.all(cam_pos[:, 1] >= -200.0) and np.all(cam_pos[:, 1] <= 200.0)
    assert np.all(cam_pos[:, 2] > 0.0) and np.all(cam_pos[:, 2] < 20.0)

    # Building clearance check if world.npz exists
    world_npz = Path("work/viz3d/v2/world/world.npz")
    if world_npz.exists():
        min_dist = assert_camera_building_clearance_world(cam_pos, world_npz_path=world_npz, min_clearance_m=0.048)
        assert min_dist >= 0.048


def test_reveal_edges():
    """Verify reveal edges sweep properly: edge_b never overtakes edge_a and edges leave visible range."""
    tracks = compute_hook_tracks()
    edge_a = tracks["edge_a"]
    edge_b = tracks["edge_b"]

    # In shot 2c (frames 220..350), edges are active
    for f in range(220, 351):
        assert edge_b[f] <= edge_a[f] + 1e-4, f"Frame {f}: edge_b ({edge_b[f]}) overtook edge_a ({edge_a[f]})"

    # Edges sweep across visible ground (x_min <= 2.4, x_max >= 5.0)
    assert edge_a[220] <= 2.41
    assert edge_a[309] >= 5.00
    # owner review 2: the colorizer edge trails the upscaler edge by 20 frames (0.65 s), not 12
    assert abs(edge_b[260] - edge_a[240]) < 1e-4
    assert edge_b[329] >= 5.00


def test_map_transition_and_focus_ids():
    """Verify map transition 88->89 occurs at frame 555 and focus object IDs are valid."""
    tracks = compute_hook_tracks()
    player = tracks["player_track"]
    focus_ids = tracks["focus_object_id"]

    # Transition from Viridian (step 88) to Route 1 (step 89) at frame 554 -> 555
    assert int(player[554, 2]) == 88
    assert int(player[555, 2]) == 89

    # Valid focus object IDs (0: Title, 1: PC Door, 2: Player)
    assert set(np.unique(focus_ids)).issubset({0, 1, 2})


def test_walk_continuity_and_hop():
    """Verify max per-frame jump <= 0.2 tile and hop within [0, 0.004] m."""
    tracks = compute_hook_tracks()
    player = tracks["player_track"]
    hop = tracks["player_hop"]

    # Check max per-frame jump
    pos_xy = player[:, :2]
    diffs = np.linalg.norm(np.diff(pos_xy, axis=0), axis=1) / 0.16
    assert np.max(diffs) <= 0.20, f"Max per-frame jump exceeded 0.2 tile: {np.max(diffs):.4f}"

    # Hop height is between 0 and 0.004 m
    assert np.all(hop >= 0.0)
    assert np.max(hop) <= 0.0041
    # Hop is 0 when player is still (f < 461 and f >= 634)
    assert np.all(hop[:461] == 0.0)
    assert np.all(hop[634:] == 0.0)


def test_camera_smoothness_and_framing():
    """Verify camera aim and pos second differences drop significantly and player stays within central 60% of frame."""
    from lab.viz3d.shots import project_point_to_frame

    tracks = compute_hook_tracks()
    cam_pos = tracks["cam_pos"]
    cam_aim = tracks["cam_aim"]
    player = tracks["player_track"]

    # Bound second difference over frames 441..end
    d2_aim = np.linalg.norm(np.diff(cam_aim[441:], n=2, axis=0), axis=1)
    d2_pos = np.linalg.norm(np.diff(cam_pos[441:], n=2, axis=0), axis=1)
    max_d2_aim = float(np.max(d2_aim))
    max_d2_pos = float(np.max(d2_pos))

    assert max_d2_aim < 0.003, f"cam_aim second difference too high: {max_d2_aim:.6f} m"
    assert max_d2_pos < 0.012, f"cam_pos second difference too high: {max_d2_pos:.6f} m"

    # Player stays inside central 60% of frame width (u in [0.20, 0.80]) across walk frames
    u_min, u_max = 1.0, 0.0
    for f in range(441, 635):
        p_world = np.array([player[f, 0], player[f, 1], 0.08])
        u, v = project_point_to_frame(p_world, cam_pos[f], cam_aim[f], lens=40.0)
        u_min = min(u_min, u)
        u_max = max(u_max, u)

    assert 0.20 <= u_min and u_max <= 0.80, f"Player outside central 60% width: [{u_min:.3f}, {u_max:.3f}]"


def test_sprite_validation():
    """Verify step 78 split sprite is replaced by step 77 sprite."""
    tracks = compute_hook_tracks()
    sprites_cls = tracks["player_sprites_cls"]
    distinct = list(tracks["distinct_steps"])

    if 77 in distinct and 78 in distinct:
        idx_77 = distinct.index(77)
        idx_78 = distinct.index(78)
        # Step 78 sprite must match step 77 sprite
        np.testing.assert_array_equal(sprites_cls[idx_78], sprites_cls[idx_77])


def test_title_margins_and_scale():
    """Assert title + ball coverage >= 40% from frame 60 (owner: 25% smaller than round 4) and margins >= 5% across frames 60-128."""
    from lab.viz3d.shots import project_point_to_frame

    tracks_path = Path("work/viz3d/v2/hook_tracks.npz")
    assert tracks_path.exists()
    tracks = np.load(tracks_path)
    cam_pos = tracks["cam_pos"]
    cam_aim = tracks["cam_aim"]

    pond_center = np.array([1.76, -4.16, 0.0], dtype=np.float32)
    yaw_deg = 35.0
    rad_yaw = np.radians(yaw_deg)
    u_r = np.array([np.cos(rad_yaw), np.sin(rad_yaw), 0.0], dtype=np.float32)
    u_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for f in range(60, 129):
        t = (f - 60) / 68.0
        ease = 3.0 * t * t - 2.0 * t * t * t
        s_xy = 0.4125 + (0.1875 - 0.4125) * ease

        # title + ball span in font units, measured on the render: text ~3.5 wide (r4_f128.png: 795 px at s_xy 0.152),
        # gap 0.15, ball 0.61; the group is centred on the pond centre
        half_w = 2.35 * s_xy  # spacing 1.08: text ~3.78 wide + gap 0.3 + ball 0.62
        p_left = pond_center - half_w * u_r + 0.12 * u_up
        p_right = pond_center + half_w * u_r + 0.12 * u_up

        u_l, _ = project_point_to_frame(p_left, cam_pos[f], cam_aim[f], lens=40.0)
        u_r_proj, _ = project_point_to_frame(p_right, cam_pos[f], cam_aim[f], lens=40.0)

        u_min = min(u_l, u_r_proj)
        u_max = max(u_l, u_r_proj)
        coverage = u_max - u_min
        left_margin = u_min
        right_margin = 1.0 - u_max

        assert coverage >= 0.40, f"Frame {f}: coverage {coverage:.1%} < 40%"
        assert left_margin >= 0.05, f"Frame {f}: left margin {left_margin:.1%} < 5%"
        assert right_margin >= 0.05, f"Frame {f}: right margin {right_margin:.1%} < 5%"

    # Frame 128 should be around 80% coverage
    s128 = 0.1875
    half_w128 = 2.35 * s128
    u_l128, _ = project_point_to_frame(pond_center - half_w128 * u_r + 0.12 * u_up, cam_pos[128], cam_aim[128], lens=40.0)
    u_r128, _ = project_point_to_frame(pond_center + half_w128 * u_r + 0.12 * u_up, cam_pos[128], cam_aim[128], lens=40.0)
    cov128 = abs(u_r128 - u_l128)
    assert 0.55 <= cov128 <= 0.70, f"Frame 128: coverage {cov128:.1%} not ~63% (25% smaller than the 81% of round 4)"

