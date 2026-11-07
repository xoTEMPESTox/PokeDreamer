"""Tests for lab.viz3d.hook (CPU only)."""
from pathlib import Path
import numpy as np
import pytest

from lab.viz3d.hook import compute_hook_tracks, assert_camera_building_clearance_world


def test_hook_tracks_structure_and_counts():
    """Verify frame counts, shot counts, and storyboard alignment."""
    tracks = compute_hook_tracks()

    total_frames = len(tracks["cam_pos"])
    assert total_frames == 591

    shot_id = tracks["shot_id"]
    # Check storyboard frame counts per shot
    expected_counts = {
        "1a": 24,
        "1b": 105,
        "1c": 21,
        "2a": 24,
        "2b": 23,
        "2c": 58,
        "2d": 36,
        "3a": 30,
        "3b": 84,
        "3c": 165,
        "3d": 21,
    }

    for sid, exp_count in expected_counts.items():
        act_count = int(np.sum(shot_id == sid))
        assert abs(act_count - exp_count) <= 1, f"Shot {sid}: expected ~{exp_count}, got {act_count}"


def test_player_positions_and_door():
    """Verify player starts at PC door tile (23, 26) and ends at step 110."""
    tracks = compute_hook_tracks()
    player = tracks["player_track"]

    # Frame 0: PC door tile (3.76, -4.24, step 75)
    np.testing.assert_allclose(player[0, :2], [3.76, -4.24], atol=1e-3)
    assert player[0, 2] == 75.0

    # Frame 590: Route 1 tall grass at step 110
    assert player[590, 2] == 110.0


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

    # In shot 2c (frames 197..254), edges are active
    for f in range(197, 255):
        assert edge_b[f] <= edge_a[f] + 1e-4, f"Frame {f}: edge_b ({edge_b[f]}) overtook edge_a ({edge_a[f]})"

    # Edits leave visible range on both sides (x_min <= 2.4, x_max >= 5.0)
    assert edge_a[197] <= 2.41
    assert edge_a[245] >= 5.00
    assert edge_b[254] >= 5.00


def test_map_transition_and_focus_ids():
    """Verify map transition 88->89 occurs at frame 405 and focus object IDs are valid."""
    tracks = compute_hook_tracks()
    player = tracks["player_track"]
    focus_ids = tracks["focus_object_id"]

    # Transition from Viridian (step 88) to Route 1 (step 89) at frame 404 -> 405
    assert int(player[404, 2]) == 88
    assert int(player[405, 2]) == 89

    # Valid focus object IDs (0: Title, 1: PC Door, 2: Player)
    assert set(np.unique(focus_ids)).issubset({0, 1, 2})
