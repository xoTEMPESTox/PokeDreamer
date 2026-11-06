"""Tests for lab.viz3d.shots (CPU only, no Blender, no GPU)."""
import numpy as np
import pytest

from lab.viz3d.shots import (
    SHOTS,
    assert_camera_building_clearance,
    compute_gym_closer_tracks,
    compute_opener_tracks,
    compute_twin_tracks,
    smooth_exponential,
)


def test_shots_dict():
    assert "opener_8s" in SHOTS
    assert "route1_opener" in SHOTS
    assert SHOTS["opener_8s"]["total_frames"] == 240
    assert SHOTS["opener_8s"]["fps"] == 30
    assert SHOTS["route1_opener"]["total_frames"] == 150


def test_smooth_exponential():
    pts = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    smoothed = smooth_exponential(pts, alpha=0.5)
    assert smoothed.shape == (3, 2)
    assert np.allclose(smoothed[0], [0.0, 0.0])
    assert np.allclose(smoothed[1], [5.0, 5.0])
    assert np.allclose(smoothed[2], [7.5, 7.5])


def test_compute_opener_tracks():
    # Mock xs, ys
    xs = np.zeros(100, dtype=np.int32)
    ys = np.zeros(100, dtype=np.int32)
    xs[74:89] = np.linspace(23, 21, 15).astype(np.int32)
    ys[74:89] = np.linspace(25, 35, 15).astype(np.int32)

    player_track, camera_aim, camera_pos, steps = compute_opener_tracks(
        xs, ys, total_frames=240
    )

    assert player_track.shape == (240, 3)
    assert camera_aim.shape == (240, 3)
    assert camera_pos.shape == (240, 3)
    assert steps.shape == (240,)

    # All values must be finite numbers
    assert np.all(np.isfinite(player_track))
    assert np.all(np.isfinite(camera_aim))
    assert np.all(np.isfinite(camera_pos))

    # Pond frames (0..74) should focus on pond area (x in 100..250, y in 350..450)
    assert np.all(camera_aim[:75, 0] >= 100.0)
    assert np.all(camera_aim[:75, 0] <= 250.0)

    # South exit hold (frames 220..239) should be near final step 88 position
    assert np.all(steps[220:] == 88)


def test_clearance_assertion():
    # Camera at safe altitude above buildings
    safe_cam = np.zeros((10, 3), dtype=np.float32)
    safe_cam[:, 0] = 300.0
    safe_cam[:, 1] = 400.0
    safe_cam[:, 2] = 120.0  # 1.2 m altitude (roofs are 0.42 m)
    min_dist = assert_camera_building_clearance(safe_cam, min_clearance_tiles=0.3)
    assert min_dist >= 0.3 * 0.16  # >= 0.048 m


def test_compute_twin_tracks():
    xs = np.full(7000, 10, dtype=np.int32)
    ys = np.linspace(5, 30, 7000).astype(np.int32)
    pt, ca, cp, st = compute_twin_tracks(xs, ys, total_frames=210, step_start=6871, step_end=6892)
    assert pt.shape == (210, 3)
    assert ca.shape == (210, 3)
    assert cp.shape == (210, 3)
    assert st.shape == (210,)
    assert np.all(np.isfinite(pt))
    assert np.all(np.isfinite(ca))
    assert np.all(np.isfinite(cp))


def test_compute_gym_closer_tracks():
    pt, ca, cp, st = compute_gym_closer_tracks(total_frames=150, orbit_deg=40.0)
    assert pt.shape == (150, 3)
    assert ca.shape == (150, 3)
    assert cp.shape == (150, 3)
    assert st.shape == (150,)
    assert np.all(np.isfinite(pt))
    assert np.all(np.isfinite(ca))
    assert np.all(np.isfinite(cp))
    # Player stays at Red's position in gym: (72, 40)
    assert np.allclose(pt[:, :2], [72.0, 40.0])
