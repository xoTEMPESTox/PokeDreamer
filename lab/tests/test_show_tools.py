"""Tests for world-model evaluation and demo tooling for new areas.

Covers:
- Per-map aggregation & bootstrap in eval_rollouts
- Trajectory saving in gym_quest (--save-traj)
- Dream GIF frame composer and end-to-end rollout generator
- Map-filtered overworld window selection in play.py
"""
import os
import sys
from pathlib import Path

# Prevent OpenMP runtime collision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    import ctypes
    _lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(_lib_bin)
        except Exception:
            pass
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_lib_bin)
            except OSError:
                pass

import numpy as np
import pytest
import torch
from PIL import Image

import lab
from lab.agent.gym_quest import run_gym_quest
from lab.data import load
from lab.dwm.dream_gif import compose_frame, generate_dream_gif, get_map_name
from lab.dwm.eval_rollouts import get_map_name as eval_get_map_name, summarize, summarize_per_map
from lab.dwm.model import CTX
from lab.play import get_overworld_windows


def test_per_map_aggregation():
    """Verify summarize_per_map splits metrics across maps 13, 51, 2, 54 with bootstrap."""
    K = 4
    # Create 8 windows:
    # 2 windows on map 13 (episodes 0, 1)
    # 2 windows on map 51 (episodes 1, 2)
    # 2 windows on map 2  (episodes 3, 4)
    # 2 windows on map 54 (episodes 4, 5)
    W = 8
    map0 = np.array([13, 13, 51, 51, 2, 2, 54, 54], dtype=np.uint8)
    episode_ids = np.array([0, 1, 1, 2, 3, 4, 4, 5], dtype=np.int64)

    # Synthetic per_window metrics
    pos_err_img = np.ones((W, K), dtype=float) * 2.0
    pos_err_nochange = np.ones((W, K), dtype=float) * 1.0
    pos_err_real = np.zeros((W, K), dtype=float)
    pos_agree = np.ones((W, K), dtype=float)
    map_acc_img = np.ones((W, K), dtype=float)
    map_acc_real = np.ones((W, K), dtype=float)
    map_acc_nochange = np.ones((W, K), dtype=float)
    wrong_map_img = np.zeros((W, K), dtype=float)
    battle_acc_img = np.ones((W, K), dtype=float)
    battle_acc_nochange = np.ones((W, K), dtype=float)
    mse_img = np.full((W, K), 0.01, dtype=float)
    mse_copy = np.full((W, K), 0.02, dtype=float)
    moved = np.ones((W, K), dtype=bool)

    # Set specific values for map 2 to verify distinct aggregation
    pos_err_img[4:6] = 5.0
    wrong_map_img[4:6] = 0.25

    per_window = {
        "map0": map0,
        "moved": moved,
        "pos_err_img": pos_err_img,
        "pos_err_nochange": pos_err_nochange,
        "pos_err_real": pos_err_real,
        "pos_agree": pos_agree,
        "map_acc_img": map_acc_img,
        "map_acc_real": map_acc_real,
        "map_acc_nochange": map_acc_nochange,
        "wrong_map_img": wrong_map_img,
        "battle_acc_img": battle_acc_img,
        "battle_acc_nochange": battle_acc_nochange,
        "mse_img": mse_img,
        "mse_copy": mse_copy,
    }

    per_map = summarize_per_map(per_window, episode_ids, maps=[13, 51, 2, 54, 99])

    # Check all target maps are present
    for target in [13, 51, 2, 54]:
        key = str(target)
        assert key in per_map, f"Map {target} missing from per_map summary"
        info = per_map[key]
        assert info["n_windows"] == 2
        assert info["n_episodes"] >= 1
        assert "wrong_map_img" in info["metrics"]
        assert "pos_err_img" in info["metrics"]
        assert "pos_err_nochange" in info["metrics"]

    # Verify Map 2 has specific aggregated values
    map2_metrics = per_map["2"]["metrics"]
    assert np.allclose(map2_metrics["pos_err_img"]["mean"], 5.0)
    assert np.allclose(map2_metrics["wrong_map_img"]["mean"], 0.25)

    # Verify Map 13 has expected values
    map13_metrics = per_map["13"]["metrics"]
    assert np.allclose(map13_metrics["pos_err_img"]["mean"], 2.0)
    assert np.allclose(map13_metrics["wrong_map_img"]["mean"], 0.0)

    # Verify empty map 99 handled cleanly without error
    assert per_map["99"]["n_windows"] == 0
    assert per_map["99"]["n_episodes"] == 0


def test_trajectory_saving(tmp_path):
    """Test --save-traj in gym_quest: trajectory npz structure, types, and off-state invariance."""
    work_dir = tmp_path / "work_gym"
    traj_dir = tmp_path / "work_traj"

    # 1. Run with save_traj=True (tiny budget 2 steps)
    res_on = run_gym_quest(
        budget=2,
        save_traj=True,
        work_dir=work_dir,
        traj_dir=traj_dir,
        seed=101,
    )
    traj_path = traj_dir / "run_seed_101.npz"
    assert traj_path.exists(), f"Expected trajectory file {traj_path} not found"

    d = np.load(traj_path)
    expected_keys = ["frames", "actions", "map_ids", "xs", "ys", "in_battles", "goals"]
    for k in expected_keys:
        assert k in d, f"Missing key {k} in trajectory npz"

    N = len(d["frames"])
    assert N >= 2
    assert d["frames"].shape == (N, 72, 80)
    assert d["frames"].dtype == np.uint8
    assert d["actions"].shape == (N,)
    assert d["actions"].dtype == np.uint8
    assert d["map_ids"].shape == (N,)
    assert d["xs"].shape == (N,)
    assert d["ys"].shape == (N,)
    assert d["in_battles"].shape == (N,)
    assert d["goals"].shape == (N,)

    # System 2 goal should be a valid string
    assert len(str(d["goals"][0])) > 0

    # 2. Run with save_traj=False
    run_gym_quest(
        budget=2,
        save_traj=False,
        work_dir=work_dir,
        traj_dir=traj_dir,
        seed=102,
    )
    traj_off_path = traj_dir / "run_seed_102.npz"
    assert not traj_off_path.exists(), "Trajectory file was created even though save_traj=False"


def test_dream_gif_composer():
    """Test compose_frame: dimensions, labels, and red flag on map mismatch."""
    real_f = np.zeros((72, 80), dtype=np.uint8)
    dream_f = np.full((72, 80), 128, dtype=np.uint8)

    # 1. Matching map: ram_map == probe_map == 2 (Pewter City)
    comp_match = compose_frame(
        real_f=real_f,
        dream_f=dream_f,
        goal="gym",
        step=42,
        ram_map=2,
        probe_map=2,
        tile_drift=1,
        upsampler=None,
        target_width=640,
    )
    assert comp_match.shape == (322, 640, 3)
    assert comp_match.dtype == np.uint8

    # 2. Mismatching map: ram_map 2 vs probe_map 51 (Viridian Forest) -> flagged in red
    comp_mismatch = compose_frame(
        real_f=real_f,
        dream_f=dream_f,
        goal="gym",
        step=42,
        ram_map=2,
        probe_map=51,
        tile_drift=8,
        upsampler=None,
        target_width=640,
    )
    assert comp_mismatch.shape == (322, 640, 3)

    # Check that banner contains prominent red pixels for the mismatch flag
    banner = comp_mismatch[288:, :]
    red_pixels = (banner[:, :, 0] > 200) & (banner[:, :, 1] < 100) & (banner[:, :, 2] < 100)
    assert np.any(red_pixels), "Expected red pixels in banner for map mismatch flag"

    # In matching map, there should be no bright red error flag text
    banner_match = comp_match[288:, :]
    red_pixels_match = (banner_match[:, :, 0] > 200) & (banner_match[:, :, 1] < 100) & (banner_match[:, :, 2] < 100)
    assert not np.any(red_pixels_match), "Matching map should not contain red mismatch flag"


def test_dream_gif_generation_toy(tmp_path):
    """Test generate_dream_gif end-to-end with mock model and probe."""
    T = 12
    toy_traj_path = tmp_path / "toy_traj.npz"
    np.savez(
        toy_traj_path,
        frames=np.zeros((T, 72, 80), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.uint8),
        map_ids=np.full(T, 2, dtype=np.uint8),
        xs=np.full(T, 10, dtype=np.uint8),
        ys=np.full(T, 15, dtype=np.uint8),
        in_battles=np.zeros(T, dtype=bool),
        goals=np.array(["gym"] * T, dtype=str),
    )

    # Mock Denoiser rollout
    class MockModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def rollout(self, ctx, actions, steps=1, aug_bucket=0):
            K = actions.shape[1] - CTX + 1
            return torch.zeros((1, K, 72, 80), dtype=torch.float32)

    # Mock Probe
    class MockProbe:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, x):
            B = x.shape[0]
            return {
                "map": torch.zeros((B, 256)),
                "x": torch.zeros((B, 64)),
                "y": torch.zeros((B, 80)),
                "battle": torch.zeros((B, 1)),
            }

    out_gif = tmp_path / "toy_dream.gif"
    res_path = generate_dream_gif(
        traj_path=toy_traj_path,
        k=4,
        start=4,
        length=8,
        out=out_gif,
        every=1,
        device="cpu",
        steps=1,
        model=MockModel(),
        probe=MockProbe(),
        upsampler=None,
    )

    assert res_path.exists()
    assert res_path.stat().st_size > 0
    im = Image.open(res_path)
    assert im.format == "GIF"
    assert im.n_frames == 8


def test_play_get_overworld_windows_start_map():
    """Verify get_overworld_windows filters accurately by start-map on extended dataset."""
    data, splits = load(extended=True)

    # Test Pewter City (2)
    wins_pewter = get_overworld_windows(data, splits["test_ext"], map_id=2)
    assert len(wins_pewter) > 0
    ar = np.arange(CTX)
    maps_pewter = data["map_ids"][wins_pewter[:, None] + ar]
    assert np.all(maps_pewter == 2), "All context frames must be on map 2 (Pewter City)"

    # Test Viridian Forest (51)
    wins_forest = get_overworld_windows(data, splits["test_ext"], map_id=51)
    assert len(wins_forest) > 0
    maps_forest = data["map_ids"][wins_forest[:, None] + ar]
    assert np.all(maps_forest == 51), "All context frames must be on map 51 (Viridian Forest)"

    # Test Route 2 (13)
    wins_route2 = get_overworld_windows(data, splits["test_ext"], map_id=13)
    assert len(wins_route2) > 0
    maps_route2 = data["map_ids"][wins_route2[:, None] + ar]
    assert np.all(maps_route2 == 13), "All context frames must be on map 13 (Route 2)"
