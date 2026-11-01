"""Tests for lab.dwm.eval_rollouts."""
import os
import sys

# Prevent OpenMP runtime collision between PyTorch, MKL, and OpenCV
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure Conda Library\bin is accessible for MKL / OpenBLAS on Windows
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

from pathlib import Path

import numpy as np
import pytest
import torch

from lab.data import load
from lab.dwm.eval_rollouts import (
    evaluate_split,
    labels_for,
    plot_drift,
    plot_grid,
    plot_mapacc,
    plot_mse,
    print_compact_table,
    summarize,
)
from lab.dwm.model import CTX, Denoiser
from lab.dwm.train import windows
from lab.eval.metrics import manhattan


def test_alignment_pure_numpy():
    """Construct fake data where xs increases by 1 each step; check alignment."""
    N = 30
    xs = np.arange(N, dtype=np.uint8)
    ys = np.zeros(N, dtype=np.uint8)
    map_ids = np.ones(N, dtype=np.uint8)
    in_battles = np.zeros(N, dtype=bool)
    frames = np.zeros((N, 72, 80), dtype=np.uint8)
    actions = np.zeros(N, dtype=np.uint8)
    episode_id = np.zeros(N, dtype=np.int64)

    fake_data = {
        "xs": xs,
        "ys": ys,
        "map_ids": map_ids,
        "in_battles": in_battles,
        "frames": frames,
        "actions": actions,
        "episode_id": episode_id,
    }

    starts = np.array([0, 2, 5])
    K = 4

    labels = labels_for(fake_data, starts, K)

    # Check true_xy[:, k-1] equals xs at s + CTX - 1 + k
    for k in range(1, K + 1):
        expected_tk = starts + CTX - 1 + k
        np.testing.assert_array_equal(labels["true_xy"][:, k - 1, 0], xs[expected_tk])
        np.testing.assert_array_equal(labels["true_xy"][:, k - 1, 1], ys[expected_tk])

    # Check xy0 at t_0 = s + CTX - 1
    expected_t0 = starts + CTX - 1
    np.testing.assert_array_equal(labels["xy0"][:, 0], xs[expected_t0])
    np.testing.assert_array_equal(labels["xy0"][:, 1], ys[expected_t0])

    # Check pos_err_nochange[:, k-1] == k
    for k in range(1, K + 1):
        np.testing.assert_array_equal(labels["pos_err_nochange"][:, k - 1], np.full(len(starts), k))


def test_nan_masking_battle_and_map_change():
    """Verify NaN masking where true_battle is True or map changed."""
    N = 25
    xs = np.arange(N, dtype=np.uint8)
    ys = np.zeros(N, dtype=np.uint8)
    map_ids = np.ones(N, dtype=np.uint8)
    in_battles = np.zeros(N, dtype=bool)

    # Window 1: battle at step s + CTX - 1 + 2
    # Window 2: map change at step s + CTX - 1 + 3
    starts = np.array([0, 5, 10])
    K = 4

    # Trigger battle in window 1 at k=2
    in_battles[starts[1] + CTX - 1 + 2] = True
    # Trigger map change in window 2 at k=3
    map_ids[starts[2] + CTX - 1 + 3] = 99

    fake_data = {
        "xs": xs,
        "ys": ys,
        "map_ids": map_ids,
        "in_battles": in_battles,
    }

    labels = labels_for(fake_data, starts, K)
    pos_err = labels["pos_err_nochange"]

    # Window 0: all finite
    assert np.all(np.isfinite(pos_err[0]))

    # Window 1: k=2 is NaN (index 1), others finite
    assert np.isnan(pos_err[1, 1])
    assert np.all(np.isfinite(pos_err[1, [0, 2, 3]]))

    # Window 2: k=3 is NaN (index 2), others finite
    assert np.isnan(pos_err[2, 2])
    assert np.all(np.isfinite(pos_err[2, [0, 1, 3]]))


def test_evaluate_split_and_summarize_cpu():
    """Evaluate 4 windows on CPU with tiny sizes and verify shapes, NaNs, and summary."""
    probe_mod = pytest.importorskip("lab.probe")
    FrameProbe = probe_mod.FrameProbe

    data, splits = load()
    all_starts = windows(data, splits["test_id"], CTX + 3)
    assert len(all_starts) >= 4
    starts = all_starts[:4]
    K = 3

    model = Denoiser()
    probe = FrameProbe()

    per_window = evaluate_split(
        model=model,
        probe=probe,
        data=data,
        starts=starts,
        K=K,
        steps=1,
        aug_bucket=0,
        device="cpu",
        batch=4,
    )

    # Check shapes of all per-window arrays
    expected_shapes_wk = [
        "true_map", "true_battle", "img_map", "img_x", "img_y", "img_battle",
        "real_map", "real_x", "real_y", "real_battle", "map_changed",
        "pos_err_img", "pos_err_real", "pos_err_nochange", "pos_agree",
        "map_acc_img", "map_acc_real", "map_acc_nochange",
        "battle_acc_img", "battle_acc_nochange",
        "mse_img", "mse_copy", "moved",
    ]
    for key in expected_shapes_wk:
        assert key in per_window, f"Missing key {key}"
        assert per_window[key].shape == (4, K), f"Wrong shape for {key}: {per_window[key].shape}"

    assert per_window["true_xy"].shape == (4, K, 2)
    assert per_window["img_xy"].shape == (4, K, 2)
    assert per_window["real_xy"].shape == (4, K, 2)
    assert per_window["xy0"].shape == (4, 2)
    assert per_window["map0"].shape == (4,)
    assert per_window["battle0"].shape == (4,)
    assert per_window["imagined_frames"].shape == (4, K, 72, 80)
    assert per_window["real_frames"].shape == (4, K, 72, 80)

    # Check NaN masking consistency
    pos_mask = per_window["true_battle"] | per_window["map_changed"]
    assert np.all(np.isnan(per_window["pos_err_img"][pos_mask]))
    assert np.all(np.isnan(per_window["pos_err_real"][pos_mask]))
    assert np.all(np.isnan(per_window["pos_err_nochange"][pos_mask]))
    assert np.all(np.isnan(per_layer := per_window["pos_agree"])[pos_mask])

    bat_mask = per_window["true_battle"]
    assert np.all(np.isnan(per_window["map_acc_img"][bat_mask]))
    assert np.all(np.isnan(per_window["map_acc_real"][bat_mask]))
    assert np.all(np.isnan(per_window["map_acc_nochange"][bat_mask]))

    # Test summarize
    episode_ids = data["episode_id"][starts]
    summary = summarize(per_window, episode_ids)

    assert summary["n_windows"] == 4
    assert summary["n_episodes"] >= 1
    assert 0.0 <= summary["valid_pos_fraction"] <= 1.0

    # Verify finite numbers for no-change baseline
    nc_metric = summary["metrics"]["pos_err_nochange"]
    assert np.all(np.isfinite(nc_metric["mean"]))
    assert np.all(np.isfinite(nc_metric["lo"]))
    assert np.all(np.isfinite(nc_metric["hi"]))

    assert np.all(np.isfinite(summary["metrics"]["map_acc_nochange"]["mean"]))
    assert np.all(np.isfinite(summary["metrics"]["battle_acc_nochange"]["mean"]))
    assert np.all(np.isfinite(summary["metrics"]["mse_copy"]["mean"]))
    assert np.all(np.isfinite(summary["psnr"]["psnr_copy"]))


def test_plot_generation_and_table(tmp_path):
    """Test plotting functions and compact table printing."""
    probe_mod = pytest.importorskip("lab.probe")
    FrameProbe = probe_mod.FrameProbe

    data, splits = load()
    all_starts = windows(data, splits["test_id"], CTX + 3)
    starts = all_starts[:4]
    K = 3

    model = Denoiser()
    probe = FrameProbe()

    per_window = evaluate_split(
        model=model,
        probe=probe,
        data=data,
        starts=starts,
        K=K,
        steps=1,
        aug_bucket=0,
        device="cpu",
        batch=4,
    )
    episode_ids = data["episode_id"][starts]
    summary = summarize(per_window, episode_ids)

    # Compact table should execute without errors
    print_compact_table(summary, "test_id")

    # Plots should be created successfully
    drift_png = tmp_path / "drift.png"
    mapacc_png = tmp_path / "mapacc.png"
    mse_png = tmp_path / "mse.png"
    grid_png = tmp_path / "grid.png"

    plot_drift(summary, "test_id", "test_stem", drift_png)
    plot_mapacc(summary, "test_id", "test_stem", mapacc_png)
    plot_mse(summary, "test_id", "test_stem", mse_png)
    plot_grid(data, starts, per_window, K, grid_png)

    assert drift_png.exists() and drift_png.stat().st_size > 0
    assert mapacc_png.exists() and mapacc_png.stat().st_size > 0
    assert mse_png.exists() and mse_png.stat().st_size > 0
    assert grid_png.exists() and grid_png.stat().st_size > 0
