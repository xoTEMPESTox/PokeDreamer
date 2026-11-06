"""RAM-grounded evaluation of diffusion world model imagined rollouts.

Measures how a world model imagines Pokemon Red futures using game RAM
(map_id, player x/y tile, in_battle flag) as ground truth.
"""
import argparse
import json
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float
from lab.dwm.train import windows
from lab.eval.metrics import (
    bootstrap_ci,
    copy_last_frame_mse,
    frame_mse,
    manhattan,
    psnr,
)

try:
    from lab.probe import FrameProbe, load_probe, predict
except ImportError:
    FrameProbe = None
    load_probe = None
    predict = None


def labels_for(data: dict, starts: np.ndarray, K: int) -> dict:
    """Extract ground-truth RAM labels for windows starting at starts with horizon K."""
    starts = np.asarray(starts)
    t0 = starts + CTX - 1
    tk = starts[:, None] + CTX - 1 + np.arange(1, K + 1)[None, :]

    xy0 = np.stack([data["xs"][t0], data["ys"][t0]], axis=-1)
    map0 = np.asarray(data["map_ids"][t0])
    battle0 = np.asarray(data["in_battles"][t0], dtype=bool)

    true_xy = np.stack([data["xs"][tk], data["ys"][tk]], axis=-1)
    true_map = np.asarray(data["map_ids"][tk])
    true_battle = np.asarray(data["in_battles"][tk], dtype=bool)

    map_changed = (true_map != map0[:, None])
    pos_invalid = true_battle | map_changed

    pos_err_nochange = manhattan(xy0[:, None, :], true_xy).astype(float)
    pos_err_nochange[pos_invalid] = np.nan

    return {
        "xy0": xy0,
        "map0": map0,
        "battle0": battle0,
        "true_xy": true_xy,
        "true_map": true_map,
        "true_battle": true_battle,
        "map_changed": map_changed,
        "pos_err_nochange": pos_err_nochange,
    }


def evaluate_split(
    model: Denoiser,
    probe,
    data: dict,
    starts: np.ndarray,
    K: int,
    steps: int = 3,
    aug_bucket: int = 0,
    device: str = "cpu",
    batch: int = 128,
) -> dict:
    """Run model rollouts and evaluate against RAM and probe readings.

    Returns dict of per-window numpy arrays of shape (W, K) unless noted.
    """
    starts = np.asarray(starts)
    W = len(starts)

    if probe is None or predict is None:
        raise RuntimeError("lab.probe is required for evaluate_split")

    model.eval()
    model.to(device)

    imagined_list = []
    real_list = []

    for i in range(0, W, batch):
        batch_starts = starts[i : i + batch]

        ctx_idx = batch_starts[:, None] + np.arange(CTX)[None, :]
        ctx_raw = data["frames"][ctx_idx]
        ctx_tensor = to_float(torch.from_numpy(ctx_raw).to(device))

        act_idx = batch_starts[:, None] + np.arange(CTX + K - 1)[None, :]
        act_tensor = torch.from_numpy(data["actions"][act_idx]).long().to(device)

        real_idx = batch_starts[:, None] + (CTX + np.arange(K))[None, :]
        real_raw = data["frames"][real_idx]
        real_list.append(real_raw)

        with torch.no_grad():
            if "cuda" in str(device):
                with torch.autocast("cuda", torch.bfloat16):
                    img_batch = model.rollout(ctx_tensor, act_tensor, steps=steps, aug_bucket=aug_bucket)
            else:
                img_batch = model.rollout(ctx_tensor, act_tensor, steps=steps, aug_bucket=aug_bucket)

        imagined_list.append(img_batch.cpu().float().numpy())

    if W > 0:
        imagined = np.concatenate(imagined_list, axis=0)
        real_frames = np.concatenate(real_list, axis=0)
    else:
        imagined = np.empty((0, K, 72, 80), dtype=np.float32)
        real_frames = np.empty((0, K, 72, 80), dtype=np.uint8)

    # Convert imagined [-1, 1] to [0, 1] for probe & MSE
    imagined_01 = (imagined + 1.0) / 2.0
    imagined_u8 = np.clip(np.round(imagined_01 * 255.0), 0, 255).astype(np.uint8)

    # Probe on imagined frames
    if W > 0:
        img_preds = predict(probe, imagined_01.reshape(-1, 72, 80), device=device, batch=1024)
        img_map = img_preds["map_id"].reshape(W, K)
        img_x = img_preds["x"].reshape(W, K)
        img_y = img_preds["y"].reshape(W, K)
        img_battle = img_preds["battle"].reshape(W, K)
        img_xy = np.stack([img_x, img_y], axis=-1)

        # Probe on real frames at t_k
        real_preds = predict(probe, real_frames.reshape(-1, 72, 80), device=device, batch=1024)
        real_map = real_preds["map_id"].reshape(W, K)
        real_x = real_preds["x"].reshape(W, K)
        real_y = real_preds["y"].reshape(W, K)
        real_battle = real_preds["battle"].reshape(W, K)
        real_xy = np.stack([real_x, real_y], axis=-1)
    else:
        img_map = np.empty((0, K), dtype=int)
        img_x = np.empty((0, K), dtype=int)
        img_y = np.empty((0, K), dtype=int)
        img_battle = np.empty((0, K), dtype=bool)
        img_xy = np.empty((0, K, 2), dtype=int)

        real_map = np.empty((0, K), dtype=int)
        real_x = np.empty((0, K), dtype=int)
        real_y = np.empty((0, K), dtype=int)
        real_battle = np.empty((0, K), dtype=bool)
        real_xy = np.empty((0, K, 2), dtype=int)

    # RAM labels
    labels = labels_for(data, starts, K)
    xy0 = labels["xy0"]
    map0 = labels["map0"]
    battle0 = labels["battle0"]
    true_xy = labels["true_xy"]
    true_map = labels["true_map"]
    true_battle = labels["true_battle"]
    map_changed = labels["map_changed"]

    # Position errors and agreement
    pos_mask = true_battle | map_changed
    pos_err_img = manhattan(img_xy, true_xy).astype(float)
    pos_err_real = manhattan(real_xy, true_xy).astype(float)
    pos_err_nochange = labels["pos_err_nochange"].copy()
    pos_agree = manhattan(img_xy, real_xy).astype(float)

    pos_err_img[pos_mask] = np.nan
    pos_err_real[pos_mask] = np.nan
    pos_err_nochange[pos_mask] = np.nan
    pos_agree[pos_mask] = np.nan

    # Map accuracy (NaN where true_battle)
    map_acc_img = (img_map == true_map).astype(float)
    map_acc_real = (real_map == true_map).astype(float)
    map_acc_nochange = (map0[:, None] == true_map).astype(float)

    map_acc_img[true_battle] = np.nan
    map_acc_real[true_battle] = np.nan
    map_acc_nochange[true_battle] = np.nan

    # Wrong-map rate (NaN where true_battle)
    wrong_map_img = (img_map != true_map).astype(float)
    wrong_map_img[true_battle] = np.nan

    # Battle accuracy
    battle_acc_img = (img_battle == true_battle).astype(float)
    battle_acc_nochange = (battle0[:, None] == true_battle).astype(float)

    # Frame MSE in [0, 1] pixel units
    real_k_01 = real_frames.astype(np.float32) / 255.0
    if W > 0:
        ctx_0_01 = data["frames"][starts + CTX - 1].astype(np.float32) / 255.0
        real_all_01 = np.concatenate([ctx_0_01[:, None], real_k_01], axis=1)
        mse_copy = copy_last_frame_mse(real_all_01)
    else:
        mse_copy = np.empty((0, K), dtype=float)

    mse_img = frame_mse(imagined_01, real_k_01)

    # Moved flag (no-change error >= 1 on valid unmasked positions)
    moved = (~np.isnan(pos_err_nochange)) & (pos_err_nochange >= 1.0)

    return {
        "true_xy": true_xy,
        "true_map": true_map,
        "true_battle": true_battle,
        "xy0": xy0,
        "map0": map0,
        "battle0": battle0,
        "img_map": img_map,
        "img_x": img_x,
        "img_y": img_y,
        "img_battle": img_battle,
        "img_xy": img_xy,
        "real_map": real_map,
        "real_x": real_x,
        "real_y": real_y,
        "real_battle": real_battle,
        "real_xy": real_xy,
        "map_changed": map_changed,
        "pos_err_img": pos_err_img,
        "pos_err_real": pos_err_real,
        "pos_err_nochange": pos_err_nochange,
        "pos_agree": pos_agree,
        "map_acc_img": map_acc_img,
        "map_acc_real": map_acc_real,
        "map_acc_nochange": map_acc_nochange,
        "wrong_map_img": wrong_map_img,
        "battle_acc_img": battle_acc_img,
        "battle_acc_nochange": battle_acc_nochange,
        "mse_img": mse_img,
        "mse_copy": mse_copy,
        "moved": moved,
        "imagined_frames": imagined_u8,
        "real_frames": real_frames,
    }


def summarize(per_window: dict, episode_ids: np.ndarray) -> dict:
    """Cluster bootstrap by episode and summarize evaluation metrics.

    Average per-window values within each episode (nanmean) -> (E, K),
    then bootstrap_ci over episodes.
    """
    episode_ids = np.asarray(episode_ids)
    unique_eps = np.unique(episode_ids)
    E = len(unique_eps)
    W = len(episode_ids)

    K = per_window["pos_err_img"].shape[1] if W > 0 else 0

    moved = per_window["moved"]
    pos_err_img_moved = np.where(moved, per_window["pos_err_img"], np.nan)
    pos_err_nochange_moved = np.where(moved, per_window["pos_err_nochange"], np.nan)
    pos_err_real_moved = np.where(moved, per_window["pos_err_real"], np.nan)
    pos_agree_moved = np.where(moved, per_window["pos_agree"], np.nan)

    raw_metrics = {
        "pos_err_img": per_window["pos_err_img"],
        "pos_err_nochange": per_window["pos_err_nochange"],
        "pos_err_real": per_window["pos_err_real"],
        "pos_agree": per_window["pos_agree"],
        "pos_err_img_moved": pos_err_img_moved,
        "pos_err_nochange_moved": pos_err_nochange_moved,
        "pos_err_real_moved": pos_err_real_moved,
        "pos_agree_moved": pos_agree_moved,
        "map_acc_img": per_window["map_acc_img"],
        "map_acc_real": per_window["map_acc_real"],
        "map_acc_nochange": per_window["map_acc_nochange"],
        "wrong_map_img": per_window.get("wrong_map_img", 1.0 - per_window["map_acc_img"]),
        "battle_acc_img": per_window["battle_acc_img"],
        "battle_acc_nochange": per_window["battle_acc_nochange"],
        "mse_img": per_window["mse_img"],
        "mse_copy": per_window["mse_copy"],
    }

    metrics_summary = {}

    for name, arr in raw_metrics.items():
        if E > 0 and K > 0:
            ep_arr = np.empty((E, K), dtype=float)
            for e_idx, ep in enumerate(unique_eps):
                mask = (episode_ids == ep)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ep_arr[e_idx] = np.nanmean(arr[mask], axis=0)

            mean, lo, hi = bootstrap_ci(ep_arr, n_boot=1000, seed=0)
        else:
            mean = np.full(K, np.nan)
            lo = np.full(K, np.nan)
            hi = np.full(K, np.nan)

        metrics_summary[name] = {
            "mean": mean.tolist(),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
        }

    # PSNR of MSE means
    if K > 0:
        mean_mse_img = np.array(metrics_summary["mse_img"]["mean"])
        mean_mse_copy = np.array(metrics_summary["mse_copy"]["mean"])
        psnr_img = psnr(mean_mse_img, max_val=1.0)
        psnr_copy = psnr(mean_mse_copy, max_val=1.0)
    else:
        psnr_img = np.array([])
        psnr_copy = np.array([])

    reported_ks = [k for k in [1, 2, 4, 8, 16, 32] if k <= K]

    # Valid fraction for position metrics
    if W > 0 and K > 0:
        valid_pos = ~np.isnan(per_window["pos_err_nochange"])
        valid_pos_frac = float(np.mean(valid_pos))
        valid_pos_frac_k = np.mean(valid_pos, axis=0).tolist()
    else:
        valid_pos_frac = 0.0
        valid_pos_frac_k = []

    # Attach at_k slices for convenience
    for name in metrics_summary:
        m = metrics_summary[name]["mean"]
        l = metrics_summary[name]["lo"]
        h = metrics_summary[name]["hi"]
        metrics_summary[name]["at_k"] = {
            str(k): {
                "mean": float(m[k - 1]),
                "lo": float(l[k - 1]),
                "hi": float(h[k - 1]),
            }
            for k in reported_ks
        }

    psnr_dict = {
        "psnr_img": psnr_img.tolist(),
        "psnr_copy": psnr_copy.tolist(),
        "at_k": {
            str(k): {
                "psnr_img": float(psnr_img[k - 1]),
                "psnr_copy": float(psnr_copy[k - 1]),
            }
            for k in reported_ks
        },
    }

    return {
        "n_windows": int(W),
        "n_episodes": int(E),
        "valid_pos_fraction": valid_pos_frac,
        "valid_pos_fraction_k": valid_pos_frac_k,
        "reported_k": reported_ks,
        "metrics": metrics_summary,
        "psnr": psnr_dict,
    }


def print_compact_table(summary: dict, split_name: str):
    """Print compact summary table to stdout."""
    reported_ks = summary["reported_k"]
    header_ks = "".join(f"{f'k={k}':>20}" for k in reported_ks)
    tot_w = 32 + 20 * len(reported_ks)
    print("=" * tot_w)
    print(f"EVALUATION RESULTS: split={split_name} | windows={summary['n_windows']} | "
          f"episodes={summary['n_episodes']} | valid_pos_fraction={summary['valid_pos_fraction']:.1%}")
    print("=" * tot_w)
    print(f"{'Metric':<32}{header_ks}")
    print("-" * tot_w)

    metric_display = [
        ("pos_err_img", "pos_err_img"),
        ("pos_err_nochange", "pos_err_nochange"),
        ("pos_err_real", "pos_err_real"),
        ("pos_agree", "pos_agree"),
        ("pos_err_img_moved", "pos_err_img (moved)"),
        ("pos_err_nochange_moved", "pos_err_nochange (moved)"),
        ("pos_err_real_moved", "pos_err_real (moved)"),
        ("map_acc_img", "map_acc_img"),
        ("wrong_map_img", "wrong_map_rate"),
        ("map_acc_real", "map_acc_real"),
        ("map_acc_nochange", "map_acc_nochange"),
        ("battle_acc_img", "battle_acc_img"),
        ("battle_acc_nochange", "battle_acc_nochange"),
        ("mse_img", "mse_img"),
        ("mse_copy", "mse_copy"),
    ]

    for key, display_name in metric_display:
        if key not in summary["metrics"]:
            continue
        row_str = f"{display_name:<32}"
        for k in reported_ks:
            val_info = summary["metrics"][key]["at_k"][str(k)]
            m = val_info["mean"]
            lo = val_info["lo"]
            hi = val_info["hi"]
            if np.isnan(m):
                cell = "nan"
            else:
                if "acc" in key or "wrong_map" in key:
                    cell = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
                elif "mse" in key:
                    cell = f"{m:.4f} [{lo:.4f},{hi:.4f}]"
                else:
                    cell = f"{m:.2f} [{lo:.2f},{hi:.2f}]"
            row_str += f"{cell:>20}"
        print(row_str)

    if "psnr" in summary and "at_k" in summary["psnr"]:
        for psnr_key, display_name in [("psnr_img", "psnr_img (dB)"), ("psnr_copy", "psnr_copy (dB)")]:
            row_str = f"{display_name:<32}"
            for k in reported_ks:
                val = summary["psnr"]["at_k"][str(k)][psnr_key]
                cell = f"{val:.2f}" if not np.isnan(val) else "nan"
                row_str += f"{cell:>20}"
            print(row_str)

    print("=" * tot_w)


def plot_drift(summary: dict, split_name: str, stem: str, out_path: Path):
    """Plot position error (tiles) vs k with shaded 95% CI."""
    k_vals = np.arange(1, len(summary["metrics"]["pos_err_img"]["mean"]) + 1)
    n_win = summary["n_windows"]
    n_ep = summary["n_episodes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Panel 1: All valid windows
    ax = axes[0]
    m_img = summary["metrics"]["pos_err_img"]
    m_nc = summary["metrics"]["pos_err_nochange"]
    m_real = summary["metrics"]["pos_err_real"]

    ax.plot(k_vals, m_img["mean"], label="Imagined (world model)", color="#1f77b4", lw=2)
    ax.fill_between(k_vals, m_img["lo"], m_img["hi"], color="#1f77b4", alpha=0.25)

    ax.plot(k_vals, m_nc["mean"], label="No-change baseline", color="#ff7f0e", lw=2, linestyle="--")
    ax.fill_between(k_vals, m_nc["lo"], m_nc["hi"], color="#ff7f0e", alpha=0.25)

    ax.plot(k_vals, m_real["mean"], label="Real probe ceiling", color="#2ca02c", lw=2, linestyle=":")
    ax.fill_between(k_vals, m_real["lo"], m_real["hi"], color="#2ca02c", alpha=0.25)

    ax.set_title("All Valid Windows")
    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Position Error (Manhattan tiles)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Moved subset
    ax = axes[1]
    m_img_m = summary["metrics"]["pos_err_img_moved"]
    m_nc_m = summary["metrics"]["pos_err_nochange_moved"]
    m_real_m = summary["metrics"]["pos_err_real_moved"]

    ax.plot(k_vals, m_img_m["mean"], label="Imagined (world model)", color="#1f77b4", lw=2)
    ax.fill_between(k_vals, m_img_m["lo"], m_img_m["hi"], color="#1f77b4", alpha=0.25)

    ax.plot(k_vals, m_nc_m["mean"], label="No-change baseline", color="#ff7f0e", lw=2, linestyle="--")
    ax.fill_between(k_vals, m_nc_m["lo"], m_nc_m["hi"], color="#ff7f0e", alpha=0.25)

    ax.plot(k_vals, m_real_m["mean"], label="Real probe ceiling", color="#2ca02c", lw=2, linestyle=":")
    ax.fill_between(k_vals, m_real_m["lo"], m_real_m["hi"], color="#2ca02c", alpha=0.25)

    ax.set_title("Moved Subset (no-change error >= 1)")
    ax.set_xlabel("Horizon k (steps)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Position Error vs Horizon k | Split: {split_name} (Windows: {n_win}, Episodes: {n_ep})", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mapacc(summary: dict, split_name: str, stem: str, out_path: Path):
    """Plot map accuracy vs k with shaded 95% CI."""
    k_vals = np.arange(1, len(summary["metrics"]["map_acc_img"]["mean"]) + 1)
    n_win = summary["n_windows"]
    n_ep = summary["n_episodes"]

    fig, ax = plt.subplots(figsize=(8, 5))
    m_img = summary["metrics"]["map_acc_img"]
    m_real = summary["metrics"]["map_acc_real"]
    m_nc = summary["metrics"]["map_acc_nochange"]

    ax.plot(k_vals, m_img["mean"], label="Imagined (world model)", color="#1f77b4", lw=2)
    ax.fill_between(k_vals, m_img["lo"], m_img["hi"], color="#1f77b4", alpha=0.25)

    ax.plot(k_vals, m_real["mean"], label="Real probe ceiling", color="#2ca02c", lw=2, linestyle=":")
    ax.fill_between(k_vals, m_real["lo"], m_real["hi"], color="#2ca02c", alpha=0.25)

    ax.plot(k_vals, m_nc["mean"], label="No-change baseline", color="#ff7f0e", lw=2, linestyle="--")
    ax.fill_between(k_vals, m_nc["lo"], m_nc["hi"], color="#ff7f0e", alpha=0.25)

    ax.set_title(f"Map Accuracy vs Horizon k | Split: {split_name} (Windows: {n_win}, Episodes: {n_ep})")
    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Map Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mse(summary: dict, split_name: str, stem: str, out_path: Path):
    """Plot pixel MSE vs k with shaded 95% CI."""
    k_vals = np.arange(1, len(summary["metrics"]["mse_img"]["mean"]) + 1)
    n_win = summary["n_windows"]
    n_ep = summary["n_episodes"]

    fig, ax = plt.subplots(figsize=(8, 5))
    m_img = summary["metrics"]["mse_img"]
    m_cp = summary["metrics"]["mse_copy"]

    ax.plot(k_vals, m_img["mean"], label="Imagined (world model)", color="#1f77b4", lw=2)
    ax.fill_between(k_vals, m_img["lo"], m_img["hi"], color="#1f77b4", alpha=0.25)

    ax.plot(k_vals, m_cp["mean"], label="Copy-last baseline", color="#ff7f0e", lw=2, linestyle="--")
    ax.fill_between(k_vals, m_cp["lo"], m_cp["hi"], color="#ff7f0e", alpha=0.25)

    ax.set_title(f"Pixel MSE vs Horizon k | Split: {split_name} (Windows: {n_win}, Episodes: {n_ep})")
    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Pixel MSE (in [0, 1])")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grid(data: dict, starts: np.ndarray, per_window: dict, K: int, out_path: Path):
    """Save 8-window comparison grid (real vs imagined) upscaled 3x nearest."""
    W = len(starts)
    if W == 0 or "imagined_frames" not in per_window:
        return

    k_check = min(8, K)
    k_check_idx = k_check - 1

    battle0 = per_window["battle0"]
    moved = per_window["moved"]

    cands_p1 = [w for w in range(W) if not battle0[w] and (moved[w, k_check_idx] if K >= 1 else True)]
    cands_p2 = [w for w in range(W) if not battle0[w] and w not in cands_p1]
    cands_p3 = [w for w in range(W) if w not in cands_p1 and w not in cands_p2]

    selected_w = (cands_p1 + cands_p2 + cands_p3)[:min(8, W)]
    if not selected_w:
        return

    horizons = [k for k in [1, 2, 4, 8, 16, 32] if k <= K]

    sep_v_w = 2
    sep_h_thin = 2
    sep_h_thick = 6

    frame_h, frame_w = 72 * 3, 80 * 3

    window_blocks = []
    for w in selected_w:
        s = starts[w]
        t0 = s + CTX - 1
        ctx_frame = data["frames"][t0]
        ctx_up = cv2.resize(ctx_frame, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        real_cols = [ctx_up]
        img_cols = [ctx_up]

        for k in horizons:
            rf = data["frames"][s + CTX - 1 + k]
            rf_up = cv2.resize(rf, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            real_cols.append(rf_up)

            imf = per_window["imagined_frames"][w, k - 1]
            imf_up = cv2.resize(imf, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            img_cols.append(imf_up)

        v_sep = np.full((frame_h, sep_v_w), 128, dtype=np.uint8)

        row_real_parts = []
        row_img_parts = []
        for i in range(len(real_cols)):
            if i > 0:
                row_real_parts.append(v_sep)
                row_img_parts.append(v_sep)
            row_real_parts.append(real_cols[i])
            row_img_parts.append(img_cols[i])

        row_real = np.concatenate(row_real_parts, axis=1)
        row_img = np.concatenate(row_img_parts, axis=1)

        h_sep_thin = np.full((sep_h_thin, row_real.shape[1]), 80, dtype=np.uint8)
        win_block = np.concatenate([row_real, h_sep_thin, row_img], axis=0)
        window_blocks.append(win_block)

    h_sep_thick = np.full((sep_h_thick, window_blocks[0].shape[1]), 255, dtype=np.uint8)
    all_rows = []
    for i, wb in enumerate(window_blocks):
        if i > 0:
            all_rows.append(h_sep_thick)
        all_rows.append(wb)

    full_grid = np.concatenate(all_rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full_grid)


MAP_NAMES_OVERRIDE = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    12: "Route 1",
    13: "Route 2",
    41: "Viridian Pokecenter",
    42: "Viridian Mart",
    47: "Forest North Gate",
    50: "Forest South Gate",
    51: "Viridian Forest",
    54: "Pewter Gym",
    58: "Pewter Pokecenter",
}


def get_map_name(map_id: int) -> str:
    return MAP_NAMES_OVERRIDE.get(int(map_id), f"Map {map_id}")


def summarize_per_map(per_window: dict, episode_ids: np.ndarray, maps=(13, 51, 2, 54)) -> dict:
    """Compute episode-bootstrap evaluation metrics separately for specified map IDs."""
    episode_ids = np.asarray(episode_ids)
    map0 = np.asarray(per_window["map0"])
    per_map = {}

    for m in maps:
        mask = (map0 == m)
        n_win = int(np.sum(mask))
        m_name = get_map_name(m)

        if n_win == 0:
            per_map[str(m)] = {
                "map_id": int(m),
                "map_name": m_name,
                "n_windows": 0,
                "n_episodes": 0,
                "metrics": {},
            }
            continue

        sub_pw = {}
        for k, v in per_window.items():
            if isinstance(v, np.ndarray) and len(v) == len(map0):
                sub_pw[k] = v[mask]
            else:
                sub_pw[k] = v

        sub_eps = episode_ids[mask]
        sub_summary = summarize(sub_pw, sub_eps)
        sub_summary["map_id"] = int(m)
        sub_summary["map_name"] = m_name
        per_map[str(m)] = sub_summary

    return per_map


def print_per_map_table(per_map: dict):
    """Print per-map summary table to stdout."""
    for map_key, m_info in per_map.items():
        n_win = m_info["n_windows"]
        n_ep = m_info["n_episodes"]
        m_name = m_info.get("map_name", f"Map {map_key}")
        if n_win == 0:
            print(f"\n--- PER-MAP EVALUATION: Map {map_key} ({m_name}) | 0 windows ---")
            continue
        reported_ks = m_info.get("reported_k", [1, 2, 4, 8, 16, 32])
        header_ks = "".join(f"{f'k={k}':>20}" for k in reported_ks)
        tot_w = 32 + 20 * len(reported_ks)
        print("=" * tot_w)
        print(f"PER-MAP EVALUATION: Map {map_key} ({m_name}) | windows={n_win} | episodes={n_ep}")
        print("=" * tot_w)
        print(f"{'Metric':<32}{header_ks}")
        print("-" * tot_w)

        map_metrics = [
            ("wrong_map_img", "wrong_map_rate"),
            ("pos_err_img", "pos_err_img"),
            ("pos_err_nochange", "pos_err_nochange"),
        ]
        for key, display_name in map_metrics:
            if key not in m_info.get("metrics", {}):
                continue
            row_str = f"{display_name:<32}"
            for k in reported_ks:
                k_str = str(k)
                if k_str in m_info["metrics"][key].get("at_k", {}):
                    val_info = m_info["metrics"][key]["at_k"][k_str]
                    m = val_info["mean"]
                    lo = val_info["lo"]
                    hi = val_info["hi"]
                    if np.isnan(m):
                        cell = "nan"
                    elif "wrong_map" in key:
                        cell = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
                    else:
                        cell = f"{m:.2f} [{lo:.2f},{hi:.2f}]"
                else:
                    cell = "nan"
                row_str += f"{cell:>20}"
            print(row_str)
        print("=" * tot_w)


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion world model imagined rollouts.")
    parser.add_argument("--run", type=str, default="main", help="Run name in work/runs/<run>")
    parser.add_argument("--data", type=str, default="base", choices=["base", "ext"], help="Dataset to load: base or ext")
    parser.add_argument("--split", type=str, default="both", choices=["test_id", "test_ood", "both", "test_ext"], help="Split to evaluate")
    parser.add_argument("--probe", type=str, default=None, help="Path to probe checkpoint")
    parser.add_argument("--k", type=int, default=32, help="Rollout horizon K")
    parser.add_argument("--n-windows", type=int, default=512, help="Number of windows per split")
    parser.add_argument("--steps", type=int, default=3, help="Euler sampler diffusion steps")
    parser.add_argument("--aug-bucket", type=int, default=0, help="Context noise augmentation bucket")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for rollout evaluation")
    parser.add_argument("--maps", type=str, default="13,51,2,54", help="Comma-separated map ids for the per-map summary (ext data)")
    parser.add_argument("--tag", type=str, default="", help="Suffix for the output file names (keeps earlier outputs)")
    args = parser.parse_args()

    dev = args.device
    ckpt_path = Path("work/runs") / args.run / "ckpt.pt"
    is_ext = (args.data == "ext")
    if args.probe is not None:
        probe_path = Path(args.probe)
    else:
        probe_path = Path("work/probe_ext/probe.pt") if is_ext else Path("work/probe/probe.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"World model checkpoint not found: {ckpt_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")

    print(f"Loading world model from {ckpt_path} on {dev}...")
    model = Denoiser()
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
    model.load_state_dict(state["ema"])
    model.to(dev)
    model.eval()

    print(f"Loading probe from {probe_path} on {dev}...")
    probe = load_probe(str(probe_path), device=dev)

    print(f"Loading dataset (data={args.data})...")
    data, splits = load(extended=is_ext)

    splits_to_eval = ["test_id", "test_ood"] if args.split == "both" else [args.split]
    for s in splits_to_eval:
        if s not in splits:
            raise ValueError(f"Split '{s}' not found in splits: {list(splits.keys())}. Did you mean --data ext?")

    Path("work/eval").mkdir(parents=True, exist_ok=True)
    Path("work/plots").mkdir(parents=True, exist_ok=True)

    for split_name in splits_to_eval:
        print(f"\n--- Evaluating split: {split_name} (K={args.k}, steps={args.steps}, aug_bucket={args.aug_bucket}) ---")
        all_starts = windows(data, splits[split_name], CTX + args.k)
        rng = np.random.default_rng(0)
        n = min(len(all_starts), args.n_windows)
        starts = rng.choice(all_starts, size=n, replace=False)
        print(f"Sampled {len(starts)} windows from {len(all_starts)} eligible windows.")

        per_window = evaluate_split(
            model=model,
            probe=probe,
            data=data,
            starts=starts,
            K=args.k,
            steps=args.steps,
            aug_bucket=args.aug_bucket,
            device=dev,
            batch=args.batch,
        )

        episode_ids = data["episode_id"][starts]
        summary = summarize(per_window, episode_ids)

        print_compact_table(summary, split_name)

        if split_name == "test_ext" or is_ext:
            per_map = summarize_per_map(per_window, episode_ids, maps=[int(m) for m in args.maps.split(",")])
            summary["per_map"] = per_map
            print_per_map_table(per_map)

        stem = f"{args.run}_{split_name}_s{args.steps}_a{args.aug_bucket}" + ("_ext" if args.data == "ext" else "") + (f"_{args.tag}" if args.tag else "")  # base keeps the old names

        # Save per-window arrays in npz
        npz_path = Path("work/eval") / f"{stem}.npz"
        save_dict = {
            "starts": starts,
            "episode": episode_ids,
            "true_xy": per_window["true_xy"],
            "true_map": per_window["true_map"],
            "true_battle": per_window["true_battle"],
            "xy0": per_window["xy0"],
            "map0": per_window["map0"],
            "battle0": per_window["battle0"],
            "img_map": per_window["img_map"],
            "img_x": per_window["img_x"],
            "img_y": per_window["img_y"],
            "img_battle": per_window["img_battle"],
            "img_xy": per_window["img_xy"],
            "real_map": per_window["real_map"],
            "real_x": per_window["real_x"],
            "real_y": per_window["real_y"],
            "real_battle": per_window["real_battle"],
            "real_xy": per_window["real_xy"],
            "map_changed": per_window["map_changed"],
            "pos_err_img": per_window["pos_err_img"],
            "pos_err_real": per_window["pos_err_real"],
            "pos_err_nochange": per_window["pos_err_nochange"],
            "pos_agree": per_window["pos_agree"],
            "map_acc_img": per_window["map_acc_img"],
            "map_acc_real": per_window["map_acc_real"],
            "map_acc_nochange": per_window["map_acc_nochange"],
            "wrong_map_img": per_window["wrong_map_img"],
            "battle_acc_img": per_window["battle_acc_img"],
            "battle_acc_nochange": per_window["battle_acc_nochange"],
            "mse_img": per_window["mse_img"],
            "mse_copy": per_window["mse_copy"],
            "moved": per_window["moved"],
        }
        np.savez(npz_path, **save_dict)
        print(f"Saved per-window metrics to {npz_path}")

        # Save summary in json
        json_path = Path("work/eval") / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary curves to {json_path}")

        # Save plots
        plot_drift(summary, split_name, stem, Path("work/plots") / f"{stem}_drift.png")
        plot_mapacc(summary, split_name, stem, Path("work/plots") / f"{stem}_mapacc.png")
        plot_mse(summary, split_name, stem, Path("work/plots") / f"{stem}_mse.png")
        plot_grid(data, starts, per_window, args.k, Path("work/plots") / f"{stem}_grid.png")
        print(f"Saved plots to work/plots/{stem}_*.png")


if __name__ == "__main__":
    main()
