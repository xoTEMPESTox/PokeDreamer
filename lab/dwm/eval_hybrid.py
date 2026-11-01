"""RAM-grounded evaluation of hybrid diffusion world model imagined rollouts.

Evaluates HybridDenoiser + RamDynamics rollouts against game RAM ground truth
and probe readings on imagined frames. Reports wrong-map rates (symbolic vs probe),
map accuracy, position drift, and pixel MSE.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from lab.data import load
from lab.dwm.hybrid import HybridDenoiser, RamDynamics
from lab.dwm.model import CTX, to_float
from lab.dwm.train import windows
from lab.eval.metrics import bootstrap_ci, manhattan
from lab.dwm.eval_rollouts import (
    labels_for,
    plot_drift,
    plot_grid,
    plot_mapacc,
    plot_mse,
    summarize,
)
from lab.probe import load_probe, predict


def evaluate_split_hybrid(
    model: HybridDenoiser,
    ram_model: RamDynamics,
    probe,
    data: dict,
    starts: np.ndarray,
    K: int,
    steps: int = 3,
    aug_bucket: int = 0,
    device: str = "cpu",
    batch: int = 128,
    guidance: float = 1.0,
) -> dict:
    """Run hybrid rollouts and evaluate against RAM ground truth and probe readings."""
    starts = np.asarray(starts)
    W = len(starts)

    if probe is None or predict is None:
        raise RuntimeError("lab.probe is required for evaluate_split_hybrid")

    model.eval()
    ram_model.eval()
    model.to(device)
    ram_model.to(device)

    imagined_list = []
    real_list = []
    pred_ram_lists = {"map": [], "x": [], "y": [], "facing": [], "battle": []}

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

        ram_ctx = {
            "map": torch.from_numpy(data["map_ids"][ctx_idx].astype(np.int64)).to(device),
            "x": torch.from_numpy(data["xs"][ctx_idx].astype(np.int64)).to(device),
            "y": torch.from_numpy(data["ys"][ctx_idx].astype(np.int64)).to(device),
            "facing": torch.from_numpy((data["facings"][ctx_idx] // 4).astype(np.int64)).to(device),
            "battle": torch.from_numpy(data["in_battles"][ctx_idx].astype(np.int64)).to(device),
        }

        with torch.no_grad():
            if "cuda" in str(device):
                with torch.autocast("cuda", torch.bfloat16):
                    img_batch, ram_batch = model.rollout(
                        ctx_tensor, ram_ctx, act_tensor, steps=steps, aug_bucket=aug_bucket, ram_model=ram_model, guidance=guidance
                    )
            else:
                img_batch, ram_batch = model.rollout(
                    ctx_tensor, ram_ctx, act_tensor, steps=steps, aug_bucket=aug_bucket, ram_model=ram_model, guidance=guidance
                )

        imagined_list.append(img_batch.cpu().float().numpy())
        for k_field in ("map", "x", "y", "facing", "battle"):
            pred_ram_lists[k_field].append(ram_batch[k_field])

    if W > 0:
        imagined = np.concatenate(imagined_list, axis=0)
        real_frames = np.concatenate(real_list, axis=0)
        pred_ram_map = np.concatenate(pred_ram_lists["map"], axis=0)
        pred_ram_x = np.concatenate(pred_ram_lists["x"], axis=0)
        pred_ram_y = np.concatenate(pred_ram_lists["y"], axis=0)
        pred_ram_facing = np.concatenate(pred_ram_lists["facing"], axis=0)
        pred_ram_battle = np.concatenate(pred_ram_lists["battle"], axis=0)
    else:
        imagined = np.empty((0, K, 72, 80), dtype=np.float32)
        real_frames = np.empty((0, K, 72, 80), dtype=np.uint8)
        pred_ram_map = np.empty((0, K), dtype=int)
        pred_ram_x = np.empty((0, K), dtype=int)
        pred_ram_y = np.empty((0, K), dtype=int)
        pred_ram_facing = np.empty((0, K), dtype=int)
        pred_ram_battle = np.empty((0, K), dtype=int)

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

    # Symbolic RAM position error
    pred_ram_xy = np.stack([pred_ram_x, pred_ram_y], axis=-1)
    pos_err_ram = manhattan(pred_ram_xy, true_xy).astype(float)
    pos_err_ram[pos_mask] = np.nan

    # Map accuracies (NaN where true_battle)
    map_acc_img = (img_map == true_map).astype(float)
    map_acc_real = (real_map == true_map).astype(float)
    map_acc_nochange = (map0[:, None] == true_map).astype(float)
    map_acc_ram = (pred_ram_map == true_map).astype(float)

    map_acc_img[true_battle] = np.nan
    map_acc_real[true_battle] = np.nan
    map_acc_nochange[true_battle] = np.nan
    map_acc_ram[true_battle] = np.nan

    # Wrong map rates (on valid overworld unchanged-map steps, following lab/eval/decompose.py)
    valid_map_steps = ~pos_mask
    wrong_map_probe = np.where(valid_map_steps, (img_map != true_map).astype(float), np.nan)
    wrong_map_ram = np.where(valid_map_steps, (pred_ram_map != true_map).astype(float), np.nan)

    # Battle accuracy
    battle_acc_img = (img_battle == true_battle).astype(float)
    battle_acc_nochange = (battle0[:, None] == true_battle).astype(float)

    # Frame MSE in [0, 1]
    real_k_01 = real_frames.astype(np.float32) / 255.0
    from lab.eval.metrics import copy_last_frame_mse, frame_mse

    if W > 0:
        ctx_0_01 = data["frames"][starts + CTX - 1].astype(np.float32) / 255.0
        real_all_01 = np.concatenate([ctx_0_01[:, None], real_k_01], axis=1)
        mse_copy = copy_last_frame_mse(real_all_01)
    else:
        mse_copy = np.empty((0, K), dtype=float)

    mse_img = frame_mse(imagined_01, real_k_01)
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
        "pos_err_ram": pos_err_ram,
        "map_acc_img": map_acc_img,
        "map_acc_real": map_acc_real,
        "map_acc_nochange": map_acc_nochange,
        "map_acc_ram": map_acc_ram,
        "wrong_map_probe": wrong_map_probe,
        "wrong_map_ram": wrong_map_ram,
        "battle_acc_img": battle_acc_img,
        "battle_acc_nochange": battle_acc_nochange,
        "mse_img": mse_img,
        "mse_copy": mse_copy,
        "moved": moved,
        "pred_ram_map": pred_ram_map,
        "pred_ram_x": pred_ram_x,
        "pred_ram_y": pred_ram_y,
        "pred_ram_facing": pred_ram_facing,
        "pred_ram_battle": pred_ram_battle,
        "pred_ram_xy": pred_ram_xy,
        "imagined_frames": imagined_u8,
        "real_frames": real_frames,
    }


def print_compact_table_hybrid(summary: dict, split_name: str):
    """Print compact summary table including symbolic and probe wrong-map rates."""
    reported_ks = summary["reported_k"]
    header_ks = "".join(f"{f'k={k}':>20}" for k in reported_ks)
    tot_w = 34 + 20 * len(reported_ks)
    print("=" * tot_w)
    print(
        f"HYBRID EVALUATION RESULTS: split={split_name} | windows={summary['n_windows']} | "
        f"episodes={summary['n_episodes']} | valid_pos_fraction={summary['valid_pos_fraction']:.1%}"
    )
    print("=" * tot_w)
    print(f"{'Metric':<34}{header_ks}")
    print("-" * tot_w)

    metric_display = [
        ("wrong_map_ram", "wrong_map_rate (symbolic RAM)"),
        ("wrong_map_probe", "wrong_map_rate (imag probe)"),
        ("map_acc_ram", "map_acc_ram (symbolic)"),
        ("map_acc_img", "map_acc_img (probe)"),
        ("map_acc_real", "map_acc_real (probe ceiling)"),
        ("map_acc_nochange", "map_acc_nochange"),
        ("pos_err_ram", "pos_err_ram (symbolic)"),
        ("pos_err_img", "pos_err_img (imagined probe)"),
        ("pos_err_nochange", "pos_err_nochange"),
        ("pos_err_real", "pos_err_real"),
        ("pos_agree", "pos_agree"),
        ("battle_acc_img", "battle_acc_img"),
        ("battle_acc_nochange", "battle_acc_nochange"),
        ("mse_img", "mse_img"),
        ("mse_copy", "mse_copy"),
    ]

    for key, display_name in metric_display:
        if key not in summary["metrics"]:
            continue
        row_str = f"{display_name:<34}"
        for k in reported_ks:
            val_info = summary["metrics"][key]["at_k"][str(k)]
            m = val_info["mean"]
            lo = val_info["lo"]
            hi = val_info["hi"]
            if np.isnan(m):
                cell = "nan"
            elif "wrong_map" in key:
                cell = f"{m * 100:.1f}% [{lo * 100:.1f}%,{hi * 100:.1f}%]"
            elif "acc" in key:
                cell = f"{m:.3f} [{lo:.3f},{hi:.3f}]"
            elif "mse" in key:
                cell = f"{m:.4f} [{lo:.4f},{hi:.4f}]"
            else:
                cell = f"{m:.2f} [{lo:.2f},{hi:.2f}]"
            row_str += f"{cell:>20}"
        print(row_str)

    if "psnr" in summary and "at_k" in summary["psnr"]:
        for psnr_key, display_name in [("psnr_img", "psnr_img (dB)"), ("psnr_copy", "psnr_copy (dB)")]:
            row_str = f"{display_name:<34}"
            for k in reported_ks:
                val = summary["psnr"]["at_k"][str(k)][psnr_key]
                cell = f"{val:.2f}" if not np.isnan(val) else "nan"
                row_str += f"{cell:>20}"
            print(row_str)

    print("=" * tot_w)


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid diffusion world model imagined rollouts.")
    parser.add_argument("--run", type=str, default="hybrid", help="Run name in work/runs/<run>")
    parser.add_argument("--split", type=str, default="both", choices=["test_id", "test_ood", "both"], help="Split")
    parser.add_argument("--k", type=int, default=32, help="Rollout horizon K")
    parser.add_argument("--n-windows", type=int, default=512, help="Number of windows per split")
    parser.add_argument("--steps", type=int, default=3, help="Euler sampler diffusion steps")
    parser.add_argument("--aug-bucket", type=int, default=0, help="Augmentation bucket")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--guidance", type=float, default=1.0, help="Classifier-free guidance weight for RAM conditioning (default: 1.0)")
    args = parser.parse_args()

    dev = args.device
    ckpt_path = Path("work/runs") / args.run / "ckpt.pt"
    probe_path = Path("work/probe/probe.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {ckpt_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")

    print(f"Loading hybrid world model from {ckpt_path} on {dev}...")
    model = HybridDenoiser()
    ram_model = RamDynamics()
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
    model.load_state_dict(state["ema"])
    ram_model.load_state_dict(state["ram"])
    model.to(dev).eval()
    ram_model.to(dev).eval()

    print(f"Loading probe from {probe_path} on {dev}...")
    probe = load_probe(str(probe_path), device=dev)

    print("Loading dataset...")
    data, splits = load()

    splits_to_eval = ["test_id", "test_ood"] if args.split == "both" else [args.split]
    Path("work/eval").mkdir(parents=True, exist_ok=True)
    Path("work/plots").mkdir(parents=True, exist_ok=True)

    for split_name in splits_to_eval:
        print(f"\n--- Evaluating hybrid split: {split_name} (K={args.k}, steps={args.steps}, guidance={args.guidance}) ---")
        all_starts = windows(data, splits[split_name], CTX + args.k)
        rng = np.random.default_rng(0)
        n = min(len(all_starts), args.n_windows)
        starts = rng.choice(all_starts, size=n, replace=False)
        print(f"Sampled {len(starts)} windows from {len(all_starts)} eligible windows.")

        per_window = evaluate_split_hybrid(
            model=model,
            ram_model=ram_model,
            probe=probe,
            data=data,
            starts=starts,
            K=args.k,
            steps=args.steps,
            aug_bucket=args.aug_bucket,
            device=dev,
            batch=args.batch,
            guidance=args.guidance,
        )

        episode_ids = data["episode_id"][starts]
        summary = summarize(per_window, episode_ids)

        # Extend summary with RAM model metrics
        unique_eps = np.unique(episode_ids)
        E = len(unique_eps)
        K = args.k
        reported_ks = [k for k in [1, 2, 4, 8, 16, 32] if k <= K]
        extra_metrics = [
            ("wrong_map_ram", per_window["wrong_map_ram"]),
            ("wrong_map_probe", per_window["wrong_map_probe"]),
            ("map_acc_ram", per_window["map_acc_ram"]),
            ("pos_err_ram", per_window["pos_err_ram"]),
        ]
        for name, arr in extra_metrics:
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
            summary["metrics"][name] = {
                "mean": mean.tolist(),
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "at_k": {
                    str(k): {"mean": float(mean[k - 1]), "lo": float(lo[k - 1]), "hi": float(hi[k - 1])}
                    for k in reported_ks
                },
            }

        print_compact_table_hybrid(summary, split_name)

        stem = f"{args.run}_{split_name}_hybrid" if args.guidance == 1.0 else f"{args.run}_{split_name}_hybrid_g{args.guidance}"
        npz_path = Path("work/eval") / f"{stem}.npz"
        json_path = Path("work/eval") / f"{stem}.json"

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
            "battle_acc_img": per_window["battle_acc_img"],
            "battle_acc_nochange": per_window["battle_acc_nochange"],
            "mse_img": per_window["mse_img"],
            "mse_copy": per_window["mse_copy"],
            "moved": per_window["moved"],
            "pred_ram_map": per_window["pred_ram_map"],
            "pred_ram_x": per_window["pred_ram_x"],
            "pred_ram_y": per_window["pred_ram_y"],
            "pred_ram_facing": per_window["pred_ram_facing"],
            "pred_ram_battle": per_window["pred_ram_battle"],
            "wrong_map_ram": per_window["wrong_map_ram"],
            "wrong_map_probe": per_window["wrong_map_probe"],
            "map_acc_ram": per_window["map_acc_ram"],
            "pos_err_ram": per_window["pos_err_ram"],
        }
        np.savez(npz_path, **save_dict)
        print(f"Saved per-window metrics to {npz_path}")

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary curves to {json_path}")

        plot_drift(summary, split_name, stem, Path("work/plots") / f"{stem}_drift.png")
        plot_mapacc(summary, split_name, stem, Path("work/plots") / f"{stem}_mapacc.png")
        plot_mse(summary, split_name, stem, Path("work/plots") / f"{stem}_mse.png")
        plot_grid(data, starts, per_window, args.k, Path("work/plots") / f"{stem}_grid.png")
        print(f"Saved plots to work/plots/{stem}_*.png")


if __name__ == "__main__":
    main()
