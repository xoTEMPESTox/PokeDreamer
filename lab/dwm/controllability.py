"""Action-controllability test for a diffusion world model of Pokemon Red.

Evaluates whether imagined player movement follows commanded direction buttons
using a trained frame->RAM probe as an automated judge.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float
from lab.dwm.train import windows
from lab.eval.metrics import bootstrap_ci
from lab.probe import load_probe, predict

DIRECTIONS = {0: "down", 1: "left", 2: "right", 3: "up"}
DIR_NAMES = ["down", "left", "right", "up"]


def make_actions(ctx_actions, d: int, K: int):
    """Construct rollout actions: real context actions (first CTX-1) followed by K copies of action d.

    Supports 1D or 2D torch.Tensor, numpy.ndarray, or list:
    - 1D: (CTX-1,) -> (CTX-1 + K,)
    - 2D: (B, CTX-1) -> (B, CTX-1 + K)
    Preserves type (torch.Tensor or np.ndarray).
    """
    if isinstance(ctx_actions, torch.Tensor):
        if ctx_actions.ndim == 1:
            k_acts = torch.full((K,), d, dtype=ctx_actions.dtype, device=ctx_actions.device)
            return torch.cat([ctx_actions, k_acts], dim=0)
        elif ctx_actions.ndim == 2:
            k_acts = torch.full((ctx_actions.shape[0], K), d, dtype=ctx_actions.dtype, device=ctx_actions.device)
            return torch.cat([ctx_actions, k_acts], dim=1)
        else:
            raise ValueError(f"ctx_actions tensor must be 1D or 2D, got shape {ctx_actions.shape}")
    else:
        arr = np.asarray(ctx_actions)
        if arr.ndim == 1:
            k_acts = np.full((K,), d, dtype=arr.dtype)
            return np.concatenate([arr, k_acts], axis=0)
        elif arr.ndim == 2:
            k_acts = np.full((arr.shape[0], K), d, dtype=arr.dtype)
            return np.concatenate([arr, k_acts], axis=1)
        else:
            raise ValueError(f"ctx_actions array must be 1D or 2D, got shape {arr.shape}")


def signed_displacement(dx, dy, d: int):
    """Commanded-axis signed displacement for direction d.

    Positive means moved in the commanded direction:
    - 0 (down): +dy
    - 1 (left): -dx
    - 2 (right): +dx
    - 3 (up): -dy
    """
    if d == 0:
        return dy
    elif d == 1:
        return -dx
    elif d == 2:
        return dx
    elif d == 3:
        return -dy
    else:
        raise ValueError(f"Invalid direction d={d}, expected 0 (down), 1 (left), 2 (right), or 3 (up)")


def orthogonal_displacement(dx, dy, d: int):
    """Orthogonal absolute displacement for direction d:

    - 0 (down) or 3 (up): |dx|
    - 1 (left) or 2 (right): |dy|
    """
    if d in (0, 3):
        return np.abs(dx) if not isinstance(dx, torch.Tensor) else torch.abs(dx)
    elif d in (1, 2):
        return np.abs(dy) if not isinstance(dy, torch.Tensor) else torch.abs(dy)
    else:
        raise ValueError(f"Invalid direction d={d}, expected 0, 1, 2, or 3")


def real_reference(data: dict, eps, d: int, K: int) -> dict:
    """Find real windows where direction button d was pressed K times in a row, all overworld,
    and report the commanded-axis displacement from RAM.

    Args:
        data: dict of arrays (actions, xs, ys, map_ids, in_battles, episode_id)
        eps: list, set, or array of episode IDs
        d: direction int (0=down, 1=left, 2=right, 3=up)
        K: horizon int

    Returns:
        dict with:
            "count": int, number of matching windows
            "mean": np.ndarray of shape (K,), mean commanded displacement at steps 1..K
            "displacements": np.ndarray of shape (N, K), signed displacements per window
            "starts": np.ndarray of shape (N,), window start indices
    """
    w = windows(data, eps, CTX + K)
    if len(w) == 0:
        return {
            "count": 0,
            "mean": np.zeros(K, dtype=float),
            "displacements": np.empty((0, K), dtype=float),
            "starts": np.empty((0,), dtype=int),
        }

    # 1. All CTX + K frames must be overworld (not in battle)
    idx = w[:, None] + np.arange(CTX + K)
    no_battle = ~data["in_battles"][idx].any(axis=1)

    # 2. Actions s+CTX-1 .. s+CTX+K-2 all equal d (length K)
    act_idx = w[:, None] + np.arange(CTX - 1, CTX + K - 1)
    same_action = (data["actions"][act_idx] == d).all(axis=1)

    # 3. Must stay on the same map as reference frame s+CTX-1
    map_idx = w[:, None] + np.arange(CTX - 1, CTX + K)
    map0 = data["map_ids"][w + CTX - 1]
    same_map = (data["map_ids"][map_idx] == map0[:, None]).all(axis=1)

    valid_mask = no_battle & same_action & same_map
    matching_starts = w[valid_mask]
    n_matches = len(matching_starts)

    if n_matches == 0:
        return {
            "count": 0,
            "mean": np.zeros(K, dtype=float),
            "displacements": np.empty((0, K), dtype=float),
            "starts": np.empty((0,), dtype=int),
        }

    step_idx = matching_starts[:, None] + (CTX - 1) + np.arange(1, K + 1)
    x0 = data["xs"][matching_starts + CTX - 1][:, None].astype(int)  # RAM is uint8; avoid wraparound
    y0 = data["ys"][matching_starts + CTX - 1][:, None].astype(int)
    dx = data["xs"][step_idx] - x0
    dy = data["ys"][step_idx] - y0
    displacements = signed_displacement(dx, dy, d)
    mean_disp = np.mean(displacements, axis=0)

    return {
        "count": int(n_matches),
        "mean": mean_disp,
        "displacements": displacements,
        "starts": matching_starts,
    }


def pick_contexts(data: dict, eps, probe, n_contexts: int, seed: int = 0, device: str = "cpu") -> np.ndarray:
    """Pick windows of length CTX from the split where all CTX frames are overworld,
    and the probe correctly reads the last context frame's map and position.
    """
    cand = windows(data, eps, CTX)
    if len(cand) == 0:
        return np.empty((0,), dtype=int)

    # Check 4 context frames all overworld
    idx = cand[:, None] + np.arange(CTX)
    overworld = ~data["in_battles"][idx].any(axis=1)
    cand = cand[overworld]
    if len(cand) == 0:
        return np.empty((0,), dtype=int)

    # Check last context frame with probe
    last_idx = cand + CTX - 1
    last_frames = data["frames"][last_idx]
    preds = predict(probe, last_frames, device=device, batch=1024)

    ram_map = data["map_ids"][last_idx]
    ram_x = data["xs"][last_idx]
    ram_y = data["ys"][last_idx]

    trustworthy = (
        (preds["map_id"] == ram_map)
        & (preds["x"] == ram_x)
        & (preds["y"] == ram_y)
        & (~preds["battle"])
    )
    valid_starts = cand[trustworthy]

    rng = np.random.default_rng(seed)
    if len(valid_starts) > n_contexts:
        selected = rng.choice(valid_starts, size=n_contexts, replace=False)
    else:
        selected = valid_starts
    return np.sort(selected)


def load_world_model(run: str, device: str = "cpu") -> Denoiser:
    """Load Denoiser EMA weights from run directory."""
    ckpt_path = Path("work/runs") / run / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model = Denoiser().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["ema"])
    model.eval()
    return model


@torch.no_grad()
def rollout_condition(model: Denoiser, ctx_float: torch.Tensor, real_actions: np.ndarray,
                      d: int, K: int, steps: int = 3, device: str = "cpu",
                      batch_size: int = 32) -> torch.Tensor:
    """Generate K-step rollouts for action condition d across all contexts."""
    B = ctx_float.shape[0]
    actions = make_actions(torch.from_numpy(real_actions).long().to(device), d, K)
    outs = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        b_ctx = ctx_float[start:end]
        b_act = actions[start:end]
        if device == "cuda":
            with torch.autocast("cuda", torch.bfloat16):
                b_out = model.rollout(b_ctx, b_act, steps=steps)
        else:
            b_out = model.rollout(b_ctx, b_act, steps=steps)
        outs.append(b_out.cpu())
    return torch.cat(outs, dim=0)


def to_serializable(val):
    """Recursively convert numpy types to standard Python types for JSON."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val) if not np.isnan(val) else None
    elif isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, np.ndarray):
        return [to_serializable(v) for v in val.tolist()]
    elif isinstance(val, list):
        return [to_serializable(v) for v in val]
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    return val


def save_displacement_plot(results: dict, stem: str, out_dir: Path):
    """Save 2x2 line plot of commanded-axis displacement vs k per direction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"control_{stem}.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    dir_titles = ["Down (0)", "Left (1)", "Right (2)", "Up (3)"]

    for idx, (d_name, title) in enumerate(zip(DIR_NAMES, dir_titles)):
        ax = axes[idx // 2, idx % 2]
        d_res = results["directions"][d_name]
        ks = d_res["k_steps"]
        mean = np.array(d_res["commanded_displacement"]["mean"])
        lo = np.array(d_res["commanded_displacement"]["ci_lo"])
        hi = np.array(d_res["commanded_displacement"]["ci_hi"])

        # Imagined error bars
        yerr = [mean - lo, hi - mean]
        ax.errorbar(ks, mean, yerr=yerr, fmt="o-", label="Imagined (95% CI)",
                    color="#1f77b4", capsize=4, capthick=1.5, lw=2)

        # Real reference
        ref = d_res["real_reference"]
        ref_mean = [ref["mean_by_k"][str(k)] for k in ks]
        ax.plot(ks, ref_mean, "s--", label=f"Real RAM (N={ref['count']})",
                color="#ff7f0e", lw=2)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Commanded Disp (tiles)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)
        if idx >= 2:
            ax.set_xlabel("Horizon k (steps)")

    split = results["split"]
    run = results["run"]
    fig.suptitle(f"World Model Controllability ({run} on {split})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_grid_visualization(imagined_by_cond: dict, ctx_frames: np.ndarray,
                           starts: np.ndarray, data: dict, stem: str, K: int,
                           out_dir: Path):
    """Save grid visualization: for 4 contexts, a block with the last context frame

    then 5 rows (down, left, right, up, pass) of imagined frames at k=1..K,
    upscaled 3x nearest, rows labelled with cv2.putText.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"control_{stem}_grid.png"

    n_contexts_to_show = min(4, len(starts))
    if n_contexts_to_show == 0:
        return

    H_u = 72 * 3   # 216
    W_u = 80 * 3   # 240
    W_lbl = 120    # width of row label box

    row_names = ["ctx", "down", "left", "right", "up", "pass"]
    row_keys = ["ctx", 0, 1, 2, 3, 7]

    all_blocks = []

    for c in range(n_contexts_to_show):
        start_idx = starts[c]
        ep_id = int(data["episode_id"][start_idx])
        map0 = int(data["map_ids"][start_idx + CTX - 1])
        x0 = int(data["xs"][start_idx + CTX - 1])
        y0 = int(data["ys"][start_idx + CTX - 1])

        # Header banner for context
        banner_h = 36
        banner_w = W_lbl + K * W_u
        banner = np.full((banner_h, banner_w, 3), 45, dtype=np.uint8)
        banner_text = f"Context #{c+1} | Ep: {ep_id} | Map: {map0} | Pos: ({x0}, {y0})"
        cv2.putText(banner, banner_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        block_rows = [banner]

        for r_name, r_key in zip(row_names, row_keys):
            # Label strip on left
            lbl = np.full((H_u, W_lbl, 3), 25, dtype=np.uint8)
            cv2.putText(lbl, r_name, (12, H_u // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            row_frames = [lbl]

            if r_key == "ctx":
                # Column 0: Last context frame
                last_f = ctx_frames[c, CTX - 1]
                last_u3 = cv2.resize(last_f, (W_u, H_u), interpolation=cv2.INTER_NEAREST)
                last_bgr = cv2.cvtColor(last_u3, cv2.COLOR_GRAY2BGR)
                cv2.putText(last_bgr, "t=0 (RAM)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                row_frames.append(last_bgr)

                # Columns 1..K-1: dark placeholder with k header
                for k in range(1, K):
                    ph = np.full((H_u, W_u, 3), 35, dtype=np.uint8)
                    cv2.putText(ph, f"k={k+1}", (W_u // 2 - 30, H_u // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA)
                    row_frames.append(ph)
            else:
                # Imagined frames k=1..K
                cond_imagined = imagined_by_cond[r_key] # (B, K, 72, 80)
                for k in range(K):
                    f_float = cond_imagined[c, k]
                    f_u8 = ((f_float + 1.0) * 127.5).clamp(0, 255).byte().numpy()
                    f_u3 = cv2.resize(f_u8, (W_u, H_u), interpolation=cv2.INTER_NEAREST)
                    f_bgr = cv2.cvtColor(f_u3, cv2.COLOR_GRAY2BGR)
                    cv2.putText(f_bgr, f"k={k+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
                    row_frames.append(f_bgr)

            row_img = np.concatenate(row_frames, axis=1)
            block_rows.append(row_img)

        # Context block image
        block_img = np.concatenate(block_rows, axis=0)
        all_blocks.append(block_img)

        # Separator between contexts
        if c < n_contexts_to_show - 1:
            sep = np.zeros((15, banner_w, 3), dtype=np.uint8)
            all_blocks.append(sep)

    final_grid = np.concatenate(all_blocks, axis=0)
    cv2.imwrite(str(out_path), final_grid)


def print_compact_table(results: dict):
    """Print compact summary table of controllability metrics."""
    split = results["split"]
    run = results["run"]
    K = results["k"]
    steps = results["steps"]
    n_ctx = results["n_contexts"]
    print(f"\n{'='*112}")
    print(f"Action-Controllability: run={run}, split={split}, K={K}, steps={steps}, n_contexts={n_ctx}")
    print(f"{'='*112}")
    print(f"{'Dir':<6} | {'k':<2} | {'Commanded Disp (95% CI)':<26} | {'Orth |dx|':<9} | {'Moved Cmd':<9} | {'Moved Opp':<9} | {'Valid':<7} | {'Real Ref (N)':<15}")
    print(f"{'-'*112}")
    for d_name in DIR_NAMES:
        d_res = results["directions"][d_name]
        ks = d_res["k_steps"]
        ref = d_res["real_reference"]
        for i, k in enumerate(ks):
            m_c = d_res["commanded_displacement"]["mean"][i]
            lo_c = d_res["commanded_displacement"]["ci_lo"][i]
            hi_c = d_res["commanded_displacement"]["ci_hi"][i]
            m_o = d_res["orthogonal_displacement"]["mean"][i]
            m_mc = d_res["frac_moved_commanded"]["mean"][i] * 100
            m_mo = d_res["frac_moved_opposite"]["mean"][i] * 100
            m_v = d_res["frac_valid"]["mean"][i] * 100
            ref_val = ref["mean_by_k"][str(k)]
            ref_cnt = ref["count"]
            ci_str = f"{m_c:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
            ref_str = f"{ref_val:+.2f} (N={ref_cnt})"
            dir_label = d_name.capitalize() if i == 0 else ""
            print(f"{dir_label:<6} | {k:<2} | {ci_str:<26} | {m_o:9.2f} | {m_mc:8.1f}% | {m_mo:8.1f}% | {m_v:6.1f}% | {ref_str:<15}")
        print(f"{'-'*112}")
    # Control condition pass
    p_res = results["pass"]
    p_ks = p_res["k_steps"]
    for i, k in enumerate(p_ks):
        m_m = p_res["manhattan"]["mean"][i]
        lo_m = p_res["manhattan"]["ci_lo"][i]
        hi_m = p_res["manhattan"]["ci_hi"][i]
        m_v = p_res["frac_valid"]["mean"][i] * 100
        p_ci = f"{m_m:.2f} [{lo_m:.2f}, {hi_m:.2f}]"
        dir_label = "Pass" if i == 0 else ""
        print(f"{dir_label:<6} | {k:<2} | |dx|+|dy| = {p_ci:<14} | {'-':<9} | {'-':<9} | {'-':<9} | {m_v:6.1f}% | {'-':<15}")
    print(f"{'='*112}\n")


def evaluate_split(model: Denoiser, probe, data: dict, splits: dict,
                   split_name: str, run: str, n_contexts: int, K: int,
                   steps: int, device: str, seed: int) -> dict:
    """Run action-controllability test on one split."""
    eps = splits[split_name]
    starts = pick_contexts(data, eps, probe, n_contexts=n_contexts, seed=seed, device=device)
    B = len(starts)
    if B == 0:
        raise RuntimeError(f"No valid contexts found for split {split_name}")

    ctx_frames = data["frames"][starts[:, None] + np.arange(CTX)]
    ctx_float = to_float(torch.from_numpy(ctx_frames)).to(device)
    real_actions = data["actions"][starts[:, None] + np.arange(CTX - 1)]

    x0 = data["xs"][starts + CTX - 1]
    y0 = data["ys"][starts + CTX - 1]
    map0 = data["map_ids"][starts + CTX - 1]

    ks = sorted(list(set(min(k, K) for k in [1, 2, 4, 8])))
    k_indices = [k - 1 for k in ks]

    conditions = [0, 1, 2, 3, 7]
    imagined_by_cond = {}
    preds_by_cond = {}

    for d in conditions:
        imagined = rollout_condition(model, ctx_float, real_actions, d, K, steps=steps, device=device)
        imagined_by_cond[d] = imagined
        f_01 = ((imagined + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
        preds = predict(probe, f_01.reshape(-1, 72, 80), device=device, batch=1024)
        preds_by_cond[d] = {
            "x": preds["x"].reshape(B, K),
            "y": preds["y"].reshape(B, K),
            "map_id": preds["map_id"].reshape(B, K),
            "battle": preds["battle"].reshape(B, K),
        }

    results = {
        "run": run,
        "split": split_name,
        "n_contexts": B,
        "k": K,
        "steps": steps,
        "device": device,
        "seed": seed,
        "directions": {},
        "pass": {},
    }

    # Evaluate directions 0..3
    for d, d_name in DIRECTIONS.items():
        pr = preds_by_cond[d]
        dx = pr["x"] - x0[:, None]
        dy = pr["y"] - y0[:, None]
        valid = (~pr["battle"]) & (pr["map_id"] == map0[:, None])

        disp_cmd = signed_displacement(dx, dy, d)
        disp_orth = orthogonal_displacement(dx, dy, d)

        disp_cmd_masked = np.where(valid, disp_cmd, np.nan)
        disp_orth_masked = np.where(valid, disp_orth, np.nan)

        moved_cmd = (valid & (disp_cmd >= 1.0)).astype(float)
        moved_opp = (valid & (disp_cmd <= -1.0)).astype(float)
        is_valid = valid.astype(float)

        # Bootstrap metrics at reporting horizons
        cmd_m, cmd_lo, cmd_hi = bootstrap_ci(disp_cmd_masked[:, k_indices], seed=seed)
        orth_m, orth_lo, orth_hi = bootstrap_ci(disp_orth_masked[:, k_indices], seed=seed)
        mc_m, mc_lo, mc_hi = bootstrap_ci(moved_cmd[:, k_indices], seed=seed)
        mo_m, mo_lo, mo_hi = bootstrap_ci(moved_opp[:, k_indices], seed=seed)
        v_m, v_lo, v_hi = bootstrap_ci(is_valid[:, k_indices], seed=seed)

        ref = real_reference(data, eps, d, K)
        ref_mean_by_k = {str(k): float(ref["mean"][k - 1]) if ref["count"] > 0 else 0.0 for k in ks}

        # Convert to 1D arrays if scalar (when len(ks) == 1)
        to_arr = lambda x: np.atleast_1d(x).tolist()

        results["directions"][d_name] = {
            "d": d,
            "k_steps": ks,
            "commanded_displacement": {
                "mean": to_arr(cmd_m),
                "ci_lo": to_arr(cmd_lo),
                "ci_hi": to_arr(cmd_hi),
            },
            "orthogonal_displacement": {
                "mean": to_arr(orth_m),
                "ci_lo": to_arr(orth_lo),
                "ci_hi": to_arr(orth_hi),
            },
            "frac_moved_commanded": {
                "mean": to_arr(mc_m),
                "ci_lo": to_arr(mc_lo),
                "ci_hi": to_arr(mc_hi),
            },
            "frac_moved_opposite": {
                "mean": to_arr(mo_m),
                "ci_lo": to_arr(mo_lo),
                "ci_hi": to_arr(mo_hi),
            },
            "frac_valid": {
                "mean": to_arr(v_m),
                "ci_lo": to_arr(v_lo),
                "ci_hi": to_arr(v_hi),
            },
            "real_reference": {
                "count": ref["count"],
                "mean_all_k": to_arr(ref["mean"]),
                "mean_by_k": ref_mean_by_k,
            },
        }

    # Evaluate pass control condition (d=7)
    p_pr = preds_by_cond[7]
    p_dx = p_pr["x"] - x0[:, None]
    p_dy = p_pr["y"] - y0[:, None]
    p_valid = (~p_pr["battle"]) & (p_pr["map_id"] == map0[:, None])
    p_manh = np.abs(p_dx) + np.abs(p_dy)
    p_manh_masked = np.where(p_valid, p_manh, np.nan)

    pm_m, pm_lo, pm_hi = bootstrap_ci(p_manh_masked[:, k_indices], seed=seed)
    pv_m, pv_lo, pv_hi = bootstrap_ci(p_valid[:, k_indices].astype(float), seed=seed)

    results["pass"] = {
        "d": 7,
        "k_steps": ks,
        "manhattan": {
            "mean": to_arr(pm_m),
            "ci_lo": to_arr(pm_lo),
            "ci_hi": to_arr(pm_hi),
        },
        "frac_valid": {
            "mean": to_arr(pv_m),
            "ci_lo": to_arr(pv_lo),
            "ci_hi": to_arr(pv_hi),
        },
    }

    # Save output artifacts
    stem = f"{run}_{split_name}_k{K}_s{steps}"
    eval_dir = Path("work/eval")
    plots_dir = Path("work/plots")
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / f"control_{stem}.json"
    with open(json_path, "w") as fh:
        json.dump(to_serializable(results), fh, indent=2)

    save_displacement_plot(results, stem, plots_dir)
    save_grid_visualization(imagined_by_cond, ctx_frames, starts, data, stem, K, plots_dir)

    print_compact_table(results)
    return results


def main():
    ap = argparse.ArgumentParser(description="Action-controllability test for Pokemon Red world model")
    ap.add_argument("--run", type=str, default="main")
    ap.add_argument("--split", type=str, default="both", choices=["test_id", "test_ood", "both"])
    ap.add_argument("--n-contexts", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, splits = load()
    probe = load_probe("work/probe/probe.pt", device=args.device)
    model = load_world_model(args.run, device=args.device)

    splits_to_run = ["test_id", "test_ood"] if args.split == "both" else [args.split]

    for s_name in splits_to_run:
        evaluate_split(
            model=model,
            probe=probe,
            data=data,
            splits=splits,
            split_name=s_name,
            run=args.run,
            n_contexts=args.n_contexts,
            K=args.k,
            steps=args.steps,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
