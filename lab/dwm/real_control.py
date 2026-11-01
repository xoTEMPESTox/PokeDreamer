"""Ground-truth controllability test of PokeDreamer diffusion world model against real game.

Rolls out the diffusion model open-loop conditioned on real Game Boy contexts,
and compares imagined trajectories against real emulator execution under identical button presses:
- Real commanded-axis displacement (RAM)
- Imagined commanded displacement (frame->RAM probe)
- Position error |imagined - real| in tiles (Manhattan, conditional on matching map)
- Map mismatch fraction
- Pixel MSE in [0, 1] between imagined and real frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lab.dwm.controllability import (
    DIRECTIONS,
    DIR_NAMES,
    load_world_model,
    make_actions,
    rollout_condition,
    signed_displacement,
    to_serializable,
)
from lab.dwm.model import to_float
from lab.env import ROM_PATH, PokemonRed, make_states
from lab.eval.metrics import bootstrap_ci
from lab.probe import load_probe, predict

CONDITIONS = {0: "down", 1: "left", 2: "right", 3: "up", 7: "pass"}
COND_ORDER = [0, 1, 2, 3, 7]


def collect_real_contexts(
    rom_path: Path,
    saves_dir: Path,
    n_contexts: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Collect diverse snapshot points starting alternately from bedroom and pallet save states.

    Runs a seeded random walk (mostly directions, some A presses) until the last 4 frames
    are all out of battle and on the same map.
    """
    bedroom_path = saves_dir / "bedroom.state"
    pallet_path = saves_dir / "pallet.state"

    if not bedroom_path.exists() or not pallet_path.exists():
        print(f"[collect_real_contexts] Save states missing in {saves_dir}, generating now...")
        make_states(rom=rom_path, out_dir=saves_dir)

    bedroom_bytes = bedroom_path.read_bytes()
    pallet_bytes = pallet_path.read_bytes()

    rng = np.random.default_rng(seed)
    contexts: list[dict[str, Any]] = []

    print(f"[collect_real_contexts] Collecting {n_contexts} contexts from real emulator...")
    with PokemonRed(rom=rom_path, state=bedroom_bytes) as env:
        for idx in range(n_contexts):
            source_name = "bedroom" if (idx % 2 == 0) else "pallet"
            source_bytes = bedroom_bytes if (idx % 2 == 0) else pallet_bytes

            env.restore(source_bytes)
            f0, r0 = env.observe()
            frames = [f0]
            actions: list[int] = []
            rams = [r0]

            # Walk a randomized number of steps to achieve spatial diversity
            target_walk = int(rng.integers(3, 25))
            steps_taken = 0

            while True:
                # 88% direction buttons, 12% A button
                act = int(rng.choice([0, 1, 2, 3, 4], p=[0.22, 0.22, 0.22, 0.22, 0.12]))
                f, r = env.step(act)
                frames.append(f)
                actions.append(act)
                rams.append(r)
                steps_taken += 1

                if steps_taken >= target_walk:
                    last_rams = rams[-4:]
                    no_battle = all(not item["in_battle"] for item in last_rams)
                    same_map = all(item["map_id"] == last_rams[-1]["map_id"] for item in last_rams)

                    if no_battle and same_map:
                        snap = env.snapshot()
                        ctx = {
                            "index": idx,
                            "source": source_name,
                            "snapshot": snap,
                            "frames": np.array(frames[-4:], dtype=np.uint8),  # (4, 72, 80)
                            "actions": np.array(actions[-3:], dtype=np.int64),  # (3,)
                            "map_id": int(last_rams[-1]["map_id"]),
                            "x": int(last_rams[-1]["x"]),
                            "y": int(last_rams[-1]["y"]),
                            "facing": int(last_rams[-1]["facing"]),
                        }
                        contexts.append(ctx)
                        break

                if steps_taken > 100:
                    # Reset if stuck in battle or persistent transition
                    env.restore(source_bytes)
                    f0, r0 = env.observe()
                    frames = [f0]
                    actions = []
                    rams = [r0]
                    steps_taken = 0

    print(f"[collect_real_contexts] Successfully collected {len(contexts)} contexts.")
    return contexts


def save_displacement_plot(
    results: dict[str, Any],
    run: str,
    K: int,
    out_path: Path,
) -> None:
    """Save 2x2 line plot comparing real vs imagined commanded displacement."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    dir_titles = ["Down (0)", "Left (1)", "Right (2)", "Up (3)"]

    for idx, (d_name, title) in enumerate(zip(DIR_NAMES, dir_titles)):
        ax = axes[idx // 2, idx % 2]
        d_res = results["conditions"][d_name]
        ks = np.array(d_res["k_steps"])

        r_mean = np.array(d_res["real_displacement"]["mean"], dtype=float)
        r_lo = np.array(d_res["real_displacement"]["ci_lo"], dtype=float)
        r_hi = np.array(d_res["real_displacement"]["ci_hi"], dtype=float)

        i_mean = np.array(d_res["imagined_displacement"]["mean"], dtype=float)
        i_lo = np.array(d_res["imagined_displacement"]["ci_lo"], dtype=float)
        i_hi = np.array(d_res["imagined_displacement"]["ci_hi"], dtype=float)

        # Real reference curve
        if not np.all(np.isnan(r_mean)):
            r_err = [r_mean - r_lo, r_hi - r_mean]
            ax.errorbar(
                ks,
                r_mean,
                yerr=r_err,
                fmt="s--",
                label="Real (RAM)",
                color="#ff7f0e",
                capsize=3,
                lw=2,
            )

        # Imagined curve
        if not np.all(np.isnan(i_mean)):
            i_err = [i_mean - i_lo, i_hi - i_mean]
            ax.errorbar(
                ks,
                i_mean,
                yerr=i_err,
                fmt="o-",
                label="Imagined (Probe 95% CI)",
                color="#1f77b4",
                capsize=4,
                capthick=1.5,
                lw=2,
            )

        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Commanded Disp (tiles)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)
        if idx >= 2:
            ax.set_xlabel("Horizon k (steps)")

    n_ctx = results["n_contexts"]
    fig.suptitle(
        f"Real-Game Controllability: {run} (N={n_ctx}, K={K})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_grid_visualization(
    contexts: list[dict[str, Any]],
    real_frames_by_cond: dict[int, np.ndarray],
    imagined_by_cond: dict[int, torch.Tensor],
    out_path: Path,
    K: int,
) -> None:
    """Save grid visualization for up to 4 contexts showing Real and Imagined frames per condition."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_show = min(4, len(contexts))
    if n_show == 0:
        return

    H_u = 72 * 3  # 216
    W_u = 80 * 3  # 240
    W_lbl = 140   # width of label banner

    all_blocks = []
    banner_w = W_lbl + K * W_u

    for c_idx in range(n_show):
        ctx = contexts[c_idx]
        map0 = ctx["map_id"]
        x0 = ctx["x"]
        y0 = ctx["y"]
        src = ctx["source"]

        # Context header banner
        banner_h = 36
        banner = np.full((banner_h, banner_w, 3), 45, dtype=np.uint8)
        header_str = f"Context #{c_idx+1} ({src}) | Map: {map0} | Initial Pos: ({x0}, {y0})"
        cv2.putText(banner, header_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        block_rows = [banner]

        for d in COND_ORDER:
            d_name = CONDITIONS[d]

            # ── 1. Real Row ──────────────────────────────────────────────────
            lbl_real = np.full((H_u, W_lbl, 3), 30, dtype=np.uint8)
            cv2.putText(lbl_real, f"Real {d_name}", (10, H_u // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            real_tiles = [lbl_real]

            for k in range(K):
                rf = real_frames_by_cond[d][c_idx, k]  # (72, 80) uint8
                rf_u3 = cv2.resize(rf, (W_u, H_u), interpolation=cv2.INTER_NEAREST)
                rf_bgr = cv2.cvtColor(rf_u3, cv2.COLOR_GRAY2BGR)
                cv2.putText(rf_bgr, f"k={k+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
                real_tiles.append(rf_bgr)

            block_rows.append(np.concatenate(real_tiles, axis=1))

            # ── 2. Imagined Row ──────────────────────────────────────────────
            lbl_imag = np.full((H_u, W_lbl, 3), 20, dtype=np.uint8)
            cv2.putText(lbl_imag, f"Imag {d_name}", (10, H_u // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2, cv2.LINE_AA)
            imag_tiles = [lbl_imag]

            for k in range(K):
                im_float = imagined_by_cond[d][c_idx, k]  # float in [-1, 1]
                im_u8 = ((im_float + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
                im_u3 = cv2.resize(im_u8, (W_u, H_u), interpolation=cv2.INTER_NEAREST)
                im_bgr = cv2.cvtColor(im_u3, cv2.COLOR_GRAY2BGR)
                cv2.putText(im_bgr, f"k={k+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1, cv2.LINE_AA)
                imag_tiles.append(im_bgr)

            block_rows.append(np.concatenate(imag_tiles, axis=1))

        context_block = np.concatenate(block_rows, axis=0)
        all_blocks.append(context_block)

        if c_idx < n_show - 1:
            sep = np.zeros((15, banner_w, 3), dtype=np.uint8)
            all_blocks.append(sep)

    final_grid = np.concatenate(all_blocks, axis=0)
    cv2.imwrite(str(out_path), final_grid)


def print_compact_table(results: dict[str, Any]) -> None:
    """Print compact summary table of controllability metrics."""
    run = results["run"]
    K = results["k"]
    n_ctx = results["n_contexts"]
    device = results["device"]

    print(f"\n{'=' * 122}")
    print(f"Ground-Truth Controllability Evaluation: run={run}, K={K}, n_contexts={n_ctx}, device={device}")
    print(f"{'=' * 122}")
    print(
        f"{'Cond':<6} | {'k':<2} | {'Real Disp (95% CI)':<22} | {'Imag Disp (95% CI)':<22} | "
        f"{'Pos Error (tiles)':<20} | {'Map Mismatch':<12} | {'Pixel MSE (x10^-3)':<18}"
    )
    print(f"{'-' * 122}")

    for d in COND_ORDER:
        d_name = CONDITIONS[d]
        d_res = results["conditions"][d_name]
        ks = d_res["k_steps"]
        for i, k in enumerate(ks):
            r_m = d_res["real_displacement"]["mean"][i]
            r_lo = d_res["real_displacement"]["ci_lo"][i]
            r_hi = d_res["real_displacement"]["ci_hi"][i]

            i_m = d_res["imagined_displacement"]["mean"][i]
            i_lo = d_res["imagined_displacement"]["ci_lo"][i]
            i_hi = d_res["imagined_displacement"]["ci_hi"][i]

            p_m = d_res["position_error"]["mean"][i]
            p_lo = d_res["position_error"]["ci_lo"][i]
            p_hi = d_res["position_error"]["ci_hi"][i]

            m_mis = d_res["map_mismatch_frac"][i] * 100.0

            mse_m = d_res["pixel_mse"]["mean"][i] * 1e3 if d_res["pixel_mse"]["mean"][i] is not None else None
            mse_lo = d_res["pixel_mse"]["ci_lo"][i] * 1e3 if d_res["pixel_mse"]["ci_lo"][i] is not None else None
            mse_hi = d_res["pixel_mse"]["ci_hi"][i] * 1e3 if d_res["pixel_mse"]["ci_hi"][i] is not None else None

            r_str = f"{r_m:+.2f} [{r_lo:+.2f}, {r_hi:+.2f}]" if r_m is not None else "N/A"
            i_str = f"{i_m:+.2f} [{i_lo:+.2f}, {i_hi:+.2f}]" if i_m is not None else "N/A"
            p_str = f"{p_m:.2f} [{p_lo:.2f}, {p_hi:.2f}]" if p_m is not None else "N/A"
            mis_str = f"{m_mis:5.1f}%"
            mse_str = f"{mse_m:.2f} [{mse_lo:.2f}, {mse_hi:.2f}]" if mse_m is not None else "N/A"

            c_label = d_name.capitalize() if i == 0 else ""
            print(
                f"{c_label:<6} | {k:<2} | {r_str:<22} | {i_str:<22} | "
                f"{p_str:<20} | {mis_str:<12} | {mse_str:<18}"
            )
        print(f"{'-' * 122}")
    print(f"{'=' * 122}\n")


def run_evaluation(
    run: str = "main",
    n_contexts: int = 64,
    k: int = 8,
    device: str = "cuda",
    steps: int = 3,
    seed: int = 0,
    rom_path: Path = ROM_PATH,
    saves_dir: Path = Path("saves"),
) -> dict[str, Any]:
    """Execute ground-truth controllability benchmark comparing world model to emulator."""
    # ── 1. Collect Contexts from Real Game ────────────────────────────────────
    contexts = collect_real_contexts(
        rom_path=rom_path,
        saves_dir=saves_dir,
        n_contexts=n_contexts,
        seed=seed,
    )
    B = len(contexts)
    K = k

    ctx_frames = np.stack([c["frames"] for c in contexts])      # (B, 4, 72, 80) uint8
    ctx_actions = np.stack([c["actions"] for c in contexts])    # (B, 3) int64
    x0 = np.array([c["x"] for c in contexts], dtype=int)        # (B,)
    y0 = np.array([c["y"] for c in contexts], dtype=int)        # (B,)
    map0 = np.array([c["map_id"] for c in contexts], dtype=int)  # (B,)

    # ── 2. Real Emulator Execution ───────────────────────────────────────────
    print(f"[run_evaluation] Running real emulator rollouts (K={K}) across {B} contexts...")
    real_frames_by_cond: dict[int, np.ndarray] = {}
    real_x_by_cond: dict[int, np.ndarray] = {}
    real_y_by_cond: dict[int, np.ndarray] = {}
    real_map_by_cond: dict[int, np.ndarray] = {}

    with PokemonRed(rom=rom_path) as env:
        for d in COND_ORDER:
            cond_frames = np.empty((B, K, 72, 80), dtype=np.uint8)
            cond_x = np.empty((B, K), dtype=int)
            cond_y = np.empty((B, K), dtype=int)
            cond_map = np.empty((B, K), dtype=int)

            for b_idx in range(B):
                env.restore(contexts[b_idx]["snapshot"])
                for step_k in range(K):
                    f_real, r_real = env.step(d)
                    cond_frames[b_idx, step_k] = f_real
                    cond_x[b_idx, step_k] = r_real["x"]
                    cond_y[b_idx, step_k] = r_real["y"]
                    cond_map[b_idx, step_k] = r_real["map_id"]

            real_frames_by_cond[d] = cond_frames
            real_x_by_cond[d] = cond_x
            real_y_by_cond[d] = cond_y
            real_map_by_cond[d] = cond_map

    # ── 3. Imagined World Model Rollouts ─────────────────────────────────────
    print(f"[run_evaluation] Loading world model ({run}) and probe on {device}...")
    model = load_world_model(run, device=device)
    probe = load_probe("work/probe/probe.pt", device=device)

    ctx_float = to_float(torch.from_numpy(ctx_frames)).to(device)

    imagined_by_cond: dict[int, torch.Tensor] = {}
    pred_x_by_cond: dict[int, np.ndarray] = {}
    pred_y_by_cond: dict[int, np.ndarray] = {}
    pred_map_by_cond: dict[int, np.ndarray] = {}
    pred_battle_by_cond: dict[int, np.ndarray] = {}

    print(f"[run_evaluation] Running imagined rollouts (steps={steps})...")
    for d in COND_ORDER:
        imagined = rollout_condition(
            model,
            ctx_float,
            ctx_actions,
            d,
            K,
            steps=steps,
            device=device,
            batch_size=32 if device == "cuda" else 8,
        )
        imagined_by_cond[d] = imagined  # (B, K, 72, 80) float in [-1, 1]

        # Convert to [0, 1] for probe
        f_01 = ((imagined + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
        preds = predict(probe, f_01.reshape(-1, 72, 80), device=device, batch=1024)

        pred_x_by_cond[d] = preds["x"].reshape(B, K)
        pred_y_by_cond[d] = preds["y"].reshape(B, K)
        pred_map_by_cond[d] = preds["map_id"].reshape(B, K)
        pred_battle_by_cond[d] = preds["battle"].reshape(B, K)

    # ── 4. Metric Computation ────────────────────────────────────────────────
    print("[run_evaluation] Computing metrics and bootstrap CIs...")
    results: dict[str, Any] = {
        "run": run,
        "n_contexts": B,
        "k": K,
        "device": device,
        "seed": seed,
        "conditions": {},
    }

    for d in COND_ORDER:
        d_name = CONDITIONS[d]

        # Real displacement
        dx_real = real_x_by_cond[d] - x0[:, None]
        dy_real = real_y_by_cond[d] - y0[:, None]
        if d in (0, 1, 2, 3):
            disp_real = signed_displacement(dx_real, dy_real, d)
        else:
            disp_real = np.abs(dx_real) + np.abs(dy_real)

        r_mean, r_lo, r_hi = bootstrap_ci(disp_real)

        # Imagined displacement
        pr_x = pred_x_by_cond[d]
        pr_y = pred_y_by_cond[d]
        pr_map = pred_map_by_cond[d]
        pr_battle = pred_battle_by_cond[d]

        dx_imag = pr_x - x0[:, None]
        dy_imag = pr_y - y0[:, None]
        if d in (0, 1, 2, 3):
            disp_imag = signed_displacement(dx_imag, dy_imag, d)
        else:
            disp_imag = np.abs(dx_imag) + np.abs(dy_imag)

        valid_imag = (pr_map == map0[:, None]) & (~pr_battle)
        disp_imag_masked = np.where(valid_imag, disp_imag, np.nan)
        i_mean, i_lo, i_hi = bootstrap_ci(disp_imag_masked)

        # Position error in tiles (Manhattan, conditional on matching map)
        r_map = real_map_by_cond[d]
        r_x = real_x_by_cond[d]
        r_y = real_y_by_cond[d]

        map_match = pr_map == r_map
        map_mismatch_frac = np.mean(~map_match, axis=0)

        manh_err = np.abs(pr_x - r_x) + np.abs(pr_y - r_y)
        manh_err_masked = np.where(map_match, manh_err, np.nan)
        p_mean, p_lo, p_hi = bootstrap_ci(manh_err_masked)

        # Pixel MSE in [0, 1]
        f_real_01 = real_frames_by_cond[d].astype(np.float32) / 255.0
        f_imag_01 = ((imagined_by_cond[d] + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
        pix_mse = np.mean((f_imag_01 - f_real_01) ** 2, axis=(-2, -1))  # (B, K)
        m_mean, m_lo, m_hi = bootstrap_ci(pix_mse)

        results["conditions"][d_name] = {
            "k_steps": list(range(1, K + 1)),
            "real_displacement": {
                "mean": r_mean,
                "ci_lo": r_lo,
                "ci_hi": r_hi,
            },
            "imagined_displacement": {
                "mean": i_mean,
                "ci_lo": i_lo,
                "ci_hi": i_hi,
            },
            "position_error": {
                "mean": p_mean,
                "ci_lo": p_lo,
                "ci_hi": p_hi,
            },
            "map_mismatch_frac": map_mismatch_frac.tolist(),
            "pixel_mse": {
                "mean": m_mean,
                "ci_lo": m_lo,
                "ci_hi": m_hi,
            },
        }

    # ── 5. Save Outputs ──────────────────────────────────────────────────────
    eval_dir = Path("work/eval")
    plots_dir = Path("work/plots")
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / f"real_control_{run}_k{K}.json"
    plot_path = plots_dir / f"real_control_{run}_k{K}.png"
    grid_path = plots_dir / f"real_control_{run}_k{K}_grid.png"

    json_path.write_text(json.dumps(to_serializable(results), indent=2))
    print(f"[run_evaluation] Saved metrics -> {json_path}")

    save_displacement_plot(results, run, K, plot_path)
    print(f"[run_evaluation] Saved plot -> {plot_path}")

    save_grid_visualization(contexts, real_frames_by_cond, imagined_by_cond, grid_path, K)
    print(f"[run_evaluation] Saved grid visualization -> {grid_path}")

    print_compact_table(to_serializable(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-game controllability benchmark for PokeDreamer.")
    parser.add_argument("--run", type=str, default="main", help="Run directory name under work/runs")
    parser.add_argument("--n-contexts", type=int, default=64, help="Number of snapshot contexts to evaluate")
    parser.add_argument("--k", type=int, default=8, help="Rollout horizon in action steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--steps", type=int, default=3, help="Euler sampling steps (default: 3)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for context generation")
    parser.add_argument("--rom", type=Path, default=ROM_PATH, help="Path to Pokemon Red ROM")
    parser.add_argument("--saves-dir", type=Path, default=Path("saves"), help="Directory for save states")
    args = parser.parse_args()

    run_evaluation(
        run=args.run,
        n_contexts=args.n_contexts,
        k=args.k,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        rom_path=args.rom,
        saves_dir=args.saves_dir,
    )


if __name__ == "__main__":
    main()
