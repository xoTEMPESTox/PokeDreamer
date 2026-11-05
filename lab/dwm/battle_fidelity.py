"""Battle fidelity evaluation for diffusion world models.

Measures how accurately a world model simulates Pokemon battles when starting from real
in-battle contexts:
1. Battle end: does the model keep showing the battle screen and end it when the real battle ends?
2. Battle-screen pixels: pixel MSE inside battles vs copy-last-frame baseline across horizon k.
3. Party HP context: real player party HP fraction trajectory and trend.

Usage: python -m lab.dwm.battle_fidelity --run wm31 --data ext
Writes work/eval/battle_fidelity_<run>_<data>.json and work/plots/battle_fidelity_<run>_<data>.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lab.data import EXT_SRC, SRC, load
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.dwm.train import windows
from lab.eval.metrics import bootstrap_ci, copy_last_frame_mse, frame_mse, psnr
from lab.probe import load_probe, predict


def battle_end_step(in_battle: np.ndarray) -> int | None:
    """First future step index (0-indexed, 0..H-1) where battle is over (~in_battle), or None if it never ends within H."""
    ended = ~np.asarray(in_battle, dtype=bool)
    if not np.any(ended):
        return None
    return int(np.argmax(ended))


def classify_battle_end_pair(r_end: int | None, p_end: int | None, tol: int = 3) -> str:
    """Classify a single window's battle end timing into one of 4 mutually exclusive categories:
    - 'within_pm3': real ends within H and predicted ends within +-tol steps
    - 'hallucinated_end': predicted ends earlier than real by > tol steps, or predicted ends when real never ends
    - 'later_or_never': predicted ends later than real by > tol steps, or real ends and predicted never ends
    - 'neither': neither ends within H
    """
    if r_end is None and p_end is None:
        return "neither"
    if r_end is None:
        return "hallucinated_end"
    if p_end is None:
        return "later_or_never"
    diff = p_end - r_end
    if abs(diff) <= tol:
        return "within_pm3"
    elif diff < -tol:
        return "hallucinated_end"
    else:
        return "later_or_never"


def classify_battle_ends(ram_ends: list[int | None] | np.ndarray, pred_ends: list[int | None] | np.ndarray,
                         tol: int = 3) -> dict:
    """Batch classification of battle-end timings and median absolute error where both end."""
    n = len(ram_ends)
    cases = [classify_battle_end_pair(r, p, tol) for r, p in zip(ram_ends, pred_ends)]
    counts = {c: cases.count(c) for c in ("within_pm3", "hallucinated_end", "later_or_never", "neither")}
    fractions = {c: float(counts[c] / n) if n else 0.0 for c in counts}
    errors = [abs(p - r) for r, p in zip(ram_ends, pred_ends) if r is not None and p is not None]
    median_err = float(np.median(errors)) if errors else None
    return {
        "counts": counts,
        "fractions": fractions,
        "both_end_count": len(errors),
        "both_end_fraction": float(len(errors) / n) if n else 0.0,
        "median_abs_error": median_err,
        "cases": cases,
        "errors": errors,
    }


def load_party_hp(extended: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load raw party_hps and party_max_hps matching the exact order/length of lab.data.load()."""
    files = sorted(glob.glob(SRC))
    if extended:
        files += sorted(f for pattern in EXT_SRC for f in glob.glob(pattern))
    hps, max_hps = [], []
    for f in files:
        d = np.load(f)
        hps.append(d["party_hps"])
        max_hps.append(d["party_max_hps"])
    return np.concatenate(hps), np.concatenate(max_hps)


def compute_hp_context(starts: np.ndarray, H: int, extended: bool = False, reported_ks: list[int] | None = None) -> dict:
    """Compute real player party HP fraction statistics and trend per window."""
    hps, max_hps = load_party_hp(extended=extended)
    t0 = starts + CTX - 1
    fut = starts[:, None] + CTX + np.arange(H)
    # Total party HP divided by total party max HP
    hp0 = hps[t0].sum(axis=-1).astype(float)
    max0 = np.maximum(max_hps[t0].sum(axis=-1).astype(float), 1.0)
    frac0 = hp0 / max0

    fut_hp = hps[fut].sum(axis=-1).astype(float)
    fut_max = np.maximum(max_hps[fut].sum(axis=-1).astype(float), 1.0)
    fut_frac = fut_hp / fut_max                                       # (W, H)

    delta = fut_frac[:, -1] - frac0
    ks = reported_ks or [k for k in [1, 2, 4, 8, 16, 32] if k <= H]
    return {
        "included": True,
        "hp_fraction_start": {"mean": float(np.mean(frac0)), "std": float(np.std(frac0))},
        "hp_fraction_end": {"mean": float(np.mean(fut_frac[:, -1])), "std": float(np.std(fut_frac[:, -1]))},
        "delta_hp_fraction": {"mean": float(np.mean(delta)), "std": float(np.std(delta))},
        "fraction_windows_hp_decreased": float(np.mean(delta < -1e-4)),
        "fraction_windows_hp_unchanged": float(np.mean(np.abs(delta) <= 1e-4)),
        "fraction_windows_hp_increased": float(np.mean(delta > 1e-4)),
        "mean_hp_fraction_at_k": {str(k): float(np.mean(fut_frac[:, k - 1])) for k in ks},
    }


def plot_battle_fidelity(summary: dict, out_path: Path):
    """Plot pixel MSE vs horizon k inside battles with bootstrap 95% CIs."""
    mse_info = summary["pixel_mse"]
    ks = np.arange(1, len(mse_info["curves"]["imagined"]["mean"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    m_img = mse_info["curves"]["imagined"]
    m_copy = mse_info["curves"]["copy_last"]

    ax.plot(ks, m_img["mean"], label=f"Imagined ({summary['run']})", color="#1f77b4", lw=2)
    ax.fill_between(ks, m_img["lo"], m_img["hi"], color="#1f77b4", alpha=0.25)

    ax.plot(ks, m_copy["mean"], label="Copy-last-frame baseline", color="#ff7f0e", lw=2, linestyle="--")
    ax.fill_between(ks, m_copy["lo"], m_copy["hi"], color="#ff7f0e", alpha=0.25)

    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Pixel MSE (inside battle)")
    ax.set_title(f"Battle Fidelity Pixel MSE: {summary['run']} ({summary['data']}) | {summary['windows']} windows")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_battle_fidelity_summary(summary: dict):
    """Print human-readable summary table of battle fidelity results."""
    print("=" * 78)
    print(f"BATTLE FIDELITY EVALUATION: run={summary['run']} data={summary['data']} split={summary['split']}")
    print(f"Windows: {summary['windows']} (from {summary['eligible_windows']} eligible) | Episodes: {summary['episodes']} | H={summary['horizon']}")
    print("-" * 78)
    print("Battle End Timing (fractions of windows):")
    b_ceil = summary["battle_end"]["probe_ceiling_real_frames"]
    b_img = summary["battle_end"]["imagined_frames"]
    print(f"  Within +-3 steps of real end:   real probe={b_ceil['within_pm3']:.3f} | imagined={b_img['within_pm3']:.3f}")
    print(f"  Hallucinated end (>3 steps early): real probe={b_ceil['hallucinated_end']:.3f} | imagined={b_img['hallucinated_end']:.3f}")
    print(f"  Later or never:                  real probe={b_ceil['later_or_never']:.3f} | imagined={b_img['later_or_never']:.3f}")
    print(f"  Neither ends within H:           real probe={b_ceil['neither']:.3f} | imagined={b_img['neither']:.3f}")
    med_ceil = f"{b_ceil['median_abs_error']:.1f}" if b_ceil['median_abs_error'] is not None else "n/a"
    med_img = f"{b_img['median_abs_error']:.1f}" if b_img['median_abs_error'] is not None else "n/a"
    print(f"  Median abs timing error (both end): real probe={med_ceil} | imagined={med_img}")
    print("-" * 78)
    print("Battle-Screen Pixel MSE inside battles (vs Copy-Last):")
    r_ks = summary["pixel_mse"]["reported_k"]
    hdr = f"{'k':<6}{'Imagined MSE [95% CI]':<28}{'Copy-Last MSE [95% CI]':<28}{'In-Battle Frac':<16}"
    print(hdr)
    for k in r_ks:
        k_str = str(k)
        info = summary["pixel_mse"]["at_k"][k_str]
        m_i = info["mse_imagined"]
        m_c = info["mse_copy_last"]
        ib = info["in_battle_fraction"]
        str_i = f"{m_i['mean']:.4f} [{m_i['lo']:.4f},{m_i['hi']:.4f}]"
        str_c = f"{m_c['mean']:.4f} [{m_c['lo']:.4f},{m_c['hi']:.4f}]"
        print(f"k={k:<4}{str_i:<28}{str_c:<28}{ib:<16.2f}")
    if summary.get("hp_context", {}).get("included"):
        hp = summary["hp_context"]
        print("-" * 78)
        print(f"Party HP Context: start frac={hp['hp_fraction_start']['mean']:.3f}, end frac={hp['hp_fraction_end']['mean']:.3f}, "
              f"delta={hp['delta_hp_fraction']['mean']:.3f} (dropped in {hp['fraction_windows_hp_decreased']:.1%} of windows)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Evaluate world model fidelity during Pokemon battles.")
    ap.add_argument("--run", default="wm31", help="world model run name (work/runs/<run>)")
    ap.add_argument("--data", choices=["base", "ext"], default="ext", help="base or extended dataset")
    ap.add_argument("--probe", default=None, help="path to probe checkpoint")
    ap.add_argument("--split", default=None, help="held-out split (default: test_ext for ext, test_id for base)")
    ap.add_argument("--horizon", type=int, default=32, help="rollout horizon H")
    ap.add_argument("--samples", type=int, default=4, help="samples per context")
    ap.add_argument("--steps", type=int, default=3, help="Euler sampling steps")
    ap.add_argument("--device", default="cuda", help="eval device (cuda or cpu)")
    ap.add_argument("--max-windows", type=int, default=0, help="subsample max windows (0 = all)")
    ap.add_argument("--batch", type=int, default=128, help="batch size for rollout and probe")
    ap.add_argument("--seed", type=int, default=0, help="random seed for window sampling and rollouts")
    ap.add_argument("--tag", default="", help="optional tag appended to output filenames")
    args = ap.parse_args()

    H, dev = args.horizon, args.device
    split = args.split or ("test_ext" if args.data == "ext" else "test_id")
    probe_path = args.probe or ("work/probe_ext/probe.pt" if args.data == "ext" else "work/probe/probe.pt")

    print(f"Loading data (data={args.data}, split={split})...")
    data, splits = load(extended=args.data == "ext")
    all_starts = windows(data, splits[split], CTX + H)
    in_b = data["in_battles"][all_starts[:, None] + np.arange(CTX)].all(axis=1)
    eligible_starts = all_starts[in_b]
    n_eligible = len(eligible_starts)
    n_eligible_eps = len(set(data["episode_id"][eligible_starts].tolist()))
    print(f"Found {n_eligible} in-battle windows across {n_eligible_eps} episodes in {split}.")

    starts = eligible_starts
    if args.max_windows and len(starts) > args.max_windows:
        rng = np.random.default_rng(args.seed)
        starts = np.sort(rng.choice(starts, size=args.max_windows, replace=False))
        print(f"Subsampled to {len(starts)} windows (max_windows={args.max_windows}, seed={args.seed}).")

    fut_idx = starts[:, None] + CTX + np.arange(H)
    real_frames = data["frames"][fut_idx]                             # (W, H, 72, 80) uint8
    ram_battle = data["in_battles"][fut_idx].astype(bool)             # (W, H) bool
    ram_ends = [battle_end_step(ram_battle[w]) for w in range(len(starts))]

    print(f"Loading probe from {probe_path} on {dev}...")
    probe = load_probe(probe_path, device=dev)
    print("Evaluating probe ceiling on real frames...")
    real_probe_preds = predict(probe, real_frames.reshape(-1, 72, 80), device=dev, batch=1024)
    real_probe_battle = real_probe_preds["battle"].reshape(len(starts), H)
    real_probe_ends = [battle_end_step(real_probe_battle[w]) for w in range(len(starts))]
    ceiling_eval = classify_battle_ends(ram_ends, real_probe_ends)

    print(f"Loading world model {args.run} on {dev}...")
    model = load_world_model(args.run, device=dev)

    # Rollouts and evaluation across samples
    imag_samples_battle = []                                          # list of (W, H) bool
    imag_samples_mse = []                                             # list of (W, H) float
    real_k_01 = real_frames.astype(np.float32) / 255.0
    ctx_0_01 = data["frames"][starts + CTX - 1].astype(np.float32) / 255.0
    real_all_01 = np.concatenate([ctx_0_01[:, None], real_k_01], axis=1)
    mse_copy_all = copy_last_frame_mse(real_all_01)                   # (W, H)
    mse_copy_battle = np.where(ram_battle, mse_copy_all, np.nan)

    for rep in range(args.samples):
        print(f"Rollout sample {rep + 1}/{args.samples}...")
        torch.manual_seed(args.seed + rep)
        out_u8 = []
        for i in range(0, len(starts), args.batch):
            s = starts[i : i + args.batch]
            ctx = to_float(torch.from_numpy(data["frames"][s[:, None] + np.arange(CTX)]).to(dev))
            act = torch.from_numpy(data["actions"][s[:, None] + np.arange(CTX + H - 1)]).long().to(dev)
            with torch.autocast("cuda", torch.bfloat16, enabled="cuda" in str(dev)):
                frames = model.rollout(ctx, act, steps=args.steps)
            u8 = ((frames.float() + 1.0) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
            out_u8.append(u8)
        sample_frames = np.concatenate(out_u8, axis=0)                # (W, H, 72, 80) uint8
        preds = predict(probe, sample_frames.reshape(-1, 72, 80), device=dev, batch=1024)
        sample_battle = preds["battle"].reshape(len(starts), H)
        imag_samples_battle.append(sample_battle)

        sample_01 = sample_frames.astype(np.float32) / 255.0
        mse_sample = frame_mse(sample_01, real_k_01)
        imag_samples_mse.append(mse_sample)

    # Battle end metrics for imagined frames
    sample_evals = []
    all_imag_errors = []
    for rep in range(args.samples):
        rep_ends = [battle_end_step(imag_samples_battle[rep][w]) for w in range(len(starts))]
        rep_ev = classify_battle_ends(ram_ends, rep_ends)
        sample_evals.append(rep_ev)
        all_imag_errors.extend(rep_ev["errors"])

    imag_mean_fractions = {
        c: float(np.mean([ev["fractions"][c] for ev in sample_evals]))
        for c in ("within_pm3", "hallucinated_end", "later_or_never", "neither")
    }
    imag_median_err = float(np.median(all_imag_errors)) if all_imag_errors else None
    imag_both_end_frac = float(np.mean([ev["both_end_fraction"] for ev in sample_evals]))

    # Pixel MSE inside battles: average across samples first, then mask with RAM in_battle
    mse_imag_avg = np.mean(imag_samples_mse, axis=0)                  # (W, H)
    mse_imag_battle = np.where(ram_battle, mse_imag_avg, np.nan)

    mean_img, lo_img, hi_img = bootstrap_ci(mse_imag_battle, n_boot=1000, seed=args.seed)
    mean_copy, lo_copy, hi_copy = bootstrap_ci(mse_copy_battle, n_boot=1000, seed=args.seed)

    reported_ks = [k for k in [1, 2, 4, 8, 16, 32] if k <= H]
    at_k_dict = {}
    for k in reported_ks:
        idx = k - 1
        at_k_dict[str(k)] = {
            "mse_imagined": {"mean": float(mean_img[idx]), "lo": float(lo_img[idx]), "hi": float(hi_img[idx])},
            "mse_copy_last": {"mean": float(mean_copy[idx]), "lo": float(lo_copy[idx]), "hi": float(hi_copy[idx])},
            "psnr_imagined": float(psnr(mean_img[idx])),
            "psnr_copy_last": float(psnr(mean_copy[idx])),
            "in_battle_fraction": float(ram_battle[:, idx].mean()),
        }

    # Party HP context
    hp_ctx = compute_hp_context(starts, H, extended=args.data == "ext", reported_ks=reported_ks)

    tag_str = f"_{args.tag}" if args.tag else ""
    stem = f"battle_fidelity_{args.run}_{args.data}{tag_str}"

    summary = {
        "run": args.run,
        "data": args.data,
        "split": split,
        "probe": str(probe_path),
        "horizon": H,
        "samples": args.samples,
        "steps": args.steps,
        "eligible_windows": int(n_eligible),
        "eligible_episodes": int(n_eligible_eps),
        "windows": int(len(starts)),
        "episodes": int(len(set(data["episode_id"][starts].tolist()))),
        "battle_end": {
            "probe_ceiling_real_frames": {
                "within_pm3": ceiling_eval["fractions"]["within_pm3"],
                "hallucinated_end": ceiling_eval["fractions"]["hallucinated_end"],
                "later_or_never": ceiling_eval["fractions"]["later_or_never"],
                "neither": ceiling_eval["fractions"]["neither"],
                "both_end_fraction": ceiling_eval["both_end_fraction"],
                "median_abs_error": ceiling_eval["median_abs_error"],
            },
            "imagined_frames": {
                "within_pm3": imag_mean_fractions["within_pm3"],
                "hallucinated_end": imag_mean_fractions["hallucinated_end"],
                "later_or_never": imag_mean_fractions["later_or_never"],
                "neither": imag_mean_fractions["neither"],
                "both_end_fraction": imag_both_end_frac,
                "median_abs_error": imag_median_err,
            },
        },
        "pixel_mse": {
            "reported_k": reported_ks,
            "at_k": at_k_dict,
            "curves": {
                "imagined": {"mean": mean_img.tolist(), "lo": lo_img.tolist(), "hi": hi_img.tolist()},
                "copy_last": {"mean": mean_copy.tolist(), "lo": lo_copy.tolist(), "hi": hi_copy.tolist()},
            },
        },
        "hp_context": hp_ctx,
    }

    out_json = Path("work/eval") / f"{stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote results to {out_json}")

    out_plot = Path("work/plots") / f"{stem}.png"
    plot_battle_fidelity(summary, out_plot)
    print(f"Wrote plot to {out_plot}")

    print_battle_fidelity_summary(summary)


if __name__ == "__main__":
    main()
