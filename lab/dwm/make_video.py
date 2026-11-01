"""Generate side-by-side 'real game vs imagination' videos for Pokemon Red diffusion world model.

CLI:
    python -m lab.dwm.make_video --run main --split test_id --n 6 --k 64 --steps 3 --device cuda --seed 0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float
from lab.dwm.train import windows

ACTION_NAMES = ["down", "left", "right", "up", "a", "b", "start", "pass"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side real vs imagination video for Pokemon Red diffusion world model."
    )
    parser.add_argument("--run", type=str, default="main", help="Run name in work/runs/<run>")
    parser.add_argument("--split", type=str, default="test_id", help="Dataset split (e.g. test_id, test_ood)")
    parser.add_argument("--n", type=int, default=6, help="Number of windows to sample")
    parser.add_argument("--k", type=int, default=64, help="Number of future rollout steps")
    parser.add_argument("--steps", type=int, default=3, help="Denoising steps per frame")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling windows")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path")
    parser.add_argument(
        "--upsample",
        action=argparse.BooleanOptionalAction,
        default=Path("work/upsampler/upsampler.pt").exists(),
        help="Pass frames through neural upsampler (default: on if checkpoint exists)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = args.device

    # Set up output paths
    if args.out is None:
        out_path = Path("work/videos") / f"{args.run}_{args.split}_k{args.k}.mp4"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path = out_path.parent / f"{out_path.stem}_sheet.png"

    # 1. Load data and choose windows
    data, splits = load()
    if args.split not in splits:
        raise KeyError(f"Split '{args.split}' not found in splits. Available: {list(splits.keys())}")
    eps = splits[args.split]

    cand = windows(data, eps, CTX + args.k)
    if len(cand) == 0:
        raise ValueError(f"No valid windows of length {CTX + args.k} found for split '{args.split}'")

    xs, ys = data["xs"], data["ys"]
    in_battles = data["in_battles"]

    # Filter criteria: last context frame not in battle, RAM position changes >= 5 times in future K steps
    not_in_battle = ~in_battles[cand + CTX - 1].astype(bool)

    if args.k > 0:
        idx_curr = cand[:, None] + (CTX + np.arange(args.k))
        idx_prev = cand[:, None] + (CTX - 1 + np.arange(args.k))
        pos_changed = (xs[idx_curr] != xs[idx_prev]) | (ys[idx_curr] != ys[idx_prev])
        num_moves = pos_changed.sum(axis=1)
    else:
        num_moves = np.zeros(len(cand), dtype=int)

    # Prefer interesting windows (not in battle and >= 5 movements)
    mask = not_in_battle & (num_moves >= 5)
    if mask.sum() < args.n:
        mask = not_in_battle & (num_moves >= min(5, args.k))
    if mask.sum() < args.n:
        mask = not_in_battle
    if mask.sum() < args.n:
        mask = np.ones(len(cand), dtype=bool)

    filtered_cand = cand[mask]
    rng = np.random.default_rng(args.seed)
    replace = len(filtered_cand) < args.n
    chosen = rng.choice(filtered_cand, size=args.n, replace=replace)

    print(f"Chosen {len(chosen)} windows (split='{args.split}'):")
    print(f"  Start indices: {chosen.tolist()}")
    print(f"  Episode IDs:   {data['episode_id'][chosen].tolist()}")
    print(f"  Map IDs:       {data['map_ids'][chosen].tolist()}")
    for i, s in enumerate(chosen):
        ep = int(data["episode_id"][s])
        map_id = int(data["map_ids"][s])
        cand_match = np.where(cand == s)[0]
        moves = int(num_moves[cand_match[0]]) if len(cand_match) > 0 and args.k > 0 else 0
        print(f"  [{i}] start: {s}, episode_id: {ep}, map_id: {map_id}, pos_changes: {moves}")

    # 2. Load model and rollout in one batch
    ckpt_path = Path("work/runs") / args.run / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    m = Denoiser()
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    m.load_state_dict(state["ema"])
    m.to(dev).eval()

    idx_ctx = chosen[:, None] + np.arange(CTX)
    ctx_u8 = data["frames"][idx_ctx]
    ctx = to_float(torch.from_numpy(ctx_u8).to(dev))

    idx_act = chosen[:, None] + np.arange(CTX + args.k - 1)
    actions = torch.from_numpy(data["actions"][idx_act]).long().to(dev)

    is_cuda = torch.device(dev).type == "cuda"
    with torch.no_grad():
        if is_cuda:
            with torch.autocast("cuda", torch.bfloat16):
                pred = m.rollout(ctx, actions, steps=args.steps)
        else:
            pred = m.rollout(ctx, actions, steps=args.steps)

    # Convert rollout predictions from float [-1, 1] to uint8 [0, 255]
    pred_u8 = ((pred.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Pre-extract real uint8 frames for t = 0..K
    idx_real = chosen[:, None] + (CTX - 1 + np.arange(args.k + 1))
    real_u8 = data["frames"][idx_real]  # shape (n, k + 1, 72, 80)

    if args.upsample:
        from lab.upsampler import load_upsampler, upsample
        upsampler_m = load_upsampler("work/upsampler/upsampler.pt", device=dev)
        real_flat = real_u8.reshape(-1, 72, 80)
        real_up = upsample(upsampler_m, real_flat, device=dev).reshape(args.n, args.k + 1, 144, 160)
        pred_flat = pred_u8.reshape(-1, 72, 80)
        pred_up = upsample(upsampler_m, pred_flat, device=dev).reshape(args.n, args.k, 144, 160)

    # 3. Compose video frames t = 0..K
    panel_h, panel_w = 288, 320
    sep_px = 6
    header_h, footer_h = 28, 28
    total_w = panel_w * 2 + sep_px
    grid_h = args.n * panel_h + (args.n - 1) * sep_px
    total_h = header_h + grid_h + footer_h

    # Static top header bar
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    size_real, _ = cv2.getTextSize("real", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x_real = (panel_w - size_real[0]) // 2
    y_real = (header_h + size_real[1]) // 2
    cv2.putText(header, "real", (x_real, y_real), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    size_imag, _ = cv2.getTextSize("imagined", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x_imag = (panel_w + sep_px) + (panel_w - size_imag[0]) // 2
    y_imag = (header_h + size_imag[1]) // 2
    cv2.putText(header, "imagined", (x_imag, y_imag), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if args.upsample:
        tag = "upsampled for display"
        size_tag, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        x_tag = (total_w - size_tag[0]) // 2
        y_tag = (header_h + size_tag[1]) // 2
        cv2.putText(header, tag, (x_tag, y_tag), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    sep_v = np.zeros((panel_h, sep_px), dtype=np.uint8)
    sep_h = np.zeros((sep_px, total_w), dtype=np.uint8)

    composed_frames = []
    for t in range(args.k + 1):
        row_images = []
        for i in range(args.n):
            if args.upsample:
                rf = real_up[i, t]
                imf = real_up[i, 0] if t == 0 else pred_up[i, t - 1]
            else:
                rf = real_u8[i, t]
                imf = real_u8[i, 0] if t == 0 else pred_u8[i, t - 1]

            rf_disp = cv2.resize(rf, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
            imf_disp = cv2.resize(imf, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

            row = np.concatenate([rf_disp, sep_v, imf_disp], axis=1)
            row_images.append(row)

        grid_rows = []
        for i, r in enumerate(row_images):
            if i > 0:
                grid_rows.append(sep_h)
            grid_rows.append(r)
        grid_gray = np.concatenate(grid_rows, axis=0)
        grid_bgr = cv2.cvtColor(grid_gray, cv2.COLOR_GRAY2BGR)

        # Footer bar with step t / K and action pressed
        footer = np.zeros((footer_h, total_w, 3), dtype=np.uint8)
        if t == 0:
            footer_text = f"step 0 / {args.k}"
        else:
            if args.n == 1:
                act_idx = int(data["actions"][chosen[0] + CTX - 2 + t])
                footer_text = f"step {t} / {args.k}  |  action: {ACTION_NAMES[act_idx]}"
            else:
                acts = [ACTION_NAMES[int(data["actions"][s + CTX - 2 + t])] for s in chosen]
                if len(set(acts)) == 1:
                    footer_text = f"step {t} / {args.k}  |  action: {acts[0]}"
                else:
                    footer_text = f"step {t} / {args.k}  |  actions: " + ", ".join(f"w{i}:{a}" for i, a in enumerate(acts))

        # Dynamically scale font if text is too wide
        scale = 0.45
        size_foot, _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        while size_foot[0] > total_w - 16 and scale > 0.25:
            scale -= 0.05
            size_foot, _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

        x_foot = max(8, (total_w - size_foot[0]) // 2)
        y_foot = (footer_h + size_foot[1]) // 2
        cv2.putText(footer, footer_text, (x_foot, y_foot), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

        full_frame = np.concatenate([header, grid_bgr, footer], axis=0)
        composed_frames.append(full_frame)

    # 4. Write MP4 with fps=4 and 1 s hold on first and last frames
    fps = 4
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    # Hold first frame for 1 s (fps frames)
    for _ in range(1 + fps):
        writer.write(composed_frames[0])

    # Middle frames t = 1 .. K - 1
    for t in range(1, args.k):
        writer.write(composed_frames[t])

    # Hold last frame for 1 s (fps frames)
    if args.k > 0:
        for _ in range(1 + fps):
            writer.write(composed_frames[args.k])

    writer.release()

    # Save PNG contact sheet (columns: 0, 1, 2, 4, 8, 16, 32, 64 clipped to K; 2 rows per window)
    cols = sorted(list(set(min(c, args.k) for c in [0, 1, 2, 4, 8, 16, 32, 64])))
    sheet_rows = []
    for i in range(args.n):
        if args.upsample:
            row_real = np.concatenate([real_up[i, t] for t in cols], axis=1)
            row_imag = np.concatenate([real_up[i, 0] if t == 0 else pred_up[i, t - 1] for t in cols], axis=1)
        else:
            row_real = np.concatenate([real_u8[i, t] for t in cols], axis=1)
            row_imag = np.concatenate([real_u8[i, 0] if t == 0 else pred_u8[i, t - 1] for t in cols], axis=1)
        sheet_rows.append(row_real)
        sheet_rows.append(row_imag)

    sheet_grid = np.concatenate(sheet_rows, axis=0)
    if args.upsample:
        sheet_up = sheet_grid
    else:
        sheet_up = cv2.resize(sheet_grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(sheet_path), sheet_up)

    # 5. Print output paths and frame size
    print(f"Video saved to: {out_path}")
    print(f"Contact sheet saved to: {sheet_path}")
    print(f"Frame size: {total_w}x{total_h} (WxH)")


if __name__ == "__main__":
    main()
