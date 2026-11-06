"""World-model frames aligned step by step with a recorded quest trajectory (for the real-vs-dream video).

Every K steps the model is re-anchored on the last 4 real frames and imagines the next K frames open-loop with
the buttons the agent actually pressed (the same scheme as lab/dwm/dream_gif.py). All chunks are independent, so
they run as batches. Output: the imagined 72x80 frame for every step in [start, end) and a flag for the first
step of each chunk.

Usage: python -m lab.video.dream --traj work/video/traj/run_seed_100.npz --start 6833 --end 7724 --k 16
       (writes work/video/dream_<start>_<end>_k<k>.npz)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from lab.dwm.model import CTX, to_float


def dream_frames(model, frames, actions, start, end, k=16, steps=3, device="cuda", batch=64):
    """frames (N,72,80) uint8 and actions (N,) of a trajectory -> imagined (end-start,72,80) uint8, anchor flags."""
    start = max(start, CTX)
    chunk_starts = list(range(start, end, k))
    out = np.zeros((end - start, *frames.shape[1:]), np.uint8)
    anchor = np.zeros(end - start, bool)
    for b0 in range(0, len(chunk_starts), batch):
        group = chunk_starts[b0:b0 + batch]
        lens = [min(k, end - c) for c in group]
        kk = max(lens)
        ctx = np.stack([frames[c - CTX:c] for c in group])
        # actions between context frames, then one per imagined step; pad the last (short) chunk with its final action
        acts = np.stack([np.pad(actions[c - CTX:c + n - 1], (0, kk - n), mode="edge") for c, n in zip(group, lens)])
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled="cuda" in str(device)):
            img = model.rollout(to_float(torch.from_numpy(ctx).to(device)), torch.from_numpy(acts).long().to(device),
                                steps=steps)
        img = np.clip(np.round((img.float().cpu().numpy() + 1) * 127.5), 0, 255).astype(np.uint8)
        for c, n, im in zip(group, lens, img):
            out[c - start:c - start + n] = im[:n]
            anchor[c - start] = True
    return out, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--run", default="wm31")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from lab.dwm.controllability import load_world_model
    t = np.load(args.traj)
    model = load_world_model(args.run, device=args.device)
    out, anchor = dream_frames(model, t["frames"], t["actions"].astype(np.int64), args.start, args.end, args.k,
                               args.steps, args.device)
    path = Path("work/video") / f"dream_{args.start}_{args.end}_k{args.k}.npz"
    np.savez_compressed(path, dream=out, anchor=anchor, start=max(args.start, CTX), end=args.end, k=args.k)
    print(f"wrote {path}: {len(out)} frames, {int(anchor.sum())} anchors")


if __name__ == "__main__":
    main()
