"""Train the diffusion world model on random windows from the train split for a fixed wall-time budget.

Usage: python -m lab.dwm.train --name main --minutes 120
Resumes from work/runs/<name>/ckpt.pt if it exists. Logs to work/runs/<name>/train.log.
"""
import argparse
import copy
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float

BATCH, LR, WD, CLIP, EMA = 64, 2e-4, 1e-2, 1.0, 0.999
# Speed notes for the next run (measured 2026-11-01 with `python -m lab.bench_speed 64 128 192`, RTX 4090):
# - Fused AdamW is the default (since the noaug ablation finished): 11.4 -> 15.2 steps/s at batch 64. The main, main60 and noaug
#   runs used the non-fused optimizer.
# - A bigger batch does not train faster: 970 / 941 / 917 samples/s at batch 64 / 128 / 192 with fused AdamW
#   (peak 5.6 / 9.6 / 13.7 GB). The GPU is already saturated at 64, so spare VRAM is not the bottleneck. Raise
#   --batch only for less gradient noise, and scale LR with it.
# - cudnn.benchmark made no difference (<1%).
EVAL_EVERY, CKPT_MINUTES, GPU_CHECK_EVERY = 5000, 15, 200
EVAL_WINDOWS, EVAL_K = 32, 32
PAUSE = Path("work/PAUSE")  # while this file exists, training checkpoints and idles (time not counted)


class GpuGuard:
    """Logs power/temperature; pauses at >=80C until <=70C; asks to stop at >=87C or >5 pauses in an hour."""

    def __init__(self, log):
        self.log, self.pauses = log, deque()
        limit, _, _ = self.read()
        log(f"gpu power limit {limit:.0f} W (set by the owner)")

    @staticmethod
    def read():
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return [float(v) for v in out.strip().split(",")]

    def check(self):
        """Returns (power, temp, stop)."""
        _, power, temp = self.read()
        if temp >= 87:
            return power, temp, True
        if temp >= 80:
            self.pauses.append(time.time())
            while self.pauses and self.pauses[0] < time.time() - 3600:
                self.pauses.popleft()
            if len(self.pauses) > 5:
                return power, temp, True
            self.log(f"pause: gpu {temp:.0f} C")
            while temp > 70:
                time.sleep(10)
                _, power, temp = self.read()
            self.log(f"resume: gpu {temp:.0f} C")
        return power, temp, False


def windows(data, episodes, length):
    """Start indices i such that frames i..i+length-1 lie in one episode from `episodes`."""
    ep = data["episode_id"]
    i = np.arange(len(ep) - length + 1)
    ok = (ep[i] == ep[i + length - 1]) & np.isin(ep[i], episodes)
    return i[ok]


@torch.no_grad()
def evaluate(ema, frames, actions, starts, out_png):
    """Open-loop rollout MSE per horizon on fixed held-out windows vs the copy-last-frame baseline."""
    idx = starts[:, None] + torch.arange(CTX + EVAL_K, device=frames.device)
    f = to_float(frames[idx])
    with torch.autocast("cuda", torch.bfloat16):
        pred = ema.rollout(f[:, :CTX], actions[idx][:, :CTX + EVAL_K - 1].long())
    true = f[:, CTX:]
    mse = ((pred - true) ** 2).mean((0, 2, 3)).cpu().numpy()
    copy_last = ((f[:, CTX - 1:CTX] - true) ** 2).mean((0, 2, 3)).cpu().numpy()
    cols = [0, 1, 3, 7, 15, 31]
    rows = []
    for w in range(4):
        rows.append(torch.cat([true[w, k] for k in cols], 1))
        rows.append(torch.cat([pred[w, k] for k in cols], 1))
    grid = ((torch.cat(rows, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()

    cv2.imwrite(str(out_png), cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    return mse, copy_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="also stop at this step (0 = no limit)")
    ap.add_argument("--no-ctx-aug", action="store_true", help="ablation: train without context-noise augmentation")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    torch.set_num_threads(4)
    run = Path("work/runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    guard = GpuGuard(log)
    data, splits = load()
    dev = "cuda"
    frames = torch.from_numpy(data["frames"]).to(dev)
    actions = torch.from_numpy(data["actions"]).to(dev)
    train_starts = torch.from_numpy(windows(data, splits["train"], CTX + 1)).to(dev)
    rng = np.random.default_rng(0)
    eval_starts = torch.from_numpy(rng.choice(windows(data, splits["test_id"], CTX + EVAL_K), EVAL_WINDOWS, replace=False)).to(dev)

    model = Denoiser().to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, eps=1e-8, fused=True)
    step, elapsed = 0, 0.0
    ckpt = run / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step, elapsed = state["step"], state["elapsed"]
        log(f"resumed at step {step}, {elapsed / 60:.1f} min trained")
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, train windows {len(train_starts)}, "
        f"batch {args.batch}, lr {LR}, budget {args.minutes} min, max steps {args.max_steps or '-'}, "
        f"context-noise aug {'OFF' if args.no_ctx_aug else 'on'}")

    def save():
        tmp = run / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "elapsed": elapsed}, tmp)
        os.replace(tmp, ckpt)

    ar = torch.arange(CTX + 1, device=dev)
    t_last, t_ckpt, losses = time.time(), time.time(), []
    while elapsed < args.minutes * 60 and not (args.max_steps and step >= args.max_steps):
        idx = train_starts[torch.randint(len(train_starts), (args.batch,), device=dev)][:, None] + ar
        f = to_float(frames[idx])
        with torch.autocast("cuda", torch.bfloat16):
            loss = model.loss(f, actions[idx][:, :CTX].long(), ctx_aug=not args.no_ctx_aug)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if step < 100:
            for g in opt.param_groups:
                g["lr"] = LR * (step + 1) / 100
        opt.step()
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - EMA)
        step += 1
        losses.append(loss.detach())

        if step % GPU_CHECK_EVERY == 0:
            now = time.time()
            elapsed += now - t_last
            sps = GPU_CHECK_EVERY / (now - t_last)
            power, temp, stop = guard.check()
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            log(f"step {step} loss {torch.stack(losses).mean().item():.4f} {sps:.1f} it/s "
                f"power {power:.0f} W temp {temp:.0f} C mem {mem:.1f} GB elapsed {elapsed / 60:.1f} min")
            losses = []
            if stop:
                save()
                log("STOP: gpu temperature guard tripped; checkpoint saved")
                return
            if now - t_ckpt > CKPT_MINUTES * 60:
                save()
                t_ckpt = time.time()
            if PAUSE.exists():
                save()
                log(f"paused ({PAUSE} exists); checkpoint saved, GPU idle")
                while PAUSE.exists():
                    time.sleep(10)
                log("resumed")
            t_last = time.time()
        if step % EVAL_EVERY == 0:
            mse, copy_last = evaluate(ema, frames, actions, eval_starts, run / f"rollout_{step:06d}.png")
            rec = {"step": step, "mse": mse.tolist(), "copy_last_mse": copy_last.tolist()}
            with open(run / "eval.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log("eval rollout mse k=1,4,8,16,32: " + " ".join(f"{mse[k - 1]:.4f}" for k in (1, 4, 8, 16, 32))
                + " | copy-last: " + " ".join(f"{copy_last[k - 1]:.4f}" for k in (1, 4, 8, 16, 32)))
            t_last = time.time()
    save()
    mse, copy_last = evaluate(ema, frames, actions, eval_starts, run / f"rollout_{step:06d}.png")
    with open(run / "eval.jsonl", "a") as fh:
        fh.write(json.dumps({"step": step, "mse": mse.tolist(), "copy_last_mse": copy_last.tolist()}) + "\n")
    log(f"DONE step {step}, {elapsed / 60:.1f} min; final rollout mse k=1,32: {mse[0]:.4f} {mse[-1]:.4f}")


if __name__ == "__main__":
    main()
