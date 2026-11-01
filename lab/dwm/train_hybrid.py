"""Train the hybrid world model (HybridDenoiser + RamDynamics) on random windows.

Usage: python -m lab.dwm.train_hybrid --name hybrid --minutes 60
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
from lab.dwm.hybrid import HybridDenoiser, RamDynamics
from lab.dwm.model import CTX, to_float
from lab.dwm.train import (
    BATCH,
    CLIP,
    CKPT_MINUTES,
    EMA,
    EVAL_EVERY,
    EVAL_K,
    EVAL_WINDOWS,
    GPU_CHECK_EVERY,
    LR,
    PAUSE,
    WD,
    windows,
)


class GpuGuard:
    """Logs power/temperature; pauses at >=80C until <=70C; asks to stop at >=87C or >5 pauses in an hour."""

    def __init__(self, log, enabled=True):
        self.log, self.pauses, self.enabled = log, deque(), enabled
        if self.enabled:
            limit, _, _ = self.read()
            log(f"gpu power limit {limit:.0f} W (set by the owner)")

    @staticmethod
    def read():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        return [float(v) for v in out.strip().split(",")]

    def check(self):
        """Returns (power, temp, stop)."""
        if not self.enabled:
            return 0.0, 0.0, False
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


@torch.no_grad()
def evaluate(ema, ram_model, frames, actions, ram_dict, starts, out_png, device="cuda", eval_k=EVAL_K):
    """Open-loop rollout MSE per horizon on fixed held-out windows vs the copy-last-frame baseline."""
    k_eval = min(eval_k, EVAL_K)
    idx = starts[:, None] + torch.arange(CTX + k_eval, device=frames.device)
    f = to_float(frames[idx])
    ctx_starts = starts[:, None] + torch.arange(CTX, device=frames.device)
    ctx_ram = {k: v[ctx_starts] for k, v in ram_dict.items()}

    if "cuda" in str(device):
        with torch.autocast("cuda", torch.bfloat16):
            pred, _ = ema.rollout(
                f[:, :CTX], ctx_ram, actions[idx][:, :CTX + k_eval - 1].long(), ram_model=ram_model
            )
    else:
        pred, _ = ema.rollout(
            f[:, :CTX], ctx_ram, actions[idx][:, :CTX + k_eval - 1].long(), ram_model=ram_model
        )

    true = f[:, CTX:]
    mse = ((pred - true) ** 2).mean((0, 2, 3)).cpu().numpy()
    copy_last = ((f[:, CTX - 1:CTX] - true) ** 2).mean((0, 2, 3)).cpu().numpy()
    cols = [c for c in [0, 1, 3, 7, 15, 31] if c < k_eval]
    rows = []
    for w in range(min(4, len(starts))):
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
    ap.add_argument("--no-ctx-aug", action="store_true", default=True, help="train without context-noise augmentation (default)")
    ap.add_argument("--ctx-aug", dest="no_ctx_aug", action="store_false", help="enable context-noise augmentation")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--cond-drop", type=float, default=0.0, help="probability of dropping RAM conditioning during training")
    ap.add_argument("--init-from", type=str, default=None, help="path to existing checkpoint to initialize weights from")
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

    dev = args.device
    guard = GpuGuard(log, enabled=("cuda" in dev))
    data, splits = load()

    frames = torch.from_numpy(data["frames"]).to(dev)
    actions = torch.from_numpy(data["actions"]).to(dev)
    map_ids = torch.from_numpy(data["map_ids"].astype(np.int64)).to(dev)
    xs = torch.from_numpy(data["xs"].astype(np.int64)).to(dev)
    ys = torch.from_numpy(data["ys"].astype(np.int64)).to(dev)
    facings = torch.from_numpy((data["facings"] // 4).astype(np.int64)).to(dev)
    in_battles = torch.from_numpy(data["in_battles"].astype(np.int64)).to(dev)
    ram_data = {"map": map_ids, "x": xs, "y": ys, "facing": facings, "battle": in_battles}

    train_starts = torch.from_numpy(windows(data, splits["train"], CTX + 1)).to(dev)
    rng = np.random.default_rng(0)
    eval_w = EVAL_WINDOWS if "cuda" in dev else 2
    eval_k = EVAL_K if "cuda" in dev else 4
    test_id_windows = windows(data, splits["test_id"], CTX + eval_k)
    n_eval = min(eval_w, len(test_id_windows))
    eval_starts = torch.from_numpy(rng.choice(test_id_windows, n_eval, replace=False)).to(dev)

    model = HybridDenoiser().to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    ram_model = RamDynamics().to(dev)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(ram_model.parameters()),
        lr=LR,
        weight_decay=WD,
        eps=1e-8,
        fused=("cuda" in dev),
    )

    step, elapsed = 0, 0.0
    ckpt = run / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        ram_model.load_state_dict(state["ram"])
        opt.load_state_dict(state["opt"])
        step, elapsed = state["step"], state["elapsed"]
        log(f"resumed at step {step}, {elapsed / 60:.1f} min trained")
    elif args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"init-from checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location=dev, weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        ram_model.load_state_dict(state["ram"])
        log(f"initialized weights from {init_path} (step reset to 0, fresh optimizer)")

    p_rend = sum(p.numel() for p in model.parameters()) / 1e6
    p_ram = sum(p.numel() for p in ram_model.parameters()) / 1e6
    log(
        f"params: renderer {p_rend:.2f}M, ram {p_ram:.2f}M (total {p_rend + p_ram:.2f}M), "
        f"train windows {len(train_starts)}, batch {args.batch}, lr {LR}, budget {args.minutes} min, "
        f"max steps {args.max_steps or '-'}, context-noise aug {'OFF' if args.no_ctx_aug else 'on'}, "
        f"cond-drop {args.cond_drop}"
    )

    def save():
        tmp = run / "ckpt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "ram": ram_model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "elapsed": elapsed,
            },
            tmp,
        )
        os.replace(tmp, ckpt)

    ar = torch.arange(CTX + 1, device=dev)
    t_last, t_ckpt, t_last_check = time.time(), time.time(), time.time()
    losses, rend_losses, ram_losses = [], [], []
    check_every = GPU_CHECK_EVERY if "cuda" in dev else 2

    while elapsed < args.minutes * 60 and not (args.max_steps and step >= args.max_steps):
        idx = train_starts[torch.randint(len(train_starts), (args.batch,), device=dev)][:, None] + ar
        f = to_float(frames[idx])
        acts = actions[idx][:, :CTX].long()
        target_ram = {k: v[idx[:, -1]] for k, v in ram_data.items()}
        ram_ctx = {k: v[idx[:, :CTX]] for k, v in ram_data.items()}

        if "cuda" in dev:
            with torch.autocast("cuda", torch.bfloat16):
                loss_rend = model.loss(f, acts, target_ram, ctx_aug=not args.no_ctx_aug, cond_drop=args.cond_drop)
                loss_ram = ram_model.loss(ram_ctx, acts, target_ram)
                loss = loss_rend + loss_ram
        else:
            loss_rend = model.loss(f, acts, target_ram, ctx_aug=not args.no_ctx_aug, cond_drop=args.cond_drop)
            loss_ram = ram_model.loss(ram_ctx, acts, target_ram)
            loss = loss_rend + loss_ram

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(ram_model.parameters()), CLIP)
        if step < 100:
            for g in opt.param_groups:
                g["lr"] = LR * (step + 1) / 100
        opt.step()
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - EMA)
        step += 1
        losses.append(loss.detach())
        rend_losses.append(loss_rend.detach())
        ram_losses.append(loss_ram.detach())

        now = time.time()
        elapsed += now - t_last
        t_last = now

        if step % check_every == 0:
            sps = check_every / max(now - (t_ckpt if step == check_every else t_last_check), 1e-6)
            t_last_check = now
            power, temp, stop = guard.check()
            mem = (torch.cuda.max_memory_allocated() / 2 ** 30) if "cuda" in dev else 0.0
            l_tot = torch.stack(losses).mean().item()
            l_rend = torch.stack(rend_losses).mean().item()
            l_ram = torch.stack(ram_losses).mean().item()
            log(
                f"step {step} loss {l_tot:.4f} (rend {l_rend:.4f} ram {l_ram:.4f}) {sps:.1f} it/s "
                f"power {power:.0f} W temp {temp:.0f} C mem {mem:.1f} GB elapsed {elapsed / 60:.1f} min"
            )
            losses, rend_losses, ram_losses = [], [], []
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
            t_last = time.time()  # time spent in guard pauses, saves and PAUSE does not count against the budget
        else:
            if step == 1:
                t_last_check = now

        if step % EVAL_EVERY == 0:
            mse, copy_last = evaluate(
                ema,
                ram_model,
                frames,
                actions,
                ram_data,
                eval_starts,
                run / f"rollout_{step:06d}.png",
                device=dev,
                eval_k=eval_k,
            )
            rec = {"step": step, "mse": mse.tolist(), "copy_last_mse": copy_last.tolist()}
            with open(run / "eval.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(
                "eval rollout mse k=1,4,8,16,32: "
                + " ".join(f"{mse[k - 1]:.4f}" for k in (1, 4, 8, 16, 32) if k <= len(mse))
                + " | copy-last: "
                + " ".join(f"{copy_last[k - 1]:.4f}" for k in (1, 4, 8, 16, 32) if k <= len(copy_last))
            )
            t_last = time.time()

    save()
    mse, copy_last = evaluate(
        ema,
        ram_model,
        frames,
        actions,
        ram_data,
        eval_starts,
        run / f"rollout_{step:06d}.png",
        device=dev,
        eval_k=eval_k,
    )
    with open(run / "eval.jsonl", "a") as fh:
        fh.write(json.dumps({"step": step, "mse": mse.tolist(), "copy_last_mse": copy_last.tolist()}) + "\n")
    k_last = min(32, len(mse))
    log(f"DONE step {step}, {elapsed / 60:.1f} min; final rollout mse k=1,{k_last}: {mse[0]:.4f} {mse[k_last - 1]:.4f}")


if __name__ == "__main__":
    main()
