"""Training-throughput microbenchmarks (speed only; nothing is saved). Results 2026-11-01: work/bench_speed.log
    python -m lab.bench_speed 64 128 192 > work/bench_speed.log
World model: batch x cudnn.benchmark x fused AdamW, same step as lab/dwm/train.py.
Upsampler (batch 64): current loop (numpy indexing on CPU, host->GPU copy, loss.item() every step, as in
lab/upsampler.py) vs dataset resident on the GPU without the per-step sync; each with cudnn.benchmark off/on.
"""
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float
from lab.dwm.train import windows
from lab.upsampler import Upsampler

WARMUP, STEPS = 30, 200
dev = "cuda"


def timeit(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    return STEPS / (time.time() - t)


def world_model(frames, actions, starts, batch, fused):
    torch.manual_seed(0)
    model = Denoiser().to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2, eps=1e-8, fused=fused)
    ar = torch.arange(CTX + 1, device=dev)

    def step():
        idx = starts[torch.randint(len(starts), (batch,), device=dev)][:, None] + ar
        f = to_float(frames[idx])
        with torch.autocast("cuda", torch.bfloat16):
            loss = model.loss(f, actions[idx][:, :CTX].long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1e-3)

    return timeit(step)


def upsampler(frames_np, classes_np, train_idx, gpu_data, batch=64):
    torch.manual_seed(0)
    model = Upsampler().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    if gpu_data:
        fr, cl = torch.from_numpy(frames_np).to(dev), torch.from_numpy(classes_np).to(dev)
        ti = torch.from_numpy(train_idx).to(dev)

    def step():
        if gpu_data:
            b = ti[torch.randint(len(ti), (batch,), device=dev)]
            x, y = fr[b].float().unsqueeze(1) / 255.0, cl[b].long()
        else:
            b = rng.choice(train_idx, size=batch, replace=True)
            x = torch.from_numpy(frames_np[b]).to(device=dev, dtype=torch.float32).unsqueeze(1) / 255.0
            y = torch.from_numpy(classes_np[b]).to(device=dev, dtype=torch.long)
        sigma = torch.rand(batch, 1, 1, 1, device=dev) * 0.05
        x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not gpu_data:
            loss.item()

    return timeit(step)


def main():
    batches = [int(b) for b in sys.argv[1:]] or [64]
    data, splits = load()
    frames = torch.from_numpy(data["frames"]).to(dev)
    actions = torch.from_numpy(data["actions"]).to(dev)
    starts = torch.from_numpy(windows(data, splits["train"], CTX + 1)).to(dev)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, warmup {WARMUP}, timed steps {STEPS}", flush=True)
    for bench in (False, True):  # heuristics first; once on, cuDNN caches the tuned algorithms
        torch.backends.cudnn.benchmark = bench
        for batch in batches:
            for fused in (False, True):
                torch.cuda.reset_peak_memory_stats()
                sps = world_model(frames, actions, starts, batch, fused)
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                torch.cuda.empty_cache()
                print(f"world model batch {batch:3d} cudnn.benchmark {bench!s:5} fused {fused!s:5} "
                      f"{sps:6.2f} steps/s {sps * batch:7.0f} samples/s peak mem {mem:.1f} GB", flush=True)
    del frames, actions, starts
    torch.cuda.empty_cache()

    classes = np.load("work/native_classes.npy")
    train_idx = np.where(np.isin(data["episode_id"], splits["train"]))[0]
    for bench in (False, True):
        torch.backends.cudnn.benchmark = bench
        for gpu_data in (False, True):
            sps = upsampler(data["frames"], classes, train_idx, gpu_data)
            torch.cuda.empty_cache()
            print(f"upsampler batch 64 cudnn.benchmark {bench!s:5} data {'gpu' if gpu_data else 'cpu+item'} "
                  f"{sps:6.2f} steps/s", flush=True)


if __name__ == "__main__":
    main()
