"""Native-resolution upsampler for 72x80 greyscale Pokemon Red frames.

Maps half-resolution frames (72x80) to crisp 144x160 Game Boy frames with exactly
4 grey shades {0, 85, 153, 255}.
"""
import argparse
import contextlib
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lab  # handles Windows DLL / OpenMP setup
from lab.data import load

SHADES = np.array([0, 85, 153, 255], dtype=np.uint8)
_LUT = np.argmin(np.abs(np.arange(256)[:, None] - SHADES.astype(np.int32)), axis=1).astype(np.uint8)


def to_classes(grey_uint8):
    """Maps greyscale uint8 (0..255) to nearest shade class index in {0, 1, 2, 3}."""
    if isinstance(grey_uint8, torch.Tensor):
        lut_t = torch.from_numpy(_LUT).to(grey_uint8.device)
        return lut_t[grey_uint8.long()]
    arr = np.asarray(grey_uint8)
    if arr.dtype != np.uint8 and np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return _LUT[arr]


def to_shades(classes):
    """Maps class indices {0, 1, 2, 3} to Game Boy shades {0, 85, 153, 255}."""
    if isinstance(classes, torch.Tensor):
        shades_t = torch.from_numpy(SHADES).to(classes.device)
        return shades_t[classes.long()]
    return SHADES[np.asarray(classes)]


def build_native_cache(out_path="work/native_classes.npy"):
    """Builds (200000, 144, 160) uint8 cache of nearest shade classes for all native frames."""
    files = sorted(glob.glob("work/hf/dataset/transitions_*.npz"))
    if not files:
        raise FileNotFoundError("No transition files found in work/hf/dataset/")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for f in files:
        with np.load(f) as d:
            total_frames += len(d["actions"])

    print(f"Building native classes cache for {total_frames} frames from {len(files)} files...")
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(total_frames, 144, 160))

    offset = 0
    t0 = time.time()
    for f_idx, f in enumerate(files):
        with np.load(f) as d:
            obs = d["obs"]
        n = len(obs)
        # Vectorized RGB to greyscale conversion
        grey = cv2.cvtColor(obs.reshape(-1, 160, 3), cv2.COLOR_RGB2GRAY).reshape(n, 144, 160)
        cache[offset : offset + n] = to_classes(grey)
        offset += n
        if (f_idx + 1) % 5 == 0 or f_idx == len(files) - 1:
            print(f"Processed {f_idx + 1}/{len(files)} files ({offset}/{total_frames} frames) in {time.time() - t0:.1f}s")

    del cache
    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"Native classes cache saved to {out_path} ({size_gb:.2f} GB)")
    return out_path


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Upsampler(nn.Module):
    """Small conv U-Net for 2x upsampling with 4 shade class logits."""

    def __init__(self, chans=(32, 64, 128)):
        super().__init__()
        self.conv_in = nn.Conv2d(1, chans[0], 3, padding=1)
        self.enc0 = ConvBlock(chans[0], chans[0])
        self.down0 = nn.Conv2d(chans[0], chans[1], 3, stride=2, padding=1)
        self.enc1 = ConvBlock(chans[1], chans[1])
        self.down1 = nn.Conv2d(chans[1], chans[2], 3, stride=2, padding=1)
        self.mid = ConvBlock(chans[2], chans[2])
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(chans[2], chans[1], 3, padding=1))
        self.dec1 = ConvBlock(chans[1] * 2, chans[1])
        self.up0 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(chans[1], chans[0], 3, padding=1))
        self.dec0 = ConvBlock(chans[0] * 2, chans[0])
        self.conv_out = nn.Conv2d(chans[0], 4, 3, padding=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h0 = self.enc0(self.conv_in(x))
        h1 = self.enc1(self.down0(h0))
        h2 = self.mid(self.down1(h1))
        d1 = self.dec1(torch.cat([self.up1(h2), h1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), h0], dim=1))
        return self.conv_out(d0)


def load_upsampler(path="work/upsampler/upsampler.pt", device="cpu") -> Upsampler:
    """Loads Upsampler model checkpoint into eval mode on target device."""
    model = Upsampler().to(device)
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Upsampler checkpoint not found at {path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unexpected checkpoint format in {path}")
    model.eval()
    return model


@torch.no_grad()
def upsample(model: Upsampler, frames, device="cpu", batch=256) -> np.ndarray:
    """frames (N,72,80) uint8 or float in [0,1] -> (N,144,160) uint8 with values in {0,85,153,255}."""
    model.eval()
    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu().numpy()
        is_uint8 = (frames.dtype == torch.uint8)
    else:
        frames_np = np.asarray(frames)
        is_uint8 = (frames_np.dtype == np.uint8)

    single_frame = False
    if frames_np.ndim == 2:
        frames_np = frames_np[None]
        single_frame = True

    n = len(frames_np)
    out = np.empty((n, 144, 160), dtype=np.uint8)
    dev = torch.device(device)
    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    for i in range(0, n, batch):
        batch_slice = frames_np[i : i + batch]
        if is_uint8:
            batch_t = torch.from_numpy(batch_slice).to(device=dev, dtype=torch.float32) / 255.0
        else:
            batch_t = torch.from_numpy(batch_slice).to(device=dev, dtype=torch.float32)

        if batch_t.dim() == 3:
            batch_t = batch_t.unsqueeze(1)

        with autocast_ctx:
            logits = model(batch_t)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        out[i : i + len(preds)] = SHADES[preds]

    if single_frame:
        return out[0]
    return out


def save_screenshot(frame_raw_u8, upsampler_or_None, out_dir="work/screenshots", step=0, device="cpu"):
    """Saves screenshots of the current frame.

    If upsampler_or_None is provided:
        - shot_<YYYYmmdd_HHMMSS>_step<N>_native.png (upsampled 144x160, exact 4 shades)
        - shot_<YYYYmmdd_HHMMSS>_step<N>_4x.png (144x160 nearest 4x to 576x640, no status bar)
        - shot_<YYYYmmdd_HHMMSS>_step<N>_raw72x80.png (raw 72x80 frame)
    If upsampler_or_None is None:
        - shot_<YYYYmmdd_HHMMSS>_step<N>_raw72x80.png (raw 72x80 frame)
        - shot_<YYYYmmdd_HHMMSS>_step<N>_8x.png (raw 72x80 nearest 8x to 576x640)
        Prints a warning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(frame_raw_u8, torch.Tensor):
        t = frame_raw_u8.detach().cpu()
        if t.dtype != torch.uint8:
            if t.min() < 0:
                t = ((t + 1.0) / 2.0).clamp(0.0, 1.0)
            if t.max() <= 1.0:
                t = (t * 255.0).round()
            t = t.byte()
        raw_np = t.numpy()
    else:
        raw_np = np.asarray(frame_raw_u8)

    while raw_np.ndim > 2:
        raw_np = raw_np[0]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"shot_{timestamp}_step{step}"
    saved_paths = []

    if upsampler_or_None is not None:
        native_u8 = upsample(upsampler_or_None, raw_np, device=device)
        while native_u8.ndim > 2:
            native_u8 = native_u8[0]

        p_native = out_dir / f"{stem}_native.png"
        cv2.imwrite(str(p_native), native_u8)
        saved_paths.append(p_native)

        frame_4x = cv2.resize(native_u8, (640, 576), interpolation=cv2.INTER_NEAREST)
        p_4x = out_dir / f"{stem}_4x.png"
        cv2.imwrite(str(p_4x), frame_4x)
        saved_paths.append(p_4x)

        p_raw = out_dir / f"{stem}_raw72x80.png"
        cv2.imwrite(str(p_raw), raw_np)
        saved_paths.append(p_raw)
    else:
        print("Warning: no upsampler loaded; saving raw and nearest 8x screenshot.")
        p_raw = out_dir / f"{stem}_raw72x80.png"
        cv2.imwrite(str(p_raw), raw_np)
        saved_paths.append(p_raw)

        frame_8x = cv2.resize(raw_np, (640, 576), interpolation=cv2.INTER_NEAREST)
        p_8x = out_dir / f"{stem}_8x.png"
        cv2.imwrite(str(p_8x), frame_8x)
        saved_paths.append(p_8x)

    for p in saved_paths:
        print(f"Saved screenshot: {p}")

    return saved_paths


def train(minutes: float, device: str = "cpu", batch: int = 32):
    """Train upsampler on train split for minutes wall-time budget."""
    torch.set_num_threads(4)
    dev = torch.device(device)
    work_dir = Path("work/upsampler")
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train.log"
    ckpt_path = work_dir / "upsampler.pt"

    with open(log_path, "a") as logf:
        def log(msg):
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        log(f"Starting training for {minutes} min on {device}, batch {batch}")
        cache_path = Path("work/native_classes.npy")
        if not cache_path.exists():
            log("Native classes cache not found. Building it first...")
            build_native_cache(cache_path)

        log("Loading native classes into RAM...")
        t0 = time.time()
        native_classes = np.load(cache_path)
        log(f"Loaded {len(native_classes)} native class frames in {time.time() - t0:.1f}s ({native_classes.nbytes / (1024**3):.2f} GB)")

        data, splits = load()
        frames_72 = data["frames"]
        ep_ids = data["episode_id"]
        train_eps = set(splits["train"])
        train_indices = np.where(np.isin(ep_ids, list(train_eps)))[0]
        log(f"Found {len(train_indices)} train frames across {len(train_eps)} episodes")

        model = Upsampler().to(dev)
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=dev, weights_only=True)
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
                elif isinstance(state, dict):
                    model.load_state_dict(state)
                log(f"Resumed from {ckpt_path}")
            except Exception as e:
                log(f"Could not resume from {ckpt_path}: {e}")

        model.train()
        # lr 1e-3 without clipping spiked twice (loss 0.02 -> 1.5) in the first run; clip + cosine decay
        lr_max = 3e-4
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max)

        def save():
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")

        rng = np.random.default_rng()
        autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

        t_start = time.time()
        t_last = time.time()
        t_ckpt = time.time()
        step = 0
        losses = []

        while (time.time() - t_start) < (minutes * 60):
            batch_idx = rng.choice(train_indices, size=batch, replace=True)
            x_raw = frames_72[batch_idx]
            y_raw = native_classes[batch_idx]

            x = torch.from_numpy(x_raw).to(device=dev, dtype=torch.float32).unsqueeze(1) / 255.0
            y = torch.from_numpy(y_raw).to(device=dev, dtype=torch.long)

            sigma = torch.rand(batch, 1, 1, 1, device=dev) * 0.05
            x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            frac = min((time.time() - t_start) / (minutes * 60), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_max * 0.5 * (1 + np.cos(np.pi * frac))
            opt.step()

            step += 1
            losses.append(loss.item())

            if step % 200 == 0:
                now = time.time()
                dt = now - t_last
                sps = 200 / dt if dt > 0 else 0
                mean_loss = sum(losses) / len(losses)
                elapsed_min = (now - t_start) / 60
                log(f"step {step} loss {mean_loss:.4f} {sps:.1f} steps/s elapsed {elapsed_min:.1f} min")
                losses = []
                t_last = time.time()

            if (time.time() - t_ckpt) >= 600:
                save()
                t_ckpt = time.time()

        save()
        if losses:
            mean_loss = sum(losses) / len(losses)
            log(f"final step {step} loss {mean_loss:.4f} elapsed {(time.time() - t_start) / 60:.1f} min")
        log(f"DONE training {step} steps. Checkpoint saved to {ckpt_path}")


def evaluate(device: str = "cpu", n_samples: int = 2000):
    """Evaluate pixel accuracy and 100% exact frames on train, test_id, test_ood."""
    dev = torch.device(device)
    ckpt_path = Path("work/upsampler/upsampler.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = load_upsampler(str(ckpt_path), device=device)
    cache_path = Path("work/native_classes.npy")
    if not cache_path.exists():
        raise FileNotFoundError(f"Native classes cache not found at {cache_path}")

    native_classes = np.load(cache_path, mmap_mode="r")
    data, splits = load()
    frames_72 = data["frames"]
    ep_ids = data["episode_id"]

    rng = np.random.default_rng(0)
    metrics = {}
    table_rows = []

    for split_name in ["train", "test_id", "test_ood"]:
        split_eps = set(splits[split_name])
        split_indices = np.where(np.isin(ep_ids, list(split_eps)))[0]
        n_eval = min(n_samples, len(split_indices))
        chosen = rng.choice(split_indices, size=n_eval, replace=False)

        eval_frames = frames_72[chosen]
        eval_targets = native_classes[chosen]

        pred_shades = upsample(model, eval_frames, device=device, batch=256)
        pred_classes = to_classes(pred_shades)

        matches = (pred_classes == eval_targets)
        pixel_acc = float(matches.mean())
        exact_acc = float(matches.all(axis=(1, 2)).mean())

        metrics[split_name] = {
            "pixel_accuracy": pixel_acc,
            "exact_frame_fraction": exact_acc,
            "samples": n_eval,
        }
        table_rows.append(f"| {split_name:<10} | {pixel_acc * 100:6.2f}% | {exact_acc * 100:6.2f}% | {n_eval:>7} |")

    out_json = Path("work/upsampler/metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))

    table = [
        "| Split      | Pixel Acc | 100% Exact Frames | Samples |",
        "|------------|-----------|-------------------|---------|",
    ] + table_rows
    table_str = "\n".join(table)
    print(table_str)
    print(f"\nSaved metrics to {out_json}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Pokemon Red Frame Upsampler")
    parser.add_argument("--build-cache", action="store_true", help="Build native classes cache")
    parser.add_argument("--train", action="store_true", help="Train upsampler")
    parser.add_argument("--eval", action="store_true", help="Evaluate upsampler")
    parser.add_argument("--minutes", type=float, default=25.0, help="Training budget in minutes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=2000, help="Number of eval samples per split")
    args = parser.parse_args()

    if args.build_cache:
        build_native_cache()
    if args.train:
        train(minutes=args.minutes, device=args.device, batch=args.batch)
    if args.eval:
        evaluate(device=args.device, n_samples=args.samples)
    if not (args.build_cache or args.train or args.eval):
        parser.print_help()


if __name__ == "__main__":
    main()
