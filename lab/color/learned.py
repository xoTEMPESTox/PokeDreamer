"""Learned colorizer for 72x80 greyscale world-model frames (Stage 2).

Predicts per-pixel colour class indices (0..22) directly from 72x80 frames,
trained on exact Stage 1 targets from real gameplay. Display only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.color.rules import PLAYER_BOX, _rule, build_tile_map, colorize_frame
from lab.upsampler import SHADES, load_upsampler, to_classes, upsample

# Class names corresponding to palette.json (23 classes)
CLASS_NAMES = [
    "grass", "tall_grass", "tree", "path", "water", "ledge", "fence",
    "flowers", "building_wall", "roof_red", "roof_blue", "door", "sign",
    "floor_indoor", "counter", "carpet", "text_box", "sprite_player",
    "sprite_npc", "battle_bg", "battle_hp_bar", "other", "rock",
]

DEFAULT_PALETTE_PATH = Path("lab/color/palette.json")
DEFAULT_LABELS_PATH = Path("lab/color/labels.json")
DEFAULT_TILES_PATH = Path("work/color/native/tiles.npz")


def load_palette(
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    tiles_path: str | Path = DEFAULT_TILES_PATH,
) -> Dict[str, Any]:
    """Loads palette dict with 'classes' and 'tiles' map."""
    with open(palette_path, "r") as f:
        pal_data = json.load(f)
    with open(labels_path, "r") as f:
        labels_data = json.load(f)

    # keys from the labels themselves (both bg and window layers), so the map is complete without work/ files;
    # the collected inventory, when present, only adds unlabelled keys (which map to rules or "other")
    keys = [f"{k.split(':')[0]}:{layer}:{k.split(':')[1]}" for k in labels_data for layer in (0, 1)]
    t_path = Path(tiles_path)
    if t_path.exists():
        keys += [str(k) for k in np.load(t_path)["keys"]]

    tiles = build_tile_map(labels_data, keys)
    return {
        "classes": pal_data["classes"],
        "tiles": tiles,
        "labels": labels_data,
    }


def class_map(px: Dict[str, Any], palette: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Returns (144, 160) integer class indices using rules.colorize_frame logic.

    Uses exactly the same rules as rules.colorize_frame:
    - Font region 128..255 -> text_box
    - Layer 2 outside player box -> sprite_npc
    - Screen-centre 16x16 box on layer 2 -> sprite_player (unless in battle)
    - In battle -> battle_bg for background/window (font rule still applies)
    - Overworld tile map lookup with fallback to labels / other.
    """
    if palette is None:
        palette = load_palette()
    classes = palette["classes"]
    names = list(classes)
    tiles = palette.get("tiles", {})
    labels = palette.get("labels", {})

    layer, tile, ts = px["layer"], px["tile"], px["tileset"]
    battle = bool(px.get("in_battle"))
    pairs, inv = np.unique(np.stack([layer.ravel(), tile.ravel()], 1), axis=0, return_inverse=True)
    lut = []
    for l, t in pairs:
        if battle:
            name = _rule(l, t) or "battle_bg"
        else:
            k = f"{ts}:{l}:{t}"
            name = tiles.get(k) or _rule(l, t) or labels.get(f"{ts}:{t}") or "other"
        lut.append(names.index(name if name in classes else "other"))
    cls = np.asarray(lut, dtype=np.int64)[inv.ravel()].reshape(layer.shape)
    if not battle:
        (y0, y1), (x0, x1) = PLAYER_BOX
        box = np.zeros(layer.shape, bool)
        box[y0:y1, x0:x1] = True
        cls[box & (layer == 2)] = names.index("sprite_player")
    return cls


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class Colorizer(nn.Module):
    """Small U-Net predicting 23 colour classes at 144x160 from 5 input channels.

    Parameter budget: <= 2.0M parameters.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 23,
        chans: Tuple[int, ...] = (32, 64, 128, 144),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, chans[0], 3, padding=1)
        self.enc0 = ResBlock(chans[0], chans[0])
        self.down0 = nn.Conv2d(chans[0], chans[1], 3, stride=2, padding=1)
        self.enc1 = ResBlock(chans[1], chans[1])
        self.down1 = nn.Conv2d(chans[1], chans[2], 3, stride=2, padding=1)
        self.enc2 = ResBlock(chans[2], chans[2])
        self.down2 = nn.Conv2d(chans[2], chans[3], 3, stride=2, padding=1)
        self.mid = ResBlock(chans[3], chans[3])
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(chans[3], chans[2], 3, padding=1),
        )
        self.dec2 = ResBlock(chans[2] * 2, chans[2])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(chans[2], chans[1], 3, padding=1),
        )
        self.dec1 = ResBlock(chans[1] * 2, chans[1])
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(chans[1], chans[0], 3, padding=1),
        )
        self.dec0 = ResBlock(chans[0] * 2, chans[0])
        self.conv_out = nn.Conv2d(chans[0], num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.enc0(self.conv_in(x))
        h1 = self.enc1(self.down0(h0))
        h2 = self.enc2(self.down1(h1))
        h3 = self.mid(self.down2(h2))
        d2 = self.dec2(torch.cat([self.up2(h3), h2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), h0], dim=1))
        return self.conv_out(d0)


def prepare_input(
    frames72: np.ndarray | torch.Tensor,
    upsampler: Optional[Any] = None,
    device: str = "cpu",
    upsampled_shades: Optional[np.ndarray | torch.Tensor] = None,
) -> torch.Tensor:
    """Constructs (N, 5, 144, 160) float tensor from 72x80 frames.

    Channels 0..3: One-hot 4-shade classes from upsampler {0, 85, 153, 255}.
    Channel 4: Raw 72x80 frame nearest-upsampled to 144x160 in [0, 1].
    """
    if isinstance(frames72, torch.Tensor):
        f72_np = frames72.detach().cpu().numpy()
    else:
        f72_np = np.asarray(frames72)

    if f72_np.ndim == 2:
        f72_np = f72_np[None]

    dev = torch.device(device)

    if upsampled_shades is None:
        if upsampler is None:
            raise ValueError("Either upsampler or upsampled_shades must be provided.")
        shades = upsample(upsampler, f72_np, device=device)
    else:
        if isinstance(upsampled_shades, torch.Tensor):
            shades = upsampled_shades.detach().cpu().numpy()
        else:
            shades = np.asarray(upsampled_shades)
        if shades.ndim == 2:
            shades = shades[None]

    # Map shades {0, 85, 153, 255} to class indices {0, 1, 2, 3}
    shade_classes = to_classes(shades)  # (N, 144, 160) uint8 in 0..3
    one_hot = F.one_hot(torch.from_numpy(shade_classes).long(), num_classes=4).permute(0, 3, 1, 2).float()

    raw_t = torch.from_numpy(f72_np).float().unsqueeze(1) / 255.0
    nearest = F.interpolate(raw_t, size=(144, 160), mode="nearest")

    x = torch.cat([one_hot, nearest], dim=1).to(device=dev)
    return x


def load_colorizer(path: str | Path = "work/color/learned/model.pt", device: str = "cpu") -> Colorizer:
    """Loads Colorizer model state_dict onto target device."""
    model = Colorizer().to(device)
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Colorizer checkpoint not found at {path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise ValueError(f"Unexpected checkpoint format in {path}")
    model.eval()
    return model


@torch.no_grad()
def colorize72(
    frames72: np.ndarray | torch.Tensor,
    model: nn.Module,
    upsampler: Any,
    palette: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    batch: int = 64,
) -> np.ndarray:
    """Colorize 72x80 greyscale frames using learned colorizer and upsampler.

    Returns (N, 144, 160, 3) uint8 RGB array (or (144, 160, 3) if single 2D frame).
    """
    if palette is None:
        palette = load_palette()
    classes = palette["classes"]
    names = list(classes)
    ramps = np.asarray([classes[n] for n in names], dtype=np.uint8)  # (23, 4, 3)

    if isinstance(frames72, torch.Tensor):
        f72_np = frames72.detach().cpu().numpy()
    else:
        f72_np = np.asarray(frames72)

    single_frame = (f72_np.ndim == 2)
    if single_frame:
        f72_np = f72_np[None]

    N = len(f72_np)
    out = np.empty((N, 144, 160, 3), dtype=np.uint8)
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    # 1. Upsample all frames to 4 DMG shades
    shades = upsample(upsampler, f72_np, device=device, batch=batch)  # (N, 144, 160)

    # 2. Shade level index: 0 lightest (255) .. 3 darkest (0)
    ramp_idx = (3 - to_classes(shades)).astype(np.int64)

    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    for i in range(0, N, batch):
        batch_f72 = f72_np[i : i + batch]
        batch_shades = shades[i : i + batch]
        x = prepare_input(batch_f72, device=device, upsampled_shades=batch_shades)
        if dev.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with autocast_ctx:
            logits = model(x)
        pred_cls = logits.argmax(dim=1).cpu().numpy()  # (B, 144, 160)

        b_ramp = ramp_idx[i : i + len(batch_f72)]
        out[i : i + len(batch_f72)] = ramps[pred_cls, b_ramp]

    if single_frame:
        return out[0]
    return out


# Multiprocess worker for dataset collection
def _collect_worker(args: Tuple[str, int, int]) -> Dict[str, Any]:
    """Runs one emulator on one route state for steps_per_state steps."""
    state_path, steps, seed = args
    from lab.color.native import pixel_tiles
    from lab.env import PokemonRed

    rng = random.Random(seed)
    palette = load_palette()

    frames72 = np.empty((steps, 72, 80), dtype=np.uint8)
    class_maps = np.empty((steps, 144, 160), dtype=np.uint8)
    cidx_arr = np.empty((steps, 144, 160), dtype=np.uint8)
    in_battles = np.empty(steps, dtype=bool)

    env = PokemonRed(state=state_path)
    try:
        for s in range(steps):
            # Directions mostly, A every 10th step to clear text
            if s % 10 == 0:
                act = 4  # button 'a'
            else:
                act = rng.choice([0, 1, 2, 3])  # down, left, right, up

            f72, _ = env.step(act)
            px = pixel_tiles(env.pyboy)
            cm = class_map(px, palette)

            frames72[s] = f72
            class_maps[s] = cm.astype(np.uint8)
            cidx_arr[s] = px["cidx"].astype(np.uint8)
            in_battles[s] = bool(px.get("in_battle", False))
    finally:
        env.close()

    return {
        "state": Path(state_path).name,
        "frames72": frames72,
        "class_maps": class_maps,
        "cidx": cidx_arr,
        "in_battles": in_battles,
    }


def collect_data(
    states_dir: str | Path = "work/route_states",
    out_dir: str | Path = "work/color/learned",
    steps_per_state: int = 4000,
    num_workers: int = 8,
) -> None:
    """Collects dataset from all route states using process pool."""
    import glob

    state_files = sorted(glob.glob(f"{states_dir}/*.state"))
    if not state_files:
        raise FileNotFoundError(f"No route states found in {states_dir}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_files = [f for f in state_files if "seed102" not in Path(f).name]
    test_files = [f for f in state_files if "seed102" in Path(f).name]

    print(f"Total states: {len(state_files)} (Train: {len(train_files)}, Test: {len(test_files)})")
    print(f"Running {steps_per_state} steps per state across {num_workers} workers...")

    for split_name, files in [("train", train_files), ("test", test_files)]:
        t0 = time.time()
        print(f"\n--- Collecting {split_name} split ({len(files)} states) ---")
        tasks = [(f, steps_per_state, 1000 + idx) for idx, f in enumerate(files)]

        all_frames72 = []
        all_class_maps = []
        all_cidx = []
        all_battles = []

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for res in pool.map(_collect_worker, tasks):
                st_name = res["state"]
                b_cnt = int(res["in_battles"].sum())
                print(f"  Processed {st_name}: {steps_per_state} steps ({b_cnt} battle frames)")
                all_frames72.append(res["frames72"])
                all_class_maps.append(res["class_maps"])
                all_cidx.append(res["cidx"])
                all_battles.append(res["in_battles"])

        cat_f72 = np.concatenate(all_frames72, axis=0)
        cat_cm = np.concatenate(all_class_maps, axis=0)
        cat_cidx = np.concatenate(all_cidx, axis=0)
        cat_battles = np.concatenate(all_battles, axis=0)

        dt = time.time() - t0
        save_file = out_path / f"data_{split_name}.npz"
        print(f"Saving {split_name} split to {save_file}...")
        np.savez_compressed(
            save_file,
            frames72=cat_f72,
            class_map=cat_cm,
            cidx=cat_cidx,
            in_battle=cat_battles,
        )
        file_size_mb = save_file.stat().st_size / (1024 * 1024)
        print(f"Saved {save_file} ({file_size_mb:.1f} MB) in {dt:.1f} s")

        # Report stats
        total_frames = len(cat_f72)
        battle_share = float(cat_battles.mean()) * 100
        print(f"[{split_name.upper()} STATS]")
        print(f"  Frame count: {total_frames:,}")
        print(f"  Battle frames share: {battle_share:.2f}% ({cat_battles.sum():,} / {total_frames:,})")
        print("  Class histogram:")
        hist = np.bincount(cat_cm.ravel(), minlength=len(CLASS_NAMES))
        total_pixels = cat_cm.size
        for c_idx, count in enumerate(hist):
            c_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"class_{c_idx}"
            print(f"    {c_idx:2d} ({c_name:<15}): {count:>11,} ({count / total_pixels * 100:5.2f}%)")


def wait_for_gpu_idle(poll_interval: int = 60) -> None:
    """Waits until no compute process is using the GPU."""
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=process_name,pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            # On Windows WDDM, desktop GUI apps may be listed.
            # Check specifically for python compute jobs.
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            python_jobs = [l for l in lines if "python" in l.lower()]
            if not python_jobs:
                print("GPU is idle. Proceeding with training.")
                break
            print(f"{time.strftime('%H:%M:%S')} GPU in use by {len(python_jobs)} python compute job(s): {python_jobs}. Waiting {poll_interval}s...")
        except Exception as e:
            print(f"Warning: nvidia-smi check failed: {e}. Assuming GPU is available.")
            break
        time.sleep(poll_interval)


def train_colorizer(
    minutes: float = 25.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    lr: float = 3e-4,
    data_dir: str | Path = "work/color/learned",
    upsampler_path: str | Path = "work/upsampler/upsampler.pt",
) -> None:
    """Trains the Colorizer U-Net on data_train.npz for ~minutes wall-time budget."""
    if device == "cuda":
        wait_for_gpu_idle()

    data_dir = Path(data_dir)
    train_npz = data_dir / "data_train.npz"
    if not train_npz.exists():
        raise FileNotFoundError(f"Train data not found at {train_npz}. Run data collection first.")

    log_path = data_dir / "train.log"
    ckpt_path = data_dir / "model.pt"

    print(f"Loading upsampler from {upsampler_path}...")
    upsampler = load_upsampler(str(upsampler_path), device=device)

    print(f"Loading training data from {train_npz}...")
    t0 = time.time()
    data = np.load(train_npz)
    train_f72 = data["frames72"]
    train_cm = data["class_map"]
    n_train = len(train_f72)
    print(f"Loaded {n_train:,} training frames in {time.time() - t0:.1f} s")

    shades_cache_path = data_dir / "train_shades.npy"
    if shades_cache_path.exists():
        print(f"Loading precomputed upsampled shades from {shades_cache_path}...")
        train_shades = np.load(shades_cache_path)
    else:
        print(f"Precomputing upsampled shades for {n_train:,} training frames on {device}...")
        t_up = time.time()
        train_shades = upsample(upsampler, train_f72, device=device, batch=256)
        np.save(shades_cache_path, train_shades)
        print(f"Saved upsampled shades cache to {shades_cache_path} in {time.time() - t_up:.1f} s")

    dev = torch.device(device)
    model = Colorizer().to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Colorizer model parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    if n_params > 2_000_000:
        raise ValueError(f"Model exceeds 2.0M parameter budget: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting training for {minutes:.1f} min on {device}, batch {batch_size}, params {n_params:,}")

    rng = np.random.default_rng(42)
    t_start = time.time()
    t_last = time.time()
    step = 0
    losses: List[float] = []

    def save_ckpt() -> None:
        tmp_path = ckpt_path.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, ckpt_path)
        log(f"Saved checkpoint to {ckpt_path}")

    while (time.time() - t_start) < (minutes * 60):
        # Sample random batch
        idx = rng.choice(n_train, size=batch_size, replace=True)
        batch_f72 = train_f72[idx]
        batch_shades = train_shades[idx]
        batch_cm = train_cm[idx]

        # Prepare 5-channel input on GPU
        shade_cls = torch.from_numpy(to_classes(batch_shades)).long().to(dev)
        one_hot = F.one_hot(shade_cls, num_classes=4).permute(0, 3, 1, 2).float()
        raw_t = torch.from_numpy(batch_f72).float().unsqueeze(1).to(dev) / 255.0
        # Slight noise augmentation on raw frame
        sigma = torch.rand(batch_size, 1, 1, 1, device=dev) * 0.05
        raw_t = (raw_t + torch.randn_like(raw_t) * sigma).clamp(0.0, 1.0)
        nearest = F.interpolate(raw_t, size=(144, 160), mode="nearest")

        x = torch.cat([one_hot, nearest], dim=1)
        if dev.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        y = torch.from_numpy(batch_cm).long().to(dev)

        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Cosine learning rate decay
        frac = min((time.time() - t_start) / (minutes * 60), 1.0)
        cur_lr = lr * 0.5 * (1.0 + np.cos(np.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.step()

        step += 1
        losses.append(loss.item())

        if step % 100 == 0:
            now = time.time()
            dt = now - t_last
            sps = 100 / dt if dt > 0 else 0
            mean_loss = sum(losses) / len(losses)
            elapsed_min = (now - t_start) / 60
            log(f"step {step} loss {mean_loss:.4f} {sps:.1f} steps/s lr {cur_lr:.2e} elapsed {elapsed_min:.1f} min")
            losses = []

            # GPU etiquette: check work/PAUSE
            pause_file = Path("work/PAUSE")
            if pause_file.exists():
                save_ckpt()
                log("PAUSED: work/PAUSE exists; idling...")
                t_pause = time.time()
                while pause_file.exists():
                    time.sleep(10)
                t_start += (time.time() - t_pause)
                log("Resumed after pause.")

            t_last = time.time()

        if step % 1000 == 0:
            save_ckpt()

    save_ckpt()
    total_elapsed = (time.time() - t_start) / 60
    log(f"DONE training {step} steps in {total_elapsed:.1f} min. Final checkpoint saved to {ckpt_path}")


def evaluate_colorizer(
    data_dir: str | Path = "work/color/learned",
    model_path: str | Path = "work/color/learned/model.pt",
    upsampler_path: str | Path = "work/upsampler/upsampler.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Evaluates trained colorizer on held-out split (data_test.npz)."""
    data_dir = Path(data_dir)
    test_npz = data_dir / "data_test.npz"
    if not test_npz.exists():
        raise FileNotFoundError(f"Test data not found at {test_npz}")

    palette = load_palette()
    classes = palette["classes"]
    names = list(classes)
    ramps = np.asarray([classes[n] for n in names], dtype=np.uint8)

    print(f"Loading Colorizer model from {model_path} on {device}...")
    model = load_colorizer(model_path, device=device)
    print(f"Loading Upsampler from {upsampler_path} on {device}...")
    upsampler = load_upsampler(str(upsampler_path), device=device)

    print(f"Loading test data from {test_npz}...")
    test_data = np.load(test_npz)
    frames72 = test_data["frames72"]
    target_cm = test_data["class_map"]
    cidx = test_data["cidx"]
    in_battle = test_data["in_battle"]
    N = len(frames72)
    print(f"Loaded {N:,} test frames.")

    shades_cache_path = data_dir / "test_shades.npy"
    if shades_cache_path.exists():
        print(f"Loading precomputed upsampled shades from {shades_cache_path}...")
        test_shades = np.load(shades_cache_path)
    else:
        print(f"Precomputing upsampled shades for {N:,} test frames on {device}...")
        t_up = time.time()
        test_shades = upsample(upsampler, frames72, device=device, batch=256)
        np.save(shades_cache_path, test_shades)
        print(f"Saved upsampled shades cache to {shades_cache_path} in {time.time() - t_up:.1f} s")

    dev = torch.device(device)
    model.eval()

    total_pixels = 0
    matching_pixels = 0
    non_battle_pixels = 0
    non_battle_matches = 0
    battle_pixels = 0
    battle_matches = 0
    rgb_differ_pixels = 0

    per_class_total = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    per_class_correct = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    print("Evaluating test split...")
    t0 = time.time()
    for i in range(0, N, batch_size):
        b_f72 = frames72[i : i + batch_size]
        b_shades = test_shades[i : i + batch_size]
        b_tgt_cm = target_cm[i : i + batch_size]
        b_cidx = cidx[i : i + batch_size]
        b_battle = in_battle[i : i + batch_size]

        shade_cls = torch.from_numpy(to_classes(b_shades)).long().to(dev)
        one_hot = F.one_hot(shade_cls, num_classes=4).permute(0, 3, 1, 2).float()
        raw_t = torch.from_numpy(b_f72).float().unsqueeze(1).to(dev) / 255.0
        nearest = F.interpolate(raw_t, size=(144, 160), mode="nearest")

        x = torch.cat([one_hot, nearest], dim=1)
        if dev.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with autocast_ctx:
                logits = model(x)
            pred_cls = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        # Pixel accuracy overall
        correct_mask = (pred_cls == b_tgt_cm)
        matching_pixels += int(correct_mask.sum())
        total_pixels += b_tgt_cm.size

        # Battle vs non-battle
        # b_battle is (B,) boolean array
        b_cnt = int(b_battle.sum())
        nb_cnt = len(b_battle) - b_cnt
        battle_pixels += b_cnt * 144 * 160
        battle_matches += int(correct_mask[b_battle].sum()) if b_cnt > 0 else 0
        non_battle_pixels += nb_cnt * 144 * 160
        non_battle_matches += int(correct_mask[~b_battle].sum()) if nb_cnt > 0 else 0

        # Per class accuracy
        for c in range(len(CLASS_NAMES)):
            c_mask = (b_tgt_cm == c)
            c_tot = int(c_mask.sum())
            if c_tot > 0:
                per_class_total[c] += c_tot
                per_class_correct[c] += int(correct_mask[c_mask].sum())

        # Final RGB comparison with Stage 1
        # Stage 1: ramps[target_cm, px["cidx"]]
        # Learned: ramps[pred_cls, ramp_idx]
        b_ramp_idx = (3 - to_classes(b_shades)).astype(np.int64)
        pred_rgb = ramps[pred_cls, b_ramp_idx]
        stage1_rgb = ramps[b_tgt_cm, b_cidx]
        rgb_diff = (pred_rgb != stage1_rgb).any(axis=-1)
        rgb_differ_pixels += int(rgb_diff.sum())

    dt = time.time() - t0
    overall_acc = matching_pixels / max(1, total_pixels)
    non_battle_acc = non_battle_matches / max(1, non_battle_pixels)
    battle_acc = battle_matches / max(1, battle_pixels)
    rgb_differ_share = rgb_differ_pixels / max(1, total_pixels)

    per_class_acc = {}
    for c in range(len(CLASS_NAMES)):
        c_name = CLASS_NAMES[c]
        c_tot = int(per_class_total[c])
        acc = float(per_class_correct[c] / c_tot) if c_tot > 0 else 0.0
        per_class_acc[c_name] = {
            "accuracy": acc,
            "total_pixels": c_tot,
        }

    metrics = {
        "overall_pixel_accuracy": float(overall_acc),
        "non_battle_pixel_accuracy": float(non_battle_acc),
        "battle_pixel_accuracy": float(battle_acc),
        "rgb_difference_share": float(rgb_differ_share),
        "total_test_frames": N,
        "total_test_pixels": total_pixels,
        "evaluation_time_seconds": float(dt),
        "per_class": per_class_acc,
    }

    metrics_file = data_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"\nEvaluation complete in {dt:.1f} s. Saved metrics to {metrics_file}")
    print(f"  Overall Pixel Accuracy: {overall_acc * 100:.2f}%")
    print(f"  Non-battle Accuracy:   {non_battle_acc * 100:.2f}%")
    print(f"  Battle Accuracy:       {battle_acc * 100:.2f}%")
    print(f"  RGB Differ Share:      {rgb_differ_share * 100:.2f}%")
    print("\nPer-class accuracy:")
    for c_name, c_info in per_class_acc.items():
        if c_info["total_pixels"] > 0:
            print(f"  {c_name:<16}: {c_info['accuracy'] * 100:6.2f}% ({c_info['total_pixels']:,} px)")
    return metrics


def make_demo(
    traj_path: str | Path = "work/gym_traj_best_s0/run_seed_100.npz",
    model_path: str | Path = "work/color/learned/model.pt",
    upsampler_path: str | Path = "work/upsampler/upsampler.pt",
    world_model_run: str = "wm31",
    out_path: str | Path = "work/color/learned/demo.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    """Generates the 6-moment world model open-loop demo image.

    Rows = 6 moments (outdoor, forest, Pewter, Pokémon Center, gym, battle).
    Columns = Real colour (colorize72) | Dream step 1 | Dream step 4 | Dream step 8.
    All colorized with colorize72, 3x nearest upscaled.
    """
    if device == "cuda":
        wait_for_gpu_idle()

    from lab.dwm.controllability import load_world_model
    from lab.dwm.model import CTX, to_float

    traj = np.load(traj_path)
    frames = traj["frames"]  # (N, 72, 80)
    actions = traj["actions"]  # (N,)
    map_ids = traj["map_ids"] if "map_ids" in traj else traj["map"]
    in_battles = traj["in_battles"] if "in_battles" in traj else traj["in_battle"]

    palette = load_palette()
    model = load_colorizer(model_path, device=device)
    upsampler = load_upsampler(str(upsampler_path), device=device)
    wm = load_world_model(world_model_run, device=device)
    wm.eval()

    # Find the 6 moments:
    # 1. Outdoor (Route 1 or Route 2, maps 12 or 13)
    # 2. Forest (Viridian Forest, map 51)
    # 3. Pewter (Pewter City, map 2)
    # 4. Pokémon Center (maps 41 or 58)
    # 5. Gym (Pewter Gym, map 54)
    # 6. Battle (in_battles == True)
    moments = {}
    N = len(frames)

    def find_idx(condition):
        for t in range(CTX, N - 10):
            if condition(t):
                return t
        return None

    moments["outdoor"] = find_idx(lambda t: map_ids[t] in (12, 13) and not in_battles[t-CTX:t+9].any())
    moments["forest"] = find_idx(lambda t: map_ids[t] == 51 and not in_battles[t-CTX:t+9].any())
    moments["pewter"] = find_idx(lambda t: map_ids[t] == 2 and not in_battles[t-CTX:t+9].any())
    moments["pokecenter"] = find_idx(lambda t: map_ids[t] in (41, 58) and not in_battles[t-CTX:t+9].any())
    moments["gym"] = find_idx(lambda t: map_ids[t] == 54 and not in_battles[t-CTX:t+9].any())
    moments["battle"] = find_idx(lambda t: bool(in_battles[t-CTX:t+9].all()))

    print(f"Selected moments start indices: {moments}")

    moment_keys = ["outdoor", "forest", "pewter", "pokecenter", "gym", "battle"]
    row_images = []

    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if "cuda" in str(device) else contextlib.nullcontext()

    for m_key in moment_keys:
        t0 = moments[m_key]
        if t0 is None:
            raise ValueError(f"Could not find moment: {m_key}")

        # Real context frames f_{t0-4..t0-1}
        ctx_raw = frames[t0 - CTX : t0]
        ctx_t = to_float(torch.from_numpy(ctx_raw).unsqueeze(0).to(device))

        # Real actions for 8 steps: length CTX + 8 - 1 = 11 actions
        act_raw = actions[t0 - CTX : t0 + 8 - 1]
        act_t = torch.from_numpy(act_raw).long().unsqueeze(0).to(device)

        # Rollout 8 steps open-loop
        with torch.no_grad():
            with autocast_ctx:
                dream_batch = wm.rollout(ctx_t, act_t, steps=3)

        dream_np = dream_batch.cpu().float().numpy()[0]  # (8, 72, 80) in [-1, 1]
        dream_u8 = np.clip(np.round(((dream_np + 1.0) / 2.0) * 255.0), 0, 255).astype(np.uint8)

        # Real frame at t0 (start of rollout)
        real_f72 = frames[t0]
        # Dream steps 1, 4, 8 (0-indexed: 0, 3, 7)
        dream_s1 = dream_u8[0]
        dream_s4 = dream_u8[3]
        dream_s8 = dream_u8[7]

        # Colorize all four frames using colorize72
        col_real = colorize72(real_f72, model, upsampler, palette, device=device)
        col_d1 = colorize72(dream_s1, model, upsampler, palette, device=device)
        col_d4 = colorize72(dream_s4, model, upsampler, palette, device=device)
        col_d8 = colorize72(dream_s8, model, upsampler, palette, device=device)

        # 3x nearest resize: 144x160 -> 432x480
        col_real_3x = cv2.resize(col_real, (480, 432), interpolation=cv2.INTER_NEAREST)
        col_d1_3x = cv2.resize(col_d1, (480, 432), interpolation=cv2.INTER_NEAREST)
        col_d4_3x = cv2.resize(col_d4, (480, 432), interpolation=cv2.INTER_NEAREST)
        col_d8_3x = cv2.resize(col_d8, (480, 432), interpolation=cv2.INTER_NEAREST)

        # Concatenate horizontally
        row = np.concatenate([col_real_3x, col_d1_3x, col_d4_3x, col_d8_3x], axis=1)
        row_images.append(row)

    # Concatenate vertically
    full_demo = np.concatenate(row_images, axis=0)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV writes BGR
    cv2.imwrite(str(out_p), cv2.cvtColor(full_demo, cv2.COLOR_RGB2BGR))
    print(f"Saved demo to {out_p} ({full_demo.shape})")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Learned Colorizer for Pokemon Red frames")
    parser.add_argument("--collect", action="store_true", help="Collect dataset via multiprocess random walk")
    parser.add_argument("--train", action="store_true", help="Train colorizer model")
    parser.add_argument("--eval", action="store_true", help="Evaluate on held-out test split")
    parser.add_argument("--demo", action="store_true", help="Generate world model demo image")
    parser.add_argument("--minutes", type=float, default=25.0, help="Training minutes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes for collection")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    if args.collect:
        collect_data(num_workers=args.workers)
    if args.train:
        train_colorizer(minutes=args.minutes, device=args.device, batch_size=args.batch)
    if args.eval:
        evaluate_colorizer(device=args.device)
    if args.demo:
        make_demo(device=args.device)


if __name__ == "__main__":
    main()
