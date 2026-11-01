"""PokeDreamer v1 re-evaluation on held-out episodes.

Faithful re-run of the v1 world model recipe (VAE, GRU Dynamics with Scheduled
Sampling and Teacher Forcing ablation, RAM Probe) trained on an episode-level
train split and evaluated on held-out episodes (test_id and test_ood).

CLI:
    python -m lab.v1_reeval --device cuda
    python -m lab.v1_reeval --quick --device cpu
"""

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from lab.data import load as load_data
from lab.eval.metrics import bootstrap_ci, manhattan, no_change_error


# ─── V1 MODEL IMPORTER (no namespace collision with repo root src) ─────────────

def load_v1_models():
    """Dynamically load v1 model classes from v1/src."""
    v1_root = Path(__file__).resolve().parents[1] / "v1" / "src"

    def _import_from_path(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    vae_mod = _import_from_path("v1_vae", v1_root / "vae.py")
    dyn_mod = _import_from_path("v1_dynamics", v1_root / "dynamics.py")
    prb_mod = _import_from_path("v1_probe", v1_root / "probe.py")

    return vae_mod.VAE, dyn_mod.LatentDynamics, prb_mod.RAMProbe


# ─── DATA PREPARATION & HELPERS ───────────────────────────────────────────────

def to_40x36(obs: np.ndarray) -> np.ndarray:
    """Convert (N, 144, 160, 3) uint8 RGB frames to (N, 36, 40, 3) uint8 RGB.

    Matches v1 cv2.resize(rgb, (40, 36), interpolation=cv2.INTER_AREA).
    """
    out = np.empty((len(obs), 36, 40, 3), dtype=np.uint8)
    for i in range(len(obs)):
        out[i] = cv2.resize(obs[i], (40, 36), interpolation=cv2.INTER_AREA)
    return out


def carve_train_val_episodes(
    train_pool: list[int],
    val_count: int = 10,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Carve validation episodes deterministically from the train pool."""
    rng = np.random.default_rng(seed)
    pool = sorted(train_pool)
    val_eps = sorted(rng.choice(pool, size=val_count, replace=False).tolist())
    train_eps = sorted([ep for ep in pool if ep not in set(val_eps)])
    return train_eps, val_eps


def get_valid_window_indices(
    episodes: list[int],
    ep_len: int = 1000,
    seq_len: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Get valid sequence window start indices strictly inside each episode.

    Returns:
        start_indices: (N,) array of start step indices
        window_episodes: (N,) array recording which episode each window belongs to
    """
    starts = []
    window_eps = []
    n_per_ep = ep_len - seq_len + 1
    for ep in episodes:
        base = ep * ep_len
        for i in range(n_per_ep):
            starts.append(base + i)
            window_eps.append(ep)
    return np.array(starts, dtype=int), np.array(window_eps, dtype=int)


def ensure_cached_frames(
    cache_path: Path,
    data: dict,
    logger,
) -> tuple[np.ndarray, dict]:
    """Build or load 40x36 frames cache and extra RAM fields from HF raw dataset."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(glob.glob("work/hf/dataset/transitions_*.npz"))
    if not raw_files:
        raise FileNotFoundError("No raw transition files found in work/hf/dataset/")

    # Extra RAM fields needed for probe training matching v1
    extra_ram = {"dialog_opens": [], "party_hps": [], "party_max_hps": []}

    if cache_path.exists():
        logger(f"[Cache] Loading 40x36 frames from {cache_path}...")
        frames = np.load(cache_path)
        logger(f"[Cache] Loaded frames array of shape {frames.shape}, dtype {frames.dtype}")
        # Still load extra RAM fields from raw npz
        for f in raw_files:
            d = np.load(f)
            extra_ram["dialog_opens"].append(d["dialog_opens"])
            extra_ram["party_hps"].append(d["party_hps"])
            extra_ram["party_max_hps"].append(d["party_max_hps"])
    else:
        logger(f"[Cache] Rebuilding 40x36 frames from {len(raw_files)} raw transition files...")
        t0 = time.time()
        all_frames = []
        for idx, f in enumerate(raw_files):
            d = np.load(f)
            raw_obs = d["obs"]  # (10000, 144, 160, 3)
            small = to_40x36(raw_obs)
            all_frames.append(small)
            extra_ram["dialog_opens"].append(d["dialog_opens"])
            extra_ram["party_hps"].append(d["party_hps"])
            extra_ram["party_max_hps"].append(d["party_max_hps"])
            if (idx + 1) % 5 == 0 or idx == len(raw_files) - 1:
                logger(f"  Processed {idx + 1}/{len(raw_files)} files...")

        frames = np.concatenate(all_frames, axis=0)
        logger(f"[Cache] Saving 40x36 frames cache to {cache_path} (took {time.time()-t0:.2f}s)...")
        np.save(cache_path, frames)

    # Concatenate extra RAM fields
    for k in extra_ram:
        extra_ram[k] = np.concatenate(extra_ram[k], axis=0)

    # Verification of alignment against data["map_ids"]
    check_indices = [0, 9999, 50000, 100000, 150000, 199999]
    for ci in check_indices:
        file_idx = ci // 10000
        within_file = ci % 10000
        d_check = np.load(raw_files[file_idx])
        assert d_check["map_ids"][within_file] == data["map_ids"][ci], (
            f"Alignment mismatch at index {ci}"
        )
    logger("[Cache] Alignment check with data['map_ids'] verified OK.")

    return frames, extra_ram


# ─── TRAINING ROUTINES ────────────────────────────────────────────────────────

class Logger:
    """Simultaneous stdout and file logging."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log_path, "a", encoding="utf-8")

    def __call__(self, msg: str = ""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def train_vae(
    model,
    frames: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1.0,
    out_dir: Path = Path("work/v1_reeval"),
    logger=print,
    quick: bool = False,
):
    """Train v1 VAE matching v1/scripts/train_vae.py."""
    logger("\n" + "=" * 70)
    logger(f"[VAE Training] Starting {epochs} epochs on device={device}")
    logger("=" * 70)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_vae.pt"

    n_train = len(train_indices)
    n_val = len(val_indices)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        perm = np.random.permutation(n_train)
        if quick:
            perm = perm[: batch_size * 2]
            cur_train_len = len(perm)
        else:
            cur_train_len = n_train

        for i in range(0, cur_train_len, batch_size):
            b_idx = train_indices[perm[i : i + batch_size]]
            x_np = frames[b_idx]  # (B, 36, 40, 3)
            # Permute to (B, 3, 36, 40) normalized [0, 1]
            x = (
                torch.from_numpy(x_np)
                .to(device, non_blocking=True)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            losses = model.loss_function(x_recon, x, mu, logvar, beta=beta)
            loss = losses["loss"]

            loss.backward()
            optimizer.step()

            train_loss += losses["loss"].item()
            train_recon += losses["recon_loss"].item()
            train_kl += losses["kl_loss"].item()

        train_loss /= cur_train_len
        train_recon /= cur_train_len
        train_kl /= cur_train_len

        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        val_eval_indices = val_indices[: batch_size * 2] if quick else val_indices
        cur_val_len = len(val_eval_indices)

        with torch.no_grad():
            for i in range(0, cur_val_len, batch_size):
                b_idx = val_eval_indices[i : i + batch_size]
                x_np = frames[b_idx]
                x = (
                    torch.from_numpy(x_np)
                    .to(device, non_blocking=True)
                    .permute(0, 3, 1, 2)
                    .float()
                    .div_(255.0)
                )

                x_recon, mu, logvar = model(x)
                losses = model.loss_function(x_recon, x, mu, logvar, beta=beta)

                val_loss += losses["loss"].item()
                val_recon += losses["recon_loss"].item()
                val_kl += losses["kl_loss"].item()

        val_loss /= cur_val_len
        val_recon /= cur_val_len
        val_kl /= cur_val_len

        elapsed = time.time() - t0
        logger(
            f"Epoch {epoch:>2}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | "
            f"Val Loss: {val_loss:.2f} (Recon: {val_recon:.2f}, KL: {val_kl:.2f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "latent_dim": model.latent_dim,
                },
                best_ckpt_path,
            )
            logger(f"   [New Best VAE] val_loss: {val_loss:.2f} -> saved {best_ckpt_path.name}")

    # Load best weights before returning
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger(f"[VAE Training Complete] Best Val Loss: {best_val_loss:.2f}")
    return model, best_val_loss


def precompute_latents(
    model,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
    logger=print,
) -> torch.Tensor:
    """Precompute frozen VAE latent means mu for all frames."""
    logger("[Precompute] Encoding all frames to latent space mu with frozen VAE...")
    t0 = time.time()
    n = len(frames)
    all_mu = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            x = (
                torch.from_numpy(chunk)
                .to(device, non_blocking=True)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            mu, _ = model.encode(x)
            all_mu.append(mu)

    all_mu = torch.cat(all_mu, dim=0)
    logger(f"[Precompute] Encoded {n} frames to latents tensor {all_mu.shape} in {time.time()-t0:.2f}s")
    return all_mu


def train_probe(
    probe,
    latents: torch.Tensor,
    ram_data: dict,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    out_dir: Path = Path("work/v1_reeval"),
    logger=print,
    quick: bool = False,
):
    """Train RAM Probe matching v1/scripts/train_probe.py."""
    logger("\n" + "=" * 70)
    logger(f"[RAM Probe Training] Starting {epochs} epochs on device={device}")
    logger("=" * 70)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_probe.pt"

    n_train = len(train_indices)
    n_val = len(val_indices)

    # Prepare target tensors on device
    pos_all = torch.tensor(
        np.stack([ram_data["xs"], ram_data["ys"]], axis=1), dtype=torch.float32, device=device
    )
    map_id_all = torch.tensor(ram_data["map_ids"], dtype=torch.long, device=device)
    in_battle_all = torch.tensor(ram_data["in_battles"], dtype=torch.float32, device=device)
    dialog_open_all = torch.tensor(ram_data["dialog_opens"], dtype=torch.float32, device=device)
    hp_norm_all = torch.tensor(
        ram_data["party_hps"] / (ram_data["party_max_hps"] + 1e-5),
        dtype=torch.float32,
        device=device,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        probe.train()
        train_loss = 0.0

        perm = np.random.permutation(n_train)
        if quick:
            perm = perm[: batch_size * 2]
            cur_train_len = len(perm)
        else:
            cur_train_len = n_train

        for i in range(0, cur_train_len, batch_size):
            b_idx = train_indices[perm[i : i + batch_size]]
            mu = latents[b_idx]

            targets = {
                "pos": pos_all[b_idx],
                "map_id": map_id_all[b_idx],
                "in_battle": in_battle_all[b_idx],
                "dialog_open": dialog_open_all[b_idx],
                "hp": hp_norm_all[b_idx],
            }

            optimizer.zero_grad()
            preds = probe(mu)
            losses = probe.loss_function(preds, targets)
            loss = losses["loss"]

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_idx)

        train_loss /= cur_train_len

        # Validation
        probe.eval()
        val_loss = 0.0
        val_pos_err = 0.0
        val_map_corr = 0.0

        val_eval_indices = val_indices[: batch_size * 2] if quick else val_indices
        cur_val_len = len(val_eval_indices)

        with torch.no_grad():
            for i in range(0, cur_val_len, batch_size):
                b_idx = val_eval_indices[i : i + batch_size]
                mu = latents[b_idx]

                targets = {
                    "pos": pos_all[b_idx],
                    "map_id": map_id_all[b_idx],
                    "in_battle": in_battle_all[b_idx],
                    "dialog_open": dialog_open_all[b_idx],
                    "hp": hp_norm_all[b_idx],
                }

                preds = probe(mu)
                losses = probe.loss_function(preds, targets)
                val_loss += losses["loss"].item() * len(b_idx)

                # Validation metrics
                pos_diff = torch.abs(preds["pos"] - targets["pos"])
                val_pos_err += torch.sum(pos_diff).item()

                pred_map = torch.argmax(preds["map_logits"], dim=-1)
                val_map_corr += torch.sum(pred_map == targets["map_id"]).item()

        val_loss /= cur_val_len
        val_pos_mae = val_pos_err / (cur_val_len * 2)  # Per-axis MAE as computed in v1
        val_manhattan = val_pos_err / cur_val_len       # Manhattan distance
        val_map_acc = (val_map_corr / cur_val_len) * 100.0

        elapsed = time.time() - t0
        logger(
            f"Epoch {epoch:>2}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pos MAE: {val_pos_mae:.2f} tiles (Manhattan: {val_manhattan:.2f}) | "
            f"Map Acc: {val_map_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "latent_dim": probe.latent_dim,
                },
                best_ckpt_path,
            )
            logger(f"   [New Best Probe] val_loss: {val_loss:.4f} -> saved {best_ckpt_path.name}")

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.eval()
    logger(f"[Probe Training Complete] Best Val Loss: {best_val_loss:.4f}")
    return probe, best_val_loss


def train_dynamics(
    model,
    latents: torch.Tensor,
    actions: np.ndarray,
    train_starts: np.ndarray,
    val_starts: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    seq_len: int = 30,
    decay_epochs: int = 15,
    min_teacher_forcing: float = 0.2,
    no_scheduled_sampling: bool = False,
    out_dir: Path = Path("work/v1_reeval"),
    ckpt_name: str = "best_dynamics.pt",
    logger=print,
    quick: bool = False,
):
    """Train LatentDynamics matching v1/scripts/train_dynamics.py."""
    mode_str = "Pure Teacher Forcing (Ablation)" if no_scheduled_sampling else "Scheduled Sampling"
    logger("\n" + "=" * 70)
    logger(f"[Dynamics Training - {mode_str}] Starting {epochs} epochs on device={device}")
    logger("=" * 70)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_ckpt_path = out_dir / ckpt_name

    n_train = len(train_starts)
    n_val = len(val_starts)

    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    offsets = np.arange(seq_len)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Teacher forcing schedule
        if no_scheduled_sampling:
            tf_ratio = 1.0
        else:
            if epoch <= decay_epochs:
                tf_ratio = 1.0 - (1.0 - min_teacher_forcing) * (epoch - 1) / (decay_epochs - 1 + 1e-8)
            else:
                tf_ratio = min_teacher_forcing

        model.train()
        train_loss = 0.0

        perm = np.random.permutation(n_train)
        if quick:
            perm = perm[: batch_size * 2]
            cur_train_len = len(perm)
        else:
            cur_train_len = n_train

        for i in range(0, cur_train_len, batch_size):
            b_starts = train_starts[perm[i : i + batch_size]]
            B = len(b_starts)
            idx_2d = torch.from_numpy(b_starts[:, None] + offsets[None, :]).to(device)

            z_seq = latents[idx_2d]  # (B, seq_len, latent_dim)
            act_seq = actions_tensor[idx_2d]  # (B, seq_len)

            input_z = z_seq[:, :-1]
            input_act = act_seq[:, :-1]
            target_z = z_seq[:, 1:]

            optimizer.zero_grad()
            pred_z, _ = model(input_z, input_act, teacher_forcing_ratio=tf_ratio)
            loss = F.mse_loss(pred_z, target_z)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        train_loss /= cur_train_len

        # Validation
        model.eval()
        val_loss_tf = 0.0
        val_loss_ar = 0.0

        val_eval_starts = val_starts[: batch_size * 2] if quick else val_starts
        cur_val_len = len(val_eval_starts)

        with torch.no_grad():
            for i in range(0, cur_val_len, batch_size):
                b_starts = val_eval_starts[i : i + batch_size]
                B = len(b_starts)
                idx_2d = torch.from_numpy(b_starts[:, None] + offsets[None, :]).to(device)

                z_seq = latents[idx_2d]
                act_seq = actions_tensor[idx_2d]

                input_z = z_seq[:, :-1]
                input_act = act_seq[:, :-1]
                target_z = z_seq[:, 1:]

                # 1. Teacher Forced validation
                pred_z_tf, _ = model(input_z, input_act, teacher_forcing_ratio=1.0)
                loss_tf = F.mse_loss(pred_z_tf, target_z)
                val_loss_tf += loss_tf.item() * B

                # 2. Pure Autoregressive validation
                pred_z_ar, _ = model(input_z, input_act, teacher_forcing_ratio=0.0)
                loss_ar = F.mse_loss(pred_z_ar, target_z)
                val_loss_ar += loss_ar.item() * B

        val_loss_tf /= cur_val_len
        val_loss_ar /= cur_val_len

        elapsed = time.time() - t0
        logger(
            f"Epoch {epoch:>2}/{epochs} ({elapsed:.1f}s) | tf_ratio: {tf_ratio:.2f} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss (TF): {val_loss_tf:.5f} | "
            f"Val Loss (AR): {val_loss_ar:.5f}"
        )

        if val_loss_ar < best_val_loss:
            best_val_loss = val_loss_ar
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss_ar": val_loss_ar,
                    "tf_ratio": tf_ratio,
                    "latent_dim": model.latent_dim,
                    "hidden_dim": model.hidden_dim,
                    "action_dim": model.action_dim,
                },
                best_ckpt_path,
            )
            logger(f"   [New Best AR] val_loss_ar: {val_loss_ar:.5f} -> saved {best_ckpt_path.name}")

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger(f"[Dynamics Training Complete - {mode_str}] Best Val Loss (AR): {best_val_loss:.5f}")
    return model, best_val_loss


# ─── EVALUATION ROUTINES ON HELD-OUT EPISODES ──────────────────────────────────

def evaluate_probe_on_split(
    probe,
    latents: torch.Tensor,
    ram_data: dict,
    episodes: list[int],
    device: torch.device,
    ep_len: int = 1000,
    batch_size: int = 1024,
) -> dict:
    """Evaluate probe map ID accuracy & position error on non-battle frames of held-out episodes.

    Computes per-episode means, then 95% bootstrap CIs via bootstrap_ci.
    """
    probe.eval()
    ep_map_accs = []
    ep_manhattans = []
    ep_per_axes = []

    pos_all = torch.tensor(
        np.stack([ram_data["xs"], ram_data["ys"]], axis=1), dtype=torch.float32, device=device
    )
    map_id_all = torch.tensor(ram_data["map_ids"], dtype=torch.long, device=device)
    in_battle_all = ram_data["in_battles"]  # numpy bool array

    with torch.no_grad():
        for ep in episodes:
            base = ep * ep_len
            ep_indices = np.arange(base, base + ep_len)
            # Filter frames that are not in battle
            non_battle_mask = ~in_battle_all[ep_indices]
            valid_indices = ep_indices[non_battle_mask]

            if len(valid_indices) == 0:
                continue

            ep_pos_err = 0.0
            ep_map_corr = 0.0
            n_ep_samples = len(valid_indices)

            for i in range(0, n_ep_samples, batch_size):
                b_idx = valid_indices[i : i + batch_size]
                mu = latents[b_idx]
                target_pos = pos_all[b_idx]
                target_map = map_id_all[b_idx]

                preds = probe(mu)
                diff = torch.abs(preds["pos"] - target_pos)
                ep_pos_err += torch.sum(diff).item()

                pred_map = torch.argmax(preds["map_logits"], dim=-1)
                ep_map_corr += torch.sum(pred_map == target_map).item()

            ep_map_acc = (ep_map_corr / n_ep_samples) * 100.0
            ep_manhattan = ep_pos_err / n_ep_samples
            ep_per_axis = ep_pos_err / (n_ep_samples * 2)

            ep_map_accs.append(ep_map_acc)
            ep_manhattans.append(ep_manhattan)
            ep_per_axes.append(ep_per_axis)

    ep_map_accs = np.array(ep_map_accs)
    ep_manhattans = np.array(ep_manhattans)
    ep_per_axes = np.array(ep_per_axes)

    map_m, map_lo, map_hi = bootstrap_ci(ep_map_accs)
    man_m, man_lo, man_hi = bootstrap_ci(ep_manhattans)
    pax_m, pax_lo, pax_hi = bootstrap_ci(ep_per_axes)

    return {
        "num_episodes": len(ep_map_accs),
        "map_accuracy": {
            "mean": float(map_m),
            "lo": float(map_lo),
            "hi": float(map_hi),
            "per_episode": ep_map_accs.tolist(),
        },
        "manhattan_pos_error": {
            "mean": float(man_m),
            "lo": float(man_lo),
            "hi": float(man_hi),
            "per_episode": ep_manhattans.tolist(),
        },
        "per_axis_pos_error": {
            "mean": float(pax_m),
            "lo": float(pax_lo),
            "hi": float(pax_hi),
            "per_episode": ep_per_axes.tolist(),
        },
    }


def evaluate_drift_on_split(
    model_ss,
    model_tf,
    probe,
    latents: torch.Tensor,
    ram_data: dict,
    episodes: list[int],
    device: torch.device,
    ep_len: int = 1000,
    seq_len: int = 30,
    batch_size: int = 128,
) -> dict:
    """Evaluate open-loop drift up to step 29 matching compare_drift.py.

    Evaluates SS, TF, and "nothing moves" baseline on all valid windows
    within each held-out episode. Aggregates per-episode means and computes
    bootstrap 95% CIs.
    """
    model_ss.eval()
    model_tf.eval()
    probe.eval()

    K = seq_len - 1  # 29 steps
    actions_tensor = torch.tensor(ram_data["actions"], dtype=torch.long, device=device)
    xs_all = torch.tensor(ram_data["xs"], dtype=torch.float32, device=device)
    ys_all = torch.tensor(ram_data["ys"], dtype=torch.float32, device=device)

    offsets = np.arange(seq_len)

    # Accumulate per-episode mean trajectories: shape (num_episodes, K)
    ep_ss_tile = []
    ep_tf_tile = []
    ep_none_tile = []
    ep_ss_latent = []
    ep_tf_latent = []

    with torch.no_grad():
        for ep in episodes:
            base = ep * ep_len
            n_windows = ep_len - seq_len + 1
            starts = base + np.arange(n_windows)

            ep_ss_tile_sum = np.zeros(K)
            ep_tf_tile_sum = np.zeros(K)
            ep_none_tile_sum = np.zeros(K)
            ep_ss_latent_sum = np.zeros(K)
            ep_tf_latent_sum = np.zeros(K)

            for i in range(0, n_windows, batch_size):
                b_starts = starts[i : i + batch_size]
                B = len(b_starts)
                idx_2d = torch.from_numpy(b_starts[:, None] + offsets[None, :]).to(device)

                z_seq = latents[idx_2d]  # (B, 30, 32)
                act_seq = actions_tensor[idx_2d]  # (B, 30)

                z_start = z_seq[:, 0]
                rollout_actions = act_seq[:, :-1]  # 29 actions

                # Rollouts
                pred_z_ss = model_ss.rollout(z_start, rollout_actions, device=device)  # (B, 29, 32)
                pred_z_tf = model_tf.rollout(z_start, rollout_actions, device=device)  # (B, 29, 32)

                gt_z = z_seq[:, 1:]  # (B, 29, 32)
                gt_pos = torch.stack(
                    [xs_all[idx_2d[:, 1:]], ys_all[idx_2d[:, 1:]]], dim=-1
                )  # (B, 29, 2)
                # Position at last observed step (step 0)
                last_pos = torch.stack(
                    [xs_all[idx_2d[:, 0]], ys_all[idx_2d[:, 0]]], dim=-1
                ).unsqueeze(1)  # (B, 1, 2)

                for k in range(K):
                    # Latent MSE
                    mse_ss = torch.mean((pred_z_ss[:, k] - gt_z[:, k]) ** 2, dim=-1)
                    mse_tf = torch.mean((pred_z_tf[:, k] - gt_z[:, k]) ** 2, dim=-1)
                    ep_ss_latent_sum[k] += torch.sum(mse_ss).item()
                    ep_tf_latent_sum[k] += torch.sum(mse_tf).item()

                    # Physical tile Manhattan error
                    gt_pos_k = gt_pos[:, k]
                    pred_pos_ss = probe(pred_z_ss[:, k])["pos"]
                    pred_pos_tf = probe(pred_z_tf[:, k])["pos"]

                    err_ss = torch.sum(torch.abs(pred_pos_ss - gt_pos_k), dim=-1)
                    err_tf = torch.sum(torch.abs(pred_pos_tf - gt_pos_k), dim=-1)
                    # "Nothing moves" baseline: assuming position at step 0
                    err_none = torch.sum(torch.abs(last_pos.squeeze(1) - gt_pos_k), dim=-1)

                    ep_ss_tile_sum[k] += torch.sum(err_ss).item()
                    ep_tf_tile_sum[k] += torch.sum(err_tf).item()
                    ep_none_tile_sum[k] += torch.sum(err_none).item()

            ep_ss_tile.append(ep_ss_tile_sum / n_windows)
            ep_tf_tile.append(ep_tf_tile_sum / n_windows)
            ep_none_tile.append(ep_none_tile_sum / n_windows)
            ep_ss_latent.append(ep_ss_latent_sum / n_windows)
            ep_tf_latent.append(ep_tf_latent_sum / n_windows)

    ep_ss_tile = np.array(ep_ss_tile)      # (num_episodes, 29)
    ep_tf_tile = np.array(ep_tf_tile)
    ep_none_tile = np.array(ep_none_tile)
    ep_ss_latent = np.array(ep_ss_latent)
    ep_tf_latent = np.array(ep_tf_latent)

    def _ci_dict(arr_2d):
        m, lo, hi = bootstrap_ci(arr_2d)
        return {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}

    return {
        "steps": list(range(1, K + 1)),
        "num_episodes": len(ep_ss_tile),
        "ss_tile_error": _ci_dict(ep_ss_tile),
        "tf_tile_error": _ci_dict(ep_tf_tile),
        "nothing_moves_tile_error": _ci_dict(ep_none_tile),
        "ss_latent_mse": _ci_dict(ep_ss_latent),
        "tf_latent_mse": _ci_dict(ep_tf_latent),
    }


# ─── MAIN DRIVER & CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-measure PokeDreamer v1 on held-out episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--quick", action="store_true",
                        help="Run tiny smoke test with few batches/epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=Path, default=Path("work/v1_reeval"),
                        help="Directory to save logs, weights, and results")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    start_time = time.time()
    logger("=" * 80)
    logger("POKEDREAMER v1 FAITHFUL RE-EVALUATION ON HELD-OUT EPISODES")
    logger(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"CLI: device={args.device}, quick={args.quick}, seed={args.seed}")
    logger("=" * 80)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    logger(f"Using PyTorch device: {device}")

    # Load v1 model classes
    VAE, LatentDynamics, RAMProbe = load_v1_models()

    # Load shared dataset and splits
    data, splits = load_data()
    train_pool = splits["train"]
    test_id = splits["test_id"]
    test_ood = splits["test_ood"]

    # Carve validation episodes out of train pool (never touching test episodes)
    val_count = 1 if args.quick else 10
    train_eps, val_eps = carve_train_val_episodes(train_pool, val_count=val_count, seed=args.seed)
    if args.quick:
        train_eps = train_eps[:2]
        test_id = test_id[:1]
        test_ood = test_ood[:1]

    logger(
        f"[Splits] Train: {len(train_eps)} eps | Val (carved): {len(val_eps)} eps | "
        f"Test ID: {len(test_id)} eps | Test OOD: {len(test_ood)} eps"
    )

    # Ensure 40x36 frames cache and extra RAM fields
    cache_path = out_dir / "frames_40x36.npy"
    frames, extra_ram = ensure_cached_frames(cache_path, data, logger)

    # Combine RAM fields
    ram_data = {
        "actions": data["actions"],
        "map_ids": data["map_ids"],
        "xs": data["xs"],
        "ys": data["ys"],
        "in_battles": data["in_battles"],
        "dialog_opens": extra_ram["dialog_opens"],
        "party_hps": extra_ram["party_hps"],
        "party_max_hps": extra_ram["party_max_hps"],
    }

    # Frame indices for VAE and Probe training
    ep_len = 1000
    train_frame_indices = np.concatenate([np.arange(ep * ep_len, (ep + 1) * ep_len) for ep in train_eps])
    val_frame_indices = np.concatenate([np.arange(ep * ep_len, (ep + 1) * ep_len) for ep in val_eps])

    # 1. Train VAE
    vae = VAE(latent_dim=32).to(device)
    vae_epochs = 1 if args.quick else 15
    vae, best_vae_loss = train_vae(
        vae,
        frames,
        train_frame_indices,
        val_frame_indices,
        device=device,
        epochs=vae_epochs,
        batch_size=128,
        lr=1e-3,
        beta=1.0,
        out_dir=out_dir,
        logger=logger,
        quick=args.quick,
    )

    # Freeze VAE and precompute latents for probe and dynamics
    for param in vae.parameters():
        param.requires_grad = False
    all_latents = precompute_latents(vae, frames, device=device, batch_size=1024, logger=logger)

    # 2. Train RAM Probe
    probe = RAMProbe(latent_dim=32, num_classes_map=256).to(device)
    probe_epochs = 1 if args.quick else 10
    probe, best_probe_loss = train_probe(
        probe,
        all_latents,
        ram_data,
        train_frame_indices,
        val_frame_indices,
        device=device,
        epochs=probe_epochs,
        batch_size=128,
        lr=1e-3,
        out_dir=out_dir,
        logger=logger,
        quick=args.quick,
    )

    # 3. Train Dynamics Models
    train_starts, _ = get_valid_window_indices(train_eps, ep_len=1000, seq_len=30)
    val_starts, _ = get_valid_window_indices(val_eps, ep_len=1000, seq_len=30)

    dyn_epochs = 1 if args.quick else 20

    # 3.1 Scheduled Sampling Dynamics
    dyn_ss = LatentDynamics(latent_dim=32, num_actions=8, action_dim=16, hidden_dim=256).to(device)
    dyn_ss, best_dyn_ss_loss = train_dynamics(
        dyn_ss,
        all_latents,
        ram_data["actions"],
        train_starts,
        val_starts,
        device=device,
        epochs=dyn_epochs,
        batch_size=128,
        lr=1e-3,
        seq_len=30,
        decay_epochs=15,
        min_teacher_forcing=0.2,
        no_scheduled_sampling=False,
        out_dir=out_dir,
        ckpt_name="best_dynamics.pt",
        logger=logger,
        quick=args.quick,
    )

    # 3.2 Pure Teacher Forcing Ablation Dynamics
    dyn_tf = LatentDynamics(latent_dim=32, num_actions=8, action_dim=16, hidden_dim=256).to(device)
    dyn_tf, best_dyn_tf_loss = train_dynamics(
        dyn_tf,
        all_latents,
        ram_data["actions"],
        train_starts,
        val_starts,
        device=device,
        epochs=dyn_epochs,
        batch_size=128,
        lr=1e-3,
        seq_len=30,
        decay_epochs=15,
        min_teacher_forcing=0.2,
        no_scheduled_sampling=True,
        out_dir=out_dir,
        ckpt_name="best_dynamics_ablation.pt",
        logger=logger,
        quick=args.quick,
    )

    # 4. Evaluation on Held-Out Episodes (test_id and test_ood)
    logger("\n" + "=" * 80)
    logger("EVALUATION ON HELD-OUT EPISODES")
    logger("=" * 80)

    eval_results = {}
    for split_name, split_eps in [("test_id", test_id), ("test_ood", test_ood)]:
        logger(f"\nEvaluating on {split_name} ({len(split_eps)} episodes)...")
        t_eval = time.time()

        probe_res = evaluate_probe_on_split(
            probe,
            all_latents,
            ram_data,
            split_eps,
            device=device,
            ep_len=1000,
        )

        drift_res = evaluate_drift_on_split(
            dyn_ss,
            dyn_tf,
            probe,
            all_latents,
            ram_data,
            split_eps,
            device=device,
            ep_len=1000,
            seq_len=30,
        )

        eval_results[split_name] = {
            "probe": probe_res,
            "drift": drift_res,
        }
        logger(f"Done {split_name} in {time.time()-t_eval:.2f}s")

    total_time = time.time() - start_time

    # Published v1 numbers for reference
    published_v1 = {
        "probe": {
            "map_id_acc_percent": 98.7,
            "pos_mae_tiles": 1.23,
        },
        "drift": {
            "steps": [1, 5, 10, 15, 20, 25, 29],
            "ss_tile_error": [3.72, 3.32, 3.30, 3.33, 3.36, 3.42, 3.47],
            "tf_tile_error": [4.06, 5.08, 6.47, 7.81, 9.13, 9.72, 10.44],
            "ss_latent_mse": [0.09268, 0.08751, 0.08842, 0.09238, 0.09830, 0.10606, 0.11197],
            "tf_latent_mse": [0.12240, 0.23061, 0.40189, 0.59876, 0.78455, 0.92376, 1.04063],
        },
    }

    results_data = {
        "meta": {
            "device": str(device),
            "quick": args.quick,
            "seed": args.seed,
            "runtime_seconds": round(total_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "published_v1": published_v1,
        "hyperparameters": {
            "vae": {
                "latent_dim": 32,
                "beta": 1.0,
                "learning_rate": 1e-3,
                "batch_size": 128,
                "epochs": vae_epochs,
                "selection": "min val_loss",
                "source_line": "v1/scripts/train_vae.py:38-47",
            },
            "probe": {
                "input_dim": 32,
                "hidden_dim": 64,
                "learning_rate": 1e-3,
                "batch_size": 128,
                "epochs": probe_epochs,
                "selection": "min val_loss",
                "source_line": "v1/scripts/train_probe.py:39-44",
            },
            "dynamics_ss": {
                "seq_len": 30,
                "hidden_dim": 256,
                "action_dim": 16,
                "decay_epochs": 15,
                "min_teacher_forcing": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 128,
                "epochs": dyn_epochs,
                "selection": "min val_loss_ar",
                "source_line": "v1/scripts/train_dynamics.py:41-56",
            },
            "dynamics_tf": {
                "seq_len": 30,
                "hidden_dim": 256,
                "action_dim": 16,
                "no_scheduled_sampling": True,
                "learning_rate": 1e-3,
                "batch_size": 128,
                "epochs": dyn_epochs,
                "selection": "min val_loss_ar",
                "source_line": "v1/scripts/train_dynamics.py:57",
            },
        },
        "training": {
            "train_episodes": len(train_eps),
            "val_episodes": len(val_eps),
            "vae_val_loss": round(float(best_vae_loss), 2),
            "probe_val_loss": round(float(best_probe_loss), 4),
            "dynamics_ss_val_loss_ar": round(float(best_dyn_ss_loss), 5),
            "dynamics_tf_val_loss_ar": round(float(best_dyn_tf_loss), 5),
        },
        "evaluation": eval_results,
    }

    # Save results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    logger(f"\n[Saved] Complete results written to {results_path}")

    # ── Print Compact Markdown Results Table ───────────────────────────────────
    logger("\n" + "=" * 80)
    logger("FINAL RESULTS SUMMARY: PokeDreamer v1 Re-measured vs Published")
    logger("=" * 80)

    # 1. Probe Comparison
    p_pub_acc = published_v1["probe"]["map_id_acc_percent"]
    p_pub_mae = published_v1["probe"]["pos_mae_tiles"]

    t_id_p = eval_results["test_id"]["probe"]
    t_ood_p = eval_results["test_ood"]["probe"]

    logger("\n### 1. RAM State Probe Evaluation")
    logger(
        "| Metric | Published v1 (No split) | test_id (16 held-out eps) [95% CI] | "
        "test_ood (45 held-out eps) [95% CI] |"
    )
    logger("|---|---|---|---|")
    logger(
        f"| Map ID Acc (%) | {p_pub_acc:.1f}% | "
        f"{t_id_p['map_accuracy']['mean']:.1f}% [{t_id_p['map_accuracy']['lo']:.1f}, {t_id_p['map_accuracy']['hi']:.1f}] | "
        f"{t_ood_p['map_accuracy']['mean']:.1f}% [{t_ood_p['map_accuracy']['lo']:.1f}, {t_ood_p['map_accuracy']['hi']:.1f}] |"
    )
    logger(
        f"| Coordinate Pos MAE (tiles/axis) | {p_pub_mae:.2f} | "
        f"{t_id_p['per_axis_pos_error']['mean']:.2f} [{t_id_p['per_axis_pos_error']['lo']:.2f}, {t_id_p['per_axis_pos_error']['hi']:.2f}] | "
        f"{t_ood_p['per_axis_pos_error']['mean']:.2f} [{t_ood_p['per_axis_pos_error']['lo']:.2f}, {t_ood_p['per_axis_pos_error']['hi']:.2f}] |"
    )
    logger(
        f"| Coordinate Manhattan (tiles) | {p_pub_mae * 2:.2f}* | "
        f"{t_id_p['manhattan_pos_error']['mean']:.2f} [{t_id_p['manhattan_pos_error']['lo']:.2f}, {t_id_p['manhattan_pos_error']['hi']:.2f}] | "
        f"{t_ood_p['manhattan_pos_error']['mean']:.2f} [{t_ood_p['manhattan_pos_error']['lo']:.2f}, {t_ood_p['manhattan_pos_error']['hi']:.2f}] |"
    )
    logger("* Note: v1 defined pos MAE as sum(|dx|+|dy|)/(2N). Manhattan distance is sum(|dx|+|dy|)/N.")

    # 2. Open-Loop Drift Comparison
    logger("\n### 2. Open-Loop Rollout Drift (Manhattan Tile Error) on Held-Out Windows")
    logger(
        "| Step | v1 Published SS | v1 Published TF | test_id SS (95% CI) | test_id TF (95% CI) | "
        "test_id Nothing Moves (95% CI) |"
    )
    logger("|---|---|---|---|---|---|")

    steps_to_report = [1, 5, 10, 15, 20, 29]
    t_id_d = eval_results["test_id"]["drift"]
    t_ood_d = eval_results["test_ood"]["drift"]

    for s in steps_to_report:
        idx = s - 1
        pub_idx = published_v1["drift"]["steps"].index(s)
        pub_ss = published_v1["drift"]["ss_tile_error"][pub_idx]
        pub_tf = published_v1["drift"]["tf_tile_error"][pub_idx]

        ss_m = t_id_d["ss_tile_error"]["mean"][idx]
        ss_lo = t_id_d["ss_tile_error"]["lo"][idx]
        ss_hi = t_id_d["ss_tile_error"]["hi"][idx]

        tf_m = t_id_d["tf_tile_error"]["mean"][idx]
        tf_lo = t_id_d["tf_tile_error"]["lo"][idx]
        tf_hi = t_id_d["tf_tile_error"]["hi"][idx]

        none_m = t_id_d["nothing_moves_tile_error"]["mean"][idx]
        none_lo = t_id_d["nothing_moves_tile_error"]["lo"][idx]
        none_hi = t_id_d["nothing_moves_tile_error"]["hi"][idx]

        logger(
            f"| Step {s:>2} | {pub_ss:.2f} | {pub_tf:.2f} | "
            f"{ss_m:.2f} [{ss_lo:.2f}, {ss_hi:.2f}] | "
            f"{tf_m:.2f} [{tf_lo:.2f}, {tf_hi:.2f}] | "
            f"{none_m:.2f} [{none_lo:.2f}, {none_hi:.2f}] |"
        )

    logger("\n### 3. Open-Loop Rollout Drift (Manhattan Tile Error) on test_ood (Held-Out Start Location)")
    logger(
        "| Step | test_ood SS (95% CI) | test_ood TF (95% CI) | test_ood Nothing Moves (95% CI) |"
    )
    logger("|---|---|---|---|")
    for s in steps_to_report:
        idx = s - 1
        ss_m = t_ood_d["ss_tile_error"]["mean"][idx]
        ss_lo = t_ood_d["ss_tile_error"]["lo"][idx]
        ss_hi = t_ood_d["ss_tile_error"]["hi"][idx]

        tf_m = t_ood_d["tf_tile_error"]["mean"][idx]
        tf_lo = t_ood_d["tf_tile_error"]["lo"][idx]
        tf_hi = t_ood_d["tf_tile_error"]["hi"][idx]

        none_m = t_ood_d["nothing_moves_tile_error"]["mean"][idx]
        none_lo = t_ood_d["nothing_moves_tile_error"]["lo"][idx]
        none_hi = t_ood_d["nothing_moves_tile_error"]["hi"][idx]

        logger(
            f"| Step {s:>2} | "
            f"{ss_m:.2f} [{ss_lo:.2f}, {ss_hi:.2f}] | "
            f"{tf_m:.2f} [{tf_lo:.2f}, {tf_hi:.2f}] | "
            f"{none_m:.2f} [{none_lo:.2f}, {none_hi:.2f}] |"
        )

    logger(f"\nTotal re-evaluation runtime: {total_time:.1f}s ({total_time/60:.2f} min)")
    logger.close()


if __name__ == "__main__":
    main()
