"""
evaluate_drift.py — Day 8-11 deliverable script.

Evaluates the multi-step rollout drift of the LatentDynamics model.
Measures both latent MSE drift and physical tile distance drift (via RAM probe).
Generates the drift curve plot out to K steps.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/evaluate_drift.py \
        --data-dir data \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --dynamics-checkpoint checkpoints/dynamics/best_dynamics.pt \
        --probe-checkpoint checkpoints/probe/best_probe.pt \
        --seq-len 30 \
        --out-dir checkpoints/dynamics
"""

import argparse
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.dynamics import LatentDynamics
from src.probe import RAMProbe
from src.vae import VAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multi-step rollout drift")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory with .npz chunk files")
    p.add_argument("--vae-checkpoint", type=Path, required=True,
                   help="Path to trained VAE checkpoint best_vae.pt")
    p.add_argument("--dynamics-checkpoint", type=Path, required=True,
                   help="Path to trained LatentDynamics checkpoint best_dynamics.pt")
    p.add_argument("--probe-checkpoint", type=Path, default=None,
                   help="Path to trained RAMProbe checkpoint (optional, for physical drift)")
    p.add_argument("--seq-len", type=int, default=30,
                   help="Maximum rollout horizon to test (default: 30)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for evaluation (default: 64)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/dynamics"),
                   help="Directory to save the evaluation plots")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_drift] Using device: {device}")

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = VAE(latent_dim=vae_ckpt['latent_dim']).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    # ── Load Dynamics ─────────────────────────────────────────────────────────
    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device)
    dynamics = LatentDynamics(
        latent_dim=dyn_ckpt['latent_dim'],
        num_actions=8,
        action_dim=dyn_ckpt['action_dim'],
        hidden_dim=dyn_ckpt['hidden_dim']
    ).to(device)
    dynamics.load_state_dict(dyn_ckpt['model_state_dict'])
    dynamics.eval()

    # ── Load Probe (Optional) ─────────────────────────────────────────────────
    probe = None
    if args.probe_checkpoint and args.probe_checkpoint.exists():
        print(f"[evaluate_drift] Loading RAM Probe from {args.probe_checkpoint}")
        probe_ckpt = torch.load(args.probe_checkpoint, map_location=device)
        probe = RAMProbe(latent_dim=vae_ckpt['latent_dim']).to(device)
        probe.load_state_dict(probe_ckpt['model_state_dict'])
        probe.eval()
    else:
        print("[evaluate_drift] No RAM Probe provided. Physical tile drift will not be computed.")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    try:
        # We load validation transitions
        dataset = PokemonDataset(args.data_dir, seq_len=args.seq_len)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Rollout Drift Evaluation ──────────────────────────────────────────────
    K = args.seq_len - 1 # Rollout horizon steps (T-1)
    
    latent_drift = np.zeros(K)
    tile_drift = np.zeros(K)
    counts = 0

    print(f"\nEvaluating rollout drift up to {K} steps...")
    
    for batch in loader:
        obs = batch['obs'].to(device) # (B, T, C, H, W)
        actions = batch['actions'].to(device) # (B, T)
        
        B, T, C, H, W = obs.shape
        
        # 1. Encode all frames to ground-truth latents
        with torch.no_grad():
            flat_obs = obs.view(B * T, C, H, W)
            mu, _ = vae.encode(flat_obs)
            z_seq = mu.view(B, T, -1) # (B, T, latent_dim)
            
        # 2. Run autoregressive rollout starting from z_0
        z_start = z_seq[:, 0]
        # actions[:, :-1] has shape (B, K)
        rollout_actions = actions[:, :-1]
        
        # Roll out
        with torch.no_grad():
            pred_z_seq = dynamics.rollout(z_start, rollout_actions, device=device) # (B, K, latent_dim)
            
        # 3. Compute step-by-step drift
        for k in range(K):
            # Ground truth latent at step k+1
            gt_z_k = z_seq[:, k + 1]
            pred_z_k = pred_z_seq[:, k]
            
            # Latent MSE drift
            mse = torch.mean((pred_z_k - gt_z_k) ** 2, dim=-1) # (B,)
            latent_drift[k] += torch.sum(mse).item()
            
            # Physical tile drift (if probe is available)
            if probe is not None:
                gt_pos = torch.stack([batch['xs'][:, k + 1], batch['ys'][:, k + 1]], dim=1).to(device) # (B, 2)
                pred_pos = probe(pred_z_k)['pos'] # (B, 2)
                
                # Manhattan distance
                manhattan = torch.sum(torch.abs(pred_pos - gt_pos), dim=-1) # (B,)
                tile_drift[k] += torch.sum(manhattan).item()
                
        counts += B

    # Normalize by total sample count
    latent_drift /= counts
    if probe is not None:
        tile_drift /= counts

    # ── Report Headline Claims ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ROLLOUT DRIFT RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Step 1  | Latent MSE: {latent_drift[0]:.6f}" + (f" | Tile Drift: {tile_drift[0]:.2f} tiles" if probe else ""))
    print(f"  Step 5  | Latent MSE: {latent_drift[4]:.6f}" + (f" | Tile Drift: {tile_drift[4]:.2f} tiles" if probe else ""))
    print(f"  Step 10 | Latent MSE: {latent_drift[9]:.6f}" + (f" | Tile Drift: {tile_drift[9]:.2f} tiles" if probe else ""))
    print(f"  Step 15 | Latent MSE: {latent_drift[14]:.6f}" + (f" | Tile Drift: {tile_drift[14]:.2f} tiles" if probe else ""))
    print(f"  Step 25 | Latent MSE: {latent_drift[24]:.6f}" + (f" | Tile Drift: {tile_drift[24]:.2f} tiles" if probe else ""))
    print("=" * 50)

    # ── Plot and Save Drift Curves ────────────────────────────────────────────
    steps = np.arange(1, K + 1)
    
    plt.figure(figsize=(10, 4))
    
    # Subplot 1: Latent Drift
    plt.subplot(1, 2, 1)
    plt.plot(steps, latent_drift, marker='o', color='blue', linewidth=2)
    plt.xlabel('Rollout Step (k)')
    plt.ylabel('Latent Space MSE')
    plt.title('Latent Space Drift Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Physical Drift (if probe available)
    if probe is not None:
        plt.subplot(1, 2, 2)
        plt.plot(steps, tile_drift, marker='s', color='orange', linewidth=2)
        plt.xlabel('Rollout Step (k)')
        plt.ylabel('Mean Manhattan Distance (tiles)')
        plt.title('Player Position Prediction Drift')
        plt.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plot_path = args.out_dir / "drift_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\n[evaluate_drift] Drift curves saved to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
