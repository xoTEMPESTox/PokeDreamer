"""
compare_drift.py — Day 14 deliverable script.

Compares the rollout drift of the scheduled-sampling dynamics model versus
the pure teacher forcing ablation model. Saves a combined comparison plot.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/compare_drift.py \
        --data-dir data \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --dynamics-checkpoint checkpoints/dynamics/best_dynamics.pt \
        --ablation-checkpoint checkpoints/dynamics/best_dynamics_ablation.pt \
        --probe-checkpoint checkpoints/probe/best_probe.pt \
        --seq-len 30
"""

import argparse
import random
import sys
from pathlib import Path
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
    p = argparse.ArgumentParser(description="Compare rollout drift of dynamics models")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory with .npz chunk files")
    p.add_argument("--vae-checkpoint", type=Path, required=True,
                   help="Path to trained VAE checkpoint")
    p.add_argument("--dynamics-checkpoint", type=Path, required=True,
                   help="Path to trained scheduled-sampling dynamics checkpoint")
    p.add_argument("--ablation-checkpoint", type=Path, required=True,
                   help="Path to trained ablation dynamics checkpoint (pure teacher forcing)")
    p.add_argument("--probe-checkpoint", type=Path, default=None,
                   help="Path to trained RAMProbe checkpoint")
    p.add_argument("--seq-len", type=int, default=30,
                   help="Maximum rollout horizon to test (default: 30)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/dynamics"),
                   help="Directory to save comparison plot")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model_drift(model, vae, probe, loader, K, device):
    latent_drift = np.zeros(K)
    tile_drift = np.zeros(K)
    counts = 0
    
    for batch in loader:
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)
        
        B, T, C, H, W = obs.shape
        
        with torch.no_grad():
            flat_obs = obs.view(B * T, C, H, W)
            mu, _ = vae.encode(flat_obs)
            z_seq = mu.view(B, T, -1)
            
        z_start = z_seq[:, 0]
        rollout_actions = actions[:, :-1]
        
        with torch.no_grad():
            pred_z_seq = model.rollout(z_start, rollout_actions, device=device)
            
        for k in range(K):
            gt_z_k = z_seq[:, k + 1]
            pred_z_k = pred_z_seq[:, k]
            
            # Latent MSE
            mse = torch.mean((pred_z_k - gt_z_k) ** 2, dim=-1)
            latent_drift[k] += torch.sum(mse).item()
            
            # Physical tile drift
            if probe is not None:
                gt_pos = torch.stack([batch['xs'][:, k + 1], batch['ys'][:, k + 1]], dim=1).to(device)
                pred_pos = probe(pred_z_k)['pos']
                manhattan = torch.sum(torch.abs(pred_pos - gt_pos), dim=-1)
                tile_drift[k] += torch.sum(manhattan).item()
                
        counts += B
        
    latent_drift /= counts
    if probe is not None:
        tile_drift /= counts
        
    return latent_drift, tile_drift


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare_drift] Using device: {device}")

    # ── Load VAE & Probe ──────────────────────────────────────────────────────
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = VAE(latent_dim=vae_ckpt['latent_dim']).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    probe = None
    if args.probe_checkpoint and args.probe_checkpoint.exists():
        probe_ckpt = torch.load(args.probe_checkpoint, map_location=device)
        probe = RAMProbe(latent_dim=vae_ckpt['latent_dim']).to(device)
        probe.load_state_dict(probe_ckpt['model_state_dict'])
        probe.eval()

    # ── Load Models ───────────────────────────────────────────────────────────
    # 1. Scheduled Sampling Model
    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device)
    model_ss = LatentDynamics(
        latent_dim=dyn_ckpt['latent_dim'],
        num_actions=8,
        action_dim=dyn_ckpt['action_dim'],
        hidden_dim=dyn_ckpt['hidden_dim']
    ).to(device)
    model_ss.load_state_dict(dyn_ckpt['model_state_dict'])
    model_ss.eval()
    
    # 2. Pure Teacher Forcing Model (Ablation)
    abl_ckpt = torch.load(args.ablation_checkpoint, map_location=device)
    model_tf = LatentDynamics(
        latent_dim=abl_ckpt['latent_dim'],
        num_actions=8,
        action_dim=abl_ckpt['action_dim'],
        hidden_dim=abl_ckpt['hidden_dim']
    ).to(device)
    model_tf.load_state_dict(abl_ckpt['model_state_dict'])
    model_tf.eval()

    # ── Load Dataset ──────────────────────────────────────────────────────────
    try:
        dataset = PokemonDataset(args.data_dir, seq_len=args.seq_len)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    K = args.seq_len - 1
    
    print("\nEvaluating Scheduled Sampling model...")
    latent_ss, tile_ss = evaluate_model_drift(model_ss, vae, probe, loader, K, device)
    
    print("Evaluating Pure Teacher Forcing model (Ablation)...")
    latent_tf, tile_tf = evaluate_model_drift(model_tf, vae, probe, loader, K, device)

    steps = np.arange(1, K + 1)
    # Save metrics to JSON file
    import json
    # Convert numpy arrays to list for json serialization
    drift_data = {
        'steps': list(map(int, steps)),
        'latent_ss': list(map(float, latent_ss)),
        'latent_tf': list(map(float, latent_tf)),
        'tile_ss': list(map(float, tile_ss)) if tile_ss is not None else None,
        'tile_tf': list(map(float, tile_tf)) if tile_tf is not None else None
    }
    json_path = args.out_dir / "drift_comparison.json"
    with open(json_path, "w") as f:
        json.dump(drift_data, f, indent=2)
        
    print(f"\n[SUCCESS] Ablation comparison data saved to: {json_path.resolve()}")
    
    # Print comparison at key intervals
    print("\nRollout Step | SS Latent MSE | TF Latent MSE | SS Tile Error | TF Tile Error")
    print("-" * 75)
    for idx in range(K):
        step = idx + 1
        if step in [1, 5, 10, 15, 20, 25, K]:
            t_ss = f"{tile_ss[idx]:.2f}" if tile_ss is not None else "N/A"
            t_tf = f"{tile_tf[idx]:.2f}" if tile_tf is not None else "N/A"
            print(f"Step {step:>11} | {latent_ss[idx]:.5f}      | {latent_tf[idx]:.5f}      | {t_ss:>12} | {t_tf:>12}")


if __name__ == "__main__":
    main()
