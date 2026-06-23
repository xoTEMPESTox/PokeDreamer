"""
train_dynamics.py — Day 6-7 deliverable script.

Trains the LatentDynamics model on sequence transitions.
Supports Scheduled Sampling and Ablation mode (Pure Teacher Forcing).

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/train_dynamics.py \
        --data-dir data \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --epochs 20 \
        --batch-size 128 \
        --seq-len 30 \
        --out-dir checkpoints/dynamics
"""

import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.dynamics import LatentDynamics
from src.vae import VAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Recurrent Latent Dynamics on Pokémon Red")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory with .npz chunk files")
    p.add_argument("--vae-checkpoint", type=Path, required=True,
                   help="Path to trained VAE checkpoint best_vae.pt")
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of epochs to train (default: 20)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size (default: 128)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--seq-len", type=int, default=30,
                   help="Sequence length for BPTT (default: 30)")
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="Dimension of GRU hidden state (default: 256)")
    p.add_argument("--action-dim", type=int, default=16,
                   help="Dimension of action embedding (default: 16)")
    p.add_argument("--decay-epochs", type=int, default=15,
                   help="Epochs to linearly decay teacher forcing to min ratio (default: 15)")
    p.add_argument("--min-teacher-forcing", type=float, default=0.2,
                   help="Minimum teacher forcing ratio at end of decay (default: 0.2)")
    p.add_argument("--no-scheduled-sampling", action="store_true",
                   help="Ablation: disable scheduled sampling (force 1.0 teacher forcing)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/dynamics"),
                   help="Directory to save model checkpoints and logs")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Validation split ratio (default: 0.1)")
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
    print(f"[train_dynamics] Using device: {device}")

    # ── Load VAE Checkpoint ───────────────────────────────────────────────────
    print(f"[train_dynamics] Loading VAE from {args.vae_checkpoint}")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    latent_dim = vae_ckpt['latent_dim']
    
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"[train_dynamics] VAE loaded OK | latent_dim={latent_dim}")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    try:
        full_dataset = PokemonDataset(args.data_dir, seq_len=args.seq_len)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[train_dynamics] Train sequences: {train_size} | Val sequences: {val_size}")

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    dynamics = LatentDynamics(
        latent_dim=latent_dim,
        num_actions=8, # standard action space size
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.Adam(dynamics.parameters(), lr=args.lr)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss_tf': [], 'val_loss_ar': []}

    print("\nStarting dynamics training...")
    for epoch in range(1, args.epochs + 1):
        # Calculate teacher forcing ratio
        if args.no_scheduled_sampling:
            tf_ratio = 1.0
        else:
            # Linear decay to min ratio
            if epoch <= args.decay_epochs:
                tf_ratio = 1.0 - (1.0 - args.min_teacher_forcing) * (epoch - 1) / (args.decay_epochs - 1 + 1e-8)
            else:
                tf_ratio = args.min_teacher_forcing

        # Train Epoch
        dynamics.train()
        train_loss = 0
        
        for batch in train_loader:
            obs = batch['obs'].to(device) # (B, T, C, H, W)
            actions = batch['actions'].to(device) # (B, T)
            
            B, T, C, H, W = obs.shape
            
            # Encode frames sequence to latents sequence
            with torch.no_grad():
                flat_obs = obs.reshape(B * T, C, H, W)
                mu, _ = vae.encode(flat_obs)
                z_seq = mu.reshape(B, T, latent_dim) # (B, T, latent_dim)
            
            # Predict transitions: z_t, action_t -> z_{t+1}
            # The inputs are steps 0 to T-2
            input_z = z_seq[:, :-1]
            input_act = actions[:, :-1]
            target_z = z_seq[:, 1:] # targets are steps 1 to T-1
            
            optimizer.zero_grad()
            pred_z, _ = dynamics(input_z, input_act, teacher_forcing_ratio=tf_ratio)
            
            # Loss is MSE between predicted next latent and VAE target latent
            loss = F.mse_loss(pred_z, target_z)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * B
            
        train_loss /= len(train_dataset)

        # Validate Epoch
        dynamics.eval()
        val_loss_tf = 0 # Teacher forced validation loss
        val_loss_ar = 0 # Pure autoregressive (imagine) validation loss
        
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                actions = batch['actions'].to(device)
                
                B, T, C, H, W = obs.shape
                
                flat_obs = obs.view(B * T, C, H, W)
                mu, _ = vae.encode(flat_obs)
                z_seq = mu.view(B, T, latent_dim)
                
                input_z = z_seq[:, :-1]
                input_act = actions[:, :-1]
                target_z = z_seq[:, 1:]
                
                # 1. Teacher Forced validation
                pred_z_tf, _ = dynamics(input_z, input_act, teacher_forcing_ratio=1.0)
                loss_tf = F.mse_loss(pred_z_tf, target_z)
                val_loss_tf += loss_tf.item() * B
                
                # 2. Pure Autoregressive (imagine/evaluation mode)
                pred_z_ar, _ = dynamics(input_z, input_act, teacher_forcing_ratio=0.0)
                loss_ar = F.mse_loss(pred_z_ar, target_z)
                val_loss_ar += loss_ar.item() * B
                
        val_loss_tf /= len(val_dataset)
        val_loss_ar /= len(val_dataset)

        history['train_loss'].append(train_loss)
        history['val_loss_tf'].append(val_loss_tf)
        history['val_loss_ar'].append(val_loss_ar)

        print(f"Epoch {epoch:>2}/{args.epochs} | tf_ratio: {tf_ratio:.2f} | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val Loss (TF): {val_loss_tf:.5f} | "
              f"Val Loss (AR): {val_loss_ar:.5f}")

        # Checkpoint if best AR validation loss
        if val_loss_ar < best_val_loss:
            best_val_loss = val_loss_ar
            ckpt_name = "best_dynamics_ablation.pt" if args.no_scheduled_sampling else "best_dynamics.pt"
            ckpt_path = args.out_dir / ckpt_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': dynamics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss_ar': val_loss_ar,
                'tf_ratio': tf_ratio,
                'latent_dim': latent_dim,
                'hidden_dim': args.hidden_dim,
                'action_dim': args.action_dim
            }, ckpt_path)
            print(f"   [New Best AR] Checkpoint saved: {ckpt_name}")

    # Save loss history to JSON file
    import json
    history_name = "history_dynamics_ablation.json" if args.no_scheduled_sampling else "history_dynamics.json"
    history_path = args.out_dir / history_name
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[train_dynamics] Loss history saved to {history_path.resolve()}")


if __name__ == "__main__":
    main()
