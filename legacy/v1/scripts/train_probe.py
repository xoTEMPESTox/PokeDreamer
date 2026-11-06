"""
train_probe.py — Day 8-11 deliverable script.

Trains the RAMProbe on VAE latents to predict game position, map, battle,
dialog, and party HP. This serves as our evaluation probe.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/train_probe.py \
        --data-dir data \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --epochs 10 \
        --batch-size 128 \
        --out-dir checkpoints/probe
"""

import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.probe import RAMProbe
from src.vae import VAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RAM Probe on VAE latents")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory with .npz chunk files")
    p.add_argument("--vae-checkpoint", type=Path, required=True,
                   help="Path to trained VAE checkpoint best_vae.pt")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of epochs to train (default: 10)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size (default: 128)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/probe"),
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
    print(f"[train_probe] Using device: {device}")

    # ── Load VAE Checkpoint ───────────────────────────────────────────────────
    print(f"[train_probe] Loading VAE from {args.vae_checkpoint}")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    latent_dim = vae_ckpt['latent_dim']
    
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"[train_probe] VAE loaded OK | latent_dim={latent_dim}")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    try:
        full_dataset = PokemonDataset(args.data_dir, seq_len=1)
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

    print(f"[train_probe] Train samples: {train_size} | Val samples: {val_size}")

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    probe = RAMProbe(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_pos_mae': [], 'val_map_acc': []}

    print("\nStarting RAM Probe training...")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        train_loss = 0
        
        for batch in train_loader:
            x = batch['obs'].to(device)
            
            # Encode frames sequence to latents sequence
            with torch.no_grad():
                mu, _ = vae.encode(x)
            
            # Target dictionary preparation
            targets = {
                'pos': torch.stack([batch['x'], batch['y']], dim=1).float().to(device),
                'map_id': batch['map_id'].to(device),
                'in_battle': batch['in_battle'].to(device),
                'dialog_open': batch['dialog_open'].to(device),
                # Normalize HP relative to max HP
                'hp': (batch['party_hp'] / (batch['party_max_hp'] + 1e-5)).to(device)
            }
            
            optimizer.zero_grad()
            preds = probe(mu)
            
            losses = probe.loss_function(preds, targets)
            loss = losses['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_dataset)

        # Validate Epoch
        probe.eval()
        val_loss = 0
        val_pos_err = 0
        val_map_corr = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['obs'].to(device)
                mu, _ = vae.encode(x)
                
                targets = {
                    'pos': torch.stack([batch['x'], batch['y']], dim=1).float().to(device),
                    'map_id': batch['map_id'].to(device),
                    'in_battle': batch['in_battle'].to(device),
                    'dialog_open': batch['dialog_open'].to(device),
                    'hp': (batch['party_hp'] / (batch['party_max_hp'] + 1e-5)).to(device)
                }
                
                preds = probe(mu)
                losses = probe.loss_function(preds, targets)
                
                val_loss += losses['loss'].item() * x.size(0)
                
                # Compute specific validation metrics
                # Manhattan distance error
                pos_diff = torch.abs(preds['pos'] - targets['pos'])
                val_pos_err += torch.sum(pos_diff).item() # sum over batch and dim
                
                # Map ID accuracy
                pred_map = torch.argmax(preds['map_logits'], dim=-1)
                val_map_corr += torch.sum(pred_map == targets['map_id']).item()
                
        val_loss /= len(val_dataset)
        val_pos_mae = val_pos_err / (len(val_dataset) * 2) # Average coordinate MAE
        val_map_acc = (val_map_corr / len(val_dataset)) * 100.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pos_mae'].append(val_pos_mae)
        history['val_map_acc'].append(val_map_acc)

        print(f"Epoch {epoch:>2}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Pos MAE: {val_pos_mae:.2f} tiles | "
              f"Map Acc: {val_map_acc:.1f}%")

        # Checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.out_dir / "best_probe.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'latent_dim': latent_dim
            }, ckpt_path)
            print(f"   [New Best] Checkpoint saved: {ckpt_path.name}")

    # Save loss history to JSON file
    import json
    history_path = args.out_dir / "loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[train_probe] Loss history saved to {history_path.resolve()}")


if __name__ == "__main__":
    main()
