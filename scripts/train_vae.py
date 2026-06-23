"""
train_vae.py — Day 4-5 deliverable script.

Trains the VAE (Encoder/Decoder) on downsampled Pokémon Red observations.
Saves checkpoints and generates reconstruction validation plots.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/train_vae.py \
        --data-dir data \
        --epochs 15 \
        --batch-size 128 \
        --latent-dim 32 \
        --beta 1.0 \
        --out-dir checkpoints/vae
"""

import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.vae import VAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAE on Pokémon Red frames")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory with .npz chunk files")
    p.add_argument("--epochs", type=int, default=15,
                   help="Number of epochs to train (default: 15)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size (default: 128)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--latent-dim", type=int, default=32,
                   help="Dimension of latent space z (default: 32)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="KL loss weighting coefficient (default: 1.0)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/vae"),
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


def save_reconstruction_plot(model: VAE, val_loader: DataLoader, device: torch.device, epoch: int, out_path: Path) -> None:
    """Saves a plot comparing original frames with VAE reconstructions using Pillow."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        x = batch['obs'].to(device)
        x_recon, _, _ = model(x)
        
        n_samples = min(x.size(0), 8)
        x = x[:n_samples].cpu().numpy()
        x_recon = x_recon[:n_samples].cpu().numpy()
        
        # Transpose to (B, H, W, C)
        x = np.transpose(x, (0, 2, 3, 1))
        x_recon = np.transpose(x_recon, (0, 2, 3, 1))
        
        # Scale to [0, 255] and cast to uint8
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
        x_recon = (x_recon * 255.0).clip(0, 255).astype(np.uint8)
        
        # Construct grid: 2 rows, n_samples columns
        grid_h = 36 * 2
        grid_w = 40 * n_samples
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i in range(n_samples):
            grid[0:36, i*40:(i+1)*40] = x[i]
            grid[36:72, i*40:(i+1)*40] = x_recon[i]
            
        img = Image.fromarray(grid)
        img.save(out_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_vae] Using device: {device}")

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

    print(f"[train_vae] Train samples: {train_size} | Val samples: {val_size}")

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_kl': [], 'val_kl': []}

    print("\nStarting VAE training...")
    for epoch in range(1, args.epochs + 1):
        # Train Epoch
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for batch in train_loader:
            x = batch['obs'].to(device) # (B, 3, 36, 40)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            
            losses = model.loss_function(x_recon, x, mu, logvar, beta=args.beta)
            loss = losses['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += losses['loss'].item()
            train_recon += losses['recon_loss'].item()
            train_kl += losses['kl_loss'].item()
            
        train_loss /= len(train_dataset)
        train_recon /= len(train_dataset)
        train_kl /= len(train_dataset)

        # Validate Epoch
        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['obs'].to(device)
                x_recon, mu, logvar = model(x)
                losses = model.loss_function(x_recon, x, mu, logvar, beta=args.beta)
                
                val_loss += losses['loss'].item()
                val_recon += losses['recon_loss'].item()
                val_kl += losses['kl_loss'].item()
                
        val_loss /= len(val_dataset)
        val_recon /= len(val_dataset)
        val_kl /= len(val_dataset)

        # Update logs
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        history['train_kl'].append(train_kl)
        history['val_kl'].append(val_kl)

        print(f"Epoch {epoch:>2}/{args.epochs} | "
              f"Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | "
              f"Val Loss: {val_loss:.2f} (Recon: {val_recon:.2f}, KL: {val_kl:.2f})")

        # Save reconstructions periodically
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            plot_path = args.out_dir / f"reconstruction_epoch_{epoch}.png"
            save_reconstruction_plot(model, val_loader, device, epoch, plot_path)
            print(f"   Reconstruction sample saved to {plot_path.name}")

        # Checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.out_dir / "best_vae.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'latent_dim': args.latent_dim
            }, ckpt_path)
            print(f"   [New Best] Checkpoint saved: {ckpt_path.name}")

    # Save loss history to JSON file
    import json
    history_path = args.out_dir / "loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[train_vae] Loss history saved to {history_path.resolve()}")


if __name__ == "__main__":
    main()
