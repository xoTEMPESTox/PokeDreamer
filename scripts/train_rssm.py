"""
train_rssm.py — PokéWorld v2.

Trains the Recurrent State-Space Model (RSSM) with discrete categorical latents
on high-resolution (80x72x3) transitions.

Usage:
    conda activate pokemon-rl
    python scripts/train_rssm.py \
        --data-dir data \
        --epochs 20 \
        --batch-size 32 \
        --seq-len 15 \
        --lr 3e-4 \
        --out-dir checkpoints/rssm_v2
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.models import (
    Encoder, Decoder, RSSMCell,
    RewardPredictor, ContinuePredictor
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Discrete RSSM World Model")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory containing transitions_*.npz")
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of epochs to train (default: 20)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--seq-len", type=int, default=15,
                   help="Sequence length for recurrent updates (default: 15)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--kl-scale", type=float, default=1.0,
                   help="Scale factor for KL divergence loss (default: 1.0)")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="Scale factor for reward prediction loss (default: 1.0)")
    p.add_argument("--continue-scale", type=float, default=1.0,
                   help="Scale factor for continue prediction loss (default: 1.0)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/rssm_v2"),
                   help="Directory to save model checkpoints and visualizations")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Validation split ratio (default: 0.1)")
    p.add_argument("--save-interval", type=int, default=5,
                   help="Save checkpoint every N epochs (default: 5)")
    p.add_argument("--max-files", type=int, default=None,
                   help="Limit number of NPZ files loaded (default: load all)")
    return p.parse_args()


def save_grid(orig: torch.Tensor, recon: torch.Tensor, path: Path) -> None:
    """Save side-by-side comparison of original and reconstructed frames."""
    # orig and recon: (B, C, H, W) in [0, 1]
    orig_np = (orig.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    recon_np = (recon.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    
    grid = []
    num_samples = min(orig.size(0), 4)
    for i in range(num_samples):
        # Concatenate horizontally
        pair = np.hstack([orig_np[i], recon_np[i]])
        grid.append(pair)
    
    # Concatenate vertically
    full_grid = np.vstack(grid)
    cv2.imwrite(str(path), cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_rssm] Using device: {device}")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    try:
        dataset = PokemonDataset(args.data_dir, seq_len=args.seq_len, max_files=args.max_files)
    except FileNotFoundError as e:
        print(f"[ERROR] Dataset initialization failed: {e}")
        print("Wait for data collection to write transitions or check your path.")
        sys.exit(1)

    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    
    # Use fixed seed for consistent split
    train_ds, val_ds = random_split(dataset, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=True)
    
    print(f"[train_rssm] Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # ── Initialize Networks ────────────────────────────────────────────────────
    encoder = Encoder(embed_dim=512).to(device)
    decoder = Decoder(latent_dim=512 + 1024).to(device)
    rssm_cell = RSSMCell(action_dim=8, det_dim=512, class_num=32, category_num=32).to(device)
    reward_predictor = RewardPredictor(latent_dim=512 + 1024).to(device)
    continue_predictor = ContinuePredictor(latent_dim=512 + 1024).to(device)

    # Combined world model parameters
    wm_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(rssm_cell.parameters()) +
        list(reward_predictor.parameters()) +
        list(continue_predictor.parameters())
    )
    optimizer = torch.optim.Adam(wm_params, lr=args.lr)

    # Metric history log
    history = {
        "train_loss": [], "train_recon": [], "train_kl": [], "train_reward": [],
        "val_loss": [], "val_recon": [], "val_kl": [], "val_reward": []
    }

    print("[train_rssm] Starting training loop...")
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        t_start = time.perf_counter()
        
        # ── Train Epoch ───────────────────────────────────────────────────────
        encoder.train()
        decoder.train()
        rssm_cell.train()
        reward_predictor.train()
        continue_predictor.train()

        train_loss_accum = 0.0
        train_recon_accum = 0.0
        train_kl_accum = 0.0
        train_reward_accum = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            obs = batch['obs'].to(device)       # (B, T, 3, 72, 80)
            actions = batch['actions'].to(device) # (B, T)
            rewards = batch['rewards'].to(device) # (B, T)
            
            # continues = 0 at starts, else 1
            continues = torch.ones_like(rewards).to(device)

            B, T, C, H, W = obs.size()
            
            # Step 1: Pre-encode frames in batch parallel
            flat_obs = obs.reshape(B * T, C, H, W)
            flat_embed = encoder(flat_obs)
            embeds = flat_embed.reshape(B, T, -1)

            # Step 2: Recurrent rollouts
            h, s = rssm_cell.get_initial_state(B, device)
            
            h_list, s_list = [], []
            prior_logits_list, post_logits_list = [], []
            
            for t in range(T):
                action_one_hot = F.one_hot(actions[:, t], num_classes=8).float()
                step_result = rssm_cell(h, s, action_one_hot, embeds[:, t])
                h, s = step_result["h"], step_result["s"]
                
                h_list.append(h)
                s_list.append(s)
                prior_logits_list.append(step_result["prior_logits"])
                post_logits_list.append(step_result["post_logits"])

            h_stack = torch.stack(h_list, dim=1) # (B, T, 512)
            s_stack = torch.stack(s_list, dim=1) # (B, T, 1024)
            prior_logits = torch.stack(prior_logits_list, dim=1) # (B, T, 1024)
            post_logits = torch.stack(post_logits_list, dim=1)   # (B, T, 1024)

            # Step 3: Decoders & predictors
            latents = torch.cat([h_stack, s_stack], dim=-1)
            flat_latents = latents.reshape(B * T, -1)
            
            flat_recons = decoder(flat_latents)
            recons = flat_recons.reshape(B, T, C, H, W)
            
            pred_rewards = reward_predictor(flat_latents).reshape(B, T)
            pred_continues = continue_predictor(flat_latents).reshape(B, T)

            # Step 4: Loss calculation
            # Reconstruction (MSE)
            recon_loss = F.mse_loss(recons, obs, reduction='mean')

            # KL loss with KL-balancing
            prior_logits_res = prior_logits.reshape(B, T, 32, 32)
            post_logits_res = post_logits.reshape(B, T, 32, 32)
            
            post_dist = torch.distributions.Categorical(logits=post_logits_res)
            prior_dist = torch.distributions.Categorical(logits=prior_logits_res)
            
            post_dist_sg = torch.distributions.Categorical(logits=post_logits_res.detach())
            prior_dist_sg = torch.distributions.Categorical(logits=prior_logits_res.detach())
            
            kl_prior = torch.distributions.kl_divergence(post_dist_sg, prior_dist).sum(dim=-1).mean()
            kl_post = torch.distributions.kl_divergence(post_dist, prior_dist_sg).sum(dim=-1).mean()
            kl_loss = 0.8 * kl_prior + 0.2 * kl_post

            # Predictor losses
            reward_loss = F.mse_loss(pred_rewards, rewards, reduction='mean')
            continue_loss = F.binary_cross_entropy(pred_continues, continues, reduction='mean')

            # Combined loss
            total_loss = (
                recon_loss +
                args.kl_scale * kl_loss +
                args.reward_scale * reward_loss +
                args.continue_scale * continue_loss
            )

            # Step 5: Optimization
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(wm_params, 10.0)
            optimizer.step()

            # Record metrics
            train_loss_accum += total_loss.item()
            train_recon_accum += recon_loss.item()
            train_kl_accum += kl_loss.item()
            train_reward_accum += reward_loss.item()

            # Save mid-epoch checkpoint and print progress every 500 batches
            if (batch_idx + 1) % 500 == 0:
                mid_ckpt_path = args.out_dir / "checkpoint_latest.pt"
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'rssm_cell_state_dict': rssm_cell.state_dict(),
                    'reward_predictor_state_dict': reward_predictor.state_dict(),
                    'continue_predictor_state_dict': continue_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, mid_ckpt_path)
                print(f"  [Progress] Epoch {epoch:02d} | Batch {batch_idx + 1:04d}/{len(train_loader):04d} | Loss: {total_loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}) | Saved latest midpoint checkpoint")

        train_loss = train_loss_accum / len(train_loader)
        train_recon = train_recon_accum / len(train_loader)
        train_kl = train_kl_accum / len(train_loader)
        train_reward = train_reward_accum / len(train_loader)

        # ── Validation Epoch ──────────────────────────────────────────────────
        encoder.eval()
        decoder.eval()
        rssm_cell.eval()
        reward_predictor.eval()
        continue_predictor.eval()

        val_loss_accum = 0.0
        val_recon_accum = 0.0
        val_kl_accum = 0.0
        val_reward_accum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                actions = batch['actions'].to(device)
                rewards = batch['rewards'].to(device)
                continues = torch.ones_like(rewards).to(device)

                B, T, C, H, W = obs.size()
                
                flat_obs = obs.reshape(B * T, C, H, W)
                flat_embed = encoder(flat_obs)
                embeds = flat_embed.reshape(B, T, -1)

                h, s = rssm_cell.get_initial_state(B, device)
                h_list, s_list = [], []
                prior_logits_list, post_logits_list = [], []
                
                for t in range(T):
                    action_one_hot = F.one_hot(actions[:, t], num_classes=8).float()
                    # Do not use Gumbel noise during validation evaluation
                    step_result = rssm_cell(h, s, action_one_hot, embeds[:, t], use_gumbel=False)
                    h, s = step_result["h"], step_result["s"]
                    h_list.append(h)
                    s_list.append(s)
                    prior_logits_list.append(step_result["prior_logits"])
                    post_logits_list.append(step_result["post_logits"])

                h_stack = torch.stack(h_list, dim=1)
                s_stack = torch.stack(s_list, dim=1)
                prior_logits = torch.stack(prior_logits_list, dim=1)
                post_logits = torch.stack(post_logits_list, dim=1)

                latents = torch.cat([h_stack, s_stack], dim=-1)
                flat_latents = latents.reshape(B * T, -1)
                
                flat_recons = decoder(flat_latents)
                recons = flat_recons.reshape(B, T, C, H, W)
                
                pred_rewards = reward_predictor(flat_latents).reshape(B, T)
                pred_continues = continue_predictor(flat_latents).reshape(B, T)

                recon_loss = F.mse_loss(recons, obs, reduction='mean')
                
                prior_logits_res = prior_logits.reshape(B, T, 32, 32)
                post_logits_res = post_logits.reshape(B, T, 32, 32)
                kl_loss = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=post_logits_res),
                    torch.distributions.Categorical(logits=prior_logits_res)
                ).sum(dim=-1).mean()

                reward_loss = F.mse_loss(pred_rewards, rewards, reduction='mean')
                continue_loss = F.binary_cross_entropy(pred_continues, continues, reduction='mean')

                val_loss = (
                    recon_loss +
                    args.kl_scale * kl_loss +
                    args.reward_scale * reward_loss +
                    args.continue_scale * continue_loss
                )

                val_loss_accum += val_loss.item()
                val_recon_accum += recon_loss.item()
                val_kl_accum += kl_loss.item()
                val_reward_accum += reward_loss.item()

        val_loss = val_loss_accum / len(val_loader)
        val_recon = val_recon_accum / len(val_loader)
        val_kl = val_kl_accum / len(val_loader)
        val_reward = val_reward_accum / len(val_loader)

        elapsed = time.perf_counter() - t_start

        # Record metrics
        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["train_reward"].append(train_reward)
        
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["val_reward"].append(val_reward)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f}) | "
              f"Time: {elapsed:.1f}s")

        # Save metrics log
        with open(args.out_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save validation reconstruction grid periodically
        if epoch == 1 or epoch % args.save_interval == 0:
            grid_path = args.out_dir / f"reconstruction_epoch_{epoch:03d}.png"
            # Get the first frame of the first batch
            save_grid(obs[:4, 0], recons[:4, 0], grid_path)
            print(f"  Saved reconstruction grid to {grid_path}")

        # Checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        if is_best or epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = args.out_dir / ("best_world_model.pt" if is_best else f"world_model_epoch_{epoch:03d}.pt")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'rssm_cell_state_dict': rssm_cell.state_dict(),
                'reward_predictor_state_dict': reward_predictor.state_dict(),
                'continue_predictor_state_dict': continue_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path} (best: {is_best})")

    print(f"\n[SUCCESS] Completed training! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
