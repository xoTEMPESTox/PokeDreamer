"""
train_policy.py — PokéWorld v2.

Trains an Actor-Critic policy entirely in the imagined latent space of a trained
RSSM world model. Evaluates the policy zero-shot on the real PyBoy emulator.

Usage:
    conda activate pokemon-rl
    python scripts/train_policy.py \
        --wm-checkpoint checkpoints/rssm_v2/best_world_model.pt \
        --epochs 50 \
        --imag-horizon 15 \
        --lr 3e-4 \
        --out-dir checkpoints/policy_v2
"""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PokemonDataset
from src.models import (
    Encoder, Decoder, RSSMCell,
    RewardPredictor, ContinuePredictor,
    Actor, Critic
)
from pyboy import PyBoy
from src.game_state import extract_game_state, screen_capture

# Actions mapping
BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Actor-Critic Policy in RSSM Imagination")
    p.add_argument("--wm-checkpoint", type=Path, default=Path("checkpoints/rssm_v2/best_world_model.pt"),
                   help="Path to trained RSSM world model checkpoint")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to PokemonRed ROM for evaluation")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory containing transition data (for starting states)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of policy training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size of starting states (default: 64)")
    p.add_argument("--imag-horizon", type=int, default=15,
                   help="Imagination rollout horizon (default: 15)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate for Actor and Critic (default: 3e-4)")
    p.add_argument("--discount", type=float, default=0.99,
                   help="Discount factor gamma (default: 0.99)")
    p.add_argument("--lambda-gae", type=float, default=0.95,
                   help="GAE lambda for value targets (default: 0.95)")
    p.add_argument("--entropy-scale", type=float, default=1e-3,
                   help="Entropy regularization weight (default: 1e-3)")
    p.add_argument("--eval-episodes", type=int, default=3,
                   help="Number of zero-shot episodes to evaluate on emulator (default: 3)")
    p.add_argument("--eval-steps", type=int, default=100,
                   help="Max steps per evaluation episode (default: 100)")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/policy_v2"),
                   help="Output directory for policy checkpoints")
    return p.parse_args()


def compute_lambda_returns(rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor, bootstrap: torch.Tensor, discount: float, lambda_gae: float) -> torch.Tensor:
    """
    Compute lambda-returns for imagined trajectories.
    rewards: (H, B)
    values: (H, B)
    continues: (H, B)
    bootstrap: (B,)
    """
    H, B = rewards.size()
    returns = torch.zeros_like(rewards)
    
    # Start recursion from boundary condition
    last_val = bootstrap
    for t in reversed(range(H)):
        rewards_t = rewards[t]
        values_next = values[t + 1] if t + 1 < H else last_val
        continues_t = continues[t]
        
        # V_t^lambda = r_t + gamma_t * ((1 - lambda) * v_{t+1} + lambda * V_{t+1}^lambda)
        returns[t] = rewards_t + discount * continues_t * (
            (1.0 - lambda_gae) * values_next + lambda_gae * (returns[t + 1] if t + 1 < H else last_val)
        )
    return returns


def evaluate_policy(actor, encoder, rssm_cell, rom_path, num_episodes: int, max_steps: int, device: torch.device) -> dict:
    """Runs zero-shot policy inside the PyBoy emulator using RSSM belief state tracking."""
    actor.eval()
    encoder.eval()
    rssm_cell.eval()

    # Find starting save state
    project_root = Path(__file__).resolve().parents[1]
    states_folder = project_root / "saves"
    save_states = list(states_folder.glob("*.state"))
    if not save_states:
        # Fall back to external folder if exists
        save_states = list((project_root / "external/PokemonRedExperiments").glob("*.state"))
        
    if not save_states:
        print("[eval] No save state found, cannot run evaluation.")
        return {"avg_steps": 0, "avg_x_drift": 0}

    episode_rewards = []
    episode_drifts = []

    print(f"\n[eval] Running {num_episodes} zero-shot evaluation episodes on emulator...")

    for ep in range(num_episodes):
        state_path = random.choice(save_states)
        pyboy = PyBoy(str(rom_path), window="null")
        with open(state_path, "rb") as f:
            pyboy.load_state(f)

        # Track belief state
        h, s = rssm_cell.get_initial_state(1, device)
        ram_start = extract_game_state(pyboy)
        start_x, start_y = ram_start.x, ram_start.y

        total_reward = 0
        step = 0
        
        while step < max_steps:
            # 1. Screen capture & downsample
            raw_frame = screen_capture(pyboy)
            downsampled_frame = cv2.resize(raw_frame, (80, 72), interpolation=cv2.INTER_AREA).astype(np.uint8)
            obs_tensor = torch.tensor(downsampled_frame, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

            # 2. Update belief state using encoder + RSSMCell
            with torch.no_grad():
                embed = encoder(obs_tensor)
                # Sample action using actor
                latent = torch.cat([h, s], dim=-1)
                action_logits = actor(latent)
                action = torch.argmax(action_logits, dim=-1).item()

                action_one_hot = F.one_hot(torch.tensor([action], device=device), num_classes=8).float()
                step_result = rssm_cell(h, s, action_one_hot, embed, use_gumbel=False)
                h, s = step_result["h"], step_result["s"]

            # 3. Advance emulator
            btn = BUTTONS[action]
            if btn != "pass":
                pyboy.button(btn, 8)
            pyboy.tick(24, render=True)

            step += 1

        ram_end = extract_game_state(pyboy)
        end_x, end_y = ram_end.x, ram_end.y
        drift = abs(end_x - start_x) + abs(end_y - start_y)
        episode_drifts.append(drift)
        pyboy.stop()
        print(f"  Episode {ep + 1} completed | Start: ({start_x}, {start_y}) -> End: ({end_x}, {end_y}) | L1 Drift: {drift}")

    return {
        "avg_drift": float(np.mean(episode_drifts))
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_policy] Using device: {device}")

    # ── Load World Model ───────────────────────────────────────────────────────
    if not args.wm_checkpoint.exists():
        print(f"[ERROR] World model checkpoint not found: {args.wm_checkpoint}")
        sys.exit(1)

    print(f"[train_policy] Loading world model from {args.wm_checkpoint}...")
    wm_data = torch.load(args.wm_checkpoint, map_location=device)

    encoder = Encoder(embed_dim=512).to(device)
    decoder = Decoder(latent_dim=512 + 1024).to(device)
    rssm_cell = RSSMCell(action_dim=8, det_dim=512, class_num=32, category_num=32).to(device)
    reward_predictor = RewardPredictor(latent_dim=512 + 1024).to(device)
    continue_predictor = ContinuePredictor(latent_dim=512 + 1024).to(device)

    encoder.load_state_dict(wm_data['encoder_state_dict'])
    decoder.load_state_dict(wm_data['decoder_state_dict'])
    rssm_cell.load_state_dict(wm_data['rssm_cell_state_dict'])
    reward_predictor.load_state_dict(wm_data['reward_predictor_state_dict'])
    continue_predictor.load_state_dict(wm_data['continue_predictor_state_dict'])

    # Freeze world model parameters
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    for p in rssm_cell.parameters(): p.requires_grad = False
    for p in reward_predictor.parameters(): p.requires_grad = False
    for p in continue_predictor.parameters(): p.requires_grad = False

    encoder.eval()
    decoder.eval()
    rssm_cell.eval()
    reward_predictor.eval()
    continue_predictor.eval()

    # ── Initialize Actor-Critic Networks ───────────────────────────────────────
    actor = Actor(latent_dim=512 + 1024, action_dim=8).to(device)
    critic = Critic(latent_dim=512 + 1024).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # Load dataset to extract real starting states
    dataset = PokemonDataset(args.data_dir, seq_len=1)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("[train_policy] Starting training in imagination...")
    history = {"actor_loss": [], "critic_loss": [], "entropy": [], "avg_val": [], "eval_drift": []}

    for epoch in range(1, args.epochs + 1):
        t_start = time.perf_counter()
        
        actor.train()
        critic.train()

        actor_loss_accum = 0.0
        critic_loss_accum = 0.0
        entropy_accum = 0.0
        val_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            obs = batch['obs'].to(device) # (B, 3, 72, 80)
            B = obs.size(0)

            # Step 1: Infer initial latent states using encoder
            with torch.no_grad():
                embed = encoder(obs)
                h0, s0 = rssm_cell.get_initial_state(B, device)
                # Feed a dummy action at start (say, pass/action=5)
                dummy_action = F.one_hot(torch.full((B,), 7, device=device), num_classes=8).float()
                step0 = rssm_cell(h0, s0, dummy_action, embed, use_gumbel=False)
                h, s = step0["h"], step0["s"]

            # Step 2: Roll out in imagination
            imag_h_list = [h]
            imag_s_list = [s]
            imag_action_list = []
            imag_logits_list = []

            # We roll out for H steps in imagination
            for t in range(args.imag_horizon):
                latent = torch.cat([h, s], dim=-1)
                
                # Actor selects action
                action_logits = actor(latent)
                # Sample action using differentiable Gumbel-Softmax
                action_sample = F.gumbel_softmax(action_logits, tau=1.0, hard=True)
                
                # Roll out the world model dynamics
                with torch.no_grad():
                    # Prior transition p(s_{t+1} | h_{t+1})
                    step_result = rssm_cell(h, s, action_sample, embed=None, use_gumbel=True)
                    h, s = step_result["h"], step_result["s"]

                imag_h_list.append(h)
                imag_s_list.append(s)
                imag_action_list.append(action_sample)
                imag_logits_list.append(action_logits)

            # Stack imagined trajectory variables
            # h: (H+1, B, det_dim), s: (H+1, B, stoch_dim)
            imag_h = torch.stack(imag_h_list, dim=0)
            imag_s = torch.stack(imag_s_list, dim=0)
            imag_actions = torch.stack(imag_action_list, dim=0) # (H, B, action_dim)
            imag_logits = torch.stack(imag_logits_list, dim=0)   # (H, B, action_dim)

            # Step 3: Predict rewards, continues, and values
            imag_latents = torch.cat([imag_h, imag_s], dim=-1) # (H+1, B, lat_dim)
            
            # Predict values for t = 0 ... H
            flat_imag_latents = imag_latents.reshape(-1, imag_latents.size(-1))
            flat_pred_values = critic(flat_imag_latents)
            pred_values = flat_pred_values.reshape(args.imag_horizon + 1, B)

            # Predict rewards and continues for t = 0 ... H-1
            flat_imag_latents_trunc = imag_latents[:-1].reshape(-1, imag_latents.size(-1))
            flat_pred_rewards = reward_predictor(flat_imag_latents_trunc)
            pred_rewards = flat_pred_rewards.reshape(args.imag_horizon, B)
            
            flat_pred_continues = continue_predictor(flat_imag_latents_trunc)
            pred_continues = flat_pred_continues.reshape(args.imag_horizon, B)

            # Step 4: Compute lambda-returns as targets for the Critic
            bootstrap = pred_values[-1] # shape (B,)
            targets = compute_lambda_returns(
                pred_rewards.detach(),
                pred_values.detach(),
                pred_continues.detach(),
                bootstrap.detach(),
                args.discount,
                args.lambda_gae
            ) # shape (H, B)

            # Step 5: Loss Calculation
            # Critic loss (fits the expected value to target returns)
            # targets is (H, B), we fit critic predictions for t = 0 ... H-1
            critic_loss = 0.5 * F.mse_loss(pred_values[:-1], targets, reduction='mean')

            # Actor loss (maximizes lambda-returns + entropy regularization)
            # We want to maximize the return, so actor_loss = -returns
            # We compute policy log probabilities and scale by returns for PG gradient or use reparameterized returns
            # Since actions are hard one-hots from gumbel_softmax, targets contains the return of the trajectory.
            # Differentiable returns flow through action_sample to actor_logits:
            actor_loss_raw = -targets
            
            # Policy entropy
            probs = F.softmax(imag_logits, dim=-1)
            log_probs = F.log_softmax(imag_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            
            actor_loss = actor_loss_raw.mean() - args.entropy_scale * entropy

            # Step 6: Backpropagate
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_opt.step()

            # Record
            actor_loss_accum += actor_loss.item()
            critic_loss_accum += critic_loss.item()
            entropy_accum += entropy.item()
            val_accum += pred_values.mean().item()

        avg_actor_loss = actor_loss_accum / len(train_loader)
        avg_critic_loss = critic_loss_accum / len(train_loader)
        avg_entropy = entropy_accum / len(train_loader)
        avg_val = val_accum / len(train_loader)
        elapsed = time.perf_counter() - t_start

        # Record metrics
        history["actor_loss"].append(avg_actor_loss)
        history["critic_loss"].append(avg_critic_loss)
        history["entropy"].append(avg_entropy)
        history["avg_val"].append(avg_val)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | "
              f"Entropy: {avg_entropy:.4f} | Avg Value: {avg_val:.4f} | Time: {elapsed:.1f}s")

        # Periodically run emulator evaluation
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            eval_metrics = evaluate_policy(
                actor, encoder, rssm_cell,
                args.rom, args.eval_episodes, args.eval_steps, device
            )
            history["eval_drift"].append(eval_metrics["avg_drift"])
            
            # Save checkpoints
            ckpt_path = args.out_dir / f"policy_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer': actor_opt.state_dict(),
                'critic_optimizer': critic_opt.state_dict(),
                'history': history
            }, ckpt_path)
            print(f"  Saved policy checkpoint to {ckpt_path} | Eval Avg L1 Drift: {eval_metrics['avg_drift']:.2f}")

        # Dump history log
        with open(args.out_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n[SUCCESS] Completed Actor-Critic Policy training in imagination!")


if __name__ == "__main__":
    main()
