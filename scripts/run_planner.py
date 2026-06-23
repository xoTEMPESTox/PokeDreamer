"""
run_planner.py — Day 12-13 deliverable script.

Demonstrates model-predictive control (MPC) lookahead planning.
Imagines future trajectories using the latent world model, probes coordinates
using the RAMProbe, and executes the best sequence of actions on PyBoy.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/run_planner.py \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --dynamics-checkpoint checkpoints/dynamics/best_dynamics.pt \
        --probe-checkpoint checkpoints/probe/best_probe.pt \
        --target-x 5 --target-y 5
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.game_state import extract_game_state, screen_capture
from src.vae import VAE
from src.dynamics import LatentDynamics
from src.probe import RAMProbe
from src.planner import LatentPlanner

# Actions: 0=DOWN, 1=LEFT, 2=RIGHT, 3=UP, 4=A, 5=B
BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MPC Latent Planner on Pokémon Red")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to Pokémon Red ROM")
    p.add_argument("--vae-checkpoint", type=Path, default=Path("checkpoints/vae/best_vae.pt"),
                   help="Path to trained VAE checkpoint")
    p.add_argument("--dynamics-checkpoint", type=Path, default=Path("checkpoints/dynamics/best_dynamics.pt"),
                   help="Path to trained LatentDynamics checkpoint")
    p.add_argument("--probe-checkpoint", type=Path, default=Path("checkpoints/probe/best_probe.pt"),
                   help="Path to trained RAMProbe checkpoint")
    p.add_argument("--save-state", type=Path, default=Path("saves/intro_done.state"),
                   help="PyBoy save state to load")
    p.add_argument("--target-x", type=int, required=True,
                   help="Target tile coordinate X")
    p.add_argument("--target-y", type=int, required=True,
                   help="Target tile coordinate Y")
    p.add_argument("--target-map", type=int, default=None,
                   help="Target map ID (optional)")
    p.add_argument("--lookahead-steps", type=int, default=15,
                   help="Planning search horizon (default: 15)")
    p.add_argument("--replan-interval", type=int, default=5,
                   help="Steps to execute before replanning (default: 5)")
    p.add_argument("--max-steps", type=int, default=150,
                   help="Max actions to execute in total (default: 150)")
    p.add_argument("--ticks-per-action", type=int, default=24,
                   help="Emulator ticks per action (default: 24)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── ROM check ─────────────────────────────────────────────────────────────
    if not args.rom.exists():
        print(f"[ERROR] ROM not found: {args.rom.resolve()}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_planner] Using device: {device}")

    # ── Load World Model Checkpoints ──────────────────────────────────────────
    print(f"[run_planner] Loading models...")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = VAE(latent_dim=vae_ckpt['latent_dim']).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device)
    dynamics = LatentDynamics(
        latent_dim=dyn_ckpt['latent_dim'],
        num_actions=8,
        action_dim=dyn_ckpt['action_dim'],
        hidden_dim=dyn_ckpt['hidden_dim']
    ).to(device)
    dynamics.load_state_dict(dyn_ckpt['model_state_dict'])
    dynamics.eval()
    
    probe_ckpt = torch.load(args.probe_checkpoint, map_location=device)
    probe = RAMProbe(latent_dim=vae_ckpt['latent_dim']).to(device)
    probe.load_state_dict(probe_ckpt['model_state_dict'])
    probe.eval()

    print("[run_planner] All models loaded successfully!")

    # ── Initialize Planner ────────────────────────────────────────────────────
    planner = LatentPlanner(dynamics, probe, device)

    # ── Start Emulator ────────────────────────────────────────────────────────
    pyboy = PyBoy(str(args.rom), window="null")
    if args.save_state.exists():
        with open(args.save_state, "rb") as f:
            pyboy.load_state(f)
        print(f"[run_planner] Loaded save state: {args.save_state.name}")
    else:
        print("[WARN] Save state not found. Starting from cold start.")

    state = extract_game_state(pyboy)
    print(f"[run_planner] Starting state: {state}")
    print(f"[run_planner] Target position: ({args.target_x}, {args.target_y})" + 
          (f" @ map {args.target_map}" if args.target_map is not None else ""))

    total_steps = 0
    t_start = time.perf_counter()

    while total_steps < args.max_steps:
        # Check arrival
        state = extract_game_state(pyboy)
        if args.target_map is not None and state.map_id != args.target_map:
            pass
        else:
            dist = abs(state.x - args.target_x) + abs(state.y - args.target_y)
            if dist <= 1:
                print(f"\n[SUCCESS] Reached target in {total_steps} actions! Final state: {state}")
                pyboy.stop()
                sys.exit(0)

        # ── 1. Perception: Encode current screen ──────────────────────────────
        raw_frame = screen_capture(pyboy)
        downsampled = vae.encode(
            torch.tensor(
                np.transpose(
                    vae_ckpt['model_state_dict'] is not None and policy_downsample(raw_frame) if False else 
                    cv2.resize(raw_frame, (40, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0,
                    (2, 0, 1)
                ), dtype=torch.float32
            ).unsqueeze(0).to(device)
        )
        
        # Extract z_t (mu)
        z_t = downsampled[0] 

        # ── 2. MPC Planning: Choose best action sequence ──────────────────────
        best_name, best_seq, best_score = planner.plan(
            z_t, 
            goal_x=args.target_x, 
            goal_y=args.target_y, 
            goal_map_id=args.target_map,
            seq_len=args.lookahead_steps
        )

        print(f"\n[Plan step {total_steps}] Chosen Macro-action: {best_name} (Imagined Score: {best_score:.2f})")
        print(f"  Actions to execute: {[BUTTONS[a] for a in best_seq[:args.replan_interval]]}")

        # ── 3. Execution: Apply action sequence for interval steps ────────────
        for idx in range(min(args.replan_interval, len(best_seq))):
            action = best_seq[idx]
            btn = BUTTONS[action]
            
            if btn != "pass":
                pyboy.button(btn, 8)
            pyboy.tick(args.ticks_per_action, render=True)
            
            total_steps += 1
            
            # Print current state
            current_state = extract_game_state(pyboy)
            print(f"    Step {total_steps:>3}: Executed {btn:<6} | Pos: ({current_state.x},{current_state.y}) @ map {current_state.map_id} | Battle: {current_state.in_battle}")

            # Early arrival check during macro-action execution
            if args.target_map is not None and current_state.map_id != args.target_map:
                continue
            
            curr_dist = abs(current_state.x - args.target_x) + abs(current_state.y - args.target_y)
            if curr_dist <= 1:
                print(f"\n[SUCCESS] Reached target in {total_steps} actions! Final state: {current_state}")
                pyboy.stop()
                sys.exit(0)

    print(f"\n[TIMEOUT] Failed to reach target in {args.max_steps} steps. Ending run.")
    pyboy.stop()


def policy_downsample(screen_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(screen_rgb, (40, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


if __name__ == "__main__":
    main()
