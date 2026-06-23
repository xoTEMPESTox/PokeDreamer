"""
generate_planner_demo.py — Day 14-15 deliverable script.

Demonstrates model-predictive control (MPC) planning over a long horizon.
Runs the planner for 150 steps (navigating up Route 1 toward Viridian City).

At each action step:
- Left View: Real emulator execution showing long-term progress.
- Right View: The planner's short-term imagined lookahead (15 steps) decoded
  back to pixels, showing what the model predicted would happen for its chosen macro-action.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/generate_planner_demo.py \
        --rom "Pokemon - Red Version (USA, Europe).gb" \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --dynamics-checkpoint checkpoints/dynamics/best_dynamics.pt \
        --probe-checkpoint checkpoints/probe/best_probe.pt \
        --save-state saves/intro_done.state \
        --out-video checkpoints/planner_navigation_demo.mp4
"""

import argparse
import sys
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

BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate long MPC planner demo video")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to Pokémon Red ROM")
    p.add_argument("--vae-checkpoint", type=Path, default=Path("checkpoints/vae/best_vae.pt"),
                   help="Path to trained VAE checkpoint")
    p.add_argument("--dynamics-checkpoint", type=Path, default=Path("checkpoints/dynamics/best_dynamics.pt"),
                   help="Path to trained LatentDynamics checkpoint")
    p.add_argument("--probe-checkpoint", type=Path, default=Path("checkpoints/probe/best_probe.pt"),
                   help="Path to trained RAMProbe checkpoint")
    p.add_argument("--save-state", type=Path, default=Path("saves/intro_done.state"),
                   help="Save state to load")
    p.add_argument("--out-video", type=Path, default=Path("checkpoints/planner_navigation_demo.mp4"),
                   help="Output MP4 file path")
    p.add_argument("--max-steps", type=int, default=150,
                   help="Total number of steps to run the demo (default: 150)")
    p.add_argument("--ticks-per-action", type=int, default=24,
                   help="Ticks per action")
    p.add_argument("--replan-interval", type=int, default=5,
                   help="MPC replan interval")
    p.add_argument("--lookahead-steps", type=int, default=15,
                   help="Lookahead steps in imagination")
    p.add_argument("--fps", type=int, default=5,
                   help="FPS of output video")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.rom.exists():
        print(f"[ERROR] ROM not found at {args.rom.resolve()}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate_planner_demo] Using device: {device}")

    # ── 1. Load Models ────────────────────────────────────────────────────────
    print("Loading models...")
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

    print("Models loaded successfully.")

    # ── 2. Initialize Planner ──────────────────────────────────────────────────
    planner = LatentPlanner(dynamics, probe, device)

    # ── 3. Start Emulator ──────────────────────────────────────────────────────
    pyboy = PyBoy(str(args.rom), window="null")
    if args.save_state.exists():
        with open(args.save_state, "rb") as f:
            pyboy.load_state(f)
        print(f"Loaded save state: {args.save_state.name}")
    else:
        print("[WARN] Save state not found. Starting from cold boot.")

    # Set navigation target: Route 1 exit (which is North)
    # Pallet town start is (5,6) on map 0. Moving up (y decrease) leads to Route 1.
    target_x = 5
    target_y = -30 # Target deep in Route 1

    # ── 4. Set up Video Writer ────────────────────────────────────────────────
    scale = 6
    view_h = 36 * scale
    view_w = 40 * scale
    border_w = 4
    
    frame_h = view_h + 40
    frame_w = view_w * 2 + border_w
    
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (frame_w, frame_h))
    print(f"Saving planner demo video to: {args.out_video.resolve()}")

    total_steps = 0
    
    # Store the currently selected imagined sequence and its name
    current_best_name = "None"
    current_best_seq = []
    current_best_score = 0.0
    imagined_z_sequence = None
    
    while total_steps < args.max_steps:
        # Get current state from WRAM
        game_state = extract_game_state(pyboy)
        
        # Check if we should replan (or start planning)
        if total_steps % args.replan_interval == 0:
            # Get current frame for VAE encoding (perception grounding)
            raw_frame = screen_capture(pyboy)
            obs = cv2.resize(raw_frame, (40, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                mu_t, _ = vae.encode(obs_tensor)
            
            # Replan
            current_best_name, current_best_seq, current_best_score = planner.plan(
                mu_t,
                goal_x=target_x,
                goal_y=target_y,
                seq_len=args.lookahead_steps
            )
            
            # Generate the imagined z sequence for the chosen best macro-action
            # to display side-by-side
            with torch.no_grad():
                # rollout shape: (1, seq_len, latent_dim)
                imagined_z_sequence = dynamics.rollout(
                    mu_t, 
                    torch.tensor(current_best_seq, dtype=torch.long, device=device).unsqueeze(0), 
                    device=device
                )

        # ── Execute current action in the macro-action block ──
        inner_idx = total_steps % args.replan_interval
        action_val = current_best_seq[inner_idx]
        btn = BUTTONS[action_val]
        
        # Apply step in real emulator
        if btn != "pass":
            pyboy.button(btn, 8)
        pyboy.tick(args.ticks_per_action, render=True)
        
        real_frame = screen_capture(pyboy)
        
        # Decode the corresponding imagined frame from our BPTT lookahead sequence
        # We display the lookahead step corresponding to the current execution index
        with torch.no_grad():
            z_t_imagined = imagined_z_sequence[:, inner_idx] # shape (1, latent_dim)
            pred_obs_tensor = vae.decode(z_t_imagined)
            
        pred_obs = pred_obs_tensor.squeeze(0).cpu().numpy()
        pred_obs = np.transpose(pred_obs, (1, 2, 0))
        
        # Contrast stretch to push soft grey values to sharp black/white (retro pixel art style)
        pred_obs = (pred_obs - pred_obs.min()) / (pred_obs.max() - pred_obs.min() + 1e-5)
        pred_obs = 1.0 / (1.0 + np.exp(-12.0 * (pred_obs - 0.5)))
        pred_obs = (pred_obs * 255.0).clip(0, 255).astype(np.uint8)
        
        # Stacking panels
        # Left panel: native high-resolution frame from the emulator
        left_panel = cv2.resize(real_frame, (view_w, view_h), interpolation=cv2.INTER_CUBIC)
        # Right panel: upscaled imagined frame with retro nearest-neighbor scaling
        right_panel = cv2.resize(pred_obs, (view_w, view_h), interpolation=cv2.INTER_NEAREST)
        
        left_panel_bgr = cv2.cvtColor(left_panel, cv2.COLOR_RGB2BGR)
        right_panel_bgr = cv2.cvtColor(right_panel, cv2.COLOR_RGB2BGR)
        
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        canvas[40:40+view_h, 0:view_w] = left_panel_bgr
        canvas[40:40+view_h, view_w:view_w+border_w] = [128, 128, 128]
        canvas[40:40+view_h, view_w+border_w:view_w+border_w+view_w] = right_panel_bgr
        
        # Text annotations
        cv2.putText(canvas, "REAL EMULATOR (MPC GROUNDED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"IMAGINED LOOKAHEAD ({current_best_name})", (view_w + border_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
        
        info_text = f"Step: {total_steps+1:03d} | WRAM Pos: ({game_state.x},{game_state.y}) | Action: {btn.upper()}"
        cv2.putText(canvas, info_text, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        
        video_writer.write(canvas)
        total_steps += 1
        print(f"  Step {total_steps:>3}/{args.max_steps} | Real Pos: ({game_state.x},{game_state.y}) | Action: {btn:<5} | Score: {current_best_score:.1f}")

    pyboy.stop()
    video_writer.release()
    print(f"\n[generate_planner_demo] Planner demo video completed successfully!")
    print(f"Saved to: {args.out_video.resolve()}")


if __name__ == "__main__":
    main()
