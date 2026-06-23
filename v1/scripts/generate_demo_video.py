"""
generate_demo_video.py — Day 14-15 deliverable script.

Generates a side-by-side comparison video of:
[ Real Emulator Trajectory (Left) | Purely Imagined Latent Rollout (Right) ]

It starts from the same initial frame (encoded by VAE to z_0) and applies the same
sequence of actions. The right side is generated entirely by rolling out the GRU
dynamics model autoregressively (z_t+1 = Dynamics(z_t, a_t)) and decoding the latents back to pixels,
WITHOUT stepping the emulator for the imagined trajectory.

Usage:
    C:\\Users\\priya\\miniconda3\\envs\\pokemon-rl\\python.exe scripts/generate_demo_video.py \
        --rom "Pokemon - Red Version (USA, Europe).gb" \
        --vae-checkpoint checkpoints/vae/best_vae.pt \
        --dynamics-checkpoint checkpoints/dynamics/best_dynamics.pt \
        --save-state saves/intro_done.state \
        --out-video checkpoints/side_by_side_demo.mp4
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
from src.game_state import screen_capture
from src.vae import VAE
from src.dynamics import LatentDynamics

BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]

# Define a pre-programmed loop of 50 actions to demonstrate walking around Pallet Town
DEMO_ACTIONS = [
    0, 0, 0, 0, # Walk Down
    2, 2, 2, 2, # Walk Right
    3, 3, 3, 3, # Walk Up
    1, 1, 1, 1, # Walk Left
    0, 0, 0, 0, # Walk Down
    2, 2, 2, 2, # Walk Right
    3, 3, 3, 3, # Walk Up
    1, 1, 1, 1, # Walk Left
    0, 2, 0, 2, 0, 2, 0, 2, # Alternating Down/Right
    3, 1, 3, 1, 3, 1, 3, 1  # Alternating Up/Left
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate side-by-side imagined vs. real video")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to Pokémon Red ROM")
    p.add_argument("--vae-checkpoint", type=Path, default=Path("checkpoints/vae/best_vae.pt"),
                   help="Path to trained VAE checkpoint")
    p.add_argument("--dynamics-checkpoint", type=Path, default=Path("checkpoints/dynamics/best_dynamics.pt"),
                   help="Path to trained LatentDynamics checkpoint")
    p.add_argument("--save-state", type=Path, default=Path("saves/intro_done.state"),
                   help="Save state to load")
    p.add_argument("--out-video", type=Path, default=Path("checkpoints/side_by_side_demo.mp4"),
                   help="Output MP4 file path")
    p.add_argument("--ticks-per-action", type=int, default=24,
                   help="Ticks per emulator action")
    p.add_argument("--fps", type=int, default=5,
                   help="Video frame rate (FPS)")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Maximum actions/steps to run the demo (default: 30)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.rom.exists():
        print(f"[ERROR] ROM not found at {args.rom.resolve()}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate_demo_video] Using device: {device}")

    # ── 1. Load Models ────────────────────────────────────────────────────────
    print("Loading checkpoints...")
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

    print("Models loaded successfully.")

    # ── 2. Start Emulator ──────────────────────────────────────────────────────
    pyboy = PyBoy(str(args.rom), window="null")
    if args.save_state.exists():
        with open(args.save_state, "rb") as f:
            pyboy.load_state(f)
        print(f"Loaded save state: {args.save_state.name}")
    else:
        print("[WARN] Save state not found. Starting from cold boot.")

    # Capture start frame
    raw_start_frame = screen_capture(pyboy)
    
    # ── 3. Initialize Imagined Rollout State ───────────────────────────────
    # Downsample and normalize initial screen to match model format
    start_obs = cv2.resize(raw_start_frame, (40, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    start_obs_tensor = torch.tensor(np.transpose(start_obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu_0, _ = vae.encode(start_obs_tensor)
        z_start = mu_0 # Starting latent z_0 (1, latent_dim)

    # ── 4. Set up Video Writer ────────────────────────────────────────────────
    # We will output frames of shape (scale*36, scale*40*2 + border_width, 3)
    scale = 6 # scale up to 240x216 per view
    view_h = 36 * scale
    view_w = 40 * scale
    border_w = 4
    
    frame_h = view_h + 40 # extra space for labels/action text
    frame_w = view_w * 2 + border_w
    
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (frame_w, frame_h))
    print(f"Saving video to: {args.out_video.resolve()}")

    steps_executed = 0
    
    # Store history of actions applied
    action_history = []
    
    # Execute the demo sequence
    max_steps = min(args.max_steps, len(DEMO_ACTIONS))
    for step_idx in range(max_steps):
        action_val = DEMO_ACTIONS[step_idx]
        action_history.append(action_val)
        btn = BUTTONS[action_val]
        
        # ────────── A. REAL EMULATOR STEP ──────────
        if btn != "pass":
            pyboy.button(btn, 8)
        pyboy.tick(args.ticks_per_action, render=True)
        
        real_frame = screen_capture(pyboy)
        
        # ────────── B. IMAGINED DYNAMICS STEP ──────────
        # Perform rollout purely in latent space starting from z_start using all actions so far
        actions_tensor = torch.tensor(action_history, dtype=torch.long, device=device).unsqueeze(0) # (1, step_idx+1)
        with torch.no_grad():
            pred_z_seq = dynamics.rollout(z_start, actions_tensor, device=device)
            z_t = pred_z_seq[:, -1] # get final latent z_t
            pred_obs_tensor = vae.decode(z_t) # decode latent to (1, 3, 36, 40)
            
        pred_obs = pred_obs_tensor.squeeze(0).cpu().numpy() # (3, 36, 40)
        pred_obs = np.transpose(pred_obs, (1, 2, 0)) # (36, 40, 3)
        
        # Contrast stretch to push soft grey values to sharp black/white (retro pixel art style)
        pred_obs = (pred_obs - pred_obs.min()) / (pred_obs.max() - pred_obs.min() + 1e-5)
        pred_obs = 1.0 / (1.0 + np.exp(-12.0 * (pred_obs - 0.5)))
        pred_obs = (pred_obs * 255.0).clip(0, 255).astype(np.uint8)
        
        # ────────── C. CREATE COMBINED VISUAL FRAME ──────────
        # Left panel: native high-resolution frame from the emulator
        left_panel = cv2.resize(real_frame, (view_w, view_h), interpolation=cv2.INTER_CUBIC)
        # Right panel: upscaled imagined frame with retro nearest-neighbor scaling
        right_panel = cv2.resize(pred_obs, (view_w, view_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert RGB to BGR for OpenCV VideoWriter
        left_panel_bgr = cv2.cvtColor(left_panel, cv2.COLOR_RGB2BGR)
        right_panel_bgr = cv2.cvtColor(right_panel, cv2.COLOR_RGB2BGR)
        
        # Create full output canvas (black background)
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # Insert left view
        canvas[40:40+view_h, 0:view_w] = left_panel_bgr
        
        # Insert border
        canvas[40:40+view_h, view_w:view_w+border_w] = [128, 128, 128] # Grey vertical divider
        
        # Insert right view
        canvas[40:40+view_h, view_w+border_w:view_w+border_w+view_w] = right_panel_bgr
        
        # Draw Labels & Text
        # Header text
        cv2.putText(canvas, "REAL EMULATOR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "IMAGINED FUTURE (WORLD MODEL)", (view_w + border_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        
        # Action/step text at footer
        info_text = f"Step: {step_idx+1:02d} | Action: {btn.upper()}"
        cv2.putText(canvas, info_text, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Write frame to video
        video_writer.write(canvas)
        steps_executed += 1
        print(f"  Processed step {steps_executed:>2}/{len(DEMO_ACTIONS)} | Action: {btn}")

    pyboy.stop()
    video_writer.release()
    print(f"\n[generate_demo_video] Demo video completed successfully!")
    print(f"Saved to: {args.out_video.resolve()}")


if __name__ == "__main__":
    main()
