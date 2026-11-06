"""
generate_demo_video_v2.py — Day 14-15 deliverable script adapted for PokéWorld v2.

Generates a side-by-side comparison video of:
[ Real Emulator Trajectory (Left) | Purely Imagined Discrete Latent RSSM Rollout (Right) ]

Starts from the same initial frame (encoded by RSSM Encoder to s_0, h_0) and applies the same
sequence of actions. The right side is generated entirely by rolling out the prior dynamics model
autoregressively (prior transition p(s_t | h_t)) and decoding the latents back to pixels,
WITHOUT stepping the emulator for the imagined trajectory.

Usage:
    conda activate pokemon-rl
    python scripts/generate_demo_video_v2.py \
        --rom "Pokemon - Red Version (USA, Europe).gb" \
        --checkpoint checkpoints/rssm_v2/best_world_model.pt \
        --save-state saves/intro_done.state \
        --out-video checkpoints/rssm_v2/side_by_side_demo_v2.mp4
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.game_state import screen_capture
from src.models import Encoder, Decoder, RSSMCell

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
    p = argparse.ArgumentParser(description="Generate side-by-side imagined vs. real video for RSSM v2")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to Pokémon Red ROM")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rssm_v2/best_world_model.pt"),
                   help="Path to trained RSSM world model checkpoint")
    p.add_argument("--save-state", type=Path, default=Path("saves/intro_done.state"),
                   help="Save state to load")
    p.add_argument("--out-video", type=Path, default=Path("checkpoints/rssm_v2/side_by_side_demo_v2.mp4"),
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
    print(f"[generate_demo_video_v2] Using device: {device}")

    # ── 1. Load Models ────────────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    encoder = Encoder(embed_dim=512).to(device)
    decoder = Decoder(latent_dim=512 + 1024).to(device)
    rssm_cell = RSSMCell(action_dim=8, det_dim=512, class_num=32, category_num=32).to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    rssm_cell.load_state_dict(checkpoint['rssm_cell_state_dict'])
    
    encoder.eval()
    decoder.eval()
    rssm_cell.eval()
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
    start_obs_tensor = torch.tensor(raw_start_frame, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        embed_0 = encoder(start_obs_tensor)
        h, s = rssm_cell.get_initial_state(1, device)
        # Dummy first action is a pass (action index 7)
        dummy_action = F.one_hot(torch.tensor([7], device=device), num_classes=8).float()
        step0 = rssm_cell(h, s, dummy_action, embed_0, use_gumbel=False)
        h, s = step0["h"], step0["s"]

    # ── 4. Set up Video Writer ────────────────────────────────────────────────
    scale = 2 # Upscale views slightly for visibility
    view_h = 144 * scale
    view_w = 160 * scale
    border_w = 4
    
    frame_h = view_h + 40 # extra space for labels/action text
    frame_w = view_w * 2 + border_w
    
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (frame_w, frame_h))
    print(f"Saving video to: {args.out_video.resolve()}")

    steps_executed = 0
    max_steps = min(args.max_steps, len(DEMO_ACTIONS))
    
    for step_idx in range(max_steps):
        action_val = DEMO_ACTIONS[step_idx]
        btn = BUTTONS[action_val]
        
        # ────────── A. REAL EMULATOR STEP ──────────
        if btn != "pass":
            pyboy.button(btn, 8)
        pyboy.tick(args.ticks_per_action, render=True)
        
        real_frame = screen_capture(pyboy)
        
        # ────────── B. IMAGINED DYNAMICS STEP (Prior rollout) ──────────
        action_tensor = F.one_hot(torch.tensor([action_val], device=device), num_classes=8).float()
        with torch.no_grad():
            # embed=None forces the model to use the prior transition p(s_t | h_t)
            step_result = rssm_cell(h, s, action_tensor, embed=None, use_gumbel=False)
            h, s = step_result["h"], step_result["s"]
            
            latent = torch.cat([h, s], dim=-1)
            pred_obs_tensor = decoder(latent) # (1, 3, 144, 160)
            
        pred_obs = pred_obs_tensor.squeeze(0).cpu().numpy() # (3, 144, 160)
        pred_obs = np.transpose(pred_obs, (1, 2, 0)) # (144, 160, 3)
        pred_obs = np.clip(pred_obs, 0, 1)

        # Contrast stretch to push soft grey values to sharp black/white (retro pixel art style)
        pred_obs = (pred_obs - pred_obs.min()) / (pred_obs.max() - pred_obs.min() + 1e-5)
        pred_obs = 1.0 / (1.0 + np.exp(-12.0 * (pred_obs - 0.5)))
        pred_obs = (pred_obs * 255.0).clip(0, 255).astype(np.uint8)
        
        # ────────── C. CREATE COMBINED VISUAL FRAME ──────────
        left_panel = cv2.resize(real_frame, (view_w, view_h), interpolation=cv2.INTER_CUBIC)
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
        cv2.putText(canvas, "IMAGINED FUTURE (DISCRETE RSSM)", (view_w + border_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        
        # Action/step text at footer
        info_text = f"Step: {step_idx+1:02d} | Action: {btn.upper()}"
        cv2.putText(canvas, info_text, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Write frame to video
        video_writer.write(canvas)
        steps_executed += 1
        print(f"  Processed step {steps_executed:>2}/{args.max_steps} | Action: {btn}")

    pyboy.stop()
    video_writer.release()
    print(f"\n[generate_demo_video_v2] Demo video completed successfully!")
    print(f"Saved to: {args.out_video.resolve()}")


if __name__ == "__main__":
    main()
