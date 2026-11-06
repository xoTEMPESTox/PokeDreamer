"""
collect_data.py — Day 1-3 deliverable script.

Runs the frozen PPO policy (or random actions) in non-deterministic mode to
collect diverse trajectories. Extracts downsampled (36, 40, 3) frames, actions,
and relevant RAM states, then dumps them in compressed .npz chunks.

Usage:
    conda activate pokemon-rl
    python scripts/collect_data.py \
        --rom "Pokemon - Red Version (USA, Europe).gb" \
        --checkpoint "external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip" \
        --total-steps 150000 \
        --episode-length 1000 \
        --save-interval 10000 \
        --out-dir data
"""

import argparse
import random
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyboy import PyBoy
from src.game_state import extract_game_state, screen_capture

# PWhiddy 8-action set mapping in controller.py
BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect game trajectories for world model training")
    p.add_argument("--rom", type=Path, default=Path("Pokemon - Red Version (USA, Europe).gb"),
                   help="Path to PokemonRed ROM")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip"),
                   help="SB3 PPO .zip checkpoint (omit for random actions)")
    p.add_argument("--total-steps", type=int, default=150000,
                   help="Total number of steps to collect (default: 150k)")
    p.add_argument("--episode-length", type=int, default=1000,
                   help="Max steps per episode before reset (default: 1000)")
    p.add_argument("--save-interval", type=int, default=10000,
                   help="Number of steps per compressed .npz chunk (default: 10k)")
    p.add_argument("--out-dir", type=Path, default=Path("data"),
                   help="Directory to save dataset chunks (default: data)")
    p.add_argument("--ticks-per-action", type=int, default=24,
                   help="Emulator ticks per action (default: 24)")
    p.add_argument("--det-prob", type=float, default=0.2,
                   help="Probability of selecting deterministic mode for an episode (default: 0.2)")
    p.add_argument("--rand-prob", type=float, default=0.1,
                   help="Probability of selecting random-action mode for an episode (default: 0.1)")
    return p.parse_args()


def find_all_states(root_dir: Path) -> list[Path]:
    """Dynamically scan for all available .state files in the project."""
    states = []
    
    # Check standard directories
    paths_to_check = [
        root_dir / "saves",
        root_dir / "external" / "PokemonRedExperiments",
    ]
    
    for folder in paths_to_check:
        if folder.exists():
            for p in folder.glob("*.state"):
                states.append(p)
                
    # Deduplicate and sort
    return sorted(list(set(states)))


def main() -> None:
    args = parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.rom.exists():
        print(f"[ERROR] ROM not found: {args.rom.resolve()}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Policy ───────────────────────────────────────────────────────────
    policy = None
    if args.checkpoint and args.checkpoint.exists():
        try:
            from src.frozen_ppo import FrozenPPO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            policy = FrozenPPO(args.checkpoint, device=device)
            print(f"[collect_data] Loaded policy: {policy} on device: {device}")
        except Exception as e:
            print(f"[WARN] Failed to load policy: {e}. Falling back to random actions.")
    
    if policy is None:
        print("[collect_data] Running in RANDOM action mode.")

    # ── Find Save States ──────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parents[1]
    save_states = find_all_states(project_root)
    if not save_states:
        print("[ERROR] No save states (.state) found in saves/ or external/ directories.")
        sys.exit(1)
        
    print(f"[collect_data] Found {len(save_states)} save states to cycle through:")
    for s in save_states:
        print(f"  - {s.relative_to(project_root)}")

    # ── Buffer Allocation ─────────────────────────────────────────────────────
    # We pre-allocate lists to append step records, then dump when save-interval is met
    obs_buffer = []
    action_buffer = []
    reward_buffer = []
    episode_starts_buffer = []
    
    # RAM states
    map_ids_buffer = []
    xs_buffer = []
    ys_buffer = []
    facings_buffer = []
    in_battles_buffer = []
    dialog_opens_buffer = []
    badges_buffer = []
    party_hps_buffer = []
    party_max_hps_buffer = []

    # Counters
    total_steps_collected = 0
    chunk_counter = 0
    episode_counter = 0

    print(f"\n[collect_data] Beginning collection of {args.total_steps} steps...")
    
    t_start = time.perf_counter()

    while total_steps_collected < args.total_steps:
        # Determine episode configuration
        roll = random.random()
        if roll < args.det_prob:
            mode = "deterministic"
            det_flag = True
            rand_flag = False
        elif roll < args.det_prob + args.rand_prob:
            mode = "random"
            det_flag = False
            rand_flag = True
        else:
            mode = "stochastic"
            det_flag = False
            rand_flag = False

        # Select a starting save state
        state_path = random.choice(save_states)
        episode_counter += 1
        
        print(f"\n[Episode {episode_counter}] Mode: {mode} | Save State: {state_path.name}")
        
        # Start a fresh PyBoy session for each episode to avoid load_state memory corruption hangs
        pyboy = PyBoy(str(args.rom), window="null")
        with open(state_path, "rb") as f:
            pyboy.load_state(f)

        if policy is not None:
            policy.reset()

        step_in_ep = 0
        ep_steps_limit = args.episode_length
        
        while step_in_ep < ep_steps_limit and total_steps_collected < args.total_steps:
            # 1. Capture screen & state *before* action is taken
            raw_frame = screen_capture(pyboy)
            # Preprocess to (36, 40, 3) matching FrozenPPO
            if policy is not None:
                downsampled_frame = policy._downsample(raw_frame)
            else:
                downsampled_frame = cv2.resize(raw_frame, (40, 36), interpolation=cv2.INTER_AREA).astype(np.uint8)
                
            ram_state = extract_game_state(pyboy)

            # 2. Decide action
            if rand_flag or policy is None:
                action = random.randint(0, 5) # Model action space is 6 actions
            else:
                action = policy.predict(raw_frame, deterministic=det_flag)

            # 3. Store step records
            obs_buffer.append(downsampled_frame)
            action_buffer.append(action)
            reward_buffer.append(0.0) # Dummy reward
            episode_starts_buffer.append(step_in_ep == 0)
            
            map_ids_buffer.append(ram_state.map_id)
            xs_buffer.append(ram_state.x)
            ys_buffer.append(ram_state.y)
            facings_buffer.append(ram_state.facing)
            in_battles_buffer.append(ram_state.in_battle)
            dialog_opens_buffer.append(ram_state.dialog_open)
            badges_buffer.append(ram_state.badges)
            
            # Pad party lists to length 6
            hps = list(ram_state.party_hp) + [0] * (6 - len(ram_state.party_hp))
            max_hps = list(ram_state.party_max_hp) + [0] * (6 - len(ram_state.party_max_hp))
            party_hps_buffer.append(hps)
            party_max_hps_buffer.append(max_hps)

            # 4. Advance emulator
            btn = BUTTONS[action]
            if btn != "pass":
                pyboy.button(btn, 8)
            pyboy.tick(args.ticks_per_action, render=True)

            step_in_ep += 1
            total_steps_collected += 1

            # 5. Periodic saving
            if len(obs_buffer) >= args.save_interval:
                chunk_counter += 1
                chunk_path = args.out_dir / f"transitions_{chunk_counter:04d}.npz"
                
                print(f"\n>> Saving chunk {chunk_counter} to {chunk_path}...")
                np.savez_compressed(
                    chunk_path,
                    obs=np.array(obs_buffer, dtype=np.uint8),
                    actions=np.array(action_buffer, dtype=np.uint8),
                    rewards=np.array(reward_buffer, dtype=np.float32),
                    episode_starts=np.array(episode_starts_buffer, dtype=bool),
                    map_ids=np.array(map_ids_buffer, dtype=np.uint8),
                    xs=np.array(xs_buffer, dtype=np.uint8),
                    ys=np.array(ys_buffer, dtype=np.uint8),
                    facings=np.array(facings_buffer, dtype=np.uint8),
                    in_battles=np.array(in_battles_buffer, dtype=bool),
                    dialog_opens=np.array(dialog_opens_buffer, dtype=bool),
                    badges=np.array(badges_buffer, dtype=np.uint8),
                    party_hps=np.array(party_hps_buffer, dtype=np.int16),
                    party_max_hps=np.array(party_max_hps_buffer, dtype=np.int16)
                )
                
                # Clear buffers
                obs_buffer.clear()
                action_buffer.clear()
                reward_buffer.clear()
                episode_starts_buffer.clear()
                map_ids_buffer.clear()
                xs_buffer.clear()
                ys_buffer.clear()
                facings_buffer.clear()
                in_battles_buffer.clear()
                dialog_opens_buffer.clear()
                badges_buffer.clear()
                party_hps_buffer.clear()
                party_max_hps_buffer.clear()

                elapsed = time.perf_counter() - t_start
                steps_per_sec = total_steps_collected / elapsed
                print(f"   Collected: {total_steps_collected}/{args.total_steps} steps | Speed: {steps_per_sec:.1f} steps/s | Elapsed: {elapsed:.1f}s")

        pyboy.stop()

    # Handle remaining steps in buffers
    if obs_buffer:
        chunk_counter += 1
        chunk_path = args.out_dir / f"transitions_{chunk_counter:04d}.npz"
        print(f"\n>> Saving final chunk {chunk_counter} to {chunk_path}...")
        np.savez_compressed(
            chunk_path,
            obs=np.array(obs_buffer, dtype=np.uint8),
            actions=np.array(action_buffer, dtype=np.uint8),
            rewards=np.array(reward_buffer, dtype=np.float32),
            episode_starts=np.array(episode_starts_buffer, dtype=bool),
            map_ids=np.array(map_ids_buffer, dtype=np.uint8),
            xs=np.array(xs_buffer, dtype=np.uint8),
            ys=np.array(ys_buffer, dtype=np.uint8),
            facings=np.array(facings_buffer, dtype=np.uint8),
            in_battles=np.array(in_battles_buffer, dtype=bool),
            dialog_opens=np.array(dialog_opens_buffer, dtype=bool),
            badges=np.array(badges_buffer, dtype=np.uint8),
            party_hps=np.array(party_hps_buffer, dtype=np.int16),
            party_max_hps=np.array(party_max_hps_buffer, dtype=np.int16)
        )

    total_time = time.perf_counter() - t_start
    print(f"\n[SUCCESS] Completed data collection! Collected {total_steps_collected} steps in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
