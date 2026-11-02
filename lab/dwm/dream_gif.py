"""Real vs Dream side-by-side GIF generator from agent trajectory npz.

Every K steps (default 16), takes the last 4 real frames as context and rolls
the world model forward open-loop for K steps with the actions the agent actually took.
Shows real (left) and imagined (right), upsampled, with a caption line:
- System 2 goal
- Step
- Map name from RAM vs probe's predicted map (flagged in red on mismatch)
- Tile drift
"""
import argparse
import contextlib
import glob
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import lab  # handles Windows DLL / OpenMP setup
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, Denoiser, to_float
from lab.probe import load_probe, predict
from lab.upsampler import load_upsampler, upsample
from src.ram_addresses import MAP_NAMES

MAP_NAME_OVERRIDES = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    12: "Route 1",
    13: "Route 2",
    41: "Viridian Pokecenter",
    42: "Viridian Mart",
    47: "Forest North Gate",
    50: "Forest South Gate",
    51: "Viridian Forest",
    54: "Pewter Gym",
    58: "Pewter Pokecenter",
}


def get_map_name(map_id: int) -> str:
    m_int = int(map_id)
    if m_int in MAP_NAME_OVERRIDES:
        return MAP_NAME_OVERRIDES[m_int]
    return MAP_NAMES.get(m_int, f"Map {m_int}")


def compose_frame(
    real_f: np.ndarray,
    dream_f: np.ndarray,
    goal: str,
    step: int,
    ram_map: int,
    probe_map: int,
    tile_drift: int,
    upsampler: Optional[Any] = None,
    device: str = "cpu",
    target_width: int = 640,
) -> np.ndarray:
    """Compose side-by-side real vs dream frame with status caption line.

    Args:
        real_f: (72, 80) or (144, 160) uint8 or float [0, 1]
        dream_f: (72, 80) or (144, 160) uint8 or float [0, 1]
        goal: System 2 goal string
        step: Trajectory step index
        ram_map: Map ID from RAM ground truth
        probe_map: Map ID predicted by probe
        tile_drift: Manhattan distance in tiles
        upsampler: Optional Upsampler model
        device: Torch device for upsampler
        target_width: Combined width for the side-by-side display (default 640)

    Returns:
        (H, W, 3) uint8 RGB image with banner
    """
    # 1. Upsample or resize real frame to (144, 160) uint8
    if real_f.shape == (72, 80):
        if upsampler is not None:
            real_144 = upsample(upsampler, real_f, device=device)
        else:
            real_144 = cv2.resize(real_f, (160, 144), interpolation=cv2.INTER_NEAREST)
    else:
        real_144 = np.asarray(real_f, dtype=np.uint8)

    # 2. Upsample or resize dream frame to (144, 160) uint8
    if dream_f.shape == (72, 80):
        if upsampler is not None:
            dream_144 = upsample(upsampler, dream_f, device=device)
        else:
            dream_144 = cv2.resize(dream_f, (160, 144), interpolation=cv2.INTER_NEAREST)
    else:
        dream_144 = np.asarray(dream_f, dtype=np.uint8)

    if real_144.ndim == 3 and real_144.shape[0] == 1:
        real_144 = real_144[0]
    if dream_144.ndim == 3 and dream_144.shape[0] == 1:
        dream_144 = dream_144[0]

    # Side-by-side horizontal concatenation: 144x320
    side_by_side = np.hstack([real_144, dream_144])

    # Scale to target_width (e.g. 640x288)
    scale = target_width / float(side_by_side.shape[1])
    h_scaled = int(round(side_by_side.shape[0] * scale))
    disp_scaled = cv2.resize(side_by_side, (target_width, h_scaled), interpolation=cv2.INTER_NEAREST)
    disp_rgb = cv2.cvtColor(disp_scaled, cv2.COLOR_GRAY2RGB)

    # Dividing line in center
    mid_x = target_width // 2
    disp_rgb[:, mid_x - 1 : mid_x + 1] = (60, 60, 60)

    # Subtle Real / Dream labels
    cv2.putText(disp_rgb, "REAL", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(disp_rgb, "DREAM", (mid_x + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Caption banner (34 px height)
    banner_h = 34
    banner = np.full((banner_h, target_width, 3), 26, dtype=np.uint8)
    banner[0, :] = (65, 65, 65)  # Separator line

    ram_name = get_map_name(ram_map)
    probe_name = get_map_name(probe_map)
    mismatch = (int(ram_map) != int(probe_map))

    # Left: Goal & Step
    text_left = f"{goal.upper()} | Step {step}"
    cv2.putText(banner, text_left, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)

    # Center: Map info (flag mismatches in RED)
    if mismatch:
        map_text = f"Map: {ram_name} != {probe_name}"
        map_color = (255, 50, 50)  # Bright Red
    else:
        map_text = f"Map: {ram_name}"
        map_color = (130, 235, 130)  # Green

    cv2.putText(banner, map_text, (220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, map_color, 1, cv2.LINE_AA)

    # Right: Tile drift
    drift_text = f"Drift: {tile_drift} tiles"
    drift_color = (255, 120, 120) if tile_drift > 2 else (225, 225, 225)
    cv2.putText(banner, drift_text, (target_width - 130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, drift_color, 1, cv2.LINE_AA)

    return np.vstack([disp_rgb, banner])


def generate_dream_gif(
    run: str = "wm31",
    traj_path: str | Path = "work/gym_traj/run_seed_100.npz",
    probe_path: Optional[str | Path] = None,
    k: int = 16,
    start: int = 0,
    length: Optional[int] = 256,
    out: Optional[str | Path] = None,
    every: int = 1,
    device: str = "cpu",
    steps: int = 3,
    fps: int = 10,
    model: Optional[Denoiser] = None,
    probe: Optional[Any] = None,
    upsampler: Optional[Any] = None,
) -> Path:
    """Generate side-by-side real vs dream GIF from trajectory npz."""
    traj_path = Path(traj_path)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # 1. Load world model if not provided
    if model is None:
        print(f"Loading world model run '{run}' on {device}...")
        model = load_world_model(run, device=device)
    model.eval()
    model.to(device)

    # 2. Load probe if not provided
    if probe is None:
        if probe_path is None:
            cand_ext = Path("work/probe_ext/probe.pt")
            cand_base = Path("work/probe/probe.pt")
            probe_path = cand_ext if cand_ext.exists() else cand_base
        probe_path = Path(probe_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe not found at {probe_path}")
        print(f"Loading probe from {probe_path} on {device}...")
        probe = load_probe(str(probe_path), device=device)

    # 3. Load upsampler if not provided
    if upsampler is None:
        up_path = Path("work/upsampler/upsampler.pt")
        if up_path.exists():
            try:
                upsampler = load_upsampler(str(up_path), device=device)
                print(f"Loaded upsampler from {up_path}")
            except Exception as e:
                print(f"Could not load upsampler: {e}; falling back to nearest 2x.")
                upsampler = None
        else:
            upsampler = None

    # 4. Load trajectory
    traj = np.load(traj_path)
    frames = traj["frames"]
    actions = traj["actions"]
    map_ids = traj["map_ids"] if "map_ids" in traj else traj["map"]
    xs = traj["xs"] if "xs" in traj else traj["x"]
    ys = traj["ys"] if "ys" in traj else traj["y"]
    goals = traj["goals"] if "goals" in traj else traj["goal"]

    total_steps = len(frames)
    start_step = max(CTX, start)
    if length is not None:
        end_step = min(total_steps, start_step + length)
    else:
        end_step = total_steps

    if start_step >= end_step:
        raise ValueError(f"Invalid range: start={start_step} >= end={end_step} (total steps={total_steps})")

    print(f"Rolling forward world model from step {start_step} to {end_step} (horizon K={k}, stride={every})...")

    composed_frames: list[np.ndarray] = []
    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if "cuda" in str(device) else contextlib.nullcontext()

    # Roll out in open-loop chunks of size K
    for chunk_start in range(start_step, end_step, k):
        cur_k = min(k, end_step - chunk_start)

        # 4 real context frames leading into chunk_start
        ctx_raw = frames[chunk_start - CTX : chunk_start]
        ctx_tensor = to_float(torch.from_numpy(ctx_raw).unsqueeze(0).to(device))

        # Actions taken by agent across context and chunk
        act_raw = actions[chunk_start - CTX : chunk_start + cur_k - 1]
        act_tensor = torch.from_numpy(act_raw).long().unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast_ctx:
                img_batch = model.rollout(ctx_tensor, act_tensor, steps=steps)

        img_np = img_batch.cpu().float().numpy()[0]  # (cur_k, 72, 80) in [-1, 1]
        img_01 = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)
        img_u8 = np.clip(np.round(img_01 * 255.0), 0, 255).astype(np.uint8)

        # Batch probe prediction on dream frames
        preds = predict(probe, img_01, device=device)

        for i in range(cur_k):
            step_idx = chunk_start + i
            ram_m = int(map_ids[step_idx])
            ram_x = int(xs[step_idx])
            ram_y = int(ys[step_idx])
            goal_s = str(goals[step_idx])

            pr_m = int(preds["map_id"][i])
            pr_x = int(preds["x"][i])
            pr_y = int(preds["y"][i])

            drift = abs(ram_x - pr_x) + abs(ram_y - pr_y)

            frame_comp = compose_frame(
                real_f=frames[step_idx],
                dream_f=img_u8[i],
                goal=goal_s,
                step=step_idx,
                ram_map=ram_m,
                probe_map=pr_m,
                tile_drift=drift,
                upsampler=upsampler,
                device=device,
            )
            composed_frames.append(frame_comp)

    # Apply stride
    selected_frames = composed_frames[::every]
    if not selected_frames:
        raise RuntimeError("No frames selected for GIF output.")

    # Convert to PIL paletted images
    pil_frames = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in selected_frames]

    out_path = Path(out) if out else Path(f"work/dream_{run}.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(pil_frames)} frames to {out_path} ...")
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"GIF saved successfully: {out_path} ({len(pil_frames)} frames, {size_mb:.2f} MB)")
    if size_mb > 15.0:
        print(f"Warning: GIF size ({size_mb:.2f} MB) exceeds 15 MB limit. Consider using --every > 1 or smaller --length.")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate real vs dream side-by-side GIF from trajectory npz.")
    parser.add_argument("--run", type=str, default="wm31", help="World model run name in work/runs/<run>")
    parser.add_argument("--probe", type=str, default=None, help="Probe checkpoint path (default: work/probe_ext/probe.pt)")
    parser.add_argument("--traj", type=str, default=None, help="Path to trajectory npz file")
    parser.add_argument("--k", type=int, default=16, help="Rollout horizon / re-contexting interval K (default 16)")
    parser.add_argument("--start", type=int, default=0, help="Starting step in trajectory (default 0)")
    parser.add_argument("--length", type=int, default=256, help="Number of steps to roll out and show (default 256)")
    parser.add_argument("--out", type=str, default=None, help="Output GIF path (default: work/dream_<run>.gif)")
    parser.add_argument("--every", type=int, default=1, help="Stride across steps for GIF frames to keep size small (default 1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--steps", type=int, default=3, help="Euler sampler diffusion steps (default 3)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second in output GIF (default 10)")
    args = parser.parse_args()

    traj_path = args.traj
    if traj_path is None:
        # Look for latest trajectory in work/gym_traj
        found = sorted(glob.glob("work/gym_traj/run_seed_*.npz"))
        if found:
            traj_path = found[-1]
            print(f"Using latest trajectory found: {traj_path}")
        else:
            raise FileNotFoundError("No trajectory file specified with --traj and none found in work/gym_traj/")

    generate_dream_gif(
        run=args.run,
        traj_path=traj_path,
        probe_path=args.probe,
        k=args.k,
        start=args.start,
        length=args.length,
        out=args.out,
        every=args.every,
        device=args.device,
        steps=args.steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
