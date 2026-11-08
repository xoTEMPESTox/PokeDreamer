"""Interactive "play inside the world model" demo for PokeDreamer.

Loads the diffusion world model (Denoiser) and optional upsampler, allowing real-time
exploration of Pokemon Red inside the model's imagination.
"""
import argparse
import contextlib
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import lab  # handles Windows DLL / OpenMP setup
from lab.data import load
from lab.dwm.model import CTX, Denoiser, to_float
from lab.dwm.train import windows
from lab.env import PokemonRed
from lab.upsampler import load_upsampler, save_screenshot, upsample

ACTION_NAMES = {
    0: "DOWN",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "A",
    5: "B",
    6: "START",
    7: "PASS",
}

SCRIPT_MAP = {
    "d": 0, "D": 0,
    "l": 1, "L": 1,
    "r": 2, "R": 2,
    "u": 3, "U": 3,
    "a": 4, "A": 4,
    "b": 5, "B": 5,
    "s": 6, "S": 6,
    ".": 7, " ": 7,
}

KEY_TO_ACTION = {
    # Windows arrow keys
    2621440: 0,  # down
    2424832: 1,  # left
    2555904: 2,  # right
    2490368: 3,  # up
    # Standard X11 / curses arrow keys
    65364: 0,
    65361: 1,
    65363: 2,
    65362: 3,
    # WASD controls
    ord("s"): 0, ord("S"): 0,
    ord("a"): 1, ord("A"): 1,
    ord("d"): 2, ord("D"): 2,
    ord("w"): 3, ord("W"): 3,
    # Game Boy buttons
    ord("z"): 4, ord("Z"): 4,  # A
    ord("x"): 5, ord("X"): 5,  # B
    13: 6, 10: 6,             # Enter -> START
    32: 7,                    # Space -> PASS
}

QUIT_KEYS = {ord("q"), ord("Q"), 27}
RESET_KEYS = {ord("r"), ord("R")}
TOGGLE_KEYS = {ord("u"), ord("U")}
SCREENSHOT_KEYS = {ord("p"), ord("P")}


def get_overworld_windows(data, split_episodes, map_id=None):
    """Find valid start indices for CTX frames entirely within an overworld episode without battles.
    Starts are >= 1 frame into the episode so the action before the first context frame exists."""
    wins = windows(data, split_episodes, CTX + 1) + 1
    in_b = data["in_battles"]
    ar = np.arange(CTX)
    idx_grid = wins[:, None] + ar
    no_battle = (~in_b[idx_grid]).all(axis=1)
    valid = wins[no_battle]
    if map_id is not None:
        maps = data["map_ids"]
        on_map = (maps[valid[:, None] + ar] == int(map_id)).all(axis=1)
        valid = valid[on_map]
    return valid


def initial_actions(data, start_idx):
    """Action buffer a_{s-1}..a_{s+CTX-2}: the actions that led into each context frame. Each key press
    drops the oldest and appends the pressed button, which the model reads as the action after the last frame."""
    return data["actions"][start_idx - 1 : start_idx + CTX - 1]


def is_anchor_step(step: int, anchor_n: int) -> bool:
    """Return True if step is an anchor step (press_count % N == 0)."""
    return anchor_n > 0 and step > 0 and (step % anchor_n == 0)


def next_anchor_in(step: int, anchor_n: int) -> int:
    """Return number of steps until next anchor."""
    return 0 if anchor_n <= 0 else (anchor_n if step % anchor_n == 0 else anchor_n - (step % anchor_n))


def anchor_status_text(step: int, anchor_n: int) -> str:
    """Status bar string for anchor state."""
    if anchor_n <= 0:
        return ""
    if is_anchor_step(step, anchor_n):
        return f"Anchor every {anchor_n} | ANCHORED"
    return f"Anchor every {anchor_n} | next anchor in {next_anchor_in(step, anchor_n)}"


def render_pane(frame_data, upsampler, use_upsampler, device):
    """Render a 72x80 frame (float tensor in [-1, 1] or uint8 numpy) to (576, 640, 3) BGR."""
    if isinstance(frame_data, torch.Tensor):
        curr = frame_data if frame_data.dim() == 3 else frame_data.unsqueeze(0)
        f01 = ((curr + 1.0) / 2.0).clamp(0.0, 1.0)
        if use_upsampler and upsampler is not None:
            frame_144 = upsample(upsampler, f01, device=device)[0]
        else:
            frame_144 = cv2.resize((f01 * 255.0).round().to(torch.uint8).cpu().numpy()[0], (160, 144), interpolation=cv2.INTER_NEAREST)
    else:
        if use_upsampler and upsampler is not None:
            frame_144 = upsample(upsampler, torch.from_numpy(frame_data).float().unsqueeze(0) / 255.0, device=device)[0]
        else:
            frame_144 = cv2.resize(frame_data, (160, 144), interpolation=cv2.INTER_NEAREST)
    disp_576 = cv2.resize(frame_144, (640, 576), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(disp_576, cv2.COLOR_GRAY2BGR)


def render_display_frame(
    ctx,
    upsampler,
    use_upsampler,
    step_count,
    last_button,
    step_ms,
    device,
    real_frame=None,
    anchor_n=0,
):
    """Render the 576x640 frame (or 576x1296 side-by-side) plus 32px status bar (total 608 height BGR)."""
    dream_bgr = render_pane(ctx[:, -1], upsampler, use_upsampler, device)
    upsampler_tag = "ON" if (use_upsampler and upsampler is not None) else ("OFF" if upsampler is not None else "N/A")

    if real_frame is not None:
        real_bgr = render_pane(real_frame, upsampler, use_upsampler, device)
        for pane, label in [(real_bgr, "REAL"), (dream_bgr, "DREAM")]:
            cv2.putText(pane, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(pane, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        panes = np.hstack([real_bgr, np.zeros((576, 16, 3), dtype=np.uint8), dream_bgr])
        w = 1296
    else:
        panes = dream_bgr
        w = 640

    status_bar = np.full((32, w, 3), 32, dtype=np.uint8)
    status_bar[0, :] = 60  # subtle separator line

    btn_str = last_button if last_button is not None else "INIT"
    ms_str = f"{step_ms:.1f} ms" if step_ms > 0 else "- ms"
    anchor_info = f" | {anchor_status_text(step_count, anchor_n)}" if anchor_n > 0 else ""
    text = f"Step: {step_count} | Button: {btn_str} | {ms_str} | Upsampler: {upsampler_tag}{anchor_info}"

    cv2.putText(status_bar, text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([panes, status_bar])


def main():
    parser = argparse.ArgumentParser(description="Play inside the PokeDreamer world model")
    parser.add_argument("--run", type=str, default="main", help="Run name in work/runs/<run>")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--data", type=str, default="base", choices=["base", "ext"], help="Dataset to load: base or ext")
    parser.add_argument("--split", type=str, default=None, help="Split for random context (train, test_id, test_ood, test_ext)")
    parser.add_argument("--start-map", type=int, default=None, help="Start context from this map ID (e.g. 2 for Pewter City, 51 for Viridian Forest)")
    parser.add_argument("--start", type=int, default=None, help="Explicit start frame index (not the first frame of an episode)")
    parser.add_argument("--script", type=str, default=None, help="Scripted button sequence for headless video recording")
    parser.add_argument("--shot", action="store_true", help="Save screenshot on last step in scripted mode")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path for scripted mode")
    parser.add_argument("--steps", type=int, default=3, help="EDM denoiser steps per frame")
    parser.add_argument("--anchor", type=int, default=0, help="Re-anchor dream model on real frames every N steps (0 = off)")
    parser.add_argument("--state", type=str, default="saves/pokedex_viridian.state", help="Save state for real emulator in anchored mode")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path for scripted mode")
    args = parser.parse_args()

    dev = torch.device(args.device)
    print(f"Loading world model from work/runs/{args.run}/ckpt.pt on {args.device}...")
    ckpt_path = Path("work/runs") / args.run / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model = Denoiser().to(dev)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    upsampler_path = Path("work/upsampler/upsampler.pt")
    if upsampler_path.exists():
        print(f"Loading upsampler from {upsampler_path}...")
        upsampler = load_upsampler(str(upsampler_path), device=args.device)
        use_upsampler = True
    else:
        print("Upsampler checkpoint not found; running with nearest 2x upsampling.")
        upsampler = None
        use_upsampler = False

    if args.anchor > 0:
        state_path = Path(args.state)
        if not state_path.exists():
            raise FileNotFoundError(f"Save state not found at {state_path}")

        print(f"Initializing PokemonRed emulator from {state_path} (anchored mode N={args.anchor})...")
        env = PokemonRed(state=str(state_path))

        def init_anchored_context():
            real_buf = deque([env.step("pass")[0] for _ in range(CTX)], maxlen=CTX)
            c = to_float(torch.from_numpy(np.stack(list(real_buf))).unsqueeze(0)).to(dev)
            a = torch.full((1, CTX), 7, dtype=torch.long, device=dev)
            return c, a, real_buf

        ctx, actions, real_frames = init_anchored_context()
        cur_real_frame = real_frames[-1]
        print(f"Initialized anchored context from emulator ({CTX} PASS frames)")
    else:
        env = None
        real_frames = None
        cur_real_frame = None

        is_ext = (args.data == "ext")
        data, splits = load(extended=is_ext)

        if args.split is not None:
            split_name = args.split
        elif is_ext:
            split_name = "test_ext"
        else:
            split_name = "test_id"

        if split_name not in splits:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(splits.keys())}")

        valid_starts = get_overworld_windows(data, splits[split_name], map_id=args.start_map)
        if len(valid_starts) == 0:
            err = f"No valid non-battle overworld windows found in split '{split_name}'"
            if args.start_map is not None:
                err += f" for map ID {args.start_map}"
            raise RuntimeError(err)

        rng = np.random.default_rng()

        def init_context(start_idx=None):
            if start_idx is None:
                start_idx = int(rng.choice(valid_starts))
            f_init = data["frames"][start_idx : start_idx + CTX]
            a_init = initial_actions(data, start_idx)
            c = to_float(torch.from_numpy(f_init).unsqueeze(0)).to(dev)
            a = torch.from_numpy(a_init).long().unsqueeze(0).to(dev)
            return c, a, start_idx

        ctx, actions, cur_start = init_context(args.start)
        print(f"Initialized context from frame {cur_start} (split {split_name}, map {data['map_ids'][cur_start]})")

    autocast_ctx = torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    # Headless scripted mode
    if args.script is not None:
        out_path = Path(args.out) if args.out else Path(f"work/videos/play_{args.run}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width = 1296 if args.anchor > 0 else 640
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 4.0, (width, 608))

        gif_frames = [] if args.gif is not None else None

        step_count = 0
        last_button = "INIT"
        step_ms = 0.0
        anchor_count = 0

        # Initial frame
        frame = render_display_frame(
            ctx, upsampler, use_upsampler, step_count, last_button, step_ms, dev,
            real_frame=cur_real_frame, anchor_n=args.anchor,
        )
        writer.write(frame)
        if gif_frames is not None:
            gif_frames.append(frame)

        print(f"Running scripted sequence of {len(args.script)} actions...")
        for char in args.script:
            btn = SCRIPT_MAP.get(char, 7)
            btn_name = ACTION_NAMES[btn]

            if args.anchor > 0:
                cur_real_frame, real_ram = env.step(btn_name.lower())
                real_frames.append(cur_real_frame)

            actions = torch.cat([actions[:, 1:], torch.tensor([[btn]], device=dev, dtype=torch.long)], dim=1)
            t0 = time.perf_counter()
            with autocast_ctx:
                next_f = model.sample(ctx, actions, steps=args.steps)
            step_ms = (time.perf_counter() - t0) * 1000

            ctx = torch.cat([ctx[:, 1:], next_f], dim=1)
            step_count += 1
            last_button = btn_name

            is_anchor = is_anchor_step(step_count, args.anchor)
            if is_anchor:
                anchor_count += 1

            if args.anchor > 0:
                print(f"step {step_count:2d} | button: {btn_name:<5} | RAM: map_id={real_ram['map_id']}, x={real_ram['x']}, y={real_ram['y']} | anchor: {is_anchor}")

            frame = render_display_frame(
                ctx, upsampler, use_upsampler, step_count, last_button, step_ms, dev,
                real_frame=cur_real_frame, anchor_n=args.anchor,
            )
            writer.write(frame)
            if gif_frames is not None:
                gif_frames.append(frame)

            if is_anchor:
                ctx = to_float(torch.from_numpy(np.stack(list(real_frames))).unsqueeze(0)).to(dev)

        writer.release()
        print(f"Scripted video written to {out_path} ({step_count + 1} frames)")
        if args.anchor > 0:
            print(f"Anchors: {anchor_count}")

        if args.gif and gif_frames:
            from PIL import Image
            Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
            pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in gif_frames]
            pil_imgs[0].save(args.gif, save_all=True, append_images=pil_imgs[1:], duration=250, loop=0)
            print(f"Scripted GIF written to {args.gif} ({len(pil_imgs)} frames)")

        if args.shot:
            curr_f01 = ((ctx[:, -1] + 1.0) / 2.0).clamp(0.0, 1.0)
            curr_u8 = (curr_f01 * 255.0).round().to(torch.uint8).cpu().numpy()[0]
            save_screenshot(curr_u8, upsampler, out_dir="work/screenshots", step=step_count, device=dev)

        if env is not None:
            env.close()
        return

    # Interactive mode
    win_name = "PokeDreamer - Pokemon Red World Model"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    step_count = 0
    last_button = None
    step_ms = 0.0
    needs_reanchor = False

    print("\nControls:")
    print("  Arrows / WASD : D-pad (Down, Left, Right, Up)")
    print("  Z             : A button")
    print("  X             : B button")
    print("  Enter         : START")
    print("  Space         : PASS (no button)")
    print("  R             : Reset context")
    print("  U             : Toggle upsampler (ON/OFF)")
    print("  P             : Save screenshot to work/screenshots/")
    print("  Q / Esc       : Quit\n")

    while True:
        display = render_display_frame(
            ctx, upsampler, use_upsampler, step_count, last_button, step_ms, dev,
            real_frame=cur_real_frame, anchor_n=args.anchor,
        )
        cv2.imshow(win_name, display)

        key = cv2.waitKeyEx(0)
        if key in QUIT_KEYS or (key & 0xFF) in QUIT_KEYS:
            break
        elif key in RESET_KEYS or (key & 0xFF) in RESET_KEYS:
            if args.anchor > 0:
                env.restore(str(args.state))
                ctx, actions, real_frames = init_anchored_context()
                cur_real_frame = real_frames[-1]
                cur_start = args.state
            else:
                ctx, actions, cur_start = init_context()
            step_count, last_button, step_ms, needs_reanchor = 0, "RESET", 0.0, False
            print(f"Reset context from {cur_start}")
            continue
        elif key in TOGGLE_KEYS or (key & 0xFF) in TOGGLE_KEYS:
            if upsampler is not None:
                use_upsampler = not use_upsampler
                print(f"Upsampler toggled {'ON' if use_upsampler else 'OFF'}")
            else:
                print("Upsampler model not loaded (no checkpoint found).")
            continue
        elif key in SCREENSHOT_KEYS or (key & 0xFF) in SCREENSHOT_KEYS:
            curr_f01 = ((ctx[:, -1] + 1.0) / 2.0).clamp(0.0, 1.0)
            curr_u8 = (curr_f01 * 255.0).round().to(torch.uint8).cpu().numpy()[0]
            save_screenshot(curr_u8, upsampler, out_dir="work/screenshots", step=step_count, device=dev)
            continue

        btn = None
        if key in KEY_TO_ACTION:
            btn = KEY_TO_ACTION[key]
        elif (key & 0xFF) in KEY_TO_ACTION:
            btn = KEY_TO_ACTION[key & 0xFF]

        if btn is not None:
            btn_name = ACTION_NAMES[btn]
            if needs_reanchor:
                ctx = to_float(torch.from_numpy(np.stack(list(real_frames))).unsqueeze(0)).to(dev)
                needs_reanchor = False

            if args.anchor > 0:
                cur_real_frame, real_ram = env.step(btn_name.lower())
                real_frames.append(cur_real_frame)

            actions = torch.cat([actions[:, 1:], torch.tensor([[btn]], device=dev, dtype=torch.long)], dim=1)
            t0 = time.perf_counter()
            with autocast_ctx:
                next_f = model.sample(ctx, actions, steps=args.steps)
            step_ms = (time.perf_counter() - t0) * 1000

            ctx = torch.cat([ctx[:, 1:], next_f], dim=1)
            step_count += 1
            last_button = btn_name

            if is_anchor_step(step_count, args.anchor):
                needs_reanchor = True

    if env is not None:
        env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
