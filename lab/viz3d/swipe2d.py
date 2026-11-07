"""CPU 2D swipe generator using real neural model outputs for frames 255-290."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lab.upsampler import load_upsampler, upsample
from lab.video.compose import get_dream_colorizer
from lab.viz3d.classes import frame_classes


def generate_swipe_panels(
    traj_path: str | Path = "work/video/traj/run_seed_100.npz",
    step: int = 75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate raw, crisp, and color 640x576 BGR panels and measure step-75 shade agreement."""
    traj = np.load(traj_path)
    frames = traj["frames"]
    raw72 = frames[step]  # (72, 80) uint8

    # 1. Raw panel: 2x bilinear upscale to 144x160, then 4x nearest to 576x640
    raw_144 = cv2.resize(raw72, (160, 144), interpolation=cv2.INTER_LINEAR)
    raw_display = cv2.resize(raw_144, (640, 576), interpolation=cv2.INTER_NEAREST)
    raw_bgr = cv2.cvtColor(raw_display, cv2.COLOR_GRAY2BGR)

    # 2. Crisp panel: upsampler output (144, 160) 4-shade grey -> 576x640 nearest
    upsampler = load_upsampler("work/upsampler/upsampler.pt", device="cpu")
    pred_sh = upsample(upsampler, raw72[None], device="cpu")[0]  # (144, 160)
    crisp_display = cv2.resize(pred_sh, (640, 576), interpolation=cv2.INTER_NEAREST)
    crisp_bgr = cv2.cvtColor(crisp_display, cv2.COLOR_GRAY2BGR)

    # 3. Colour panel: dream colorizer output (144, 160, 3) uint8 -> 576x640 nearest
    colorizer, _ = get_dream_colorizer(device="cpu")
    color_out = colorizer(raw72[None])[0]  # (144, 160, 3) RGB
    color_bgr = cv2.cvtColor(color_out, cv2.COLOR_RGB2BGR)
    color_display = cv2.resize(color_bgr, (640, 576), interpolation=cv2.INTER_NEAREST)

    # 4. Measure agreement with native shades from run_seed_100_color.npz
    agreement_pct = 0.0
    color_npz = Path("work/video/traj/run_seed_100_color.npz")
    if color_npz.exists():
        c_data = np.load(color_npz)
        native_rgb = c_data["color"][step]
        _, native_sh = frame_classes(native_rgb)
        mapping = {255: 0, 153: 1, 85: 2, 0: 3}
        pred_indices = np.zeros_like(pred_sh, dtype=np.uint8)
        for v, k in mapping.items():
            pred_indices[pred_sh == v] = k
        agreement_pct = float(np.mean(pred_indices == native_sh) * 100.0)

    return raw_bgr, crisp_bgr, color_display, agreement_pct


def composite_swipe_frame(
    raw_bgr: np.ndarray,
    crisp_bgr: np.ndarray,
    color_bgr: np.ndarray,
    edge_a_x: float,
    edge_b_x: float,
    frame_w: int = 1920,
    frame_h: int = 1080,
    origin_x: int = 640,
    origin_y: int = 190,
    disp_w: int = 640,
    disp_h: int = 576,
    active_labels: Tuple[bool, bool, bool] = (True, True, True),
    bg_image: np.ndarray | None = None,
) -> np.ndarray:
    """Composite a single 1920x1080 2D swipe frame in composer play layout (no text overlay)."""
    # Background canvas: blurred 3D frame if provided, else (25, 20, 15) in BGR
    if bg_image is not None:
        if bg_image.shape[:2] != (frame_h, frame_w):
            canvas = cv2.resize(bg_image, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = bg_image.copy()
    else:
        canvas = np.full((frame_h, frame_w, 3), (25, 20, 15), dtype=np.uint8)

    # Build the 640x576 composited game screen
    # Local coordinate x in [0, disp_w] corresponds to canvas X in [origin_x, origin_x + disp_w]
    screen = raw_bgr.copy()
    loc_a = int(round(edge_a_x - origin_x))
    loc_b = int(round(edge_b_x - origin_x))

    loc_a = max(0, min(disp_w, loc_a))
    loc_b = max(0, min(disp_w, loc_b))

    # Crisp replaces raw left of edge A
    if loc_a > 0:
        screen[:, :loc_a] = crisp_bgr[:, :loc_a]
    # Colour replaces crisp left of edge B
    if loc_b > 0:
        screen[:, :loc_b] = color_bgr[:, :loc_b]

    # Draw cyan hairline and glow for Edge A
    cyan_bgr = (252, 211, 125)  # Light cyan RGB (125, 211, 252)
    if 0 < loc_a < disp_w:
        glow_radius = 6
        for dx in range(-glow_radius, glow_radius + 1):
            gx = loc_a + dx
            if 0 <= gx < disp_w:
                weight = math.exp(-((dx / 2.5) ** 2)) * 0.70
                screen[:, gx] = cv2.addWeighted(
                    screen[:, gx], 1.0 - weight,
                    np.full_like(screen[:, gx], cyan_bgr), weight, 0
                )
        screen[:, loc_a] = cyan_bgr

    # Draw cyan hairline and glow for Edge B
    if 0 < loc_b < disp_w:
        glow_radius = 6
        for dx in range(-glow_radius, glow_radius + 1):
            gx = loc_b + dx
            if 0 <= gx < disp_w:
                weight = math.exp(-((dx / 2.5) ** 2)) * 0.70
                screen[:, gx] = cv2.addWeighted(
                    screen[:, gx], 1.0 - weight,
                    np.full_like(screen[:, gx], cyan_bgr), weight, 0
                )
        screen[:, loc_b] = cyan_bgr

    # Place screen into canvas
    canvas[origin_y : origin_y + disp_h, origin_x : origin_x + disp_w] = screen

    # Rounded border around game frame: (60, 75, 95) BGR = (95, 75, 60)
    border_bgr = (95, 75, 60)
    cv2.rectangle(
        canvas,
        (origin_x - 3, origin_y - 3),
        (origin_x + disp_w + 2, origin_y + disp_h + 2),
        border_bgr,
        thickness=2,
    )

    return canvas


def build_all_2d_swipe_frames(
    out_dir: str | Path = "work/viz3d/v2",
) -> Tuple[Path, Path, float]:
    """Generate all 90 frames for frames 351-440 (3.0s), swipe_strip.png, and report agreement."""
    out_path = Path(out_dir).resolve()
    frames_2d_dir = out_path / "frames_2d"
    frames_2d_dir.mkdir(parents=True, exist_ok=True)

    raw_bgr, crisp_bgr, color_bgr, agreement_pct = generate_swipe_panels()

    # 1. Save swipe_strip.png: raw | crisp | colour side by side
    strip = np.hstack([raw_bgr, crisp_bgr, color_bgr])
    strip_path = out_path / "swipe_strip.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"Saved swipe strip to {strip_path}")

    # Find nearest rendered 3D frame for blurred background
    bg_candidate = None
    for cand_path in [
        out_path / "frames" / "frame_0250.png",
        out_path / "frames" / "frame_0254.png",
        out_path / "checks_r4" / "r4_f280.png",
        out_path / "checks_r4" / "r4_f200.png",
        out_path.parent / "checks" / "frame_0270.png",
    ]:
        if cand_path.exists():
            bg_candidate = cv2.imread(str(cand_path))
            break

    darkened_bg = None
    if bg_candidate is not None:
        blurred = cv2.GaussianBlur(bg_candidate, (51, 51), 25)
        darkened_bg = np.clip(blurred.astype(np.float32) * 0.60, 0, 255).astype(np.uint8)

    origin_x = 640
    disp_w = 640

    # 2. Generate 90 frames for 351..440 (3.0 s)
    # Frames 351..380 (30 frames): 72x80 input screen
    # Frames 381..395 (15 frames): Edge A sweeps across screen
    # Frames 396..410 (15 frames): Hold on upscaler (crisp)
    # Frames 411..425 (15 frames): Edge B sweeps across screen
    # Frames 426..440 (15 frames): Hold on colorizer
    total_frames = 90
    for i, frame_idx in enumerate(range(351, 441)):
        if i < 30:
            edge_a = origin_x - 10.0
            edge_b = origin_x - 10.0
        elif i < 45:
            t_a = (i - 30) / 14.0
            edge_a = origin_x + t_a * disp_w
            edge_b = origin_x - 10.0
        elif i < 60:
            edge_a = origin_x + disp_w + 10.0
            edge_b = origin_x - 10.0
        elif i < 75:
            t_b = (i - 60) / 14.0
            edge_a = origin_x + disp_w + 10.0
            edge_b = origin_x + t_b * disp_w
        else:
            edge_a = origin_x + disp_w + 10.0
            edge_b = origin_x + disp_w + 10.0

        frame_bgr = composite_swipe_frame(
            raw_bgr, crisp_bgr, color_bgr,
            edge_a_x=edge_a, edge_b_x=edge_b,
            bg_image=darkened_bg,
        )
        out_frame_file = frames_2d_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_frame_file), frame_bgr)

    # Also keep legacy frames 255..290 for backwards compatibility
    for i, frame_idx in enumerate(range(255, 291)):
        t_edge = i / 35.0
        edge_a = origin_x + t_edge * disp_w
        edge_b = origin_x + max(0.0, (t_edge - 0.3) * disp_w)
        frame_bgr = composite_swipe_frame(
            raw_bgr, crisp_bgr, color_bgr,
            edge_a_x=edge_a, edge_b_x=edge_b,
            bg_image=darkened_bg,
        )
        cv2.imwrite(str(frames_2d_dir / f"frame_{frame_idx:04d}.png"), frame_bgr)

    print(f"Generated {total_frames} 2D swipe frames in {frames_2d_dir}")
    print(f"Step 75 native-shade agreement: {agreement_pct:.2f}%")
    return frames_2d_dir, strip_path, agreement_pct


if __name__ == "__main__":
    build_all_2d_swipe_frames()
