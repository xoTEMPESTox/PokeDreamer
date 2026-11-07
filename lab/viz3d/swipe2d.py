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
) -> np.ndarray:
    """Composite a single 1920x1080 2D swipe frame in composer play layout with labels."""
    # Background canvas: (15, 20, 25) in BGR = (25, 20, 15)
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

    # Draw two-tier labels under the frame per COPY_REVIEW.md §4
    # Headline at y = 800, sub-line at y = 835
    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Use standard DejaVu font if available, else default
    try:
        font_headline = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        try:
            font_headline = ImageFont.truetype("arialbd.ttf", 24)
            font_sub = ImageFont.truetype("arial.ttf", 17)
        except Exception:
            font_headline = ImageFont.load_default()
            font_sub = ImageFont.load_default()

    col_white = (240, 245, 250)
    col_grey = (148, 163, 184)
    col_dim = (71, 85, 105)

    # Headline text
    text_headline = "The real screen, restored by two small networks"
    bbox_h = draw.textbbox((0, 0), text_headline, font=font_headline)
    text_w_h = bbox_h[2] - bbox_h[0]
    draw.text(((frame_w - text_w_h) // 2, 795), text_headline, fill=col_white, font=font_headline)

    # Three-part technical sub-line
    p1 = "72x80 input"
    p2 = "detail: 91% of held-out frames pixel-exact"
    p3 = "colour: 99.27% of pixels right"
    sub_text = f"{p1}  ·  {p2}  ·  {p3}"

    bbox_s = draw.textbbox((0, 0), sub_text, font=font_sub)
    text_w_s = bbox_s[2] - bbox_s[0]
    draw.text(((frame_w - text_w_s) // 2, 832), sub_text, fill=col_grey, font=font_sub)

    out_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return out_bgr


def build_all_2d_swipe_frames(
    out_dir: str | Path = "work/viz3d/v2",
) -> Tuple[Path, Path, float]:
    """Generate all 36 frames (255-290), swipe_strip.png, and report step-75 agreement."""
    out_path = Path(out_dir).resolve()
    frames_2d_dir = out_path / "frames_2d"
    frames_2d_dir.mkdir(parents=True, exist_ok=True)

    raw_bgr, crisp_bgr, color_bgr, agreement_pct = generate_swipe_panels()

    # 1. Save swipe_strip.png: raw | crisp | colour side by side
    strip = np.hstack([raw_bgr, crisp_bgr, color_bgr])
    strip_path = out_path / "swipe_strip.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"Saved swipe strip to {strip_path}")

    # 2. Generate 36 frames for 255..290 (1.2 s)
    # Origin X = 640, frame width 640 -> sweeps from x = 640 to 1280
    # Over 0.9 s (27 frames, 255..281):
    # Edge A: frames 255..275 (sweep over 20 frames)
    # Edge B: frames 263..283 (follows 8 frames behind)
    # Hold at full colour for frames 284..290 (7 frames)
    origin_x = 640
    disp_w = 640
    total_frames = 36  # frames 255 to 290 inclusive

    for i, frame_idx in enumerate(range(255, 291)):
        # Calculate edge A position
        if i <= 20:
            t_a = i / 20.0
            edge_a = origin_x + t_a * disp_w
        else:
            edge_a = origin_x + disp_w + 10.0

        # Calculate edge B position (starts at frame index 8 = frame 263)
        if i < 8:
            edge_b = origin_x - 10.0
        elif i <= 28:
            t_b = (i - 8) / 20.0
            edge_b = origin_x + t_b * disp_w
        else:
            edge_b = origin_x + disp_w + 10.0

        act_labels = (True, i >= 2, i >= 8)

        frame_bgr = composite_swipe_frame(
            raw_bgr, crisp_bgr, color_bgr,
            edge_a_x=edge_a, edge_b_x=edge_b,
            active_labels=act_labels,
        )

        out_frame_file = frames_2d_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out_frame_file), frame_bgr)

    print(f"Generated {total_frames} 2D swipe frames in {frames_2d_dir}")
    print(f"Step 75 native-shade agreement: {agreement_pct:.2f}%")
    return frames_2d_dir, strip_path, agreement_pct


if __name__ == "__main__":
    build_all_2d_swipe_frames()
