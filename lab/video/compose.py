"""Storyboard-driven video composer for PokeDreamer.

Generates 1080p 30fps video from a storyboard JSON, trajectory color/greyscale data,
and world-model dream frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canvas specifications
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
DEFAULT_FPS = 30
BG_COLOR = (15, 20, 25)  # #0f1419 very dark blue-grey in RGB
BG_COLOR_BGR = (25, 20, 15)


def calculate_segment_frames(storyboard: List[Dict[str, Any]], fps: int = DEFAULT_FPS) -> List[int]:
    """Computes output frame count for each segment from its seconds."""
    return [int(round(s["seconds"] * fps)) for s in storyboard]


def compute_step_indices(
    start: int,
    end: int,
    num_frames: int,
    speed: Any = "fit",
    ranges: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """Computes array of step indices of length num_frames for a segment.

    If ranges is provided (e.g. [[s1, e1], [s2, e2]]), steps are drawn across those ranges.
    Otherwise, steps are drawn from start to end (inclusive).
    """
    if ranges is not None and len(ranges) > 0:
        step_list: List[int] = []
        for r in ranges:
            step_list.extend(range(r[0], r[1] + 1))
        steps = np.array(step_list, dtype=int)
    else:
        if start == end:
            return np.full(num_frames, start, dtype=int)
        steps = np.arange(start, end + 1, dtype=int)

    if len(steps) == 0:
        return np.zeros(num_frames, dtype=int)

    if speed == "fit" or str(speed).lower() == "fit":
        indices = np.linspace(0, len(steps) - 1, num_frames).round().astype(int)
        return steps[indices]
    else:
        spd = float(speed)
        raw_idx = (np.arange(num_frames) * spd).round().astype(int)
        return steps[np.clip(raw_idx, 0, len(steps) - 1)]


def get_hud_badge(
    in_battle: bool, by: str
) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Returns (badge_text, bg_rgb, border_rgb, text_rgb) based on step flags.

    Priority:
    1. in_battles -> "rules: battle" (amber)
    2. by == "bfs" -> "path-finding fallback" (grey)
    3. otherwise -> "world-model policy" (green)
    """
    if in_battle:
        return "rules: battle", (75, 45, 12), (180, 110, 25), (252, 211, 77)
    elif str(by).lower() == "bfs":
        return "path-finding fallback", (45, 55, 68), (100, 116, 139), (203, 213, 225)
    else:
        return "world-model policy", (18, 65, 36), (46, 125, 50), (110, 231, 145)


class FontManager:
    """Loads fonts bundled in matplotlib's data path."""

    def __init__(self):
        import matplotlib

        mpl_data = Path(matplotlib.get_data_path())
        fonts_dir = mpl_data / "fonts" / "ttf"
        self.regular_path = str(fonts_dir / "DejaVuSans.ttf")
        self.bold_path = str(fonts_dir / "DejaVuSans-Bold.ttf")
        if not os.path.exists(self.regular_path):
            raise FileNotFoundError(f"Matplotlib font not found: {self.regular_path}")
        self.cache: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self.cache:
            path = self.bold_path if bold else self.regular_path
            self.cache[key] = ImageFont.truetype(path, size)
        return self.cache[key]


def get_dream_colorizer(device: str = "cpu") -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Returns a colorizer callable (N, 72, 80) -> (N, 144, 160, 3) and a status string.

    Automatically uses learned colorizer if weights exist, otherwise falls back to upsampler greyscale.
    """
    model_path = Path("work/color/learned/model.pt")
    upsampler_path = Path("work/upsampler/upsampler.pt")

    try:
        from lab.color.learned import colorize72, load_colorizer, load_palette
        from lab.upsampler import load_upsampler

        if model_path.exists() and upsampler_path.exists():
            model = load_colorizer(str(model_path), device=device)
            upsampler = load_upsampler(str(upsampler_path), device=device)
            palette = load_palette()

            def _learned_fn(frames72: np.ndarray) -> np.ndarray:
                return colorize72(frames72, model, upsampler, palette, device=device, batch=32)

            return _learned_fn, "Stage-2 learned colorizer (work/color/learned/model.pt)"
    except Exception as e:
        print(f"Notice: could not load Stage-2 colorizer ({e})")

    try:
        from lab.upsampler import load_upsampler, upsample

        if upsampler_path.exists():
            upsampler = load_upsampler(str(upsampler_path), device=device)

            def _grey_fn(frames72: np.ndarray) -> np.ndarray:
                shades = upsample(upsampler, frames72, device=device, batch=64)
                return np.repeat(shades[..., None], 3, axis=-1)

            return _grey_fn, "Greyscale upsampler (work/upsampler/upsampler.pt)"
    except Exception as e:
        print(f"Notice: could not load upsampler ({e})")

    # Simple nearest neighbor fallback
    def _fallback_fn(frames72: np.ndarray) -> np.ndarray:
        out = []
        for f in frames72:
            resized = cv2.resize(f, (160, 144), interpolation=cv2.INTER_NEAREST)
            out.append(np.repeat(resized[..., None], 3, axis=-1))
        return np.array(out, dtype=np.uint8)

    return _fallback_fn, "Nearest-neighbor fallback"


def draw_hud_card(
    draw: ImageDraw.ImageDraw,
    fonts: FontManager,
    box: Tuple[int, int, int, int],
    map_name: str,
    goal: str,
    leg: str,
    in_battle: bool,
    by: str,
    step: int,
    progress_text: Optional[str] = None,
):
    """Draws left-column HUD card with status, goals, badge, and counter."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(20, 26, 35), outline=(45, 58, 75), width=2)

    # Title / header
    draw.text((x0 + 30, y0 + 26), "POKEDREAMER STATUS", font=fonts.get(14, bold=True), fill=(100, 116, 139))
    draw.line([(x0 + 30, y0 + 52), (x1 - 30, y0 + 52)], fill=(34, 45, 58), width=1)

    curr_y = y0 + 68

    # 1. Map name
    draw.text((x0 + 30, curr_y), "CURRENT MAP", font=fonts.get(13), fill=(148, 163, 184))
    draw.text((x0 + 30, curr_y + 22), map_name or "—", font=fonts.get(22, bold=True), fill=(248, 250, 252))
    curr_y += 66

    # 2. System 2 Goal
    draw.text((x0 + 30, curr_y), "SYSTEM 2 GOAL", font=fonts.get(13), fill=(148, 163, 184))
    draw.text((x0 + 30, curr_y + 22), str(goal).upper() if goal else "—", font=fonts.get(22, bold=True), fill=(248, 250, 252))
    curr_y += 66

    # 3. Active Leg
    draw.text((x0 + 30, curr_y), "ACTIVE NAVIGATION LEG", font=fonts.get(13), fill=(148, 163, 184))
    leg_str = leg if leg else "—"
    draw.text((x0 + 30, curr_y + 22), leg_str, font=fonts.get(20, bold=True), fill=(147, 197, 253))
    curr_y += 66

    # 4. Control Badge
    draw.text((x0 + 30, curr_y), "CONTROL MODE", font=fonts.get(13), fill=(148, 163, 184))
    badge_text, badge_bg, badge_border, badge_fg = get_hud_badge(in_battle, by)
    badge_font = fonts.get(18, bold=True)
    tb = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw, bh = tb[2] - tb[0], tb[3] - tb[1]
    pill_w = bw + 32
    pill_h = bh + 16
    pill_y = curr_y + 22
    draw.rounded_rectangle(
        [x0 + 30, pill_y, x0 + 30 + pill_w, pill_y + pill_h],
        radius=8,
        fill=badge_bg,
        outline=badge_border,
        width=2,
    )
    draw.text((x0 + 30 + pill_w // 2, pill_y + pill_h // 2), badge_text, font=badge_font, fill=badge_fg, anchor="mm")
    curr_y += 68

    # 5. Press Counter (step + 1)
    draw.text((x0 + 30, curr_y), "PRESSES", font=fonts.get(13), fill=(148, 163, 184))
    draw.text((x0 + 30, curr_y + 22), f"{step + 1:,}", font=fonts.get(30, bold=True), fill=(56, 189, 248))
    curr_y += 66

    # 6. Optional level progress (for montage)
    if progress_text:
        draw.text((x0 + 30, curr_y), "PROGRESS", font=fonts.get(13), fill=(148, 163, 184))
        draw.text((x0 + 30, curr_y + 22), progress_text, font=fonts.get(20, bold=True), fill=(250, 204, 21))


def render_title_frame(
    bg_frame: Optional[np.ndarray],
    text_lines: List[str],
    caption: str,
    fonts: FontManager,
    fade_alpha: float = 1.0,
) -> np.ndarray:
    """Renders a title frame with centered text lines and Viridian background."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)

    if bg_frame is not None:
        scaled = cv2.resize(bg_frame, (800, 720), interpolation=cv2.INTER_NEAREST)
        tinted = (scaled.astype(np.float32) * 0.28).astype(np.uint8)
        paste_x = (CANVAS_WIDTH - 800) // 2
        paste_y = (CANVAS_HEIGHT - 720) // 2
        img.paste(Image.fromarray(tinted), (paste_x, paste_y))

    draw = ImageDraw.Draw(img)

    # Decorative frame outline
    cx, cy = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2
    card_w, card_h = 1040, 420
    draw.rounded_rectangle(
        [cx - card_w // 2, cy - card_h // 2, cx + card_w // 2, cy + card_h // 2],
        radius=18,
        fill=(20, 26, 35),
        outline=(50, 65, 85),
        width=2,
    )

    # Text lines
    title = text_lines[0] if len(text_lines) > 0 else "PokeDreamer"
    draw.text((cx, cy - 90), title, font=fonts.get(56, bold=True), fill=(255, 255, 255), anchor="mm")
    draw.line([(cx - 250, cy - 40), (cx + 250, cy - 40)], fill=(45, 58, 75), width=1)

    if len(text_lines) > 1:
        draw.text((cx, cy + 15), text_lines[1], font=fonts.get(28, bold=True), fill=(226, 232, 240), anchor="mm")
    if len(text_lines) > 2:
        draw.text((cx, cy + 65), text_lines[2], font=fonts.get(28, bold=True), fill=(226, 232, 240), anchor="mm")

    # If caption is distinct from text lines, show it; otherwise do not duplicate
    combined_text = " ".join(text_lines)
    if caption and caption not in combined_text:
        draw.text((cx, CANVAS_HEIGHT - 100), caption, font=fonts.get(24), fill=(160, 175, 195), anchor="mm")

    res = np.asarray(img)
    if fade_alpha < 0.999:
        res = (res.astype(np.float32) * max(0.0, min(1.0, fade_alpha))).astype(np.uint8)
    return res


def render_end_frame(
    text_lines: List[str],
    caption: str,
    fonts: FontManager,
    fade_alpha: float = 1.0,
) -> np.ndarray:
    """Renders the end card with the exact results numbers."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    cx, cy = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2

    # Outer card
    card_w, card_h = 1400, 560
    draw.rounded_rectangle(
        [cx - card_w // 2, cy - card_h // 2, cx + card_w // 2, cy + card_h // 2],
        radius=18,
        fill=(20, 26, 35),
        outline=(50, 65, 85),
        width=2,
    )

    # Title header
    draw.text((cx, cy - 210), "POKEDREAMER // EXPERIMENT RESULTS", font=fonts.get(20, bold=True), fill=(100, 116, 139), anchor="mm")
    draw.line([(cx - card_w // 2 + 50, cy - 180), (cx + card_w // 2 - 50, cy - 180)], fill=(34, 45, 58), width=1)

    colors = [
        (250, 204, 21),   # Gold for badges
        (74, 222, 128),   # Green for learned navigation
        (56, 189, 248),   # Cyan for 0 real-game steps
    ]

    base_y = cy - 110
    for idx, line in enumerate(text_lines):
        color = colors[idx % len(colors)]
        size = 34
        while size > 18 and draw.textlength(line, font=fonts.get(size, bold=True)) > card_w - 100:
            size -= 2                                     # shrink a long line until it fits inside the card
        draw.text((cx, base_y + idx * 85), line, font=fonts.get(size, bold=True), fill=color, anchor="mm")

    # Footer note
    draw.text((cx, cy + 220), "Autonomous Quest // Boulder Badge Achieved", font=fonts.get(18), fill=(148, 163, 184), anchor="mm")

    res = np.asarray(img)
    if fade_alpha < 0.999:
        res = (res.astype(np.float32) * max(0.0, min(1.0, fade_alpha))).astype(np.uint8)
    return res


def render_play_frame(
    color_frame: np.ndarray,
    map_name: str,
    goal: str,
    leg: str,
    in_battle: bool,
    by: str,
    step: int,
    caption: str,
    fonts: FontManager,
    scale: int = 4,
    progress_text: Optional[str] = None,
) -> np.ndarray:
    """Renders a standard play or montage frame with Game Boy canvas and left HUD."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fw = 160 * scale
    fh = 144 * scale

    if scale == 4:
        fx = 640
        fy = 190
        hud_box = (100, 190, 560, 190 + fh)
    else:  # scale == 5
        fx = 760
        fy = 150
        hud_box = (80, 150, 680, 150 + fh)

    # 1. Scale and paste game frame
    scaled = cv2.resize(color_frame, (fw, fh), interpolation=cv2.INTER_NEAREST)
    img.paste(Image.fromarray(scaled), (fx, fy))

    # 2. Rounded border around game frame
    draw.rounded_rectangle([fx - 3, fy - 3, fx + fw + 3, fy + fh + 3], radius=10, outline=(60, 75, 95), width=2)

    # 3. HUD on the left column
    draw_hud_card(
        draw=draw,
        fonts=fonts,
        box=hud_box,
        map_name=map_name,
        goal=goal,
        leg=leg,
        in_battle=in_battle,
        by=by,
        step=step,
        progress_text=progress_text,
    )

    # 4. Caption centred under the frame
    if caption:
        cap_x = fx + fw // 2
        cap_y = fy + fh + 60
        draw.text((cap_x, cap_y), caption, font=fonts.get(24, bold=True), fill=(241, 245, 249), anchor="mm")

    return np.asarray(img)


def render_split_frame(
    real_frame: np.ndarray,
    dream_frame: Optional[np.ndarray],
    caption: str,
    fonts: FontManager,
    is_anchor_flashing: bool = False,
) -> np.ndarray:
    """Renders side-by-side real vs dream frame comparison (each 4x)."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fw, fh = 640, 576
    lx, ly = 280, 190
    rx, ry = 1000, 190

    # 1. Labels above frames
    draw.text((lx + fw // 2, ly - 40), "REAL GAME", font=fonts.get(26, bold=True), fill=(226, 232, 240), anchor="mm")
    draw.text((rx + fw // 2, ly - 40), "WORLD MODEL'S IMAGINATION", font=fonts.get(26, bold=True), fill=(96, 165, 250), anchor="mm")

    # 2. Real frame (left)
    scaled_real = cv2.resize(real_frame, (fw, fh), interpolation=cv2.INTER_NEAREST)
    img.paste(Image.fromarray(scaled_real), (lx, ly))
    draw.rounded_rectangle([lx - 3, ly - 3, lx + fw + 3, ly + fh + 3], radius=10, outline=(60, 75, 95), width=2)

    # 3. Dream frame (right)
    if dream_frame is not None:
        scaled_dream = cv2.resize(dream_frame, (fw, fh), interpolation=cv2.INTER_NEAREST)
        img.paste(Image.fromarray(scaled_dream), (rx, ry))
        draw.rounded_rectangle([rx - 3, ry - 3, rx + fw + 3, ry + fh + 3], radius=10, outline=(60, 75, 95), width=2)

        # Flashing re-anchor tag
        if is_anchor_flashing:
            tag_text = "re-anchored on 4 real frames"
            tag_font = fonts.get(18, bold=True)
            tb = draw.textbbox((0, 0), tag_text, font=tag_font)
            pw, ph = (tb[2] - tb[0]) + 28, (tb[3] - tb[1]) + 14
            px, py = rx + fw - pw - 20, ry + 20
            draw.rounded_rectangle([px, py, px + pw, py + ph], radius=8, fill=(15, 35, 65), outline=(56, 189, 248), width=2)
            draw.text((px + pw // 2, py + ph // 2), tag_text, font=tag_font, fill=(125, 211, 252), anchor="mm")
    else:
        # Placeholder when dream frames pending
        draw.rounded_rectangle([rx, ry, rx + fw, ry + fh], radius=10, fill=(18, 24, 32), outline=(45, 58, 72), width=2)
        draw.text(
            (rx + fw // 2, ry + fh // 2),
            "world model frames pending",
            font=fonts.get(24, bold=True),
            fill=(148, 163, 184),
            anchor="mm",
        )

    # 4. Caption centred under both frames
    if caption:
        draw.text((CANVAS_WIDTH // 2, ly + fh + 60), caption, font=fonts.get(24, bold=True), fill=(241, 245, 249), anchor="mm")

    return np.asarray(img)


def compose_video(
    storyboard_path: str | Path,
    out_path: str | Path,
    scale: int = 4,
    traj_color_path: str | Path = "work/video/traj/run_seed_100_color.npz",
    traj_grey_path: str | Path = "work/video/traj/run_seed_100.npz",
    dream_path: str | Path = "work/video/dream_4_8718_k10.npz",
    quest_json_path: str | Path = "work/video/quest_s100/run_seed_100.json",
    fps: int = DEFAULT_FPS,
    stills_dir: Optional[str | Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Renders the complete 60s video according to the storyboard specification."""
    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)

    # Load trajectory color data
    print(f"Loading color trajectory: {traj_color_path}")
    color_npz = np.load(traj_color_path)
    colors = color_npz["color"]
    map_ids = color_npz["map_ids"]
    goals = color_npz["goals"]
    in_battles = color_npz["in_battles"]
    legs = color_npz["leg"]
    bys = color_npz["by"]
    total_traj_steps = len(colors)

    # Load map name resolver
    try:
        from lab.dwm.eval_rollouts import get_map_name
    except ImportError:
        def get_map_name(mid: int) -> str:
            return f"Map {mid}"

    # Load quest timeline for level lookup if available
    timeline_levels = []
    if os.path.exists(quest_json_path):
        with open(quest_json_path, "r") as f:
            q_data = json.load(f)
            timeline_levels = [(e["presses"], e.get("level", 8)) for e in q_data.get("timeline", [])]

    def get_level(step_idx: int) -> int:
        lvl = 8
        for p, l in timeline_levels:
            if step_idx >= p:
                lvl = l
        return lvl

    # Check dream frames
    dream_file = Path(dream_path)
    has_dream = dream_file.exists()
    dream_frames_data = None
    dream_anchors_data = None
    dream_start = 4

    dream_colorizer_fn = None
    colorizer_name = "None"
    if has_dream:
        print(f"Loading world model dream frames: {dream_path}")
        d_npz = np.load(dream_path)
        dream_frames_data = d_npz["dream"]
        dream_anchors_data = d_npz["anchor"]
        dream_start = int(d_npz.get("start", 4))
        dream_colorizer_fn, colorizer_name = get_dream_colorizer(device=device)
        print(f"Using colorizer for dream: {colorizer_name}")
    else:
        print("World model dream frames pending; rendered placeholder for split segment.")

    # Calculate segment frame allocations
    seg_frames = calculate_segment_frames(storyboard, fps=fps)
    total_frames = sum(seg_frames)
    total_duration = total_frames / fps

    print("\n" + "=" * 80)
    print(f"{'SEGMENT TABLE':^80}")
    print("=" * 80)
    print(f"{'#':<3} {'Type':<10} {'Start':<7} {'End':<7} {'Speed':<8} {'Frames':<8} {'Sec':<6} {'Caption'}")
    print("-" * 80)
    for i, s in enumerate(storyboard):
        cap = s.get("caption", "")
        if len(cap) > 30:
            cap = cap[:27] + "..."
        print(
            f"{i+1:<3} {s['type']:<10} {s.get('start', 0):<7} {s.get('end', 0):<7} "
            f"{str(s.get('speed', 'fit')):<8} {seg_frames[i]:<8} {s['seconds']:<6.1f} {cap}"
        )
    print("-" * 80)
    print(f"Total video duration: {total_duration:.2f} s ({total_frames} frames @ {fps} fps)")
    print("=" * 80 + "\n")

    fonts = FontManager()

    # Pre-render frames segment by segment
    rendered_segments: List[List[np.ndarray]] = []
    stills_to_save: Dict[str, np.ndarray] = {}

    for seg_idx, seg in enumerate(storyboard):
        seg_type = seg["type"]
        n_frames = seg_frames[seg_idx]
        speed = seg.get("speed", "fit")
        ranges = seg.get("ranges", None)
        start_step = seg.get("start", 0)
        end_step = seg.get("end", 0)
        caption = seg.get("caption", "")
        text_lines = seg.get("text", [])

        steps = compute_step_indices(start_step, end_step, n_frames, speed=speed, ranges=ranges)
        seg_out: List[np.ndarray] = []

        # Colorize dream chunk if split segment
        dream_colorized = None
        if seg_type == "split" and has_dream and dream_frames_data is not None:
            raw_dream_list = []
            for st in steps:
                d_idx = st - dream_start
                if 0 <= d_idx < len(dream_frames_data):
                    raw_dream_list.append(dream_frames_data[d_idx])
                else:
                    raw_dream_list.append(np.zeros((72, 80), dtype=np.uint8))
            raw_dream_arr = np.array(raw_dream_list, dtype=np.uint8)
            assert dream_colorizer_fn is not None
            dream_colorized = dream_colorizer_fn(raw_dream_arr)

        anchor_flash_countdown = 0

        for f_idx in range(n_frames):
            st = steps[f_idx]
            prev_st = steps[max(0, f_idx - 1)]
            st_clamped = min(st, total_traj_steps - 1)
            cf = colors[st_clamped]
            m_id = int(map_ids[st_clamped])
            m_name = get_map_name(m_id)
            g_str = str(goals[st_clamped])
            l_str = str(legs[st_clamped])
            bat = bool(in_battles[st_clamped])
            by_str = str(bys[st_clamped])

            if seg_type == "title":
                fade = 1.0
                if f_idx < 15:
                    fade = (f_idx + 1) / 15.0
                elif f_idx >= n_frames - 15:
                    fade = (n_frames - f_idx) / 15.0
                frame = render_title_frame(cf, text_lines, caption, fonts, fade_alpha=fade)

            elif seg_type == "end":
                fade = 1.0
                if f_idx < 15:
                    fade = (f_idx + 1) / 15.0
                elif f_idx >= n_frames - 15:
                    fade = (n_frames - f_idx) / 15.0
                frame = render_end_frame(text_lines, caption, fonts, fade_alpha=fade)

            elif seg_type == "split":
                # Check if any step in [prev_st, st] was an anchor step
                if has_dream and dream_anchors_data is not None:
                    step_range_to_check = range(min(prev_st, st), max(prev_st, st) + 1)
                    hit = any(
                        dream_anchors_data[s - dream_start]
                        for s in step_range_to_check
                        if 0 <= s - dream_start < len(dream_anchors_data)
                    )
                    if hit:
                        anchor_flash_countdown = 15  # 0.5 s @ 30 fps

                dream_img = dream_colorized[f_idx] if dream_colorized is not None else None
                frame = render_split_frame(
                    real_frame=cf,
                    dream_frame=dream_img,
                    caption=caption,
                    fonts=fonts,
                    is_anchor_flashing=(anchor_flash_countdown > 0),
                )
                if anchor_flash_countdown > 0:
                    anchor_flash_countdown -= 1

            elif seg_type == "montage":
                lvl = get_level(st_clamped)
                prog_str = f"Level {lvl} -> 14"
                frame = render_play_frame(
                    color_frame=cf,
                    map_name=m_name,
                    goal=g_str,
                    leg=l_str,
                    in_battle=bat,
                    by=by_str,
                    step=st_clamped,
                    caption=caption,
                    fonts=fonts,
                    scale=scale,
                    progress_text=prog_str,
                )

            else:  # "play"
                frame = render_play_frame(
                    color_frame=cf,
                    map_name=m_name,
                    goal=g_str,
                    leg=l_str,
                    in_battle=bat,
                    by=by_str,
                    step=st_clamped,
                    caption=caption,
                    fonts=fonts,
                    scale=scale,
                )

            seg_out.append(frame)

        rendered_segments.append(seg_out)

        # Save representative still for this segment
        mid_idx = n_frames // 2
        still_key = f"segment_{seg_idx+1:02d}_{seg_type}"
        stills_to_save[still_key] = seg_out[mid_idx]

    # Save stills if directory provided
    if stills_dir:
        stills_path = Path(stills_dir)
        stills_path.mkdir(parents=True, exist_ok=True)
        for k, s_frame in stills_to_save.items():
            out_img = Image.fromarray(s_frame)
            p = stills_path / f"{k}.png"
            out_img.save(p)
            print(f"Saved still: {p}")

    # Assemble video with 0.3s crossfades between segments
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (CANVAS_WIDTH, CANVAS_HEIGHT))

    crossfade_frames = int(round(0.3 * fps))  # 9 frames
    half_fade = crossfade_frames // 2         # 4 frames

    print(f"Encoding video with {crossfade_frames}-frame (0.3s) crossfades...")
    t_start = time.time()
    frames_written = 0

    num_segs = len(rendered_segments)
    for i in range(num_segs):
        cur_seg = rendered_segments[i]
        n_cur = len(cur_seg)

        for j in range(n_cur):
            is_crossfade = False
            blended_frame = None

            if i < num_segs - 1 and j >= n_cur - half_fade:
                # In fade-out portion of boundary
                next_seg = rendered_segments[i + 1]
                offset = j - (n_cur - half_fade)
                alpha = (offset + 1) / float(crossfade_frames + 1)
                blended_frame = cv2.addWeighted(cur_seg[j], 1.0 - alpha, next_seg[0], alpha, 0)
                is_crossfade = True
            elif i > 0 and j < (crossfade_frames - half_fade):
                # In fade-in portion of boundary
                prev_seg = rendered_segments[i - 1]
                offset = half_fade + j
                alpha = (offset + 1) / float(crossfade_frames + 1)
                blended_frame = cv2.addWeighted(prev_seg[-1], 1.0 - alpha, cur_seg[j], alpha, 0)
                is_crossfade = True

            frame_to_write = blended_frame if is_crossfade else cur_seg[j]
            bgr = cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frames_written += 1

    writer.release()
    t_elapsed = time.time() - t_start
    print(f"Successfully generated {out_file}: {frames_written} frames in {t_elapsed:.2f}s ({frames_written/t_elapsed:.1f} fps)")

    return {
        "out_file": str(out_file),
        "total_frames": frames_written,
        "total_duration": frames_written / fps,
        "has_dream": has_dream,
        "colorizer_used": colorizer_name,
        "segments": storyboard,
        "seg_frames": seg_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Compose storyboard video for PokeDreamer")
    parser.add_argument("--storyboard", default="lab/video/storyboard.json", help="Path to storyboard JSON")
    parser.add_argument("--out", default="work/video/draft_v1.mp4", help="Path to output MP4")
    parser.add_argument("--scale", type=int, default=4, choices=[4, 5], help="Game Boy frame scale factor")
    parser.add_argument("--traj-color", default="work/video/traj/run_seed_100_color.npz")
    parser.add_argument("--traj-grey", default="work/video/traj/run_seed_100.npz")
    parser.add_argument("--dream", default="work/video/dream_4_8718_k10.npz")
    parser.add_argument("--stills-dir", default="work/video/draft_v1_stills")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    compose_video(
        storyboard_path=args.storyboard,
        out_path=args.out,
        scale=args.scale,
        traj_color_path=args.traj_color,
        traj_grey_path=args.traj_grey,
        dream_path=args.dream,
        fps=args.fps,
        stills_dir=args.stills_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
