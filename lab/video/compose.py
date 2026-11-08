"""Storyboard-driven video composer for PokeDreamer.

Generates 1080p 30fps video from a storyboard JSON, trajectory color/greyscale data,
and world-model dream frames.
"""
from __future__ import annotations

import argparse
import json
import math
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
# Light palette for the website's light theme (info screens only; the video uses the dark one)
LIGHT_INFO = {"bg": (248, 250, 252), "panel": (255, 255, 255), "border": (203, 213, 225), "title": (15, 23, 42),
              "body": (51, 65, 85), "muted": (71, 85, 105), "faint": (100, 116, 139), "cyan": (8, 145, 178),
              "sky": (2, 132, 199), "gold": (217, 119, 6)}

# Play and Montage segment layout (6x Game Boy frame)
PLAY_SCALE = 6
GAME_W = 160 * PLAY_SCALE  # 960
GAME_H = 144 * PLAY_SCALE  # 864
PLAY_OUTER_MARGIN = 50     # outer margin on left and right (about 40-60 px)
PLAY_GAP = 40              # gap between status panel and game frame
PANEL_W = CANVAS_WIDTH - 2 * PLAY_OUTER_MARGIN - GAME_W - PLAY_GAP  # 820 px
PANEL_H = GAME_H           # 864 px
PLAY_FRAME_Y = 94          # top coordinate for game frame and status panel
HEADING_Y = 47             # vertical center for top heading band
HEADING_FONT_SIZE = 48     # heading font size (bold, 44-52 px)
HEADING_ICON_SIZE = 42     # pokeball icon size in heading band

# Split segment layout (5x Game Boy frames side-by-side)
SPLIT_SCALE = 5
SPLIT_FRAME_W = 160 * SPLIT_SCALE  # 800
SPLIT_FRAME_H = 144 * SPLIT_SCALE  # 720
SPLIT_GAP = 80                     # gap between real and dream frames
SPLIT_MARGIN_X = (CANVAS_WIDTH - 2 * SPLIT_FRAME_W - SPLIT_GAP) // 2  # 120 px
SPLIT_FRAME_Y = 150                # top coordinate for frames

# Mappings for status panel goals, legs, and badge subtext
GOAL_MAP: Dict[str, str] = {  # research-first labels (owner review 6); keys are the planner's goals
    "grind": "Grind: level up",
    "heal": "Heal: restore HP",
    "travel_to_pewter": "Travel: chain of System 1 walks",
    "gym": "Gym: earn the badge",
    "": "—",
}

LEG_MAP: Dict[str, str] = {  # the System 1 policy names as trained (lab/results/compute.md, run JSON legs[].leg)
    "to_grass": "to_grass",
    "to_viridian_pc": "to_viridian_pc",
    "leave_viridian_pc": "leave_viridian_pc",
    "viridian": "viridian",
    "to_route2": "to_route2",
    "to_gate": "to_gate",
    "forest": "forest",
    "to_pewter": "to_pewter",
    "to_pewter_pc": "to_pewter_pc",
    "leave_pewter_pc": "leave_pewter_pc",
    "to_gym": "to_gym",
    "": "—",
}

BADGE_SUBTEXT: Dict[str, str] = {
    "Backup path-finder (BFS)": "fallback after a System 1 miss",  # the only pill whose text needs explaining
}

_ICON_CACHE: Dict[int, Image.Image] = {}


def get_pokeball_icon(size: int = HEADING_ICON_SIZE) -> Image.Image:
    """Procedurally generates a Poké Ball-style icon using PIL primitives.

    Drawn supersampled at 4x with PIL primitives (pieslices, rectangles, ellipses)
    and downscaled with Lanczos antialiasing for crisp, smooth edges.
    """
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]

    scale = 4
    s = size * scale
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    cx, cy = s // 2, s // 2
    r = s // 2 - 2 * scale

    # Upper red dome
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], start=180, end=360, fill=(230, 45, 45, 255))
    # Lower white dome
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], start=0, end=180, fill=(245, 245, 248, 255))

    # Center dividing band
    band_h = max(2, int(r * 0.22))
    draw.rectangle([cx - r, cy - band_h // 2, cx + r, cy + band_h // 2], fill=(24, 28, 36, 255))

    # Outer circle border
    border_w = max(2, int(r * 0.12))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(24, 28, 36, 255), width=border_w)

    # Center button outer ring
    btn_r = int(r * 0.36)
    draw.ellipse([cx - btn_r, cy - btn_r, cx + btn_r, cy + btn_r], fill=(24, 28, 36, 255))

    # Center button face
    inner_r = int(r * 0.24)
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=(245, 245, 248, 255))

    # Center button core dot
    core_r = max(2, int(r * 0.12))
    draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=(210, 215, 225, 255), outline=(140, 150, 165, 255), width=max(1, scale))

    resample = getattr(Image, "Resampling", Image).LANCZOS
    icon = im.resize((size, size), resample=resample)
    _ICON_CACHE[size] = icon
    return icon


def draw_heading(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: FontManager,
    cx: int = CANVAS_WIDTH // 2,
    cy: int = HEADING_Y,
    text: str = "PokeDreamer",
    font_size: int = HEADING_FONT_SIZE,
    fill: Tuple[int, int, int] = (255, 255, 255),
    icon_size: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Draws procedural Poké Ball icon and bold heading text centered at (cx, cy)."""
    if icon_size is None:
        icon_size = max(16, int(round(font_size * 0.88)))

    font = fonts.get(font_size, bold=True)
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    gap = max(8, int(font_size * 0.28))
    total_w = icon_size + gap + tw

    start_x = cx - total_w // 2
    icon_x = start_x
    icon_y = cy - icon_size // 2
    text_x = start_x + icon_size + gap

    icon = get_pokeball_icon(icon_size)
    img.paste(icon, (icon_x, icon_y), icon)
    draw.text((text_x, cy), text, font=font, fill=fill, anchor="lm")

    return (start_x, cy - font_size // 2, start_x + total_w, cy + font_size // 2)


def calculate_segment_frames(storyboard: List[Dict[str, Any]], fps: int = DEFAULT_FPS) -> List[int]:
    """Computes output frame count for each segment from its seconds (or frame count for clips)."""
    counts: List[int] = []
    for s in storyboard:
        if s.get("type") == "clip":
            start_f = int(s.get("start_frame", 0))
            end_f = int(s.get("end_frame", 0))
            hold_s = float(s.get("hold_s", 0.0))
            counts.append((end_f - start_f) + int(round(hold_s * fps)))
        else:
            counts.append(int(round(s["seconds"] * fps)))
    return counts


def load_clip_frames(
    src: str | Path,
    start_frame: int,
    end_frame: int,
    hold_s: float = 0.0,
    fps: int = DEFAULT_FPS,
) -> List[np.ndarray]:
    """Loads frames [start_frame, end_frame) from video src, resized to 1920x1080 if needed, with hold."""
    src_path = str(src)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Clip source not found: {src_path}")

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Could not open clip video: {src_path}")

    # Skip sequentially: CAP_PROP_POS_FRAMES seeks can land on a nearby keyframe in mp4v files.
    for _ in range(start_frame):
        cap.grab()

    frames: List[np.ndarray] = []
    num_to_read = max(0, end_frame - start_frame)
    last_frame = None
    for _ in range(num_to_read):
        ret, frame_bgr = cap.read()
        if not ret:
            if last_frame is not None:
                frame_rgb = last_frame.copy()
            else:
                frame_rgb = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[0] != CANVAS_HEIGHT or frame_rgb.shape[1] != CANVAS_WIDTH:
                frame_rgb = cv2.resize(frame_rgb, (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)
            last_frame = frame_rgb
        frames.append(frame_rgb)
    cap.release()

    hold_frames = int(round(hold_s * fps))
    if hold_frames > 0 and len(frames) > 0:
        last = frames[-1]
        for _ in range(hold_frames):
            frames.append(last.copy())

    return frames


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


def battle_flash_mask(colors: np.ndarray, in_battles: np.ndarray) -> np.ndarray:
    """True for the steps that flash when sped up (owner: "flashbang").

    Every wild/trainer battle starts with the same transition, one frame per press: ~3 recoloured overworld
    frames, a dark frame, then 2-3 pure white frames before the battle screen. Masked: each onset through the
    end of its first run of flat frames within 12 steps (luminance std < 20: white/dark screens measure 0-10,
    battle and overworld screens >= 40), plus any other flat frame.
    """
    lum = colors[:, ::4, ::4].astype(np.float32).mean(axis=3)
    flat = lum.reshape(len(lum), -1).std(axis=1) < 20.0
    mask = flat.copy()
    onsets = np.where(np.diff(in_battles.astype(np.int8)) == 1)[0] + 1
    for o in onsets:
        f = np.where(flat[o : o + 12])[0]
        end = o + 4
        if len(f):
            end = o + f[0]
            while end + 1 < len(flat) and flat[end + 1]:
                end += 1
        mask[o : end + 1] = True
    return mask


def deflash_steps(steps: np.ndarray, flash: np.ndarray) -> np.ndarray:
    """Shows the last non-flash step instead of each flash step (a short hold), so the timing is unchanged."""
    out = steps.copy()
    for i, s in enumerate(steps):
        while s > 0 and flash[s]:
            s -= 1
        out[i] = s
    return out


def get_hud_badge(
    in_battle: bool, by: str
) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Returns (badge_text, bg_rgb, border_rgb, text_rgb) based on step flags.

    Priority:
    1. in_battles -> "Battling" (amber)
    2. by == "bfs" -> "Backup path-finder (BFS)" (grey)
    3. by == "policy" -> "System 1 policy, dream-trained" (green)
    4. otherwise (no navigation leg pressing) -> "Menus, dialogue, waiting" (dark grey)
    """
    if in_battle:
        return "Battling", (75, 45, 12), (180, 110, 25), (252, 211, 77)
    elif str(by).lower() == "bfs":
        return "Backup path-finder (BFS)", (45, 55, 68), (100, 116, 139), (203, 213, 225)
    elif str(by).lower() == "policy":
        return "System 1 policy, dream-trained", (18, 65, 36), (46, 125, 50), (110, 231, 145)
    else:  # no navigation leg is pressing: menus, dialogue, the nurse-counter walk, grind pacing (scripted)
        return "Menus, dialogue, waiting", (40, 46, 56), (90, 100, 115), (190, 198, 210)


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


def truncate_reason(reason: str, max_chars: int = 180) -> str:
    """Truncates reason at a word boundary, appending '…' if truncated."""
    if len(reason) <= max_chars:
        return reason
    cut = reason[:max_chars]
    last_space = cut.rfind(" ")
    if last_space != -1:
        cut = cut[:last_space]
    return cut.rstrip(".,;:!? ") + "…"


def get_latest_decision(timeline: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    """Returns the latest timeline entry with presses <= step, or None if none qualify."""
    candidates = [e for e in timeline if e.get("presses", 0) <= step]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("presses", 0))


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    """Wraps text into lines that fit within max_width pixels using font."""
    lines: List[str] = []
    words = text.split()
    if not words:
        return []
    cur_line = words[0]
    for w in words[1:]:
        test_line = cur_line + " " + w
        w_px = font.getlength(test_line) if hasattr(font, "getlength") else font.getsize(test_line)[0]
        if w_px <= max_width:
            cur_line = test_line
        else:
            lines.append(cur_line)
            cur_line = w
    lines.append(cur_line)
    return lines


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
    is_llm_planner: bool = False,
    planner_decision: Optional[Dict[str, Any]] = None,
    is_decision_highlighted: bool = False,
    img: Optional[Image.Image] = None,
):
    """Draws left-column HUD card with status, goals, badge, and counter. `img` (the canvas) is needed for logos."""
    x0, y0, x1, y1 = box
    box_h = y1 - y0

    # Large layout for 6x (or any height >= 700), compact fallback for smaller boxes
    is_large = box_h >= 700

    radius = 16 if is_large else 14
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=(20, 26, 35), outline=(45, 58, 75), width=2)

    pad_x = 40 if is_large else 30
    avail_w = (x1 - x0) - 2 * pad_x

    if is_large:
        # Header: one line (the section cards carry the rest); it stays so a shared still is honest on its own
        header = (
            "Walking: System 1, dream-trained · System 2: local LLM"
            if is_llm_planner
            else "Walking: System 1, dream-trained"
        )
        draw.text((x0 + pad_x, y0 + 38), header, font=fonts.get(19), fill=(248, 250, 252))
        draw.line([(x0 + pad_x, y0 + 82), (x1 - pad_x, y0 + 82)], fill=(34, 45, 58), width=2)

        if is_llm_planner:
            step_y = 110
            curr_y = y0 + 104
        else:
            step_y = 118 if progress_text else 128
            curr_y = y0 + 106

        def _draw_section(label: str, sub_label: str, val_text: str, default_size: int, fill: Tuple[int, int, int], is_bold: bool = True):
            draw.text((x0 + pad_x, curr_y), label, font=fonts.get(16, bold=True), fill=(148, 163, 184))
            if sub_label:
                lw = draw.textlength(label, font=fonts.get(16, bold=True))
                draw.text((x0 + pad_x + lw + 12, curr_y + 2), sub_label, font=fonts.get(13), fill=(100, 116, 139))
            sz = default_size
            while sz > 16 and draw.textlength(val_text, font=fonts.get(sz, bold=is_bold)) > avail_w:
                sz -= 2
            draw.text((x0 + pad_x, curr_y + 28), val_text, font=fonts.get(sz, bold=is_bold), fill=fill)

        # 1. Map name
        _draw_section("ENVIRONMENT", "", "real game, Game Boy emulator", default_size=30, fill=(248, 250, 252))
        curr_y += step_y

        # 2. System 2 Goal / The Plan
        plan_sub = "local LLM planner" if is_llm_planner else ""
        goal_val = GOAL_MAP.get(str(goal), "—") if goal else "—"
        _draw_section("SYSTEM 2 PLAN", plan_sub, goal_val, default_size=32, fill=(248, 250, 252))
        curr_y += step_y

        # 3. Active Leg / Walking to
        leg_val = LEG_MAP.get(str(leg), "—") if leg else "—"
        _draw_section("SYSTEM 1 POLICY", "trained in imagination", leg_val, default_size=30, fill=(147, 197, 253))
        curr_y += step_y

        # 4. Control Badge
        draw.text((x0 + pad_x, curr_y), "ACTIVE CONTROLLER", font=fonts.get(16, bold=True), fill=(148, 163, 184))

        badge_text, badge_bg, badge_border, badge_fg = get_hud_badge(in_battle, by)
        badge_font = fonts.get(22, bold=True)
        tb = draw.textbbox((0, 0), badge_text, font=badge_font)
        bw, bh = tb[2] - tb[0], tb[3] - tb[1]
        pill_w = bw + 36
        pill_h = bh + 16
        pill_y = curr_y + 24
        draw.rounded_rectangle(
            [x0 + pad_x, pill_y, x0 + pad_x + pill_w, pill_y + pill_h],
            radius=10,
            fill=badge_bg,
            outline=badge_border,
            width=2,
        )
        draw.text((x0 + pad_x + pill_w // 2, pill_y + pill_h // 2), badge_text, font=badge_font, fill=badge_fg, anchor="mm")

        badge_sub = BADGE_SUBTEXT.get(badge_text)
        if badge_sub:
            draw.text((x0 + pad_x, pill_y + pill_h + 6), badge_sub, font=fonts.get(13), fill=(100, 116, 139))
        curr_y += (step_y + 10) if is_llm_planner else step_y

        # 5. Press Counter (step + 1)
        draw.text((x0 + pad_x, curr_y), "ACTIONS", font=fonts.get(16, bold=True), fill=(148, 163, 184))
        draw.text((x0 + pad_x + draw.textlength("ACTIONS", font=fonts.get(16, bold=True)) + 12, curr_y + 2), "button presses",
                  font=fonts.get(13), fill=(100, 116, 139))
        draw.text((x0 + pad_x, curr_y + 28), f"{step + 1:,}", font=fonts.get(42 if is_llm_planner else 44, bold=True), fill=(56, 189, 248))
        curr_y += (step_y - 2) if is_llm_planner else step_y

        # 6. Optional level progress (for montage)
        if progress_text:
            _draw_section("GRIND PROGRESS", "", progress_text, default_size=30, fill=(250, 204, 21))

        # 7. Planner card (for harness)
        if is_llm_planner and planner_decision is not None:
            card_x0 = x0 + pad_x
            card_x1 = x1 - pad_x
            card_y0 = curr_y + 6
            card_y1 = y1 - 22
            card_w = card_x1 - card_x0

            if is_decision_highlighted:
                c_fill = (24, 38, 54)
                c_border = (56, 189, 248)
                b_width = 3
            else:
                c_fill = (16, 22, 30)
                c_border = (45, 58, 75)
                b_width = 2

            draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=12, fill=c_fill, outline=c_border, width=b_width)

            # Card Header, with the Gemma and Ollama logos in front when present (work/video/logos/SOURCE.txt)
            hdr_x = card_x0 + 18
            for key in ("gemma", "ollama"):
                logo_path = Path("work/video/logos") / f"{key}.png"
                if img is not None and logo_path.exists():
                    logo_im = Image.open(logo_path).convert("RGBA")
                    logo_im = logo_im.resize((22, 22), resample=getattr(Image, "Resampling", Image).LANCZOS)
                    img.paste(logo_im, (hdr_x, card_y0 + 10), logo_im)
                    hdr_x += 28
            hdr_text = "System 2 planner: gemma4:31b (local, via Ollama)"
            draw.text((hdr_x, card_y0 + 13), hdr_text, font=fonts.get(16, bold=True), fill=(248, 250, 252))

            # Latency tag when highlighted
            if is_decision_highlighted:
                latency = float(planner_decision.get("latency", 0.0))
                tag_str = f"thought for {latency:.1f} s"
                f_tag = fonts.get(14, bold=True)
                tb_tag = draw.textbbox((0, 0), tag_str, font=f_tag)
                t_w, t_h = tb_tag[2] - tb_tag[0], tb_tag[3] - tb_tag[1]
                p_w = t_w + 20
                p_h = t_h + 10
                p_x0 = card_x1 - 18 - p_w
                p_y0 = card_y0 + 9
                draw.rounded_rectangle([p_x0, p_y0, p_x0 + p_w, p_y0 + p_h], radius=6, fill=(15, 45, 75), outline=(56, 189, 248), width=2)
                draw.text((p_x0 + p_w // 2, p_y0 + p_h // 2), tag_str, font=f_tag, fill=(125, 211, 252), anchor="mm")

            draw.line([(card_x0 + 18, card_y0 + 40), (card_x1 - 18, card_y0 + 40)], fill=(34, 45, 58), width=1)

            dec_goal = str(planner_decision.get("goal", ""))
            goal_mapped = GOAL_MAP.get(dec_goal, dec_goal)
            dec_line = f"Goal: {dec_goal} — {goal_mapped}"
            draw.text((card_x0 + 18, card_y0 + 48), dec_line, font=fonts.get(17, bold=True), fill=(250, 204, 21))

            raw_reason = str(planner_decision.get("reason", ""))
            trunc = truncate_reason(raw_reason, max_chars=180)
            quoted = f'"{trunc}"'
            wrapped_lines = wrap_text(quoted, fonts.get(15), max_width=card_w - 36)
            for idx, line in enumerate(wrapped_lines[:4]):
                draw.text((card_x0 + 18, card_y0 + 76 + idx * 24), line, font=fonts.get(15), fill=(203, 213, 225))

    else:
        # Compact fallback for smaller scale/box
        draw.text((x0 + 30, y0 + 16), "Walking: dream-trained System 1 policy.", font=fonts.get(13), fill=(248, 250, 252))
        draw.line([(x0 + 30, y0 + 44), (x1 - 30, y0 + 44)], fill=(34, 45, 58), width=1)

        curr_y = y0 + 56

        def _draw_compact_section(label: str, sub_label: str, val_text: str, default_size: int, fill: Tuple[int, int, int]):
            draw.text((x0 + 30, curr_y), label, font=fonts.get(13), fill=(148, 163, 184))
            if sub_label:
                lw = draw.textlength(label, font=fonts.get(13))
                draw.text((x0 + 30 + lw + 8, curr_y + 1), sub_label, font=fonts.get(11), fill=(100, 116, 139))
            draw.text((x0 + 30, curr_y + 18), val_text, font=fonts.get(default_size, bold=True), fill=fill)

        # 1. Map name
        _draw_compact_section("ENVIRONMENT", "", "real game, Game Boy emulator", 20, (248, 250, 252))
        curr_y += 56

        # 2. System 2 Goal
        goal_val = GOAL_MAP.get(str(goal), "—") if goal else "—"
        _draw_compact_section("SYSTEM 2 PLAN", "", goal_val, 18, (248, 250, 252))
        curr_y += 56

        # 3. Active Leg
        leg_val = LEG_MAP.get(str(leg), "—") if leg else "—"
        _draw_compact_section("SYSTEM 1 POLICY", "trained in imagination", leg_val, 18, (147, 197, 253))
        curr_y += 56

        # 4. Control Badge
        draw.text((x0 + 30, curr_y), "ACTIVE CONTROLLER", font=fonts.get(13), fill=(148, 163, 184))
        badge_text, badge_bg, badge_border, badge_fg = get_hud_badge(in_battle, by)
        badge_font = fonts.get(16, bold=True)
        tb = draw.textbbox((0, 0), badge_text, font=badge_font)
        bw, bh = tb[2] - tb[0], tb[3] - tb[1]
        pill_w = bw + 24
        pill_h = bh + 12
        pill_y = curr_y + 18
        draw.rounded_rectangle(
            [x0 + 30, pill_y, x0 + 30 + pill_w, pill_y + pill_h],
            radius=6,
            fill=badge_bg,
            outline=badge_border,
            width=2,
        )
        draw.text((x0 + 30 + pill_w // 2, pill_y + pill_h // 2), badge_text, font=badge_font, fill=badge_fg, anchor="mm")
        curr_y += 58

        # 5. Press Counter (step + 1)
        draw.text((x0 + 30, curr_y), "ACTIONS", font=fonts.get(13), fill=(148, 163, 184))
        draw.text((x0 + 30, curr_y + 18), f"{step + 1:,}", font=fonts.get(26, bold=True), fill=(56, 189, 248))
        curr_y += 56

        # 6. Optional level progress (for montage)
        if progress_text:
            _draw_compact_section("GRIND PROGRESS", "", progress_text, 18, (250, 204, 21))


def render_title_frame(
    bg_frame: Optional[np.ndarray],
    text_lines: List[str],
    caption: str,
    fonts: FontManager,
    fade_alpha: float = 1.0,
    subcaption: Optional[str] = None,
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
    draw_heading(img, draw, fonts, cx=cx, cy=cy - 90, text=title, font_size=56, icon_size=48)
    draw.line([(cx - 250, cy - 40), (cx + 250, cy - 40)], fill=(45, 58, 75), width=1)

    if len(text_lines) > 1:
        draw.text((cx, cy + 15), text_lines[1], font=fonts.get(28, bold=True), fill=(226, 232, 240), anchor="mm")
    if len(text_lines) > 2:
        draw.text((cx, cy + 65), text_lines[2], font=fonts.get(28, bold=True), fill=(226, 232, 240), anchor="mm")

    # Subcaption or distinct caption shown under the card
    note = subcaption if subcaption else caption
    combined_text = " ".join(text_lines)
    if note and note not in combined_text:
        draw.text((cx, CANVAS_HEIGHT - 100), note, font=fonts.get(20), fill=(160, 175, 195), anchor="mm")

    res = np.asarray(img)
    if fade_alpha < 0.999:
        res = (res.astype(np.float32) * max(0.0, min(1.0, fade_alpha))).astype(np.uint8)
    return res


END_HEADING = "PokeDreamer · results"
END_FOOTER = "Walking: zero real-game training steps · battles: scripted · System 2: rules (3 seeds x 10 runs) or local LLM (10 runs)"


def render_end_frame(
    text_lines: List[str],
    caption: str,
    fonts: FontManager,
    fade_alpha: float = 1.0,
    heading: str = END_HEADING,
    footer: str = END_FOOTER,
) -> np.ndarray:
    """Renders the end card: a heading, up to 6 coloured lines, a footer (all from the storyboard when given)."""
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
    draw_heading(img, draw, fonts, cx=cx, cy=cy - 210, text=heading, font_size=20, icon_size=22, fill=(100, 116, 139))
    draw.line([(cx - card_w // 2 + 50, cy - 180), (cx + card_w // 2 - 50, cy - 180)], fill=(34, 45, 58), width=1)

    colors = [
        (250, 204, 21),   # Gold for badges
        (74, 222, 128),   # Green for learned navigation
        (56, 189, 248),   # Cyan for restored color/detail
        (167, 139, 250),  # Violet for the LLM planner
        (244, 114, 182),  # Pink for the upscaler/colorizer
    ]

    many = len(text_lines) > 4  # up to 6 lines fit between the rule and the footer
    base_y, step = (cy - 130, 64) if many else (cy - 110, 85)
    for idx, line in enumerate(text_lines):
        color = colors[idx % len(colors)]
        size = 30 if many else 34
        while size > 18 and draw.textlength(line, font=fonts.get(size, bold=True)) > card_w - 100:
            size -= 2                                     # shrink a long line until it fits inside the card
        draw.text((cx, base_y + idx * step), line, font=fonts.get(size, bold=True), fill=color, anchor="mm")

    # Footer note
    draw.text(
        (cx, cy + 220),
        footer,
        font=fonts.get(18),
        fill=(148, 163, 184),
        anchor="mm",
    )

    res = np.asarray(img)
    if fade_alpha < 0.999:
        res = (res.astype(np.float32) * max(0.0, min(1.0, fade_alpha))).astype(np.uint8)
    return res


def render_spec_end_frame(seg: Dict[str, Any], t_s: float, fonts: FontManager) -> np.ndarray:
    """End screen as a research spec sheet (owner review 7, Fable's design): 'what we built' panels on the left,
    'by the numbers' charts and tiles on the right, a spec strip; two accents, staggered reveal, fade out."""
    W, H = CANVAS_WIDTH, CANVAS_HEIGHT
    seconds = float(seg["seconds"])
    CYAN, GOLD = (6, 182, 212), (245, 158, 11)
    PANEL, BORDER, TITLE, BODY, MUTED, FAINT = (20, 26, 35), (45, 58, 75), (248, 250, 252), (203, 213, 225), (148, 163, 184), (100, 116, 139)

    def ease(a: float, b: float) -> float:
        x = min(1.0, max(0.0, (t_s - a) / (b - a)))
        return x * x * (3.0 - 2.0 * x)

    out = 1.0 - ease(seconds - 0.5, seconds)
    img = Image.new("RGBA", (W, H), BG_COLOR + (255,))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def col(c, a):
        return c + (int(255 * a * out),)

    def spaced(s: str) -> str:
        return " ".join(s.upper())

    # title band
    a = ease(0.0, 0.3)
    icon = get_pokeball_icon(28)
    ov.alpha_composite(icon, (64, 79))
    d.text((104, 72), seg["title"], font=fonts.get(44, bold=True), fill=col(TITLE, a))
    d.text((1856, 92), seg["title_right"], font=fonts.get(22), fill=col(MUTED, a), anchor="rm")
    d.line([(64, 150), (1856, 150)], fill=col(BORDER, a), width=2)

    # left: what we built
    d.text((64, 184), spaced(seg["left_label"]), font=fonts.get(16, bold=True), fill=col(MUTED, a))
    for k, (name, spec, note) in enumerate(seg["panels"]):
        ap = ease(0.3 + 0.15 * k, 0.6 + 0.15 * k)
        y = 220 + 134 * k + int((1.0 - ap) * 24)
        d.rounded_rectangle([64, y, 980, y + 118], radius=8, fill=col(PANEL, ap), outline=col(BORDER, ap), width=2)
        d.rectangle([64, y + 8, 68, y + 110], fill=col(CYAN, ap))
        d.text((88, y + 14), name, font=fonts.get(26, bold=True), fill=col(TITLE, ap))
        d.text((88, y + 54), spec, font=fonts.get(18), fill=col(BODY, ap))
        d.text((88, y + 84), note, font=fonts.get(15), fill=col(FAINT, ap))

    # right: by the numbers
    d.text((1020, 184), spaced(seg["right_label"]), font=fonts.get(16, bold=True), fill=col(MUTED, a))
    grow = ease(0.9, 1.5)
    bars = seg["bars"]
    ab = ease(0.6, 0.9)
    d.text((1020, 220), bars["title"], font=fonts.get(18), fill=col(MUTED, ab))
    vmax = max(v for _, v in bars["rows"])
    for k, (label, v) in enumerate(bars["rows"]):
        y = 256 + 50 * k
        d.text((1020, y + 17), label, font=fonts.get(18), fill=col(BODY, ab), anchor="lm")
        bw = max(2, int(470 * v / vmax * grow))
        d.rectangle([1290, y, 1290 + bw, y + 34], fill=col(CYAN, ab))
        d.text((1300 + bw, y + 17), f"{v:g}", font=fonts.get(20, bold=True), fill=col(TITLE, ab * grow), anchor="lm")
    d.text((1856, 466), bars["total"], font=fonts.get(20, bold=True), fill=col(TITLE, ab * grow), anchor="rm")

    cmp_ = seg["compare"]
    ac = ease(0.75, 1.05)
    d.text((1020, 500), cmp_["title"], font=fonts.get(18), fill=col(MUTED, ac))
    cmax = max(r[1] for r in cmp_["rows"])
    for k, (label, v, accent) in enumerate(cmp_["rows"]):
        y = 536 + 50 * k
        d.text((1020, y + 17), label, font=fonts.get(18), fill=col(BODY, ac), anchor="lm")
        bw = max(2, int(470 * v / cmax * grow))
        d.rectangle([1290, y, 1290 + bw, y + 34], fill=col(GOLD if accent == "gold" else CYAN, ac))
        d.text((1300 + bw, y + 17), f"{v:,.1f}".replace(".0", ""), font=fonts.get(20, bold=True), fill=col(TITLE, ac * grow), anchor="lm")
    d.text((1020, 650), cmp_["note"], font=fonts.get(15), fill=col(FAINT, ac))

    count = ease(1.0, 1.6)
    for k, (value, caption, accent) in enumerate(seg["tiles"]):
        at = ease(0.9 + 0.1 * k, 1.2 + 0.1 * k)
        x = 1020 + 288 * k
        d.rounded_rectangle([x, 710, x + 260, 860], radius=8, fill=col(PANEL, at), outline=col(BORDER, at), width=2)
        if "/" in value:  # count up: "30/30"
            n, den = value.split("/")
            shown = f"{round(int(n) * count)}/{den}"
        else:  # count up: "91.4%" keeps its decimals
            num = value.rstrip("%")
            dec = len(num.split(".")[1]) if "." in num else 0
            shown = f"{float(num) * count:.{dec}f}%"
        d.text((x + 130, 768), shown, font=fonts.get(56, bold=True), fill=col(GOLD if accent == "gold" else CYAN, at), anchor="mm")
        for j, line in enumerate(caption.split("\n")):
            d.text((x + 130, 818 + 20 * j), line, font=fonts.get(15), fill=col(MUTED, at), anchor="mm")

    # spec strip
    a_s = ease(1.4, 1.8)
    for j, line in enumerate(seg["strip"]):
        d.text((W // 2, 936 + 32 * j), line, font=fonts.get(18), fill=col(MUTED if j < len(seg["strip"]) - 1 else FAINT, a_s), anchor="mm")

    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


_TRAJ_DATA_CACHE: Dict[str, Any] = {}


def get_traj_frames(path: str = "work/video/traj/run_seed_100.npz") -> np.ndarray:
    """Load and cache trajectory greyscale frames."""
    if path not in _TRAJ_DATA_CACHE:
        if os.path.exists(path):
            _TRAJ_DATA_CACHE[path] = np.load(path)["frames"]
        else:
            _TRAJ_DATA_CACHE[path] = np.zeros((10000, 72, 80), dtype=np.uint8)
    return _TRAJ_DATA_CACHE[path]


def format_diagram_tile_count(value: str, count: float) -> str:
    """Animate tile counter string over count in [0, 1]. Handles 30/30, 91.4%, 99.27%, −19%, 5 of 6, 676K."""
    count = min(1.0, max(0.0, count))
    if "/" in value:
        n, den = value.split("/")
        return f"{round(int(n) * count)}/{den}"
    if " of " in value:
        n, den = value.split(" of ")
        return f"{round(int(n) * count)} of {den}"
    if value.endswith("K"):
        n = float(value[:-1])
        return f"{round(n * count)}K"
    if value.endswith("M"):
        n = float(value[:-1])
        return f"{round(n * count)}M"
    if value.endswith("%"):
        prefix = ""
        raw = value[:-1]
        if raw.startswith("−"):
            prefix = "−"
            raw = raw[1:]
        elif raw.startswith("-"):
            prefix = "-"
            raw = raw[1:]
        dec = len(raw.split(".")[1]) if "." in raw else 0
        v = float(raw) * count
        if dec > 0:
            return f"{prefix}{v:.{dec}f}%"
        else:
            return f"{prefix}{round(v)}%"
    try:
        n = float(value)
        return str(round(n * count))
    except ValueError:
        return value


def draw_diagram_arrow(
    draw: ImageDraw.ImageDraw,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int, int],
    width: int = 3,
    head_size: float = 12.0,
    style: str = "flow",
    progress: float = 1.0,
) -> None:
    """Draw a straight or polyline arrow with draw-in animation and arrowhead."""
    if len(points) < 2 or progress <= 0.0:
        return

    segs = []
    total_len = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        l = math.hypot(x2 - x1, y2 - y1)
        segs.append((x1, y1, x2, y2, l))
        total_len += l

    if total_len <= 0:
        return

    target_len = total_len * min(1.0, max(0.0, progress))
    curr_len = 0.0
    drawn_pts: List[Tuple[float, float]] = [(points[0][0], points[0][1])]
    last_dir = (1.0, 0.0)

    for x1, y1, x2, y2, l in segs:
        if l == 0:
            continue
        ux, uy = (x2 - x1) / l, (y2 - y1) / l
        if curr_len + l <= target_len:
            drawn_pts.append((x2, y2))
            curr_len += l
            last_dir = (ux, uy)
        else:
            rem = target_len - curr_len
            px, py = x1 + ux * rem, y1 + uy * rem
            drawn_pts.append((px, py))
            last_dir = (ux, uy)
            curr_len = target_len
            break

    if style == "dashed":
        dash_len = 8.0
        gap_len = 6.0
        for i in range(len(drawn_pts) - 1):
            sx, sy = drawn_pts[i]
            ex, ey = drawn_pts[i + 1]
            seg_l = math.hypot(ex - sx, ey - sy)
            if seg_l == 0:
                continue
            dux, duy = (ex - sx) / seg_l, (ey - sy) / seg_l
            d_curr = 0.0
            is_dash = True
            while d_curr < seg_l:
                d_next = min(seg_l, d_curr + (dash_len if is_dash else gap_len))
                if is_dash:
                    draw.line(
                        [(sx + dux * d_curr, sy + duy * d_curr), (sx + dux * d_next, sy + duy * d_next)],
                        fill=color,
                        width=width,
                    )
                d_curr = d_next
                is_dash = not is_dash
    else:
        for i in range(len(drawn_pts) - 1):
            draw.line([drawn_pts[i], drawn_pts[i + 1]], fill=color, width=width)

    if curr_len >= 8.0 and style != "no_head":
        tip = drawn_pts[-1]
        ux, uy = last_dir
        bx, by = tip[0] - ux * head_size, tip[1] - uy * head_size
        hw = head_size * 0.5
        nx, ny = -uy, ux
        p1 = (bx + nx * hw, by + ny * hw)
        p2 = (bx - nx * hw, by - ny * hw)
        draw.polygon([tip, p1, p2], fill=color)


def render_diagram_frame(seg: Dict[str, Any], t_s: float, fonts: FontManager, theme: str = "dark") -> np.ndarray:
    """Generic diagram renderer driven entirely by JSON elements in a 'diagram' segment."""
    W, H = CANVAS_WIDTH, CANVAS_HEIGHT
    seconds = float(seg.get("seconds", 8.0))
    CYAN, GOLD = (6, 182, 212), (245, 158, 11)
    PANEL, BORDER, TITLE, BODY, MUTED, FAINT = (
        (20, 26, 35),
        (45, 58, 75),
        (248, 250, 252),
        (203, 213, 225),
        (148, 163, 184),
        (100, 116, 139),
    )
    BORDER_GREY = (45, 58, 75)
    bg = BG_COLOR
    if theme == "light":
        L = LIGHT_INFO
        CYAN, GOLD = L["cyan"], L["gold"]
        PANEL, BORDER, TITLE, BODY, MUTED, FAINT = L["panel"], L["border"], L["title"], L["body"], L["muted"], L["faint"]
        BORDER_GREY, bg = BORDER, L["bg"]

    COLOR_LOOKUP = {
        "cyan": CYAN,
        "gold": GOLD,
        "title": TITLE,
        "body": BODY,
        "muted": MUTED,
        "faint": FAINT,
        "border": BORDER,
        "border_grey": BORDER_GREY,
        "panel": PANEL,
        "white": (255, 255, 255),
    }

    def resolve_color(col_val: Any, default: Tuple[int, int, int] = BODY) -> Tuple[int, int, int]:
        if isinstance(col_val, (tuple, list)):
            return tuple(col_val[:3])
        if isinstance(col_val, str):
            return COLOR_LOOKUP.get(col_val.lower(), default)
        return default

    def ease(a: float, b: float) -> float:
        if b <= a:
            return 1.0 if t_s >= a else 0.0
        x = min(1.0, max(0.0, (t_s - a) / (b - a)))
        return x * x * (3.0 - 2.0 * x)

    out = 1.0 - ease(seconds - 0.5, seconds)

    img = Image.new("RGBA", (W, H), bg + (255,))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def col(c: Any, a: float) -> Tuple[int, int, int, int]:
        rgb = resolve_color(c)
        return rgb + (int(255 * a * out),)

    def spaced(s: str) -> str:
        return " ".join(s.upper())

    # Title band
    a_hdr = ease(0.0, 0.3)
    icon = get_pokeball_icon(28)
    if a_hdr * out < 0.999:
        icon_faded = icon.copy()
        r, g, b, a_ch = icon_faded.split()
        a_ch = a_ch.point(lambda p: int(p * a_hdr * out))
        icon_faded.putalpha(a_ch)
        ov.alpha_composite(icon_faded, (64, 79))
    else:
        ov.alpha_composite(icon, (64, 79))

    title_text = seg.get("title", "PokeDreamer v3")
    d.text((104, 72), title_text, font=fonts.get(44, bold=True), fill=col(TITLE, a_hdr))

    if "title_right" in seg:
        d.text((1856, 92), seg["title_right"], font=fonts.get(22), fill=col(MUTED, a_hdr), anchor="rm")

    d.line([(64, 150), (1856, 150)], fill=col(BORDER, a_hdr), width=2)

    # Section labels
    if "left_label" in seg and seg["left_label"]:
        f16 = fonts.get(16, bold=True)
        lbl = spaced(seg["left_label"])
        d.text((64, 184), lbl, font=f16, fill=col(MUTED, a_hdr))
        hl = seg.get("left_highlight")  # owner: the key term on a yellow highlighter box, dark text
        if hl:
            hx = 64 + int(d.textlength(lbl + "   ", font=f16))
            htxt = spaced(hl)
            bb = d.textbbox((hx, 184), htxt, font=f16)
            d.rounded_rectangle([bb[0] - 8, bb[1] - 6, bb[2] + 8, bb[3] + 6], radius=6, fill=col((250, 204, 21), a_hdr))
            d.text((hx, 184), htxt, font=f16, fill=col((15, 20, 25), a_hdr))
    if "right_label" in seg and seg["right_label"]:
        rx = seg.get("right_label_x", 1240)
        d.text((rx, 184), spaced(seg["right_label"]), font=fonts.get(16, bold=True), fill=col(MUTED, a_hdr))

    # Process elements
    elements = seg.get("elements", [])
    for elem in elements:
        kind = elem.get("kind", elem.get("type", ""))
        t0 = float(elem.get("t0", 0.0))

        if kind == "box":
            a = ease(t0, t0 + 0.3)
            if a <= 0.0:
                continue
            y_off = int((1.0 - a) * 24)
            x0, y0, x1, y1 = elem["rect"]
            box_rect = [x0, y0 + y_off, x1, y1 + y_off]
            d.rounded_rectangle(box_rect, radius=8, fill=col(PANEL, a), outline=col(BORDER, a), width=2)

            accent = elem.get("accent", "cyan")
            if accent and accent != "none":
                acc_rgb = resolve_color(accent, CYAN)
                d.rectangle([x0, y0 + y_off + 8, x0 + 4, y1 + y_off - 8], fill=col(acc_rgb, a))

            title = elem.get("title")
            t_size = elem.get("title_size", 24)
            padding_x = elem.get("padding_x", 16 if (x1 - x0) <= 300 else 20)
            max_inner_w = (x1 - x0) - 2 * padding_x

            curr_y = y0 + y_off + elem.get("title_y_pad", 10 if (y1 - y0) <= 120 else 14)
            if title:
                d.text((x0 + padding_x, curr_y), title, font=fonts.get(t_size, bold=True), fill=col(TITLE, a))
                curr_y += t_size + elem.get("title_gap", 10)

            body_sz = elem.get("body_size", 16 if (x1 - x0) <= 650 else 18)
            f_body = fonts.get(body_sz)
            lines = elem.get("lines", [])
            line_spacing = elem.get("line_spacing", body_sz + 4)
            for item in lines:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                    l_y = item.get("y", curr_y) + y_off
                    d.text((x0 + padding_x, l_y), txt, font=f_body, fill=col(BODY, a))
                    curr_y = l_y + line_spacing
                else:
                    wrapped = wrap_text(str(item), f_body, max_inner_w)
                    for wl in wrapped:
                        d.text((x0 + padding_x, curr_y), wl, font=f_body, fill=col(BODY, a))
                        curr_y += line_spacing

            note_sz = elem.get("note_size", 14 if (y1 - y0) <= 120 else 15)
            f_note = fonts.get(note_sz)
            notes = elem.get("notes", elem.get("note", []))
            if isinstance(notes, (str, dict)):
                notes = [notes]
            note_spacing = elem.get("note_spacing", note_sz + 3)
            for item in notes:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                    n_y = item.get("y", curr_y) + y_off
                    d.text((x0 + padding_x, n_y), txt, font=f_note, fill=col(FAINT, a))
                    curr_y = n_y + note_spacing
                else:
                    wrapped = wrap_text(str(item), f_note, max_inner_w)
                    for wn in wrapped:
                        d.text((x0 + padding_x, curr_y), wn, font=f_note, fill=col(FAINT, a))
                        curr_y += note_spacing

        elif kind == "arrow":
            prog = ease(t0, t0 + 0.3)
            if prog <= 0.0:
                continue
            pts = elem["points"]
            style = elem.get("style", "flow")
            if style == "loop":
                arrow_color = col(GOLD, 1.0)
            elif style == "dashed":
                arrow_color = col(GOLD, 1.0)
            else:
                arrow_color = col(CYAN, 1.0)

            draw_diagram_arrow(
                d,
                pts,
                color=arrow_color,
                width=elem.get("width", 3),
                head_size=elem.get("head_size", 12.0),
                style=style,
                progress=prog,
            )

            label = elem.get("label")
            if label and "label_pos" in elem and prog > 0.0:
                a_lbl = ease(t0, t0 + 0.3)
                l_sz = elem.get("label_size", 16)
                l_col = elem.get("label_colour", elem.get("label_color", "muted"))
                l_anchor = elem.get("label_anchor", "mm")
                d.text(
                    tuple(elem["label_pos"]),
                    label,
                    font=fonts.get(l_sz),
                    fill=col(l_col, a_lbl),
                    anchor=l_anchor,
                )

        elif kind == "chip":
            a = ease(t0, t0 + 0.3)
            if a <= 0.0:
                continue
            y_off = int((1.0 - a) * 24)
            x0, y0, x1, y1 = elem["rect"]
            chip_rect = [x0, y0 + y_off, x1, y1 + y_off]
            d.rounded_rectangle(chip_rect, radius=6, fill=col(PANEL, a), outline=col(CYAN, a), width=1)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2 + y_off
            f_chip = fonts.get(elem.get("font_size", 15))
            d.text((cx, cy), elem.get("text", ""), font=f_chip, fill=col(TITLE, a), anchor="mm")

        elif kind == "label":
            a = ease(t0, t0 + 0.3)
            if a <= 0.0:
                continue
            txt = elem.get("text", "")
            if elem.get("spaced", False):
                txt = spaced(txt)
            sz = elem.get("size", 16)
            bold = elem.get("bold", False)
            c_name = elem.get("colour", elem.get("color", "muted"))
            anchor = elem.get("anchor", "lt")
            d.text(tuple(elem["pos"]), txt, font=fonts.get(sz, bold=bold), fill=col(c_name, a), anchor=anchor)

        elif kind == "thumb":
            a = ease(t0, t0 + 0.3)
            if a <= 0.0:
                continue
            y_off = int((1.0 - a) * 24)
            x0, y0, x1, y1 = elem["rect"]
            tw, th = x1 - x0, y1 - y0
            step = int(elem.get("step", 0))
            traj_p = elem.get("traj_path", "work/video/traj/run_seed_100.npz")
            frames = get_traj_frames(traj_p)
            step_clamped = min(max(0, step), len(frames) - 1)
            frame_raw = frames[step_clamped]

            resized = cv2.resize(frame_raw, (tw, th), interpolation=cv2.INTER_NEAREST)
            alpha_ch = np.full_like(resized, int(255 * a * out))
            rgba = np.stack([resized, resized, resized, alpha_ch], axis=-1)
            thumb_im = Image.fromarray(rgba, mode="RGBA")
            ov.alpha_composite(thumb_im, (x0, y0 + y_off))
            d.rectangle([x0, y0 + y_off, x1 - 1, y1 + y_off - 1], outline=col(BORDER, a), width=1)

        elif kind == "bars":
            a = ease(t0, t0 + 0.3)
            if a <= 0.0:
                continue
            grow = ease(t0, t0 + elem.get("grow_duration", 0.6))
            x0, y0, x1, y1 = elem["rect"]
            title = elem.get("title", "")
            d.text((x0, y0), title, font=fonts.get(18), fill=col(MUTED, a))

            rows = elem.get("rows", [])
            vmax = max((r[1] for r in rows), default=1.0)
            bar_start = elem.get("bar_start_x", x0 + 200)
            bar_w_max = elem.get("bar_width", x1 - bar_start - 120)

            for k, row_data in enumerate(rows):
                lbl, val = row_data[0], row_data[1]
                row_c = row_data[2] if len(row_data) > 2 else "cyan"
                y_r = y0 + 36 + 48 * k
                d.text((x0, y_r + 17), lbl, font=fonts.get(18), fill=col(BODY, a), anchor="lm")
                bw = max(2, int(bar_w_max * (val / vmax) * grow))
                d.rectangle([bar_start, y_r, bar_start + bw, y_r + 34], fill=col(row_c, a))

                if val >= 100:
                    val_str = f"{val:,.1f}".replace(".0", "")
                else:
                    val_str = f"{val:g}"
                d.text(
                    (bar_start + bw + 10, y_r + 17),
                    val_str,
                    font=fonts.get(20, bold=True),
                    fill=col(TITLE, a * grow),
                    anchor="lm",
                )

            total = elem.get("total")
            if total:
                d.text(
                    (bar_start + bar_w_max + 60, y0 + 36 + 48 * len(rows) + 8),
                    total,
                    font=fonts.get(20, bold=True),
                    fill=col(TITLE, a * grow),
                    anchor="rm",
                )

            note = elem.get("note")
            if note:
                d.text((x0, y1 - 10), note, font=fonts.get(15), fill=col(FAINT, a), anchor="ld")

        elif kind == "tiles":
            count = ease(t0, t0 + elem.get("count_duration", 0.6))
            tiles_list = elem.get("tiles", [])
            val_sz_default = elem.get("value_size", 40)
            for k, tile in enumerate(tiles_list):
                x0, y0, x1, y1 = tile["rect"]
                t_k0 = t0 + 0.05 * k
                a_tile = ease(t_k0, t_k0 + 0.3)
                if a_tile <= 0.0:
                    continue
                y_off = int((1.0 - a_tile) * 24)
                d.rounded_rectangle(
                    [x0, y0 + y_off, x1, y1 + y_off],
                    radius=8,
                    fill=col(PANEL, a_tile),
                    outline=col(BORDER, a_tile),
                    width=2,
                )

                shown = format_diagram_tile_count(tile["value"], count)
                t_col = tile.get("colour", tile.get("color", "cyan"))
                val_sz = tile.get("value_size", val_sz_default)
                cx = (x0 + x1) // 2
                val_y = y0 + y_off + int((y1 - y0) * 0.32)
                d.text((cx, val_y), shown, font=fonts.get(val_sz, bold=True), fill=col(t_col, a_tile), anchor="mm")

                lines = tile.get("caption", "").split("\n")
                cap_y = val_y + val_sz // 2 + 8
                line_sp = 16 if (y1 - y0) <= 120 else 18
                max_tile_w = (x1 - x0) - 24
                for j, cl in enumerate(lines):
                    cap_sz = tile.get("caption_size", elem.get("caption_size", 15))
                    f_cap = fonts.get(cap_sz)
                    while cap_sz > 11 and f_cap.getlength(cl) > max_tile_w:
                        cap_sz -= 1
                        f_cap = fonts.get(cap_sz)
                    d.text((cx, cap_y + line_sp * j), cl, font=f_cap, fill=col(MUTED, a_tile), anchor="mm")

        elif kind == "table":
            x0, y0, x1, y1 = elem["rect"]
            a_box = ease(t0, t0 + 0.3)
            if a_box > 0.0:
                y_off = int((1.0 - a_box) * 24)
                d.rounded_rectangle(
                    [x0, y0 + y_off, x1, y1 + y_off],
                    radius=8,
                    fill=col(PANEL, a_box),
                    outline=col(BORDER, a_box),
                    width=2,
                )
                rows = elem.get("rows", [])
                stagger = elem.get("stagger", 0.1)
                row_h = (y1 - y0) / max(1, len(rows))
                for k, (dot_c, txt) in enumerate(rows):
                    row_t0 = t0 + stagger * k
                    a_row = ease(row_t0, row_t0 + 0.3)
                    if a_row <= 0.0:
                        continue
                    cy = y0 + y_off + int((k + 0.5) * row_h)
                    dot_r = 5
                    d.ellipse(
                        [x0 + 24 - dot_r, cy - dot_r, x0 + 24 + dot_r, cy + dot_r],
                        fill=col(dot_c, a_row),
                    )
                    d.text((x0 + 44, cy), txt, font=fonts.get(18), fill=col(BODY, a_row), anchor="lm")

        elif kind == "strip":
            a_s = ease(t0, t0 + 0.4)
            if a_s > 0.0:
                lines = elem.get("lines", [])
                y_start = elem.get("y_start", 936)
                dy = elem.get("dy", 32)
                for j, line in enumerate(lines):
                    c = MUTED if j < len(lines) - 1 else FAINT
                    d.text((W // 2, y_start + dy * j), line, font=fonts.get(18), fill=col(c, a_s), anchor="mm")

    # If strip is defined at segment root level (for backward compatibility / convenience)
    if "strip" in seg and not any(elem.get("kind") == "strip" for elem in elements):
        a_s = ease(1.4, 1.8)
        if a_s > 0.0:
            for j, line in enumerate(seg["strip"]):
                c = MUTED if j < len(seg["strip"]) - 1 else FAINT
                d.text((W // 2, 936 + 32 * j), line, font=fonts.get(18), fill=col(c, a_s), anchor="mm")

    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


def render_value_frame(seg: Dict[str, Any], t_s: float, fonts: FontManager, theme: str = "dark") -> np.ndarray:
    """Renders a two-column info screen ('value' segment) showing research and engineering value."""
    W, H = CANVAS_WIDTH, CANVAS_HEIGHT
    seconds = float(seg.get("seconds", 10.0))
    CYAN, GOLD = (56, 189, 248), (250, 204, 21)
    PANEL, BORDER, TITLE, BODY, MUTED, FAINT = (
        (20, 26, 35),
        (45, 58, 75),
        (248, 250, 252),
        (203, 213, 225),
        (148, 163, 184),
        (100, 116, 139),
    )
    DARK_TEXT = (15, 20, 25)
    bg = BG_COLOR
    if theme == "light":  # GOLD stays: the header highlighter reads on both backgrounds
        L = LIGHT_INFO
        CYAN = L["sky"]
        PANEL, BORDER, TITLE, BODY, MUTED, FAINT = L["panel"], L["border"], L["title"], L["body"], L["muted"], L["faint"]
        bg = L["bg"]

    def ease(a: float, b: float) -> float:
        if b <= a:
            return 1.0 if t_s >= a else 0.0
        x = min(1.0, max(0.0, (t_s - a) / (b - a)))
        return x * x * (3.0 - 2.0 * x)

    out = 1.0 - ease(seconds - 0.5, seconds)

    def col(c: Tuple[int, int, int], a: float) -> Tuple[int, int, int, int]:
        return c + (int(255 * max(0.0, min(1.0, a)) * out),)

    def spaced(s: str) -> str:
        return " ".join(s.upper())

    img = Image.new("RGBA", (W, H), bg + (255,))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # Header band with Poké Ball icon, title, title_right, and rule
    a_hdr = ease(0.0, 0.3)
    icon = get_pokeball_icon(28)
    if a_hdr * out < 0.999:
        icon_faded = icon.copy()
        r, g, b, a_ch = icon_faded.split()
        a_ch = a_ch.point(lambda p: int(p * a_hdr * out))
        icon_faded.putalpha(a_ch)
        ov.alpha_composite(icon_faded, (64, 79))
    else:
        ov.alpha_composite(icon, (64, 79))

    title_text = seg.get("title", "PokeDreamer v3")
    d.text((104, 72), title_text, font=fonts.get(44, bold=True), fill=col(TITLE, a_hdr))
    if "title_right" in seg:
        d.text((1856, 92), seg["title_right"], font=fonts.get(22), fill=col(MUTED, a_hdr), anchor="rm")
    d.line([(64, 150), (1856, 150)], fill=col(BORDER, a_hdr), width=2)

    # Headline
    a_head = ease(0.0, 0.4)
    headline = seg.get("headline", "")
    if headline and a_head > 0.0:
        d.text((64, 172), headline, font=fonts.get(34, bold=True), fill=col(TITLE, a_head))

    # Two equal columns
    columns = seg.get("columns", [])
    num_cols = len(columns)
    if num_cols > 0:
        margin_x = 64
        total_w = 1856 - margin_x
        col_gap = 48
        col_w = (total_w - (num_cols - 1) * col_gap) // num_cols
        y_hdr = 226
        f_hdr = fonts.get(18, bold=True)
        a_col_hdr = ease(0.35, 0.65)

        max_items = max((len(c.get("items", [])) for c in columns), default=0)
        item_h = 160 if max_items >= 4 else 190
        item_gap = 16 if max_items >= 4 else 22
        y_items_start = 274

        for c_idx, c_data in enumerate(columns):
            cx0 = margin_x + c_idx * (col_w + col_gap)
            cx1 = cx0 + col_w

            # Column header sits on the yellow highlighter box
            if a_col_hdr > 0.0:
                hdr_raw = c_data.get("header", "")
                htxt = spaced(hdr_raw)
                bb = d.textbbox((cx0 + 14, y_hdr), htxt, font=f_hdr)
                d.rounded_rectangle([cx0, bb[1] - 5, bb[2] + 14, bb[3] + 5], radius=6, fill=col(GOLD, a_col_hdr))
                d.text((cx0 + 14, y_hdr), htxt, font=f_hdr, fill=col(DARK_TEXT, a_col_hdr))

            # Column items: staggered ~0.25 s apart
            items = c_data.get("items", [])
            for r_idx, item in enumerate(items):
                t_item = 0.5 + 0.25 * (r_idx * num_cols + c_idx)
                a_item = ease(t_item, t_item + 0.35)
                if a_item <= 0.0:
                    continue

                iy = y_items_start + r_idx * (item_h + item_gap)
                y_off = int((1.0 - a_item) * 16)
                iy_box = iy + y_off

                # Outer card box
                d.rounded_rectangle(
                    [cx0, iy_box, cx1, iy_box + item_h],
                    radius=8,
                    fill=col(PANEL, a_item),
                    outline=col(BORDER, a_item),
                    width=2,
                )
                # Left accent stripe
                d.rectangle([cx0, iy_box + 8, cx0 + 4, iy_box + item_h - 8], fill=col(CYAN, a_item))

                pad_l = 24
                pad_r = 24
                metric_w = 210
                max_text_w = col_w - pad_l - metric_w - 20

                # Lead: bold white
                lead_str = item.get("lead", "")
                lead_sz = 26 if max_items <= 3 else 24  # readable at 1080p
                f_lead = fonts.get(lead_sz, bold=True)
                while lead_sz > 14 and f_lead.getlength(lead_str) > max_text_w:
                    lead_sz -= 1
                    f_lead = fonts.get(lead_sz, bold=True)
                lead_y = iy_box + (22 if max_items <= 3 else 18)
                d.text((cx0 + pad_l, lead_y), lead_str, font=f_lead, fill=col(TITLE, a_item))

                # Line: body colour, wrapped to at most 2 lines, shrinking font if needed
                line_str = item.get("line", "")
                line_sz = 21 if max_items <= 3 else 20
                f_line = fonts.get(line_sz)
                w_lines = wrap_text(line_str, f_line, max_text_w)
                while len(w_lines) > 2 and line_sz > 10:
                    line_sz -= 1
                    f_line = fonts.get(line_sz)
                    w_lines = wrap_text(line_str, f_line, max_text_w)
                w_lines = w_lines[:2]

                line_y = lead_y + lead_sz + 10
                line_sp = line_sz + 8
                for l_t in w_lines:
                    d.text((cx0 + pad_l, line_y), l_t, font=f_line, fill=col(BODY, a_item))
                    line_y += line_sp

                # Metric and source on the right
                metric_rx = cx1 - pad_r
                has_metric = "metric" in item and item["metric"]
                has_source = "source" in item and item["source"]

                if has_metric:
                    m_str = str(item["metric"])
                    m_sz = 36 if max_items <= 3 else 32
                    f_m = fonts.get(m_sz, bold=True)
                    while m_sz > 16 and f_m.getlength(m_str) > metric_w:
                        m_sz -= 2
                        f_m = fonts.get(m_sz, bold=True)

                    if has_source:
                        my = iy_box + (item_h // 2) - 10
                    else:
                        my = iy_box + (item_h // 2)
                    d.text((metric_rx, my), m_str, font=f_m, fill=col(CYAN, a_item), anchor="rm")

                    if has_source:
                        s_str = str(item["source"])
                        f_s = fonts.get(13)
                        sy = my + (m_sz // 2) + 10
                        d.text((metric_rx, sy), s_str, font=f_s, fill=col(FAINT, a_item), anchor="rm")

    # Footer
    footer = seg.get("footer", "")
    if footer:
        a_foot = ease(0.4, 0.8)
        if a_foot > 0.0:
            d.text((W // 2, 1020), footer, font=fonts.get(18), fill=col(FAINT, a_foot), anchor="mm")

    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


def render_card_frame(
    bg_frame: np.ndarray,
    t: float,
    headline: str,
    line: str,
    fonts: FontManager,
    seconds: float = 2.0,
) -> np.ndarray:
    """Section intro card (owner review 3, Fable's design) at progress t in [0, 1].

    So 2-2.5 s of text is not a blank screen, the next section's own gameplay plays behind it (slowed, blurred,
    muted towards the UI background, slow push-in). Over it, staggered: the section name as the headline sliding up,
    a rule drawing in left to right, then a one-line grey subtitle; everything fades out at the end (owner, 2026-11-08:
    the section name is the heading, not a small tag over a description).
    """
    W, H = CANVAS_WIDTH, CANVAS_HEIGHT

    def ease(a: float, b: float) -> float:  # smoothstep over [a, b] in seconds, so long cards are readable sooner
        x = min(1.0, max(0.0, (t * seconds - a) / (b - a)))
        return x * x * (3.0 - 2.0 * x)

    zoom = 1.0 + 0.06 * t
    big = cv2.resize(bg_frame, (W, int(W * bg_frame.shape[0] / bg_frame.shape[1])), interpolation=cv2.INTER_NEAREST)
    ch, cw = int(H / zoom), int(W / zoom)
    y0, x0 = (big.shape[0] - ch) // 2, (W - cw) // 2
    bg = cv2.resize(big[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), 14)
    bg = (bg.astype(np.float32) * 0.38 + np.array(BG_COLOR, np.float32) * 0.62).astype(np.uint8)

    img = Image.fromarray(bg).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    cx, cy = W // 2, H // 2
    out = 1.0 - ease(seconds - 0.5, seconds)
    a_head, a_line = ease(0.0, 0.6) * out, ease(0.4, 1.1) * out
    f_head, f_line = fonts.get(60, bold=True), fonts.get(28)

    hy = cy - 40 + int((1.0 - ease(0.0, 0.6)) * 24)
    d.text((cx, hy), headline, font=f_head, fill=(248, 250, 252, int(255 * a_head)), anchor="mm")
    hw = int(d.textlength(headline, font=f_head))
    rw = int(hw * ease(0.4, 1.0))
    if rw > 0:
        d.rectangle([cx - hw // 2, cy + 12, cx - hw // 2 + rw, cy + 14], fill=(147, 197, 253, int(255 * out)))
    d.text((cx, cy + 56), line, font=f_line, fill=(190, 200, 214, int(255 * a_line)), anchor="mm")
    img.alpha_composite(ov)
    return np.asarray(img.convert("RGB"))


def render_bakeoff_frame(
    rows: List[Dict[str, Any]],
    title: str,
    footer: str,
    fonts: FontManager,
    fade_alpha: float = 1.0,
) -> np.ndarray:
    """Renders the full-frame bakeoff card with 7-row local LLM comparison table."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    cx, cy = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2

    card_w, card_h = 1420, 720
    x0 = cx - card_w // 2
    x1 = cx + card_w // 2
    y0 = cy - card_h // 2
    y1 = cy + card_h // 2

    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(20, 26, 35), outline=(50, 65, 85), width=2)

    # Heading band
    draw_heading(img, draw, fonts, cx=cx, cy=y0 + 38, text="PokeDreamer · System 2 bake-off", font_size=20, icon_size=22, fill=(100, 116, 139))
    draw.line([(x0 + 50, y0 + 64), (x1 - 50, y0 + 64)], fill=(34, 45, 58), width=1)

    # Title
    draw.text((cx, y0 + 104), title, font=fonts.get(32, bold=True), fill=(248, 250, 252), anchor="mm")

    # Table columns
    col_planner_x = x0 + 80
    col_badges_x = x0 + 590
    col_presses_x = x0 + 910
    col_vram_x = x1 - 80

    th_y = y0 + 155
    f_th = fonts.get(16, bold=True)
    c_th = (148, 163, 184)
    draw.text((col_planner_x, th_y), "PLANNER", font=f_th, fill=c_th, anchor="lm")
    draw.text((col_badges_x, th_y), "BADGES / 10", font=f_th, fill=c_th, anchor="mm")
    draw.text((col_presses_x, th_y), "MEDIAN PRESSES", font=f_th, fill=c_th, anchor="rm")
    draw.text((col_vram_x, th_y), "PEAK GPU MEMORY", font=f_th, fill=c_th, anchor="rm")

    draw.line([(x0 + 50, y0 + 180), (x1 - 50, y0 + 180)], fill=(45, 58, 75), width=1)

    # Rows
    row_start_y = y0 + 218
    row_step = 54
    logos_dir = Path("work/video/logos")

    for i, r in enumerate(rows):
        ry = row_start_y + i * row_step
        is_hl = bool(r.get("highlight", False))

        if is_hl:
            draw.rounded_rectangle([x0 + 50, ry - 18, x1 - 50, ry + 26], radius=8, fill=(28, 48, 72), outline=(56, 189, 248), width=2)
            c_planner = (255, 255, 255)
            c_badges = (250, 204, 21)
            c_presses = (56, 189, 248)
            c_vram = (248, 250, 252)
            f_row = fonts.get(22, bold=True)
        else:
            c_planner = (180, 195, 215) if i == 0 else (226, 232, 240)
            c_badges = (250, 204, 21) if r.get("badges") == "10/10" else (203, 213, 225)
            c_presses = (226, 232, 240)
            c_vram = (148, 163, 184)
            f_row = fonts.get(21, bold=(i == 0))

        # Logo check
        key = r.get("key", "")
        logo_path = logos_dir / f"{key}.png"
        text_x = col_planner_x
        if key and logo_path.exists():
            try:
                logo_im = Image.open(logo_path).convert("RGBA")
                logo_im = logo_im.resize((26, 26), resample=getattr(Image, "Resampling", Image).LANCZOS)
                img.paste(logo_im, (text_x, ry - 13), logo_im)
                text_x += 34
            except Exception:
                pass

        draw.text((text_x, ry), r.get("planner", ""), font=f_row, fill=c_planner, anchor="lm")
        draw.text((col_badges_x, ry), r.get("badges", ""), font=f_row, fill=c_badges, anchor="mm")
        draw.text((col_presses_x, ry), r.get("presses", ""), font=f_row, fill=c_presses, anchor="rm")
        draw.text((col_vram_x, ry), r.get("vram", ""), font=f_row, fill=c_vram, anchor="rm")

    # Divider before footer
    draw.line([(x0 + 50, y1 - 66), (x1 - 50, y1 - 66)], fill=(34, 45, 58), width=1)
    draw.text((cx, y1 - 34), footer, font=fonts.get(18), fill=(148, 163, 184), anchor="mm")

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
    scale: int = PLAY_SCALE,
    progress_text: Optional[str] = None,
    subcaption: Optional[str] = None,
    is_llm_planner: bool = False,
    planner_decision: Optional[Dict[str, Any]] = None,
    is_decision_highlighted: bool = False,
) -> np.ndarray:
    """Renders a standard play or montage frame with Game Boy canvas and left HUD."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fw = 160 * scale
    fh = 144 * scale

    if scale == 6:
        fx = CANVAS_WIDTH - PLAY_OUTER_MARGIN - fw
        fy = PLAY_FRAME_Y
        hud_box = (PLAY_OUTER_MARGIN, fy, fx - PLAY_GAP, fy + fh)
    elif scale == 5:
        fx = 760
        fy = 150
        hud_box = (80, 150, 680, 150 + fh)
    else:  # scale == 4
        fx = 640
        fy = 190
        hud_box = (100, 190, 560, 190 + fh)

    # 1. Heading band at the top centre
    draw_heading(img, draw, fonts, cx=CANVAS_WIDTH // 2, cy=HEADING_Y, text="PokeDreamer", font_size=HEADING_FONT_SIZE)

    # 2. Scale and paste game frame
    scaled = cv2.resize(color_frame, (fw, fh), interpolation=cv2.INTER_NEAREST)
    img.paste(Image.fromarray(scaled), (fx, fy))

    # 3. Rounded border around game frame
    draw.rounded_rectangle([fx - 3, fy - 3, fx + fw + 3, fy + fh + 3], radius=10, outline=(60, 75, 95), width=2)

    # 4. HUD on the left column
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
        is_llm_planner=is_llm_planner,
        planner_decision=planner_decision,
        is_decision_highlighted=is_decision_highlighted,
        img=img,
    )

    # 5. Caption centred under the game frame
    if caption or subcaption:
        cap_x = fx + fw // 2
        if subcaption:
            cap_y = fy + fh + (42 if scale == 6 else 44)
            sub_y = cap_y + 30
            if caption:
                sz = 26
                while sz > 18 and draw.textlength(caption, font=fonts.get(sz, bold=True)) > fw:
                    sz -= 1
                draw.text((cap_x, cap_y), caption, font=fonts.get(sz, bold=True), fill=(241, 245, 249), anchor="mm")
            sz_sub = 20
            while sz_sub > 14 and draw.textlength(subcaption, font=fonts.get(sz_sub)) > fw:
                sz_sub -= 1
            draw.text((cap_x, sub_y), subcaption, font=fonts.get(sz_sub), fill=(148, 163, 184), anchor="mm")
        else:
            cap_y = fy + fh + (58 if scale == 6 else 60)
            sz = 24
            while sz > 16 and draw.textlength(caption, font=fonts.get(sz, bold=True)) > fw:
                sz -= 1
            draw.text((cap_x, cap_y), caption, font=fonts.get(sz, bold=True), fill=(241, 245, 249), anchor="mm")

    return np.asarray(img)


def render_split_frame(
    real_frame: np.ndarray,
    dream_frame: Optional[np.ndarray],
    caption: str,
    fonts: FontManager,
    is_anchor_flashing: bool = False,
    scale: int = SPLIT_SCALE,
    subcaption: Optional[str] = None,
) -> np.ndarray:
    """Renders side-by-side real vs dream frame comparison (each 5x)."""
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fw, fh = 160 * scale, 144 * scale
    lx = SPLIT_MARGIN_X
    rx = lx + fw + SPLIT_GAP
    ly = ry = SPLIT_FRAME_Y

    # Heading band at top centre
    draw_heading(img, draw, fonts, cx=CANVAS_WIDTH // 2, cy=HEADING_Y, text="PokeDreamer", font_size=HEADING_FONT_SIZE)

    # 1. Labels above frames
    draw.text((lx + fw // 2, ly - 36), "The real game", font=fonts.get(24, bold=True), fill=(226, 232, 240), anchor="mm")

    f_main = fonts.get(24, bold=True)
    f_sub = fonts.get(14)
    main_txt = "The world model's imagination"
    sub_txt = "diffusion world model, same button presses"
    w_main = draw.textlength(main_txt, font=f_main)
    w_sub = draw.textlength(sub_txt, font=f_sub)
    gap = 12
    total_w = w_main + gap + w_sub
    start_x = (rx + fw // 2) - total_w / 2

    draw.text((start_x, ly - 36), main_txt, font=f_main, fill=(96, 165, 250), anchor="lm")
    draw.text((start_x + w_main + gap, ly - 36), sub_txt, font=f_sub, fill=(148, 163, 184), anchor="lm")

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
            tag_text = "re-anchor: 4 real context frames"
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
    if caption or subcaption:
        cap_x = CANVAS_WIDTH // 2
        if subcaption:
            cap_y = ly + fh + 48
            sub_y = cap_y + 32
            if caption:
                sz = 26
                while sz > 18 and draw.textlength(caption, font=fonts.get(sz, bold=True)) > CANVAS_WIDTH - 80:
                    sz -= 1
                draw.text((cap_x, cap_y), caption, font=fonts.get(sz, bold=True), fill=(241, 245, 249), anchor="mm")
            sz_sub = 20
            while sz_sub > 14 and draw.textlength(subcaption, font=fonts.get(sz_sub)) > CANVAS_WIDTH - 80:
                sz_sub -= 1
            draw.text((cap_x, sub_y), subcaption, font=fonts.get(sz_sub), fill=(148, 163, 184), anchor="mm")
        else:
            cap_y = ly + fh + 60
            sz = 24
            while sz > 16 and draw.textlength(caption, font=fonts.get(sz, bold=True)) > CANVAS_WIDTH - 80:
                sz -= 1
            draw.text((cap_x, cap_y), caption, font=fonts.get(sz, bold=True), fill=(241, 245, 249), anchor="mm")

    return np.asarray(img)


def load_storyboard(path: str | Path) -> list:
    """The storyboard is UTF-8 (arrows, middle dots, é); Windows' default cp1252 would garble them on screen."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compose_video(
    storyboard_path: str | Path,
    out_path: str | Path,
    scale: int = PLAY_SCALE,
    traj_color_path: str | Path = "work/video/traj/run_seed_100_color.npz",
    traj_grey_path: str | Path = "work/video/traj/run_seed_100.npz",
    dream_path: str | Path = "work/video/dream_4_8718_k10.npz",
    quest_json_path: str | Path = "work/video/quest_s100/run_seed_100.json",
    fps: int = DEFAULT_FPS,
    stills_dir: Optional[str | Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Renders the complete 60s video according to the storyboard specification."""
    storyboard = load_storyboard(storyboard_path)

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
    flash_main = battle_flash_mask(colors, in_battles)

    # Load map name resolver
    try:
        from lab.dwm.eval_rollouts import get_map_name
    except ImportError:
        def get_map_name(mid: int) -> str:
            return f"Map {mid}"

    # The run records the level only at planner decisions, so a "live" level would lag behind the battles on screen;
    # the montage shows the grind's span instead (start level from the run JSON -> target 14).
    start_level = 8
    if os.path.exists(quest_json_path):
        with open(quest_json_path, "r", encoding="utf-8") as f:
            start_level = json.load(f).get("level_start", 8)

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
        cap = s.get("caption", s.get("tag", s.get("headline", s.get("src", ""))))
        if len(cap) > 30:
            cap = cap[:27] + "..."
        start_val = s.get("start", s.get("start_frame", 0))
        end_val = s.get("end", s.get("end_frame", 0))
        sec_val = s.get("seconds", seg_frames[i] / fps)
        print(
            f"{i+1:<3} {s['type']:<10} {start_val:<7} {end_val:<7} "
            f"{str(s.get('speed', 'fit')):<8} {seg_frames[i]:<8} {sec_val:<6.1f} {cap}"
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

        if seg_type == "clip":
            seg_out = load_clip_frames(
                src=seg["src"],
                start_frame=int(seg.get("start_frame", 0)),
                end_frame=int(seg.get("end_frame", 0)),
                hold_s=float(seg.get("hold_s", 0.0)),
                fps=fps,
            )
            rendered_segments.append(seg_out)
            mid_idx = n_frames // 2
            still_key = f"segment_{seg_idx+1:02d}_{seg_type}"
            stills_to_save[still_key] = seg_out[mid_idx]
            if seg_type not in stills_to_save:
                stills_to_save[seg_type] = seg_out[mid_idx]
            continue

        speed = seg.get("speed", "fit")
        ranges = seg.get("ranges", None)
        start_step = seg.get("start", 0)
        end_step = seg.get("end", 0)
        caption = seg.get("caption", "")
        subcaption = seg.get("subcaption", None)
        text_lines = seg.get("text", [])

        steps = compute_step_indices(start_step, end_step, n_frames, speed=speed, ranges=ranges)
        if seg_type in ("play", "montage", "split"):  # split: real and dream use these same steps, so both hold together
            steps = deflash_steps(steps, flash_main)
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

        # Harness-specific data
        harness_colors = None
        harness_map_ids = None
        harness_goals = None
        harness_in_battles = None
        harness_legs = None
        harness_bys = None
        harness_timeline: List[Dict[str, Any]] = []
        if seg_type == "harness":
            harness_traj_path = seg.get("traj", "work/video/traj_llm/run_seed_100_color.npz")
            harness_json_path = seg.get("run_json", "work/video/quest_s100_llm/run_seed_100.json")
            print(f"Loading harness trajectory: {harness_traj_path}")
            h_npz = np.load(harness_traj_path)
            harness_colors = h_npz["color"]
            harness_map_ids = h_npz["map_ids"]
            harness_goals = h_npz["goals"]
            harness_in_battles = h_npz["in_battles"]
            harness_legs = h_npz["leg"]
            harness_bys = h_npz["by"]
            steps = deflash_steps(steps, battle_flash_mask(harness_colors, harness_in_battles))
            if os.path.exists(harness_json_path):
                with open(harness_json_path, "r", encoding="utf-8") as f:
                    harness_timeline = json.load(f).get("timeline", [])

        anchor_flash_countdown = 0
        decision_highlight_countdown = 0

        for f_idx in range(n_frames):
            st = steps[f_idx]
            prev_st = steps[max(0, f_idx - 1)]

            if seg_type == "harness":
                assert harness_colors is not None
                st_clamped = min(st, len(harness_colors) - 1)
                cf = harness_colors[st_clamped]
                m_id = int(harness_map_ids[st_clamped])
                m_name = get_map_name(m_id)
                g_str = str(harness_goals[st_clamped])
                l_str = str(harness_legs[st_clamped])
                bat = bool(harness_in_battles[st_clamped])
                by_str = str(harness_bys[st_clamped])

                if f_idx == 0:
                    crossed = any(entry.get("presses", 0) == st for entry in harness_timeline)
                else:
                    low_st = min(prev_st, st)
                    high_st = max(prev_st, st)
                    crossed = any(low_st < entry.get("presses", 0) <= high_st for entry in harness_timeline)

                if crossed:
                    decision_highlight_countdown = 20

                is_highlighted = decision_highlight_countdown > 0
                if decision_highlight_countdown > 0:
                    decision_highlight_countdown -= 1

                dec_entry = get_latest_decision(harness_timeline, st_clamped)

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
                    subcaption=subcaption,
                    is_llm_planner=True,
                    planner_decision=dec_entry,
                    is_decision_highlighted=is_highlighted,
                )

                if is_highlighted and "harness_decision" not in stills_to_save:
                    stills_to_save["harness_decision"] = frame
                elif not is_highlighted and "harness_between" not in stills_to_save and f_idx > 30:
                    stills_to_save["harness_between"] = frame

            elif seg_type == "spec_end":
                frame = render_spec_end_frame(seg, f_idx / fps, fonts)
                if f_idx == n_frames // 2:
                    stills_to_save["spec_end"] = frame

            elif seg_type == "diagram":
                frame = render_diagram_frame(seg, f_idx / fps, fonts)
                if f_idx == n_frames // 2:
                    stills_to_save[f"diagram_{seg_idx:02d}"] = frame

            elif seg_type == "value":
                frame = render_value_frame(seg, f_idx / fps, fonts)
                if f_idx == n_frames // 2:
                    stills_to_save[f"value_{seg_idx:02d}"] = frame

            elif seg_type == "card":
                if f_idx == 0:
                    bg_path = seg.get("bg_traj", traj_color_path)
                    bg_npz = np.load(bg_path) if bg_path != traj_color_path else None
                    bg_colors = bg_npz["color"] if bg_npz is not None else colors
                    bg_flash = battle_flash_mask(bg_colors, bg_npz["in_battles"]) if bg_npz is not None else flash_main
                    bg_steps = deflash_steps(np.linspace(seg["bg_start"], seg["bg_end"], n_frames).round().astype(int), bg_flash)
                frame = render_card_frame(
                    bg_colors[bg_steps[f_idx]],
                    f_idx / max(1, n_frames - 1),
                    headline=seg.get("headline", ""),
                    line=seg.get("line", ""),
                    fonts=fonts,
                    seconds=float(seg["seconds"]),
                )
                if f_idx == int(n_frames * 0.6):
                    stills_to_save[f"card_{seg_idx:02d}"] = frame

            elif seg_type == "bakeoff":
                fade = 1.0
                if f_idx < 15:
                    fade = (f_idx + 1) / 15.0
                elif f_idx >= n_frames - 15:
                    fade = (n_frames - f_idx) / 15.0
                frame = render_bakeoff_frame(
                    rows=seg.get("rows", []),
                    title=seg.get("title", "Six local LLMs tested as the planner"),
                    footer=seg.get("footer", ""),
                    fonts=fonts,
                    fade_alpha=fade,
                )
                if f_idx == n_frames // 2:
                    stills_to_save["bakeoff"] = frame

            else:
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
                    frame = render_title_frame(cf, text_lines, caption, fonts, fade_alpha=fade, subcaption=subcaption)

                elif seg_type == "end":
                    fade = 1.0
                    if f_idx < 15:
                        fade = (f_idx + 1) / 15.0
                    elif f_idx >= n_frames - 15:
                        fade = (n_frames - f_idx) / 15.0
                    frame = render_end_frame(text_lines, caption, fonts, fade_alpha=fade,
                                             heading=seg.get("heading", END_HEADING), footer=seg.get("footer", END_FOOTER))

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
                        subcaption=subcaption,
                    )
                    if anchor_flash_countdown > 0:
                        anchor_flash_countdown -= 1

                elif seg_type == "montage":
                    prog_str = seg.get("progress", f"Level {start_level} → 14")  # per-excerpt level span from the run timeline
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
                        subcaption=subcaption,
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
                        subcaption=subcaption,
                    )

                if seg_type == "play":
                    if bat and "play_battle" not in stills_to_save:
                        stills_to_save["play_battle"] = frame
                    elif not bat and by_str.lower() == "policy" and "play_policy" not in stills_to_save:
                        stills_to_save["play_policy"] = frame
                    elif not bat and by_str == "" and "play_scripted" not in stills_to_save:
                        stills_to_save["play_scripted"] = frame

            seg_out.append(frame)

        rendered_segments.append(seg_out)

        # Save representative still for this segment
        mid_idx = n_frames // 2
        still_key = f"segment_{seg_idx+1:02d}_{seg_type}"
        stills_to_save[still_key] = seg_out[mid_idx]
        if seg_type not in stills_to_save:
            stills_to_save[seg_type] = seg_out[mid_idx]

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
    parser.add_argument("--traj-color", default="work/video/traj/run_seed_100_color.npz")
    parser.add_argument("--traj-grey", default="work/video/traj/run_seed_100.npz")
    parser.add_argument("--dream", default="work/video/dream_4_8718_k10.npz")
    parser.add_argument("--stills-dir", default=None, help="Directory to save stills")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    compose_video(
        storyboard_path=args.storyboard,
        out_path=args.out,
        scale=PLAY_SCALE,
        traj_color_path=args.traj_color,
        traj_grey_path=args.traj_grey,
        dream_path=args.dream,
        fps=args.fps,
        stills_dir=args.stills_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
