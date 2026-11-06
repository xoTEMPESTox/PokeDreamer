"""Tests for lab.video.compose and lab.video.storyboard.json."""
import json
from pathlib import Path

import numpy as np
import pytest

from lab.video.compose import (
    calculate_segment_frames,
    compute_step_indices,
    get_hud_badge,
    render_title_frame,
    render_play_frame,
    render_end_frame,
    render_split_frame,
    draw_heading,
    get_pokeball_icon,
    FontManager,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    PLAY_SCALE,
    GAME_W,
    GAME_H,
    PLAY_OUTER_MARGIN,
    PLAY_GAP,
    PANEL_W,
    PANEL_H,
    PLAY_FRAME_Y,
    HEADING_Y,
    HEADING_FONT_SIZE,
    HEADING_ICON_SIZE,
    SPLIT_SCALE,
    SPLIT_FRAME_W,
    SPLIT_FRAME_H,
    SPLIT_GAP,
    SPLIT_MARGIN_X,
    SPLIT_FRAME_Y,
    BG_COLOR,
)


def test_segment_frame_counts_sum_to_seconds_times_fps():
    """Verify segment frame counts sum to requested seconds * fps on toy storyboard."""
    toy_storyboard = [
        {"type": "title", "seconds": 2.5},
        {"type": "play", "seconds": 3.0},
        {"type": "split", "seconds": 4.5},
    ]
    fps = 30
    counts = calculate_segment_frames(toy_storyboard, fps=fps)
    assert counts == [75, 90, 135]
    assert sum(counts) == int(round(10.0 * fps)) == 300


def test_actual_storyboard_json_frame_counts():
    """Verify lab/video/storyboard.json totals exactly 60.0 s (1800 frames at 30 fps)."""
    sb_path = Path("lab/video/storyboard.json")
    assert sb_path.exists(), "lab/video/storyboard.json must exist"

    with open(sb_path, "r") as f:
        storyboard = json.load(f)

    fps = 30
    counts = calculate_segment_frames(storyboard, fps=fps)
    assert sum(counts) == 1800, f"Expected 1800 frames, got {sum(counts)}"
    total_seconds = sum(s["seconds"] for s in storyboard)
    assert abs(total_seconds - 60.0) < 1e-6, f"Expected 60.0 seconds, got {total_seconds}"


def test_storyboard_end_card_exact_numbers():
    """Verify end card has exact required numbers and phrases."""
    sb_path = Path("lab/video/storyboard.json")
    with open(sb_path, "r") as f:
        storyboard = json.load(f)

    end_segments = [s for s in storyboard if s.get("type") == "end"]
    assert len(end_segments) >= 1, "Storyboard must have at least one end segment"
    end_text = end_segments[0].get("text", [])

    assert "30/30 badges" in end_text
    assert "14/30 runs with every navigation leg learned in imagination" in end_text
    assert "0 real-game steps of navigation training" in end_text


def test_fit_speed_maps_step_range():
    """Verify 'fit' speed maps a step range onto the segment's requested frames."""
    # Toy range [0, 9] onto 5 frames
    steps_5 = compute_step_indices(start=0, end=9, num_frames=5, speed="fit")
    assert len(steps_5) == 5
    assert steps_5[0] == 0
    assert steps_5[-1] == 9
    assert np.all(np.diff(steps_5) >= 0)  # monotonically increasing

    # Single-step hold (e.g. title or still)
    steps_still = compute_step_indices(start=78, end=78, num_frames=10, speed="fit")
    assert len(steps_still) == 10
    assert np.all(steps_still == 78)

    # Multi-range montage
    ranges = [[10, 20], [50, 60]]
    steps_multi = compute_step_indices(start=10, end=60, num_frames=8, speed="fit", ranges=ranges)
    assert len(steps_multi) == 8
    assert steps_multi[0] == 10
    assert steps_multi[-1] == 60
    assert np.all(np.diff(steps_multi) >= 0)


def test_numeric_speed_mapping():
    """Verify numeric step progression when speed is specified in steps per frame."""
    steps = compute_step_indices(start=100, end=200, num_frames=6, speed=2.0)
    assert len(steps) == 6
    assert np.array_equal(steps, [100, 102, 104, 106, 108, 110])


def test_hud_badge_text_and_priority():
    """Verify HUD badge text and colors for in_battle / bfs / policy."""
    # 1. in_battle=True -> amber "rules: battle"
    text, bg, border, fg = get_hud_badge(in_battle=True, by="policy")
    assert text == "rules: battle"

    text, bg, border, fg = get_hud_badge(in_battle=True, by="bfs")
    assert text == "rules: battle"

    # 2. in_battle=False, by="bfs" -> grey "path-finding fallback"
    text, bg, border, fg = get_hud_badge(in_battle=False, by="bfs")
    assert text == "path-finding fallback"

    # 3. in_battle=False, by="policy" or empty -> green "world-model policy"
    text, bg, border, fg = get_hud_badge(in_battle=False, by="policy")
    assert text == "world-model policy"

    text, bg, border, fg = get_hud_badge(in_battle=False, by="")
    assert text == "world-model policy"


def test_toy_rendering_dimensions_and_types():
    """Verify render functions generate frames of exactly (1080, 1920, 3) uint8."""
    fonts = FontManager()
    toy_frame = np.zeros((144, 160, 3), dtype=np.uint8)

    # Play frame (6x layout)
    play_img = render_play_frame(
        color_frame=toy_frame,
        map_name="Viridian City",
        goal="grind",
        leg="to_grass",
        in_battle=False,
        by="policy",
        step=100,
        caption="Walking test",
        fonts=fonts,
        scale=PLAY_SCALE,
    )
    assert play_img.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert play_img.dtype == np.uint8

    # Title frame
    title_img = render_title_frame(
        bg_frame=toy_frame,
        text_lines=["Title", "Subtitle"],
        caption="Title caption",
        fonts=fonts,
    )
    assert title_img.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert title_img.dtype == np.uint8

    # End frame
    end_img = render_end_frame(
        text_lines=["30/30 badges", "0 steps"],
        caption="",
        fonts=fonts,
    )
    assert end_img.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert end_img.dtype == np.uint8

    # Split frame
    split_img = render_split_frame(
        real_frame=toy_frame,
        dream_frame=toy_frame,
        caption="Split caption",
        fonts=fonts,
        is_anchor_flashing=True,
    )
    assert split_img.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert split_img.dtype == np.uint8


def test_play_layout_geometry_centering_and_bounds():
    """Verify 6x play layout: game frame at 960x864, status panel enlarged, centred block, 40-60px margins."""
    assert PLAY_SCALE == 6
    assert GAME_W == 960
    assert GAME_H == 864
    assert PANEL_H == 864
    assert 40 <= PLAY_OUTER_MARGIN <= 60

    # Total block width = panel + gap + game
    total_block_w = PANEL_W + PLAY_GAP + GAME_W
    # Block is horizontally centered: left margin == right margin
    left_margin = PLAY_OUTER_MARGIN
    right_margin = CANVAS_WIDTH - (left_margin + total_block_w)
    assert left_margin == right_margin, f"Block not centered: left={left_margin}, right={right_margin}"
    assert (left_margin + left_margin + total_block_w) / 2 == CANVAS_WIDTH / 2 == 960

    # Vertical fit: heading + gap + game/panel + gap + caption <= CANVAS_HEIGHT
    assert PLAY_FRAME_Y == 94
    assert PLAY_FRAME_Y + GAME_H == 958
    assert HEADING_Y == 47
    assert HEADING_Y < PLAY_FRAME_Y
    # Space above heading and below heading are balanced
    assert HEADING_Y > 20
    assert PLAY_FRAME_Y - HEADING_Y > 20

    # Caption is below the game frame and comfortably inside the canvas
    caption_y = PLAY_FRAME_Y + GAME_H + 58  # 1016
    assert caption_y < CANVAS_HEIGHT
    bottom_margin = CANVAS_HEIGHT - caption_y
    assert bottom_margin > 40, f"Caption too close to canvas bottom: {bottom_margin}px"


def test_split_layout_geometry_centering_and_bounds():
    """Verify split layout: 5x frames side by side (800x720 each), centred in canvas with labels."""
    assert SPLIT_SCALE == 5
    assert SPLIT_FRAME_W == 800
    assert SPLIT_FRAME_H == 720

    two_frames_w = 2 * SPLIT_FRAME_W + SPLIT_GAP
    left_margin = SPLIT_MARGIN_X
    right_margin = CANVAS_WIDTH - (left_margin + two_frames_w)
    assert left_margin == right_margin, f"Split frames not centered: left={left_margin}, right={right_margin}"

    # Vertical fit
    assert SPLIT_FRAME_Y + SPLIT_FRAME_H <= CANVAS_HEIGHT
    caption_y = SPLIT_FRAME_Y + SPLIT_FRAME_H + 60
    assert caption_y < CANVAS_HEIGHT


def test_procedural_pokeball_icon():
    """Verify Poké Ball icon is procedurally drawn with PIL primitives and has expected dimensions."""
    icon = get_pokeball_icon(42)
    assert icon.size == (42, 42)
    assert icon.mode == "RGBA"

    arr = np.asarray(icon)
    # Check top half has red-dominant pixels
    top_half = arr[:15, 10:32]
    # Red channel significantly higher than blue/green
    has_red = np.any((top_half[..., 0] > 180) & (top_half[..., 1] < 100) & (top_half[..., 2] < 100))
    assert has_red, "Poké Ball top half must contain red pixels"

    # Check bottom half has white/light pixels
    bottom_half = arr[27:40, 10:32]
    has_white = np.any((bottom_half[..., 0] > 200) & (bottom_half[..., 1] > 200) & (bottom_half[..., 2] > 200))
    assert has_white, "Poké Ball bottom half must contain white pixels"


def test_heading_band_present_in_rendered_frames():
    """Verify the PokeDreamer heading band is rendered on play, montage, and split frames."""
    fonts = FontManager()
    toy_frame = np.zeros((144, 160, 3), dtype=np.uint8)

    # 1. Play frame
    play_frame = render_play_frame(
        color_frame=toy_frame,
        map_name="Pallet Town",
        goal="explore",
        leg="to_route1",
        in_battle=False,
        by="policy",
        step=50,
        caption="Testing heading",
        fonts=fonts,
    )
    # Heading region: HEADING_Y +/- 15, centered horizontally
    heading_region = play_frame[HEADING_Y - 15 : HEADING_Y + 15, CANVAS_WIDTH // 2 - 150 : CANVAS_WIDTH // 2 + 150]
    # Should not be uniform background
    diff_from_bg = np.abs(heading_region.astype(int) - np.array(BG_COLOR))
    assert np.any(diff_from_bg > 30), "Play frame heading region must contain rendered heading"

    # 2. Split frame
    split_frame = render_split_frame(
        real_frame=toy_frame,
        dream_frame=toy_frame,
        caption="Split heading test",
        fonts=fonts,
    )
    heading_region_split = split_frame[HEADING_Y - 15 : HEADING_Y + 15, CANVAS_WIDTH // 2 - 150 : CANVAS_WIDTH // 2 + 150]
    diff_from_bg_split = np.abs(heading_region_split.astype(int) - np.array(BG_COLOR))
    assert np.any(diff_from_bg_split > 30), "Split frame heading region must contain rendered heading"

    # 3. Montage frame (play frame with progress_text)
    montage_frame = render_play_frame(
        color_frame=toy_frame,
        map_name="Viridian Forest",
        goal="grind",
        leg="to_grass",
        in_battle=False,
        by="policy",
        step=2000,
        caption="Montage heading test",
        fonts=fonts,
        progress_text="Level 8 -> 14",
    )
    heading_region_montage = montage_frame[HEADING_Y - 15 : HEADING_Y + 15, CANVAS_WIDTH // 2 - 150 : CANVAS_WIDTH // 2 + 150]
    diff_from_bg_montage = np.abs(heading_region_montage.astype(int) - np.array(BG_COLOR))
    assert np.any(diff_from_bg_montage > 30), "Montage frame heading region must contain rendered heading"
