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
    FontManager,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
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

    # Play frame
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
        scale=4,
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
