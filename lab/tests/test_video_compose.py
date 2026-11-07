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
    render_bakeoff_frame,
    render_diagram_frame,
    render_value_frame,
    load_clip_frames,
    truncate_reason,
    get_latest_decision,
    load_storyboard,
    wrap_text,
    draw_heading,
    get_pokeball_icon,
    FontManager,
    GOAL_MAP,
    LEG_MAP,
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
    """Verify lab/video/storyboard.json totals exactly 130.87 s (3926 frames at 30 fps) across 21 segments."""
    sb_path = Path("lab/video/storyboard.json")
    assert sb_path.exists(), "lab/video/storyboard.json must exist"

    with open(sb_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    fps = 30
    counts = calculate_segment_frames(storyboard, fps=fps)
    assert sum(counts) == 3926, f"Expected 3926 frames, got {sum(counts)}"
    total_seconds = sum(s["seconds"] for s in storyboard)
    assert abs(total_seconds - 130.867) < 1e-3, f"Expected 130.867 seconds, got {total_seconds}"


def test_storyboard_end_card_exact_numbers():
    """The three diagram screens (8 s + 9 s + 8 s: world model, harness, numbers).
    Verifies chart numbers (8779.5 / 7144.5, 50.0 GPU-hours), tiles, and
    honesty phrases in strip/table."""
    with open(Path("lab/video/storyboard.json"), "r", encoding="utf-8") as f:
        storyboard = json.load(f)
    diagram_segs = [s for s in storyboard if s.get("type") == "diagram"]
    assert len(diagram_segs) == 3
    assert [s["seconds"] for s in diagram_segs] == [8.0, 9.0, 8.0]

    screen3 = diagram_segs[2]
    strip = " ".join(screen3["strip"])
    assert "hand-written script" in strip and "rule-based planner" in strip and "zero real-game training steps" in strip
    assert [tl[0] for tl in screen3["tiles"]] == ["30/30", "91.4%", "99.27%"]
    assert [r[1] for r in screen3["compare"]["rows"]] == [8779.5, 7144.5]   # lab/results/v3_system2.md
    assert round(sum(r[1] for r in screen3["bars"]["rows"]), 1) == 50.0       # lab/results/compute.md

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
    """Verify HUD badge text and colors for in_battle / bfs / policy / scripted."""
    # 1. in_battle=True -> amber "Battle rules"
    text, bg, border, fg = get_hud_badge(in_battle=True, by="policy")
    assert text == "Battling"

    text, bg, border, fg = get_hud_badge(in_battle=True, by="bfs")
    assert text == "Battling"

    # 2. in_battle=False, by="bfs" -> grey "Backup path-finder"
    text, bg, border, fg = get_hud_badge(in_battle=False, by="bfs")
    assert text == "Backup path-finder (BFS)"

    # 3. in_battle=False, by="policy" -> green "AI trained in the dream"
    text, bg, border, fg = get_hud_badge(in_battle=False, by="policy")
    assert text == "System 1 policy, dream-trained"

    # 4. in_battle=False, by="" (no navigation leg pressing) -> grey scripted badge, not the policy
    text, bg, border, fg = get_hud_badge(in_battle=False, by="")
    assert text == "Menus, dialogue, waiting"


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


def test_hud_badge_does_not_credit_the_policy_for_scripted_steps():
    # by == "" outside battle means no navigation leg is pressing (menus, dialogue, the nurse-counter walk, grind
    # pacing): 2,757 of the 8,718 recorded steps. Showing "AI trained in the dream" there would overclaim.
    text, _, _, _ = get_hud_badge(in_battle=False, by="")
    assert "dream-trained" not in text.lower() and "menus" in text.lower()
    assert "dream-trained" in get_hud_badge(in_battle=False, by="policy")[0].lower()


@pytest.mark.skipif(not Path("work/video/traj/run_seed_100_color.npz").exists(),
                    reason="recorded run not present (work/ is not in the repo)")
def test_mapping_dicts_cover_all_trajectory_legs_and_goals():
    """Verify GOAL_MAP and LEG_MAP cover every leg and goal in run_seed_100_color.npz."""
    traj_path = Path("work/video/traj/run_seed_100_color.npz")

    data = np.load(traj_path)
    unique_goals = set(str(g) for g in data["goals"])
    unique_legs = set(str(l) for l in data["leg"])

    for g in unique_goals:
        assert g in GOAL_MAP, f"Goal {g!r} missing from GOAL_MAP"
        assert GOAL_MAP[g] != "", f"Goal {g!r} mapped to empty string"

    for l in unique_legs:
        assert l in LEG_MAP, f"Leg {l!r} missing from LEG_MAP"
        assert LEG_MAP[l] != "", f"Leg {l!r} mapped to empty string"


def test_storyboard_subcaptions_present_and_match_spec():
    """Every play/montage/split segment has a caption; subcaptions are optional since the intro cards (owner
    review 3) carry the context, but never empty when present. Every card has a tag, headline and line."""
    sb_path = Path("lab/video/storyboard.json")
    with open(sb_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    for i, seg in enumerate(storyboard):
        if seg.get("type") in ("play", "montage", "split"):
            assert seg.get("caption"), f"Segment {i} ({seg['type']}) has no caption"
            assert seg.get("subcaption", "x") != "", f"Segment {i} subcaption is empty"
        if seg.get("type") == "card":
            assert seg.get("tag") and seg.get("headline") and seg.get("line"), f"Card {i} is missing text"


def test_no_do_not_say_phrases():
    """Verify no storyboard strings or HUD texts violate COPY_REVIEW.md §6 do-not-say list."""
    do_not_say_patterns = [
        "learned to play pokémon",
        "learned to play pokemon",
        "plays pokémon by itself",
        "plays pokemon by itself",
        "ai beats the gym on its own",
        "beats the gym on its own",
        "autonomous",
        "end-to-end",
        "learned to walk kanto",
        "never needs the backup",
        "100% success",
        "the world model is accurate",
        "predicts the game",
    ]

    sb_path = Path("lab/video/storyboard.json")
    with open(sb_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    strings_to_check = []
    for s in storyboard:
        if s.get("caption"):
            strings_to_check.append(s["caption"])
        if s.get("subcaption"):
            strings_to_check.append(s["subcaption"])
        for t in s.get("text", []):
            strings_to_check.append(t)
        for k in ("tag", "headline", "line", "line2", "heading", "footer", "title", "title_right", "left_label", "right_label"):
            if s.get(k):
                strings_to_check.append(s[k])
        for panel in s.get("panels", []):
            strings_to_check.extend(panel)
        strings_to_check.extend(s.get("strip", []))
        for elem in s.get("elements", []):
            for k in ("title", "label", "text", "note", "total"):
                if elem.get(k):
                    strings_to_check.append(str(elem[k]))
            for line in elem.get("lines", []):
                if isinstance(line, dict):
                    strings_to_check.append(line.get("text", ""))
                else:
                    strings_to_check.append(str(line))
            for note in elem.get("notes", []):
                if isinstance(note, dict):
                    strings_to_check.append(note.get("text", ""))
                else:
                    strings_to_check.append(str(note))
            for r in elem.get("rows", []):
                if len(r) > 0 and isinstance(r[0], str):
                    strings_to_check.append(r[0])
                if len(r) > 1 and isinstance(r[1], str):
                    strings_to_check.append(r[1])
            for tl in elem.get("tiles", []):
                if isinstance(tl, dict) and "caption" in tl:
                    strings_to_check.append(tl["caption"])
                elif isinstance(tl, (list, tuple)) and len(tl) > 1:
                    strings_to_check.append(tl[1])
        for col in s.get("columns", []):
            if col.get("header"):
                strings_to_check.append(col["header"])
            for item in col.get("items", []):
                for ik in ("lead", "line", "metric", "source"):
                    if item.get(ik):
                        strings_to_check.append(str(item[ik]))

    for g_val in GOAL_MAP.values():
        strings_to_check.append(g_val)
    for l_val in LEG_MAP.values():
        strings_to_check.append(l_val)

    for text in strings_to_check:
        t_low = text.lower()
        for phrase in do_not_say_patterns:
            assert phrase not in t_low, f"Found forbidden phrase {phrase!r} in text: {text!r}"


def test_diagram_element_kinds_render_without_error():
    """Verify that every diagram element kind renders without error at t = 0, mid, and end."""
    fonts = FontManager()
    toy_seg = {
        "type": "diagram",
        "seconds": 8.0,
        "title": "Architecture Test",
        "title_right": "Test · 1 / 1",
        "left_label": "TEST LEFT",
        "right_label": "TEST RIGHT",
        "elements": [
            {"kind": "box", "rect": [64, 236, 364, 420], "title": "Box", "lines": ["Line 1"], "notes": ["Note 1"], "t0": 0.3},
            {"kind": "arrow", "points": [[100, 100], [200, 200]], "style": "flow", "t0": 0.5, "label": "Flow", "label_pos": [150, 150]},
            {"kind": "arrow", "points": [[200, 200], [300, 200]], "style": "loop", "t0": 0.6},
            {"kind": "arrow", "points": [[300, 200], [400, 200]], "style": "dashed", "t0": 0.7},
            {"kind": "chip", "rect": [400, 400, 500, 450], "text": "Chip", "t0": 0.4},
            {"kind": "label", "pos": [100, 500], "text": "Label", "size": 16, "t0": 0.2},
            {"kind": "thumb", "rect": [84, 290, 148, 348], "step": 100, "t0": 0.3},
            {"kind": "bars", "rect": [64, 600, 500, 750], "title": "Bars", "rows": [["A", 10.0, "cyan"], ["B", 20.0, "gold"]], "note": "Bar note", "total": "total 30.0", "t0": 0.8},
            {"kind": "tiles", "t0": 0.9, "tiles": [{"rect": [64, 800, 200, 920], "value": "30/30", "colour": "gold", "caption": "Cap\nSubcap"}]},
            {"kind": "table", "rect": [300, 800, 700, 950], "rows": [["cyan", "Row 1"], ["gold", "Row 2"]], "t0": 1.0},
            {"kind": "strip", "lines": ["Strip 1", "Strip 2"], "t0": 1.2},
        ],
    }

    for t in (0.0, 4.0, 8.0):
        frame = render_diagram_frame(toy_seg, t, fonts)
        assert frame.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
        assert frame.dtype == np.uint8


def test_harness_card_reason_truncation():
    """Verify reason truncation occurs at a word boundary and ends with '…', never mid-word."""
    # 1. Short text <= max_chars is preserved verbatim without ellipsis
    short_text = "Level up Charmander on Route 1."
    assert truncate_reason(short_text, max_chars=180) == short_text
    assert truncate_reason(short_text, max_chars=len(short_text)) == short_text

    # 2. Text longer than max_chars is cut at a word boundary
    long_text = "The starter has reached the target level of 13, which is sufficient to challenge Brock. I will now travel north through Route 2 and Viridian Forest to reach Pewter City."
    # len is 170. If max_chars=80:
    # "The starter has reached the target level of 13, which is sufficient to challenge Brock."
    trunc_80 = truncate_reason(long_text, max_chars=80)
    assert trunc_80.endswith("…")
    # Content before ellipsis
    prefix = trunc_80[:-1]
    # The last word in prefix must be a complete word in long_text (e.g. followed by a space in long_text)
    assert prefix in long_text
    word_end_idx = long_text.find(prefix) + len(prefix)
    assert word_end_idx == len(long_text) or long_text[word_end_idx] in " .,;:!?", f"Truncated mid-word at index {word_end_idx}: {long_text[word_end_idx-5:word_end_idx+5]}"

    # 3. Test on all reasons in run_seed_100.json
    llm_json_path = Path("work/video/quest_s100_llm/run_seed_100.json")
    if llm_json_path.exists():
        with open(llm_json_path, "r") as f:
            data = json.load(f)
        for entry in data.get("timeline", []):
            reason = entry["reason"]
            for limit in [50, 80, 120, 150, 180]:
                tr = truncate_reason(reason, max_chars=limit)
                if len(reason) > limit:
                    assert tr.endswith("…")
                    assert not tr[:-1].endswith(" ")
                    assert len(tr) <= limit + 1
                    # Verify no mid-word cut: tr[:-1] must be a prefix of reason at a word boundary
                    prefix = tr[:-1]
                    assert reason.startswith(prefix)
                    if len(prefix) < len(reason):
                        after_char = reason[len(prefix)]
                        assert after_char in " .,;:!?'\"-\n", f"Mid-word cut on '{prefix[-10:]}' followed by '{after_char}' in '{reason}'"


def test_decision_lookup_by_presses():
    """Verify get_latest_decision returns latest timeline entry with presses <= step."""
    toy_timeline = [
        {"presses": 72, "goal": "grind", "latency": 10.8},
        {"presses": 1020, "goal": "heal", "latency": 13.7},
        {"presses": 1405, "goal": "grind", "latency": 12.2},
        {"presses": 4697, "goal": "travel_to_pewter", "latency": 21.5},
        {"presses": 6130, "goal": "gym", "latency": 10.9},
    ]

    # Before first decision
    assert get_latest_decision(toy_timeline, 0) is None
    assert get_latest_decision(toy_timeline, 71) is None

    # Exactly at decisions
    assert get_latest_decision(toy_timeline, 72)["goal"] == "grind"
    assert get_latest_decision(toy_timeline, 1020)["goal"] == "heal"
    assert get_latest_decision(toy_timeline, 4697)["goal"] == "travel_to_pewter"
    assert get_latest_decision(toy_timeline, 6130)["goal"] == "gym"

    # Between decisions
    assert get_latest_decision(toy_timeline, 100)["goal"] == "grind"
    assert get_latest_decision(toy_timeline, 1019)["goal"] == "grind"
    assert get_latest_decision(toy_timeline, 1021)["goal"] == "heal"
    assert get_latest_decision(toy_timeline, 3000)["goal"] == "grind"
    assert get_latest_decision(toy_timeline, 4700)["goal"] == "travel_to_pewter"

    # After last decision
    assert get_latest_decision(toy_timeline, 7044)["goal"] == "gym"
    assert get_latest_decision(toy_timeline, 99999)["goal"] == "gym"


def test_bakeoff_rows_load_from_storyboard():
    """Verify bakeoff segment in storyboard.json loads 7 rows with correct structure."""
    sb_path = Path("lab/video/storyboard.json")
    with open(sb_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    bakeoff_segs = [s for s in storyboard if s.get("type") == "bakeoff"]
    assert len(bakeoff_segs) == 1, "Expected exactly 1 bakeoff segment"

    bakeoff = bakeoff_segs[0]
    assert bakeoff.get("title") == "Six local LLMs as the System 2 planner"
    assert "RTX 4090" in bakeoff.get("footer", "")

    rows = bakeoff.get("rows", [])
    assert len(rows) == 7, f"Expected 7 table rows, got {len(rows)}"

    # Rule planner must be first, labelled 'rule-based (reference)'
    assert rows[0]["planner"] == "default planner (reference)"
    assert rows[0]["badges"] == "10/10"

    # gemma4:31b must be present and highlighted
    gemma_rows = [r for r in rows if "gemma4:31b" in r.get("planner", "")]
    assert len(gemma_rows) == 1
    assert gemma_rows[0].get("highlight") is True

    # Every row must have required columns
    for r in rows:
        assert "planner" in r and len(r["planner"]) > 0
        assert "badges" in r and len(r["badges"]) > 0
        assert "presses" in r and len(r["presses"]) > 0
        assert "vram" in r and len(r["vram"]) > 0


def test_render_bakeoff_frame_dimensions():
    """Verify render_bakeoff_frame generates (1080, 1920, 3) uint8 frame."""
    fonts = FontManager()
    toy_rows = [
        {"planner": "rules (reference)", "badges": "10/10", "presses": "8,508.5", "vram": "—"},
        {"planner": "gemma4:31b", "badges": "10/10", "presses": "7,239", "vram": "22.4 GB", "highlight": True},
    ]
    frame = render_bakeoff_frame(
        rows=toy_rows,
        title="Six local LLMs tested as the planner",
        footer="Test footer",
        fonts=fonts,
    )
    assert frame.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert frame.dtype == np.uint8


def test_load_storyboard_keeps_utf8_text_on_a_cp1252_locale(tmp_path):
    # Windows' default codepage turned the caption's "→" into "âť’" on screen (draft_v7)
    sb = [{"type": "montage", "caption": "Levelling up: Charmander 8 → 14", "subcaption": "a · b · Pokémon"}]
    path = tmp_path / "storyboard.json"
    path.write_text(json.dumps(sb, ensure_ascii=False), encoding="utf-8")
    loaded = load_storyboard(path)
    assert loaded[0]["caption"] == "Levelling up: Charmander 8 → 14"
    assert loaded[0]["subcaption"] == "a · b · Pokémon"


def test_battle_flash_mask_covers_the_battle_start_transition_and_flat_frames():
    """A battle start (recoloured overworld, dark frame, white frames) is masked up to its last flat frame."""
    from lab.video.compose import battle_flash_mask

    rng = np.random.default_rng(0)
    colors = rng.integers(60, 200, size=(20, 144, 160, 3), dtype=np.uint8)  # textured frames
    colors[8] = 40    # dark transition frame (flat)
    colors[9] = 241   # white flash frames (flat)
    colors[10] = 241
    colors[15] = 0    # an unrelated flat frame elsewhere
    in_battles = np.zeros(20, dtype=bool)
    in_battles[5:14] = True  # onset at step 5
    mask = battle_flash_mask(colors, in_battles)
    assert mask[5:11].all()            # onset .. last white frame
    assert not mask[11:14].any()       # the battle screen itself stays
    assert not mask[:5].any() and mask[15]


def test_deflash_steps_holds_the_last_clean_step_and_keeps_length():
    from lab.video.compose import deflash_steps

    flash = np.zeros(12, dtype=bool)
    flash[5:8] = True
    out = deflash_steps(np.arange(12), flash)
    assert len(out) == 12
    assert out[5:8].tolist() == [4, 4, 4] and out[8] == 8 and out[4] == 4


def test_clip_segment_loading_and_hold(tmp_path):
    """Verify loading a clip segment from MP4: frame count, hold frames, first/last frame match, held frames equal last."""
    import cv2
    clip_path = tmp_path / "tiny_clip.mp4"
    w, h = 64, 36
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, float(fps), (w, h))

    # 12 frames with distinct solid colors (BGR)
    src_frames_bgr = []
    for i in range(12):
        f = np.full((h, w, 3), (i * 20, (i * 20 + 50) % 256, (255 - i * 20) % 256), dtype=np.uint8)
        src_frames_bgr.append(f)
        writer.write(f)
    writer.release()

    # Read back through load_clip_frames: frames [2, 8) with hold 0.2 s (6 frames @ 30 fps)
    start_f = 2
    end_f = 8
    hold_s = 0.2
    expected_hold_frames = int(round(hold_s * fps))  # 6
    expected_total_frames = (end_f - start_f) + expected_hold_frames  # 6 + 6 = 12

    loaded_frames = load_clip_frames(clip_path, start_frame=start_f, end_frame=end_f, hold_s=hold_s, fps=fps)
    assert len(loaded_frames) == expected_total_frames
    assert all(f.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3) for f in loaded_frames)

    # First frame matches source frame 2 (resized to 1920x1080)
    first_expected = cv2.resize(cv2.cvtColor(src_frames_bgr[start_f], cv2.COLOR_BGR2RGB), (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)
    assert np.max(np.abs(loaded_frames[0].astype(int) - first_expected.astype(int))) <= 6

    # Last read frame (index 5) matches source frame 7 (resized)
    last_expected = cv2.resize(cv2.cvtColor(src_frames_bgr[end_f - 1], cv2.COLOR_BGR2RGB), (CANVAS_WIDTH, CANVAS_HEIGHT), interpolation=cv2.INTER_LINEAR)
    assert np.max(np.abs(loaded_frames[5].astype(int) - last_expected.astype(int))) <= 6

    # Held frames equal the last read frame exactly
    last_read_frame = loaded_frames[5]
    for h_idx in range(6, expected_total_frames):
        assert np.array_equal(loaded_frames[h_idx], last_read_frame)

    # Also test calculate_segment_frames on this clip
    sb_toy = [{"type": "clip", "src": str(clip_path), "start_frame": start_f, "end_frame": end_f, "hold_s": hold_s}]
    assert calculate_segment_frames(sb_toy, fps=fps) == [expected_total_frames]


def test_value_screen_rendering():
    """Verify value screen renders at t=0, 1.5, 8.0 with placeholder copy and 3-item columns.
    Checks dimensions (1080, 1920, 3), progression between t=0 and t=8, and very long line handling."""
    import copy
    fonts = FontManager()

    # Load placeholder copy from storyboard.json
    with open(Path("lab/video/storyboard.json"), "r", encoding="utf-8") as f:
        sb = json.load(f)
    val_segs = [s for s in sb if s.get("type") == "value"]
    assert len(val_segs) >= 1
    seg = val_segs[0]

    # Test at t=0, t=1.5, t=8 with 4-item placeholder copy
    f0 = render_value_frame(seg, 0.0, fonts)
    f1_5 = render_value_frame(seg, 1.5, fonts)
    f8 = render_value_frame(seg, 8.0, fonts)

    for frame in (f0, f1_5, f8):
        assert frame.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
        assert frame.dtype == np.uint8

    # Frames differ between t=0 and t=8 (fade-in / stagger)
    assert not np.array_equal(f0, f8)
    assert not np.array_equal(f0, f1_5)
    assert not np.array_equal(f1_5, f8)

    # Test with 3-item columns
    seg3 = copy.deepcopy(seg)
    for col in seg3["columns"]:
        col["items"] = col["items"][:3]

    f3_0 = render_value_frame(seg3, 0.0, fonts)
    f3_1_5 = render_value_frame(seg3, 1.5, fonts)
    f3_8 = render_value_frame(seg3, 8.0, fonts)

    for frame in (f3_0, f3_1_5, f3_8):
        assert frame.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
        assert frame.dtype == np.uint8

    assert not np.array_equal(f3_0, f3_8)

    # Test with a very long line: must wrap/shrink and not crash
    seg_long = copy.deepcopy(seg)
    seg_long["columns"][0]["items"][0]["line"] = "A" + " very long repetitive text line for overflow testing" * 30
    seg_long["columns"][1]["items"][0]["lead"] = "A very long headline title lead that should shrink properly"
    seg_long["columns"][0]["items"][0]["metric"] = "999,999,999/1,000,000,000"
    f_long = render_value_frame(seg_long, 5.0, fonts)
    assert f_long.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert f_long.dtype == np.uint8
