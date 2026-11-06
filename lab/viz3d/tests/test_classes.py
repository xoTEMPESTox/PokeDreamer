"""Tests for lab.viz3d.classes (CPU only, no Blender, no GPU)."""
import json
from pathlib import Path
import numpy as np
import pytest

from lab.viz3d.classes import (
    load_palette,
    frame_classes,
    sprite_regions,
    stitch,
    agreement,
)


def test_load_palette():
    names, lut = load_palette()
    assert len(names) == 23
    assert "grass" in names
    assert "building_wall" in names
    assert "sprite_npc" in names
    # Collision color (245, 245, 240) defaults to building_wall
    wall_idx = names.index("building_wall")
    assert lut[(245, 245, 240)] == (wall_idx, 0)


def test_collision_rule():
    names, _ = load_palette()
    wall_idx = names.index("building_wall")
    npc_idx = names.index("sprite_npc")

    # Load palette to get exact colours
    with open("lab/color/palette.json") as f:
        pal = json.load(f)["classes"]

    c_collision = np.array([245, 245, 240], dtype=np.uint8)
    c_npc_shade1 = np.array(pal["sprite_npc"][1], dtype=np.uint8)
    c_grass_shade0 = np.array(pal["grass"][0], dtype=np.uint8)

    img = np.zeros((144, 160, 3), dtype=np.uint8)
    img[:, :] = c_grass_shade0

    # Put a collision pixel at (50, 50) with an NPC shade 1 pixel at (50, 51) (dist 1)
    img[50, 50] = c_collision
    img[50, 51] = c_npc_shade1

    # Put another collision pixel at (100, 100), far from any NPC pixels
    img[100, 100] = c_collision

    cls, shade = frame_classes(img)

    # (50, 50) is near NPC shade 1 -> sprite_npc shade 0
    assert cls[50, 50] == npc_idx
    assert shade[50, 50] == 0

    # (100, 100) is isolated -> building_wall shade 0
    assert cls[100, 100] == wall_idx
    assert shade[100, 100] == 0


def test_sprite_regions():
    cls = np.zeros((144, 160), dtype=np.int8)
    names, _ = load_palette()
    npc_idx = names.index("sprite_npc")

    # Component 1: 5x5 = 25 pixels (>= 20 px)
    cls[10:15, 10:15] = npc_idx

    # Component 2: 3x3 = 9 pixels (< 20 px)
    cls[30:33, 30:33] = npc_idx

    regions = sprite_regions(cls, which="sprite_npc", min_pixels=20)
    assert len(regions) == 1
    r0, r1, c0, c1, mask = regions[0]
    assert (r0, r1, c0, c1) == (10, 15, 10, 15)
    assert mask.shape == (5, 5)
    assert np.all(mask)


def test_stitch_toy():
    names, _ = load_palette()
    grass_idx = names.index("grass")
    path_idx = names.index("path")

    with open("lab/color/palette.json") as f:
        pal = json.load(f)["classes"]

    c_grass0 = pal["grass"][0]
    c_path1 = pal["path"][1]

    # Two synthetic frames: frame 0 at (x=4, y=4), frame 1 at (x=5, y=4) (moved 16 px right)
    f0 = np.zeros((144, 160, 3), dtype=np.uint8)
    f0[:, :] = c_grass0

    f1 = np.zeros((144, 160, 3), dtype=np.uint8)
    f1[:, :] = c_grass0
    # Add a distinctive patch in f1
    f1[20:40, 20:40] = c_path1

    frames = np.stack([f0, f1], axis=0)
    xs = np.array([4, 5])
    ys = np.array([4, 4])

    res = stitch(frames, xs, ys)
    cls, shade, origin, count = res["cls"], res["shade"], res["origin"], res["count"]

    assert origin == (0, 0)
    # H should be 144, W should be 160 + 16 = 176
    assert cls.shape == (144, 176)
    # Unseen pixels on the left or outside must be -1
    # Check that majority class is grass for most pixels
    assert np.all(cls[50:100, 50:100] == grass_idx)
    assert np.all(count > 0)


def test_agreement_synthetic():
    rng = np.random.default_rng(42)
    with open("lab/color/palette.json") as f:
        pal = json.load(f)["classes"]

    # Pick non-sprite colors: grass and path
    valid_colors = pal["grass"] + pal["path"]
    big_canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    for r in range(200):
        for c in range(200):
            big_canvas[r, c] = valid_colors[(r + c) % len(valid_colors)]

    # Slice frame 0 at (r=16, c=16) -> corresponds to (x=5, y=5)
    f0 = big_canvas[16 : 16 + 144, 16 : 16 + 160]
    # Slice frame 1 at (r=32, c=16) -> moved south by 16 px -> (x=5, y=6)
    f1 = big_canvas[32 : 32 + 144, 16 : 16 + 160]

    frames = np.stack([f0, f1], axis=0)
    xs = np.array([5, 5])
    ys = np.array([5, 6])

    agr = agreement(frames, xs, ys)
    assert pytest.approx(agr, 1e-6) == 1.0


def test_agreement_real_steps():
    c_data = np.load("work/video/traj/run_seed_100_color.npz")
    r_data = np.load("work/video/traj/run_seed_100.npz")

    frames = c_data["color"][6833:6866]
    xs = r_data["xs"][6833:6866]
    ys = r_data["ys"][6833:6866]

    agr = agreement(frames, xs, ys)
    assert agr >= 0.95
