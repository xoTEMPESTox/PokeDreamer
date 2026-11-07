"""Tests for lab.viz3d.world (CPU only)."""
from pathlib import Path
import numpy as np
import pytest

from lab.viz3d.classes import load_palette
from lab.viz3d.bpy_build import CLASS_NAMES
from lab.viz3d.world import (
    FORBIDDEN_BLADE_CLASSES,
    compute_scatter_mask,
    create_ground_images,
    extract_world_elements,
)


def test_palette_classes_convention():
    """Pin the convention: first 23 palette names must match bpy_build.CLASS_NAMES in exact order."""
    names, _ = load_palette()
    assert names[:23] == CLASS_NAMES


def test_toy_map_join_offset():
    """Test Route 1 offset join logic on toy arrays."""
    # Toy Viridian: 4x4 tiles (32x32 px)
    # Toy Route 1: 4x2 tiles (32x16 px)
    # Joined world: 64x32 px, with Route 1 at row 32, col 8
    H1_px, W1_px = 32, 32
    H12_px, W12_px = 32, 16
    r1_row = 32
    r1_col = 8

    toy_map1 = np.full((H1_px, W1_px), 10, dtype=np.int8)
    toy_map12 = np.full((H12_px, W12_px), 20, dtype=np.int8)

    total_H = H1_px + H12_px
    total_W = W1_px
    combined = np.zeros((total_H, total_W), dtype=np.int8)

    combined[:H1_px, :W1_px] = toy_map1
    combined[r1_row : r1_row + H12_px, r1_col : r1_col + W12_px] = toy_map12

    # Check boundaries
    assert combined[0, 0] == 10
    assert combined[31, 31] == 10
    assert combined[r1_row, r1_col] == 20
    assert combined[r1_row + H12_px - 1, r1_col + W12_px - 1] == 20
    # Left strip is not Route 1
    assert combined[r1_row, 0] == 0
    # Right strip is not Route 1
    assert combined[r1_row, r1_col + W12_px] == 0


def test_mask_channels_exclusive_of_forbidden_classes():
    """Verify that blade mask channels (R, G, B) are strictly 0 on forbidden classes."""
    names, _ = load_palette()
    name_to_idx = {n: i for i, n in enumerate(names[:23])}

    # Create synthetic map containing all 23 classes
    all_classes = np.arange(23, dtype=np.int8).reshape(1, 23)
    mask = compute_scatter_mask(all_classes)  # (1, 23, 4)

    for forbidden_name in FORBIDDEN_BLADE_CLASSES:
        if forbidden_name in name_to_idx:
            idx = name_to_idx[forbidden_name]
            # R, G, B must all be 0.0 on forbidden classes
            assert mask[0, idx, 0] == 0.0, f"R channel was >0 for {forbidden_name}"
            assert mask[0, idx, 1] == 0.0, f"G channel was >0 for {forbidden_name}"
            assert mask[0, idx, 2] == 0.0, f"B channel was >0 for {forbidden_name}"
            # Alpha must be 0.0 on forbidden classes
            assert mask[0, idx, 3] == 0.0, f"Alpha was >0 for {forbidden_name}"

    # Check permitted blade classes
    grass_idx = name_to_idx["grass"]
    assert mask[0, grass_idx, 0] == 1.0
    assert mask[0, grass_idx, 3] == 1.0

    tg_idx = name_to_idx["tall_grass"]
    assert mask[0, tg_idx, 1] == 1.0
    assert mask[0, tg_idx, 3] == 1.0

    fl_idx = name_to_idx["flowers"]
    assert mask[0, fl_idx, 2] == 1.0
    assert mask[0, fl_idx, 3] == 1.0


def test_raw_is_exact_2x2_mean_of_crisp():
    """Verify that ground_raw is exactly the 2x2 mean of ground_crisp."""
    H, W = 16, 16
    np.random.seed(123)
    toy_shade = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)
    toy_cls = np.zeros((H, W), dtype=np.int8)

    crisp, raw, color = create_ground_images(toy_cls, toy_shade)

    assert raw.shape == (H // 2, W // 2)
    assert crisp.shape == (H, W)

    expected_mean = (
        crisp[0::2, 0::2].astype(np.float32)
        + crisp[0::2, 1::2].astype(np.float32)
        + crisp[1::2, 0::2].astype(np.float32)
        + crisp[1::2, 1::2].astype(np.float32)
    ) / 4.0

    np.testing.assert_allclose(raw.astype(np.float32), np.round(expected_mean), atol=1e-5)
