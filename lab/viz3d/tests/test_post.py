"""Tests for lab.viz3d.post (CPU only)."""

from pathlib import Path
import json
import numpy as np
import pytest

from lab.viz3d.post import (
    create_vignette_mask,
    create_chromatic_maps,
    apply_chromatic_aberration,
    apply_grain,
    create_battle_play_frame,
    compose_single_frame,
)
from lab.video.compose import FontManager


def test_vignette_mask_properties():
    """Verify vignette mask is 1.0 at center and darkens toward edges."""
    v = create_vignette_mask(1080, 1920)
    assert v.shape == (1080, 1920, 1)
    # Center (540, 960) should be 1.0
    assert abs(v[540, 960, 0] - 1.0) < 1e-4
    # Corner should be ~0.528 (1.0 - 0.22 * sqrt(2)^2.2)
    assert v[0, 0, 0] < 0.65
    assert v[0, 0, 0] > 0.45


def test_chromatic_aberration():
    """Verify chromatic aberration applies radial shift to R and B channels."""
    maps = create_chromatic_maps(1080, 1920, max_shift=1.5)
    # Toy image with single-pixel high-contrast dot
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[100, 100] = [255, 255, 255]
    ab = apply_chromatic_aberration(img, *maps)
    assert ab.shape == (1080, 1920, 3)
    # R and B should have shifted slightly relative to G
    assert np.any(ab[99:102, 99:102] > 0)


def test_grain_deterministic():
    """Verify grain is deterministic for a given frame index and varies across frames."""
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    g0_a = apply_grain(base, frame_idx=10, sigma=2.0)
    g0_b = apply_grain(base, frame_idx=10, sigma=2.0)
    g1 = apply_grain(base, frame_idx=11, sigma=2.0)

    np.testing.assert_array_equal(g0_a, g0_b)
    assert not np.array_equal(g0_a, g1)
    assert abs(float(g0_a.mean()) - 128.0) < 1.0


def test_insert_replacement_range():
    """Verify 2D swipe insert replaces exactly frames 255..290."""
    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920)
    captions = {}

    dummy_3d = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    dummy_2d = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    # Frame 254: should be 3D
    out_254 = compose_single_frame(
        254, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_254[540, 960].mean()) - 50.0) < 5.0

    # Frame 255: should be 2D insert
    out_255 = compose_single_frame(
        255, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_255[540, 960].mean()) - 200.0) < 5.0

    # Frame 290: should be 2D insert
    out_290 = compose_single_frame(
        290, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_290[540, 960].mean()) - 200.0) < 5.0

    # Frame 291: should be 3D
    out_291 = compose_single_frame(
        291, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_291[540, 960].mean()) - 50.0) < 5.0


def test_battle_frame_wipe():
    """Verify battle frame wipes in during frames 578..590."""
    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920)
    captions = {}

    dummy_3d = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    dummy_battle = np.full((1080, 1920, 3), 220, dtype=np.uint8)

    # Frame 577: no wipe yet
    out_577 = compose_single_frame(
        577, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert abs(float(out_577[540, 100].mean()) - 40.0) < 5.0

    # Frame 584 (half-way wipe): left side is battle frame (220), right side is 3D (31 with vignette)
    out_584 = compose_single_frame(
        584, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert abs(float(out_584[540, 100].mean()) - 220.0) < 5.0
    assert 25.0 < float(out_584[540, 1800].mean()) < 40.0

    # Frame 590 (final frame): full battle frame
    out_590 = compose_single_frame(
        590, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert abs(float(out_590[540, 100].mean()) - 220.0) < 5.0
    assert abs(float(out_590[540, 1800].mean()) - 220.0) < 5.0
