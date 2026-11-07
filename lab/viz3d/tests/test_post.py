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
    """Verify 2D swipe insert replaces exactly frames 351..440 (shot 2d); 255 is the 3D sweep now."""
    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920)
    captions = {}

    dummy_3d = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    dummy_2d = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    # Frame 350: should be 3D
    out_254 = compose_single_frame(
        350, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_254[540, 960].mean()) - 50.0) < 5.0

    # Frame 351: should be 2D insert
    out_255 = compose_single_frame(
        351, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_255[540, 960].mean()) - 200.0) < 5.0

    # Frame 440: should be 2D insert
    out_290 = compose_single_frame(
        440, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_290[540, 960].mean()) - 200.0) < 5.0

    # Frame 255 (mid 3D sweep) and 441: should be 3D
    out_old = compose_single_frame(
        255, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_old[540, 960].mean()) - 50.0) < 5.0
    out_291 = compose_single_frame(
        441, dummy_3d, fonts, captions, vignette, chroma_maps, insert_2d_frame=dummy_2d
    )
    assert abs(float(out_291[540, 960].mean()) - 50.0) < 5.0


def test_battle_frame_wipe():
    """Verify battle frame wipes in during frames 710..721 only (584 is mid-walk now)."""
    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920)
    captions = {}

    dummy_3d = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    dummy_battle = np.full((1080, 1920, 3), 220, dtype=np.uint8)

    # Frame 709: no wipe yet
    out_pre = compose_single_frame(
        709, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert float(out_pre[540, 100].mean()) < 100.0  # 3D, not the battle frame (220)

    # Frame 584 (mid-walk): no wipe
    out_mid = compose_single_frame(
        584, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert float(out_mid[540, 100].mean()) < 100.0  # 3D, not the battle frame (220)

    # Frame 716 (half-way wipe): left side is battle frame (220), right side is 3D (31 with vignette)
    out_half = compose_single_frame(
        716, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert abs(float(out_half[540, 100].mean()) - 220.0) < 5.0
    assert 25.0 < float(out_half[540, 1800].mean()) < 40.0

    # Frame 722 (final frame): full battle frame
    out_end = compose_single_frame(
        722, dummy_3d, fonts, captions, vignette, chroma_maps, battle_frame=dummy_battle
    )
    assert abs(float(out_end[540, 100].mean()) - 220.0) < 5.0
    assert abs(float(out_end[540, 1800].mean()) - 220.0) < 5.0


def test_caption_anchor_moves_text():
    """Verify changing caption anchor moves rendered text on a toy frame."""
    from PIL import Image
    from lab.viz3d.post import draw_caption

    fonts = FontManager()
    cap_top = {
        "headline": "Test Anchor Headline",
        "sub": "Subtext line",
        "anchor": "top_center",
        "dx": 0,
        "dy": 0,
        "size": 24,
    }
    cap_bot = {
        "headline": "Test Anchor Headline",
        "sub": "Subtext line",
        "anchor": "bottom_center",
        "dx": 0,
        "dy": 0,
        "size": 24,
    }

    img_top = Image.new("RGBA", (1920, 1080), (0, 0, 0, 255))
    draw_caption(img_top, cap_top, fonts)
    arr_top = np.array(img_top)
    y_coords_top = np.where(arr_top[..., 0] > 100)[0]

    img_bot = Image.new("RGBA", (1920, 1080), (0, 0, 0, 255))
    draw_caption(img_bot, cap_bot, fonts)
    arr_bot = np.array(img_bot)
    y_coords_bot = np.where(arr_bot[..., 0] > 100)[0]

    assert len(y_coords_top) > 0 and len(y_coords_bot) > 0
    # Top center text must be in top quarter, bottom center in bottom quarter
    assert np.mean(y_coords_top) < 300.0
    assert np.mean(y_coords_bot) > 700.0
