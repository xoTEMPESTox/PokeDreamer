"""Tests for lab.viz3d.swipe2d."""
import numpy as np
import pytest

from lab.viz3d.swipe2d import composite_swipe_frame


def test_swipe_compositor_toy_edges():
    """Verify swipe compositor picks the right panel on each side of each edge."""
    # Create distinct toy panels (H=100, W=100)
    raw_toy = np.full((100, 100, 3), 10, dtype=np.uint8)
    crisp_toy = np.full((100, 100, 3), 100, dtype=np.uint8)
    color_toy = np.full((100, 100, 3), 200, dtype=np.uint8)

    # Origin at X=50, W=100 (so range is [50, 150])
    # Edge A at 110, Edge B at 80
    frame = composite_swipe_frame(
        raw_toy, crisp_toy, color_toy,
        edge_a_x=110, edge_b_x=80,
        frame_w=200, frame_h=200,
        origin_x=50, origin_y=50,
        disp_w=100, disp_h=100,
    )

    # Sample screen region
    # Canvas X in [50, 150], Y in [50, 150]
    # At local X=15 (canvas X=65 < edge_b 80): should be color (200)
    assert frame[100, 65, 0] == 200

    # At local X=45 (canvas X=95, between edge_b 80 and edge_a 110): should be crisp (100)
    assert frame[100, 95, 0] == 100

    # At local X=80 (canvas X=130 > edge_a 110): should be raw (10)
    assert frame[100, 130, 0] == 10


def test_swipe_panels_output_sizes():
    """Verify full canvas dimensions and strip shapes."""
    raw_toy = np.zeros((576, 640, 3), dtype=np.uint8)
    crisp_toy = np.zeros((576, 640, 3), dtype=np.uint8)
    color_toy = np.zeros((576, 640, 3), dtype=np.uint8)

    frame = composite_swipe_frame(
        raw_toy, crisp_toy, color_toy,
        edge_a_x=900, edge_b_x=750,
    )

    assert frame.shape == (1080, 1920, 3)
