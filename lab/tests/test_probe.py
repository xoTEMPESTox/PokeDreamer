"""Tests for FrameProbe and predict function."""
import numpy as np
import pytest

from lab.probe import FrameProbe, predict


def test_probe_predict_shapes_and_dtypes():
    probe = FrameProbe()

    # Test uint8 input of shape (3, 72, 80)
    frames_u8 = np.random.randint(0, 256, size=(3, 72, 80), dtype=np.uint8)
    preds_u8 = predict(probe, frames_u8, device="cpu", batch=2)

    assert isinstance(preds_u8, dict)
    assert "map_id" in preds_u8 and "x" in preds_u8 and "y" in preds_u8 and "battle" in preds_u8

    assert preds_u8["map_id"].shape == (3,)
    assert np.issubdtype(preds_u8["map_id"].dtype, np.integer)

    assert preds_u8["x"].shape == (3,)
    assert np.issubdtype(preds_u8["x"].dtype, np.integer)

    assert preds_u8["y"].shape == (3,)
    assert np.issubdtype(preds_u8["y"].dtype, np.integer)

    assert preds_u8["battle"].shape == (3,)
    assert preds_u8["battle"].dtype == bool

    # Test float32 input of shape (3, 72, 80)
    frames_f32 = np.random.rand(3, 72, 80).astype(np.float32)
    preds_f32 = predict(probe, frames_f32, device="cpu", batch=2)

    assert isinstance(preds_f32, dict)
    assert "map_id" in preds_f32 and "x" in preds_f32 and "y" in preds_f32 and "battle" in preds_f32

    assert preds_f32["map_id"].shape == (3,)
    assert np.issubdtype(preds_f32["map_id"].dtype, np.integer)

    assert preds_f32["x"].shape == (3,)
    assert np.issubdtype(preds_f32["x"].dtype, np.integer)

    assert preds_f32["y"].shape == (3,)
    assert np.issubdtype(preds_f32["y"].dtype, np.integer)

    assert preds_f32["battle"].shape == (3,)
    assert preds_f32["battle"].dtype == bool
