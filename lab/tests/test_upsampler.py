"""Tests for the native-resolution upsampler and shade quantization."""
import cv2
import numpy as np
import pytest
import torch

from lab.upsampler import SHADES, Upsampler, save_screenshot, to_classes, to_shades, upsample


def test_untrained_upsampler_forward():
    model = Upsampler()
    x = torch.rand(2, 1, 72, 80)
    out = model(x)
    assert out.shape == (2, 4, 144, 160)
    assert out.dtype == torch.float32


def test_upsample_inference_uint8_and_float():
    model = Upsampler().eval()

    # uint8 input
    u8_frames = np.random.randint(0, 256, (3, 72, 80), dtype=np.uint8)
    out_u8 = upsample(model, u8_frames, device="cpu")
    assert out_u8.shape == (3, 144, 160)
    assert out_u8.dtype == np.uint8
    unique_u8 = set(np.unique(out_u8))
    assert unique_u8.issubset({0, 85, 153, 255})

    # float input in [0, 1]
    f_frames = np.random.rand(2, 72, 80).astype(np.float32)
    out_f = upsample(model, f_frames, device="cpu")
    assert out_f.shape == (2, 144, 160)
    assert out_f.dtype == np.uint8
    unique_f = set(np.unique(out_f))
    assert unique_f.issubset({0, 85, 153, 255})


def test_shade_quantization():
    # Exact shades
    exact_shades = np.array([0, 85, 153, 255], dtype=np.uint8)
    classes = to_classes(exact_shades)
    np.testing.assert_array_equal(classes, [0, 1, 2, 3])

    # In-between and boundary values
    # Midpoints: 0 vs 85 -> 42.5; 85 vs 153 -> 119; 153 vs 255 -> 204
    test_vals = np.array([0, 20, 42, 43, 85, 100, 119, 120, 153, 180, 204, 205, 230, 255], dtype=np.uint8)
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=np.uint8)
    np.testing.assert_array_equal(to_classes(test_vals), expected)

    # Inverse mapping
    np.testing.assert_array_equal(to_shades(classes), exact_shades)


def test_to_classes_tensor():
    t_vals = torch.tensor([0, 42, 43, 85, 119, 120, 153, 204, 205, 255], dtype=torch.uint8)
    expected = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=torch.uint8)
    assert torch.equal(to_classes(t_vals), expected)


def test_save_screenshot(tmp_path):
    raw = np.random.randint(0, 256, (72, 80), dtype=np.uint8)
    model = Upsampler().eval()

    # 1. With untrained Upsampler
    paths = save_screenshot(raw, model, out_dir=tmp_path, step=42, device="cpu")
    assert len(paths) == 3
    names = {p.name for p in paths}
    assert any(n.endswith("_native.png") for n in names)
    assert any(n.endswith("_4x.png") for n in names)
    assert any(n.endswith("_raw72x80.png") for n in names)

    for p in paths:
        assert p.exists()
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if p.name.endswith("_native.png"):
            assert img.shape == (144, 160)
            assert set(np.unique(img)).issubset({0, 85, 153, 255})
        elif p.name.endswith("_4x.png"):
            assert img.shape == (576, 640)
        elif p.name.endswith("_raw72x80.png"):
            assert img.shape == (72, 80)
            np.testing.assert_array_equal(img, raw)

    # 2. Without upsampler (None)
    paths_none = save_screenshot(raw, None, out_dir=tmp_path, step=99, device="cpu")
    assert len(paths_none) == 2
    names_none = {p.name for p in paths_none}
    assert any(n.endswith("_raw72x80.png") for n in names_none)
    assert any(n.endswith("_8x.png") for n in names_none)

    for p in paths_none:
        assert p.exists()
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if p.name.endswith("_raw72x80.png"):
            assert img.shape == (72, 80)
            np.testing.assert_array_equal(img, raw)
        elif p.name.endswith("_8x.png"):
            assert img.shape == (576, 640)


def test_play_initial_actions_align_with_context():
    # The model predicts frame t+1 from frames t-3..t and actions a_{t-3}..a_t (a_t pressed after frame t).
    # Before the first key press, the buffer must hold a_{s-1}..a_{s+2} so that appending the pressed
    # button gives a_s, a_{s+1}, a_{s+2}, button for context frames s..s+3.
    import numpy as np
    from lab.dwm.model import CTX
    from lab.play import initial_actions
    data = {"actions": np.arange(20, dtype=np.uint8)}
    s = 5
    buf = initial_actions(data, s)
    np.testing.assert_array_equal(buf, np.arange(s - 1, s + CTX - 1))
    after_press = np.concatenate([buf[1:], [7]])
    np.testing.assert_array_equal(after_press, [s, s + 1, s + 2, 7])
