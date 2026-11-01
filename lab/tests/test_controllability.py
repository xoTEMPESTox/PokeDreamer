"""Unit tests for action-controllability pure functions without checkpoints."""
import copy
import numpy as np
import pytest
import torch

from lab.dwm.controllability import (
    make_actions,
    orthogonal_displacement,
    real_reference,
    signed_displacement,
)
from lab.dwm.model import CTX


def test_make_actions_numpy_1d():
    # Context actions of length CTX-1 = 3
    ctx_actions = np.array([4, 5, 6], dtype=np.uint8)
    res = make_actions(ctx_actions, d=0, K=4)
    expected = np.array([4, 5, 6, 0, 0, 0, 0], dtype=np.uint8)
    assert isinstance(res, np.ndarray)
    assert res.shape == (7,)
    np.testing.assert_array_equal(res, expected)


def test_make_actions_numpy_2d():
    ctx_actions = np.array([[1, 2, 3], [7, 6, 5]], dtype=int)
    res = make_actions(ctx_actions, d=2, K=2)
    expected = np.array([[1, 2, 3, 2, 2], [7, 6, 5, 2, 2]], dtype=int)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 5)
    np.testing.assert_array_equal(res, expected)


def test_make_actions_torch_1d():
    ctx_actions = torch.tensor([1, 2, 3], dtype=torch.long)
    res = make_actions(ctx_actions, d=1, K=3)
    expected = torch.tensor([1, 2, 3, 1, 1, 1], dtype=torch.long)
    assert isinstance(res, torch.Tensor)
    assert torch.equal(res, expected)


def test_make_actions_torch_2d():
    ctx_actions = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    res = make_actions(ctx_actions, d=3, K=2)
    expected = torch.tensor([[1, 2, 3, 3, 3], [4, 5, 6, 3, 3]], dtype=torch.long)
    assert isinstance(res, torch.Tensor)
    assert torch.equal(res, expected)


def test_signed_displacement_conventions():
    # 0 = down: +dy (y increases downward)
    assert signed_displacement(dx=10, dy=2, d=0) == 2
    assert signed_displacement(dx=10, dy=-3, d=0) == -3

    # 1 = left: -dx (x decreases leftward)
    assert signed_displacement(dx=-4, dy=10, d=1) == 4
    assert signed_displacement(dx=4, dy=10, d=1) == -4

    # 2 = right: +dx (x increases rightward)
    assert signed_displacement(dx=5, dy=-10, d=2) == 5
    assert signed_displacement(dx=-5, dy=-10, d=2) == -5

    # 3 = up: -dy (y decreases upward)
    assert signed_displacement(dx=10, dy=-6, d=3) == 6
    assert signed_displacement(dx=10, dy=6, d=3) == -6

    # Invalid direction raises ValueError
    with pytest.raises(ValueError):
        signed_displacement(0, 0, d=4)


def test_signed_displacement_vectorized():
    dx = np.array([2, -3, 4, -5])
    dy = np.array([1, 2, -3, -4])

    np.testing.assert_array_equal(signed_displacement(dx, dy, d=0), [1, 2, -3, -4])
    np.testing.assert_array_equal(signed_displacement(dx, dy, d=1), [-2, 3, -4, 5])
    np.testing.assert_array_equal(signed_displacement(dx, dy, d=2), [2, -3, 4, -5])
    np.testing.assert_array_equal(signed_displacement(dx, dy, d=3), [-1, -2, 3, 4])


def test_orthogonal_displacement():
    # 0 (down) and 3 (up): orthogonal is |dx|
    assert orthogonal_displacement(dx=3, dy=5, d=0) == 3
    assert orthogonal_displacement(dx=-4, dy=5, d=3) == 4

    # 1 (left) and 2 (right): orthogonal is |dy|
    assert orthogonal_displacement(dx=3, dy=-7, d=1) == 7
    assert orthogonal_displacement(dx=-4, dy=6, d=2) == 6


def test_real_reference_constructed_walk():
    # Construct a synthetic episode of 10 steps
    n_frames = 10
    K = 3
    data = {
        "episode_id": np.zeros(n_frames, dtype=int),
        "in_battles": np.zeros(n_frames, dtype=bool),
        "map_ids": np.zeros(n_frames, dtype=int),
        "actions": np.full(n_frames, 7, dtype=int),
        "xs": np.full(n_frames, 10, dtype=int),
        "ys": np.full(n_frames, 20, dtype=int),
    }

    # Window starts at s=1. Window length = CTX + K = 4 + 3 = 7 (indices 1..7).
    # Reference frame is s + CTX - 1 = 1 + 3 = 4.
    # Future actions are at indices 4, 5, 6 (length 3).
    data["actions"][4:7] = 0  # 3 'down' actions in a row

    # Position at frame 4 is (10, 20)
    # Step 1 (frame 5): moves down 1 tile -> (10, 21), dy=1
    data["ys"][5] = 21
    # Step 2 (frame 6): moves down another tile -> (10, 22), dy=2
    data["ys"][6] = 22
    # Step 3 (frame 7): wall blocks movement -> (10, 22), dy=2
    data["ys"][7] = 22

    res = real_reference(data, eps=[0], d=0, K=K)
    assert res["count"] == 1
    assert res["starts"][0] == 1
    np.testing.assert_array_equal(res["displacements"][0], [1, 2, 2])
    np.testing.assert_allclose(res["mean"], [1.0, 2.0, 2.0])

    # Test direction 1 (left) on another window starting at s=2
    data_left = copy.deepcopy(data)
    data_left["actions"][5:8] = 1  # 3 'left' actions
    data_left["xs"][5] = 15        # start pos at frame 5
    data_left["xs"][6] = 14        # step 1: dx = -1 -> left disp = 1
    data_left["xs"][7] = 13        # step 2: dx = -2 -> left disp = 2
    data_left["xs"][8] = 13        # step 3: dx = -2 -> left disp = 2 (wall)

    res_l = real_reference(data_left, eps=[0], d=1, K=K)
    assert res_l["count"] == 1
    assert res_l["starts"][0] == 2
    np.testing.assert_array_equal(res_l["displacements"][0], [1, 2, 2])
    np.testing.assert_allclose(res_l["mean"], [1.0, 2.0, 2.0])

    # Test battle exclusion: window should be excluded if in_battles=True
    data_battle = copy.deepcopy(data)
    data_battle["in_battles"][5] = True
    res_b = real_reference(data_battle, eps=[0], d=0, K=K)
    assert res_b["count"] == 0
    np.testing.assert_allclose(res_b["mean"], [0.0, 0.0, 0.0])

    # Test map transition exclusion: window should be excluded if map changes
    data_map = copy.deepcopy(data)
    data_map["map_ids"][6] = 1
    res_m = real_reference(data_map, eps=[0], d=0, K=K)
    assert res_m["count"] == 0

    # Test episode mismatch: asking for episode 1 when only episode 0 exists
    res_ep = real_reference(data, eps=[1], d=0, K=K)
    assert res_ep["count"] == 0


def test_real_reference_uint8_ram_moving_up():
    # RAM coordinates are uint8 in the real cache; moving up (dy < 0) must not wrap around to 255.
    n_frames, K = 10, 2
    data = {
        "episode_id": np.zeros(n_frames, dtype=int),
        "in_battles": np.zeros(n_frames, dtype=bool),
        "map_ids": np.zeros(n_frames, dtype=np.uint8),
        "actions": np.full(n_frames, 7, dtype=np.uint8),
        "xs": np.full(n_frames, 10, dtype=np.uint8),
        "ys": np.full(n_frames, 20, dtype=np.uint8),
    }
    data["actions"][3:5] = 3  # window s=0: reference frame 3, two 'up' presses
    data["ys"][4:] = 19
    data["ys"][5:] = 18
    res = real_reference(data, eps=[0], d=3, K=K)
    assert res["count"] == 1
    np.testing.assert_array_equal(res["displacements"][0], [1, 2])
