"""Tests for lab/v1_reeval.py."""

import cv2
import numpy as np
import pytest

from lab.v1_reeval import (
    carve_train_val_episodes,
    get_valid_window_indices,
    to_40x36,
)


def test_to_40x36_conversion():
    """Verify 40x36 conversion exactly matches cv2.resize with INTER_AREA."""
    rng = np.random.default_rng(123)
    # Tiny fake array of 4 frames at native 144x160 RGB
    fake_obs = rng.integers(0, 256, size=(4, 144, 160, 3), dtype=np.uint8)

    converted = to_40x36(fake_obs)
    assert converted.shape == (4, 36, 40, 3)
    assert converted.dtype == np.uint8

    for i in range(len(fake_obs)):
        expected = cv2.resize(fake_obs[i], (40, 36), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(converted[i], expected)


def test_carve_train_val_episodes():
    """Verify episode split helper partitions pool disjointly and deterministically."""
    train_pool = list(range(100))
    train_eps, val_eps = carve_train_val_episodes(train_pool, val_count=10, seed=42)

    # Correct lengths
    assert len(val_eps) == 10
    assert len(train_eps) == 90

    # Disjointness
    assert set(train_eps).isdisjoint(set(val_eps))

    # Union reconstructs original pool
    assert set(train_eps) | set(val_eps) == set(train_pool)

    # Determinism
    t2, v2 = carve_train_val_episodes(train_pool, val_count=10, seed=42)
    assert train_eps == t2
    assert val_eps == v2

    # Different seed yields different split
    t3, v3 = carve_train_val_episodes(train_pool, val_count=10, seed=999)
    assert val_eps != v3


def test_get_valid_window_indices_never_mix_episodes():
    """Verify sequence window start indices never cross episode boundaries."""
    episodes = [0, 3, 7, 12]
    ep_len = 100
    seq_len = 30

    starts, window_eps = get_valid_window_indices(episodes, ep_len=ep_len, seq_len=seq_len)

    expected_per_ep = ep_len - seq_len + 1  # 71 windows per episode
    assert len(starts) == len(episodes) * expected_per_ep
    assert len(window_eps) == len(starts)

    for start_idx, ep_id in zip(starts, window_eps):
        ep_start = ep_id * ep_len
        ep_end = (ep_id + 1) * ep_len

        # Window must start at or after episode start
        assert start_idx >= ep_start
        # Entire window of length seq_len must end at or before episode end
        assert start_idx + seq_len <= ep_end
        # The window must not touch the next episode
        assert start_idx + seq_len - 1 < ep_end
