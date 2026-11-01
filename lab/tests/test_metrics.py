"""Tests for evaluation metrics on tiny hand-computable arrays."""
import numpy as np
import pytest

from lab.eval.metrics import (
    bootstrap_ci,
    copy_last_frame_mse,
    frame_mse,
    majority_class_accuracy,
    manhattan,
    no_change_error,
    psnr,
)


def test_manhattan():
    # 1D pair
    p1 = np.array([3, 7])
    t1 = np.array([1, 2])
    assert manhattan(p1, t1) == 7.0

    # 2D pairs
    pred = np.array([[1, 2], [5, 10]])
    true = np.array([[2, 4], [5, 6]])
    # |1-2| + |2-4| = 3; |5-5| + |10-6| = 4
    np.testing.assert_allclose(manhattan(pred, true), [3.0, 4.0])

    # 3D array (W, K, 2)
    pred_3d = np.zeros((2, 3, 2))
    true_3d = np.ones((2, 3, 2))
    np.testing.assert_allclose(manhattan(pred_3d, true_3d), np.full((2, 3), 2.0))


def test_no_change_error():
    # Walk of known steps: W=1, K+1=4 (K=3)
    # context is [10, 10]
    walk = np.array([[[10, 10], [11, 10], [11, 13], [8, 12]]])
    # k=1: |11-10| + |10-10| = 1.0
    # k=2: |11-10| + |13-10| = 4.0
    # k=3: |8-10| + |12-10| = 4.0
    res = no_change_error(walk)
    assert res.shape == (1, 3)
    np.testing.assert_allclose(res, [[1.0, 4.0, 4.0]])

    # Multiple windows
    walk_2w = np.array([
        [[0, 0], [1, 1]],
        [[5, 5], [5, 5]],
    ])
    np.testing.assert_allclose(no_change_error(walk_2w), [[2.0], [0.0]])


def test_copy_last_frame_mse():
    # frames: shape (W=2, K+1=3, H=2, W_pixel=2)
    frames = np.zeros((2, 3, 2, 2), dtype=float)
    # Window 0
    # k=0: all 0.0
    # k=1: all 1.0 -> (1.0 - 0.0)^2 = 1.0
    frames[0, 1] = 1.0
    # k=2: 1 pixel is 2.0, 3 pixels are 0.0 -> (4)/4 = 1.0
    frames[0, 2, 0, 0] = 2.0

    # Window 1
    # k=0: all 0.5
    frames[1, 0] = 0.5
    # k=1: all 0.5 -> MSE = 0.0
    frames[1, 1] = 0.5
    # k=2: all 1.0 -> (1.0 - 0.5)^2 = 0.25
    frames[1, 2] = 1.0

    res = copy_last_frame_mse(frames)
    assert res.shape == (2, 2)
    np.testing.assert_allclose(res, [[1.0, 1.0], [0.0, 0.25]])


def test_frame_mse():
    p = np.full((1, 1, 2, 2), 0.5)
    t = np.zeros((1, 1, 2, 2))
    np.testing.assert_allclose(frame_mse(p, t), [[0.25]])


def test_psnr():
    assert np.isclose(psnr(0.01, max_val=1.0), 20.0)
    assert np.isclose(psnr(0.1, max_val=1.0), 10.0)
    assert np.isclose(psnr(1.0, max_val=1.0), 0.0)
    assert np.isinf(psnr(0.0, max_val=1.0))
    # elementwise array test
    arr = np.array([0.01, 0.1])
    np.testing.assert_allclose(psnr(arr, max_val=1.0), [20.0, 10.0])


def test_bootstrap_ci_constant():
    # 1D constant array returns (c, c, c)
    v1 = np.full(50, 4.25)
    mean1, lo1, hi1 = bootstrap_ci(v1)
    assert mean1.shape == ()
    assert lo1.shape == ()
    assert hi1.shape == ()
    assert np.isclose(mean1, 4.25)
    assert np.isclose(lo1, 4.25)
    assert np.isclose(hi1, 4.25)

    # 2D constant array returns (c, c, c) of shape (K,)
    v2 = np.full((40, 3), 7.5)
    mean2, lo2, hi2 = bootstrap_ci(v2)
    assert mean2.shape == (3,)
    assert lo2.shape == (3,)
    assert hi2.shape == (3,)
    np.testing.assert_allclose(mean2, [7.5, 7.5, 7.5])
    np.testing.assert_allclose(lo2, [7.5, 7.5, 7.5])
    np.testing.assert_allclose(hi2, [7.5, 7.5, 7.5])


def test_bootstrap_ci_nan_masked():
    # NaN-masked 1D input ignores NaNs
    v1 = np.array([5.0, np.nan, 5.0, np.nan, 5.0])
    mean1, lo1, hi1 = bootstrap_ci(v1)
    assert np.isclose(mean1, 5.0)
    assert np.isclose(lo1, 5.0)
    assert np.isclose(hi1, 5.0)

    # 2D with NaNs
    v2 = np.array([
        [2.0, np.nan],
        [2.0, 8.0],
        [np.nan, 8.0],
        [2.0, 8.0],
    ])
    mean2, lo2, hi2 = bootstrap_ci(v2)
    np.testing.assert_allclose(mean2, [2.0, 8.0])
    np.testing.assert_allclose(lo2, [2.0, 8.0])
    np.testing.assert_allclose(hi2, [2.0, 8.0])


def test_majority_class_accuracy():
    # train has majority label 1 (count 3 vs count 1)
    train = [1, 1, 1, 2]
    # test has two 1s out of four entries
    test = [1, 2, 1, 3]
    assert majority_class_accuracy(train, test) == 0.5

    # 100% accuracy when all match
    assert majority_class_accuracy([5, 5, 6], [5, 5]) == 1.0

    # 0% accuracy when none match
    assert majority_class_accuracy([5, 5, 6], [6, 7]) == 0.0
