"""Evaluation metrics for PokeDreamer world model evaluation (pure numpy)."""
import warnings
import numpy as np


def manhattan(pred_xy, true_xy):
    """Compute Manhattan distance |dx| + |dy| over the last axis of (..., 2) arrays."""
    pred = np.asarray(pred_xy, dtype=float)
    true = np.asarray(true_xy, dtype=float)
    return np.sum(np.abs(pred - true), axis=-1)


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """Compute bootstrap CI (mean, lo, hi) resampling axis 0 with replacement, ignoring NaNs."""
    arr = np.asarray(values, dtype=float)
    w = arr.shape[0]
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, w, size=(n_boot, w))
    boot_samples = arr[boot_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        boot_means = np.nanmean(boot_samples, axis=1)
        mean = np.nanmean(arr, axis=0)
        lo = np.nanpercentile(boot_means, 100.0 * (alpha / 2.0), axis=0)
        hi = np.nanpercentile(boot_means, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return mean, lo, hi


def no_change_error(true_xy):
    """Compute (W, K) Manhattan error of predicting position[k] = position[0] for k=1..K."""
    arr = np.asarray(true_xy, dtype=float)
    diff = np.abs(arr[:, 1:, :] - arr[:, :1, :])
    return np.sum(diff, axis=-1)


def copy_last_frame_mse(frames):
    """Compute (W, K) per-window MSE of frames[:, k] vs frames[:, 0] for k=1..K."""
    arr = np.asarray(frames, dtype=float)
    diff = arr[:, 1:, ...] - arr[:, :1, ...]
    return np.mean(diff ** 2, axis=(-2, -1))


def frame_mse(pred, true):
    """Compute (W, K) mean squared error over pixel dimensions for each window and step."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    return np.mean((p - t) ** 2, axis=(-2, -1))


def psnr(mse, max_val=1.0):
    """Compute peak signal-to-noise ratio elementwise from mean squared error."""
    mse_arr = np.asarray(mse, dtype=float)
    with np.errstate(divide="ignore"):
        val = 10.0 * np.log10((float(max_val) ** 2) / mse_arr)
    if np.ndim(mse) == 0:
        return float(val)
    return val


def majority_class_accuracy(train_labels, test_labels):
    """Compute test accuracy of always predicting the most common label in train_labels."""
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if len(train) == 0 or len(test) == 0:
        return 0.0
    vals, counts = np.unique(train, return_counts=True)
    majority = vals[np.argmax(counts)]
    return float(np.mean(test == majority))
