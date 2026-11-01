"""The extended dataset must append the collected episodes after the HF ones without mixing episode IDs or splits."""
import numpy as np

from lab.data import merge


def _toy(n_eps, start_map):
    n = 4 * n_eps
    starts = np.zeros(n, bool)
    starts[::4] = True
    return {"frames": np.zeros((n, 72, 80), np.uint8), "map_ids": np.full(n, start_map), "episode_starts": starts,
            "episode_id": np.cumsum(starts) - 1}


def test_merge_offsets_episodes_and_keeps_base_splits():
    base, ext = _toy(3, 38), _toy(2, 51)
    base_splits = {"train": [0, 1], "test_id": [2], "test_ood": []}
    ext_splits = {"train": [0], "test": [1]}
    data, splits = merge(base, base_splits, ext, ext_splits)
    assert len(data["frames"]) == 20 and list(np.unique(data["episode_id"])) == [0, 1, 2, 3, 4]
    assert splits["train"] == [0, 1, 3] and splits["test_id"] == [2] and splits["test_ext"] == [4]
    assert (data["map_ids"][data["episode_id"] == 4] == 51).all()
