"""Tests for Go-Explore exploration data collection (lab.collect)."""
from pathlib import Path
import glob
import numpy as np
import pytest

from lab.collect import (
    DEFAULT_STATE_PATH,
    ArchiveEntry,
    cell_key,
    choose_exploratory_action,
    run_episode,
    select_explore_cell,
)
from lab.data import to_small
from lab.env import ROM_PATH, PokemonRed
from src.game_state import extract_game_state

ROM_EXISTS = ROM_PATH.exists()
STATE_EXISTS = DEFAULT_STATE_PATH.exists()


def test_archive_cell_key():
    """Verify archive cell key function behavior and floor division."""
    assert cell_key(1, 29, 20) == (1, 9, 6)
    assert cell_key(13, 8, 71) == (13, 2, 23)
    assert cell_key(51, 0, 0) == (51, 0, 0)
    # Numpy integers compatibility
    assert cell_key(np.uint8(1), np.uint8(15), np.uint8(18)) == (1, 5, 6)


def test_select_explore_cell_weights():
    """Verify that cell selection favors rarely chosen cells and applies recency boost."""
    rng = np.random.default_rng(42)
    c1 = (1, 0, 0)
    c2 = (13, 0, 0)
    c3 = (51, 0, 0)

    # c1 chosen 100 times, found long ago (iter 1)
    # c2 chosen 0 times, found long ago (iter 1)
    # c3 chosen 0 times, found recently (iter 45 vs cur_iter 50)
    archive = {
        c1: ArchiveEntry(snapshot=b"", times_chosen=100, first_iter=1),
        c2: ArchiveEntry(snapshot=b"", times_chosen=0, first_iter=1),
        c3: ArchiveEntry(snapshot=b"", times_chosen=0, first_iter=45),
    }

    counts = {c1: 0, c2: 0, c3: 0}
    for _ in range(1000):
        chosen = select_explore_cell(archive, rng, cur_iter=50)
        counts[chosen] += 1

    # c3 has 2x weight of c2, and both much higher than c1
    assert counts[c3] > counts[c2]
    assert counts[c2] > counts[c1]
    assert counts[c1] < 100


def test_select_explore_cell_healthy_filter():
    """Verify that low HP triggers candidate filtering to healthy cells."""
    rng = np.random.default_rng(42)
    c_weak = (1, 0, 0)
    c_healthy = (13, 0, 0)
    archive = {
        c_weak: ArchiveEntry(snapshot=b"", times_chosen=0, hp_frac=0.1, first_iter=1),
        c_healthy: ArchiveEntry(snapshot=b"", times_chosen=0, hp_frac=1.0, first_iter=1),
    }
    for _ in range(50):
        chosen = select_explore_cell(archive, rng, cur_iter=10, prefer_healthy=True)
        assert chosen == c_healthy


def test_choose_exploratory_action():
    """Verify exploratory action behavior: never START, battle reflex, bump re-pick."""
    rng = np.random.default_rng(42)

    # 1. Battle reflex: always A (4)
    ram_battle = {"map_id": 51, "x": 10, "y": 10, "facing": 0, "in_battle": True, "dialog_open": False}
    act, _, _ = choose_exploratory_action(ram_battle, prev_pos=None, cur_dir=0, steps_left=0, rng=rng)
    assert act == 4

    # 2. Never START (6) over 500 actions
    ram_overworld = {"map_id": 1, "x": 10, "y": 10, "facing": 0, "in_battle": False, "dialog_open": False}
    cur_dir, steps_left, prev_pos, prev_act = 0, 0, None, -1
    for _ in range(500):
        act, cur_dir, steps_left = choose_exploratory_action(
            ram_overworld, prev_pos, cur_dir, steps_left, rng, prev_act=prev_act
        )
        assert act != 6, "Action must never be START (6)"
        prev_pos = (ram_overworld["map_id"], ram_overworld["x"], ram_overworld["y"])
        prev_act = act

    # 3. Bump re-pick switches away from cur_dir
    pos = (1, 10, 10)
    act, next_dir, _ = choose_exploratory_action(
        ram_overworld, prev_pos=pos, cur_dir=3, steps_left=0, rng=rng, prev_act=3
    )
    # If bumped while moving up (3), next_dir must not be 3 (unless rolled A/B)
    if act in (0, 1, 2, 3):
        assert act != 3


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="Start state not present")
def test_short_episode_and_dataset_format(tmp_path):
    """Verify that collecting an episode produces arrays with exact dataset keys, shapes, and dtypes."""
    env = PokemonRed(rom=ROM_PATH, state=DEFAULT_STATE_PATH)
    st0 = extract_game_state(env.pyboy)
    c0 = cell_key(st0.map_id, st0.x, st0.y)

    archive = {
        c0: ArchiveEntry(
            snapshot=env.snapshot(),
            times_chosen=0,
            map_id=st0.map_id,
            x=st0.x,
            y=st0.y,
            hp_frac=1.0,
            first_iter=0,
        )
    }
    rng = np.random.default_rng(123)

    ep_length = 50
    ep_data, ep_stats = run_episode(env, archive, rng, episode_idx=0, episode_length=ep_length)
    env.close()

    # Save to compressed npz
    out_file = tmp_path / "transitions_test.npz"
    np.savez_compressed(out_file, **ep_data)
    assert out_file.exists()

    loaded = np.load(out_file)

    # Required keys and shapes
    expected_shapes = {
        "obs": (ep_length, 144, 160, 3),
        "actions": (ep_length,),
        "rewards": (ep_length,),
        "episode_starts": (ep_length,),
        "map_ids": (ep_length,),
        "xs": (ep_length,),
        "ys": (ep_length,),
        "facings": (ep_length,),
        "in_battles": (ep_length,),
        "dialog_opens": (ep_length,),
        "badges": (ep_length,),
        "party_hps": (ep_length, 6),
        "party_max_hps": (ep_length, 6),
    }

    expected_dtypes = {
        "obs": np.uint8,
        "actions": np.uint8,
        "rewards": np.float32,
        "episode_starts": bool,
        "map_ids": np.uint8,
        "xs": np.uint8,
        "ys": np.uint8,
        "facings": np.uint8,
        "in_battles": bool,
        "dialog_opens": bool,
        "badges": np.uint8,
        "party_hps": np.int16,
        "party_max_hps": np.int16,
    }

    for k, shape in expected_shapes.items():
        assert k in loaded, f"Missing key '{k}' in saved npz"
        assert loaded[k].shape == shape, f"Key '{k}' expected shape {shape}, got {loaded[k].shape}"
        assert loaded[k].dtype == expected_dtypes[k], f"Key '{k}' expected dtype {expected_dtypes[k]}, got {loaded[k].dtype}"

    # episode_starts check
    assert loaded["episode_starts"][0] == True
    assert not np.any(loaded["episode_starts"][1:])

    # lab.data to_small compatibility
    small_frames = to_small(loaded["obs"])
    assert small_frames.shape == (ep_length, 72, 80)
    assert small_frames.dtype == np.uint8

    # Compare against an actual HF file if available
    hf_files = sorted(glob.glob("work/hf/dataset/transitions_*.npz"))
    if hf_files:
        ref_data = np.load(hf_files[0])
        for k in expected_shapes.keys():
            assert k in ref_data, f"Key '{k}' missing from reference dataset file"
            assert loaded[k].dtype == ref_data[k].dtype, f"Dtype mismatch for '{k}': {loaded[k].dtype} vs ref {ref_data[k].dtype}"
            # Check non-leading dimensions match
            assert loaded[k].shape[1:] == ref_data[k].shape[1:], f"Trailing shape mismatch for '{k}'"
