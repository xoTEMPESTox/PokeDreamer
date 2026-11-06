"""test_gym_quest_color.py — Test --save-color recording in gym quest.

Verifies:
1. run_gym_quest records full resolution color frames (144x160x3 uint8) when save_color=True.
2. Number of color frames equals the trajectory length.
3. Color npz contains required keys: color, map_ids, goals, in_battles, leg, by.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from lab.agent.gym_quest import run_gym_quest
from lab.env import ROM_PATH


@pytest.mark.skipif(not ROM_PATH.exists(), reason="ROM not available")
def test_gym_quest_save_color(tmp_path: Path):
    # Prefer an existing route state if available, else starter state
    candidate_states = [
        Path("work/route_states/to_grass.state"),
        Path("saves/pokedex_viridian.state"),
    ]
    state_path = None
    for cand in candidate_states:
        if cand.is_file():
            state_path = cand
            break
    if state_path is None:
        pytest.skip("No suitable route save state found")

    seed = 100
    budget = 15

    res = run_gym_quest(
        budget=budget,
        seed=seed,
        save_traj=True,
        save_color=True,
        state_path=state_path,
        work_dir=tmp_path,
        traj_dir=tmp_path,
    )

    traj_path = tmp_path / f"run_seed_{seed}.npz"
    color_path = tmp_path / f"run_seed_{seed}_color.npz"

    assert traj_path.is_file(), f"Trajectory npz missing: {traj_path}"
    assert color_path.is_file(), f"Color npz missing: {color_path}"

    with np.load(traj_path, allow_pickle=True) as tdata, np.load(color_path, allow_pickle=True) as cdata:
        # Check keys
        for key in ["color", "map_ids", "goals", "in_battles", "leg", "by"]:
            assert key in cdata, f"Missing key in color npz: {key}"

        color_arr = cdata["color"]
        traj_arr = tdata["frames"]

        # Check shape and dtype
        assert color_arr.ndim == 4, f"Expected 4D array, got ndim={color_arr.ndim}"
        assert color_arr.shape[1:] == (144, 160, 3), f"Expected (*, 144, 160, 3), got {color_arr.shape}"
        assert color_arr.dtype == np.uint8, f"Expected uint8 dtype, got {color_arr.dtype}"

        # Check lengths match
        assert len(color_arr) == len(traj_arr), f"Length mismatch: {len(color_arr)} vs {len(traj_arr)}"
        assert len(color_arr) > 0, "Expected at least one recorded frame"

        # Check auxiliary arrays lengths match
        assert len(cdata["map_ids"]) == len(color_arr)
        assert len(cdata["goals"]) == len(color_arr)
        assert len(cdata["in_battles"]) == len(color_arr)
        assert len(cdata["leg"]) == len(color_arr)
        assert len(cdata["by"]) == len(color_arr)

        # Check valid 'by' values
        unique_bys = set(cdata["by"])
        assert unique_bys.issubset({"policy", "bfs", ""}), f"Unexpected by values: {unique_bys}"
