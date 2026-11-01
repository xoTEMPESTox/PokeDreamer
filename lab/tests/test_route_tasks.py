"""Tests for route tasks along the gym route and the transition counter."""
from pathlib import Path
import numpy as np

from lab.agent.route_exits import count_transitions
from lab.agent.tasks import TASKS
from lab.env import PokemonRed


def test_transition_counter_lagged_warp():
    """At warps, Game Boy RAM updates wCurMap 1-2 steps before x/y. The transition counter
    must correctly record the exit tile on m before the transition."""
    # Toy episode: player on map 1 walks down to door (4, 11).
    # At step 4, map_id becomes 2, but RAM coordinates lag: (4, 11) lingers for 1 step
    # before updating to (10, 5) on map 2 at step 5.
    maps = np.array([1, 1, 1, 1, 2, 2, 2])
    xs   = np.array([4, 4, 4, 4, 4, 10, 10])
    ys   = np.array([8, 9, 10, 11, 11, 5, 5])
    eps  = np.zeros(7, int)

    counts = count_transitions(maps, xs, ys, episodes=eps, skip_first=0, step_back=1)
    assert (1, 2) in counts
    assert counts[(1, 2)][(4, 11)] == 1
    assert len(counts[(1, 2)]) == 1

    # Multi-episode with map change across episode boundary
    maps2 = np.array([1, 1, 2, 3, 3, 3])
    xs2   = np.array([5, 5, 5, 12, 12, 20])
    ys2   = np.array([2, 1, 1, 8, 8, 30])
    eps2  = np.array([0, 0, 0, 1, 1, 1])  # transition 2->3 straddles episode boundary, should not count
    counts2 = count_transitions(maps2, xs2, ys2, episodes=eps2, skip_first=0, step_back=1)
    assert (1, 2) in counts2
    assert counts2[(1, 2)][(5, 1)] == 1
    assert (2, 3) not in counts2


def test_new_tasks_start_states_exist_and_load():
    """Each new task along the gym route must have an existing save state that loads in the emulator."""
    expected_start_maps = {
        "to_route2": 1,
        "forest": 50,
        "to_pewter": 47,
        "to_gym": 2,
    }
    for task_name, start_map in expected_start_maps.items():
        assert task_name in TASKS, f"Missing task {task_name} in TASKS"
        spec = TASKS[task_name]
        state_path = Path(spec["state"])
        assert state_path.exists(), f"State file {state_path} does not exist"

        env = PokemonRed(state=spec["state"])
        _, ram = env.step(5)  # press B
        env.close()
        assert ram["map_id"] == start_map, f"Task {task_name} expected start map {start_map}, got {ram['map_id']}"


def test_new_tasks_start_maps_have_exits():
    """Each new task must have non-empty exits for its start map, route maps, and valid train_maps."""
    expected_start_maps = {
        "to_route2": 1,
        "forest": 50,
        "to_pewter": 47,
        "to_gym": 2,
    }
    for task_name, start_map in expected_start_maps.items():
        spec = TASKS[task_name]
        exits = spec["exits"]
        assert start_map in exits, f"Start map {start_map} not in exits for task {task_name}"
        assert len(exits[start_map]) > 0, f"No exits defined for start map {start_map} in task {task_name}"
        for tile in exits[start_map]:
            assert isinstance(tile, tuple) and len(tile) == 2
            assert isinstance(tile[0], int) and isinstance(tile[1], int)

        assert "success_map" in spec
        assert "train_maps" in spec
        assert start_map in spec["train_maps"]
