"""Tests for model-free PPO baseline (lab/agent/ppo_real.py)."""
import multiprocessing as mp
import numpy as np
import pytest
import torch

from lab.agent.imagine_rl import HALLUCINATION_PENALTY, STEP_COST, WARP_BONUS
from lab.agent.ppo_real import (
    TASK_ROUTES,
    compute_gae,
    compute_reward,
    worker_fn,
)
from lab.agent.tasks import TASKS
from lab.env import ROM_PATH

ROM_EXISTS = ROM_PATH.exists()


def test_gae_toy_trajectory_termination_and_truncation():
    """Test GAE advantages and lambda-returns on a trajectory with termination and truncation.

    Step 0: normal step (not done)
    Step 1: truncation (episode cap reached, bootstraps from next_values[1])
    Step 2: normal step (not done)
    Step 3: termination (success reached, next_values[3] is 0, done)
    """
    rewards = torch.tensor([1.0, 2.0, 0.5, 5.0], dtype=torch.float32)
    values = torch.tensor([2.0, 3.0, 1.0, 2.5], dtype=torch.float32)
    terminated = torch.tensor([False, False, False, True], dtype=torch.bool)
    truncated = torch.tensor([False, True, False, False], dtype=torch.bool)
    next_values = torch.tensor([3.0, 4.0, 2.5, 0.0], dtype=torch.float32)

    gamma = 0.95
    lam = 0.9

    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, next_values, gamma=gamma, lam=lam
    )

    expected_adv = torch.tensor([4.2440, 2.8000, 4.0125, 2.5000], dtype=torch.float32)
    expected_ret = torch.tensor([6.2440, 5.8000, 5.0125, 5.0000], dtype=torch.float32)

    assert torch.allclose(advantages, expected_adv, atol=1e-4), f"Got adv: {advantages}"
    assert torch.allclose(returns, expected_ret, atol=1e-4), f"Got ret: {returns}"


def test_reward_progress_same_map():
    """Test compute_reward on same-map movement."""
    exits = TASKS["leave_house"]["exits"]  # 38: [(7, 1)], 37: [(2, 7)]
    route = TASK_ROUTES["leave_house"]

    # Bedroom (38): exit at (7, 1). Distance from (3, 6) is 9; from (3, 5) is 8.
    prev_ram = {"map_id": 38, "x": 3, "y": 6}
    cur_ram = {"map_id": 38, "x": 3, "y": 5}
    r, idx = compute_reward(prev_ram, cur_ram, exits, route=route, best_route_idx=0)
    assert np.isclose(r, 1.0 - STEP_COST)
    assert idx == 0

    # Moving away increases distance by 1
    cur_ram_away = {"map_id": 38, "x": 3, "y": 7}
    r_away, idx = compute_reward(prev_ram, cur_ram_away, exits, route=route, best_route_idx=0)
    assert np.isclose(r_away, -1.0 - STEP_COST)

    # Wall bump
    r_bump, idx = compute_reward(prev_ram, prev_ram, exits, route=route, best_route_idx=0)
    assert np.isclose(r_bump, -STEP_COST)


def test_route_order_in_out_loop_net_negative():
    """Test that the in-out door loop (e.g. at Oak's lab) is strictly net negative.

    Entering Oak's lab costs -HALLUCINATION_PENALTY.
    Stepping back out into Pallet Town (index 0 <= best_route_idx 0) also costs
    -HALLUCINATION_PENALTY, preventing the +3 infinite reward cycle.
    """
    exits = TASKS["route1"]["exits"]
    route = TASK_ROUTES["route1"]  # [0, 12]
    best_idx = 0  # starts in Pallet Town (map 0)

    # 1. Step from Pallet Town (0) into Oak's lab (40)
    outside_lab = {"map_id": 0, "x": 12, "y": 11}
    inside_lab = {"map_id": 40, "x": 4, "y": 11}
    r_enter, best_idx = compute_reward(outside_lab, inside_lab, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r_enter, -HALLUCINATION_PENALTY), f"Expected {-HALLUCINATION_PENALTY}, got {r_enter}"
    assert best_idx == 0

    # 2. Step from inside Oak's lab (40) back out to Pallet Town (0)
    r_exit, best_idx = compute_reward(inside_lab, outside_lab, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r_exit, -HALLUCINATION_PENALTY), f"Expected {-HALLUCINATION_PENALTY}, got {r_exit}"
    assert best_idx == 0

    # Total two-step cycle must be strictly net negative
    total_cycle = r_enter + r_exit
    assert total_cycle < 0.0, f"Expected net negative, got {total_cycle}"
    assert np.isclose(total_cycle, -2.0 * HALLUCINATION_PENALTY)


def test_route_order_forward_progress_two_bonuses():
    """Test that forward progress 38 -> 37 -> 0 in leave_house earns two +WARP_BONUS rewards."""
    exits = TASKS["leave_house"]["exits"]
    route = TASK_ROUTES["leave_house"]  # [38, 37, 0]
    best_idx = 0  # starts on map 38

    # 1. Warp 38 -> 37 at bedroom stairs (7, 1)
    stairs_38 = {"map_id": 38, "x": 7, "y": 1}
    floor1_37 = {"map_id": 37, "x": 7, "y": 1}
    r1, best_idx = compute_reward(stairs_38, floor1_37, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r1, WARP_BONUS), f"Expected {WARP_BONUS}, got {r1}"
    assert best_idx == 1  # reached map 37

    # 2. Warp 37 -> 0 at front door (2, 7)
    door_37 = {"map_id": 37, "x": 2, "y": 7}
    pallet_0 = {"map_id": 0, "x": 5, "y": 5}
    r2, best_idx = compute_reward(door_37, pallet_0, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r2, WARP_BONUS), f"Expected {WARP_BONUS}, got {r2}"
    assert best_idx == 2  # reached map 0


def test_route_order_backtrack_no_second_bonus():
    """Test that backtracking 37 -> 38 -> 37 earns no second warp bonus."""
    exits = TASKS["leave_house"]["exits"]
    route = TASK_ROUTES["leave_house"]  # [38, 37, 0]
    best_idx = 1  # already reached map 37

    # 1. Backtrack 37 -> 38 (walk back upstairs)
    floor1_stairs = {"map_id": 37, "x": 7, "y": 1}
    bedroom_stairs = {"map_id": 38, "x": 7, "y": 1}
    r_back, best_idx = compute_reward(floor1_stairs, bedroom_stairs, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r_back, -HALLUCINATION_PENALTY), f"Expected {-HALLUCINATION_PENALTY}, got {r_back}"
    assert best_idx == 1  # still 1

    # 2. Return 38 -> 37 (walk back downstairs to already visited map index)
    r_return, best_idx = compute_reward(bedroom_stairs, floor1_stairs, exits, route=route, best_route_idx=best_idx)
    assert np.isclose(r_return, -HALLUCINATION_PENALTY), f"Expected {-HALLUCINATION_PENALTY}, got {r_return}"
    assert best_idx == 1  # no second bonus


def test_post_warp_coordinate_lag_loop_net_negative():
    """Test that the RAM coordinate lag warp loop is strictly net negative with the lag filter."""
    exits = TASKS["route1"]["exits"]
    route = TASK_ROUTES["route1"]
    best_idx = 0

    # User trace:
    # t=0 U 0(12,12) -> 40(12,11) r=-2.00
    r0, best_idx = compute_reward(
        {"map_id": 0, "x": 12, "y": 12},
        {"map_id": 40, "x": 12, "y": 11},
        exits,
        route=route,
        best_route_idx=best_idx,
    )
    assert np.isclose(r0, -HALLUCINATION_PENALTY)

    # t=2 R 40(12,11) -> 40(5,11) coordinates update 1-2 steps later: |dx|+|dy| = 7 > 2 -> r = -STEP_COST
    r1, best_idx = compute_reward(
        {"map_id": 40, "x": 12, "y": 11},
        {"map_id": 40, "x": 5, "y": 11},
        exits,
        route=route,
        best_route_idx=best_idx,
    )
    assert np.isclose(r1, -STEP_COST)

    # t=4 D 40(5,11) -> 0(5,11) r=-2.00
    r2, best_idx = compute_reward(
        {"map_id": 40, "x": 5, "y": 11},
        {"map_id": 0, "x": 5, "y": 11},
        exits,
        route=route,
        best_route_idx=best_idx,
    )
    assert np.isclose(r2, -HALLUCINATION_PENALTY)

    # t=5 D 0(5,11) -> 0(12,11) coordinates update: |dx|+|dy| = 7 > 2 -> r = -STEP_COST
    r3, best_idx = compute_reward(
        {"map_id": 0, "x": 5, "y": 11},
        {"map_id": 0, "x": 12, "y": 11},
        exits,
        route=route,
        best_route_idx=best_idx,
    )
    assert np.isclose(r3, -STEP_COST)

    # t=6 D 0(12,11) -> 0(12,12) r=-1.05 (moving 1 step away from route 1 exit)
    r4, best_idx = compute_reward(
        {"map_id": 0, "x": 12, "y": 11},
        {"map_id": 0, "x": 12, "y": 12},
        exits,
        route=route,
        best_route_idx=best_idx,
    )
    assert np.isclose(r4, -1.0 - STEP_COST)

    total_loop = r0 + r1 + r2 + r3 + r4
    assert total_loop < 0.0, f"Expected loop to be net negative, got {total_loop}"
    assert np.isclose(total_loop, -2.0 * HALLUCINATION_PENALTY - 2.0 * STEP_COST - 1.0 - STEP_COST)


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
def test_worker_process_step_and_reset():
    """Test that a spawned worker process starts, resets, and steps correctly."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=worker_fn,
        args=(child_conn, "route1", str(ROM_PATH), 42),
        daemon=True,
    )
    proc.start()

    try:
        # Reset
        parent_conn.send(("reset", None))
        assert parent_conn.poll(20.0), "Worker reset timed out"
        obs = parent_conn.recv()

        assert isinstance(obs, np.ndarray), "Obs should be numpy array"
        assert obs.shape == (4, 72, 80), f"Expected shape (4, 72, 80), got {obs.shape}"
        assert obs.dtype == np.uint8, f"Expected dtype uint8, got {obs.dtype}"

        # Step
        parent_conn.send(("step", 0))  # down
        assert parent_conn.poll(20.0), "Worker step timed out"
        res = parent_conn.recv()
        assert not (isinstance(res, tuple) and len(res) == 2 and res[0] == "error"), f"Worker error: {res}"

        next_obs, reward, term, trunc, info, presses = res
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (4, 72, 80)
        assert next_obs.dtype == np.uint8
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(presses, int) and presses >= 1
    finally:
        parent_conn.send(("close", None))
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
