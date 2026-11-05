"""Tests for learned grind/heal movement tasks, mpc success_tiles, and quest wiring fallback."""
from pathlib import Path
from unittest.mock import patch
import numpy as np

from lab.agent.gym_quest import grind_skill, heal_skill, run_leg
from lab.agent.mpc import run_episode
from lab.agent.nav import MAP_VIRIDIAN_CITY
from lab.agent.tasks import TASKS, geodesic_fields, train_geodesic_fields
from lab.data import load
from lab.env import PokemonRed

NEW_TASKS = [
    "leave_viridian_pc",
    "to_grass",
    "to_viridian_pc",
    "to_pewter_pc",
    "leave_pewter_pc",
]

EXPECTED_START_TILES = {
    "leave_viridian_pc": (41, 3, 4),
    "to_grass": (1, 23, 26),
    "to_viridian_pc": (12, 13, 30),
    "to_pewter_pc": (2, 18, 35),
    "leave_pewter_pc": (58, 3, 4),
}


def test_new_tasks_well_formed_and_reachable():
    """All 5 new tasks must be well-formed, start states must exist and match expected (map, x, y),
    and every exit tile must be reachable (finite geodesic distance) from its start tile in the
    extended dataset's train split."""
    data, splits = load(extended=True)
    train_mask = np.isin(data["episode_id"], splits["train"])
    train_maps = data["map_ids"][train_mask]
    train_xs = data["xs"][train_mask]
    train_ys = data["ys"][train_mask]
    train_eps = data["episode_id"][train_mask]

    for task_name in NEW_TASKS:
        assert task_name in TASKS, f"Task {task_name} missing from TASKS"
        spec = TASKS[task_name]
        assert "state" in spec
        assert "exits" in spec
        assert "success_map" in spec
        assert "train_maps" in spec
        assert Path(spec["state"]).is_file(), f"State file {spec['state']} does not exist"

        expected_map, expected_x, expected_y = EXPECTED_START_TILES[task_name]
        with PokemonRed(state=spec["state"]) as env:
            _, ram = env.observe()
            assert (int(ram["map_id"]), int(ram["x"]), int(ram["y"])) == (expected_map, expected_x, expected_y)

        # Verify geodesic distance from start tile to exit on start map
        fields = geodesic_fields(train_maps, train_xs, train_ys, train_eps, spec["exits"])
        assert expected_map in fields, f"Start map {expected_map} not in geodesic fields for {task_name}"
        f = fields[expected_map]
        dist = f[expected_y, expected_x]
        unreachable_val = f.max()
        assert dist < unreachable_val, (
            f"Start tile ({expected_x}, {expected_y}) on map {expected_map} has unreachable "
            f"distance {dist} (max {unreachable_val}) for task {task_name}"
        )

        # Every exit tile must have distance 0 in its map's field
        for m, tiles in spec["exits"].items():
            if m in fields:
                for tx, ty in tiles:
                    if 0 <= tx < fields[m].shape[1] and 0 <= ty < fields[m].shape[0]:
                        assert fields[m][ty, tx] == 0, f"Exit tile ({tx}, {ty}) on map {m} has non-zero distance"


def test_mpc_success_tiles_logic():
    """mpc run_episode success requires both success_map AND (x, y) in success_tiles[map] when specified."""
    fields = train_geodesic_fields(TASKS["to_grass"]["exits"], extended=True)

    # 1. With success_tiles and max_steps=30: reaches map 12 at step 13, but does NOT succeed because not on grass
    rng = np.random.default_rng(200)
    res_with, _ = run_episode(
        policy="greedy",
        model=None,
        probe=None,
        rng=rng,
        max_steps=30,
        n=1,
        h=1,
        device="cpu",
        task="to_grass",
        fields=fields,
    )
    assert not res_with["success"], "Must not succeed when reaching success_map on non-success tiles"
    assert res_with["trace"][-1][0] == 12, "Must have reached map 12 within 30 steps"

    # 2. Without success_tiles and max_steps=30: succeeds immediately upon reaching map 12 at step 13
    spec_notiles = dict(TASKS["to_grass"])
    del spec_notiles["success_tiles"]
    with patch.dict(TASKS, {"to_grass_notiles": spec_notiles}):
        rng = np.random.default_rng(200)
        res_without, _ = run_episode(
            policy="greedy",
            model=None,
            probe=None,
            rng=rng,
            max_steps=30,
            n=1,
            h=1,
            device="cpu",
            task="to_grass_notiles",
            fields=fields,
        )
        assert res_without["success"], "Must succeed on reaching success_map when success_tiles is absent"
        assert res_without["steps"] == 13, f"Expected success at step 13, got {res_without['steps']}"


def test_gym_quest_wiring_fallback():
    """When policy files are missing, learned navigation legs must fall back to BFS cleanly and log to run_legs."""
    # 1. run_leg fallback with missing policy
    run_legs = []
    with PokemonRed(state="work/route_states/leave_viridian_pc.state") as env:
        res = run_leg(
            env=env,
            leg="leave_viridian_pc",
            target_map=MAP_VIRIDIAN_CITY,
            policy_path="work/agent_rl/nonexistent_policy/policy.pt",
            fallback_maps=[MAP_VIRIDIAN_CITY],
            budget=200,
        )
        assert res["success"], "Fallback must successfully reach target_map"
        assert res["reached_by_policy"] is False, "Missing policy must not reach target by policy"
        assert res["fallback_presses"] > 0, "Fallback presses must be recorded"
        assert int(env.observe()[1]["map_id"]) == MAP_VIRIDIAN_CITY

    # 2. heal_skill fallback with missing policies
    run_legs = []
    with PokemonRed(state="work/route_states/to_grass.state") as env:
        h_res = heal_skill(
            env,
            budget=800,
            nav="learned",
            leg_policies={"to_viridian_pc": "work/agent_rl/nonexistent/policy.pt",
                          "leave_viridian_pc": "work/agent_rl/nonexistent/policy.pt"},
            run_legs=run_legs,
        )
        assert h_res["healed"], "heal_skill must heal starter"
        assert h_res["success"], "heal_skill must return success=True"
        assert len(run_legs) >= 1, "Must log leg attempts to run_legs"
        for l in run_legs:
            assert l["reached_by_policy"] is False
            assert "leg" in l and "policy_presses" in l and "fallback_presses" in l
        assert int(env.observe()[1]["map_id"]) == MAP_VIRIDIAN_CITY

    # 2b. a policy that wastes its whole budget must not starve the heal (gym_learned_moves_s0, 2026-11-06: to_pewter_pc
    # used all 800 presses, BFS entered the PC in ~10, then the counter walk got budget 0 and the agent stayed unhealed
    # on the PC mat)
    run_legs = []
    steps = iter([1, 2] * 5000)  # left, right: stays put
    with PokemonRed(state="work/route_states/to_pewter_pc.state") as env, \
            patch("lab.agent.gym_quest.load_leg_policy", return_value=object()), \
            patch("lab.agent.gym_quest.learned_action", side_effect=lambda net, frames, device: next(steps)):
        h_res = heal_skill(env, budget=800, nav="learned", run_legs=run_legs,
                           leg_policies={"to_pewter_pc": __file__, "leave_pewter_pc": __file__})
        assert run_legs[0]["leg"] == "to_pewter_pc" and run_legs[0]["policy_presses"] >= 800
        assert h_res["healed"], "a wasted policy budget must leave the counter walk its own budget"
        assert int(env.observe()[1]["map_id"]) == 2, "must leave the Pewter PC"

    # 3. grind_skill fallback with missing to_grass policy
    run_legs = []
    with PokemonRed(state="work/route_states/to_grass.state") as env:
        g_res = grind_skill(
            env,
            target_level=8,  # starter is level 8, so it won't need to pace
            hp_threshold=0.35,
            budget=500,
            nav="learned",
            leg_policies={"to_grass": "work/agent_rl/nonexistent/policy.pt"},
            run_legs=run_legs,
        )
        assert any(l["leg"] == "to_grass" for l in run_legs)
        to_grass_log = [l for l in run_legs if l["leg"] == "to_grass"][0]
        assert to_grass_log["reached_by_policy"] is False
        assert to_grass_log["fallback_presses"] > 0
