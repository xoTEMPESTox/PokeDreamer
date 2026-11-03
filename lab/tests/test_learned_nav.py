"""test_learned_nav.py — Tests for learned (frames-only) navigation in gym quest.

Tests:
1. Leg selection from stage: verifies stage maps map to expected legs.
2. Fallback when policy file is missing: run_leg falls back to navigate_to.
3. Leg runner with a stub policy on a real state: tiny budget triggers policy presses then fallback.
4. --nav bfs default unchanged: defaults and behaviour stay pure BFS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from lab.agent.gym_quest import (
    DEFAULT_LEG_POLICIES,
    STAGE_TO_LEG,
    TRAVEL_LEGS,
    first_travel_leg,
    leg_for_stage,
    plan_gym,
    run_leg,
    travel_to_pewter_skill,
    fight_gym_skill,
    run_gym_quest,
)
from lab.agent.nav import (
    navigate_to,
    MAP_VIRIDIAN_CITY,
    MAP_ROUTE_2,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE,
    MAP_VIRIDIAN_FOREST,
    MAP_VIRIDIAN_FOREST_NORTH_GATE,
    MAP_PEWTER_CITY,
    MAP_PEWTER_GYM,
)
from lab.env import ROM_PATH, PokemonRed

SAVE_STATE_PATH = Path("saves/pokedex_viridian.state")


# ── Test 1: Leg selection from stage ─────────────────────────────────────────
def test_leg_selection_from_stage():
    """Verify that stage target maps map to the expected learned policy leg."""
    assert leg_for_stage(MAP_VIRIDIAN_CITY) == "viridian"
    assert leg_for_stage(MAP_ROUTE_2) == "to_route2"
    assert leg_for_stage(MAP_VIRIDIAN_FOREST_SOUTH_GATE) == "to_gate"
    assert leg_for_stage(MAP_VIRIDIAN_FOREST_NORTH_GATE) == "forest"
    assert leg_for_stage(MAP_PEWTER_CITY) == "to_pewter"
    assert leg_for_stage(MAP_PEWTER_GYM) == "to_gym"

    # All 6 legs must have default policy file entries
    expected_legs = ["viridian", "to_route2", "to_gate", "forest", "to_pewter", "to_gym"]
    for leg in expected_legs:
        assert leg in DEFAULT_LEG_POLICIES
        assert DEFAULT_LEG_POLICIES[leg].endswith(".pt")

    # Verify TRAVEL_LEGS sequence matches the route to Pewter
    travel_leg_names = [leg for leg, _, _ in TRAVEL_LEGS]
    assert travel_leg_names == ["viridian", "to_route2", "to_gate", "forest", "to_pewter"]

    # Unknown map raises ValueError
    with pytest.raises(ValueError, match="Unknown leg for target stage map"):
        leg_for_stage(999)


# ── Test 2: Fallback when policy file is missing ──────────────────────────────
@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_fallback_when_policy_missing():
    """Verify run_leg falls back to navigate_to when the policy checkpoint does not exist."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        # Player starts in Viridian City (map 1)
        cur_map = env.observe()[1]["map_id"]
        assert cur_map == MAP_VIRIDIAN_CITY

        nonexistent_path = Path("work/agent_rl/definitely_missing_leg_policy/policy.pt")
        assert not nonexistent_path.exists()

        res = run_leg(
            env=env,
            leg="to_route2",
            target_map=MAP_ROUTE_2,
            policy_path=nonexistent_path,
            fallback_maps=[MAP_ROUTE_2],
            budget=150,
        )

        assert res["leg"] == "to_route2"
        assert res["reached_by_policy"] is False
        assert res["policy_presses"] == 0
        assert res["fallback_presses"] > 0
        assert res["success"] is True
        assert env.observe()[1]["map_id"] == MAP_ROUTE_2


# ── Test 3: Leg runner with stub policy on real state ─────────────────────────
class StubPolicy(torch.nn.Module):
    """Stub policy returning a deterministic distribution over actions."""

    def __init__(self, action_idx: int = 0):
        super().__init__()
        self.action_idx = action_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        logits = torch.full((batch_size, 4), -1e9)
        logits[:, self.action_idx] = 10.0
        value = torch.zeros(batch_size, 1)
        return logits, value


@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_leg_runner_with_stub_policy_tiny_budget():
    """Verify run_leg runs the learned policy for tiny budget then falls back to navigate_to."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        cur_map = env.observe()[1]["map_id"]
        assert cur_map == MAP_VIRIDIAN_CITY

        stub = StubPolicy(action_idx=0)  # action 0 = "down"
        tiny_budget = 4

        res = run_leg(
            env=env,
            leg="to_route2",
            target_map=MAP_ROUTE_2,
            policy_path="stub_policy_path",
            fallback_maps=[MAP_ROUTE_2],
            budget=tiny_budget,
            policy_net=stub,
            fallback_budget=150,
        )

        # Policy ran exactly up to tiny_budget presses and did not reach Route 2 (target is north)
        assert res["policy_presses"] == tiny_budget
        assert res["reached_by_policy"] is False

        # Fallback was invoked and completed the navigation
        assert res["fallback_presses"] > 0
        assert res["success"] is True
        assert env.observe()[1]["map_id"] == MAP_ROUTE_2


# ── Test 4: --nav bfs default unchanged ───────────────────────────────────────
def test_nav_bfs_default_unchanged():
    """Verify default navigation mode is 'bfs' across the skill functions."""
    import inspect
    sig_travel = inspect.signature(travel_to_pewter_skill)
    assert sig_travel.parameters["nav"].default == "bfs"

    sig_fight = inspect.signature(fight_gym_skill)
    assert sig_fight.parameters["nav"].default == "bfs"

    sig_quest = inspect.signature(run_gym_quest)
    assert sig_quest.parameters["nav"].default == "bfs"


@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_travel_skill_bfs_does_not_call_policy():
    """Verify travel_to_pewter_skill under nav='bfs' records no legs and runs purely BFS."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        legs_recorded: list[dict[str, Any]] = []
        # Run a short budget step towards Route 2
        res = travel_to_pewter_skill(
            env=env,
            budget=60,
            nav="bfs",
            run_legs=legs_recorded,
        )
        # In bfs mode, run_legs is never populated
        assert len(legs_recorded) == 0
        assert res["presses"] > 0


GATE_STATE = Path("work/route_states/seed100_map50.state")
ROUTE2_STATE = Path("work/route_states/seed100_map13.state")
TO_GATE_POLICY = Path("work/agent_rl/to_gate_wm31/policy.pt")


def test_planner_no_top_up_heal_past_viridian():
    """Past Viridian (Route 2, the gates, the forest) the top-up heal is skipped: navigate_to cannot route back to
    the Viridian Pokecenter from there, so the quest pressed one button per planner tick until its budget ran out
    (work/gym_learned_partial, seeds 100 and 106, stuck on map 50 at HP 35/38)."""
    for m in (MAP_ROUTE_2, MAP_VIRIDIAN_FOREST_SOUTH_GATE, MAP_VIRIDIAN_FOREST, MAP_VIRIDIAN_FOREST_NORTH_GATE):
        st = {"level": 14, "hp": 35, "max_hp": 38, "hp_frac": 35 / 38, "map_id": m}
        assert plan_gym(st, target_level=14) == "travel_to_pewter"
    st = {"level": 14, "hp": 35, "max_hp": 38, "hp_frac": 35 / 38, "map_id": MAP_VIRIDIAN_CITY}
    assert plan_gym(st, target_level=14) == "heal"


@pytest.mark.skipif(not ROM_PATH.exists() or not ROUTE2_STATE.exists() or not TO_GATE_POLICY.exists(),
                    reason="ROM, Route 2 state or to_gate policy not available")
def test_leg_settles_after_reaching_target():
    """A leg reached by the policy returns with the new map's coordinates loaded (RAM updates x, y a few frames
    after the map id), so the next leg's BFS fallback can plan. It used to fail at once with 0 presses."""
    with PokemonRed(state=ROUTE2_STATE) as env:
        res = run_leg(env, "to_gate", MAP_VIRIDIAN_FOREST_SOUTH_GATE, TO_GATE_POLICY, [MAP_VIRIDIAN_FOREST_SOUTH_GATE])
        assert res["reached_by_policy"] is True
        assert navigate_to(env, MAP_VIRIDIAN_FOREST, budget=500)["success"]


@pytest.mark.skipif(not ROM_PATH.exists() or not GATE_STATE.exists(), reason="ROM or gate state not available")
def test_learned_travel_resumes_at_current_leg():
    """From the forest's south gate, learned travel starts with the forest leg. It used to restart with the
    viridian leg, spending 1000 presses per call walking nowhere (work/gym_learned_partial, seed 101)."""
    missing = "work/agent_rl/missing/policy.pt"
    with PokemonRed(state=GATE_STATE) as env:
        legs: list[dict[str, Any]] = []
        res = travel_to_pewter_skill(env, budget=4500, nav="learned",
                                     leg_policies={leg: missing for leg in DEFAULT_LEG_POLICIES}, run_legs=legs)
        assert legs[0]["leg"] == "forest"
        assert res["success"]


@pytest.mark.parametrize("map_id,y,leg", [(MAP_ROUTE_2, 60, "to_gate"), (MAP_ROUTE_2, 5, "to_pewter"),
                                          (MAP_VIRIDIAN_FOREST, 30, "forest"), (MAP_VIRIDIAN_CITY, 10, "viridian")])
def test_first_travel_leg(map_id, y, leg):
    """Route 2 is two regions split by the forest: south of it the next leg is to_gate, north of it to_pewter."""
    assert TRAVEL_LEGS[first_travel_leg(map_id, y)][0] == leg


@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_leg_already_on_target_is_not_counted():
    """A leg that starts on its target map (e.g. after a heal trip) presses nothing and is not 'reached by policy'."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        assert env.observe()[1]["map_id"] == MAP_VIRIDIAN_CITY
        res = run_leg(env, "viridian", MAP_VIRIDIAN_CITY, "work/agent_rl/missing/policy.pt", [MAP_VIRIDIAN_CITY])
        assert res["reached_by_policy"] is None
        assert res["policy_presses"] == 0 and res["fallback_presses"] == 0
