"""Tests for System 1 / System 2 agent (system12.py)."""
from pathlib import Path
import pytest

from lab.agent.system12 import DETOUR_EXITS, Goal, navigate, party_status, plan
from lab.agent.tasks import _PALLET_DETOURS
from lab.env import ROM_PATH, PokemonRed

STARTER_PALLET_STATE = Path("saves/starter_pallet.state")
ROM_EXISTS = ROM_PATH.exists()
PALLET_EXISTS = STARTER_PALLET_STATE.exists()


def test_plan_rules():
    """Verify System 2 v0 planner rules on hand-made status dictionaries."""
    # 1. Low HP triggers "heal" regardless of level
    status_low_hp = {"hp": 6, "max_hp": 20, "hp_frac": 0.30, "level": 6}
    goal = plan(status_low_hp, target_level=10)
    assert goal == "heal"
    assert "HP low" in goal.reason
    g, reason = plan(status_low_hp, target_level=10)
    assert g == "heal" and isinstance(reason, str)

    # 2. Borderline low HP (< 0.35)
    status_edge_low = {"hp": 34, "max_hp": 100, "hp_frac": 0.34, "level": 9}
    assert plan(status_edge_low, target_level=10) == "heal"

    # 3. Healthy HP (>= 0.35) and level < target_level triggers "grind"
    status_grind = {"hp": 20, "max_hp": 20, "hp_frac": 1.0, "level": 6}
    goal_grind = plan(status_grind, target_level=10)
    assert goal_grind == "grind"
    assert "Level 6 < target 10" in goal_grind.reason

    # 4. Target level reached triggers "done"
    status_done = {"hp": 25, "max_hp": 25, "hp_frac": 1.0, "level": 10}
    goal_done = plan(status_done, target_level=10)
    assert goal_done == "done"
    assert "Target level 10 reached" in goal_done.reason

    # 5. Above target level triggers "done"
    status_over = {"hp": 10, "max_hp": 30, "hp_frac": 0.33, "level": 11}
    # Note: low HP takes priority over done if hp_frac < 0.35
    assert plan(status_over, target_level=10) == "heal"
    status_over_healthy = {"hp": 30, "max_hp": 30, "hp_frac": 1.0, "level": 11}
    assert plan(status_over_healthy, target_level=10) == "done"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not PALLET_EXISTS, reason="saves/starter_pallet.state not present")
def test_party_status():
    """Verify party_status on the starter state returns level >= 5, party_size 1, hp > 0."""
    with PokemonRed(state=STARTER_PALLET_STATE) as env:
        st = party_status(env)

        assert isinstance(st, dict)
        assert st["party_size"] == 1, f"Expected party_size 1, got {st['party_size']}"
        assert st["level"] >= 5, f"Expected starter level >= 5, got {st['level']}"
        assert st["hp"] > 0, f"Expected hp > 0, got {st['hp']}"
        assert st["max_hp"] >= st["hp"], f"Expected max_hp >= hp, got {st['max_hp']} vs {st['hp']}"
        assert 0.0 < st["hp_frac"] <= 1.0, f"Expected hp_frac in (0, 1], got {st['hp_frac']}"
        assert st["map_id"] == 0, f"Expected starter map_id 0 (Pallet Town), got {st['map_id']}"
        assert st["map"] == 0
        assert not st["in_battle"], "Expected not in battle at start"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not PALLET_EXISTS, reason="saves/starter_pallet.state not present")
def test_short_navigate():
    """Verify a short navigate from the starter state to Route 1 succeeds within 200 presses."""
    with PokemonRed(state=STARTER_PALLET_STATE) as env:
        exits = {
            0: [(10, 0), (11, 0)],
            **_PALLET_DETOURS,
        }
        res = navigate(env, exits=exits, target_map=12, budget=200)

        assert res["success"] is True, f"Navigate failed: {res}"
        assert res["presses"] <= 200, f"Navigate exceeded budget: {res['presses']} presses"
        _, ram = env.observe()
        assert ram["map_id"] == 12, f"Expected map_id 12 (Route 1), got {ram['map_id']}"
