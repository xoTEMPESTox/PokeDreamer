"""Tests for System 1 / System 2 catching agent (catch_quest.py)."""
from pathlib import Path
import pytest

from lab.agent.catch_quest import plan_catch
from lab.agent.system12 import ROUTE1_SOUTH_WAYPOINTS
from lab.env import ROM_PATH, PokemonRed
from lab.make_pokedex_state import (
    ITEM_POKE_BALL,
    MART_TO_ROUTE1_PATH,
    bag,
    money,
    pokedex_status,
)
from src.ram_addresses import IN_BATTLE, PARTY_SIZE

POKEDEX_VIRIDIAN_STATE = Path("saves/pokedex_viridian.state")
ROM_EXISTS = ROM_PATH.exists()
STATE_EXISTS = POKEDEX_VIRIDIAN_STATE.exists()


def test_plan_catch_rules():
    """Verify System 2 v0 catch planner rules on hand-crafted status dictionaries."""
    # 1. Low HP (< 0.5) triggers "heal" regardless of ball count or catches
    st_low_hp = {"hp": 10, "max_hp": 25, "hp_frac": 0.40, "balls": 5, "money": 2000, "catches": 0}
    goal = plan_catch(st_low_hp, catch_target=10)
    assert goal == "heal"
    assert "HP low" in goal.reason
    g, reason = plan_catch(st_low_hp, catch_target=10)
    assert g == "heal" and isinstance(reason, str)

    # Low HP with 0 balls still triggers "heal" first
    st_low_hp_no_balls = {"hp": 11, "max_hp": 25, "hp_frac": 0.44, "balls": 0, "money": 500, "catches": 2}
    assert plan_catch(st_low_hp_no_balls, catch_target=10) == "heal"

    # 2. Healthy HP (>= 0.5), balls == 0, and money >= 200 triggers "buy"
    st_buy = {"hp": 20, "max_hp": 25, "hp_frac": 0.80, "balls": 0, "money": 600, "catches": 3}
    goal_buy = plan_catch(st_buy, catch_target=10)
    assert goal_buy == "buy"
    assert "Out of balls" in goal_buy.reason
    assert "money $600 >= 200" in goal_buy.reason

    # Boundary at exactly $200 money triggers "buy"
    st_buy_edge = {"hp": 25, "max_hp": 25, "hp_frac": 1.0, "balls": 0, "money": 200, "catches": 3}
    assert plan_catch(st_buy_edge, catch_target=10) == "buy"

    # 3. Healthy HP (>= 0.5), balls == 0, and money < 200 triggers "done: out of money"
    st_no_money = {"hp": 25, "max_hp": 25, "hp_frac": 1.0, "balls": 0, "money": 175, "catches": 4}
    goal_no_money = plan_catch(st_no_money, catch_target=10)
    assert goal_no_money == "done: out of money"
    assert "cannot afford more" in goal_no_money.reason

    # 4. Healthy HP, balls > 0, catches < catch_target triggers "catch"
    st_catch = {"hp": 25, "max_hp": 25, "hp_frac": 1.0, "balls": 5, "money": 2175, "catches": 0}
    goal_catch = plan_catch(st_catch, catch_target=10)
    assert goal_catch == "catch"
    assert "Catches 0 < target 10" in goal_catch.reason

    # Catches partially complete
    st_catch_mid = {"hp": 15, "max_hp": 25, "hp_frac": 0.60, "balls": 3, "money": 1000, "catches": 9}
    assert plan_catch(st_catch_mid, catch_target=10) == "catch"

    # 5. Catch target reached triggers "done"
    st_done = {"hp": 20, "max_hp": 25, "hp_frac": 0.80, "balls": 2, "money": 800, "catches": 10}
    goal_done = plan_catch(st_done, catch_target=10)
    assert goal_done == "done"
    assert "Target catches 10 reached" in goal_done.reason

    # Over target
    st_over = {"hp": 25, "max_hp": 25, "hp_frac": 1.0, "balls": 4, "money": 800, "catches": 12}
    assert plan_catch(st_over, catch_target=10) == "done"

    # 6. Healing latch in Viridian / Pokémon Center
    st_healing = {"hp": 20, "max_hp": 25, "hp_frac": 0.80, "balls": 0, "money": 500, "catches": 1, "map_id": 41}
    assert plan_catch(st_healing, catch_target=10, current_goal="heal") == "heal"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_initial_state_bag_has_5_poke_balls():
    """Verify from pokedex_viridian.state the bag contains 5 Poké Balls, money is 2175, owned count is 1."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        items = bag(env)
        ball_count = sum(it["count"] for it in items if it["id"] == ITEM_POKE_BALL)
        assert ball_count == 5, f"Expected 5 Poké Balls in initial bag, got {ball_count}"

        m = money(env)
        assert m == 2175, f"Expected $2,175 money, got {m}"

        dex = pokedex_status(env)
        assert dex["owned_count"] == 1, f"Expected 1 owned Pokémon (Charmander), got {dex['owned_count']}"
        assert dex["pokedex_obtained"] is True


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_scripted_wild_battle_entered_within_300_presses():
    """Verify that from pokedex_viridian.state, walking to the grass patch triggers a wild battle within 300 presses."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        snap_before = env.snapshot()
        presses = 0

        # Step south from Viridian Mart to Route 1 border
        for a in MART_TO_ROUTE1_PATH:
            _, ram = env.step(a)
            presses += 1

        # Follow south waypoints towards the grass patch
        for target in ROUTE1_SOUTH_WAYPOINTS:
            while (ram["x"], ram["y"]) != target and presses < 300:
                if ram["in_battle"]:
                    break
                if ram["x"] < target[0]: a = "right"
                elif ram["x"] > target[0]: a = "left"
                elif ram["y"] > target[1]: a = "up"
                elif ram["y"] < target[1]: a = "down"
                _, ram = env.step(a)
                presses += 1
            if ram["in_battle"]:
                break

        # Pace in the tall grass patch until battle triggers
        while not ram["in_battle"] and presses < 300:
            act = "left" if ram["x"] >= 14 else "right"
            _, ram = env.step(act)
            presses += 1

        assert ram["in_battle"] is True, f"Failed to enter wild battle within 300 presses (used {presses})"
        assert presses <= 300, f"Exceeded 300 presses: used {presses}"
        assert env.pyboy.memory[IN_BATTLE] == 1, "Expected wild battle flag (1) in RAM"

        # Restore state after test
        env.restore(snap_before)
        _, ram_restored = env.observe()
        assert ram_restored["map_id"] == 1, "Expected restored map 1 (Viridian City)"
        assert (ram_restored["x"], ram_restored["y"]) == (29, 20), "Expected restored pos (29, 20)"
        assert not ram_restored["in_battle"], "Expected not in battle after restore"
