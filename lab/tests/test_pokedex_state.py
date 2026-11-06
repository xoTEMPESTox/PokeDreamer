"""Tests for Pokemon Red Pokédex + Poké Balls save state."""
from pathlib import Path

import numpy as np
import pytest

from lab.env import ROM_PATH, PokemonRed
from lab.make_pokedex_state import (
    ITEM_OAKS_PARCEL,
    ITEM_POKE_BALL,
    ROUTE2_PATH,
    bag,
    money,
    pokedex_status,
)
from lab.game.ram_addresses import HP_ADDRESSES, IN_BATTLE, PARTY_SIZE

POKEDEX_VIRIDIAN_STATE = Path("saves/pokedex_viridian.state")
ROM_EXISTS = ROM_PATH.exists()
STATE_EXISTS = POKEDEX_VIRIDIAN_STATE.exists()


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_pokedex_viridian_state_invariants():
    """Verify pokedex_viridian.state has map 1 (Viridian), party >= 1, not in battle."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        frame, ram = env.observe()

        assert isinstance(frame, np.ndarray), "Frame must be a numpy array"
        assert frame.shape == (72, 80), f"Expected shape (72, 80), got {frame.shape}"
        assert frame.dtype == np.uint8, f"Expected dtype uint8, got {frame.dtype}"

        # Map must be Viridian City (map 1)
        assert ram["map_id"] == 1, f"Expected map_id 1 (Viridian City), got {ram['map_id']}"
        assert ram["in_battle"] is False, "Expected in_battle False"
        assert env.pyboy.memory[IN_BATTLE] == 0, "Expected IN_BATTLE RAM byte == 0"

        # Party size must be >= 1 and starter alive
        party_sz = env.pyboy.memory[PARTY_SIZE]
        assert party_sz >= 1, f"Expected party size >= 1, got {party_sz}"

        cur_hp = (env.pyboy.memory[HP_ADDRESSES[0]] << 8) | env.pyboy.memory[HP_ADDRESSES[0] + 1]
        assert cur_hp > 0, f"Starter HP must be > 0, got {cur_hp}"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_bag_has_poke_balls_and_pokedex():
    """Verify bag has >= 5 Poké Balls, Oak's Parcel is removed, and Pokédex flag is set."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        items = bag(env)
        ball_count = sum(it["count"] for it in items if it["id"] == ITEM_POKE_BALL)
        assert ball_count >= 5, f"Expected >= 5 Poké Balls in bag, found {ball_count}"

        # Oak's Parcel must not be in bag anymore
        parcel_present = any(it["id"] == ITEM_OAKS_PARCEL for it in items)
        assert not parcel_present, "Oak's Parcel should have been delivered to Oak"

        # Pokédex status
        dex = pokedex_status(env)
        assert dex["pokedex_obtained"], f"Pokédex should be obtained ({dex})"
        assert dex["event_got_pokedex"], "EVENT_GOT_POKEDEX event flag must be set"
        assert dex["owned_count"] >= 1, "Owned Pokémon count must be >= 1"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_player_free_movement():
    """Verify player is free to walk without dialogue locks."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        _, ram0 = env.observe()
        initial_pos = (ram0["x"], ram0["y"])

        moved = False
        for act in ["left", "right", "up", "down"]:
            with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env2:
                _, ram_dir = env2.step(act)
                if (ram_dir["x"], ram_dir["y"]) != initial_pos:
                    moved = True
                    break
        assert moved, f"Player is trapped at {initial_pos}; no tested direction changed position"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_route2_reachable_within_150_presses():
    """Verify Route 2 (map 13) is reachable within 150 presses (old man unblocked)."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        presses = 0
        r = None
        for move in ROUTE2_PATH:
            _, r = env.step(move)
            presses += 1
            if r["map_id"] == 13:
                break

        assert r is not None, "No steps executed"
        assert presses <= 150, f"Expected Route 2 within 150 presses, took {presses}"
        assert r["map_id"] == 13, f"Expected to reach Route 2 (map 13), reached map {r['map_id']} at ({r['x']}, {r['y']})"
        assert r["in_battle"] is False, "Expected not in battle upon entering Route 2"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STATE_EXISTS, reason="saves/pokedex_viridian.state not present")
def test_money_and_bag_readers():
    """Verify money() and bag() RAM readers return valid values."""
    with PokemonRed(state=POKEDEX_VIRIDIAN_STATE) as env:
        m = money(env)
        assert isinstance(m, int)
        assert m > 0, f"Expected positive money, got {m}"

        b = bag(env)
        assert isinstance(b, list)
        assert len(b) >= 1
        assert all("id" in item and "count" in item for item in b)
