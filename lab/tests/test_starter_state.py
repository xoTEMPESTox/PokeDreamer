"""Tests for Pokemon Red starter save states."""
from pathlib import Path

import numpy as np
import pytest

from lab.env import ROM_PATH, PokemonRed
from src.ram_addresses import HP_ADDRESSES, IN_BATTLE, PARTY_SIZE

STARTER_PALLET_STATE = Path("saves/starter_pallet.state")
STARTER_LAB_STATE = Path("saves/starter_lab.state")
ROM_EXISTS = ROM_PATH.exists()
PALLET_EXISTS = STARTER_PALLET_STATE.exists()


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not PALLET_EXISTS, reason="saves/starter_pallet.state not present")
def test_starter_pallet_state_invariants():
    """Verify saves/starter_pallet.state satisfies map 0, party size 1, not in battle."""
    with PokemonRed(state=STARTER_PALLET_STATE) as env:
        frame, ram = env.observe()

        assert isinstance(frame, np.ndarray), "Frame must be a numpy array"
        assert frame.shape == (72, 80), f"Expected shape (72, 80), got {frame.shape}"
        assert frame.dtype == np.uint8, f"Expected dtype uint8, got {frame.dtype}"

        assert ram["map_id"] == 0, f"Expected map_id 0 (Pallet Town), got {ram['map_id']}"
        assert ram["in_battle"] is False, "Expected in_battle False"
        assert env.pyboy.memory[IN_BATTLE] == 0, "Expected IN_BATTLE RAM byte == 0"

        party_size = env.pyboy.memory[PARTY_SIZE]
        assert party_size == 1, f"Expected party size 1, got {party_size}"

        # Current HP must be positive
        cur_hp = (env.pyboy.memory[HP_ADDRESSES[0]] << 8) | env.pyboy.memory[HP_ADDRESSES[0] + 1]
        assert cur_hp > 0, f"Starter HP must be > 0, got {cur_hp}"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not PALLET_EXISTS, reason="saves/starter_pallet.state not present")
def test_starter_pallet_state_movement():
    """Verify player is free to walk: one 'up' press changes y coordinate or some direction moves player."""
    with PokemonRed(state=STARTER_PALLET_STATE) as env:
        _, ram0 = env.observe()
        initial_pos = (ram0["x"], ram0["y"])

        # Check movement: one 'up' press changes y coordinate, or another direction moves player
        _, ram_up = env.step("up")
        if ram_up["y"] != ram0["y"]:
            assert True, f"Y coordinate changed from {ram0['y']} to {ram_up['y']}"
        else:
            # Check other directions
            moved = False
            for act in ["left", "right", "down"]:
                with PokemonRed(state=STARTER_PALLET_STATE) as env2:
                    _, ram_dir = env2.step(act)
                    if (ram_dir["x"], ram_dir["y"]) != initial_pos:
                        moved = True
                        break
            assert moved, f"Player is trapped at {initial_pos}; no tested direction changed position"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not STARTER_LAB_STATE.exists(), reason="saves/starter_lab.state not present")
def test_starter_lab_state_invariants():
    """Verify saves/starter_lab.state satisfies map 40, party size 1, not in battle."""
    with PokemonRed(state=STARTER_LAB_STATE) as env:
        _, ram = env.observe()

        assert ram["map_id"] == 40, f"Expected map_id 40 (Oak's Lab), got {ram['map_id']}"
        assert ram["in_battle"] is False, "Expected in_battle False"
        assert env.pyboy.memory[PARTY_SIZE] == 1, f"Expected party size 1, got {env.pyboy.memory[PARTY_SIZE]}"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
@pytest.mark.skipif(not PALLET_EXISTS, reason="saves/starter_pallet.state not present")
def test_route1_reachable_without_oak():
    """Verify Route 1 (map 12) is reachable from starter_pallet.state without Oak stopping player."""
    with PokemonRed(state=STARTER_PALLET_STATE) as env:
        path = ["left"] * 3 + ["up"] * 10 + ["right"] + ["up"] * 3
        r = None
        for move in path:
            _, r = env.step(move)

        assert r is not None
        assert r["map_id"] == 12, f"Expected to reach Route 1 (map 12), reached map {r['map_id']} at ({r['x']}, {r['y']})"
        assert r["in_battle"] is False, "Expected not in battle on Route 1"
