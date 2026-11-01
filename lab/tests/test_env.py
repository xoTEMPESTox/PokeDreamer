"""Tests for Pokemon Red PyBoy environment wrapper."""
from pathlib import Path

import numpy as np
import pytest

from lab.env import ROM_PATH, PokemonRed, make_states

BEDROOM_STATE = Path("saves/bedroom.state")
ROM_EXISTS = ROM_PATH.exists()


@pytest.fixture(scope="module", autouse=True)
def ensure_save_states():
    if ROM_EXISTS and not BEDROOM_STATE.exists():
        make_states(ROM_PATH, Path("saves"))


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
def test_fresh_env_bedroom_state():
    """A fresh PokemonRed(state='saves/bedroom.state') returns (72, 80) uint8 and map 38."""
    with PokemonRed(state=BEDROOM_STATE) as env:
        frame, ram = env.observe()

        assert isinstance(frame, np.ndarray), "Frame must be a numpy array"
        assert frame.shape == (72, 80), f"Expected shape (72, 80), got {frame.shape}"
        assert frame.dtype == np.uint8, f"Expected dtype uint8, got {frame.dtype}"

        assert isinstance(ram, dict), "RAM state must be a dict"
        assert ram["map_id"] == 38, f"Expected map_id 38, got {ram['map_id']}"
        assert ram["x"] == 3, f"Expected x=3, got {ram['x']}"
        assert ram["y"] == 6, f"Expected y=6, got {ram['y']}"
        assert ram["in_battle"] is False, "Expected in_battle False"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
def test_snapshot_restore_determinism():
    """snapshot/restore gives identical frames after the same action sequence (determinism)."""
    with PokemonRed(state=BEDROOM_STATE) as env:
        snap = env.snapshot()
        assert isinstance(snap, bytes) and len(snap) > 0, "Snapshot must be non-empty bytes"

        # Action sequence: right, up, left, down, a, pass
        actions = [2, 3, 1, 0, 4, 7]

        # Branch 1
        frames1 = []
        for act in actions:
            f, _ = env.step(act)
            frames1.append(f)

        # Restore
        env.restore(snap)

        # Branch 2
        frames2 = []
        for act in actions:
            f, _ = env.step(act)
            frames2.append(f)

        # Assert identical
        for i, (f1, f2) in enumerate(zip(frames1, frames2)):
            assert np.array_equal(f1, f2), f"Discrepancy at action step {i}"


@pytest.mark.skipif(not ROM_EXISTS, reason="ROM file not present")
def test_movement_right_changes_x():
    """One 'right' press from (3,6) changes x by 1."""
    with PokemonRed(state=BEDROOM_STATE) as env:
        _, ram0 = env.observe()
        assert (ram0["x"], ram0["y"]) == (3, 6), f"Initial position must be (3, 6), got ({ram0['x']}, {ram0['y']})"

        _, ram1 = env.step(2)  # 2 = right
        assert ram1["x"] == 4, f"Expected x=4 after right, got {ram1['x']}"
        assert ram1["y"] == 6, f"Expected y=6 after right, got {ram1['y']}"
