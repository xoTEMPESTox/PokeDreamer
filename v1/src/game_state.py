"""
GameState dataclass and extractor.

Reads all relevant RAM fields from a running PyBoy instance in a single call,
and simultaneously captures the current screen frame.  Both are derived from
the same emulator tick, guaranteeing synchronisation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyboy import PyBoy

from src.ram_addresses import (
    PLAYER_X, PLAYER_Y, MAP_ID,
    PLAYER_FACING, IN_BATTLE, TEXT_BOX_ID,
    BADGE_FLAGS, PARTY_SIZE,
    HP_ADDRESSES, MAX_HP_ADDRESSES,
    MAP_NAMES, FACING_NAMES,
)


@dataclass
class GameState:
    """Snapshot of all relevant game state at a single emulator tick."""

    # ── Position ──────────────────────────────────────────────────────────────
    map_id: int          # Raw map ID byte
    x: int               # Player tile X
    y: int               # Player tile Y
    facing: int          # 0=down, 4=up, 8=left, 12=right

    # ── Status flags ─────────────────────────────────────────────────────────
    in_battle: bool      # True when a battle is active
    dialog_open: bool    # True when a text/dialog box is on screen

    # ── Party ─────────────────────────────────────────────────────────────────
    party_hp: list[int]      = field(default_factory=list)  # current HP per slot
    party_max_hp: list[int]  = field(default_factory=list)  # max HP per slot

    # ── Progression ──────────────────────────────────────────────────────────
    badges: int = 0      # raw bitmask; bin(badges).count('1') → badge count

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def badge_count(self) -> int:
        return bin(self.badges).count("1")

    @property
    def map_name(self) -> str:
        return MAP_NAMES.get(self.map_id, f"Unknown(0x{self.map_id:02X})")

    @property
    def facing_name(self) -> str:
        return FACING_NAMES.get(self.facing, f"Unknown({self.facing})")

    @property
    def total_hp(self) -> int:
        return sum(self.party_hp)

    @property
    def total_max_hp(self) -> int:
        return sum(self.party_max_hp)

    def to_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "map_name": self.map_name,
            "x": self.x,
            "y": self.y,
            "facing": self.facing_name,
            "in_battle": self.in_battle,
            "dialog_open": self.dialog_open,
            "badges": self.badge_count,
            "party_hp": self.party_hp,
            "party_max_hp": self.party_max_hp,
        }

    def __str__(self) -> str:
        hp_str = "/".join(
            f"{h}/{m}" for h, m in zip(self.party_hp, self.party_max_hp)
        ) or "—"
        flags = []
        if self.in_battle:
            flags.append("BATTLE")
        if self.dialog_open:
            flags.append("DIALOG")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return (
            f"[{self.map_name}] ({self.x}, {self.y}) "
            f"facing={self.facing_name} "
            f"badges={self.badge_count} "
            f"HP={hp_str}"
            f"{flag_str}"
        )


def _read_u16_big(pyboy: "PyBoy", hi_addr: int, lo_addr: int) -> int:
    """Read a 2-byte big-endian unsigned int from two separate RAM addresses."""
    return (pyboy.memory[hi_addr] << 8) | pyboy.memory[lo_addr]


def extract_game_state(pyboy: "PyBoy") -> GameState:
    """
    Read all RAM fields and return a GameState snapshot.

    This must be called *after* pyboy.tick() so the memory reflects
    the result of the last action.
    """
    mem = pyboy.memory

    # ── Basic position / map ──────────────────────────────────────────────────
    map_id  = mem[MAP_ID]
    x       = mem[PLAYER_X]
    y       = mem[PLAYER_Y]
    facing  = mem[PLAYER_FACING]

    # ── Flags ─────────────────────────────────────────────────────────────────
    in_battle   = mem[IN_BATTLE] != 0
    dialog_open = mem[TEXT_BOX_ID] != 0

    # ── Badges ────────────────────────────────────────────────────────────────
    badges = mem[BADGE_FLAGS]

    # ── Party HP ──────────────────────────────────────────────────────────────
    n_pokemon = min(mem[PARTY_SIZE], 6)
    party_hp: list[int] = []
    party_max_hp: list[int] = []

    for slot in range(n_pokemon):
        # Each HP is stored as 2 bytes: hi byte at the listed address, lo byte at +1
        cur_hp  = (mem[HP_ADDRESSES[slot]] << 8)  | mem[HP_ADDRESSES[slot] + 1]
        max_hp  = (mem[MAX_HP_ADDRESSES[slot]] << 8) | mem[MAX_HP_ADDRESSES[slot] + 1]
        party_hp.append(cur_hp)
        party_max_hp.append(max_hp)

    return GameState(
        map_id=map_id,
        x=x,
        y=y,
        facing=facing,
        in_battle=in_battle,
        dialog_open=dialog_open,
        party_hp=party_hp,
        party_max_hp=party_max_hp,
        badges=badges,
    )


def screen_capture(pyboy: "PyBoy") -> np.ndarray:
    """
    Return the current Game Boy screen as a (144, 160, 3) uint8 RGB array.

    Called *after* the same tick as extract_game_state() so the frame
    and RAM state are perfectly synchronised.
    """
    # PyBoy ≥2.0: pyboy.screen.image returns a PIL Image
    img = pyboy.screen.image          # PIL Image (160×144 RGBA or RGB)
    arr = np.array(img)               # (144, 160, 4) or (144, 160, 3)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]           # drop alpha channel
    return arr.astype(np.uint8)
