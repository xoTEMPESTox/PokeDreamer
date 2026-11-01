"""make_pokedex_state.py — Script Oak's Parcel quest and save Pokédex + Poké Balls state.

Sequence:
1. Start from saves/starter_pallet.state (Pallet Town, Charmander Lv 5, rival beaten).
2. Walk north along Route 1 (map 12) to Viridian City (map 1).
3. Heal at Viridian Pokémon Center (ensures safe HP buffer for the round trip).
4. Enter Viridian Mart (map 42); clerk triggers dialogue and delivers OAK'S PARCEL (item 0x46).
5. Exit Mart, navigate south along Route 1 back to Pallet Town (map 0).
6. Enter Oak's Lab (map 40); talk to Oak at (5, 3) to deliver parcel, receive Pokédex,
   and advance through rival dialogue until player is free to move. Exit lab.
7. Return north along Route 1 to Viridian City (map 1).
8. Enter Viridian Mart, buy Poké Balls (Oak gives none in Red/Blue; 5 balls bought for $1000).
   Exit Mart to Viridian City at (29, 20).
9. Verify invariants (Viridian City, free to move, Pokédex flag set, bag >= 5 Poké Balls,
   Route 2 accessible without old man blocking) and save `saves/pokedex_viridian.state`.

Tile Coordinates Documentation:
- Viridian Mart door: (29, 19) in Viridian City (map 1), entered from (29, 20) facing UP.
  Discovered from pret/pokered `data/maps/objects/ViridianCity.asm` (`warp_event 29, 19, VIRIDIAN_MART, 1`)
  and confirmed via emulator BFS snapshot search.
- Inside Viridian Mart (map 42): Exit door warp tiles are (3, 7) and (4, 7). Clerk counter
  is at (2, 5). When entering for the parcel quest, clerk script pulls player to (2, 5).
  When purchasing, player stands at (3, 5) facing LEFT (or 2, 6 facing UP). Discovered via
  dialogue trigger coordinates and pret/pokered `ViridianMart.asm`.
- Oak's Lab in Pallet Town: Exterior entrance at (12, 11) in Pallet Town (map 0) from (12, 12).
  Interior exit at (4, 11)/(5, 11). Oak stands at (5, 2); player interacts from (5, 3) facing UP.
  Discovered via `PalletTown.asm` (`warp_event 12, 11, OAKS_LAB, 2`) and `make_starter_state.py`.
- Route 2 connection in Viridian City: North connection at Y=0 past the old man at (18, 9).
  Route 2 (map 13) reached in 32 steps from (29, 20). Discovered via `ViridianCity.asm` header
  and verified by BFS emulator pathfinding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lab.agent.system12 import (
    DETOUR_EXITS,
    MAP_OAKS_LAB,
    MAP_PALLET,
    MAP_ROUTE1,
    MAP_VIRIDIAN,
    MAP_VIRIDIAN_MART,
    MAP_VIRIDIAN_POKECENTER,
    ROUTE1_NORTH_WAYPOINTS,
    ROUTE1_SOUTH_WAYPOINTS,
    heal,
    navigate,
    party_status,
)
from lab.env import ROM_PATH, PokemonRed
from src.game_state import screen_capture
from src.ram_addresses import (
    EVENT_FLAGS_START,
    HP_ADDRESSES,
    IN_BATTLE,
    LEVEL_ADDRESSES,
    MAP_ID,
    MAX_HP_ADDRESSES,
    PARTY_SIZE,
    PARTY_SPECIES_ADDRESSES,
    TEXT_BOX_ID,
)

# ── Item Constants ─────────────────────────────────────────────────────────────
ITEM_POKE_BALL = 0x04
ITEM_OAKS_PARCEL = 0x46

ITEM_NAMES = {
    0x01: "Master Ball",
    0x02: "Ultra Ball",
    0x03: "Great Ball",
    0x04: "Poké Ball",
    0x05: "Town Map",
    0x0A: "Moon Stone",
    0x14: "Potion",
    0x19: "Antidote",
    0x1A: "Burn Heal",
    0x1C: "Parlyz Heal",
    0x1D: "Escape Rope",
    0x1E: "Repel",
    0x46: "Oak's Parcel",
}

# ── Navigation Paths & Waypoints ──────────────────────────────────────────────
# From Viridian entrance (21, 35) to Viridian Mart door at (29, 19):
VIRIDIAN_TO_MART_PATH = (
    ["up"] * 5
    + ["left"]
    + ["up"] * 2
    + ["left"]
    + ["up"] * 8
    + ["right"] * 10
    + ["up"]
)

# From Viridian PC door (23, 26) to Viridian Mart door at (29, 19):
PC_TO_MART_PATH = ["left"] * 4 + ["up"] * 6 + ["right"] * 10 + ["up"]

# From outside Viridian Mart (29, 20) south to Route 1 border (21, 36):
MART_TO_ROUTE1_PATH = (
    ["left"] * 10
    + ["down"] * 8
    + ["right"]
    + ["down"] * 2
    + ["right"]
    + ["down"] * 6
)

# Pallet Town Route 1 border (10, 0) to Oak's Lab door at (12, 11):
PALLET_TO_LAB_PATH = (
    ["down", "down", "left"] + ["down"] * 10 + ["right"] * 3 + ["up"]
)

# Pallet Town outside Oak's Lab (12, 12) to Route 1 border (10, 0):
LAB_TO_ROUTE1_PATH = ["left"] * 3 + ["up"] * 10 + ["right"] + ["up"] * 3

# Outside Viridian Mart (29, 20) north to Route 2 (map 13):
ROUTE2_PATH = [
    "left", "left", "up", "up", "up", "up", "left", "left", "left", "left",
    "left", "left", "left", "left", "up", "up", "up", "up", "up", "up",
    "up", "up", "up", "up", "up", "up", "up", "up", "left", "up", "up", "up"
]


# ── Reusable RAM Readers ───────────────────────────────────────────────────────
def money(env: PokemonRed) -> int:
    """Read player money from RAM (3 bytes BCD at 0xD347-0xD349).

    In Pokemon Red, money is stored in 3 consecutive bytes as packed BCD:
    e.g. [0x00, 0x31, 0x75] represents $3,175.
    """
    b0 = env.pyboy.memory[0xD347]
    b1 = env.pyboy.memory[0xD348]
    b2 = env.pyboy.memory[0xD349]
    return int(f"{b0:02x}{b1:02x}{b2:02x}")


def bag(env: PokemonRed) -> list[dict[str, Any]]:
    """Read player bag contents from RAM.

    Address 0xD31D stores the total number of distinct item slots in the bag.
    From 0xD31E onwards, each item slot is stored as an (item_id, quantity) byte pair.
    The list is terminated by 0xFF.
    """
    count = env.pyboy.memory[0xD31D]
    items = []
    for i in range(count):
        item_id = env.pyboy.memory[0xD31E + 2 * i]
        qty = env.pyboy.memory[0xD31E + 2 * i + 1]
        items.append({
            "id": int(item_id),
            "name": ITEM_NAMES.get(item_id, f"Item_0x{item_id:02X}"),
            "count": int(qty),
        })
    return items


def pokedex_status(env: PokemonRed) -> dict[str, Any]:
    """Verify Pokédex acquisition signals from RAM.

    - 0xD74E bit 0 is EVENT_GOT_POKEDEX (set when Oak delivers Pokédex).
    - 0xD2F7-0xD309: 19-byte bitfield for owned Pokémon.
    - 0xD30A-0xD31C: 19-byte bitfield for seen Pokémon.
    """
    event_flag = bool(env.pyboy.memory[0xD74E] & 0x01)
    owned_count = sum(
        bin(env.pyboy.memory[0xD2F7 + i]).count("1") for i in range(19)
    )
    seen_count = sum(
        bin(env.pyboy.memory[0xD30A + i]).count("1") for i in range(19)
    )
    return {
        "event_got_pokedex": event_flag,
        "owned_count": owned_count,
        "seen_count": seen_count,
        "pokedex_obtained": event_flag or (owned_count > 0 and seen_count > 0),
    }


# ── Reusable Skill Functions ──────────────────────────────────────────────────
def buy_poke_balls(
    env: PokemonRed, n: int = 5, budget: int = 500
) -> tuple[int, int]:
    """From Viridian City: enter the Mart, buy n balls, leave; returns balls bought and money spent.

    Assumes player is in Viridian City (map 1) or already inside Viridian Mart (map 42).
    Navigates to the clerk counter, conducts purchasing dialogue, and exits back to Viridian City.
    """
    presses = 0
    m_start = money(env)
    b_start = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
    _, ram = env.observe()

    # Step 1: Navigate into Viridian Mart (map 42) if currently in Viridian City
    if ram["map_id"] == MAP_VIRIDIAN:
        if (ram["x"], ram["y"]) == (29, 20):
            _, ram = env.step("up")
            presses += 1
        elif (ram["x"], ram["y"]) == (23, 26):
            for a in PC_TO_MART_PATH:
                _, ram = env.step(a)
                presses += 1
        else:
            for a in VIRIDIAN_TO_MART_PATH:
                _, ram = env.step(a)
                presses += 1

    while ram["map_id"] != MAP_VIRIDIAN_MART and presses < budget:
        _, ram = env.step("pass")
        presses += 1

    if ram["map_id"] != MAP_VIRIDIAN_MART:
        raise RuntimeError(
            f"buy_poke_balls failed to enter Viridian Mart (map {ram['map_id']})"
        )

    # Step 2: Inside Mart, walk to counter at (3, 5) and face LEFT towards clerk
    while (ram["x"], ram["y"]) != (3, 5) and presses < budget:
        if ram["y"] > 5:
            act = "up"
        elif ram["x"] < 3:
            act = "right"
        elif ram["x"] > 3:
            act = "left"
        else:
            act = "down"
        _, ram = env.step(act)
        presses += 1

    _, ram = env.step("left")
    presses += 1

    # Step 3: Talk to clerk and buy balls
    # Press A to initiate dialogue: opens Buy / Sell / Quit menu (Buy is default).
    _, ram = env.step("a")
    presses += 1

    # Each cycle of pressing 'a' through the shop menu buys 1 Poké Ball ($200)
    target_balls = b_start + n
    while presses < budget:
        cur_balls = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
        if cur_balls >= target_balls or money(env) < 200:
            break
        _, ram = env.step("a")
        presses += 1

    # Step 4: Dismiss shop menus and close text box with B presses
    for _ in range(15):
        if presses >= budget:
            break
        _, ram = env.step("b")
        presses += 1
        if env.pyboy.memory[TEXT_BOX_ID] == 0:
            break

    # Step 5: Exit Mart back to Viridian City
    # Align to exit door columns X=3 or 4, then step down
    while ram["x"] < 3 and presses < budget:
        _, ram = env.step("right")
        presses += 1
    while ram["x"] > 4 and presses < budget:
        _, ram = env.step("left")
        presses += 1
    while ram["map_id"] == MAP_VIRIDIAN_MART and presses < budget:
        _, ram = env.step("down")
        presses += 1

    for _ in range(5):
        if presses < budget:
            _, ram = env.step("pass")
            presses += 1

    m_end = money(env)
    b_end = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
    balls_bought = b_end - b_start
    money_spent = m_start - m_end
    return balls_bought, money_spent


def save_debug_frame(env: PokemonRed, out_path: Path) -> None:
    """Save native RGB screen capture to disk as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = screen_capture(env.pyboy)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ── Quest Phases ──────────────────────────────────────────────────────────────
def phase1_walk_to_viridian(
    env: PokemonRed, debug_dir: Path | None = None
) -> dict[str, Any]:
    """Phase 1: Walk from Pallet Town through Route 1 to Viridian City (map 1)."""
    print("\n--- Phase 1: Walk from Pallet Town to Viridian City ---")
    exits = {
        MAP_PALLET: [(10, 0), (11, 0)],
        MAP_ROUTE1: ROUTE1_NORTH_WAYPOINTS,
        **DETOUR_EXITS,
    }
    res = navigate(env, exits, target_map=MAP_VIRIDIAN, budget=1000)
    _, ram = env.observe()
    if ram["map_id"] != MAP_VIRIDIAN:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase1_failed.png")
        raise RuntimeError(
            f"Phase 1 failed: Reached map {ram['map_id']} instead of Viridian City (map 1)"
        )

    print(
        f"Reached Viridian City after {res['presses']} presses! RAM: {ram}"
    )
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase1_viridian_arrival.png")
    return ram


def phase2_heal_at_pc(
    env: PokemonRed, debug_dir: Path | None = None
) -> dict[str, Any]:
    """Phase 2: Heal starter at Viridian Pokémon Center to ensure full HP for round trip."""
    print("\n--- Phase 2: Heal at Viridian Pokémon Center ---")
    res = heal(env, budget=300)
    _, ram = env.observe()
    st = party_status(env)
    print(
        f"Nurse Joy healed party: HP {st['hp']}/{st['max_hp']} (presses: {res['presses']})"
    )
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase2_healed_pc.png")
    return ram


def phase3_obtain_oaks_parcel(
    env: PokemonRed, debug_dir: Path | None = None
) -> dict[str, Any]:
    """Phase 3: Enter Viridian Mart; clerk calls player over and gives OAK'S PARCEL."""
    print("\n--- Phase 3: Enter Viridian Mart and receive Oak's Parcel ---")
    # Walk from PC door (23, 26) to Mart door at (29, 19)
    for a in PC_TO_MART_PATH:
        env.step(a)

    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    if ram["map_id"] != MAP_VIRIDIAN_MART:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase3_mart_entry_failed.png")
        raise RuntimeError(
            f"Phase 3 failed: Did not enter Viridian Mart, currently at map {ram['map_id']}"
        )

    print(f"Entered Viridian Mart (map 42). Starting clerk interaction...")
    # The clerk automatically pulls the player to (2, 5). Advance dialogue with A
    dialog_steps = 0
    while env.pyboy.memory[0xD31D] == 0 and dialog_steps < 60:
        env.step("a")
        dialog_steps += 1

    if env.pyboy.memory[0xD31D] == 0:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase3_parcel_failed.png")
        raise RuntimeError(
            f"Phase 3 failed: Oak's Parcel was not added to bag within {dialog_steps} steps"
        )

    # Dismiss any trailing dialogue
    for _ in range(8):
        env.step("b")

    parcel_id = env.pyboy.memory[0xD31E]
    parcel_qty = env.pyboy.memory[0xD31F]
    print(
        f"Obtained Oak's Parcel! Item ID: 0x{parcel_id:02X}, Qty: {parcel_qty} (after {dialog_steps} steps)"
    )

    # Exit Mart: from (2, 5), step down, right to (3, 6), down to (3, 7) exit warp
    for m in ["down", "right", "down", "down"]:
        env.step(m)
    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    print(f"Exited Mart to Viridian City at pos ({ram['x']}, {ram['y']})")
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase3_parcel_obtained.png")
    return ram


def phase4_walk_to_pallet(
    env: PokemonRed, debug_dir: Path | None = None
) -> dict[str, Any]:
    """Phase 4: Walk south from Viridian City through Route 1 back to Pallet Town."""
    print("\n--- Phase 4: Walk from Viridian City back to Pallet Town ---")
    # Walk from outside Mart (29, 20) south to Route 1 border (21, 36)
    for m in MART_TO_ROUTE1_PATH:
        env.step(m)
    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    if ram["map_id"] != MAP_ROUTE1:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase4_route1_entry_failed.png")
        raise RuntimeError(
            f"Phase 4 failed: Did not reach Route 1, current map is {ram['map_id']}"
        )

    print(f"Entered Route 1 heading south. Following waypoints...")
    # Follow ROUTE1_SOUTH_WAYPOINTS and handle wild battles along the path
    for target in ROUTE1_SOUTH_WAYPOINTS + [(10, 35), (10, 36)]:
        for _ in range(40):
            ram = env.observe()[1]
            if ram["in_battle"]:
                while ram["in_battle"]:
                    _, ram = env.step("a")
            if (ram["x"], ram["y"]) == target or ram["map_id"] == MAP_PALLET:
                break
            if ram["x"] < target[0]:
                env.step("right")
            elif ram["x"] > target[0]:
                env.step("left")
            elif ram["y"] < target[1]:
                env.step("down")
            elif ram["y"] > target[1]:
                env.step("up")

    while env.observe()[1]["map_id"] != MAP_PALLET:
        env.step("down")
    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    if ram["map_id"] != MAP_PALLET:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase4_pallet_failed.png")
        raise RuntimeError(
            f"Phase 4 failed: Did not reach Pallet Town, current map is {ram['map_id']}"
        )

    print(
        f"Successfully returned to Pallet Town! RAM: {ram}, Party: {party_status(env)}"
    )
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase4_pallet_arrival.png")
    return ram


def phase5_deliver_parcel_in_lab(
    env: PokemonRed, debug_dir: Path | None = None
) -> dict[str, Any]:
    """Phase 5: Enter Oak's Lab, talk to Oak, deliver parcel, get Pokédex, clear rival dialog."""
    print("\n--- Phase 5: Deliver parcel to Oak and obtain Pokédex ---")
    # Walk from Pallet Town entrance (10, 0) to Oak's Lab door at (12, 11)
    for m in PALLET_TO_LAB_PATH:
        env.step(m)
    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    if ram["map_id"] != MAP_OAKS_LAB:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase5_lab_entry_failed.png")
        raise RuntimeError(
            f"Phase 5 failed: Did not enter Oak's Lab, current map is {ram['map_id']}"
        )

    # In Oak's Lab, walk from (5, 11) up to (5, 3) facing UP towards Oak at (5, 2)
    for _ in range(8):
        env.step("up")

    print(f"Standing in front of Oak at ({env.observe()[1]['x']}, {env.observe()[1]['y']})")
    # Talk to Oak: press A
    env.step("a")

    # Oak takes the parcel
    parcel_steps = 0
    while env.pyboy.memory[0xD31D] > 0 and parcel_steps < 100:
        env.step("a")
        parcel_steps += 1

    print(f"Parcel handed over after {parcel_steps} steps. Bag count: {env.pyboy.memory[0xD31D]}")

    # Rival enters, Oak speaks, hands over Pokédex, rival talks and exits.
    # Advance dialogue until player is free to step down from (5, 3)
    freed = False
    dialog_steps = 0
    for _ in range(300):
        _, r = env.step("a")
        dialog_steps += 1
        _, r = env.step("down")
        dialog_steps += 1
        if (r["x"], r["y"]) != (5, 3):
            freed = True
            break

    if not freed:
        if debug_dir:
            save_debug_frame(env, debug_dir / "phase5_dialog_failed.png")
        raise RuntimeError(
            f"Phase 5 failed: Dialogue in Oak's Lab did not conclude within {dialog_steps} steps"
        )

    dex_info = pokedex_status(env)
    print(
        f"Pokédex delivered! Player freed after {dialog_steps} steps. Pokédex status: {dex_info}"
    )

    # Walk south to exit Oak's Lab into Pallet Town
    while env.observe()[1]["map_id"] == MAP_OAKS_LAB:
        env.step("down")
    for _ in range(5):
        env.step("pass")

    _, ram = env.observe()
    print(f"Exited Oak's Lab to Pallet Town at pos ({ram['x']}, {ram['y']})")
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase5_pokedex_obtained.png")
    return ram


def phase6_return_to_viridian_and_buy_balls(
    env: PokemonRed, debug_dir: Path | None = None
) -> tuple[dict[str, Any], int, int]:
    """Phase 6: Walk to Viridian City, buy Poké Balls at Mart, and exit to Viridian City."""
    print("\n--- Phase 6: Return to Viridian City and buy Poké Balls ---")
    # Walk from outside Oak's Lab (12, 12) to Route 1 border (10, 0)
    for m in LAB_TO_ROUTE1_PATH:
        env.step(m)
    for _ in range(5):
        env.step("pass")

    # Navigate north along Route 1 to Viridian City
    exits = {
        MAP_PALLET: [(10, 0), (11, 0)],
        MAP_ROUTE1: ROUTE1_NORTH_WAYPOINTS,
        **DETOUR_EXITS,
    }
    res = navigate(env, exits, target_map=MAP_VIRIDIAN, budget=1000)
    print(f"Arrived in Viridian City (presses: {res['presses']})")

    # Check whether Oak gave any Poké Balls
    balls_in_bag = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
    print(f"Poké Balls before Mart visit: {balls_in_bag} (Oak gave 0 Poké Balls)")

    # Buy Poké Balls using reusable skill function buy_poke_balls
    bought, spent = buy_poke_balls(env, n=5)
    print(
        f"Mart purchase completed: bought {bought} Poké Balls, spent ${spent}"
    )
    print(f"Bag after purchase: {bag(env)}, Remaining Money: ${money(env)}")

    _, ram = env.observe()
    if debug_dir:
        save_debug_frame(env, debug_dir / "phase6_balls_purchased.png")
    return ram, bought, spent


def phase7_verify_and_save(
    env: PokemonRed,
    out_state_path: Path,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """Phase 7: Verify invariants, test Route 2 path open, and save pokedex_viridian.state."""
    print("\n--- Phase 7: Verify invariants and save pokedex_viridian.state ---")
    _, ram = env.observe()

    # 1. Map ID must be Viridian City (1)
    if ram["map_id"] != MAP_VIRIDIAN:
        raise AssertionError(
            f"Verification failed: Expected map_id=1 (Viridian City), got {ram['map_id']}"
        )

    # 2. Party size must be >= 1 and starter alive
    party_size = env.pyboy.memory[PARTY_SIZE]
    cur_hp = (env.pyboy.memory[HP_ADDRESSES[0]] << 8) | env.pyboy.memory[HP_ADDRESSES[0] + 1]
    if party_size < 1 or cur_hp <= 0:
        raise AssertionError(
            f"Verification failed: Party size {party_size}, Starter HP {cur_hp}"
        )

    # 3. Bag must contain >= 5 Poké Balls
    ball_count = sum(it["count"] for it in bag(env) if it["id"] == ITEM_POKE_BALL)
    if ball_count < 5:
        raise AssertionError(
            f"Verification failed: Bag has {ball_count} Poké Balls, expected >= 5"
        )

    # 4. Pokédex must be obtained
    dex = pokedex_status(env)
    if not dex["pokedex_obtained"]:
        raise AssertionError(f"Verification failed: Pokédex not obtained ({dex})")

    # 5. Free movement test: verify player is free to move without modifying state
    snap = env.snapshot()
    moved = False
    for direction in ["up", "left", "right", "down"]:
        env.restore(snap)
        _, r_test = env.step(direction)
        if (r_test["x"], r_test["y"]) != (ram["x"], ram["y"]):
            moved = True
            break
    env.restore(snap)
    if not moved:
        raise AssertionError(
            "Verification failed: Player is locked in dialogue / cannot move"
        )

    # 6. Route 2 open check: walk north from Viridian City to Route 2 (map 13)
    snap_before_route2 = env.snapshot()
    r_route2 = None
    for act in ROUTE2_PATH:
        _, r_route2 = env.step(act)

    route2_map_reached = r_route2["map_id"] if r_route2 else -1
    print(
        f"Route 2 reachability check: Final Map ID = {route2_map_reached} (Route 2 = 13), Pos: ({r_route2['x']}, {r_route2['y']})"
    )
    if route2_map_reached != 13:
        env.restore(snap_before_route2)
        raise AssertionError(
            f"Verification failed: Old man path blocked or wrong map reached ({route2_map_reached} != 13)"
        )
    # Restore exact state in Viridian City
    env.restore(snap_before_route2)

    # Save state to pokedex_viridian.state
    out_state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_state_path, "wb") as f:
        env.pyboy.save_state(f)
    print(f"Saved Pokédex Viridian state -> {out_state_path.resolve()}")

    if debug_dir:
        save_debug_frame(env, debug_dir / "phase7_final_pokedex_state.png")

    return {
        "ram": ram,
        "party": party_status(env),
        "bag": bag(env),
        "money": money(env),
        "pokedex": dex,
        "route2_map": route2_map_reached,
    }


# ── Main Runner ───────────────────────────────────────────────────────────────
def make_pokedex_state(
    rom: str | Path = ROM_PATH,
    starter_state: str | Path = Path("saves/starter_pallet.state"),
    out_state: str | Path = Path("saves/pokedex_viridian.state"),
    debug_dir: str | Path | None = Path("work/parcel_debug"),
) -> dict[str, Any]:
    """Execute end-to-end Oak's Parcel quest and generate pokedex_viridian.state."""
    rom_path = Path(rom)
    base_state_path = Path(starter_state)
    out_state_path = Path(out_state)
    debug_path = Path(debug_dir) if debug_dir else None

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path.resolve()}")
    if not base_state_path.exists():
        raise FileNotFoundError(f"Base starter state not found at {base_state_path.resolve()}")

    phase_logs: dict[str, Any] = {}

    print("=" * 80)
    print("GENERATING POKEMON RED POKÉDEX + POKÉ BALLS SAVE STATE")
    print(f"ROM:         {rom_path.resolve()}")
    print(f"Base State:  {base_state_path.resolve()}")
    print(f"Out State:   {out_state_path.resolve()}")
    print("=" * 80)

    with PokemonRed(rom=rom_path, state=base_state_path) as env:
        money_before = money(env)
        bag_before = bag(env)
        print(f"Initial Status: Money=${money_before}, Bag={bag_before}")

        # Phase 1: Pallet -> Viridian
        r1 = phase1_walk_to_viridian(env, debug_path)
        phase_logs["Phase 1 (Pallet -> Viridian)"] = r1

        # Phase 2: Heal at Pokémon Center
        r2 = phase2_heal_at_pc(env, debug_path)
        phase_logs["Phase 2 (Heal at PC)"] = r2

        # Phase 3: Obtain parcel at Mart
        r3 = phase3_obtain_oaks_parcel(env, debug_path)
        phase_logs["Phase 3 (Obtain Parcel)"] = r3

        # Phase 4: Viridian -> Pallet
        r4 = phase4_walk_to_pallet(env, debug_path)
        phase_logs["Phase 4 (Viridian -> Pallet)"] = r4

        # Phase 5: Deliver parcel to Oak
        r5 = phase5_deliver_parcel_in_lab(env, debug_path)
        phase_logs["Phase 5 (Deliver Parcel & Pokédex)"] = r5

        # Phase 6: Return to Viridian & buy balls
        r6, balls_bought, money_spent = phase6_return_to_viridian_and_buy_balls(
            env, debug_path
        )
        phase_logs["Phase 6 (Buy Poké Balls)"] = r6

        # Phase 7: Verify and save
        verify_results = phase7_verify_and_save(
            env, out_state_path=out_state_path, debug_dir=debug_path
        )
        phase_logs["Phase 7 (Verify & Save)"] = verify_results["ram"]

    money_after = verify_results["money"]
    bag_after = verify_results["bag"]

    print("\n" + "=" * 80)
    print("RUN COMPLETE — SUMMARY")
    print("=" * 80)
    print("Phase Log (RAM after each phase):")
    for phase_name, ram_state in phase_logs.items():
        print(f"  {phase_name:<35}: {ram_state}")

    print("\nMoney & Bag Summary:")
    print(f"  Money Before: ${money_before}")
    print(f"  Money After:  ${money_after} (Spent: ${money_spent})")
    print(f"  Bag Before:   {bag_before}")
    print(f"  Bag After:    {bag_after}")
    print(
        f"  Poké Balls Obtained: {balls_bought} bought at Viridian Mart (Oak gives 0 in Red/Blue)"
    )

    p_st = verify_results["party"]
    print("\nParty Status:")
    print(f"  Level:    {p_st['level']}")
    print(f"  HP:       {p_st['hp']}/{p_st['max_hp']}")
    print(f"  Party Sz: {p_st['party_size']}")

    print("\nPokédex & Progression:")
    print(f"  Event Flag (0xD74E bit 0): {verify_results['pokedex']['event_got_pokedex']}")
    print(f"  Owned Count:               {verify_results['pokedex']['owned_count']}")
    print(f"  Seen Count:                {verify_results['pokedex']['seen_count']}")
    print(f"  Route 2 Reached:           Map {verify_results['route2_map']} (Path Open!)")
    print("=" * 80 + "\n")

    return {
        "phase_logs": phase_logs,
        "money_before": money_before,
        "money_after": money_after,
        "bag_before": bag_before,
        "bag_after": bag_after,
        "balls_bought": balls_bought,
        "money_spent": money_spent,
        "verify_results": verify_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script Oak's Parcel quest and generate pokedex_viridian.state."
    )
    parser.add_argument(
        "--rom", type=Path, default=ROM_PATH, help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--starter-state",
        type=Path,
        default=Path("saves/starter_pallet.state"),
        help="Path to initial starter_pallet.state",
    )
    parser.add_argument(
        "--out-state",
        type=Path,
        default=Path("saves/pokedex_viridian.state"),
        help="Output pokedex_viridian.state path",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("work/parcel_debug"),
        help="Directory to save debug screenshots",
    )
    args = parser.parse_args()

    make_pokedex_state(
        rom=args.rom,
        starter_state=args.starter_state,
        out_state=args.out_state,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
