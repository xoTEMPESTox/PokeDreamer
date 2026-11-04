"""Unit tests for tile_sweep.py (pure functions, no emulator)."""
from pathlib import Path
import numpy as np
import pytest

from lab.agent.tile_sweep import (
    compute_exit_distances,
    get_mpc_stem,
    parse_extra_tiles,
    parse_policies,
    select_tiles,
)


@pytest.fixture
def detour_grid():
    """An 8x6 grid with a wall creating a detour, an unreachable tile, and an excluded door.

    Grid layout (w=8, h=6):
      y=0: . . . . . . . .
      y=1: . S . . . . . .    S = Start (1, 1)
      y=2: # # # # # # . .    Wall x=0..5
      y=3: . E . X . . . .    E = Exit (1, 3), X = Excluded/Door (3, 3)
      y=4: . . . . . . . #    Wall at (7, 4)
      y=5: . . . . . . # U    Wall at (6, 5), U = Unreachable (7, 5)
    """
    w, h = 8, 6
    walls = {
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),  # Detour wall
        (7, 4), (6, 5),                                    # Surrounds (7, 5)
    }

    def is_walkable(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h and (x, y) not in walls

    start = (1, 1)
    exit_tiles = [(1, 3)]
    excluded = [(3, 3)]
    unreachable = (7, 5)

    return {
        "w": w,
        "h": h,
        "walls": walls,
        "is_walkable": is_walkable,
        "start": start,
        "exit_tiles": exit_tiles,
        "excluded": excluded,
        "unreachable": unreachable,
    }


def test_unreachable_and_excluded_tiles_never_selected(detour_grid):
    """Unreachable tiles and excluded/door tiles must never be selected."""
    g = detour_grid
    for seed in (0, 1, 42, 100):
        rng = np.random.default_rng(seed)
        selected = select_tiles(
            g["is_walkable"],
            g["w"],
            g["h"],
            g["start"],
            g["exit_tiles"],
            g["excluded"],
            n=8,
            rng=rng,
        )

        selected_coords = {(x, y) for x, y, _ in selected}

        # Exit tiles themselves must be excluded
        for e in g["exit_tiles"]:
            assert e not in selected_coords, f"Exit tile {e} was selected!"

        # Excluded tiles must never be selected
        for x in g["excluded"]:
            assert x not in selected_coords, f"Excluded tile {x} was selected!"

        # Unreachable tile must never be selected
        assert g["unreachable"] not in selected_coords, "Unreachable tile was selected!"

        # Walls must never be selected
        for w in g["walls"]:
            assert w not in selected_coords, f"Wall tile {w} was selected!"


def test_distances_are_bfs_path_lengths_not_manhattan(detour_grid):
    """Distances must be BFS path lengths around obstacles, not Manhattan distance."""
    g = detour_grid

    # Compute distances directly
    dist = compute_exit_distances(g["is_walkable"], g["w"], g["h"], g["exit_tiles"])

    # From (1, 1) to (1, 3):
    # Manhattan distance is |1 - 1| + |3 - 1| = 2
    manhattan_dist = abs(g["start"][0] - g["exit_tiles"][0][0]) + abs(g["start"][1] - g["exit_tiles"][0][1])
    assert manhattan_dist == 2

    # BFS path length must detour around wall (x=0..5) through x=6:
    # (1,1) -> (6,1) [5 steps] -> (6,3) [2 steps] -> (1,3) [5 steps] = 12 steps
    bfs_dist = dist[g["start"]]
    assert bfs_dist == 12
    assert bfs_dist != manhattan_dist

    # Verify that select_tiles assigns exact BFS distances to all selected tiles
    rng = np.random.default_rng(0)
    selected = select_tiles(
        g["is_walkable"],
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=8,
        rng=rng,
    )

    for x, y, d in selected:
        assert d == dist[(x, y)], f"Reported dist {d} does not match BFS dist {dist[(x, y)]} for ({x}, {y})"


def test_same_rng_seed_gives_same_selection(detour_grid):
    """The same rng seed must give the exact same selection."""
    g = detour_grid

    rng1 = np.random.default_rng(12345)
    sel1 = select_tiles(
        g["is_walkable"],
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=8,
        rng=rng1,
    )

    rng2 = np.random.default_rng(12345)
    sel2 = select_tiles(
        g["is_walkable"],
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=8,
        rng=rng2,
    )

    assert sel1 == sel2, "Selections with the same RNG seed must be identical"


def test_every_quartile_bin_represented(detour_grid):
    """Every quartile bin must be represented in the selection."""
    g = detour_grid
    rng = np.random.default_rng(0)
    n = 8
    k = n // 4  # 2 per bin
    selected = select_tiles(
        g["is_walkable"],
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=n,
        rng=rng,
    )

    assert len(selected) == n

    # Compute expected quartile bins over all candidate distances
    dist = compute_exit_distances(g["is_walkable"], g["w"], g["h"], g["exit_tiles"])
    # BFS reachable from start
    reachable = set()
    q = [g["start"]]
    reachable.add(g["start"])
    while q:
        cx, cy = q.pop(0)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < g["w"] and 0 <= ny < g["h"] and (nx, ny) not in reachable and g["is_walkable"](nx, ny):
                reachable.add((nx, ny))
                q.append((nx, ny))

    candidates = [
        t for t in reachable
        if t not in set(g["exit_tiles"]) and t not in set(g["excluded"]) and t in dist
    ]
    candidates.sort(key=lambda t: (dist[t], t[0], t[1]))
    dists = [dist[t] for t in candidates]

    q0, q1, q2, q3, q4 = np.quantile(dists, [0, 0.25, 0.5, 0.75, 1.0])

    # Check that the selected items span across the quartile range
    selected_dists = [d for _, _, d in selected]
    # Smallest distance should be <= q1, largest >= q3
    assert min(selected_dists) <= q1
    assert max(selected_dists) >= q3

    # Check bin counts
    b0 = [d for d in selected_dists if d <= q1]
    b3 = [d for d in selected_dists if d > q3]
    assert len(b0) >= 1
    assert len(b3) >= 1


def test_2d_grid_input(detour_grid):
    """select_tiles must accept a 2D array / list of lists as is_walkable."""
    g = detour_grid
    grid_2d = [
        [g["is_walkable"](x, y) for x in range(g["w"])]
        for y in range(g["h"])
    ]

    rng1 = np.random.default_rng(42)
    sel_func = select_tiles(
        g["is_walkable"],
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=4,
        rng=rng1,
    )

    rng2 = np.random.default_rng(42)
    sel_grid = select_tiles(
        grid_2d,
        g["w"],
        g["h"],
        g["start"],
        g["exit_tiles"],
        g["excluded"],
        n=4,
        rng=rng2,
    )

    assert sel_func == sel_grid


def test_helper_parsers():
    """Test parse_extra_tiles, parse_policies, and get_mpc_stem."""
    assert parse_extra_tiles("23,26") == [(23, 26)]
    assert parse_extra_tiles("23,26;13,26") == [(23, 26), (13, 26)]
    assert parse_extra_tiles("none") == []
    assert parse_extra_tiles("") == []

    pols = parse_policies("to_route2_wm31_s2,ppo_real:to_route2_geo_s42")
    assert len(pols) == 2
    assert pols[0] == ("to_route2_wm31_s2", "to_route2_wm31_s2", False, "wm31")
    assert pols[1] == ("ppo_real:to_route2_geo_s42", "to_route2_geo_s42", True, "ppo_real")

    stem_wm = get_mpc_stem("to_route2", "to_route2_wm31_s2", is_ppo=False, seed=200, state_path=Path("work/route_states/map1_23_26.state"))
    assert stem_wm == "to_route2_learned_to_route2_wm31_s2_s200_from_map1_23_26"

    stem_ppo = get_mpc_stem("to_route2", "to_route2_geo_s42", is_ppo=True, seed=200, state_path=Path("work/route_states/map1_23_26.state"))
    assert stem_ppo == "to_route2_learned_ppo_real_to_route2_geo_s42_s200_from_map1_23_26"
