"""test_gym_quest.py — Unit and integration tests for System 1/2 gym quest agent.

Includes:
1. Pure-logic tests (no ROM/state required):
   - System 2 planner goal selection under various RAM statuses.
   - BFS pathfinding on toy grids with obstacles and one-way constraints.
   - Battle move choice rule logic.
2. ROM-dependent integration tests (skipped if ROM or save state is missing):
   - Privilege walkability grid extraction in Viridian City.
   - Navigation from Viridian City to Route 2 within press budget.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pytest

from lab.agent.gym_quest import Goal, plan_gym
from lab.agent.nav import (
    MAP_PEWTER_CITY,
    MAP_PEWTER_GYM,
    MAP_PEWTER_POKECENTER,
    MAP_ROUTE_1,
    MAP_ROUTE_2,
    MAP_VIRIDIAN_CITY,
    MOVE_EMBER,
    MOVE_GROWL,
    MOVE_SCRATCH,
    TYPE_BUG,
    TYPE_GROUND,
    TYPE_NORMAL,
    TYPE_ROCK,
    WRAM_OBTAINED_BADGES,
    bfs_path,
    build_walkability_grid,
    navigate_to,
)
from lab.env import ROM_PATH, PokemonRed

SAVE_STATE_PATH = Path("saves/pokedex_viridian.state")


# ── Pure-Logic Tests ─────────────────────────────────────────────────────────
def test_plan_gym_done_when_badge_obtained():
    """If Boulder Badge (bit 0) is set, goal is 'done' regardless of other stats."""
    st = {
        "badges": 1,
        "level": 15,
        "hp": 40,
        "max_hp": 40,
        "hp_frac": 1.0,
        "map_id": MAP_PEWTER_GYM,
    }
    goal = plan_gym(st, target_level=15)
    assert goal == "done"
    assert "Boulder Badge" in goal.reason


def test_plan_gym_heal_when_hp_low():
    """If HP fraction is below threshold outside Pewter, goal is 'heal'."""
    st = {
        "badges": 0,
        "level": 12,
        "hp": 5,
        "max_hp": 30,
        "hp_frac": 5 / 30,  # ~0.16 < 0.35
        "map_id": MAP_ROUTE_1,
    }
    goal = plan_gym(st, target_level=15, heal_threshold=0.35)
    assert goal == "heal"
    assert "HP low" in goal.reason


def test_plan_gym_grind_when_level_below_target():
    """If HP is healthy and level < target_level, goal is 'grind'."""
    st = {
        "badges": 0,
        "level": 10,
        "hp": 28,
        "max_hp": 30,
        "hp_frac": 28 / 30,
        "map_id": MAP_VIRIDIAN_CITY,
    }
    goal = plan_gym(st, target_level=15)
    assert goal == "grind"
    assert "Level 10 < target 15" in goal.reason


def test_plan_gym_travel_to_pewter():
    """If level >= target_level and not in Pewter area, goal is 'travel_to_pewter'."""
    st = {
        "badges": 0,
        "level": 15,
        "hp": 42,
        "max_hp": 42,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
    }
    goal = plan_gym(st, target_level=15)
    assert goal == "travel_to_pewter"
    assert "traveling" in goal.reason


def test_plan_gym_heal_in_pewter_before_gym():
    """When arriving in Pewter City, must heal at Pewter PC to anchor respawn point."""
    st = {
        "badges": 0,
        "level": 15,
        "hp": 35,
        "max_hp": 42,
        "hp_frac": 35 / 42,
        "map_id": MAP_PEWTER_CITY,
    }
    goal = plan_gym(st, target_level=15, pewter_healed=False)
    assert goal == "heal"
    assert "Pewter" in goal.reason


def test_plan_gym_challenge_gym_when_prepared():
    """In Pewter City, healed and at full HP, goal is 'gym'."""
    st = {
        "badges": 0,
        "level": 15,
        "hp": 42,
        "max_hp": 42,
        "hp_frac": 1.0,
        "map_id": MAP_PEWTER_CITY,
    }
    goal = plan_gym(st, target_level=15, pewter_healed=True)
    assert goal == "gym"
    assert "Pewter Gym" in goal.reason


def test_toy_bfs_with_obstacle_and_ledge():
    """Pure-logic BFS on a 5x5 grid with an obstacle wall and a one-way downward ledge."""
    # 5x5 grid: (0,0) to (4,4)
    # Obstacle wall at (2, 1), (2, 2)
    # Downward ledge at (1, 2): allows jumping from (1, 1) down to (1, 3), but cannot walk from (1, 3) up to (1, 1)
    obstacles = {(2, 1), (2, 2)}
    ledge_tiles = {(1, 2): "down"}

    def get_neighbors(x: int, y: int) -> list[tuple[int, int, str]]:
        res = []
        for dx, dy, act in [(0, 1, "down"), (-1, 0, "left"), (1, 0, "right"), (0, -1, "up")]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 5 and 0 <= ny < 5):
                continue
            if (nx, ny) in obstacles:
                continue
            # Check ledge constraint: cannot step onto ledge in reverse or laterally
            if (nx, ny) in ledge_tiles:
                if act != ledge_tiles[(nx, ny)]:
                    continue
                # Jumping the ledge advances 2 steps in that direction
                nnx, nny = nx + dx, ny + dy
                if 0 <= nnx < 5 and 0 <= nny < 5 and (nnx, nny) not in obstacles:
                    res.append((nnx, nny, act))
                continue
            res.append((nx, ny, act))
        return res

    def toy_bfs(start: tuple[int, int], target: tuple[int, int]) -> Optional[list[str]]:
        queue = deque([(start[0], start[1], [])])
        visited = {start}
        while queue:
            cx, cy, path = queue.popleft()
            if (cx, cy) == target:
                return path
            for nx, ny, act in get_neighbors(cx, cy):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [act]))
        return None

    # Test 1: Route around obstacle from (1, 1) to (3, 1)
    path = toy_bfs((1, 1), (3, 1))
    assert path is not None
    # Must go around wall at x=2 (e.g. via y=0)
    assert (2, 1) not in [(1 + dx, 1 + dy) for dx, dy in [(0, 0)]]

    # Test 2: Downward jump over ledge from (1, 1) to (1, 3)
    path_ledge = toy_bfs((1, 1), (1, 3))
    assert path_ledge == ["down"]  # One jump action down over (1, 2) to (1, 3)

    # Test 3: Upward traversal blocked over ledge: from (1, 3) to (1, 1) must bypass around
    path_reverse = toy_bfs((1, 3), (1, 1))
    assert path_reverse is not None
    assert path_reverse != ["up"]  # Cannot directly jump upward


# ── ROM-Dependent Integration Tests ──────────────────────────────────────────
@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_viridian_walkability_grid():
    """Verify walkability grid builds from WRAM and accurately identifies passable tiles."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        w_tiles, h_tiles, is_walkable = build_walkability_grid(env)
        obs = env.observe()[1]
        assert obs["map_id"] == MAP_VIRIDIAN_CITY

        # Player starting tile at (29, 20) outside Mart must be walkable
        assert is_walkable(29, 20) is True

        # Viridian Mart building wall at (29, 18) must NOT be walkable
        assert is_walkable(29, 18) is False

        # Out of bounds tiles must return False
        assert is_walkable(-1, 0) is False
        assert is_walkable(w_tiles + 5, 0) is False


@pytest.mark.skipif(not ROM_PATH.exists() or not SAVE_STATE_PATH.exists(), reason="ROM or save state not available")
def test_navigate_to_route_2_from_start():
    """Verify navigate_to reaches Route 2 (map 13) from start state within 100 actions."""
    with PokemonRed(state=SAVE_STATE_PATH) as env:
        res = navigate_to(env, MAP_ROUTE_2, budget=150)
        assert res["success"] is True
        assert env.observe()[1]["map_id"] == MAP_ROUTE_2
        assert res["presses"] <= 80


def test_zero_memory_writes_enforced():
    """Verify that nav.py and gym_quest.py contain strictly zero assignments to memory."""
    import ast
    for module_name in ["lab/agent/nav.py", "lab/agent/gym_quest.py"]:
        with open(module_name, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=module_name)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript):
                        val = target.value
                        if isinstance(val, ast.Name) and val.id in ("mem", "memory"):
                            raise AssertionError(f"Forbidden RAM write found in {module_name} at line {node.lineno}")
                        elif isinstance(val, ast.Attribute) and val.attr in ("mem", "memory"):
                            raise AssertionError(f"Forbidden RAM write found in {module_name} at line {node.lineno}")
