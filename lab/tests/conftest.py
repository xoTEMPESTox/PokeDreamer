"""Pytest configuration and automatic marker assignment for PokeDreamer v3.

Automatically marks and skips tests requiring external resources (ROM, route
save states, dataset cache, or model weights) when those resources are not present
on the machine, or when simulated via POKEDREAMER_CI=1.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from lab.env import ROM_PATH

# CI simulation flag
CI_MODE = os.environ.get("POKEDREAMER_CI", "").strip().lower() in ("1", "true", "yes")

# Resource checks
ROM_FILE_EXISTS = (not CI_MODE) and ROM_PATH.exists()
ROUTE_STATES_DIR = Path("work/route_states")
ROUTE_STATES_EXIST = (not CI_MODE) and ROUTE_STATES_DIR.is_dir() and any(ROUTE_STATES_DIR.glob("*.state"))
ROM_AVAILABLE = ROM_FILE_EXISTS and ROUTE_STATES_EXIST

CACHE_BASE = Path("work/cache_72x80.npz")
CACHE_EXT = Path("work/cache_ext_72x80.npz")
DATA_AVAILABLE = (not CI_MODE) and CACHE_BASE.exists() and CACHE_EXT.exists()

WEIGHTS_AVAILABLE = (not CI_MODE) and any(Path("work").glob("**/*.pt"))

# Mapping from test module filename to required resource markers
MODULE_REQUIREMENTS: dict[str, list[str]] = {
    # Modules requiring ROM and/or route save states
    "test_env.py": ["requires_rom"],
    "test_starter_state.py": ["requires_rom"],
    "test_pokedex_state.py": ["requires_rom"],
    "test_route_tasks.py": ["requires_rom"],
    "test_system12.py": ["requires_rom"],
    "test_catch_quest.py": ["requires_rom"],
    "test_collect.py": ["requires_rom"],
    "test_gym_quest.py": ["requires_rom"],
    "test_ppo_real.py": ["requires_rom"],
    "test_learned_nav.py": ["requires_rom", "requires_weights"],
    "test_move_tasks.py": ["requires_rom", "requires_data", "requires_weights"],
    # Modules requiring dataset cache (work/cache_*)
    "test_eval_rollouts.py": ["requires_data"],
    "test_show_tools.py": ["requires_data"],
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_rom: test module requires Pokemon Red ROM or emulator save states")
    config.addinivalue_line("markers", "requires_data: test module requires dataset cache in work/cache_*")
    config.addinivalue_line("markers", "requires_weights: test module requires model weights/checkpoints in work/")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        # Determine module file name
        mod_name = Path(item.fspath).name if hasattr(item, "fspath") else Path(item.path).name
        reqs = MODULE_REQUIREMENTS.get(mod_name, [])

        for req in reqs:
            item.add_marker(getattr(pytest.mark, req))

        # Check if any required resource is missing and apply skip
        if "requires_rom" in reqs and not ROM_AVAILABLE:
            reason = "Pokemon Red ROM or route save states not available"
            if CI_MODE:
                reason += " (POKEDREAMER_CI=1 set)"
            item.add_marker(pytest.mark.skip(reason=reason))
        elif "requires_data" in reqs and not DATA_AVAILABLE:
            reason = "Dataset cache (work/cache_*.npz) not available"
            if CI_MODE:
                reason += " (POKEDREAMER_CI=1 set)"
            item.add_marker(pytest.mark.skip(reason=reason))
        elif "requires_weights" in reqs and not WEIGHTS_AVAILABLE:
            reason = "Model checkpoints/weights under work/ not available"
            if CI_MODE:
                reason += " (POKEDREAMER_CI=1 set)"
            item.add_marker(pytest.mark.skip(reason=reason))
