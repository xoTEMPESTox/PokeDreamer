"""
controller.py — External Controller (System 1 driver).

The controller owns the emulator loop.  It:
  1. Receives a Goal from the Planner
  2. Runs the frozen PPO (or random) step by step
  3. Monitors distance to target / stuck condition
  4. Returns a GoalResult when done (success, stuck, or timeout)

The PPO NEVER sees the goal — it only sees the raw screen, exactly as
during training.  The goal is used purely by the controller's monitoring logic.

Action mapping (PWhiddy 6-action):
  0=DOWN, 1=LEFT, 2=RIGHT, 3=UP, 4=A, 5=B
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
from pyboy import PyBoy

from src.game_state import GameState, extract_game_state, screen_capture
from src.goal import Goal, GoalResult, GoalType

if TYPE_CHECKING:
    from src.frozen_ppo import FrozenPPO


# PWhiddy's 8-action set
BUTTONS = ["down", "left", "right", "up", "a", "b", "start", "pass"]


@dataclass
class ControllerConfig:
    """Tunable parameters for the controller loop."""

    # How many emulator ticks to advance per action (24 ≈ 0.4s game-time)
    ticks_per_action: int = 24

    # Goal-completion: consider arrived when Manhattan distance ≤ this
    arrival_threshold: int = 1

    # Stuck detection: mark stuck if best_distance doesn't improve for K steps
    stuck_window: int = 40          # steps to look back
    stuck_distance_threshold: int = 2  # must improve by at least this many tiles

    # Hard timeout per goal (steps)
    max_steps_per_goal: int = 2000

    # Logging
    log_every_n_steps: int = 50
    verbose: bool = True
    deterministic: bool = True


@dataclass
class StepRecord:
    """One row in the controller's per-step log."""
    step: int
    action: int
    action_name: str
    map_id: int
    x: int
    y: int
    dist_to_goal: Optional[float]
    in_battle: bool
    dialog_open: bool
    badges: int


class Controller:
    """
    Drives the frozen PPO toward a Goal using RAM-state monitoring.

    Parameters
    ----------
    pyboy : PyBoy
        Running emulator instance (caller owns lifecycle).
    policy : FrozenPPO | None
        Frozen PPO wrapper.  If None, uses random actions (for testing).
    config : ControllerConfig
        Tunable hyperparameters.
    log_file : Path | None
        If set, appends JSONL step records to this file.
    """

    def __init__(
        self,
        pyboy: PyBoy,
        policy: Optional["FrozenPPO"] = None,
        config: Optional[ControllerConfig] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        self.pyboy = pyboy
        self.policy = policy
        self.cfg = config or ControllerConfig()
        self.log_file = log_file

        self._log_handle = None
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(log_file, "a", buffering=1)

        # Reset policy frame buffer at construction
        if self.policy is not None:
            self.policy.reset()

    # ── Core loop ─────────────────────────────────────────────────────────────

    def run(self, goal: Goal) -> GoalResult:
        """
        Execute a single goal.

        Runs until: arrival, stuck, timeout, or battle/dialog interruption
        (caller can re-issue a new goal after inspecting the result).
        """
        if self.cfg.verbose:
            print(f"\n[Controller] ▶ {goal}" + (f"  reason='{goal.reason}'" if goal.reason else ""))

        # Reset policy frame stack for fresh episode-like context
        if self.policy is not None:
            self.policy.reset()

        step_records: list[StepRecord] = []
        best_dist: float = float("inf")
        steps_since_improvement = 0
        t_start = time.perf_counter()

        for step in range(self.cfg.max_steps_per_goal):

            # ── 1. Capture obs + state ────────────────────────────────────────
            frame = screen_capture(self.pyboy)
            state = extract_game_state(self.pyboy)

            # ── 2. Compute distance to target ─────────────────────────────────
            dist = self._distance(state, goal)

            # ── 3. Check arrival ──────────────────────────────────────────────
            if dist is not None and dist <= self.cfg.arrival_threshold:
                result = GoalResult(
                    goal=goal,
                    status=GoalResult.Status.SUCCESS,
                    steps_taken=step,
                    final_x=state.x,
                    final_y=state.y,
                    final_map_id=state.map_id,
                )
                self._log_result(result)
                return result

            # ── 4. Choose action ──────────────────────────────────────────────
            if self.policy is not None:
                action = self.policy.predict(frame, deterministic=self.cfg.deterministic)
            else:
                import random
                action = random.randint(0, 5)  # 6-action space (no START)

            # ── 5. Step emulator ──────────────────────────────────────────────
            btn = BUTTONS[action]
            if btn != "pass":
                self.pyboy.button(btn, 8)
            self.pyboy.tick(self.cfg.ticks_per_action, render=True)

            # ── 6. Re-read state after tick ───────────────────────────────────
            state = extract_game_state(self.pyboy)
            dist  = self._distance(state, goal)

            # ── 7. Stuck detection ────────────────────────────────────────────
            if dist is not None:
                if dist < best_dist - self.cfg.stuck_distance_threshold:
                    best_dist = dist
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1  # no target = treat as stuck

            # ── 8. Record step ────────────────────────────────────────────────
            record = StepRecord(
                step=step,
                action=action,
                action_name=BUTTONS[action],
                map_id=state.map_id,
                x=state.x,
                y=state.y,
                dist_to_goal=dist,
                in_battle=state.in_battle,
                dialog_open=state.dialog_open,
                badges=state.badge_count,
            )
            step_records.append(record)
            self._write_step(record)

            # ── 9. Logging ────────────────────────────────────────────────────
            if self.cfg.verbose and step % self.cfg.log_every_n_steps == 0:
                elapsed = time.perf_counter() - t_start
                dist_str = f"{dist:.1f}" if dist is not None else "—"
                stuck_str = f"stuck={steps_since_improvement}/{self.cfg.stuck_window}"
                print(
                    f"  step {step:>4}  {state}  "
                    f"dist={dist_str}  {stuck_str}  [{elapsed:.1f}s]"
                )

            # ── 10. Stuck trigger ─────────────────────────────────────────────
            if steps_since_improvement >= self.cfg.stuck_window:
                result = GoalResult(
                    goal=goal,
                    status=GoalResult.Status.STUCK,
                    steps_taken=step,
                    final_x=state.x,
                    final_y=state.y,
                    final_map_id=state.map_id,
                    notes=f"No improvement for {steps_since_improvement} steps (best_dist={best_dist:.1f})",
                )
                self._log_result(result)
                return result

        # ── Timeout ───────────────────────────────────────────────────────────
        state = extract_game_state(self.pyboy)
        result = GoalResult(
            goal=goal,
            status=GoalResult.Status.TIMEOUT,
            steps_taken=self.cfg.max_steps_per_goal,
            final_x=state.x,
            final_y=state.y,
            final_map_id=state.map_id,
            notes=f"Timeout after {self.cfg.max_steps_per_goal} steps",
        )
        self._log_result(result)
        return result

    # ── Distance metric ───────────────────────────────────────────────────────

    def _distance(self, state: GameState, goal: Goal) -> Optional[float]:
        """
        Manhattan distance to the goal's target tile.

        Returns None if:
          - goal has no target position (e.g. ENTER_BUILDING)
          - goal specifies a different map_id and player hasn't arrived yet
        """
        if goal.type == GoalType.WAIT:
            return None

        if goal.x is None or goal.y is None:
            return None

        # If goal specifies a map and we're on the wrong map, return large dist
        if goal.map_id is not None and state.map_id != goal.map_id:
            return float("inf")

        return abs(state.x - goal.x) + abs(state.y - goal.y)

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _write_step(self, record: StepRecord) -> None:
        if self._log_handle is None:
            return
        row = {
            "step":        record.step,
            "action":      record.action,
            "action_name": record.action_name,
            "map_id":      record.map_id,
            "x":           record.x,
            "y":           record.y,
            "dist":        record.dist_to_goal,
            "in_battle":   record.in_battle,
            "dialog":      record.dialog_open,
            "badges":      record.badges,
        }
        self._log_handle.write(json.dumps(row) + "\n")

    def _log_result(self, result: GoalResult) -> None:
        if self.cfg.verbose:
            icon = "[OK]" if result.succeeded else ("[STUCK]" if result.status == GoalResult.Status.STUCK else "[TIMEOUT]")
            print(f"\n[Controller] {icon} {result}")

    def close(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *_) -> None:
        self.close()
