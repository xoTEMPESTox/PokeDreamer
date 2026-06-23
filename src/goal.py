"""
Goal schema for the hierarchical agent.

Goals are emitted by the VLM Planner (System 2) and consumed by the
external Controller (which drives the frozen PPO).

Keep this schema simple and extensible — new goal types are added here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GoalType(str, Enum):
    GOTO          = "goto"           # Move to (map_id, x, y)
    ENTER_BUILDING = "enter_building" # Walk through a building entrance
    TALK_TO_NPC   = "talk_to_npc"   # Engage dialogue with an NPC
    ENGAGE_BATTLE  = "engage_battle"  # Intentionally start a battle
    USE_ITEM      = "use_item"       # Use an item from the bag
    WAIT          = "wait"           # Hold position for K steps (dialogue / cutscene)


@dataclass
class Goal:
    """
    A single goal issued by the Planner to the Controller.

    Only fields relevant to the specific GoalType need to be set;
    all others default to None.
    """
    type: GoalType

    # ── GOTO fields ───────────────────────────────────────────────────────────
    map_id: Optional[int] = None   # Target map (None = same map as current)
    x: Optional[int]      = None   # Target tile X
    y: Optional[int]      = None   # Target tile Y

    # ── TALK_TO_NPC fields ───────────────────────────────────────────────────
    npc_id: Optional[str] = None   # Human-readable NPC identifier (for logging)

    # ── USE_ITEM fields ───────────────────────────────────────────────────────
    item_name: Optional[str] = None

    # ── Metadata (set by Planner, logged by Controller) ───────────────────────
    reason: str = ""               # Why this goal was issued (for logging/debug)
    priority: int = 0              # Higher = more urgent (reserved for future use)

    def __str__(self) -> str:
        if self.type == GoalType.GOTO:
            map_str = f"map={self.map_id}" if self.map_id is not None else "same_map"
            return f"GOTO({map_str}, x={self.x}, y={self.y})"
        if self.type == GoalType.ENTER_BUILDING:
            return "ENTER_BUILDING"
        if self.type == GoalType.TALK_TO_NPC:
            return f"TALK_TO_NPC({self.npc_id or '?'})"
        if self.type == GoalType.ENGAGE_BATTLE:
            return "ENGAGE_BATTLE"
        if self.type == GoalType.USE_ITEM:
            return f"USE_ITEM({self.item_name or '?'})"
        if self.type == GoalType.WAIT:
            return "WAIT"
        return f"Goal({self.type})"

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "map_id": self.map_id,
            "x": self.x,
            "y": self.y,
            "npc_id": self.npc_id,
            "item_name": self.item_name,
            "reason": self.reason,
        }

    @classmethod
    def goto(cls, x: int, y: int, map_id: Optional[int] = None, reason: str = "") -> "Goal":
        """Convenience constructor for the most common goal type."""
        return cls(type=GoalType.GOTO, x=x, y=y, map_id=map_id, reason=reason)

    @classmethod
    def enter_building(cls, reason: str = "") -> "Goal":
        return cls(type=GoalType.ENTER_BUILDING, reason=reason)

    @classmethod
    def talk_to_npc(cls, npc_id: str = "", reason: str = "") -> "Goal":
        return cls(type=GoalType.TALK_TO_NPC, npc_id=npc_id, reason=reason)

    @classmethod
    def wait(cls, reason: str = "") -> "Goal":
        return cls(type=GoalType.WAIT, reason=reason)


@dataclass
class GoalResult:
    """Outcome of a goal execution reported by the Controller."""

    class Status(str, Enum):
        SUCCESS  = "success"
        STUCK    = "stuck"     # No progress for K steps
        TIMEOUT  = "timeout"   # Max steps exceeded
        ABORTED  = "aborted"   # Externally cancelled

    goal: Goal
    status: Status
    steps_taken: int = 0
    final_x: Optional[int] = None
    final_y: Optional[int] = None
    final_map_id: Optional[int] = None
    notes: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == self.Status.SUCCESS

    def __str__(self) -> str:
        pos = f"({self.final_x},{self.final_y})@map{self.final_map_id}"
        return (
            f"GoalResult[{self.status.value}] {self.goal} "
            f"→ {pos} in {self.steps_taken} steps"
            + (f" | {self.notes}" if self.notes else "")
        )
