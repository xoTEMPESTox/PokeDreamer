"""llm_planner.py — LLM System 2 planner for the PokeDreamer gym quest.

Interacts locally with Ollama (e.g. gemma4:26b) via standard library urllib.request
to make high-level quest decisions with schema-constrained JSON outputs and thinking.
Falls back to the System 2 rule planner (plan_gym) on any hallucination or malformed response.
"""
from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from lab.agent.gym_quest import (
    Goal,
    MAP_PEWTER_CITY,
    MAP_PEWTER_GYM,
    MAP_PEWTER_POKECENTER,
    MAP_ROUTE_1,
    MAP_ROUTE_2,
    MAP_VIRIDIAN_CITY,
    MAP_VIRIDIAN_FOREST,
    MAP_VIRIDIAN_FOREST_NORTH_GATE,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE,
    MAP_VIRIDIAN_POKECENTER,
    plan_gym,
)

# ── Map Name Mapping ─────────────────────────────────────────────────────────
MAP_NAMES: dict[int, str] = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    12: "Route 1",
    13: "Route 2",
    37: "Red's House 1F",
    38: "Red's House 2F",
    39: "Rival's House",
    40: "Oak's Lab",
    41: "Viridian PokeCenter",
    42: "Viridian Mart",
    47: "Viridian Forest North Gate",
    50: "Viridian Forest South Gate",
    51: "Viridian Forest",
    54: "Pewter Gym",
    58: "Pewter PokeCenter",
}

PEWTER_MAP_IDS: set[int] = {MAP_PEWTER_CITY, MAP_PEWTER_GYM, MAP_PEWTER_POKECENTER}
PAST_VIRIDIAN_MAP_IDS: set[int] = {
    MAP_ROUTE_2,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE,
    MAP_VIRIDIAN_FOREST,
    MAP_VIRIDIAN_FOREST_NORTH_GATE,
}

# ── Ollama JSON Schema ───────────────────────────────────────────────────────
PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "enum": ["done", "heal", "grind", "travel_to_pewter", "gym"],
            "description": "High-level goal to execute next",
        },
        "target_level": {
            "type": "integer",
            "minimum": 8,
            "maximum": 25,
            "description": "Target level for grinding (integer between 8 and 25)",
        },
        "reason": {
            "type": "string",
            "description": "Explanation and tactical reasoning for choosing this goal",
        },
        "note": {
            "type": "string",
            "description": "Short strategic note to store in notebook (may be empty string)",
        },
    },
    "required": ["goal", "target_level", "reason", "note"],
    "additionalProperties": False,
}

# ── Web Search Tool Definition ───────────────────────────────────────────────
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search Bulbapedia for game knowledge: maps, gym leaders, Pokémon types, moves.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Bulbapedia search query (e.g. Brock, Pewter Gym, Onix, Rock type)",
                }
            },
            "required": ["query"],
        },
    },
}

# ── Prompts ──────────────────────────────────────────────────────────────────
# Honest System Prompt: contains pure game mechanics and harness constraints,
# but strictly NO rule planner thresholds (e.g. no 0.35, no >= 14, no 12-15).
SYSTEM_PROMPT: str = """You are the System 2 tactical planner for Pokémon Red playing the gym quest.
Your ultimate objective is to earn the Boulder Badge by defeating Gym Leader Brock at Pewter Gym.

Each step, you select exactly one high-level goal from these five:

1. "done":
   - Precondition: Boulder Badge MUST already be obtained in RAM (wObtainedBadges bit 0 set).
   - What it does: Finishes the quest successfully.
   - Constraint: NEVER select "done" if the Boulder Badge is not in possession.

2. "heal":
   - What it does: Visits the Pokémon Center, restores party HP to full with Nurse Joy, and sets the respawn anchor.
   - Locations:
     * Viridian Pokémon Center (Map 41) when in Viridian City (Map 1) or Route 1 (Map 12).
     * Pewter Pokémon Center (Map 58) when in Pewter City (Map 2).
   - Precondition / Routing Constraint: System 1 cannot route back south from Route 2 (Map 13) or Viridian Forest (Maps 47, 50, 51) to Viridian. Healing cannot be executed from Route 2 or Viridian Forest.

3. "grind":
   - What it does: Paces in Route 1 tall grass, fights wild battles, levels up starter Pokémon.
   - Safety stop (how the skill works): it returns as soon as the target level is reached or HP falls below the
     grind safety stop shown in the state; if HP is already below it, grind does nothing (about 1 press).
   - Preconditions / Routing Constraints:
     * Can ONLY be executed from Viridian City / Route 1 area.
     * System 1 cannot execute grinding from the Pewter area (maps 2, 54, 58), Route 2 (Map 13), or Viridian Forest (maps 47, 50, 51).
   - Specify your desired `target_level` (integer between 8 and 25) in your output.

4. "travel_to_pewter":
   - What it does: Walks north from Viridian City through Route 2 and Viridian Forest to Pewter City.
   - Preconditions / Routing Constraints:
     * Must NOT be in the Pewter area (maps 2, 54, 58).
     * The route north is one-way for the harness navigation skill: once past Viridian onto Route 2 or into Viridian Forest, you cannot route back south to Viridian to heal or grind.

5. "gym":
   - What it does: Enters Pewter Gym (Map 54), challenges Jr. Trainer and Brock.
   - Preconditions:
     * Must be in Pewter area (maps 2, 54, 58). NEVER choose "gym" outside Pewter!

Important World Mechanics:
- If your Pokémon faints (blackout), you lose money and respawn at the last visited Pokémon Center where you healed.
- If you faint before healing at Pewter Pokémon Center, you are sent back to Viridian Pokémon Center.
- The quest has a fixed budget of button presses (shown in the state, with the presses used so far); the quest
  fails if the budget runs out before the badge is won.
Always output valid JSON complying strictly with the requested schema.
"""


# ── Exceptions ───────────────────────────────────────────────────────────────
class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama connection or timeout fails."""
    pass


# ── Ollama HTTP Client (stdlib only) ─────────────────────────────────────────
class OllamaClient:
    """Minimal Ollama chat client using only urllib.request and json."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
        retry_delay: float = 5.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retry_delay = retry_delay

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: Optional[dict[str, Any]] = None,
        think: bool = True,
        keep_alive: Optional[str] = "30m",
        options: Optional[dict[str, Any]] = None,
        retry_delay: Optional[float] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """POST /api/chat with model, messages, format schema, think, and options.

        Retries once after retry_delay (default 5s) on connection/timeout error.
        """
        url = f"{self.host}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": bool(think),
        }
        if schema is not None:
            payload["format"] = schema
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if tools is not None:
            payload["tools"] = tools

        opts: dict[str, Any] = {
            "temperature": 0.1,
            "seed": 42,
            "num_ctx": 16384,
        }
        if options:
            opts.update(options)
        payload["options"] = opts

        delay = retry_delay if retry_delay is not None else self.retry_delay

        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_resp = resp.read().decode("utf-8")
                resp_json = json.loads(raw_resp)
        except (urllib.error.URLError, socket.timeout, TimeoutError, http.client.HTTPException) as first_err:
            # Connection error mid-run: retry once after retry_delay
            time.sleep(delay)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_resp = resp.read().decode("utf-8")
                    resp_json = json.loads(raw_resp)
            except (urllib.error.URLError, socket.timeout, TimeoutError, http.client.HTTPException) as second_err:
                raise OllamaConnectionError(
                    f"Ollama request to {url} failed after retry: {second_err}"
                ) from second_err

        msg = resp_json.get("message", {})
        raw_content = msg.get("content", "")
        thinking = msg.get("thinking", "") or resp_json.get("thinking", "")
        tool_calls = msg.get("tool_calls", None)

        parsed_content: Any = raw_content
        if isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
            except Exception:
                parsed_content = raw_content

        return {
            "content": parsed_content,
            "thinking": thinking,
            "total_duration": resp_json.get("total_duration", 0),
            "prompt_eval_count": resp_json.get("prompt_eval_count", 0),
            "eval_count": resp_json.get("eval_count", 0),
            "raw_content": raw_content if isinstance(raw_content, str) else json.dumps(raw_content),
            "tool_calls": tool_calls,
            "message": msg,
        }

    def unload(self, model: str, timeout: Optional[float] = None) -> bool:
        """POST /api/generate with keep_alive: 0 to free VRAM immediately."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "keep_alive": 0,
        }
        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, socket.timeout, TimeoutError, http.client.HTTPException) as err:
            raise OllamaConnectionError(f"Failed to unload model {model} at {url}: {err}") from err


def unload_model(
    model: str = "gemma4:26b",
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> bool:
    """Convenience helper to unload an Ollama model from VRAM."""
    client = OllamaClient(host=host, timeout=timeout)
    return client.unload(model=model)


# ── Stall Guard Helper ───────────────────────────────────────────────────────
def check_stall_guard(
    consecutive_stalls: int,
    total_decisions: int,
    max_decisions: int = 60,
    stall_limit: int = 5,
) -> Optional[str]:
    """Check if LLM run should be stopped due to decision cap or stalling.

    Returns:
        "decision_cap", "stalled", or None.
    """
    if total_decisions >= max_decisions:
        return "decision_cap"
    if consecutive_stalls >= stall_limit:
        return "stalled"
    return None


# ── Decision Validation Helper ───────────────────────────────────────────────
def validate_decision(
    decision: Any,
    status: dict[str, Any],
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validate model output against schema and RAM state.

    Returns:
        (is_valid, rejection_reason, hallucination_kind)
    """
    if not isinstance(decision, dict):
        return False, "Output is not a valid JSON dictionary", "malformed_output"

    goal = decision.get("goal")
    target_level = decision.get("target_level")
    reason = decision.get("reason")
    note = decision.get("note")

    valid_goals = {"done", "heal", "grind", "travel_to_pewter", "gym"}
    if goal not in valid_goals:
        return False, f"Invalid goal '{goal}', must be one of {valid_goals}", "malformed_output"

    if not isinstance(target_level, int) or isinstance(target_level, bool):
        return False, f"Invalid target_level '{target_level}', must be integer 8-25", "malformed_output"

    if target_level < 8 or target_level > 25:
        return False, f"target_level {target_level} out of range [8, 25]", "malformed_output"

    if not isinstance(reason, str) or not reason.strip():
        return False, "Reason must be a non-empty string", "malformed_output"

    if not isinstance(note, str):
        return False, "Note must be a string", "malformed_output"

    # RAM state checks
    map_id = status.get("map_id", status.get("map", 0))
    badges = status.get("badges", 0)
    in_pewter_area = map_id in PEWTER_MAP_IDS
    past_viridian = map_id in PAST_VIRIDIAN_MAP_IDS

    # 1. done without badge bit set
    if goal == "done" and not bool(badges & 0x01):
        return False, "Goal 'done' selected but Boulder Badge bit 0 is not set in RAM", "done_without_badge"

    # 2. gym outside Pewter area
    if goal == "gym" and not in_pewter_area:
        return False, f"Goal 'gym' selected while outside Pewter area (map {map_id})", "gym_outside_pewter"

    # 3. travel_to_pewter while in Pewter area
    if goal == "travel_to_pewter" and in_pewter_area:
        return False, f"Goal 'travel_to_pewter' selected while already in Pewter area (map {map_id})", "travel_while_in_pewter"

    # 4. unexecutable_goal:
    # (a) grind in Pewter area
    if goal == "grind" and in_pewter_area:
        return False, f"unexecutable_goal: Goal 'grind' cannot be executed in Pewter area (map {map_id})", "unexecutable_goal"

    # (b) heal or grind on Route 2 / Viridian Forest (one-way route north)
    if goal in ("heal", "grind") and past_viridian:
        return False, f"unexecutable_goal: Goal '{goal}' cannot be executed from map {map_id} (cannot route south on one-way route)", "unexecutable_goal"

    return True, None, None


# ── System 2 LLM Planner Class ───────────────────────────────────────────────
class LLMPlanner:
    """System 2 LLM planner with Ollama backend, schema validation, and rule fallback."""

    def __init__(
        self,
        model: str = "gemma4:26b",
        think: bool = True,
        host: str = "http://localhost:11434",
        client: Optional[Any] = None,
        timeout: float = 300.0,
        max_notebook_notes: int = 12,
        max_notebook_chars: int = 800,
        options: Optional[dict[str, Any]] = None,
        tools: tuple[str, ...] | list[str] = (),
        max_tool_calls: int = 3,
    ) -> None:
        self.model = model
        self.think = think
        self.host = host
        self.timeout = timeout
        self.client = client if client is not None else OllamaClient(host=host, timeout=timeout)
        self.max_notebook_notes = max_notebook_notes
        self.max_notebook_chars = max_notebook_chars
        self.options = options or {"temperature": 0.1, "seed": 42, "num_ctx": 16384}
        self.tools: tuple[str, ...] = tuple(tools) if tools else ()
        self.max_tool_calls: int = max_tool_calls
        self.total_tool_calls: int = 0

        # Internal state
        self.notebook: list[str] = []
        self.decision_history: list[dict[str, Any]] = []
        self.latencies: list[float] = []
        self.total_decisions: int = 0
        self.llm_decisions: int = 0
        self.fallback_decisions: int = 0
        self.total_tokens: int = 0
        self.load_seconds: float = 0.0
        self.hallucinations: dict[str, int] = {
            "done_without_badge": 0,
            "gym_outside_pewter": 0,
            "travel_while_in_pewter": 0,
            "unexecutable_goal": 0,
            "malformed_output": 0,
        }
        self.last_decision_record: dict[str, Any] = {}
        self.target_level: Optional[int] = None

    def warmup(self) -> float:
        """Warm up Ollama model before the first decision without counting as a decision."""
        t0 = time.time()
        try:
            self._call_client(
                messages=[{"role": "user", "content": "ping"}],
                schema=None,
                think=False,
                keep_alive="30m",
            )
        except Exception:
            pass
        self.load_seconds = round(time.time() - t0, 3)
        return self.load_seconds

    def add_note(self, note: str) -> None:
        """Append a note to the notebook with count and char capping."""
        note = note.strip()
        if not note:
            return
        if len(note) > self.max_notebook_chars:
            note = note[:self.max_notebook_chars]
        self.notebook.append(note)
        if len(self.notebook) > self.max_notebook_notes:
            self.notebook = self.notebook[-self.max_notebook_notes:]
        while self.notebook and sum(len(n) for n in self.notebook) > self.max_notebook_chars:
            self.notebook.pop(0)

    def record_outcome(
        self,
        presses_used: int,
        status_after: dict[str, Any],
    ) -> None:
        """Update the most recent decision with its execution outcome."""
        if self.decision_history:
            last = self.decision_history[-1]
            last["presses_used"] = presses_used
            last["status_after"] = dict(status_after)

    def _build_user_message(
        self,
        status: dict[str, Any],
        pewter_healed: bool,
        heal_threshold: float = 0.35,
    ) -> str:
        """Format current RAM status, recent decisions history, and notebook."""
        map_id = status.get("map_id", status.get("map", 0))
        map_name = MAP_NAMES.get(map_id, f"Map {map_id}")
        x = status.get("x", 0)
        y = status.get("y", 0)
        facing = status.get("facing", 0)
        level = status.get("level", 0)
        hp = status.get("hp", 0)
        max_hp = status.get("max_hp", 1)
        hp_frac = status.get("hp_frac", (hp / max_hp) if max_hp > 0 else 0.0)
        badges = status.get("badges", 0)
        has_badge = bool(badges & 0x01)
        in_battle = status.get("in_battle", False)
        party_size = status.get("party_size", 1)

        target_str = (
            f"{self.target_level}"
            if self.target_level is not None
            else "None (none chosen yet)"
        )

        lines: list[str] = [
            "### CURRENT GAME BOY RAM STATUS",
            f"- Map: ID {map_id} ({map_name}) | Coordinates: ({x}, {y}) | Facing: {facing}",
            f"- Starter: Level {level} | HP: {hp}/{max_hp} ({hp_frac:.2%}) | Party Size: {party_size} | In Battle: {in_battle}",
            f"- Badges: 0x{badges:02X} (Boulder Badge: {'YES' if has_badge else 'NO'})",
            f"- Pewter Center Healed: {'YES' if pewter_healed else 'NO'}",
            f"- Your last chosen grind target level: {target_str}",
            f"- Grind skill safety stop: HP below {heal_threshold:.0%} of max",
            f"- Button presses used: {status.get('presses_used', 0)} of the {status.get('press_budget', '?')} budget",
            "",
            "### RECENT DECISIONS (Last ~8)",
        ]

        recent = self.decision_history[-8:]
        if not recent:
            lines.append("(No previous decisions yet. This is step 1.)")
        else:
            for idx, item in enumerate(recent, 1):
                g = item.get("goal", "unknown")
                r = item.get("reason", "")
                src = item.get("source", "llm")
                p_used = item.get("presses_used")
                st_a = item.get("status_after")

                prefix = f"{idx}. Goal: {g}"
                if src != "llm":
                    prefix += f" [{src}]"
                prefix += f" | Reason: {r}"

                if p_used is not None and st_a:
                    m_a = st_a.get("map_id", 0)
                    m_a_name = MAP_NAMES.get(m_a, f"Map {m_a}")
                    prefix += f" -> Outcome: {p_used} presses, Level {st_a.get('level')}, HP {st_a.get('hp')}/{st_a.get('max_hp')}, Map {m_a} ({m_a_name})"
                lines.append(prefix)

        lines.append("")
        lines.append("### STRATEGIC NOTEBOOK")
        if not self.notebook:
            lines.append("(Notebook is empty)")
        else:
            for n_idx, n_text in enumerate(self.notebook, 1):
                lines.append(f"{n_idx}. {n_text}")

        lines.append("")
        lines.append("Decide the next goal, target_level (8-25), reason, and note in JSON.")
        return "\n".join(lines)

    def _call_client(
        self,
        messages: list[dict[str, Any]],
        schema: Optional[dict[str, Any]] = PLANNER_SCHEMA,
        think: Optional[bool] = None,
        keep_alive: str = "30m",
        options: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Invoke client with error and format handling."""
        use_think = self.think if think is None else think
        use_options = dict(self.options)
        if options:
            use_options.update(options)

        if hasattr(self.client, "chat") and callable(self.client.chat):
            if tools is not None:
                return self.client.chat(
                    model=self.model,
                    messages=messages,
                    schema=schema,
                    think=use_think,
                    keep_alive=keep_alive,
                    options=use_options,
                    tools=tools,
                )
            return self.client.chat(
                model=self.model,
                messages=messages,
                schema=schema,
                think=use_think,
                keep_alive=keep_alive,
                options=use_options,
            )
        elif callable(self.client):
            if tools is not None:
                return self.client(
                    model=self.model,
                    messages=messages,
                    schema=schema,
                    think=use_think,
                    keep_alive=keep_alive,
                    options=use_options,
                    tools=tools,
                )
            return self.client(
                model=self.model,
                messages=messages,
                schema=schema,
                think=use_think,
                keep_alive=keep_alive,
                options=use_options,
            )
        else:
            raise RuntimeError(f"Client {self.client} is neither callable nor has a chat method")

    def __call__(
        self,
        status: dict[str, Any],
        target_level: int = 15,
        pewter_healed: bool = False,
        heal_threshold: float = 0.35,
    ) -> Goal:
        """Produce next quest Goal, validating output and falling back to rule planner on error."""
        self.total_decisions += 1
        effective_fallback_target = self.target_level if self.target_level is not None else target_level

        user_content = self._build_user_message(status, pewter_healed, heal_threshold)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        client_error: Optional[Exception] = None
        resp_data: dict[str, Any] = {}
        decision_tool_calls: list[dict[str, Any]] = []
        extra_prompt_tokens: int = 0
        extra_eval_tokens: int = 0

        if "web_search" in self.tools:
            try:
                from lab.agent.web_search import web_search

                tool_sys_prompt = SYSTEM_PROMPT.strip() + "\nYou may search Bulbapedia for game knowledge: maps, gym leaders, Pokémon types, moves.\n"
                tool_messages = [
                    {"role": "system", "content": tool_sys_prompt},
                    {"role": "user", "content": user_content},
                ]
                tool_calls_count = 0
                while tool_calls_count < self.max_tool_calls:
                    turn_resp = self._call_client(
                        messages=tool_messages,
                        schema=PLANNER_SCHEMA if tool_calls_count > 0 else None,
                        think=self.think,
                        keep_alive="30m",
                        tools=[WEB_SEARCH_TOOL],
                    )

                    t_calls = turn_resp.get("tool_calls")
                    if not t_calls and isinstance(turn_resp.get("message"), dict):
                        t_calls = turn_resp["message"].get("tool_calls")

                    # Fallback check for schema-based action protocol
                    if not t_calls:
                        c_val = turn_resp.get("content")
                        if isinstance(c_val, dict) and c_val.get("action") == "search" and "query" in c_val:
                            t_calls = [{"function": {"name": "web_search", "arguments": {"query": c_val["query"]}}}]

                    if not t_calls:
                        resp_data = turn_resp
                        break

                    extra_prompt_tokens += int(turn_resp.get("prompt_eval_count", 0))
                    extra_eval_tokens += int(turn_resp.get("eval_count", 0))

                    asst_msg = turn_resp.get("message")
                    if not asst_msg or not isinstance(asst_msg, dict):
                        asst_msg = {
                            "role": "assistant",
                            "content": turn_resp.get("raw_content", ""),
                            "tool_calls": t_calls,
                        }
                    tool_messages.append(asst_msg)

                    for tc in t_calls:
                        if tool_calls_count >= self.max_tool_calls:
                            break
                        fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                        fn_name = fn_info.get("name", "web_search")
                        fn_args = fn_info.get("arguments", {})
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                fn_args = {"query": fn_args}
                        query_str = fn_args.get("query", "") if isinstance(fn_args, dict) else str(fn_args)

                        t_tool_start = time.time()
                        try:
                            results = web_search(query_str) if fn_name == "web_search" else []
                        except Exception:
                            results = []
                        call_lat = round(time.time() - t_tool_start, 3)
                        titles = [r.get("title", "") for r in results if isinstance(r, dict)]

                        decision_tool_calls.append({
                            "tool": fn_name,
                            "query": query_str,
                            "titles": titles,
                            "latency": call_lat,
                        })
                        tool_calls_count += 1
                        self.total_tool_calls += 1

                        tool_messages.append({
                            "role": "tool",
                            "content": json.dumps(results),
                        })

                # Check if final schema-constrained call is needed
                need_final = False
                if not resp_data:
                    need_final = True
                else:
                    cand = resp_data.get("content") if isinstance(resp_data, dict) else None
                    is_c_valid, _, _ = validate_decision(cand, status)
                    if not is_c_valid:
                        need_final = True
                        if resp_data.get("raw_content"):
                            tool_messages.append({"role": "assistant", "content": resp_data.get("raw_content")})

                if need_final:
                    if resp_data:
                        extra_prompt_tokens += int(resp_data.get("prompt_eval_count", 0))
                        extra_eval_tokens += int(resp_data.get("eval_count", 0))
                    resp_data = self._call_client(
                        messages=tool_messages,
                        schema=PLANNER_SCHEMA,
                        think=self.think,
                        keep_alive="30m",
                        tools=None,
                    )
            except (OllamaConnectionError, urllib.error.URLError, TimeoutError, socket.timeout):
                raise
            except Exception as err:
                client_error = err
        if "web_search" not in self.tools:
            try:
                resp_data = self._call_client(messages, schema=PLANNER_SCHEMA, think=self.think, keep_alive="30m")
            except (OllamaConnectionError, urllib.error.URLError, TimeoutError, socket.timeout):
                # Connection and timeout errors must raise clearly
                raise
            except Exception as err:
                client_error = err

        latency = round(time.time() - t0, 3)
        self.latencies.append(latency)

        thinking = ""
        prompt_tokens = 0
        eval_tokens = 0
        raw_output: Any = ""

        if isinstance(resp_data, dict):
            # If client returned a standard dict or chat response
            if "content" in resp_data:
                parsed_decision = resp_data.get("content")
                thinking = str(resp_data.get("thinking", ""))
                prompt_tokens = int(resp_data.get("prompt_eval_count", 0))
                eval_tokens = int(resp_data.get("eval_count", 0))
                if extra_prompt_tokens:
                    prompt_tokens += extra_prompt_tokens
                if extra_eval_tokens:
                    eval_tokens += extra_eval_tokens
                raw_output = resp_data.get("raw_content", parsed_decision)
            else:
                # Direct JSON output from client
                parsed_decision = resp_data
                raw_output = resp_data
        elif isinstance(resp_data, str):
            try:
                parsed_decision = json.loads(resp_data)
            except Exception:
                parsed_decision = resp_data
            raw_output = resp_data
        else:
            parsed_decision = resp_data
            raw_output = str(resp_data)

        self.total_tokens += (prompt_tokens + eval_tokens)

        # Truncate thinking to ~2000 chars
        truncated_thinking = thinking[:2000] if thinking else ""

        if client_error is not None:
            is_valid = False
            rejection_reason = f"Client error: {client_error}"
            hallucination_kind = "malformed_output"
        else:
            is_valid, rejection_reason, hallucination_kind = validate_decision(parsed_decision, status)

        if is_valid:
            # Valid LLM decision
            self.llm_decisions += 1
            goal_str = parsed_decision["goal"]
            reason_str = parsed_decision["reason"]
            self.target_level = parsed_decision["target_level"]
            note_str = parsed_decision.get("note", "")
            if note_str:
                self.add_note(note_str)

            final_goal = Goal(goal_str, reason=reason_str, target_level=self.target_level)
            source = "llm"
            rej_rec = None
        else:
            # Hallucination / Malformed / Unexecutable -> Fall back to rule planner
            kind = hallucination_kind or "malformed_output"
            self.hallucinations[kind] = self.hallucinations.get(kind, 0) + 1
            self.fallback_decisions += 1

            fallback_goal = plan_gym(
                status,
                target_level=effective_fallback_target,
                pewter_healed=pewter_healed,
                heal_threshold=heal_threshold,
            )
            final_goal = Goal(
                str(fallback_goal),
                reason=f"[rules_fallback: {rejection_reason}] {fallback_goal.reason}",
                target_level=effective_fallback_target,
            )
            source = "rules_fallback"
            rej_rec = rejection_reason

        # Record timeline record
        self.last_decision_record = {
            "source": source,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "eval_tokens": eval_tokens,
            "raw_output": raw_output,
            "thinking": truncated_thinking,
            "rejection_reason": rej_rec,
            "goal": str(final_goal),
            "reason": final_goal.reason,
            "target_level": getattr(final_goal, "target_level", effective_fallback_target),
            "tool_calls": decision_tool_calls,
        }

        # Append to history
        self.decision_history.append({
            "goal": str(final_goal),
            "reason": final_goal.reason,
            "source": source,
            "presses_used": None,
            "status_after": None,
        })

        return final_goal

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics block for result JSON."""
        mean_lat = round(float(np.mean(self.latencies)), 3) if self.latencies else 0.0
        max_lat = round(float(np.max(self.latencies)), 3) if self.latencies else 0.0
        return {
            "type": "llm",
            "model": self.model,
            "load_seconds": self.load_seconds,
            "decisions": self.total_decisions,
            "llm_decisions": self.llm_decisions,
            "hallucinations": dict(self.hallucinations),
            "fallbacks": self.fallback_decisions,
            "mean_latency": mean_lat,
            "max_latency": max_lat,
            "total_tokens": self.total_tokens,
            "notebook": list(self.notebook),
            "tool_calls": self.total_tool_calls,
        }
