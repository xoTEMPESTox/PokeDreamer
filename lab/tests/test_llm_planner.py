"""test_llm_planner.py — Unit tests for LLM System 2 planner and Ollama client.

All tests use a fake client or mock transport; no network calls or Ollama GPU loads are performed.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

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
    plan_gym,
    run_gym_quest,
)
from lab.agent.llm_planner import (
    LLMPlanner,
    OllamaClient,
    OllamaConnectionError,
    PLANNER_SCHEMA,
    SYSTEM_PROMPT,
    check_stall_guard,
    unload_model,
    validate_decision,
)


# ── Fixtures & Helper Fakes ──────────────────────────────────────────────────
class FakeOllamaClient:
    """Fake client simulating Ollama chat responses without network calls."""

    def __init__(self, response_data: dict | str, thinking: str = "Thinking step-by-step..."):
        self.response_data = response_data
        self.thinking = thinking
        self.last_messages: list[dict[str, str]] = []
        self.last_schema: dict | None = None
        self.last_think: bool | None = None
        self.last_options: dict | None = None
        self.last_keep_alive: str | None = None
        self.call_count: int = 0

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict | None = None,
        think: bool = True,
        keep_alive: str = "30m",
        options: dict | None = None,
        **kwargs,
    ) -> dict:
        self.call_count += 1
        self.last_messages = messages
        self.last_schema = schema
        self.last_think = think
        self.last_keep_alive = keep_alive
        self.last_options = options

        raw_str = (
            self.response_data
            if isinstance(self.response_data, str)
            else json.dumps(self.response_data)
        )
        parsed = self.response_data if isinstance(self.response_data, dict) else None
        if parsed is None and isinstance(self.response_data, str):
            try:
                parsed = json.loads(self.response_data)
            except Exception:
                parsed = self.response_data

        return {
            "content": parsed,
            "thinking": self.thinking,
            "total_duration": 1_500_000_000,  # 1.5s in ns
            "prompt_eval_count": 150,
            "eval_count": 45,
            "raw_content": raw_str,
        }


# ── Item 1: Prompt Policy Leaks & Target Level Hiding Tests ──────────────────
def test_system_prompt_has_no_rule_thresholds():
    """Verify SYSTEM_PROMPT contains no leaked rule planner thresholds."""
    # Check that rule policy thresholds are absent
    forbidden_snippets = [
        "0.35",
        "< 0.35",
        ">= 14",
        "12-15",
        "100%",
        "full HP",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in SYSTEM_PROMPT, f"Forbidden snippet '{snippet}' found in SYSTEM_PROMPT"

    # Check that game facts and mechanics are present
    assert "Boulder Badge" in SYSTEM_PROMPT
    assert "Pewter Gym" in SYSTEM_PROMPT
    assert "Route 1" in SYSTEM_PROMPT
    assert "Viridian Forest" in SYSTEM_PROMPT
    assert "blackout" in SYSTEM_PROMPT or "faints" in SYSTEM_PROMPT
    assert "one-way" in SYSTEM_PROMPT


def test_prompt_shows_no_target_level_until_llm_chooses():
    """The user message must show none chosen yet until the LLM has chosen one."""
    fake_client = FakeOllamaClient({
        "goal": "grind",
        "target_level": 12,
        "reason": "Grinding starter",
        "note": "First note",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 8,
        "hp": 25,
        "max_hp": 25,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
        "x": 20,
        "y": 20,
    }

    # Step 1: LLM has not chosen a target level yet
    planner(status, target_level=15)
    sent_msg_step1 = fake_client.last_messages[1]["content"]
    assert "Your last chosen grind target level: None (none chosen yet)" in sent_msg_step1
    assert "Target Level: 15" not in sent_msg_step1
    assert "15" not in sent_msg_step1.split("Your last chosen grind target level:")[1].split("\n")[0]

    # Step 2: Now the LLM chose target_level 12 on step 1
    planner(status, target_level=15)
    sent_msg_step2 = fake_client.last_messages[1]["content"]
    assert "Your last chosen grind target level: 12" in sent_msg_step2


def test_rule_fallback_uses_cli_target_before_llm_choice_and_llm_choice_after():
    """Rule fallback uses CLI target before LLM chooses, and LLM target after."""
    # Step 1: Malformed output, LLM hasn't chosen a target -> uses CLI target 14
    fake_client = FakeOllamaClient("malformed json")
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 8,
        "hp": 25,
        "max_hp": 25,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
    }

    goal1 = planner(status, target_level=14)
    assert goal1 == "grind"
    assert getattr(goal1, "target_level", None) == 14
    assert planner.target_level is None  # still None

    # Step 2: LLM makes valid choice with target_level 11
    fake_client.response_data = {
        "goal": "grind",
        "target_level": 11,
        "reason": "Level 8 to 11",
        "note": "",
    }
    goal2 = planner(status, target_level=14)
    assert goal2.target_level == 11
    assert planner.target_level == 11

    # Step 3: LLM outputs malformed -> fallback uses LLM's chosen target (11), not CLI (14)
    status_lv12 = dict(status, level=12)
    fake_client.response_data = "malformed again"
    goal3 = planner(status_lv12, target_level=14)
    # Level 12 >= target 11, so rule planner chooses travel_to_pewter (not grind)
    assert goal3 == "travel_to_pewter"
    assert getattr(goal3, "target_level", None) == 11


# ── Item 2: Unexecutable Goal Hallucinations Tests ───────────────────────────
def test_unexecutable_goal_grind_in_pewter_rejected():
    """Grind in Pewter area maps (2, 54, 58) is rejected as unexecutable_goal."""
    pewter_maps = [MAP_PEWTER_CITY, MAP_PEWTER_GYM, MAP_PEWTER_POKECENTER]
    for m in pewter_maps:
        fake_client = FakeOllamaClient({
            "goal": "grind",
            "target_level": 15,
            "reason": "Grind in Pewter",
            "note": "",
        })
        planner = LLMPlanner(client=fake_client)
        status = {
            "badges": 0,
            "level": 14,
            "hp": 38,
            "max_hp": 38,
            "hp_frac": 1.0,
            "map_id": m,
        }
        goal = planner(status, target_level=15, pewter_healed=True)
        assert planner.hallucinations["unexecutable_goal"] == 1
        assert planner.fallback_decisions == 1
        assert planner.last_decision_record["source"] == "rules_fallback"
        assert "unexecutable_goal" in str(planner.last_decision_record["rejection_reason"])


def test_unexecutable_goal_heal_or_grind_past_viridian_rejected():
    """Heal or grind on Route 2 / Viridian Forest (13, 50, 51, 47) is rejected as unexecutable_goal."""
    past_viridian_maps = [
        MAP_ROUTE_2,
        MAP_VIRIDIAN_FOREST_SOUTH_GATE,
        MAP_VIRIDIAN_FOREST,
        MAP_VIRIDIAN_FOREST_NORTH_GATE,
    ]
    for m in past_viridian_maps:
        for bad_goal in ["heal", "grind"]:
            fake_client = FakeOllamaClient({
                "goal": bad_goal,
                "target_level": 15,
                "reason": f"Try {bad_goal} past Viridian",
                "note": "",
            })
            planner = LLMPlanner(client=fake_client)
            status = {
                "badges": 0,
                "level": 14,
                "hp": 20,
                "max_hp": 38,
                "hp_frac": 20 / 38,
                "map_id": m,
            }
            goal = planner(status, target_level=15)
            assert planner.hallucinations["unexecutable_goal"] == 1
            assert planner.fallback_decisions == 1
            assert planner.last_decision_record["source"] == "rules_fallback"
            assert "unexecutable_goal" in str(planner.last_decision_record["rejection_reason"])


# ── Item 3: Stall Guard Tests ────────────────────────────────────────────────
def test_stall_guard_helper():
    """Unit test the stall guard logic."""
    # Under decision cap
    assert check_stall_guard(consecutive_stalls=0, total_decisions=59, max_decisions=60) is None
    # At or above decision cap
    assert check_stall_guard(consecutive_stalls=0, total_decisions=60, max_decisions=60) == "decision_cap"
    assert check_stall_guard(consecutive_stalls=0, total_decisions=65, max_decisions=60) == "decision_cap"

    # Under stall limit
    assert check_stall_guard(consecutive_stalls=4, total_decisions=10, max_decisions=60) is None
    # At or above stall limit
    assert check_stall_guard(consecutive_stalls=5, total_decisions=10, max_decisions=60) == "stalled"
    assert check_stall_guard(consecutive_stalls=7, total_decisions=10, max_decisions=60) == "stalled"


def test_stall_guard_decision_cap_in_run_gym_quest():
    """run_gym_quest terminates with decision_cap when total decisions reach max_llm_decisions."""
    fake_client = FakeOllamaClient({
        "goal": "heal",
        "target_level": 15,
        "reason": "Healing at Viridian PC",
        "note": "",
    })
    fake_planner = LLMPlanner(client=fake_client)

    # Patch heal_pkmn_center and PokemonRed to simulate execution
    with patch("lab.agent.gym_quest.PokemonRed") as MockEnv:
        mock_env_inst = MagicMock()
        mock_env_inst.step.return_value = (None, {"map_id": MAP_VIRIDIAN_CITY})
        mock_env_inst.observe.return_value = (None, {"map_id": MAP_VIRIDIAN_CITY, "x": 23, "y": 26, "in_battle": False})
        mock_env_inst.pyboy.memory = [0] * 65536
        MockEnv.return_value.__enter__.return_value = mock_env_inst

        with patch("lab.agent.gym_quest.heal_pkmn_center", return_value={"healed": True}):
            with patch("lab.agent.gym_quest.party_status", return_value={
                "level": 14, "hp": 38, "max_hp": 38, "hp_frac": 1.0, "party_size": 1,
                "map_id": MAP_VIRIDIAN_CITY, "x": 23, "y": 26, "in_battle": False, "facing": 0
            }):
                res = run_gym_quest(
                    planner="llm",
                    llm_planner=fake_planner,
                    max_llm_decisions=3,
                    budget=10000,
                )
                assert res["stopped_reason"] == "decision_cap"
                assert res["badge_obtained"] is False


def test_stall_guard_stalled_in_run_gym_quest():
    """run_gym_quest terminates with stalled when 5 consecutive decisions use <= 1 press."""
    fake_client = FakeOllamaClient({
        "goal": "heal",
        "target_level": 15,
        "reason": "Healing stall",
        "note": "",
    })
    fake_planner = LLMPlanner(client=fake_client)

    with patch("lab.agent.gym_quest.PokemonRed") as MockEnv:
        mock_env_inst = MagicMock()
        mock_env_inst.step.return_value = (None, {"map_id": MAP_VIRIDIAN_CITY})
        mock_env_inst.observe.return_value = (None, {"map_id": MAP_VIRIDIAN_CITY, "x": 23, "y": 26, "in_battle": False})
        mock_env_inst.pyboy.memory = [0] * 65536
        MockEnv.return_value.__enter__.return_value = mock_env_inst

        # Skill that consumes 0 presses
        with patch("lab.agent.gym_quest.heal_pkmn_center", return_value={"healed": True}):
            with patch("lab.agent.gym_quest.party_status", return_value={
                "level": 14, "hp": 38, "max_hp": 38, "hp_frac": 1.0, "party_size": 1,
                "map_id": MAP_VIRIDIAN_CITY, "x": 23, "y": 26, "in_battle": False, "facing": 0
            }):
                res = run_gym_quest(
                    planner="llm",
                    llm_planner=fake_planner,
                    max_llm_decisions=60,
                    budget=10000,
                )
                assert res["stopped_reason"] == "stalled"
                assert res["badge_obtained"] is False


# ── Item 4: Timeouts, Context, Retry & Warmup Tests ──────────────────────────
def test_ollama_client_retry_once_on_connection_error():
    """Client retries once after retry_delay on connection error, then succeeds."""
    client = OllamaClient(host="http://localhost:11434", retry_delay=0.01)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {"content": json.dumps({"goal": "heal", "target_level": 15, "reason": "ok", "note": ""})},
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    side_effects = [
        urllib.error.URLError("Connection reset"),
        mock_resp,
    ]

    with patch("urllib.request.urlopen", side_effect=side_effects) as mock_urlopen:
        res = client.chat(model="gemma4:26b", messages=[{"role": "user", "content": "hi"}], retry_delay=0.01)
        assert mock_urlopen.call_count == 2
        assert res["content"]["goal"] == "heal"


def test_ollama_client_retry_once_failure_raises_clear_exception():
    """Client retries once after retry_delay; if it fails again, raises OllamaConnectionError."""
    client = OllamaClient(host="http://localhost:11434", retry_delay=0.01)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")) as mock_urlopen:
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.chat(model="gemma4:26b", messages=[{"role": "user", "content": "hi"}], retry_delay=0.01)
        assert mock_urlopen.call_count == 2
        assert "failed after retry" in str(exc_info.value)


def test_warmup_call_and_load_seconds():
    """warmup() calls the model with keep_alive '30m' and logs load_seconds without counting as a decision."""
    fake_client = FakeOllamaClient({"goal": "grind", "target_level": 14, "reason": "ok", "note": ""})
    planner = LLMPlanner(client=fake_client)

    load_sec = planner.warmup()
    assert load_sec >= 0.0
    assert planner.load_seconds == load_sec
    assert planner.total_decisions == 0  # NOT counted as decision
    assert len(planner.decision_history) == 0
    assert fake_client.last_keep_alive == "30m"
    assert fake_client.last_think is False

    stats = planner.get_stats()
    assert stats["load_seconds"] == load_sec


def test_ollama_client_defaults_num_ctx_and_timeout():
    """Client defaults timeout to 300s, num_ctx to 16384, and keep_alive to 30m."""
    client = OllamaClient(host="http://localhost:11434")
    assert client.timeout == 300.0

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {"content": "{}"},
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        client.chat(model="gemma4:26b", messages=[])
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["keep_alive"] == "30m"
        assert body["options"]["num_ctx"] == 16384


# ── Existing Core Tests (Preserved & Updated) ────────────────────────────────
def test_valid_output_becomes_right_goal_and_target_level():
    """Valid JSON output becomes the right Goal instance and target_level."""
    fake_client = FakeOllamaClient({
        "goal": "grind",
        "target_level": 14,
        "reason": "Level 8 is too low for Brock, grinding on Route 1",
        "note": "Pacing tall grass until level 14",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 8,
        "hp": 25,
        "max_hp": 25,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
        "x": 21,
        "y": 30,
        "facing": 0,
        "in_battle": False,
        "party_size": 1,
    }

    goal = planner(status, target_level=15)

    assert isinstance(goal, Goal)
    assert goal == "grind"
    assert goal.reason == "Level 8 is too low for Brock, grinding on Route 1"
    assert getattr(goal, "target_level", None) == 14
    assert planner.target_level == 14
    assert planner.llm_decisions == 1
    assert planner.fallback_decisions == 0
    assert planner.notebook == ["Pacing tall grass until level 14"]
    assert planner.last_decision_record["source"] == "llm"
    assert planner.last_decision_record["prompt_tokens"] == 150
    assert planner.last_decision_record["eval_tokens"] == 45
    assert "Thinking step-by-step" in planner.last_decision_record["thinking"]


def test_done_without_badge_rejected_and_fallback():
    """Selecting 'done' without the badge bit in RAM is counted as hallucination and falls back to rules."""
    fake_client = FakeOllamaClient({
        "goal": "done",
        "target_level": 15,
        "reason": "Quest is finished",
        "note": "We beat the gym",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 15,
        "hp": 42,
        "max_hp": 42,
        "hp_frac": 1.0,
        "map_id": MAP_PEWTER_CITY,
        "x": 10,
        "y": 10,
    }

    goal = planner(status, target_level=15, pewter_healed=True)
    assert goal == "gym"  # Fallback to rule planner
    assert planner.fallback_decisions == 1
    assert planner.llm_decisions == 0
    assert planner.hallucinations["done_without_badge"] == 1
    assert planner.last_decision_record["source"] == "rules_fallback"


def test_gym_outside_pewter_rejected_and_fallback():
    """Selecting 'gym' while outside Pewter area is counted as hallucination and falls back to rules."""
    fake_client = FakeOllamaClient({
        "goal": "gym",
        "target_level": 15,
        "reason": "Challenge Brock right away",
        "note": "Entering gym",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 15,
        "hp": 42,
        "max_hp": 42,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
        "x": 20,
        "y": 20,
    }

    goal = planner(status, target_level=15)
    assert goal == "travel_to_pewter"  # Fallback to rule planner
    assert planner.fallback_decisions == 1
    assert planner.llm_decisions == 0
    assert planner.hallucinations["gym_outside_pewter"] == 1
    assert planner.last_decision_record["source"] == "rules_fallback"


def test_travel_to_pewter_while_in_pewter_rejected_and_fallback():
    """Selecting 'travel_to_pewter' while already in Pewter area is counted as hallucination."""
    fake_client = FakeOllamaClient({
        "goal": "travel_to_pewter",
        "target_level": 15,
        "reason": "Traveling to Pewter",
        "note": "Walking",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 0,
        "level": 15,
        "hp": 30,
        "max_hp": 42,
        "hp_frac": 30 / 42,
        "map_id": MAP_PEWTER_CITY,
        "x": 10,
        "y": 10,
    }

    goal = planner(status, target_level=15, pewter_healed=False)
    assert goal == "heal"
    assert planner.fallback_decisions == 1
    assert planner.hallucinations["travel_while_in_pewter"] == 1
    assert planner.last_decision_record["source"] == "rules_fallback"


def test_malformed_outputs_rejected_and_fallback():
    """Malformed JSON, missing schema properties, or invalid values fall back to rule planner."""
    status = {
        "badges": 0,
        "level": 8,
        "hp": 25,
        "max_hp": 25,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
        "x": 20,
        "y": 20,
    }

    malformed_cases = [
        "Not a JSON string at all",
        {"target_level": 14, "reason": "Missing goal"},
        {"goal": "invalid_goal_name", "target_level": 14, "reason": "Bad goal", "note": ""},
        {"goal": "grind", "target_level": 50, "reason": "Level too high", "note": ""},
        {"goal": "grind", "target_level": 5, "reason": "Level too low", "note": ""},
        {"goal": "grind", "target_level": "fourteen", "reason": "Not int", "note": ""},
        {"goal": "grind", "target_level": 14, "reason": "", "note": ""},
        ["not", "a", "dict"],
    ]

    for bad_case in malformed_cases:
        fake_client = FakeOllamaClient(bad_case)
        planner = LLMPlanner(client=fake_client)

        goal = planner(status, target_level=15)
        assert goal == "grind"
        assert planner.fallback_decisions == 1
        assert planner.hallucinations["malformed_output"] == 1
        assert planner.last_decision_record["source"] == "rules_fallback"


def test_notebook_cap():
    """Notebook caps entries at max_notes (12) and total length at max_chars (800)."""
    fake_client = FakeOllamaClient({
        "goal": "grind",
        "target_level": 14,
        "reason": "Test note",
        "note": "A" * 50,
    })
    planner = LLMPlanner(client=fake_client, max_notebook_notes=12, max_notebook_chars=800)

    for i in range(20):
        planner.add_note(f"Note-{i:02d}: " + "X" * 40)

    assert len(planner.notebook) <= 12
    total_chars = sum(len(n) for n in planner.notebook)
    assert total_chars <= 800
    assert "Note-00" not in planner.notebook[0]
    assert "Note-19" in planner.notebook[-1]

    planner.add_note("Z" * 900)
    assert len(planner.notebook) == 1
    assert len(planner.notebook[0]) <= 800


def test_prompt_contains_ram_fields():
    """Prompt sent to the model contains all required RAM state fields."""
    fake_client = FakeOllamaClient({
        "goal": "grind",
        "target_level": 14,
        "reason": "Test RAM prompt",
        "note": "RAM check note",
    })
    planner = LLMPlanner(client=fake_client)

    status = {
        "badges": 1,
        "level": 12,
        "hp": 27,
        "max_hp": 35,
        "hp_frac": 27 / 35,
        "party_size": 1,
        "map_id": MAP_VIRIDIAN_CITY,
        "x": 23,
        "y": 26,
        "facing": 4,
        "in_battle": False,
    }

    planner(status, target_level=14, pewter_healed=True)

    sent_messages = fake_client.last_messages
    assert len(sent_messages) == 2
    user_msg = sent_messages[1]["content"]

    assert "Map: ID 1 (Viridian City)" in user_msg
    assert "(23, 26)" in user_msg
    assert "Level 12" in user_msg
    assert "27/35" in user_msg
    assert "0x01" in user_msg
    assert "Boulder Badge: YES" in user_msg
    assert "Pewter Center Healed: YES" in user_msg


def test_ollama_client_unload_model():
    """unload_model sends POST /api/generate with keep_alive: 0 to free VRAM."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"{}"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        ok = unload_model(model="gemma4:26b", host="http://localhost:11434")
        assert ok is True
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/generate"
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "gemma4:26b"
        assert body["keep_alive"] == 0


def test_planner_stats_and_record_outcome():
    """Planner stats block and record_outcome correctly track history and summary metrics."""
    fake_client = FakeOllamaClient({
        "goal": "heal",
        "target_level": 15,
        "reason": "Healing at Viridian PC",
        "note": "Health top-up",
    })
    planner = LLMPlanner(client=fake_client)

    st_before = {
        "badges": 0,
        "level": 9,
        "hp": 10,
        "max_hp": 25,
        "hp_frac": 10 / 25,
        "map_id": MAP_VIRIDIAN_CITY,
    }
    goal = planner(st_before, target_level=15)
    assert goal == "heal"

    st_after = dict(st_before)
    st_after["hp"] = 25
    st_after["hp_frac"] = 1.0
    st_after["map_id"] = MAP_VIRIDIAN_CITY
    planner.record_outcome(presses_used=120, status_after=st_after)

    stats = planner.get_stats()
    assert stats["type"] == "llm"
    assert stats["model"] == "gemma4:26b"
    assert stats["decisions"] == 1
    assert stats["llm_decisions"] == 1
    assert stats["fallbacks"] == 0
    assert stats["total_tokens"] == 195
    assert len(stats["notebook"]) == 1
    assert stats["notebook"][0] == "Health top-up"
    assert "done_without_badge" in stats["hallucinations"]
    assert "unexecutable_goal" in stats["hallucinations"]
    assert stats["hallucinations"]["unexecutable_goal"] == 0

    assert planner.decision_history[-1]["presses_used"] == 120
    assert planner.decision_history[-1]["status_after"]["hp"] == 25


def test_goal_unpacking_and_target_level():
    """Goal tuple unpacking and target_level attribute behave as expected."""
    g = Goal("grind", "level up", target_level=14)
    goal_str, reason = g
    assert goal_str == "grind"
    assert reason == "level up"
    assert g.target_level == 14
    assert g == "grind"


def test_cli_flags_for_planner():
    """Verify that gym_quest CLI parser correctly accepts all flags including --max-llm-decisions."""
    from lab.agent.gym_quest import main

    with patch("sys.argv", [
        "gym_quest.py",
        "--planner", "llm",
        "--llm-model", "gemma4:26b",
        "--no-think",
        "--max-llm-decisions", "45",
    ]):
        with patch("lab.agent.gym_quest.run_gym_quest") as mock_run:
            with patch("lab.agent.llm_planner.unload_model") as mock_unload:
                mock_run.return_value = {
                    "presses_total": 100,
                    "badge_obtained": True,
                    "blackouts": 0,
                    "level_final": 15,
                    "level_at_gym": 15,
                    "battles_won": 5,
                    "seed": 100,
                }
                main()
                assert mock_run.call_count == 1
                kwargs = mock_run.call_args[1]
                assert kwargs["planner"] == "llm"
                assert kwargs["llm_model"] == "gemma4:26b"
                assert kwargs["llm_think"] is False
                assert kwargs["max_llm_decisions"] == 45
                assert mock_unload.call_count == 1
