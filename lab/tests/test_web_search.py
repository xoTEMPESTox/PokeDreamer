"""test_web_search.py — Unit tests for Bulbapedia web search tool and LLM planner tool calling.

Tests cover:
  (a) tools off -> identical request payload to today;
  (b) a tool_call reply -> one search, then final JSON decision accepted;
  (c) the max_tool_calls cap is respected;
  (d) a network error -> [], and the decision still happens;
  (e) the cache hit path.
No network calls or GPU loads are performed.
"""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from lab.agent.gym_quest import (
    Goal,
    MAP_PEWTER_CITY,
    MAP_VIRIDIAN_CITY,
    run_gym_quest,
)
from lab.agent.llm_planner import (
    LLMPlanner,
    OllamaClient,
    PLANNER_SCHEMA,
    SYSTEM_PROMPT,
    WEB_SEARCH_TOOL,
)
from lab.agent.web_search import (
    USER_AGENT,
    _clean_snippet,
    web_search,
)


# ── Web Search Unit Tests ────────────────────────────────────────────────────
def test_clean_snippet_html_stripping():
    """Verify HTML tags and entities are stripped from search snippets."""
    raw = 'Brock&#039;s <span class="searchmatch">Onix</span> is a Rock/Ground &amp; tough.'
    cleaned = _clean_snippet(raw)
    assert cleaned == "Brock's Onix is a Rock/Ground & tough."


def test_web_search_empty_query():
    """Empty or whitespace queries return [] immediately."""
    assert web_search("") == []
    assert web_search("   \n\t  ") == []


# ── Requirement 3(e): Cache Hit Path ────────────────────────────────────────
def test_cache_hit_path(tmp_path):
    """(e) The cache hit path: second query loads from on-disk JSON cache without network call."""
    mock_resp_search = MagicMock()
    mock_resp_search.read.return_value = json.dumps({
        "query": {
            "search": [{"pageid": 60694, "title": "Brock", "snippet": "Pewter Gym Leader"}]
        }
    }).encode("utf-8")
    mock_resp_search.__enter__.return_value = mock_resp_search

    mock_resp_extract = MagicMock()
    mock_resp_extract.read.return_value = json.dumps({
        "query": {
            "pages": {
                "60694": {
                    "pageid": 60694,
                    "title": "Brock",
                    "fullurl": "https://bulbapedia.bulbagarden.net/wiki/Brock",
                    "extract": "Brock is the Gym Leader of Pewter City's Gym. He specializes in Rock-type Pokémon.",
                }
            }
        }
    }).encode("utf-8")
    mock_resp_extract.__enter__.return_value = mock_resp_extract

    # First call: cache miss, urlopen called twice
    with patch("urllib.request.urlopen", side_effect=[mock_resp_search, mock_resp_extract]) as mock_urlopen:
        res1 = web_search("Brock", cache_dir=tmp_path)
        assert len(res1) == 1
        assert res1[0]["title"] == "Brock"
        assert "Rock-type" in res1[0]["snippet"]
        assert res1[0]["url"] == "https://bulbapedia.bulbagarden.net/wiki/Brock"
        assert mock_urlopen.call_count == 2

    # Verify on-disk cache file was written
    cache_file = tmp_path / "cache.json"
    assert cache_file.exists()

    # Second call with same query (case-insensitive): cache hit, urlopen NOT called
    with patch("urllib.request.urlopen") as mock_urlopen2:
        res2 = web_search("brock", cache_dir=tmp_path)
        assert mock_urlopen2.call_count == 0
        assert res2 == res1


# ── Requirement 3(d): Network Error Handling ────────────────────────────────
def test_network_error_returns_empty_and_decision_happens(tmp_path):
    """(d) Network error -> [] from web_search, and planner decision still completes."""
    # 1. web_search handles URLError, HTTPError, TimeoutError by returning []
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        results = web_search("Geodude", cache_dir=tmp_path)
        assert results == []

    # 2. Planner receives empty results when web_search encounters an error, but still produces goal
    call_count = 0

    class ErrorHandlingClient:
        def chat(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "Geodude"}}}],
                    "raw_content": "",
                    "thinking": "Searching Bulbapedia for Geodude...",
                    "prompt_eval_count": 50,
                    "eval_count": 10,
                }
            else:
                return {
                    "content": {
                        "goal": "grind",
                        "target_level": 14,
                        "reason": "Search returned empty, fallback to leveling starter",
                        "note": "",
                    },
                    "raw_content": json.dumps({
                        "goal": "grind",
                        "target_level": 14,
                        "reason": "Search returned empty, fallback to leveling starter",
                        "note": "",
                    }),
                    "thinking": "No search results, proceeding with grind.",
                    "prompt_eval_count": 80,
                    "eval_count": 25,
                }

    with patch("lab.agent.web_search.web_search", return_value=[]):
        planner = LLMPlanner(tools=("web_search",), client=ErrorHandlingClient())
        status = {
            "badges": 0,
            "level": 8,
            "hp": 25,
            "max_hp": 25,
            "hp_frac": 1.0,
            "map_id": MAP_VIRIDIAN_CITY,
        }
        goal = planner(status)
        assert goal == "grind"
        assert getattr(goal, "target_level", None) == 14
        assert len(planner.last_decision_record["tool_calls"]) == 1
        assert planner.last_decision_record["tool_calls"][0]["query"] == "Geodude"
        assert planner.last_decision_record["tool_calls"][0]["titles"] == []


# ── Requirement 3(a): Tools Off Identical Payload ───────────────────────────
def test_tools_off_identical_request_payload_to_today():
    """(a) Tools off -> identical request payload to today (no tools, PLANNER_SCHEMA, base prompt)."""
    captured_payloads = []

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {
            "content": json.dumps({
                "goal": "grind",
                "target_level": 14,
                "reason": "Level up on Route 1",
                "note": "Standard pacing",
            })
        },
        "prompt_eval_count": 120,
        "eval_count": 30,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    def fake_urlopen(req, timeout=300.0):
        body = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(body)
        return mock_resp

    client = OllamaClient(host="http://localhost:11434")
    planner = LLMPlanner(client=client)  # default tools=()

    status = {
        "badges": 0,
        "level": 8,
        "hp": 25,
        "max_hp": 25,
        "hp_frac": 1.0,
        "map_id": MAP_VIRIDIAN_CITY,
    }

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        goal = planner(status)
        assert goal == "grind"

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]

    # Verify byte-for-byte identical structure to baseline
    assert "tools" not in payload
    assert payload["format"] == PLANNER_SCHEMA
    assert payload["model"] == "gemma4:26b"
    assert payload["think"] is True
    assert payload["keep_alive"] == "30m"
    assert payload["options"]["num_ctx"] == 16384

    # Verify system prompt has NO Bulbapedia tool sentence
    sys_msg = payload["messages"][0]["content"]
    assert sys_msg == SYSTEM_PROMPT
    assert "Bulbapedia" not in sys_msg
    assert "web_search" not in sys_msg

    # Verify decision record and stats tool calls
    assert planner.last_decision_record["tool_calls"] == []
    assert planner.total_tool_calls == 0
    assert planner.get_stats()["tool_calls"] == 0


# ── Requirement 3(b): Tool Call Reply -> One Search -> Accepted ─────────────
def test_tool_call_reply_one_search_then_final_decision():
    """(b) A tool_call reply -> one search, then final JSON decision accepted and logged."""
    turns = 0
    captured_messages = []

    class MultiTurnClient:
        def chat(self, **kwargs):
            nonlocal turns
            turns += 1
            captured_messages.append(list(kwargs["messages"]))

            if turns == 1:
                # Turn 1: model calls tool web_search
                assert kwargs.get("tools") is not None
                assert kwargs.get("schema") is None
                return {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "Brock Pewter Gym"},
                            }
                        }
                    ],
                    "raw_content": "",
                    "thinking": "Let me look up Brock's team.",
                    "prompt_eval_count": 110,
                    "eval_count": 20,
                }
            else:
                # Turn 2: model sees tool response, outputs schema-constrained decision
                assert kwargs.get("schema") == PLANNER_SCHEMA
                return {
                    "content": {
                        "goal": "gym",
                        "target_level": 14,
                        "reason": "Brock has Rock types; level 14 starter ready",
                        "note": "Onix and Geodude verified",
                    },
                    "raw_content": json.dumps({
                        "goal": "gym",
                        "target_level": 14,
                        "reason": "Brock has Rock types; level 14 starter ready",
                        "note": "Onix and Geodude verified",
                    }),
                    "thinking": "Brock uses Geodude lv12 and Onix lv14.",
                    "prompt_eval_count": 220,
                    "eval_count": 45,
                }

    mock_search_results = [
        {
            "title": "Brock",
            "url": "https://bulbapedia.bulbagarden.net/wiki/Brock",
            "snippet": "Brock uses Geodude and Onix at the Pewter Gym.",
        }
    ]

    with patch("lab.agent.web_search.web_search", return_value=mock_search_results) as mock_search:
        planner = LLMPlanner(tools=("web_search",), client=MultiTurnClient())
        status = {
            "badges": 0,
            "level": 14,
            "hp": 42,
            "max_hp": 42,
            "hp_frac": 1.0,
            "map_id": MAP_PEWTER_CITY,
        }
        goal = planner(status, pewter_healed=True)

        assert goal == "gym"
        assert goal.reason == "Brock has Rock types; level 14 starter ready"
        assert planner.llm_decisions == 1
        assert planner.fallback_decisions == 0

        # Verify search was executed with correct query
        assert mock_search.call_count == 1
        assert mock_search.call_args[0][0] == "Brock Pewter Gym"

        # Verify tool calls logging in decision record
        tool_records = planner.last_decision_record["tool_calls"]
        assert len(tool_records) == 1
        assert tool_records[0]["tool"] == "web_search"
        assert tool_records[0]["query"] == "Brock Pewter Gym"
        assert tool_records[0]["titles"] == ["Brock"]
        assert tool_records[0]["latency"] >= 0.0

        # Verify summary stats
        stats = planner.get_stats()
        assert stats["tool_calls"] == 1
        assert stats["total_tokens"] == (110 + 20 + 220 + 45)

        # Verify message conversation history in Turn 2 included role="tool"
        turn2_msgs = captured_messages[1]
        roles = [m["role"] for m in turn2_msgs]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles


# ── Requirement 3(c): Max Tool Calls Cap ────────────────────────────────────
def test_max_tool_calls_cap_is_respected():
    """(c) The max_tool_calls cap is strictly respected even if model continuously returns tool calls."""
    turns = 0

    class ContinuousToolCallerClient:
        def chat(self, **kwargs):
            nonlocal turns
            turns += 1
            if kwargs.get("tools") is not None:
                return {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": f"query_turn_{turns}"},
                            }
                        }
                    ],
                    "raw_content": "",
                    "thinking": f"Calling tool turn {turns}",
                }
            else:
                # Tools disabled, schema enforced
                return {
                    "content": {
                        "goal": "travel_to_pewter",
                        "target_level": 14,
                        "reason": "Max searches reached, proceeding with quest",
                        "note": "",
                    },
                    "raw_content": json.dumps({
                        "goal": "travel_to_pewter",
                        "target_level": 14,
                        "reason": "Max searches reached, proceeding with quest",
                        "note": "",
                    }),
                }

    mock_results = [{"title": "Test Title", "url": "https://example.com", "snippet": "Test snippet"}]

    with patch("lab.agent.web_search.web_search", return_value=mock_results) as mock_search:
        planner = LLMPlanner(tools=("web_search",), max_tool_calls=3, client=ContinuousToolCallerClient())
        status = {
            "badges": 0,
            "level": 14,
            "hp": 40,
            "max_hp": 40,
            "hp_frac": 1.0,
            "map_id": MAP_VIRIDIAN_CITY,
        }
        goal = planner(status)

        assert goal == "travel_to_pewter"
        assert mock_search.call_count == 3  # Exactly 3, not more!
        assert len(planner.last_decision_record["tool_calls"]) == 3
        assert planner.total_tool_calls == 3


# ── Timeline & Run JSON Integration Test ────────────────────────────────────
def test_gym_quest_timeline_and_summary_contain_tool_calls():
    """Verify tool calls are tracked in gym_quest timeline entries and planner stats."""
    class FakePlanner:
        def __init__(self):
            self.total_decisions = 0
            self.last_decision_record = {
                "source": "llm",
                "latency": 0.5,
                "prompt_tokens": 100,
                "eval_tokens": 20,
                "raw_output": "{}",
                "thinking": "",
                "rejection_reason": None,
                "tool_calls": [
                    {"tool": "web_search", "query": "Brock", "titles": ["Brock"], "latency": 0.12}
                ],
            }

        def __call__(self, status, target_level=15, pewter_healed=False, heal_threshold=0.35):
            self.total_decisions += 1
            return Goal("done", reason="Badge won", target_level=15)

        def get_stats(self):
            return {
                "type": "llm",
                "model": "gemma4:31b",
                "decisions": 1,
                "tool_calls": 1,
            }

    fake_planner = FakePlanner()

    with patch("lab.agent.gym_quest.PokemonRed") as MockEnv:
        mock_env_inst = MagicMock()
        mock_env_inst.step.return_value = (None, {"map_id": MAP_PEWTER_CITY})
        mock_env_inst.observe.return_value = (None, {"map_id": MAP_PEWTER_CITY, "x": 10, "y": 10, "in_battle": False})
        mock_mem = [0] * 65536
        mock_mem[0xD356] = 1  # Boulder Badge obtained
        mock_env_inst.pyboy.memory = mock_mem
        MockEnv.return_value.__enter__.return_value = mock_env_inst

        with patch("lab.agent.gym_quest.party_status", return_value={
            "level": 14, "hp": 40, "max_hp": 40, "hp_frac": 1.0, "party_size": 1,
            "map_id": MAP_PEWTER_CITY, "x": 10, "y": 10, "in_battle": False, "facing": 0, "badges": 1
        }):
            res = run_gym_quest(
                planner="llm",
                llm_planner=fake_planner,
                llm_tools=("web_search",),
                budget=500,
            )

            # Check timeline entry
            timeline = res["timeline"]
            assert len(timeline) >= 1
            assert "tool_calls" in timeline[0]
            assert len(timeline[0]["tool_calls"]) == 1
            assert timeline[0]["tool_calls"][0]["query"] == "Brock"

            # Check planner summary block
            assert "planner" in res
            assert res["planner"]["tool_calls"] == 1


def test_cli_flags_for_llm_tools():
    """Verify gym_quest CLI parser correctly accepts --llm-tools web_search."""
    from lab.agent.gym_quest import main

    with patch("sys.argv", [
        "gym_quest.py",
        "--planner", "llm",
        "--llm-model", "gemma4:26b",
        "--no-think",
        "--llm-tools", "web_search",
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
                assert kwargs["llm_tools"] == ("web_search",)
