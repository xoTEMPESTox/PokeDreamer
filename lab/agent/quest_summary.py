"""Summarise gym_quest run directories (run_seed_*.json) into one row each, for the results pages.

Usage: python -m lab.agent.quest_summary work/harness_llm_s0 work/gym_learned_best_s0
"""
import json
import statistics
import sys
from pathlib import Path


def summarize(runs: list) -> dict:
    """Badges, presses, blackouts, navigation fallback and planner decisions over a list of run dicts."""
    presses = [r["presses_total"] for r in runs]
    fallback = [sum(leg["fallback_presses"] for leg in r.get("legs", [])) for r in runs]
    # reached_by_policy is None for legs that did not run; same definition as v3_gym's "no fallback anywhere"
    # (work/agg_moves.py): a leg cut short by a blackout counts as not reached, with 0 fallback presses.
    legs = [leg for r in runs for leg in r.get("legs", []) if leg["reached_by_policy"] is not None]
    planners = [r.get("planner") or {} for r in runs]
    decisions = sum(p.get("decisions", 0) for p in planners)
    latency_sum = sum(p.get("mean_latency", 0.0) * p.get("decisions", 0) for p in planners)
    return {
        "runs": len(runs),
        "badges": sum(bool(r["badge_obtained"]) for r in runs),
        "median_presses": statistics.median(presses),
        "min_presses": min(presses),
        "max_presses": max(presses),
        "blackouts": sum(r["blackouts"] for r in runs),
        "median_level_at_gym": statistics.median(r["level_at_gym"] for r in runs if r["level_at_gym"] is not None),
        "all_legs_by_policy_runs": sum(
            all(leg["reached_by_policy"] for leg in r.get("legs", []) if leg["reached_by_policy"] is not None) for r in runs),
        "fallback_presses": sum(fallback),
        "legs_reached_by_policy": sum(bool(leg["reached_by_policy"]) for leg in legs),
        "leg_runs": len(legs),
        "decisions": decisions,
        "rejected_decisions": sum(p.get("fallbacks", 0) for p in planners),
        "mean_decision_seconds": latency_sum / decisions if decisions else 0.0,
    }


def main():
    for d in sys.argv[1:]:
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(d).glob("run_seed_*.json"))]
        print(d, json.dumps(summarize(runs)))


if __name__ == "__main__":
    main()
