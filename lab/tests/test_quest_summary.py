from lab.agent.quest_summary import summarize


def _run(presses, badge, fallback, by_policy, decisions=4, latency=10.0, rejected=0):
    return {
        "presses_total": presses, "badge_obtained": badge, "blackouts": 1, "level_at_gym": 14,
        "legs": [{"fallback_presses": fallback, "reached_by_policy": by_policy}],
        "planner": {"decisions": decisions, "mean_latency": latency, "fallbacks": rejected},
    }


def test_counts_fallback_runs_and_weights_latency_by_decisions():
    s = summarize([_run(7000, True, 0, True, decisions=2, latency=10.0),
                   _run(9000, True, 35, False, decisions=6, latency=20.0, rejected=1),
                   _run(18000, False, 0, True)])
    assert s["badges"] == 2 and s["median_presses"] == 9000
    assert s["all_legs_by_policy_runs"] == 2 and s["fallback_presses"] == 35
    assert s["legs_reached_by_policy"] == 2 and s["leg_runs"] == 3
    assert s["rejected_decisions"] == 1
    assert abs(s["mean_decision_seconds"] - (2 * 10 + 6 * 20 + 4 * 10) / 12) < 1e-9


def test_leg_cut_short_without_fallback_still_breaks_all_legs_by_policy_and_none_legs_are_ignored():
    cut_short = _run(7000, True, 0, False)  # e.g. a blackout ended the leg: not reached, 0 fallback presses
    not_run = _run(7000, True, 0, True)
    not_run["legs"].append({"fallback_presses": 0, "reached_by_policy": None})
    s = summarize([cut_short, not_run])
    assert s["all_legs_by_policy_runs"] == 1 and s["fallback_presses"] == 0 and s["leg_runs"] == 2
