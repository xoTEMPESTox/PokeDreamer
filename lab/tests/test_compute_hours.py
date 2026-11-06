from lab.compute_hours import training_minutes

BASE = "23:26:37 DONE 900 iterations, 21.8 min; policy work\\agent_rl\\leg_s0\\policy.pt"


def test_resumed_extension_counts_only_the_increment_and_tee_copies_once(tmp_path):
    (tmp_path / "agent_rl" / "leg_s0").mkdir(parents=True)
    (tmp_path / "agent_rl" / "leg_it1800_s0").mkdir(parents=True)
    (tmp_path / "agent_rl" / "leg_s0" / "train.log").write_text(BASE + "\n")
    # the extension's log starts as a copy of the base log, then resumes from its checkpoint (cumulative minutes)
    (tmp_path / "agent_rl" / "leg_it1800_s0" / "train.log").write_text(
        BASE + "\n09:06:21 resumed at iteration 900\n09:28:07 DONE 1800 iterations, 43.5 min; policy x\n")
    (tmp_path / "chain_leg.log").write_text(BASE + "\n")  # tee copy
    mins = training_minutes(tmp_path)["imagination-RL policies"]
    assert [round(m, 1) for m in mins] == [21.8, 21.7]


def test_crash_resume_without_an_earlier_end_line_counts_the_full_run(tmp_path):
    (tmp_path / "runs" / "wm").mkdir(parents=True)
    (tmp_path / "runs" / "wm" / "train.log").write_text(
        "09:41:56 resumed at step 4200, 15.1 min trained\n11:58:01 DONE step 114800, 150.1 min; final rollout\n")
    assert training_minutes(tmp_path)["world model (diffusion)"] == [150.1]
