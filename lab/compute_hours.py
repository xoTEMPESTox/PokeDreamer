"""Compute tally for v3 from the logs in work/ (python -m lab.compute_hours).

Counts each finished training run once, from the run time its log prints when it ends:
- diffusion world model (lab.dwm.train): "DONE step N, X min"
- imagination-RL policy (lab.agent.imagine_rl): "DONE N iterations, X min"
- upsampler: "final step N loss L elapsed X min"
- stage-2 colorizer (lab.color.learned): "DONE training N steps in X min"
The minutes are cumulative across resumes (checkpoints store the elapsed time), so when a log resumes after an earlier
end line (the it1200/it1800/it3560 extensions continue the base run), only the increment counts. The chain_*.log files
are tee copies of the per-run logs, so identical lines are counted once.
Also: the System 2 LLM quests (elapsed_seconds in each run JSON; the GPU serves the LLM) and the real-game PPO
baseline (CPU; last "(... steps/s, X min)" line of each log).
Not counted: evaluation, video/3D rendering, data collection, and runs that never printed an end line.
"""
import json
import re
from pathlib import Path

WORK = Path("work")
PATTERNS = [
    ("world model (diffusion)", re.compile(r"\d\d:\d\d:\d\d DONE step \d+, ([\d.]+) min")),
    ("imagination-RL policies", re.compile(r"\d\d:\d\d:\d\d DONE \d+ iterations, ([\d.]+) min")),
    ("upsampler", re.compile(r"\d\d:\d\d:\d\d final step \d+ loss [\d.]+ elapsed ([\d.]+) min")),
    ("colorizer (stage 2)", re.compile(r"\d\d:\d\d:\d\d DONE training \d+ steps in ([\d.]+) min")),
]
PPO_LINE = re.compile(r"\(\s*[\d.]+ steps/s,\s*([\d.]+) min\)")


def training_minutes(work=WORK):
    """{category: [minutes per run]} from the end lines of every log under work/ (hf downloads excluded)."""
    seen, out = set(), {name: [] for name, _ in PATTERNS}
    for log in sorted(work.rglob("*.log")):
        if "hf" in log.relative_to(work).parts[:1]:
            continue
        prev, resumed = None, False
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "resumed at" in line:
                resumed = True
            for name, pat in PATTERNS:
                m = pat.search(line)
                if m:
                    minutes = float(m.group(1))
                    if m.group(0) not in seen:
                        seen.add(m.group(0))
                        out[name].append(minutes - prev if resumed and prev is not None else minutes)
                    prev, resumed = minutes, False
    return out


def llm_quest_minutes(work=WORK):
    """Minutes of LLM-planned quests: elapsed_seconds of every run JSON under work/sys2_bakeoff*/."""
    runs = sorted(work.glob("sys2_bakeoff*/*/run_seed_*.json"))
    return [json.loads(p.read_text(encoding="utf-8"))["elapsed_seconds"] / 60 for p in runs]


def ppo_minutes(work=WORK):
    """Minutes of each real-game PPO run: the last progress line of work/ppo_real/*/train.log."""
    out = []
    for log in sorted(work.glob("ppo_real/*/train.log")):
        mins = PPO_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if mins:
            out.append(float(mins[-1]))
    return out


def main():
    rows = list(training_minutes().items())
    gpu_total = sum(sum(v) for _, v in rows)
    llm = llm_quest_minutes()
    ppo = ppo_minutes()
    print("| what | device | runs | hours |")
    print("|---|---|---|---|")
    for name, mins in rows:
        print(f"| {name} | GPU | {len(mins)} | {sum(mins) / 60:.1f} |")
    print(f"| **training total** | GPU | {sum(len(v) for _, v in rows)} | **{gpu_total / 60:.1f}** |")
    print(f"| System 2 LLM quests (bake-off, incl. aborted) | GPU (LLM) + CPU (game) | {len(llm)} | {sum(llm) / 60:.1f} |")
    print(f"| real-game PPO baseline | CPU | {len(ppo)} | {sum(ppo) / 60:.1f} |")


if __name__ == "__main__":
    main()
