# System 2 bake-off: local LLM planners vs the rule planner (measured)

This page presents a benchmark evaluating whether local Large Language Models (LLMs) can replace the rule-based System 2 planner in the end-to-end gym quest to earn the Boulder Badge. Six open-weight models (running via Ollama with 16384 context and JSON schema validation against RAM) are tested across 10 fixed seeds (100-109) with privileged BFS navigation. Five of the six models match the rule baseline at 10/10 badges, with `gemma4:31b` achieving zero rejected decisions, 10/10 badges, and 7,239 median presses (vs 8,508.5 for the rules), while `gemma4:12b` achieves 10/10 badges within a 10.2 GB peak GPU memory footprint.

**Reproduce:** Reproducible via [lab/README.md §3.13](../README.md#313-system-2-llm-planner-bake-off) (`python -m lab.agent.gym_quest --runs 10 --seed 100 --target-level 14 --nav bfs --planner llm --llm-model gemma4:31b --work-dir work/sys2_bakeoff/gemma4_31b`).

---

**Question.** Can a local LLM replace the rule planner (System 2) in the System 1/2 gym quest and still earn the Boulder Badge? What does each model cost in VRAM and time?

**Protocol** (declared before the runs; identical for every model):
- Start state: Pokédex save (Charmander L8).
- Quest: grind → heal → Viridian Forest → Pewter → gym.
- Seeds 100-109, 10 quests per model, budget 18,000 presses.
- Navigation is BFS (reads the map from RAM), so the planner is the only variable.
- Ollama, thinking on, num_ctx 16384, JSON-schema output.
  - Every decision is validated against RAM.
  - A rejected decision (malformed, or "hallucinated": e.g. gym outside Pewter, done without badge, an unexecutable goal) falls back to the rule planner for that step.
- `--target-level 14` is only the starting value; each LLM picks its own grind target.
- Rule-planner reference: `work/gym_rng` (same seeds and save).
- Code: `lab/agent/llm_planner.py`, `lab/agent/gym_quest.py` (`--planner llm`).

## Results

| planner | badges | median presses | min-max | own decisions | rejected (kind) | blackouts | level at gym (median) | s / decision | peak GPU mem |
|---|---|---|---|---|---|---|---|---|---|
| rules (reference) | 10/10 | 8,508.5 | 8,039-10,240 | – | – | 1 | 14 | – | – |
| gemma4:26b (MoE) | **8/10** | 11,624 (9,930.5 over its 8 badge runs) | 6,362-18,000 | 113/115 | 2 malformed | 8 | 15 | 8.0 | 20.9 GB |
| qwen3.6:35b-a3b (MoE) | 10/10 | 8,355 | 5,155-9,952 | 79/97 | 18 malformed | 17 | 14 | 34.6 | 22.4 GB |
| gpt-oss:20b | 10/10 | **6,233** | 2,377-15,290 | 85/97 | 8 malformed, 3 unexecutable, 1 gym outside Pewter | 14 | 12 | 15.3 | 14.4 GB |
| **gemma4:31b** | **10/10** | 7,239 | 6,087-8,265 | **89/89** | **0** | 3 | 13 | 13.0 | 22.4 GB |
| gemma4:12b | 10/10 | 8,145 | 5,351-9,713 | 83/90 | 7 malformed | 4 | 14 | 11.7 | **10.2 GB** |
| qwen3.5:9b | 10/10 | 10,246.5 | 9,937-14,561 | 102/107 | 5 malformed | **2** | 15 | 45.4 | **7.8 GB** |

Raw data:
- per-run JSON `work/sys2_bakeoff/<model>/run_seed_10X.json` (key `planner`: decisions, llm_decisions, hallucinations, mean_latency, notebook);
- logs `work/sys2_bakeoff/<model>.log`;
- GPU memory `work/sys2_bakeoff/vram.log`.

Definitions:
- **Median presses** is `statistics.median` over all 10 runs for every model. A failed run counts at the press where it stopped:
  - gemma4:26b seed 102 used the whole 18,000 budget;
  - seed 103 was stopped by the stall guard at 15,427 (5 decisions in a row used ≤ 1 press).

  Both rank above the 5th and 6th runs, so the median (11,624) does not depend on how a failure is counted. Corrected 2026-11-07: this page earlier gave 9,930.5 (badge runs only) and said both failures used the whole budget.
- **s / decision** is the per-run mean latency, weighted by each run's decision count.
- **Peak GPU memory** is the highest total `memory.used` sampled every 30 s while the model was loaded. It includes ~1.2 GB used by the desktop with no model loaded. It covers the planner only; no world model was loaded during these runs.

## What it shows

- **Every model except gemma4:26b matched the rules on badges (10/10).** gemma4:26b failed twice: it kept raising its own grind target (10→13→15→17→20). One run hit the 18,000 budget; the stall guard stopped the other at 15,427.
- **gemma4:31b is the cleanest planner.**
  - Every one of its 89 decisions passed validation.
  - It used the fewest presses of any 10/10 model with low blackouts: 3 blackouts vs 1 for the rules, 7,239 vs 8,508.5 presses.
  - The cost: it needs ~22 GB, so it cannot share the 24 GB card with the world model.
- **gemma4:12b is the practical candidate for planner + world model on one GPU.**
  - 10/10 badges, 8,145 presses, 4 blackouts, 11.7 s per decision, ~10 GB peak.
  - 7 of its 90 decisions were rejected as malformed and handled by the rules.
- **qwen3.5:9b is the safest small model but the slowest.** 2 blackouts in 10, ~1,700 more presses than the rules, and 45 s per decision, because it thinks at length.
- **gpt-oss:20b is fast but careless.** It has the lowest median presses, because it goes to the gym at level 12. It also has 14 blackouts and 12 rejected decisions, including one attempt at the gym from outside Pewter.
- **qwen3.6:35b-a3b relied on the fallback the most.** 18 of its 97 decisions (19%) were malformed. Its thinking used ~14.8k of the 16,384-token context, leaving no room for the JSON answer. Under this equal protocol it is reported as "79 of 97 decisions its own". A 32k-context rerun is planned only if it becomes the chosen planner.
- **The LLMs black out more than the rules.** Only qwen3.5:9b (2) is close to the rules' 1. Except gpt-oss (median level 12 at the gym), the LLMs are as levelled as the rules (13-15 vs 14). The hypothesis is that the extra blackouts come from when they choose to heal: the rule planner heals at an explicit HP threshold, and an LLM only learns that threshold from the prompt.

## Full harness: LLM System 2 + dream-trained System 1 (10 runs, measured 2026-11-07)

Both halves are swapped in:
- System 2: `gemma4:31b`.
- System 1: the ten world-model navigation legs (`--nav learned --leg-policies` from `work/best_s0_legs.txt`, the set behind the 30/30 headline in [v3_gym](v3_gym.md)).

Everything else matches the rule-planner quests with the same legs: seeds 100-109, the same save state, target level 14 (`lab/experiments/quests.sh`, `work/gym_learned_best_s0`). The third column is the bake-off row above: the same LLM with map-reading (BFS) navigation.

| | **LLM planner + dream-trained legs** | rule planner + same legs | LLM planner + BFS navigation |
|---|---|---|---|
| badges | **10/10** | 10/10 | 10/10 |
| median presses [min, max] | **7,144.5** [6,800, 7,868] | 8,779.5 [8,074, 10,572] | 7,239 [6,087, 8,265] |
| blackouts (total) | 5 | 10 | 3 |
| median level at gym | 13 | 14 | 13 |
| runs with every leg driven by its world-model policy | **9/10** | 8/10 | – |
| leg runs reached by policy | 224/225 | 229/231 | – |
| navigation presses by the BFS fallback | **0** | 22 | all |
| planner decisions (rejected) | 99 (0) | 112 (–) | 89 (0) |
| mean decision time | 14.5 s | – | 13.0 s |

- **The full system wins every run, and with fewer presses than the rules on the same legs.**
  - The slowest LLM run (7,868) used fewer presses than the fastest rule run (8,074): no overlap in 10 vs 10.
  - The saving is mostly grinding: median 4,083.5 vs 5,495 presses. The LLM usually judges level 13 enough for Brock, while the rules grind to 14.
  - It also blacked out half as often (5 vs 10).
  - This is a planning difference on top of the same System 1, not better navigation: travel presses are similar (median 1,046 vs 999.5).
- **No navigation fallback in any LLM run**, and every decision passed validation (99/99).
  - The one leg not reached by policy (seed 102's `to_viridian_pc`) was cut short by a blackout, with 0 fallback presses.
  - That leg is why the run counts as 9/10 under v3_gym's "no fallback anywhere" definition (`work/agg_moves.py`).
- **The LLM's plans vary between runs of the same seed.** Seed 100 took 7,658 presses here and 7,045 in the video recording (below), with the same seed and legs. Single runs are anecdotes, which is why this was measured over 10.
- Same caveat as the bake-off: 10 runs per setup, no confidence intervals.

Raw data:
- runs: `work/harness_llm_s0/run_seed_10X.json`;
- chain log: `work/chain_harness_llm.log`;
- the table: `python -m lab.agent.quest_summary work/harness_llm_s0 work/gym_learned_best_s0 work/sys2_bakeoff/gemma4_31b` (saved in `work/harness_llm_s0/compare.txt`);
- per-phase medians from the same JSONs (`phase_presses`).

### The video's run (seed 100, one recording)
The video shows a separate recording of seed 100 (recorded with `--save-traj --save-color` for the video).

| | LLM planner (gemma4:31b) | rule planner, same legs and seed |
|---|---|---|
| badge | yes | yes |
| presses | 7,045 | 8,718 |
| level at gym | 14 | 14 |
| blackouts | 1 | 0 |
| planner decisions (rejected) | 8 (0) | 11 (–) |
| navigation presses by the BFS fallback | 0 | 0 |

- The LLM left Route 1 for Pewter at 18/36 HP and healed in Pewter; the rules heal in Viridian first.
- Of the LLM run's 20 leg runs, 19 reached their goal by policy; the 20th (`to_viridian_pc`) was cut short by the blackout.
- Mean decision time: 13.7 s.
- One run per planner. The comparison between planners is the 10-run table above.

A first attempt the same night omitted `--leg-policies`. With that flag missing, `gym_quest --nav learned` loads older default legs: to_route2 without the move gate, the 900-iteration to_pewter_pc, and leave_pewter_pc without merged centres. That run still won the badge (11,042 presses), but 4 legs fell back to BFS. It is kept in `work/video/quest_s100_llm_defaultlegs/` and not used.

Command: `python -m lab.agent.gym_quest --runs 1 --seed 100 --target-level 14 --nav learned --planner llm --llm-model gemma4:31b --leg-policies "$(cat work/best_s0_legs.txt)" --save-traj --save-color --traj-dir work/video/traj_llm --work-dir work/video/quest_s100_llm`.

Raw data:
- `work/video/quest_s100_llm/run_seed_100.json`;
- log `work/video/record_s100_llm.log`;
- rule run `work/video/quest_s100/run_seed_100.json`.

## Caveats

- 10 seeds per model, so differences of 1-2 badges or a few hundred presses are within noise. No confidence intervals were computed.
- Only System 2 is an LLM. Battles, BFS navigation and the grind/heal skills are the same code for every row, so the badge rate measures planning on top of a strong fixed System 1.
- The cause of each rejected decision was not logged per decision. Only the counts by kind are stored. The qwen3.6 cause above comes from its token counts.
- One prompt per model, no prompt tuning per model. Two prompt fixes were applied to every model before these runs: stating the grind skill's safety stop and stating the press budget. The aborted pre-fix runs are kept in `work/sys2_bakeoff_aborted1` and `work/sys2_bakeoff_aborted2`.
- Note: Press medians are computed from exact distribution medians and latency means are decision-weighted.
