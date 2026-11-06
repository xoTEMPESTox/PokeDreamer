# System 1/2 in the real game: level up and heal (measured)

This page evaluates an integrated System 1 / System 2 loop in the real Game Boy emulator executing an autonomous leveling and healing routine from Pallet Town to Route 1 and Viridian City. It examines the trade-off between grinding speed and blackout risk across varying health thresholds for triggering Pokémon Center visits. Across a 50-run sweep per threshold (seeds 200-249), blackout rates decrease monotonically as the heal threshold rises (from 1.76 per run at 0.2 to 0.38 at 0.8), while total button presses exhibit a U-shaped curve that optimizes around an HP threshold of 0.4 (3,257 mean presses).

**Reproduce:** No single Section 3 command reproduces this suite; run via `python -m lab.agent.system12`; see [lab/README.md §1](../README.md#1-directory--module-layout).

---

Code: `lab/agent/system12.py`. Start: `saves/starter_pallet.state` (Charmander L6 after the rival battle, Pallet Town; made by `python -m lab.make_starter_state` from the Pokémon Red ROM). No human input.

**What is what (honest labels):**
- **System 2 v0:** a rule-based planner over RAM: heal when the HP fraction is below a threshold; else grind until a target level; re-plans after every skill.
- **System 1 skills v0:**
  - greedy navigation with exits from recorded transitions;
  - a battle reflex (press A: FIGHT, first move);
  - grinding by pacing a Route 1 grass patch found by exploration ((13..14, 30));
  - a scripted Pokémon Center heal (door (23,26), nurse dialog).

  None of these skills is learned yet.

## First runs (target level 10, budget 4,000; `work/system12/`)
3/3 reached level 10 from level 6 in 1,646 / 1,673 / 1,690 presses. In runs 0 and 2 the planner chose "heal" at low HP (6/23, 8/25), but the walk north through Route 1's tall grass at low HP led to battles and a **blackout**, which returns the player home with full HP. So those runs used 0 heals and 2 blackouts each. System 2 decided correctly; the System 1 trek failed.

## Heal-threshold experiment (target level 12, budget 6,000, 10 runs each, seeds 100-109; `work/system12_h{0.35,0.5,0.65}/summary.json`)

| System 2 heals below | reached L12 | presses, mean ± sd | battles won / started | blackouts (runs with any) | heals succeeded / done |
|---|---|---|---|---|---|
| 0.35 | 10/10 | 3,326 ± 149 | 358 / 367 | 9 (7) | 24 / 24 |
| 0.50 | 10/10 | 3,535 ± 186 | 371 / 378 | 7 (5) | 35 / 35 |
| 0.65 | 10/10 | 3,604 ± 400 | 364 / 367 | 3 (3) | 44 / 44 |

Reading: healing earlier cuts blackouts by two thirds (9 to 3) for ~8% more presses. All 103 heals succeeded, and 1,093 of 1,112 wild battles were won (98%). It is a real System 2 trade-off (safety vs efficiency) that a context-aware planner should set per situation, e.g. heal earlier when the Pokémon Center is far or the route crosses tall grass.

Each run takes ~4 s of CPU (headless PyBoy), so planner variants can be compared over many runs cheaply.

## Heal-threshold sweep, 50 runs per threshold (fresh seeds 200-249, target L12, budget 6,000)
Raw: `work/system12_sweep/h*/summary.json`; summary with bootstrap 95% CIs: `work/sweep_cpu_summary.json`; script `work/sweep_cpu.sh`.

| heal below | reached L12 | presses, mean [95% CI] | blackouts per run [95% CI] | runs with ≥1 blackout | heals per run (all succeeded) |
|---|---|---|---|---|---|
| 0.2 | 46/50 | 3,312 [3,118, 3,547] | 1.76 [1.50, 2.02] | 46 | 1.1 |
| 0.3 | 47/50 | 3,329 [3,156, 3,545] | 1.24 [1.02, 1.46] | 41 | 1.8 |
| 0.4 | 50/50 | **3,257** [3,207, 3,313] | 0.88 [0.66, 1.10] | 33 | 2.6 |
| 0.5 | 50/50 | 3,320 [3,271, 3,369] | 0.60 [0.42, 0.78] | 26 | 3.3 |
| 0.6 | 50/50 | 3,438 [3,381, 3,492] | 0.54 [0.38, 0.70] | 24 | 3.9 |
| 0.7 | 50/50 | 3,565 [3,505, 3,623] | 0.46 [0.30, 0.64] | 19 | 4.7 |
| 0.8 | 50/50 | 3,788 [3,727, 3,853] | **0.38** [0.24, 0.52] | 18 | 6.0 |

Reading:
- Blackouts fall monotonically with the threshold.
- Presses are U-shaped. Healing too late costs blackouts, and some runs then miss the target: 4/50 and 3/50 runs ran out of budget at 0.2 and 0.3. Healing too early costs trips to the Pokémon Center.
- The press-optimal fixed rule is ~0.4; the safest is 0.8, for +16% presses.

A contextual System 2 should beat every fixed threshold, e.g. heal earlier when the Pokémon Center is far or the route has grass. This 50-run sweep serves as the primary result; the 10-run numbers are consistent with it.

## Learned vs greedy navigation inside the loop (10 runs each, target L12, heal < 0.5, seeds 100-109; `work/system12_nav_{greedy,learned}/summary.json`)
| System 1 navigation | reached L12 | presses | blackouts | heals |
|---|---|---|---|---|
| greedy (RAM) | 10/10 | 3,535 | 0.7 / run | 3.5 / run, all succeeded |
| learned (route1 + viridian policies) | 0/10 | 6,000 (budget) | 0 | 0 |

The greedy row reproduces the heal-0.5 row above exactly (3,535 presses, 7 blackouts): runs are deterministic per seed. The learned row fails because its Route 1 -> Viridian leg uses the initial unclipped viridian policy, which stalls at Route 1 row 28 (0/100 standalone, diagnosed in `v3_agent.md`). Runs reach L7, drop to 4/23 HP, choose "heal", and never reach the Pokémon Center.
