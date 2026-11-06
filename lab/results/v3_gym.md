# First gym: Boulder Badge (Brock) in the real game by a System 1/2 agent (measured)

This page reports the end-to-end evaluation of PokeDreamer v3 on earning the Boulder Badge (defeating Gym Leader Brock) in the real Game Boy emulator without human intervention. Starting from Charmander L8 in Viridian City, the agent orchestrates grinding, healing, route traversal, and gym battles. In the headline configuration with all 10 navigation legs driven by policies trained entirely inside the diffusion world model's imagination, the agent achieves 30/30 badges across three random seeds, with 14 of 30 quests completing the entire journey from Viridian City through Brock without triggering privileged pathfinding fallback anywhere.

**Reproduce:** Reproducible via [lab/README.md §3.1](../README.md#31-brock-gym-quest-1010-badges-with-bfs-navigation) (BFS navigation baseline), [lab/README.md §3.2](../README.md#32-brock-gym-quest-with-5-learned-world-model-travel-legs) (5 travel legs), [lab/README.md §3.3](../README.md#33-brock-gym-quest-with-all-10-learned-legs-5-travel--5-movement) (all 10 learned legs), and [lab/README.md §3.4](../README.md#34-standalone-travel-leg-real-game-evaluation-viridian-forest) (Viridian Forest standalone eval).

**Figures:**
- `work/figures/fig_quest_presses.png`: Distribution of total GameBoy button presses to obtain the Boulder Badge across quest policy configurations: rules + path-finding baseline (grey), 5 travel legs learned (gateall), and all 10 legs learned (first and best retrains), with individual 10-run evaluations shown as strip points and seed medians marked.
- `work/figures/fig_leg_paths.png`: Real-game evaluation traces (20 random episodes per leg, with ±0.25 tile jitter and alpha 0.35) of the learned navigation policies on each leg's primary traversal map, showing observed training tiles (light grey), target exits (dark), successful paths (green), and failures (red).
- `work/figures/fig_leg_failures.png`: Real-game failure traces (20 episodes each, with ±0.25 tile jitter and alpha 0.35) from weaker policy seeds: (left) Viridian Forest seed 1 (69/100): 31 failures, 15 of them ending at (25,12) beside the Antidote item ball at (25,11), a spot the training data barely covers; (right) to_pewter_pc seed 1 (0/100) reaching the Pokémon Center door (13,26) but repeatedly failing to press up.
- `work/figures/fig_leg_success.png`: Evaluation success rate (/100) and median steps across seeds 0, 1, and 2 of the learned gate policies compared against the greedy geodesic baseline (hollow markers).
- `work/figures/fig_door_warp.png`: Imagined vs real map-change rates per action across Pokémon Center door and mat tile groups; cells outlined in red indicate imagined map changes that never happen in the real game (imagined − real ≥ 0.3), which erroneously paid training reward.

---

Code: `lab/agent/gym_quest.py`, `lab/agent/nav.py`. Start: `saves/pokedex_viridian.state`, made from the Pokémon Red ROM by `python -m lab.make_pokedex_state`: Viridian City, Charmander L8 at 14/25 HP, Pokédex, road north open. No human input.

**Honest labels (what is and is not learned):**

| part | how it works | learned? |
|---|---|---|
| System 2 | rule planner over RAM status: heal (start, before the forest, in Pewter, HP < 35%), grind on Route 1 to L14, travel to Pewter, gym | no, rules |
| System 1: navigation | BFS over the current map's walkability read from game memory and ROM (map blocks, tileset collision list via `wTilesetCollisionPtr` 0xD530, NPC sprites); warp tiles from the pokered disassembly | no, privileged planner |
| System 1: battle | FIGHT menu by reading the cursor tile; move by rule: Ember vs rock/ground/bug types or Defense ≥ 20 > Special, otherwise Scratch (saves Ember PP); A through evolution | no, rules |
| world model | not used in this quest | n/a |

**Results (10 runs, seeds 100-109, budget 18,000 presses, target L14; `work/gym_rng/run_seed_*.json`, `summary.json`, log `work/gym_rng.log`):**

| badge | presses mean ± sd [min, max] | blackouts | battles won | level at gym / after |
|---|---|---|---|---|
| **10/10** | 8,791 ± 696 [8,039, 10,240] (~59 min of game time) | 1 in total (seed 103, recovered) | 100% (42-56 per run) | 14 / 16 (Charmeleon) |

Presses per phase (mean over runs):
- grind: 5,932 (67%);
- heal: 1,249;
- travel Viridian -> Pewter: 627, including the Viridian Forest maze;
- gym: 983 (Jr. Trainer + Brock).

**Run-to-run variation, an important caveat found in review.** The game is deterministic for a fixed input sequence, and the agent uses no randomness. The initial evaluation (`work/gym/`) therefore replayed one trajectory 10 times: 8,664 presses each, identical battles. Fix: the seed sets 0-599 idle frames before the first press. The game's RNG advances every frame, so encounters, damage rolls and catches differ. The 10/10 above is with this fix: 10 different trajectories.

**Independent verification:**
- Replayed seeds 100 (pre-fix) and 103 (post-fix, the blackout run) with every emulator's `pyboy` wrapped in a proxy that raises on any memory write.
- Both runs ended in the Pewter Gym (map 54) with `wObtainedBadges` (0xD356) bit 0 = 1, read directly from RAM. Lead Pokémon L16 (species 178 = Charmeleon). 0 write attempts.
- Seed 103 reproduced exactly (8,556 presses, 1 blackout).
- Static test `test_zero_memory_writes_enforced` (AST scan of both modules).
- GIF: `work/gym_rng/run_seed_100.gif`, 1.1 MB, captioned with the System 2 goal.

**What this shows, and what it does not:**
- It shows the full System 1/2 loop (plan, navigate, battle, heal, recover from a blackout) beating the first gym end to end in the real game, reliably.
- It does **not** show learning. Navigation reads the map from memory, and battles and plans are rules. The learned navigation legs are evaluated in the sections below.

## With world-model-trained travel legs (measured)
Same quest, seeds 100-109, rules planner, `gym_quest --nav learned --leg-policies ...`.
- Each travel leg from Viridian to the gym door (to_route2, to_gate, forest, to_pewter, to_gym) is driven by a policy trained only in wm31's imagination. It sees frames, not RAM.
- If the policy hasn't reached the leg's target within 1,000 presses, the BFS navigator finishes the leg. This is logged per leg.
- Still rules/RAM: grind and heal movement (BFS), battles, the planner. The viridian leg is skipped because the quest starts in Viridian.

| leg policies | badges | median presses [min, max] | blackouts | to_route2 by policy | forest by policy | other legs by policy | runs with no fallback | raw |
|---|---|---|---|---|---|---|---|---|
| step clip, seed 0 | 10/10 | 10,687.5 [9,457, 13,224] | 5 | 0/11 | 8/11 | all | 0/10 | `work/gym_learned` |
| step clip, seed 1 | 10/10 | 10,287.5 [9,585, 12,114] | 0 | 0/10 | 8/10 | all | 0/10 | `work/gym_learned_s1` |
| step clip, seed 2 | 10/10 | 9,888 [8,659, 11,403] | 5 | 12/12 | 6/12 | all | 4/10 | `work/gym_learned_s2` |
| **gate to_route2**, seed 0 | 10/10 | 9,547 [8,510, 10,658] | 4 | **11/11** | 7/11 | all | **6/10** | `work/gym_learned_gate_s0` |
| **gate to_route2**, seed 1 | 10/10 | 9,337 [8,332, 12,573] | 6 | **11/11** | 9/11 | all | **8/10** | `work/gym_learned_gate_s1` |
| **gate to_route2**, seed 2 | 10/10 | 9,364 [8,797, 11,343] | 1 | **10/10** | 2/10 | all | 2/10 | `work/gym_learned_gate_s2` |
| **gate to_route2 + gate forest**, seed 0 | 10/10 | 9,095.5 [8,421, 11,428] | 4 | 11/11 | **11/11** | all | **10/10** | `work/gym_learned_gate2_s0` |
| **gate to_route2 + gate forest**, seed 1 | 10/10 | 9,453.5 [8,677, 11,624] | 5 | 10/10 | 7/10 | all | **7/10** | `work/gym_learned_gate2_s1` |
| **gate to_route2 + gate forest**, seed 2 | 10/10 | 9,027.5 [8,387, 10,681] | 1 | 10/10 | 9/10 | all | **9/10** | `work/gym_learned_gate2_s2` |

The "gate to_route2" rows swap only the to_route2 policy for the `--move-gate 2` retrain (`v3_tile_sweep.md`). The "+ gate forest" rows also swap the forest policy for its `--move-gate 2` retrain (`work/agent_rl/forest_wm31_gate_s*`, 1,780 iterations). to_gate, to_pewter and to_gym are the step-clip policies of that seed. Legs can run more than 10 times because a blackout restarts them.

**Reading.**
- With the gate to_route2 policies, the leg out of Viridian is driven by its policy in 32/32 attempts. The step-clip seeds 0/1 managed 0/21.
- 16 of 30 quests now travel Viridian → gym door entirely on world-model policies.
- Every remaining fallback is in Viridian Forest, and all came from running out of the 1,000-press budget, none from blackouts.
- Seed 2's forest policy (47/100 in its own eval) is the weakest. Cause (diagnosed from `work/agent/diag_forest.log`):
  - It presses "up" at (25,12) into the Viridian Forest Antidote item ball, which blocks (25,11) until it is picked up. The quest and eval states never pick it up. No eval episode of any policy, nor the BFS path-follower, ever stands on (25,11) (about 1,000 episodes); seeds 0/1 and BFS go round via (26,11).
  - In the recorded data, only 3 of 104 forest episodes visit (25,12). In 2 of them the player picks the item up ("found ANTIDOTE!" frame), then walks through (6 of 12 recorded "up" presses moved).
  - wm31 imagines "up" from (25,12) succeeding in 25/39 real contexts. On the identical frames of a blocked press, policy seed 2 keeps P(up) ≈ 1 for 986 presses.
  - So this is a rare, state-dependent obstacle that the data barely covers, not reward hacking: the stationary press earns only the step cost under either reward.

**Forest retrained with the gate (measured).**
- Own eval (100 episodes, seed 200, cap 1000; `work/agent/eval_forest_wm31_gate_s*.log`): seeds 0/1/2 **100 / 69 / 97** (median 560 / 626 / 677 presses). The step-clip seeds scored 92 / 97 / 47.
- Totals are 266 vs 236 of 300. With 3 seeds that is no evidence either reward is better for forest.
- The Antidote stall moved between seeds instead of disappearing. It accounts for 16 of the 31 gate seed-1 failures, 32 of 53 for step-clip seed 2 and 2 of 8 for step-clip seed 0. That fits the data-coverage diagnosis.
- In the quests, the gate forest policies drive the forest leg 27/31 times. **26/30 quests travel Viridian → gym door with no fallback, including all 10 of seed 0** (`work/gym_learned_gate2_s0`: 0 fallback presses).
- Still rules/RAM: grind and heal movement, battles, the planner.
- Caveat: gate seed 2 trained in bursts between pauses. Its iteration count, 1,780, is unchanged.

## All five travel legs on the gate reward (measured)
to_gate, to_pewter and to_gym retrained with `--move-gate 2`, seeds 0-2, with the same iteration budgets as the step-clip runs (`work/chain_gate_legs.sh`, then `work/chain_resume_1105.sh`).

Own evals (100 episodes, seed 200, cap 1000; `work/agent/eval_<leg>_wm31_<variant>.log`): successes/100 (median presses), seeds 0/1/2.

| leg | step clip | gate |
|---|---|---|
| to_gate | 100/100/100 (233/207/94) | 100/100/100 (94/94/94) |
| to_pewter | 100/100/100 (28/29/28) | 100/100/100 (28/28/28) |
| to_gym | 100/100/97 (48/46/48) | 100/100/100 (48/48/48) |
| to_gym from the quest's arrival tile | n/a | 100/100/100 (44/44/44) |

Gym quest with all five gate legs (`work/gym_learned_gateall_s{0,1,2}`). The grind and heal movement is BFS, and the battles and the planner are rules.

| leg policies | badges | median presses [min, max] | blackouts | forest by policy | other travel legs by policy | runs with no travel fallback |
|---|---|---|---|---|---|---|
| all 5 gate, seed 0 | 10/10 | 9,075.5 [7,819, 10,785] | 4 | 11/12 | all | 9/10 |
| all 5 gate, seed 1 | 10/10 | 9,598.5 [8,355, 11,226] | 6 | 11/13 | all | 8/10 |
| all 5 gate, seed 2 | 10/10 | 9,051 [8,407, 11,351] | 5 | 12/14 | all | 8/10 |

- **25/30 quests travel Viridian → gym door on world-model policies with no fallback.** Every fallback is in the forest.
- The gate gives the same success as the step clip on these three legs. It takes the shorter route on to_gate (median 94 presses vs 233/207 for step-clip seeds 0/1).
- Counting rule: leg records with `reached_by_policy` None (already at the target, 0 presses) are skipped, not counted as misses.
- Note on seed 2: The initial seed-2 quest (kept in `work/gym_learned_gateall_s2_partial_to_grass`) loaded a partially trained `to_grass` checkpoint through default fallback while that leg was training, which inflated the grind presses (mean 8,670 vs 5,932). The final re-run passes the five movement legs as BFS, matching seeds 0 and 1. To reproduce any quest in this file now that movement policies exist, pass every leg explicitly.

## Learned grind and heal movement: all 10 navigation legs (measured)
Five movement legs trained in imagination with `--move-gate 2`, seeds 0-2 (`work/chain_moves.sh`).
- Iterations: to_grass and to_viridian_pc 1,800; to_pewter_pc 900; the two leave legs 600.
- Own evals (100 episodes, seed 200, cap 1000; `work/agent/eval_<leg>_wm31_gate_s<S>.log`): successes/100 (median presses), seeds 0/1/2.
- Greedy-geodesic baseline: `work/agent/<leg>_greedy_geo_s200.json`.

| leg | seed 0 | seed 1 | seed 2 | greedy |
|---|---|---|---|---|
| to_grass | 100 (53) | 100 (134) | 100 (51) | 100 (118) |
| to_viridian_pc | 100 (151) | 100 (148) | 100 (151) | 100 (241) |
| leave_viridian_pc | 100 (5) | 100 (4) | 100 (5) | 100 (8) |
| to_pewter_pc | 32 (369.5) | **0** | 64 (19) | 100 (18) |
| leave_pewter_pc | 72 (286) | 68 (408.5) | 100 (101) | 100 (8) |

Gym quest with all 10 navigation legs on world-model policies (`work/gym_learned_moves_s{0,1,2}`, summarised by `work/agg_moves.py`).
- Each leg falls back to BFS if its policy misses. Battles and the planner are rules.
- Heal budget: each heal step gets the whole budget (the initial seed-0 run with a shared budget is kept in `work/gym_learned_moves_s0_sharedbudget`; median 11,406.5; 13 starved leave legs).

| seed | badges | median presses [min, max] | blackouts | to_pewter_pc by policy | leave_pewter_pc by policy | forest by policy | world-model press share | runs with no fallback anywhere |
|---|---|---|---|---|---|---|---|---|
| 0 | 10/10 | 10,637 [9,769, 11,768] | 5 | 8/18 | 11/18 | 10/11 | 0.428 | 1/10 |
| 1 | 10/10 | 11,400 [9,682, 12,517] | 3 | 0/17 | 12/17 | 6/10 | 0.461 | 0/10 |
| 2 | 10/10 | 9,260.5 [8,457, 11,144] | 5 | 16/21 | 16/16 | 10/10 | 0.376 | **6/10** |

- **30/30 badges. In 6 of seed 2's 10 quests, every navigation leg (grind, heal, travel, gym) is driven by a policy trained only in imagination.**
- All other legs reach their target by policy every time, except one to_viridian_pc miss in seed 1. The BFS share of presses is ≤ 0.010. The remaining presses are battles, dialogue and the counter walk.
- **Why the Pewter PC legs fail** (diagnosis; `work/eval/door_warp_wm31.json`, `lab/dwm/door_warp.py`). The imagination reward pays for two kinds of misread; the second explains leave_pewter_pc (see the retrains below):
  - **On the door tile:** up enters in imagination 100% of the time and the other buttons ≤ 6%. The policies almost never press it there: seed 1 never, seed 0 32 times in ~18.9k presses (eval traces).
  - **Beside the door:** imagined entries happen after up from (12,26)/(14,26) (65-66%) and after down from (13,27) (80%). Real rate: 0%. The reward paid +5 for any map change within one tile of the exit.
  - **On the mat:** inside the Pewter PC, right on the mat (4,7) gives an imagined "exit" 66% of the time. 82/84 of these are the probe reading the frame as the Viridian PC (41); all Centers share one interior. The probe reads 23% of held-out real Pewter PC frames as 41 (`work/eval/probe_pc_confusion.log`).
  - **Seed 2's to_pewter_pc misses are a different failure:** down at the start tile (18,35) leaves Pewter onto Route 2, then battles and an apparent blackout end in Viridian.
- **Fixes:**
  - `--warp-dist 0` and `--merge-centers`.
- **to_pewter_pc is not explained by the misreads:**
  - `--warp-dist 0` gives s0 22/100 and s1 0/100.
  - An imagined-rollout audit near the door (`work/eval/door_rollout_audit.log`) shows the s1 policy also steps sideways in imagination (up 1 of 86 presses on the door tile), and there it earns less than s2: return 1.11 vs 3.63.
  - Up on the door tile enters in imagination 100% of the time from every arrival direction (`work/eval/door_arrival_wm31.log`).
  - So this is a local optimum of the policy optimisation, not a sim-to-real gap.
- **Retrains, one change per leg, same iterations (`work/chain_pewter_fix.sh`; `work/agent/eval_<leg>_wm31_gate_{wd0,mc}_s<S>.log`):**

| leg | old reward s0/s1/s2 | fixed s0/s1/s2 |
|---|---|---|
| leave_pewter_pc, `--merge-centers` | 72 (286) / 68 (408.5) / 100 (101) | **100 (5) / 100 (4) / 100 (4)** |
| to_pewter_pc, `--warp-dist 0` | 32 (369.5) / 0 / 64 (19) | 22 (219) / 0 / 0 |

  - The probe-confusion fix works on every seed and makes the leg as short as leave_viridian_pc (median 4-5; greedy 8).
  - The warp-distance fix doesn't help to_pewter_pc and is probably worse, so the old reward stays for that leg.
  - **to_pewter_pc was under-trained.** The old-reward control s0, continued from 900 to 1,800 iterations, scores **100/100, median 15** (greedy 18; `work/agent/eval_to_pewter_pc_wm31_gate_it1800_s0.log`). At 900 iterations the policy sat in the local optimum shown by the rollout audit; more training leaves it. **s1 at 1,800 is still 0/100** (`work/agent/eval_to_pewter_pc_wm31_gate_it1800_s1.log`; 64 episodes end on the door tile). **s2 at 1,800: 100/100, median 64.5.** At 1,800 iterations to_pewter_pc scores 100 / 0 / 100 (900: 32 / 0 / 64): two of three seeds leave the local optimum with more training.
  - A prior expectation that additional iterations would not resolve the stall was disproven: 1,800 iterations reached 100/100 on seeds 0 and 2.
  - **leave_pewter_pc is the opposite case.** The old-reward control s0 at 1,200 iterations gets 87/100, median 269 (`work/agent/eval_leave_pewter_pc_wm31_gate_it1200_s0.log`), against 100/100, median 5 for `--merge-centers` at 600. More training doesn't remove the mat-loitering that the probe misread rewards.

### Best leg set (measured)
Leg set: to_pewter_pc continued to 1,800 iterations (old reward) and leave_pewter_pc `--merge-centers`; the other 8 legs as in the moves quests. All 10 legs are passed explicitly. Runs: `work/gym_learned_best_s{0,1,2}`, script `work/chain_pewter_1800.sh`, summary `work/agg_moves.py`.

| seed | badges | median presses [min, max] | blackouts | to_pewter_pc by policy | leave_pewter_pc by policy | forest by policy | BFS press share | runs with no fallback anywhere |
|---|---|---|---|---|---|---|---|---|
| 0 | 10/10 | 8,779.5 [8,074, 10,572] | 10 | 18/18 | 18/18 | 10/11 | 0.000 | **8/10** |
| 1 | 10/10 | 10,587.5 [9,160, 11,770] | 1 | 0/17 | 16/16 | 7/11 | 0.010 | 0/10 |
| 2 | 10/10 | 8,934.5 [8,362, 9,846] | 3 | 16/16 | 16/16 | 7/11 | 0.002 | 6/10 |

- **30/30 badges. In 14 of 30 runs every navigation leg is driven by a world-model policy** (in the moves quests: 7/30).
- The remaining misses are seed 1's to_pewter_pc (0/100 even at 1,800 iterations), the forest (24/33), and 2 of 138 to_viridian_pc legs.
- Seed 0 had 10 blackouts against 5 in its moves quest. Battles are governed by scripted rules; individual encounter damage rolls account for the blackout variation.
