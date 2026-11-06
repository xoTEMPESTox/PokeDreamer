# Start-tile robustness of to_route2 leg policies and the ledge trap (measured)

This page examines the start-tile generalization of `to_route2` navigation policies across 26 reachable tiles in Viridian City, diagnosing a ledge-jump trap and evaluating its potential-based fix. Policies trained in imagination with step clipping suffered seed-dependent looping around a ledge obstacle (seeds 0 and 1 scored 1/100 and 0/100 from quest start (23,26)), farming non-telescoping clipped rewards. Replacing step clipping with a potential-based move gate (`--move-gate 2`) completely eliminates the trap, achieving 2,339/2,340 (99.96%) successes across 26 start tiles and 100/100 on all three seeds from the quest arrival tile.

**Reproduce:** Reproducible via [lab/README.md §3.14](../README.md#314-start-tile-robustness-sweep-across-26-reachable-tiles) (`python -m lab.agent.tile_sweep --task to_route2 --policies to_route2_wm31_gate_s0 --n-tiles 24 --episodes 30 --workers 8 --seed 0`).

---

**Question.** `to_route2` policies (Viridian City → Route 2) trained in the world model score 1/100, 0/100 and 100/100 (seeds 0, 1, 2) from the tile where the gym quest starts the leg, (23,26). How does success depend on the start tile? Do the world-model policies differ from PPO trained in the real game?

## Protocol
- Tool: `lab/agent/tile_sweep.py`.
- Tiles: 24 tiles sampled from the 600 Viridian tiles reachable from the eval start, stratified by BFS path distance to the north exit (quartile edges 1/22/29/35/51, `np.random.default_rng(0)`). Plus the training/eval start tile (29,23) and the quest tile (23,26). No tile was skipped.
- Start states: walked by BFS from the task's eval state, plus 4 idle frames, position checked in RAM.
- Evaluation: the standard evaluator `lab.agent.mpc --policy learned --episodes 30 --max-steps 1000 --seed 200 --device cpu --state <tile>`.
- Policies (6):
  - world model: `work/agent_rl/to_route2_wm31{,_s1,_s2}`, trained in imagination with geodesic reward + `--step-clip 1`, 1800 iterations;
  - real-game PPO: `work/ppo_real/to_route2_geo_s{42,1,2}`, trained from the single eval state.
- Raw data:
  - `work/agent/sweep_to_route2/summary_6policies.csv`, `table_6policies.md`;
  - per-run JSON `work/agent/to_route2_learned_*_from_map1_<x>_<y>.json`.

## Results (26 tiles × 30 episodes per policy)

| policy | successes | tiles ≥95% | tiles <50% | where the failures end |
|---|---|---|---|---|
| wm31 seed 0 | 623/780 (79.9%) | 19/26 | 5 | looping at the ledge, x 18-19, y 26-28 |
| wm31 seed 1 | 508/780 (65.1%) | 16/26 | 9 | the same ledge loop (one tile: stuck at (24,14-15)) |
| wm31 seed 2 | **780/780 (100%)** | 26/26 | 0 | – |
| PPO seed 42 | 600/780 (76.9%) | 14/26 | 6 | west side: walks out to Route 22 |
| PPO seed 1 | 466/780 (59.7%) | 11/26 | 11 | Route 22, or stuck in the south-west corner (4,28) |
| PPO seed 2 | 763/780 (97.8%) | 23/26 | 0 | – |
| **wm31 + move gate, seed 0** | **780/780 (100%)** | 26/26 | 0 | – |
| **wm31 + move gate, seed 1** | **780/780 (100%)** | 26/26 | 0 | – |
| **wm31 + move gate, seed 2** | **779/780 (99.9%)** | 26/26 | 0 | 1 episode from (32,2): pressed left at (21,6) for 986 presses (not diagnosed) |

Raw data for all 9 policies: `work/agent/sweep_to_route2/summary.csv`, `table.md`, `run_gate.log`; the 6 step-clip/PPO rows are unchanged from `summary_6policies.csv`.

Family totals: world model with step clip 1,911/2,340 (81.7%), PPO 1,829/2,340 (78.2%), **world model with move gate 2,339/2,340 (99.96%)**. Between the step-clip and PPO families, with 3 seeds each, seed-to-seed variance dominates, so that comparison is **no evidence that either family generalizes better.**

Path length (17 tiles where step-clip s2, PPO s2 and all 3 gate seeds are 30/30). Median over tiles of (median presses − BFS distance):
- step-clip seed 2: 2.0;
- gate seeds 0/1/2: 3.0 each;
- PPO seed 2: 25.5.

PPO, trained from one start tile, wanders from the others before it finds the exit.

Note: An initial observation that imagination policies might be more brittle off their start tile than PPO was based on a single tile and disproven by the systematic 26-tile sweep.

## The two families fail for different reasons
- **PPO** trains from one real start state, (29,23). Its failures are in regions it never visited, the west of the city, where it takes the wrong exit (Route 22). That's a coverage failure.
- **World model:** imagination RL starts from recorded contexts anywhere in Viridian, 124,240 contexts on map 1. The failing area has some of the densest coverage: 5,000-12,000 contexts within 2 tiles of (23,26), (22,29) and (24,30). So it's not coverage.
  - Every failing episode of seeds 0 and 1 ends in the same 6-press loop (`work/agent/to_route2_learned_to_route2_wm31_s200_from_map1_23_26.json`, ~3,500 repeats over 30 episodes):
    - (18,26) –down→ (18,27) → (18,28), a ledge jump; RAM passes through (18,27) mid-jump;
    - → (19,28) → (19,27) → (19,26), walking back up through the gap in the ledge row at x=19;
    - → (18,26), and repeat.
  - Policy probabilities on **real** recorded frames at (18,26): P(down) = 0.995 (seed 0), 1.0 (seed 1), 0.001 (seed 2, which picks up with 0.97). Smoke run of `work/ledge_diag.py`.

## Cause: the per-step reward clip, not the world model
Full run, 64 recorded contexts per tile, on GPU with imagine_rl's exact sampling (`work/ledge_diag.py cuda 64` → `work/agent/ledge_diag.log`; the earlier 4-context smoke run is superseded). Diffusion sampling is unseeded. A rerun (`ledge_diag_regen.log`, `ledge_diag.json`) gives 43/64 for the landing, 27/64 for the (18,19) read and 51/64 for the (19,8) read; the counts below are from the first run:
- **wm31 imagines the jump landing correctly.** Two presses of "down" from (18,26) are read as (18,28) in 45 of 64 contexts, as in all 15 recorded real presses.
- **Its imagined mid-jump frame is often misread.** The probe reads the first imagined frame as (18,19) in 34/64 contexts, where RAM says (18,27).
  - On the 15 real mid-jump frames the probe is right 15/15, and 41/41 at the two neighbouring columns (scratchpad `probe_midjump.py`). So this is wm31's imagined frame, not the probe.
  - Imagined frames at (19,26) are read as (19,8) in 48/64 contexts after "right" from (18,26). Here the probe has a real ambiguity too: it reads 38/285 real frames at (19,26) as (19,8), frames from all splits.
- **The reward pays for the loop.** The real geodesic distance goes 26 → 31 (mid-jump tile) → 30, and the clip of ±1 tile per step turns that into −1, +1. The 4 steps back up each earn +1.
  - So the closed loop earns **+4 progress** (+3.7 after step costs) in the real game.
  - In imagination it pays too. Summing the mean clipped rewards of each segment from real contexts on its tile: jump −0.52, right at (18,28) +0.95, up twice at (19,28) +1.65, left at (19,26) +0.84, about **+2.9 per loop**.
  - Seeds 0 and 1 learned to farm it; seed 2 didn't.
- Clipping each step's progress is not potential-based (rewards don't telescope), so any loop with one large real drop pays.
- PPO's real-game reward (`ppo_real.py`) gates on |dx|+|dy| ≤ 2 instead of clipping. That telescopes, which is consistent with no PPO seed looping there.
- Other legs: revisits are 0-7% of moves and the most frequent moves aren't cycles (to_gate, forest, to_pewter, to_gym seed-0/1 evals). So the trap is specific to this ledge.

## Fix (implemented and measured)
`imagine_rl --move-gate 2`, with tests in `lab/tests/test_imagine_rl.py`:
- a probe-read tile counts only if it's within 2 tiles of the last accepted tile and has an observed path to the exit;
- progress is measured from that accepted tile;
- probe misreads earn nothing, which was the reason for the clip, and closed loops earn exactly 0. Tested on the real field values around the ledge: the clip pays more than 3.5 for the loop, the gate pays 0.
- Both misreads above, (18,19) and (19,8), are more than 2 tiles from the last accepted tile, so the gate rejects them.

Retrained `to_route2` with `--move-gate 2`, seeds 0-2, the same 1,800 iterations and everything else unchanged (`work/chain_gate.sh`, `work/agent_rl/to_route2_wm31_gate_s{0,1,2}`, about 44 min each at the 300 W cap). Real game, 100 episodes, seed 200, cap 1000:

| policy | standard start (29,23) | quest tile (23,26) |
|---|---|---|
| step clip, seeds 0 / 1 / 2 | 100 / 96 / 100 (median 35 / 35 / 36 presses) | **1 / 0 / 100** |
| move gate, seeds 0 / 1 / 2 | 100 / 100 / 100 (median 35 / 36 / 39) | **100 / 100 / 100** (median 36 / 38 / 39) |

Logs: `work/agent/eval_to_route2_wm31_gate_s*.log` and `*_quest_start.log`. The sweep rows are in the results table above: 2,339/2,340 over the 26 tiles.

**Conclusion (measured):** The seed-dependent failure was a reward that pays for a loop. A potential-based gate removes it for all 3 seeds without costing path length (median excess over BFS 3 presses vs 2).
