# PokeDreamer v3 Experimental Results Index

This directory contains tracked research appendix pages documenting the empirical findings for PokeDreamer v3: a diffusion world model of Pokémon Red, navigation policies trained entirely inside the model's imagination, and a dual System 1 / System 2 architecture evaluated in the real game on a Game Boy emulator (PyBoy).

All measurements are grounded in Game Boy RAM state and preserve exact empirical counts.

---

## Results Index

| Page | Scope / Task | Headline Number | Primary Source Artifacts |
|---|---|---|---|
| [`v3_gym.md`](v3_gym.md) | End-to-end Brock gym quest (Boulder Badge) with 10 learned navigation legs | **30/30 badges** (100%), **14/30 quests** with zero fallback | `work/gym_learned_best_s*` |
| [`v3_agent.md`](v3_agent.md) | Zero-shot real-game transfer from imagination RL & MPC across 3 navigation routes | **100/100 successes** on `leave_house` (28 presses), `route1` (22 presses), `viridian` (284 presses) | `work/agent_rl/` |
| [`v3_dwm_eval.md`](v3_dwm_eval.md) | RAM-grounded rollout evaluation, error decomposition, and controllability of wm31 | **41.1% wrong-map rate** at k=32 (position error 9.51 tiles vs 3.85 no-change) | `work/eval/` |
| [`v3_tile_sweep.md`](v3_tile_sweep.md) | Start-tile robustness sweep across 26 Viridian tiles and ledge-trap mitigation | Move gate achieves **2,339/2,340 (99.96%)** successes across 26 start tiles | `work/agent/sweep_to_route2/` |
| [`v3_battle_fidelity.md`](v3_battle_fidelity.md) | Battle onset and simulation fidelity evaluation of wm31 (Path B go/no-go) | **71.9% hallucinated early battle ends**, pixel MSE worse than copy-last for k ≥ 4 | `work/eval/battle_fidelity_*` |
| [`v3_system2.md`](v3_system2.md) | System 2 local LLM planner bake-off across 6 open-weight models | **5/6 models match rules** (10/10 badges); `gemma4:31b` has 0 rejected decisions | `work/sys2_bakeoff/` |
| [`v3_catch.md`](v3_catch.md) | Pokémon catching quest and System 2 strategy evaluation (throw vs. weaken) | Weaken-first: **48/50 runs reach 10 catches** vs 1/50 for throw (86.5% vs 36.8% catch rate per ball; 50 runs per strategy, seeds 200-249) | `work/catch_sweep/*/summary.json` |
| [`v3_system12.md`](v3_system12.md) | Integrated System 1/2 leveling and healing loop across health thresholds | HP threshold 0.4 optimizes presses (**3,257 mean**); 0.8 minimizes blackouts (0.38/run) | `work/system12_sweep/` |
| [`upsampler.md`](upsampler.md) | Native-resolution 72x80 -> 144x160 4-shade Game Boy frame upsampler | **99.980% pixel accuracy** on test_id (91.4% frames 100% exact) | `work/upsampler/` |
| [`colorizer.md`](colorizer.md) | Per-tile colour for real frames (exact) and for world-model frames (learned); display only | Stage 2 pixel class accuracy **99.27%** on held-out save states; stage 1 screen rebuild 99.9412% | `work/color/learned/metrics.json`, `work/color/native/validate.log` |
| [`probe.md`](probe.md) | Frame-to-RAM diagnostic probe extracting map ID, tile (x, y), and battle flag | **99.36% map accuracy**, 0.397 tiles mean Manhattan error on test_id | `work/probe/` |
| [`dataset_stats.md`](dataset_stats.md) | Dataset distribution, movement frequencies, battle occurrences, and action counts | **200,000 steps** across 200 episodes (139 train / 16 test_id / 45 test_ood) | `work/cache_72x80.npz` |
| [`speed.md`](speed.md) | Hardware training throughput benchmarks and project GPU compute accounting | **15.15 steps/s** (970 samples/s) with fused AdamW; GPU-hour ledger as of 2026-11-05 | `work/bench_speed.log` |
| [`compute.md`](compute.md) | Training time per model family and parameter count of each shipped model | **50.0 GPU-hours** of logged v3 training (76 runs, one RTX 4090); shipped world model 16.7M params, 3.6 h | run logs in `work/runs/`, `work/agent_rl/`, `work/upsampler/`, `work/color/learned/` |

---

## How to Read These Numbers

### 1. RAM Ground Truth
All evaluation metrics are grounded in physical Game Boy memory states extracted via PyBoy (`lab/env.py`). Visual frame quality (pixel MSE) can often appear sharp and coherent even when the underlying world state has drifted into invalid game territory (e.g., walls or nonexistent warps). Evaluating model predictions against physical RAM coordinates (map ID, (x, y) tile coordinates, HP, and battle status) separates genuine simulation fidelity from visually plausible hallucinations.

### 2. Held-Out Splits by Episode
Evaluations are strictly partitioned by whole gameplay episodes rather than random frame sampling. The base dataset consists of 200 episodes (1,000 steps each): 139 training episodes, 16 in-distribution test episodes (`test_id`), and 45 out-of-distribution Pallet-start episodes (`test_ood`). The extended dataset adds 60 held-out exploratory episodes (`test_ext`) covering Route 2, Viridian Forest, Pewter City, and the Gym. Test windows are sampled exclusively from held-out episodes.

### 3. Seeds and Trajectory Randomness
The PyBoy Game Boy emulator is fully deterministic for an identical button sequence. To measure robust statistical distributions rather than replaying identical trajectories, evaluation scripts inject a seed-dependent delay of 0 to 599 idle frames prior to the first agent action. Because the Game Boy hardware RNG advances on every clock cycle, this initial frame offset diversifies wild encounter triggers, battle damage rolls, and RNG outcomes across runs.

### 4. Bootstrap Confidence Intervals
Confidence intervals (typically reported as 95% CIs: `[lower, upper]`) are computed via cluster bootstrapping over entire episodes (1,000 resamples), rather than treating individual rollout windows as independent observations. This accounts for strong intra-episode temporal autocorrelation.

### 5. Counting Rules
- **Skipped legs:** In end-to-end quest logs, leg records where `reached_by_policy` is `None` (instances where the agent already occupies the destination tile upon leg initialization, requiring 0 button presses) are skipped and not counted as failures.
- **Fallbacks:** In learned navigation evaluations, if an imagination-trained policy fails to reach the destination within its per-leg press budget (typically 1,000 presses), a privileged BFS pathfinding routine completes the traversal. The fallback rate is reported per leg and per quest.
- **Blackout recovery:** If the party faints during grinding or traversal, the emulator returns the player to the last visited Pokémon Center (or bedroom). System 2 detects this state change via RAM, restores health, and re-initiates the necessary quest stage.

### 6. Claim Labels
Claims in these pages remain strictly labeled as originally established:
- **Measured:** Empirically verified against emulator RAM or checkpoint evaluations.
- **Implemented:** Code constructed and verified via unit tests, but not yet evaluated end-to-end.
- **Planned:** Proposed design directions or cost estimates.
