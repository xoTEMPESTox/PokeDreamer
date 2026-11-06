# System 1 proof of concept: acting in the real game by planning and RL inside the diffusion world model (measured)

This page measures the real-game transfer of navigation agents trained entirely inside the diffusion world model's imagination, evaluating both Model Predictive Control (MPC) and actor-critic Reinforcement Learning (RL) against privileged baselines and real-game PPO. On the `leave_house` task, an imagination-trained RL policy achieves 100/100 real-game successes (median 28 presses) using only 4 screen frames at test time. On `route1`, imagination RL achieves 100/100 successes (median 22 presses), beating privileged greedy heuristics; on the full Pallet-to-Viridian route, combining geodesic path distance rewards with step clipping achieves 100/100 real-game transfers (median 284 presses) without any real-environment interaction during training.

**Reproduce:** Reproducible via [lab/README.md §3.5](../README.md#35-headline-imagination-rl-leave-house) (`leave_house`), [lab/README.md §3.6](../README.md#36-harder-task-imagination-rl-route-1) (`route1`), [lab/README.md §3.7](../README.md#37-pallet-town--route-1--viridian-city-geodesic--clip) (`viridian` geodesic+clip), and [lab/README.md §3.8](../README.md#38-model-free-ppo-in-the-real-game-viridian-geodesic) (real-game PPO baseline).

---

All numbers **measured** in the real game (Pokémon Red ROM, SHA-1 ea9bcae6...; harness `lab/env.py` reproduces the recorded step exactly, see `v3_dwm_eval.md`). Code: `lab/agent/mpc.py`.

## Task and policies
- **leave_house**: start in Red's bedroom (`saves/bedroom.state` = the dataset's bedroom start: map 38, (3,6), facing up). Success = reach Pallet Town (map 0) within 60 button presses. Route: bedroom stairs (7,1), then the 1st floor, then the front door (2,7). Exit tiles come from the dataset's own map transitions. The recorded data left the house only 3 times in 200k steps.
- **MPC in the world model** (System 1 PoC): before each real press, imagine 64 random button sequences of 4 steps with the diffusion world model (`work/runs/main`, EMA), read the imagined map/tile with the frame probe, score progress toward the current map's exit, then press the first button of the best sequence. Uses the current RAM position and the exit tiles; never sees the future.
- **Greedy** (model-free control with the same privileged information): step toward the exit using RAM coordinates; after a press that did not move the player, try a different random direction.
- **Random**: uniform over the 4 directions.

## Results (`work/agent/*.json`, logs `work/agent_*.log`)

| policy | episodes | success (95% Wilson CI) | steps when successful: median (IQR), range |
|---|---|---|---|
| MPC in the world model | 21 | **21/21 = 100% [84.5, 100]** | **27 (26-29)**, 23-49 |
| Greedy with RAM | 200 | 195/200 = 97.5% [94.3, 98.9] | 36 (30-40), 24-59 |
| Random | 200 | 0/200 = 0% (30/200 reached the 1st floor) | n/a |
| (shortest route, scripted in `legacy/v1/scripts/make_save_state.py`) | n/a | n/a | ~22 |

MPC needs **9 fewer presses** than greedy: median difference -9 [-10, -6] (bootstrap 95% CI); Mann-Whitney U=761, z=-4.73, p=2e-6. Both know where the exits are. The difference is foresight: the planner imagines which presses run into furniture or walls, while greedy has to bump into them to find out.

The MPC episodes come from 5 seeds (1-4: 2/2/2/3 episodes, run as 4 parallel processes; 5: 12 episodes). A first 7-episode run (seed 0) crashed on a shared save-RAM file before saving its JSON; its log `work/chain_v3b.log` shows 7/7 successes in 24-41 steps, consistent with but not included in the table.

## The planner first exploited the world model (important negative result)
The first scoring rewarded any imagined map change (stage bedroom < 1st floor < Pallet). In the first real episode the planner pressed **left into the bedroom wall** for 60 steps: for those sequences the world model imagines teleporting downstairs or outside (`work/videos/leave_house_mpc_naive_scoring_ep0.mp4`). This is the measured map-hallucination weakness (wrong-map rate 13% by k=8), now shown to be exploitable by an optimiser. Fix: only progress on the current real map counts; an imagined map change is credited only right next to the exit tile, where a real warp can happen; horizon 4 (where imagined position is reliable). Same model, same task: 0/1 then 21/21.

## What this does and does not show
- **Does**: decisions made by imagining outcomes in a learned pixel simulator of Pokemon Red transfer to the real game, reliably and close to the shortest route, on a task the data almost never showed (3 recorded house exits). The simulator must be queried with safeguards against its known hallucinations.
- **Does not**: learn a policy (this is planning, not RL); the reward uses hand-specified exit tiles and the current RAM position; the task is short (~25 presses) and nearly solved by a 10-line heuristic with the same privileged information. The world model's contribution is efficiency (-9 presses), not feasibility.
- Speed: ~0.25 s per decision on the RTX 4090 in one process (64 x 4 imagined steps x 3 denoising passes). Four parallel processes were ~6x slower in total (GPU time-slicing on Windows), so run GPU-bound agents in one process and parallelise emulator-only work.

## Videos
- `work/videos/leave_house_mpc_main_n64_h4_s1_ep0.mp4`, `..._s0_ep0.mp4`: successful episodes; each frame shows the real screen before the decision, the agent's best imagined next 4 frames, and the pressed button.
- `work/videos/leave_house_mpc_naive_scoring_ep0.mp4`: the exploited planner (pressing left into the wall while imagining teleports).

## System 1 trained purely in imagination: RL policy, zero-shot in the real game (measured)
`lab/agent/imagine_rl.py`: a small CNN actor-critic trained **only on rollouts imagined by the frozen diffusion world model**, never interacting with the emulator. Starts: real 4-frame contexts from the train split in the bedroom and 1st floor, sampled 50/50 per map. 10 imagined steps per rollout, batch 128, TD(lambda) advantage actor-critic, entropy 0.03. Reward: progress toward the current map's exit read by the frame probe on imagined frames, with the planner's safeguard (an imagined map change ends the rollout: +5 next to the exit tile, -2 elsewhere as a hallucinated warp). 640 iterations / 30.7 min on the RTX 4090 (~0.8M imagined steps). Log `work/agent_rl/leave_house/train.log`.

**At test time the policy sees only the last 4 real screen frames**: no RAM, no coordinates, no planning (`python -m lab.agent.mpc --policy learned`). 100 fresh real-game episodes per checkpoint, seed 200 (the 30-episode checks during training used seed 100):

| policy (real game, leave_house, 60-press cap) | success (95% CI) | presses: median (IQR), range | at test time it uses |
|---|---|---|---|
| **RL in imagination, final (it 640), pre-declared headline** | **100/100 [96.3, 100]** | **28 (24-28), 24-32** | 4 screen frames |
| RL checkpoint it 600 | 100/100 | 24 | 4 screen frames |
| RL checkpoint it 400 | 100/100 | 25 | 4 screen frames |
| RL checkpoint it 200 (before the 50/50 start fix) | 0/100 | n/a | 4 screen frames |
| MPC planning in the world model | 21/21 [84.5, 100] | 27 (26-29), 23-49 | frames + RAM position + exit tiles |
| Greedy with RAM | 195/200 [94.3, 98.9] | 36 (30-40), 24-59 | RAM position + exit tiles |
| Random | 0/200 | n/a | n/a |

- RL vs greedy: -8 presses [-9, -6] (bootstrap 95% CI), Mann-Whitney z=-11.3, p<1e-10. RL vs MPC: +1 [-1, +2], p=0.15 (no detectable difference); RL is far more consistent (max 32 vs 49 presses) and ~40x cheaper per episode (one small CNN forward per press vs 64x4 imagined steps).
- The it-200 checkpoint (unbalanced starts: 1st floor = 1.5% of contexts) reached the 1st floor in 30/30 earlier checks, then walked back upstairs; balancing the starts fixed it within ~200 iterations.
- The final checkpoint is slightly slower than it 400/600 (28 vs 24-25 presses); checkpoints were not selected on real-game results.
- Video: `work/videos/leave_house_learned_leave_house_s300_ep0.mp4` (final policy, 28 presses).

What it shows: **a policy trained entirely inside the learned diffusion simulator transfers zero-shot to the real game, near the shortest route (~22), from pixels alone.** Limits: one short task; training reward used hand-specified exit tiles and the probe (privileged information during training, not at test time); the world model was trained on data from a PPO agent in the same area.

## Harder task: Pallet Town -> Route 1 (measured)
Start: `saves/starter_pallet.state` (scripted from the ROM: Charmander, rival beaten, standing at Oak's Lab door, so Oak no longer stops you). Success: Route 1 (map 12) within 100 presses. Exits from recorded transitions, with detour exits for buildings (`lab/agent/tasks.py`). RL trained in imagination for 35 min (1,440 iterations; start contexts 50/50 Pallet Town (4,026) / Oak's Lab (12,111); log `work/agent_rl/route1/train.log`; imagined hallucinated warps fell from 30% to ~5%). Pre-declared: final checkpoint, 100 fresh episodes, seed 200.

| policy (real game) | success (95% Wilson CI) | presses: median (IQR), range | uses at test time |
|---|---|---|---|
| **RL trained purely in imagination** | **100/100 [96.3, 100]** | **22 (22-23), 22-24** | 4 screen frames |
| Greedy with RAM coordinates | 61/100 [51.2, 70.0] | 57 (40-76), 24-100 | RAM position + exits |
| Random | 0/100 | n/a | n/a |

The learned policy needs 35 fewer presses than greedy on successful episodes (bootstrap 95% CI [-40, -24]; Mann-Whitney z=-10.6, p<1e-10), with an almost constant route length. On this open-town task with fences and houses, the imagination-trained policy beats the privileged hand-coded rule on both reliability and efficiency (on "leave the house" they were close). A 30-episode check at it 700 also gave 30/30 (median 22). GIF: `work/videos/route1_policy.gif` (22 presses).

## Harder task: Pallet Town -> Route 1 -> Viridian City (measured): negative, diagnosed
Same start state; success: Viridian City (map 1). Route 1 has tall grass (wild battles; scripted reflex: press A) and a maze of ledges and trees. RL in imagination 45 min (1,700 iterations; starts balanced over Pallet / Route 1 / Oak's Lab; log `work/agent_rl/viridian/train.log`). Pre-declared eval: 100 episodes, seed 200, max 300 presses (`work/agent/viridian_*_s200.json`).

| policy (real game, max 300 presses) | success |
|---|---|
| RL in imagination (Manhattan distance reward) | 0/100 [0, 3.7] |
| Greedy with RAM coordinates | 0/100 |
| Random | 0/100 |

**Why (checked):**
- **Wrong reward shape.** Every learned episode ended on Route 1 at row y=28, x 10-15 (`trace`). In the recorded data, the only step north from row 28 is at x=7, and further detours follow (x=12-14 at row 24, x=5/9 at row 20). Distance-to-exit in a straight line has a local optimum there.
  - Fix: `geodesic_fields` gives path distance over the train split's observed one-tile moves. Moves are directed, so ledges count one way. Tiles never reached get max+1. Route 1 entry: 58 moves on this graph vs 35 in a straight line.
- **The 300-press cap is too tight once there is grass.** A 5-episode smoke test of greedy with the geodesic field gave 2/5. The 3 failures had used 220-248 of their 300 presses on battle A-presses, and their field values were consistent.
- **The world model almost never imagines wild encounters** (see `v3_dwm_eval.md`, encounter section): battle onset within 10 steps is 27% real vs 3% imagined. So the policy cannot learn in imagination that grass costs time.

**1,000-press evaluation (`work/chain_night4.sh` (internal run script; not published), protocol written before any result; 100 episodes, seed 200; `work/agent/viridian_*_s200.json`):**

| policy (real game) | success | presses: median (outside battle) |
|---|---|---|
| greedy with RAM, **geodesic** field | **100/100** | 266 (97) |
| greedy with RAM, Manhattan | 0/100 | n/a |
| RL in imagination, Manhattan reward (`viridian`) | 0/100 | n/a |
| RL in imagination, geodesic reward (`viridian_geo`) | 0/100 | n/a |
| random | 0/100 | n/a |

The geodesic field is right: privileged greedy on it solves the task every time.

`viridian_geo` fails differently from the Manhattan policy. 95/100 episodes end in Pallet at y=2, x 10-13, shuffling left/right; it never presses Up into the 2-tile exit corridor. Checked:
- On training-data frames there the policy gives P(up) = 0.00, vs 0.44-0.49 for the Manhattan-trained policies, so this is not sim-to-real.
- The exact imagined training reward favours Up there: always-Up return +2.2 vs -0.2 for left/right.
- Imagined Up warps to Route 1 in 99% of rollouts within 3 steps.

So learning failed despite a correct reward. 15-20% of imagined rollouts in that region contain a single-step reward of -26..-31 (probe misread / hallucinated jump), and the run's value loss was 3-6x higher. Hypothesis: those spikes break the advantage estimates.

Fix: `--step-clip 1`; a press moves at most one tile. **Result, `viridian_geo_clip`** (geodesic + clip, 45 min imagination RL, 0 real steps; [`../experiments/headline_rl.sh`](../experiments/headline_rl.sh), same pre-declared eval; `work/agent/viridian_learned_viridian_geo_clip_s200.json`):
- **100/100** real-game successes Pallet -> Viridian, median 284 presses (95 outside battle);
- every episode reached and crossed Route 1;
- comparable to real-game PPO (100/100, 249 / 85, 54k real steps) and privileged greedy-geodesic (100/100, 266 / 97);
- value loss stayed ~1-2 (unclipped run: 28-78);
- GIF: `work/videos/viridian_geo_clip_policy.gif`.

This confirms the diagnosis: reward spikes from probe misreads / hallucinated jumps broke the actor-critic.

**Ablation: Manhattan + clip, without geodesic (`viridian_clip`).**
- Setup: 45 min imagination RL, 1,240 it total; the same pre-declared eval (`work/agent/viridian_learned_viridian_clip_s200.json`).
- Result: **0/100**. 95/100 episodes end on Route 1 at y=28, x 14-15: the Manhattan stall that the geodesic field was introduced for. 0 dialogue presses.
- Complete 2×2 for Pallet -> Viridian, imagination RL, 0 real steps:

| | no clip | clip ±1 tile |
|---|---|---|
| Manhattan | 0/100 | 0/100 |
| geodesic | 0/100 | **100/100** |

Both parts are needed: the path-distance reward gets the agent past the Route 1 detour, and the clip keeps probe-misread spikes out of the advantages. One seed per cell.

**Model-free PPO in the real game on the same task** (`lab/agent/ppo_real.py`, geodesic reward, cap 1000, 12 emulators; `work/ppo_real/viridian_geo/eval.jsonl`):

| env steps | wall time | success / 30 | median presses |
|---|---|---|---|
| 26k | 50 s | 43% | 634 |
| 51k | 2.5 min | 87% | 480 |
| 76k | 3.6 min | 97% | 320 |
| 103k | 5 min | 100% | 314 |

Final evaluations (100 episodes, seed 200, cap 1000):

| PPO in the real game | env steps trained | success | presses: median (outside battle), range |
|---|---|---|---|
| geodesic reward | 102,881 | 95/100 | 262 (89), 82-593 |
| Manhattan reward | 54,028 (30/30 already at 27k) | **100/100** | 249 (85), 72-727 |

Files: `work/agent/viridian_learned_ppo_real_viridian{,_geo}_s200.json`; curves in `work/ppo_real/viridian{,_geo}/eval.jsonl`.

**PPO across seeds** (Manhattan reward, the same as the imagination policies; seed 42 evaluated every 25k steps, so its "first ≥95%" is an upper bound; seeds 1-2 every 5k (viridian 10k); `work/ppo_real/<task>{,_s1,_s2}/eval.jsonl`, `work/agent/*_ppo_real_*_s200.json`, `work/chain_ppo2.sh` (internal run script; not published)):

| task | env steps to first ≥95% (seed 42 / 1 / 2) | final 100 eps (success) | median presses |
|---|---|---|---|
| leave_house | ≤26,112 / 21,504 / 15,360 | 100, 100, 100 | 24, 24, 24 |
| route1 | ≤26,112 / 10,752 / 10,752 | 98, 100, 100 | 30, 17, 17.5 |
| viridian (cap 1000) | ≤27,153 / 32,447 / 41,453 | 100, 100, 99 | 249, 261, 241 |

**Reading:**
- The Manhattan reward that traps the imagination policy at Route 1 row 28 does not trap PPO in the real game. So the viridian failure is about learning inside imagination, not about the reward. Candidate causes: the 10-step imagined horizon, and how the world model renders the ledge/tree barrier.
- **Across all three navigation tasks, model-free PPO in this fast emulator reaches ≥95% within 10.8k-41.5k real steps with the Manhattan reward (3 seeds); the geodesic-reward viridian run took 76k-103k steps, 4-5 min of CPU:**
  - leave_house 100/100 (24);
  - route1 98/100 (30);
  - viridian 100/100 (249).
- The imagination-trained policy (0 real steps) matches or beats it on the two short tasks (100/100, 28 and 22 presses) and fails the long one without geodesic reward + clipping.
- Honest framing: here the world model is not a sample-efficiency win. Its value is as a measured object (what imagination gets wrong, in game units) and a reusable simulator. It would pay off where the real environment is slow or unavailable.
