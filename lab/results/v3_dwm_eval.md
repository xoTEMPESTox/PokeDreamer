# Diffusion world model: RAM-grounded evaluation (measured)

This page evaluates the 16.7M-parameter DIAMOND-style diffusion world model (`main` and `wm31`) against Game Boy RAM ground truth across rollouts of 1 to 32 steps (0.40 to 13 seconds of gameplay). While pixel MSE outperforms trivial baselines up to horizon k=32, RAM-grounded position error reveals that long-horizon degradation is primarily driven by map-identity hallucinations (41.1% wrong map at k=32 on `test_id`) rather than within-map drift. Additional evaluations benchmark action controllability, context-noise augmentation ablations, RAM-conditioned hybrid rendering, wild battle onset frequencies, and zero-shot transfer across newly collected areas (`test_ext`).

**Reproduce:** Reproducible via [lab/README.md §3.9](../README.md#39-diffusion-world-model-rollout-evaluation--error-decomposition) (`python -m lab.dwm.eval_rollouts --run wm31 --data ext --split test_ext --n-windows 512 --k 32 --steps 3 --device cuda` followed by `python -m lab.eval.decompose work/eval/wm31_test_ext_s3_a0_ext.npz`).

---

All numbers **measured** unless marked otherwise. Mean [95% CI]; CIs are a cluster bootstrap over **episodes** (1000 resamples), not windows.

**Units:** k counts *steps*. One step = one button press followed by 24 Game Boy frames (`pyboy.button(btn, 8)` then `pyboy.tick(24)`) ≈ 0.40 s at 59.7 fps, i.e. time for exactly one tile of walking. k = 8 ≈ 3 s, k = 32 ≈ 13 s, the 64-step videos ≈ 26 s. These are not video frames.

## Setup (implemented)
- Model: DIAMOND-style EDM denoiser U-Net (16.7M params) on 72x80 greyscale, conditioned on the last 4 frames and 4 actions; 3-step Euler sampling; GameNGen-style context-noise augmentation. Code: `lab/dwm/model.py`, `lab/dwm/train.py`.
- Training: run `work/runs/main`, 88,600 steps x batch 64 in 120.2 min on the RTX 4090 (log `work/runs/main/train.log`; peak 62 C, no thermal pauses). Train split = 139 episodes. EMA weights used everywhere.
- Evaluation: 512 random 36-frame windows per split (4 real context frames + 32 imagined steps on the real action sequence), seed 0. `lab/dwm/eval_rollouts.py`, outputs `work/eval/main_<split>_s3_a0.{json,npz}`, plots `work/plots/main_<split>_s3_a0_*.png`.
- Reading imagined frames: a CNN probe trained on real train-split frames (`probe.md`). On real held-out frames it reads position to 0.40 tiles (test_id), so in-distribution errors below are the world model's, not the probe's.
- Position metrics exclude steps where the true frame is in battle or the true map changed (tile coordinates are per map): 77% of (window, k) entries are valid on test_id.

## In-distribution held-out episodes (test_id: 16 episodes, 512 windows)

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| pixel MSE, imagined | 0.014 [0.01,0.02] | 0.019 [0.01,0.03] | 0.033 [0.02,0.05] | 0.052 [0.04,0.07] | 0.090 [0.06,0.12] | 0.127 [0.09,0.16] |
| pixel MSE, copy last frame | 0.035 [0.02,0.05] | 0.052 [0.03,0.07] | 0.067 [0.05,0.09] | 0.087 [0.06,0.11] | 0.117 [0.08,0.15] | 0.149 [0.10,0.19] |
| position error (tiles), imagined | 0.59 [0.20,1.07] | 0.90 [0.40,1.50] | 1.52 [0.61,2.63] | 3.10 [1.46,5.11] | 5.57 [3.27,8.28] | 9.51 [5.66,14.35] |
| position error, no-change baseline | 0.46 [0.31,0.61] | 0.78 [0.54,1.03] | 1.27 [0.83,1.72] | 2.13 [1.36,3.00] | 3.14 [1.97,4.45] | 3.85 [2.55,5.16] |
| ...moving windows only, imagined | 0.22 [0.01,0.63] | 0.57 [0.20,1.07] | 2.09 [0.86,3.64] | 3.79 [2.17,5.68] | 7.24 [5.20,9.66] | 12.13 [8.44,17.85] |
| ...moving windows only, no-change | 1.00 [1.00,1.00] | 1.84 [1.75,1.91] | 2.81 [2.46,3.11] | 4.20 [3.46,4.94] | 5.54 [4.42,6.64] | 6.48 [5.34,7.58] |
| map accuracy, imagined | 0.982 | 0.961 | 0.929 | 0.850 | 0.746 | 0.571 |
| map accuracy, no-change | 0.997 | 0.990 | 0.977 | 0.947 | 0.925 | 0.915 |
| battle-state accuracy, imagined / no-change | 0.986 / 1.000 | 0.980 / 0.994 | 0.967 / 0.987 | 0.943 / 0.959 | 0.892 / 0.924 | 0.838 / 0.857 |

**Where the error comes from** (`python -m lab.eval.decompose work/eval/main_test_id_s3_a0.npz`):

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| imagined frame shows the wrong map | 1.8% | 3.2% | 5.7% | 12.6% | 23.1% | 41.1% |
| position error when the imagined map is right | 0.47 [0.10,0.92] | 0.54 [0.20,0.95] | 0.99 [0.42,1.68] | 1.71 [0.72,2.92] | 2.02 [1.14,3.00] | 5.03 [2.51,8.44] |
| no-change baseline on the same windows | 0.46 [0.31,0.61] | 0.81 [0.56,1.06] | 1.30 [0.86,1.76] | 2.26 [1.37,3.25] | 3.41 [2.07,4.95] | 4.35 [2.68,6.02] |

Reading:
1. **Pixel metrics and RAM metrics disagree.** By pixel MSE the model beats copy-last-frame at every horizon up to 32. By RAM position it is worse than the trivial no-change predictor from k ~ 8 on. Sharp, plausible frames are not the same as a correct world state; this gap is the point of RAM-grounded evaluation.
2. **Short-horizon dynamics are learned.** When the player actually moves, the model tracks it far better than no-change at k = 1-2 (0.22 vs 1.00, 0.57 vs 1.84 tiles), and stays at or below the baseline to k ~ 8 (`work/plots/main_test_id_s3_a0_drift.png`, right panel).
3. **The dominant long-horizon failure is map identity, not in-map drift.** The wrong-map rate grows from 2% (k=1) to 41% (k=32). When the map is right, position error stays at or below no-change up to k = 16 (CIs overlap beyond k = 4, so this is a trend, not a proven gap). A 4-frame context cannot remember which building/route it is in once the screen looks ambiguous; this motivates memory or RAM-state conditioning.
4. Light context noise at inference (bucket 1, sigma 0.078) did not help: position error at k=8 was 3.89 vs 3.10 with clean context (`work/eval/main_test_id_s3_a1.json`). Bucket 0 was the pre-registered setting.

## Held-out start location (test_ood: 45 Pallet-start episodes, 512 windows): inconclusive on position
- The probe itself is unreliable here (4.8-5.4 tiles error and 87% map accuracy on *real* frames), so position numbers mostly measure the probe. The player also barely moves in these windows (no-change error 0.12-0.91 tiles).
- Pixel MSE is worse than copy-last at every k (k=1: 0.017 vs 0.011; k=32: 0.096 vs 0.068): scenes that are mostly static (Oak's Lab, Pallet) get spurious changes. These maps are 9% and 3% of training frames.
- Map accuracy of imagined frames falls from 0.86 (k=1) to 0.51 (k=32); real-frame probe accuracy is 0.87.

## Action controllability (`lab/dwm/controllability.py`)
- Unfiltered contexts (`work/eval/control_main_test_id_k8_s3.json`, 128 contexts): the imagined player moves the commanded way for left/right/up (k=1: +0.53, +0.52, +0.61 tiles; CIs above 0; wrong-way moves 0-1.6%), but far less than the real game. **Confounded**: many contexts have a text box open (e.g. "RED is playing the SNES!"), where the real game ignores direction buttons; the dataset has no usable dialog flag. See the movement-possible rerun below.
- **Movement-possible contexts** (`--require-move 4`: contexts and real-reference windows where RAM shows the player moving within 4 steps; `work/eval/control_main_test_id_k8_s3_free4.json`, 128 contexts, CPU run, log `work/control_free4_main.log`). Commanded-axis displacement in tiles, imagined vs real game (real = windows where the PPO agent pressed the same button k times, same filter):

| button | k=1 imagined | k=1 real | moved commanded way (k=1) | moved wrong way (k=1) | k=8 imagined | k=8 real |
|---|---|---|---|---|---|---|
| down | +0.95 [0.55,1.52] | +1.00 (N=18) | 60% | 0.8% | +1.66 [0.27,2.81] | +5.83 |
| left | +0.93 [0.84,1.02] | +1.00 (N=35) | 89% | 0.8% | +2.46 [1.27,3.58] | +7.60 |
| right | +1.03 [0.95,1.13] | +1.00 (N=38) | 95% | 0.0% | +2.24 [1.06,3.21] | +8.00 |
| up | +0.02 [-1.03,0.74] | +0.97 (N=94) | 76% | 5.5% | +3.01 [2.10,3.92] | +7.80 |

  - **One step ahead the model is a calibrated simulator** for left/right/down (~1 tile per press, as in the game). "Up" at k=1 moves the commanded way in 76% of contexts, but a few large wrong-way readings (5.5% of contexts) pull the mean to ~0; the success fractions are the robust statistic there (per-context values were not saved, so no median).
  - Over 8 presses the reference windows were biased toward free walking; from the same states the real game moves 2-4 tiles too (see real control below). Orthogonal drift does grow (up to 2.3 tiles for right at k=8).
- No button pressed ("pass", 0.4% of training actions): the imagined player still moves 0.64 tiles at k=1 and 2.84 at k=8 in movement-possible contexts (0.58 / 1.61 unfiltered). The model partly predicts the data policy's habitual walking instead of respecting "no input". For training an agent inside the model this matters: the agent's actions would be partly overridden by a prior.

## Ground truth from the real game (measured)
The Pokémon Red ROM (SHA-1 ea9bcae6..., checksums OK) runs in `lab/env.py`, which reproduces the recorded step exactly: the bedroom save state equals all 81 recorded bedroom starts (map 38, (3,6), facing up), and frames after 30 random steps are pixel-identical to dataset frames at the same map/tile/facing (mean abs diff 0.0000; `python -m lab.env --check`).

**Controllability vs the emulator** (`python -m lab.dwm.real_control --run main --n-contexts 64 --k 8`; `work/eval/real_control_main_k8.json`, plot `work/plots/real_control_main_k8.png`). 64 real contexts from seeded random walks starting in the bedroom and in Pallet Town. Each button is pressed K times in the emulator and in the world model from the same 4 real frames:

| button | k=1 real / imagined (tiles) | k=8 real / imagined | position error k=1 |
|---|---|---|---|
| down | +0.61 / +0.68 | +1.66 [1.17,2.22] / +1.39 [0.71,2.21] | 0.30 |
| left | +0.70 / +0.64 | +2.25 [1.80,2.75] / +0.75 [0.00,1.50] | 0.16 |
| right | +0.78 / +0.95 | +4.08 [3.33,4.86] / +4.36 [3.26,5.69] | 0.14 |
| up | +0.58 / -0.02 | +2.02 [1.53,2.53] / +1.73 [0.69,2.66] | 0.94 |
| no button | 0.00 / +1.02 [0.36,1.92] | k=2: 0.00 / +1.98 | 1.02 |

- **Correction to the movement-possible analysis:** "it under-moves over 8 presses (1.7-3.0 vs 5.8-8.0)" was mostly an artefact of the reference. The real game also moves only 2-4 tiles in 8 presses from these states (walls in small rooms); the PPO-reference windows were selected where the agent walked freely. Against ground truth from the same states, right/down/up match within CIs at k=8; left under-moves.
- **The clearest real flaw: hallucinated motion without input.** With no button, the real player stays put, but the imagined one moves ~1 tile per step. "Up" also starts badly (k=1 ~0 vs +0.58 real) and mid-horizon "down" overshoots (k=3-6).
- Imagined-vs-real map mismatch is 9-23% already at k=1. Some of that is likely probe error on Pallet Town frames (probe map accuracy on held-out Pallet-start frames is 87%); not yet separated.

## Context-noise augmentation ablation (measured, single seed)
Identical recipe without context noise (`--no-ctx-aug`, run `work/runs/noaug`), stopped at step 50,200, vs the main run's snapshot at the same step (`work/runs/main60`, copied at 60.1 min). Same eval as above (`work/eval/{main60,noaug}_*_s3_a0.json`, decomposition and controllability logged in `work/chain_rest.log`). Step-matched, not time-matched (63.7 vs 60.1 min).

test_id (16 episodes):

| k | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| pixel MSE, with aug / no aug | 0.012 / 0.010 | 0.040 / 0.030 | 0.064 / 0.044 | 0.097 / 0.070 | 0.129 [0.09,0.17] / 0.094 [0.07,0.12] |
| position error (tiles), with / without | 0.55 / 0.55 | 1.37 / 1.46 | 2.80 / 3.42 | 4.76 / 6.11 | 7.16 [3.77,11.26] / 9.11 [4.79,14.54] |
| wrong-map rate, with / without | 1.8% / 2.2% | 5.7% / 7.5% | 13.8% / 12.8% | 26.5% / 26.2% | 44.2% / 40.0% |
| map accuracy, with / without | 0.982 / 0.978 | 0.928 / 0.911 | 0.835 / 0.847 | 0.710 / 0.723 | 0.548 / 0.577 |

test_ood (45 episodes): map accuracy of imagined frames at k=32 is **0.47 [0.35,0.59] with aug vs 0.73 [0.64,0.81] without**, and imagined-vs-real probe agreement 10.0 [8.4,11.5] vs 5.9 [4.5,7.5] tiles: non-overlapping CIs, both favouring no augmentation. Pixel MSE also favours no augmentation at every k.

Movement-possible controllability (`--require-move 4`, test_id): no difference. At k=1 left/right move +0.91/+1.07 tiles (aug) vs +0.94/+1.14 (no aug), commanded way in 87-95% of contexts for both; "pass" drifts 0.60 vs 0.64 tiles at k=1.

Reading:
1. **The hypothesis "context-noise augmentation is what fixes the 2-5 frame collapse" is not supported.** Without it the model is equally stable (pixel MSE at k=32 0.094 vs copy-last 0.149; rollout grid `work/plots/noaug_test_id_s3_a0_grid.png` tracks the real game to k=16 and keeps scenes coherent to k=32). The stability over v1/v2 comes from the diffusion recipe itself (EDM + 4-frame pixel context), as DIAMOND reported.
2. At this budget augmentation is neutral in-distribution (position trends better with it at long horizons, pixel MSE worse; all CIs overlap) and **harmful out of distribution** (map consistency). Plausible cause: 90% of training samples see noisy context, so at a fixed step budget the model gets less practice with the clean context it sees at inference. Not tested: lower noise levels, more steps, or 2+ seeds.
3. Training longer did not clearly help the RAM metrics either: main at 120 min (88.6k steps) vs main60 (50.2k): test_id position error at k=32 9.51 vs 7.16, map accuracy 0.571 vs 0.548 (CIs overlap). Long-horizon state errors are not just under-training; they point at the 4-frame memory limit.

## Classifier-free guidance on the hybrid's RAM conditioning (measured): negative
Fine-tune of the hybrid (`work/runs/hybrid_cfg`): init from `work/runs/hybrid`, RAM condition dropped for 15% of samples (learned null embedding), 25,000 steps, 30.9 min. Evaluated with guidance w = 1, 2, 3 on the same 512 windows (`work/eval/hybrid_cfg_*`). test_id, k=32:

| model | wrong map in rendered frames | position error (tiles) | pixel MSE | wrong map, state model |
|---|---|---|---|---|
| hybrid (no CFG) | 25.9% [16, 38] | 8.73 | 0.099 | 1.1% |
| CFG fine-tune, w=1 | 30.0% [19, 42] | 9.26 | 0.107 | 0.9% |
| CFG fine-tune, w=2 | 34.4% [22, 49] | 10.02 | 0.117 | 0.9% |
| CFG fine-tune, w=3 | 37.9% [26, 50] | 9.80 | 0.129 | 0.9% |

Guidance does not make the renderer follow the state; rendered map errors grow with w (CIs overlap, monotone trend), and pixel error grows too. test_ood shows the same direction (wrong map 18.9% -> 20.4 / 20.9 / 24.0%). Interpretation: the renderer's map errors come from its drifting 4-frame pixel context, not from a weak RAM signal, so amplifying the conditioned-vs-unconditioned difference amplifies noise.

## Context noise on the hybrid (measured): no effect on the renderer
`hybrid_ctxaug`: fine-tuned from `hybrid` for 25k steps (31 min) with context-noise augmentation plus RAM-condition dropout 0.15 (`work/chain_night3.log`). The budget matches the CFG fine-tune. The idea: with corrupted pixel context, the renderer would have to rely on the RAM state. Same 512 windows per split; k=32; 95% cluster-bootstrap CI.

| k=32 | test_id rendered wrong-map | test_id pos err (tiles) | test_id pixel MSE | test_ood rendered wrong-map | test_ood state pos err |
|---|---|---|---|---|---|
| hybrid | 25.9% [15.8, 38.0] | 8.73 | 0.099 | 18.9% [11.9, 27.3] | 2.42 |
| hybrid_cfg, g=1 | 30.0% [18.8, 41.9] | 9.26 | 0.107 | 20.4% | 2.55 |
| hybrid_ctxaug, g=1 | 25.5% [14.8, 38.1] | 7.52 | 0.117 | 25.4% [17.5, 33.6] | 3.30 |
| hybrid_ctxaug, g=2 | 26.1% [16.7, 37.3] | 8.19 | 0.126 | 18.0% [11.7, 24.7] | 3.30 |

Files: `work/eval/hybrid_ctxaug_test_{id,ood}_hybrid{,_g2.0}.json`.

Reading: neither guidance nor context noise makes the renderer follow its conditioning. In distribution the rendered map error stays ~25% at k=32, while the symbolic state is ~0%. The fine-tune also made the RAM model worse out of distribution (3.30 vs 2.42 tiles). The knowledge lives in the symbolic state, and the renderer keeps copying its own drifting frames.

## Does the world model imagine wild encounters? (measured): almost never
`lab/dwm/battle_onset.py --run main --map 12`. Setup:
- held-out test_id episodes;
- Route 1 contexts with no battle in the 4 context frames: 1,222 windows from 9 episodes;
- open-loop rollouts driven by the real next 10 button presses;
- 4 samples per window;
- raw: `work/eval/battle_onset_main_map12.json`.

| | battle starts within 10 steps | frames in battle |
|---|---|---|
| RAM (ground truth) | 27.2% | 15.8% |
| probe on real frames | 25.5% | 13.9% |
| probe on imagined frames | 2.9% (9.8% in any of 4 samples) | 0.4% |

Of the windows where a real battle starts, an imagined sample shows one only 6.0% of the time. The probe reads real battles well, so the gap is the model's.

The model has learned the typical outcome, "walk on", and almost never produces the rare stochastic event. Plausible reasons:
- 3-step deterministic Euler sampling;
- the transition into a battle is a short screen flash;
- the encounter roll is random, so it cannot be predicted from pixels.

Consequence: an agent trained in this imagination never pays for walking through grass. That explains the "battle 0.000" in the viridian RL logs; route1 has no grass, so it was consistent there.

**More sampling steps don't help** (same windows; `work/eval/battle_onset_{main_map12_s10,noaug_map12}.json`):

| model, Euler steps | imagined onset within 10 | any of 4 samples |
|---|---|---|
| main, 3 | 2.9% | 9.8% |
| main, 10 | 3.1% | 10.0% |
| noaug, 3 | 4.5% | 10.7% |

So it is not a sampler artefact: the learned conditional puts almost no mass on "battle starts".

## wm31: the world model on the new areas (measured)
`wm31`: noaug's checkpoint (step 50,200) trained 150.1 min more (114,800 steps, batch 64, lr 2e-4, fresh optimizer, no context noise) on base train + 540 collected episodes Viridian City -> Pewter Gym (`work/runs/wm31/train.log`). 676,284 train windows, 20.5% from the original areas. HF splits unchanged (`lab/data.py` `merge`), so test_id/test_ood stay held out; test_ext = 60 held-out collected episodes. Same eval as above (512 windows, seed 0, 3 Euler steps, k=32, episode-cluster bootstrap). Log `work/chain_show_a.log`.

**New areas** (test_ext, ext probe `work/probe_ext/probe.pt`; `work/eval/wm31_test_ext_s3_a0_ext.{json,npz}`, plots `work/plots/wm31_test_ext_s3_a0_ext_*.png`). The probe reads real held-out frames to 0.44 tiles at k=1 (map 99%), so the errors below are the world model's.

| k | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| wrong-map rate | 4.9% | 12.8% | 24.0% | 36.0% | 46.5% [39.2,54.0] |
| position error (tiles) / no-change | 1.62 / 0.79 | 4.40 / 2.45 | 6.77 / 3.54 | 10.50 / 4.75 | 14.04 [11.8,16.5] / 5.82 |
| pixel MSE / copy-last | 0.018 / 0.048 | 0.044 / 0.099 | 0.065 / 0.122 | 0.092 / 0.140 | 0.127 [0.111,0.144] / 0.160 |

Per map (windows whose first context frame is on that map; k = 1 / 8 / 32):

| map (windows, episodes) | wrong-map rate | position error / no-change | pixel MSE / copy-last |
|---|---|---|---|
| Route 2 (96, 15) | 2.4 / 15.6 / 34.0% | 1.3/0.8, 8.6/3.8, 19.7/5.2 | 0.014/0.040, 0.054/0.121, 0.133/0.154 |
| Viridian Forest (76, 11) | 3.0 / 37.4 / 54.7% | 3.7/0.7, 16.4/2.9, 22.5/4.2 | 0.033/0.045, 0.126/0.155, 0.193/0.211 |
| Pewter City (111, 17) | 8.3 / 36.1 / 61.9% | 2.1/1.0, 7.0/4.7, 19.8/8.4 | 0.018/0.104, 0.051/0.166, 0.126/0.184 |
| Pewter Gym (15, 4) | 6.2 / 36.1 / 30.6% | 0.5/0.2, 1.3/1.3, 5.7/4.3 | 0.033/0.011, 0.107/0.096, 0.167/0.168 |

**Original areas** (base probe, same 512 windows as main/noaug; `work/eval/wm31_test_{id,ood}_s3_a0.json`):

| test_id, k = 1 / 8 / 32 | wm31 | main | noaug | no-change / copy-last |
|---|---|---|---|---|
| position error (tiles) | 0.66 / 3.99 / 8.99 | 0.59 / 3.10 / 9.51 | 0.55 / 3.42 / 9.11 | 0.46 / 2.13 / 3.85 |
| pixel MSE | 0.020 / 0.093 / 0.146 | 0.014 / 0.052 / 0.127 | 0.010 / 0.044 / 0.094 | 0.035 / 0.087 / 0.149 |
| wrong-map rate | 3.9 / 27.9 / 50.8% | - | 2.2 / 12.8 / 40.0% | |

test_ood, k=8: position error 8.33 (wm31) vs 6.09 (main) vs 7.08 (noaug); pixel MSE 0.072 vs 0.057 vs 0.050 (copy-last 0.043).

Reading:
1. In the new areas wm31 beats copy-last on pixels at every horizon. Probe-read position is about 2x the no-change error throughout, the same profile main has on the original areas (3.10 vs 2.13 at k=8). Viridian Forest is the weakest map: by k=4 the position is 14 tiles off, against 1.9 for no-change, and 25% of rollouts read as another map. The repetitive tree tiles give the 4-frame context little to anchor on.
2. **wm31 regressed on the original areas.** Compared with its own starting point (noaug), pixel MSE at k=8 doubled (0.093 vs 0.044; from k=4 to 32 it only ties copy-last, e.g. 0.093 vs 0.087 at k=8, 0.146 vs 0.149 at k=32) and the wrong-map rate at k=8 doubled (27.9% vs 12.8%). Plausible cause: only 20.5% of training windows come from those areas, so the fixed-capacity model shifted towards the new ones. One seed, one run.
3. For imagination RL (10-step rollouts) wm31 in the new areas is roughly as wrong as main was on Route 1, where the viridian policy reached 100/100. So the per-leg RL on wm31 was run anyway (`work/chain_show_b.sh` (internal run script; not published)); the forest leg is the one most likely to suffer.

Display: `work/plots/dream_wm31_journey.gif` (seed-100 gym run, steps 8071-8665, Viridian -> Pewter) and `work/plots/dream_wm31_brock.gif` (steps 9123-9587, the gym); real vs dream side by side, every 16 steps re-seeded from the last 4 real frames and run open-loop on the logged actions (every 2nd step shown), probe map and tile drift in the caption.

## Figures (display)
- `work/plots/main_test_id_s3_a0_grid.png`: real vs imagined at k=1..32 (raw 72x80, nearest upscaled).
- `work/videos/main_test_id_k64.mp4`, `work/videos/main_test_ood_k64.mp4` (+ `_sheet.png`): 64-step side-by-side rollouts, raw 72x80.
- Upsampled (native 144x160, labelled "upsampled for display") versions are documented in `upsampler.md`. **All metrics above are computed on raw 72x80 output.**

## Caveats
- One training run, one seed; 16 in-distribution test episodes, hence the wide CIs.
- The probe is trained on the same maps the world model saw; it cannot judge maps absent from training.
- Rollouts follow the logged action sequence of the PPO agent; they are open-loop and never see real frames after the 4-frame context.
- v1/v2 are not compared numerically: v1 was trained on frame-level random splits of all 200 episodes (no held-out episodes exist for it); v2's latent collapsed.


## Drift per leg map (measured)
wm31 on held-out test_ext: 4,096 windows, K=32, 3 sampling steps (`lab.dwm.eval_rollouts --run wm31 --data ext --split test_ext --k 32 --n-windows 4096 --maps 1,12,13,51,2,54,41,58,47,50 --tag permap`; `work/eval/wm31_test_ext_s3_a0_ext_permap.{json,npz}`). Position error is split as in `lab/eval/decompose.py`: only frames whose imagined map is right, next to the no-change baseline on the same windows. Episode means, 95% bootstrap CIs over episodes (`work/eval/drift_permap_decomposed.txt`).

| start map | episodes | wrong map, k=8 | pos. error on the right map, k=8: imagined | no-change | k=32: imagined | no-change | probe on real frames, k=8 |
|---|---|---|---|---|---|---|---|
| Viridian City (1) | 26 | 0.09 [0.06, 0.12] | **2.71** [1.62, 4.12] | 3.64 [3.27, 3.99] | 6.84 [4.15, 10.24] | 6.35 [5.54, 6.99] | 0.02 |
| Route 2 (13) | 19 | 0.18 [0.10, 0.25] | 6.23 [3.04, 10.86] | 4.01 [3.62, 4.54] | 12.17 [8.92, 16.28] | 5.79 [5.07, 6.48] | 0.41 |
| Viridian Forest (51) | 12 | 0.18 [0.12, 0.25] | **15.65** [12.40, 19.34] | 3.81 [2.86, 5.18] | 25.07 [21.89, 27.99] | 5.56 [4.63, 6.49] | 1.96 |
| Pewter City (2) | 18 | 0.25 [0.20, 0.30] | 4.99 [3.71, 6.51] | 3.86 [3.49, 4.38] | 13.80 [12.28, 15.40] | 6.57 [6.03, 7.14] | 0.02 |
| Pewter Gym (54) | 6 | 0.32 [0.11, 0.53] | 1.54 [0.10, 3.91] | 1.12 [0.64, 1.67] | 5.40 [2.16, 9.23] | 1.57 [0.58, 2.56] | 0.13 |
| Viridian PC (41) | 3 | 0.20 [0.00, 0.33] | 0.26 [0.00, 0.60] | 3.50 [2.50, 4.75] | 3.10 [1.40, 6.20] | 3.76 [3.29, 4.60] | |
| Pewter PC (58) | 2 | **0.76** [0.53, 1.00] | 0.48 | 3.11 | 0.17 | 3.50 | |
| Forest north gate (47) | 4 | 0.50 [0.42, 0.61] | 0.31 [0.03, 0.59] | 3.47 [2.82, 4.46] | 2.96 [2.50, 3.57] | 6.13 [5.00, 6.83] | |
| Forest south gate (50) | 11 | 0.35 [0.21, 0.49] | 0.33 [0.05, 0.69] | 3.20 [2.64, 3.83] | 1.17 [0.00, 1.95] | 5.21 [2.19, 9.00] | |

Route 1 (12) has 2 held-out episodes (27 windows) and is left out.

**Reading:**
- **Viridian City, the best-covered map, beats no-change on the right map** (2.71 vs 3.64 tiles at k=8).
- **The forest drifts most.** 15.65 tiles at k=8 on the right map, where the probe's own error on real frames is about 2. So it's the world model, not the measuring probe: it imagines forest that looks plausible but is somewhere else. This fits the forest being the hardest leg, and its stall beside the Antidote ball.
- **Small interiors keep position but lose the map.** Centers and gates stay within half a tile when the map is right. They are read as another map often: the Pewter PC 0.27 at k=1 and 0.76 at k=8, against 0.05 and 0.20 for the Viridian PC. This is the probe-confusion between identical Center interiors that broke leave_pewter_pc (`--merge-centers`, `v3_gym.md`).
- The imagination RL rollouts are 10 steps long, so the k ≤ 8 columns are the ones the policies train on.
