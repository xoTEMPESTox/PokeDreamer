# Battle simulation fidelity and Path B evaluation (measured)

This page evaluates whether the diffusion world model (`wm31`) accurately simulates Pokémon battles, serving as the empirical go/no-go evaluation for training battle policies in imagination. While the model initiates battles with plausible frequencies on grass routes (matching 53-74% of real onsets), 71.9% of imagined battle windows terminate prematurely (>3 steps early compared to RAM ground truth). Furthermore, battle screen visual quality degrades below the trivial copy-last baseline by step k=4, demonstrating that battle mechanics cannot currently be learned inside the unconditioned pixel world model.

**Reproduce:** Reproducible via [lab/README.md §3.10](../README.md#310-battle-onset-evaluation-wm31-on-tall-grass) (`python -m lab.dwm.battle_onset --run wm31 --data ext --map 13`) and [lab/README.md §3.11](../README.md#311-battle-fidelity-evaluation-inside-battle-screens) (`python -m lab.dwm.battle_fidelity --run wm31 --data ext --max-windows 1024`).

**Figures:**
- `work/figures/fig_battle_fidelity.png`: Battle fidelity evaluation of wm31: (left) battle-end timing categories showing 71.9% hallucinated early ends in imagination vs 0.9% for the real probe ceiling; (right) battle-screen pixel MSE vs horizon k with 95% bootstrap CIs, where copy-last surpasses imagination for k >= 4.

---

Question: could a battle policy be trained inside `wm31`, the way the navigation legs were? Every number below is measured on held-out `test_ext` episodes, with RAM as the ground truth. Run: [`../experiments/battle_eval.sh`](../experiments/battle_eval.sh), log `work/chain_battle_eval.log`.

## 1. Battle onset: `lab.dwm.battle_onset --run wm31 --data ext`
Files: `work/eval/battle_onset_wm31_map{12,13,51}_ext.json`. Settings: horizon 10, 4 samples, 3 denoising steps.

| map | windows (episodes) | battle within 10 steps: RAM / imagined | in-battle frame fraction: RAM / probe on real frames / imagined | imagined onset when a real one happens |
|---|---|---|---|---|
| Route 1 (12) | 136 (3) | 0.49 / 0.40 | 0.289 / 0.235 / **0.099** | 0.74 |
| Route 2 (13) | 7,513 (20) | 0.077 / 0.061 | 0.046 / 0.041 / **0.015** | 0.65 |
| Viridian Forest (51) | 4,048 (14) | 0.120 / 0.120 | 0.069 / 0.065 / **0.029** | 0.53 |

- `wm31` starts a battle in about the right number of windows: when a real battle starts, it imagines one 53-74% of the time.
- **But the imagined battles are short.** The in-battle share of imagined frames is a third to a half of the real share.
- Route 1 has only 3 held-out episodes, so treat its row as anecdotal.

## 2. Inside a battle: `lab.dwm.battle_fidelity --run wm31 --data ext --max-windows 1024`
Files: `work/eval/battle_fidelity_wm31_ext.json`, plot `work/plots/battle_fidelity_wm31_ext.png`.
- Sample: 1,024 random windows whose 4 context frames are all in battle, from 8,913 eligible windows in 29 episodes.
- Settings: horizon 32, 4 samples.

**Battle-end timing, as fractions of windows.** "Real probe" is the probe read on the real frames, which serves as the measurement ceiling.

| | real probe | imagined |
|---|---|---|
| ends within ±3 steps of the RAM end | 0.508 | 0.151 |
| **ends > 3 steps early (hallucinated end)** | 0.009 | **0.719** |
| ends later or never | 0.002 | 0.109 |
| neither ends within 32 steps | 0.481 | 0.020 |
| median absolute timing error (both end) | 0 | 8 steps |

**Pixel MSE on battle frames vs copy-last,** with 95% bootstrap CIs:

| k | imagined | copy-last |
|---|---|---|
| 1 | 0.036 [0.032, 0.040] | 0.040 [0.032, 0.049] |
| 2 | 0.061 [0.056, 0.065] | 0.067 [0.055, 0.079] |
| 4 | 0.096 [0.089, 0.102] | 0.086 [0.073, 0.100] |
| 8 | 0.119 [0.113, 0.125] | 0.087 [0.075, 0.100] |
| 32 | 0.211 [0.207, 0.216] | 0.158 [0.139, 0.177] |

- Party HP in these windows, from the raw files: mean HP fraction 0.727 at the window start and 0.715 at the end; HP dropped in 21% of windows.

## Verdict (measured)
**No-go for training battle policies inside wm31 as it is.**
- **Battles end early.** In 72% of windows the imagined battle ends more than 3 steps before the real one, and the real battle almost never ends that early (0.9%). A policy trained in this world would learn that battles end by themselves within about 8 steps, whatever it presses.
- **Battle screens get worse than copying the last frame.** Imagined battle frames lose to copy-last from k=4 onward, so the model doesn't carry battle state (menus, HP bars) forward.
- **Onset is roughly right, duration is wrong.** That fits a 4-frame context: one battle screen looks like any other, so the model can't tell how far into the battle it is.

Architectural options for Path B (costs are planned estimates, not measured):
1. A battle simulator built from RAM: learn the transition of battle RAM state (HP, moves, turn) from the dataset instead of from pixels. It is cheap, and it measures what a world model would need to know.
2. A pixel world model with longer context or RAM conditioning, retrained on battle windows. This costs GPU hours, and whether it would sustain battles is uncertain.
3. Keep battles as rules (the current System 2) and call Path A finished. The quests already win 30/30 with learned navigation.

## Full held-out sample (measured 2026-11-06)
The same evaluation on all 8,913 eligible in-battle windows (29 test_ext episodes) instead of 1,024 (`lab.dwm.battle_fidelity --run wm31 --data ext --max-windows 0 --tag full`; `work/eval/battle_fidelity_wm31_ext_full.json`, log `work/battle_fidelity_full.log`).

| | real probe | imagined |
|---|---|---|
| ends within ±3 steps of the RAM end | 0.492 | 0.128 |
| **ends > 3 steps early** | 0.011 | **0.746** |
| ends later or never | 0.001 | 0.106 |
| neither ends within 32 steps | 0.496 | 0.019 |
| median absolute timing error (both end) | 0 | 9 steps |

| k | imagined MSE [95% CI] | copy-last MSE [95% CI] |
|---|---|---|
| 1 | 0.0345 [0.0332, 0.0357] | 0.0348 [0.0324, 0.0373] |
| 2 | 0.0629 [0.0614, 0.0645] | 0.0556 [0.0521, 0.0590] |
| 4 | 0.0941 [0.0921, 0.0959] | 0.0785 [0.0742, 0.0829] |
| 8 | 0.1193 [0.1174, 0.1212] | 0.0808 [0.0767, 0.0850] |
| 16 | 0.1617 [0.1597, 0.1634] | 0.1056 [0.1005, 0.1104] |
| 32 | 0.2099 [0.2086, 0.2114] | 0.1431 [0.1367, 0.1490] |

With the full sample the verdict holds and tightens: imagined battles end early in 74.6% of windows, and imagined battle frames are worse than copy-last from k=2 (non-overlapping CIs), not from k=4 as in the 1,024-window sample. At k=1 the two are tied.
