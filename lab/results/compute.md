# Compute and model sizes for v3 (measured)

Everything in v3 ran on one desktop GPU (RTX 4090) and its CPU. This page tallies the training time each run logged when it finished, and the parameter count of each shipped checkpoint.

**Reproduce:** `python -m lab.compute_hours` (reads the run logs in `work/`, which stay local; counting rules in the module docstring, tested in `lab/tests/test_compute_hours.py`). Tallied 2026-11-07.

## Training time (all v3 experiments, including ablations and failed recipes)

| what | device | runs | hours |
|---|---|---|---|
| diffusion world models | GPU | 6 | 8.4 |
| imagination-RL navigation policies | GPU | 66 | 40.4 |
| upsampler | GPU | 2 | 0.8 |
| colorizer (stage 2) | GPU | 2 | 0.4 |
| **training total** | GPU | 76 | **50.0** |
| System 2 LLM quests (bake-off, including aborted runs) | GPU (LLM) + CPU (game) | 61 | 3.9 |
| real-game PPO baseline | CPU | 29 | 3.2 |

**Counting rules:**
- A run is counted from the minutes on its end line. The minutes are cumulative across resumes. When a run extends an earlier one (the it1200/it1800/it3560 policies), only the extra minutes count.
- Not counted:
  - evaluation (frame probe, drift, battle fidelity, real-game quests);
  - data collection;
  - video and 3D rendering;
  - runs that never printed an end line.

## What the shipped pipeline cost

| model | parameters | training |
|---|---|---|
| world model `wm31` (DIAMOND-style EDM U-Net, 72×80 greyscale) | 16.7M | 3.6 h: 63.7 min (`noaug`, the initialisation) + 150.1 min |
| 10 navigation legs (imagination RL, one policy per leg) | 1.58M each | 6.4 h in total, 14.5-48.1 min per leg |
| frame probe (map and position from a frame; used for rewards) | 3.1M | – |
| upsampler 72×80 → 144×160 | 0.72M | 25 min |
| colorizer stage 2 (per-pixel colour class) | 1.91M | 25 min |

Sources:
- **Parameter counts:** floating-point tensors in each checkpoint (`ema` weights for `wm31`, `policy` for the legs, BatchNorm statistics excluded), counted from `work/runs/wm31/ckpt.pt`, `work/agent_rl/to_route2_wm31_gate_s0/policy.pt`, `work/probe_ext/probe.pt`, `work/upsampler/upsampler.pt`, `work/color/learned/model.pt`.
- **Leg minutes:** the end lines of `work/agent_rl/<leg>/train.log` for the 10 legs in [`lab/WEIGHTS.md`](../WEIGHTS.md). `to_pewter_pc_wm31_gate_it1800_s0` is 43.5 min including its 21.8-min base run.
- **Probe training:** not timed.
