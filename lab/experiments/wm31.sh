#!/usr/bin/env bash
# ==============================================================================
# Experiment: Diffusion World Model wm31 (Training, Rollouts, and Decomposition)
#
# Reproduces:
#   1. Dataset extended cache build (Route 2, Viridian Forest, Pewter City)
#   2. FrameProbe retraining on extended dataset
#   3. DIAMOND-style EDM Diffusion World Model (wm31) training for 150 min on RTX 4090
#   4. 32-step open-loop rollout evaluation on held-out test windows (test_ext)
#   5. Error decomposition into wrong-map hallucinations vs in-map drift
#
# Results Page & Tables:
#   - agent_notes/results/v3_dwm_eval.md: Table 1 "Rollout Metrics", Table 2 "Error Decomposition"
#   - lab/README.md: Section 3.9
#
# Approximate Time (from work/runs/wm31/train.log on RTX 4090):
#   - probe_ext training: 10.4 min (622 s on 8 CPU threads; work/probe_ext/probe_metrics.json)
#   - wm31 training: 150.1 min (114,800 steps; work/runs/wm31/train.log:613)
#   - eval_rollouts (512 windows, k=32): ~5 min
#   Total GPU time: ~155 min.
#
# Outputs Written:
#   - Dataset cache: work/cache_ext_72x80.npz, work/splits_ext.json
#   - Probe weights: work/probe_ext/probe.pt, work/probe_ext/probe_metrics.json
#   - Model checkpoint: work/runs/wm31/ckpt.pt, work/runs/wm31/train.log
#   - Rollout evals: work/eval/wm31_test_ext_s3_a0_ext.npz, work/eval/wm31_test_ext_s3_a0_ext.json
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
mkdir -p work/runs work/eval work/probe_ext

echo "=== 1. Building Extended Dataset Cache ==="
$PY -u -m lab.data --ext

echo "=== 2. Training FrameProbe on Extended Dataset ==="
$PY -u -m lab.probe --data ext > work/probe_ext.log 2>&1

echo "=== 3. Training Diffusion World Model (wm31) ==="
$PY -u -m lab.dwm.train \
  --name wm31 \
  --data ext \
  --init-from work/runs/noaug/ckpt.pt \
  --no-ctx-aug \
  --minutes 150

echo "=== 4. Evaluating 32-Step Rollouts on Held-Out Windows ==="
$PY -u -m lab.dwm.eval_rollouts \
  --run wm31 \
  --data ext \
  --split test_ext \
  --n-windows 512 \
  --k 32 \
  --steps 3 \
  --device cuda

echo "=== 5. Decomposing Position Errors ==="
$PY -u -m lab.eval.decompose work/eval/wm31_test_ext_s3_a0_ext.npz

echo "WM31_EXPERIMENT_DONE"
