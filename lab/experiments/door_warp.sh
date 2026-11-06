#!/usr/bin/env bash
# ==============================================================================
# Experiment: Building Entrance and Door/Mat Warp Transition Analysis
#
# Reproduces:
#   - Measures imagined vs real map-change transition rates from real train-split
#     contexts on and beside Pokemon Center doors and exit mats (Pewter PC door,
#     Viridian PC door, Pewter PC mat, Viridian PC mat, Pewter south edge).
#   - Diagnostic finding: wm31 imagines building entry after sideways presses beside
#     the door in 65-66% of samples, and the probe misreads Pewter PC mat frames as
#     Viridian PC in 23% of frames. This diagnosis justifies --warp-dist 0 and
#     --merge-centers in imagine_rl.
#
# Results Page & Tables:
#   - agent_notes/PROGRESS.md: lines 496-510 ("Pewter PC legs: reward pays for misreads")
#   - agent_notes/results/v3_gym.md: lines 129-140 ("Why the Pewter PC legs fail")
#   - lab/README.md: Section 1 (lab/dwm/door_warp.py)
#
# Approximate Time:
#   - CPU Time: ~4 min (default --device cpu, 32 contexts x 4 samples per tile)
#   - GPU Time: ~1 min if run on cuda
#
# Outputs Written:
#   - work/eval/door_warp_wm31.json
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
mkdir -p work/eval

echo "=== Running Door Warp and Mat Transition Analysis ==="
$PY -u -m lab.dwm.door_warp \
  --run wm31 \
  --probe work/probe_ext/probe.pt \
  --device cpu

echo "DOOR_WARP_EXPERIMENT_DONE"
