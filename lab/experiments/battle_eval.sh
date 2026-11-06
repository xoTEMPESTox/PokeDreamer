#!/usr/bin/env bash
# ==============================================================================
# Experiment: Battle Onset and In-Battle Fidelity Evaluations
#
# Reproduces:
#   1. Battle onset measurement: evaluates whether wm31 imagines wild battle transitions
#      within 10 steps on tall-grass routes (Route 1 / map 12, Route 2 / map 13,
#      Viridian Forest / map 51).
#   2. Battle fidelity evaluation: measures open-loop rollout MSE across horizons
#      k=1..32 against copy-last baseline and quantifies early battle termination
#      hallucinations (battles ending >3 steps early) over 1,024 random in-battle windows.
#
# Results Page & Tables:
#   - agent_notes/results/v3_battle_fidelity.md:
#       - Table 1: "Battle Onset" (lines 8-13)
#       - Table 2: "Battle Screen Fidelity" (lines 25-42)
#   - lab/README.md: Sections 3.10 and 3.11
#
# GPU Training / Eval Time (from work/chain_battle_eval.sh on RTX 4090):
#   - battle_onset: ~2 min per map (~6 min total for maps 12, 13, 51)
#   - battle_fidelity (1024 windows): ~12 min
#   Total GPU time: ~18 min.
#
# Outputs Written:
#   - work/eval/battle_onset_wm31_map12_ext.json
#   - work/eval/battle_onset_wm31_map13_ext.json
#   - work/eval/battle_onset_wm31_map51_ext.json
#   - work/eval/battle_fidelity_wm31_ext.json
#   - work/plots/battle_fidelity_wm31_ext.png
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
mkdir -p work/eval work/plots

echo "=== Evaluating Battle Onset across Grass Maps (12, 13, 51) ==="
for m in 12 13 51; do
  $PY -u -m lab.dwm.battle_onset \
    --run wm31 \
    --data ext \
    --map "$m"
done

echo "=== Evaluating In-Battle Screen Fidelity (1024 windows) ==="
$PY -u -m lab.dwm.battle_fidelity \
  --run wm31 \
  --data ext \
  --max-windows 1024

echo "BATTLE_EVAL_EXPERIMENT_DONE"
