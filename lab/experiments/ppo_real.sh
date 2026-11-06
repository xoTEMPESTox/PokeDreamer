#!/usr/bin/env bash
# ==============================================================================
# Experiment: Model-Free PPO in the Real Game (CPU Baselines)
#
# Reproduces:
#   1. Pallet -> Route 1 -> Viridian City navigation task:
#      - Geodesic reward baseline (viridian_geo): 95/100 successes, median 262 presses
#        in 103k steps (4-5 minutes on CPU with 12 parallel emulators).
#      - Manhattan reward control (viridian).
#   2. 5 travel legs (to_route2, to_gate, forest, to_pewter, to_gym) across 3 seeds
#      (42, 1, 2) with geodesic reward over the extended dataset, matching the reward
#      shape of the world-model imagination RL policies.
#
# Results Page & Tables:
#   - agent_notes/results/v3_agent.md: Table 3 "Real-Game PPO vs World Model" (lines 124-140)
#   - lab/README.md: Section 3.8 and Section 5 note 2
#
# Approximate Time:
#   - GPU Time: 0 min (pure CPU PyBoy emulator parallel workers)
#   - CPU Time: ~4-5 min per run with 12 workers (safety budget 30-90 min).
#
# Outputs Written:
#   - work/ppo_real/viridian_geo/eval.jsonl, work/ppo_real/viridian_geo/policy.pt
#   - work/ppo_real/viridian/eval.jsonl, work/ppo_real/viridian/policy.pt
#   - work/ppo_real/{leg}_geo_s{seed}/eval.jsonl, work/ppo_real/{leg}_geo_s{seed}/policy.pt
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
mkdir -p work/ppo_real

echo "=== 1. Viridian City Task: PPO Geodesic Reward ==="
$PY -u -m lab.agent.ppo_real \
  --task viridian \
  --name viridian_geo \
  --geodesic \
  --cap 1000 \
  --envs 12 \
  --minutes 90 \
  --max-env-steps 3000000

echo "=== 2. Viridian City Task: PPO Manhattan Control ==="
$PY -u -m lab.agent.ppo_real \
  --task viridian \
  --name viridian \
  --cap 1000 \
  --envs 12 \
  --minutes 90 \
  --max-env-steps 3000000

echo "=== 3. 5 Travel Legs across 3 Seeds (42, 1, 2) ==="
for t in to_route2 to_gate forest to_pewter to_gym; do
  for S in 42 1 2; do
    n="${t}_geo_s${S}"
    echo "=== Running PPO on $t (Seed $S) ==="
    $PY -u -m lab.agent.ppo_real \
      --task "$t" \
      --name "$n" \
      --seed "$S" \
      --geodesic \
      --data ext \
      --cap 1000 \
      --envs 12 \
      --minutes 30 \
      --eval-every 5000
  done
done

echo "PPO_REAL_EXPERIMENT_DONE"
