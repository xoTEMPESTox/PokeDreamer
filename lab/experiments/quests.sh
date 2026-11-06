#!/usr/bin/env bash
# ==============================================================================
# Experiment: End-to-End Brock Gym Quests (BFS, 5 Gate Legs, 10 Learned Legs)
#
# Reproduces:
#   1. Privileged BFS navigation baseline: 10/10 badges, 8,791 +- 696 mean button presses
#   2. 5 learned travel legs (gate reward, seeds 0-2): 10/10 badges each
#   3. All 10 learned legs (5 travel + 5 movement, best set with to_pewter_pc at
#      1,800 iterations and leave_pewter_pc with --merge-centers, seeds 0-2):
#      10/10 badges each, evaluating full sim-to-real transfer across all overworld
#      navigation routines.
#
# Why all 10 legs are passed explicitly:
#   In lab/agent/gym_quest.py, --leg-policies accepts key=value mappings for leg tasks.
#   If any movement leg (e.g. to_grass, leave_pewter_pc) is omitted from --leg-policies,
#   the quest runner automatically falls back to privileged BFS pathfinding for that
#   omitted leg. Explicitly specifying all 10 legs ensures that every overworld trip—both
#   long travel legs and short grind/heal loops—runs purely using the learned neural
#   network policies trained in imagination.
#
# Results Page & Tables:
#   - agent_notes/results/v3_gym.md:
#       - BFS Baseline Table (lines 15-25)
#       - 5 Gate Legs Table (lines 90-97)
#       - 10 Learned Legs Table (lines 150-175)
#   - lab/README.md: Sections 3.1, 3.2, 3.3
#
# Approximate Time:
#   - GPU Time: 0 min (CPU-only PyBoy emulation)
#   - CPU Time: ~15-20 min per 10-run quest batch (~2 hours total across 7 batches)
#
# Outputs Written:
#   - work/gym_rng/summary.json and run_seed_100.json .. 109.json
#   - work/gym_learned_gateall_s{0..2}/summary.json
#   - work/gym_learned_best_s{0..2}/summary.json
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
P="work/agent_rl"

# 1. Privileged BFS baseline (10 runs, seed 100)
echo "=== Running Brock Gym Quest with BFS Navigation (10 runs) ==="
mkdir -p work/gym_rng
$PY -u -m lab.agent.gym_quest \
  --runs 10 \
  --seed 100 \
  --target-level 14 \
  --nav bfs \
  --work-dir work/gym_rng

# 2. Brock Gym Quest with 5 Learned Travel Legs (seeds 0, 1, 2)
for S in 0 1 2; do
  echo "=== Running Gym Quest with 5 Travel Legs (Seed $S) ==="
  out_dir="work/gym_learned_gateall_s${S}"
  mkdir -p "$out_dir"
  $PY -u -m lab.agent.gym_quest \
    --runs 10 \
    --seed 100 \
    --target-level 14 \
    --nav learned \
    --leg-policies "\
to_route2=$P/to_route2_wm31_gate_s$S/policy.pt,\
to_gate=$P/to_gate_wm31_gate_s$S/policy.pt,\
forest=$P/forest_wm31_gate_s$S/policy.pt,\
to_pewter=$P/to_pewter_wm31_gate_s$S/policy.pt,\
to_gym=$P/to_gym_wm31_gate_s$S/policy.pt" \
    --work-dir "$out_dir"
done

# 3. Brock Gym Quest with All 10 Learned Legs (5 travel + 5 movement, best set, seeds 0, 1, 2)
for S in 0 1 2; do
  echo "=== Running Gym Quest with All 10 Learned Legs (Seed $S) ==="
  out_dir="work/gym_learned_best_s${S}"
  mkdir -p "$out_dir"
  $PY -u -m lab.agent.gym_quest \
    --runs 10 \
    --seed 100 \
    --target-level 14 \
    --nav learned \
    --leg-policies "\
to_route2=$P/to_route2_wm31_gate_s$S/policy.pt,\
to_gate=$P/to_gate_wm31_gate_s$S/policy.pt,\
forest=$P/forest_wm31_gate_s$S/policy.pt,\
to_pewter=$P/to_pewter_wm31_gate_s$S/policy.pt,\
to_gym=$P/to_gym_wm31_gate_s$S/policy.pt,\
to_grass=$P/to_grass_wm31_gate_s$S/policy.pt,\
to_viridian_pc=$P/to_viridian_pc_wm31_gate_s$S/policy.pt,\
leave_viridian_pc=$P/leave_viridian_pc_wm31_gate_s$S/policy.pt,\
to_pewter_pc=$P/to_pewter_pc_wm31_gate_it1800_s$S/policy.pt,\
leave_pewter_pc=$P/leave_pewter_pc_wm31_gate_mc_s$S/policy.pt" \
    --work-dir "$out_dir"
done

echo "QUESTS_EXPERIMENT_DONE"
