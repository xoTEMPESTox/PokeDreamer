#!/usr/bin/env bash
# ==============================================================================
# Experiment: 5 Travel Legs x 3 Seeds (Gate Reward)
#
# Reproduces:
#   - 5 travel legs (to_route2, to_gate, forest, to_pewter, to_gym) across 3 seeds (0, 1, 2)
#   - Trained purely in wm31 world-model imagination with potential-based --move-gate 2
#   - Standard real-game evaluation: 100 episodes, seed 200, 1000 step cap on CPU
#   - Start-tile quest evaluations for to_route2 and to_gym from the exact tiles where
#     the gym quest hands off control
#
# Results Page & Tables:
#   - agent_notes/results/v3_gym.md: Table "Learned Navigation Legs (seeds 0, 1, 2)"
#   - lab/README.md: Section 3.2 and Section 3.4
#
# GPU Training Time (from work/agent_rl/<run>/train.log on RTX 4090):
#   - to_route2: s0: 44.2 min (1800 it), s1: 44.3 min (1800 it), s2: 44.3 min (1800 it)
#   - to_gate:   s0: 43.9 min (1780 it), s1: 43.8 min (1780 it), s2: 43.8 min (1780 it)
#   - forest:    s0: 44.0 min (1780 it), s1: 44.1 min (1780 it), s2: 48.2 min (1780 it)
#   - to_pewter: s0: 44.7 min (1820 it), s1: 44.9 min (1820 it), s2: 44.3 min (1820 it)
#   - to_gym:    s0: 45.7 min (1860 it), s1: 46.1 min (1860 it), s2: 45.3 min (1860 it)
#   Total GPU time: ~672 min (~11.2 hours); ~44-48 min per leg.
#
# Outputs Written:
#   - Policies: work/agent_rl/{leg}_wm31_gate_s{S}/policy.pt
#   - Training logs: work/agent_rl/{leg}_wm31_gate_s{S}/train.log
#   - Eval JSONs: work/agent/{leg}_learned_{leg}_wm31_gate_s{S}_s200.json
#   - Eval logs: work/agent/eval_{leg}_wm31_gate_s{S}.log
#   - Quest start eval logs: work/agent/eval_{leg}_wm31_gate_s{S}_quest_start.log
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
P="work/agent_rl"

mkdir -p "$P" work/agent

# Iteration budgets matched across all runs (derived from work/chain_gate_legs.sh)
declare -A ITERS=(
  [to_route2]=1800
  [to_gate]=1780
  [forest]=1780
  [to_pewter]=1820
  [to_gym]=1860
)

for S in 0 1 2; do
  for t in to_route2 to_gate forest to_pewter to_gym; do
    n="${t}_wm31_gate_s${S}"
    echo "=== Training $n (iterations: ${ITERS[$t]}) ==="
    $PY -u -m lab.agent.imagine_rl \
      --task "$t" \
      --name "$n" \
      --seed "$S" \
      --run wm31 \
      --probe work/probe_ext/probe.pt \
      --data ext \
      --geodesic \
      --move-gate 2 \
      --minutes 120 \
      --max-iters "${ITERS[$t]}"

    echo "=== Evaluating $n on real game (standard) ==="
    $PY -u -m lab.agent.mpc \
      --task "$t" \
      --policy learned \
      --rl-name "$n" \
      --episodes 100 \
      --max-steps 1000 \
      --seed 200 \
      --device cpu > "work/agent/eval_${n}.log" 2>&1

    # Quest-start state evaluations for legs whose start tile differs from the task default
    if [ "$t" = "to_route2" ]; then
      echo "=== Evaluating $n from quest start tile ==="
      $PY -u -m lab.agent.mpc \
        --task "$t" \
        --policy learned \
        --rl-name "$n" \
        --episodes 100 \
        --max-steps 1000 \
        --seed 200 \
        --device cpu \
        --state work/route_states/quest_to_route2_map1_23_26.state > "work/agent/eval_${n}_quest_start.log" 2>&1
    elif [ "$t" = "to_gym" ]; then
      echo "=== Evaluating $n from quest start tile ==="
      $PY -u -m lab.agent.mpc \
        --task "$t" \
        --policy learned \
        --rl-name "$n" \
        --episodes 100 \
        --max-steps 1000 \
        --seed 200 \
        --device cpu \
        --state work/route_states/quest_to_gym_map2_13_26.state > "work/agent/eval_${n}_quest_start.log" 2>&1
    fi
  done
done

echo "TRAVEL_LEGS_EXPERIMENT_DONE"
