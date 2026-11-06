#!/usr/bin/env bash
# ==============================================================================
# Experiment: Headline Imagination RL Tasks (Leave House, Route 1, Viridian)
#
# Reproduces:
#   1. leave_house: 100/100 real-game successes, median 28 button presses
#      (zero-shot from 4 screen frames)
#   2. route1: 100/100 real-game successes, median 22 button presses
#   3. viridian (Pallet Town -> Route 1 -> Viridian City): 100/100 real-game successes,
#      median 284 button presses (trained with geodesic reward + step clipping)
#
# Results Page & Tables:
#   - agent_notes/results/v3_agent.md:
#       - Table 1: "Headline Result: Leave House" (lines 40-50)
#       - Table 2: "Harder Task: Route 1" (lines 60-70)
#       - Section: "Pallet Town -> Route 1 -> Viridian City" (lines 100-120)
#   - lab/README.md: Sections 3.5, 3.6, 3.7
#
# GPU Training Time (from work/agent_rl/<run>/train.log on RTX 4090):
#   - leave_house:       30.7 min (640 iters)
#   - route1:            35.1 min (1440 iters)
#   - viridian_geo_clip: 45.3 min (1820 iters)
#   Total GPU time: ~111 min.
#
# Outputs Written:
#   - Policies: work/agent_rl/{task}/policy.pt
#   - Training logs: work/agent_rl/{task}/train.log
#   - Eval JSONs:
#       work/agent/leave_house_learned_leave_house_s200.json
#       work/agent/route1_learned_route1_s200.json
#       work/agent/viridian_learned_viridian_geo_clip_s200.json
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
mkdir -p work/agent_rl work/agent

# 1. Leave House (Red's bedroom -> Pallet Town)
echo "=== Training leave_house in imagination ==="
$PY -u -m lab.agent.imagine_rl \
  --task leave_house \
  --name leave_house \
  --minutes 30

echo "=== Evaluating leave_house on real game ==="
$PY -u -m lab.agent.mpc \
  --task leave_house \
  --policy learned \
  --rl-name leave_house \
  --episodes 100 \
  --max-steps 60 \
  --seed 200 \
  --device cpu

# 2. Route 1 (Pallet Town -> North entrance of Route 1)
echo "=== Training route1 in imagination ==="
$PY -u -m lab.agent.imagine_rl \
  --task route1 \
  --name route1 \
  --minutes 30

echo "=== Evaluating route1 on real game ==="
$PY -u -m lab.agent.mpc \
  --task route1 \
  --policy learned \
  --rl-name route1 \
  --episodes 100 \
  --max-steps 100 \
  --seed 200 \
  --device cpu

# 3. Viridian City (Pallet Town -> Route 1 -> Viridian City, Geodesic + Clip)
echo "=== Training viridian_geo_clip in imagination ==="
$PY -u -m lab.agent.imagine_rl \
  --task viridian \
  --name viridian_geo_clip \
  --minutes 45 \
  --geodesic \
  --step-clip 1

echo "=== Evaluating viridian_geo_clip on real game ==="
$PY -u -m lab.agent.mpc \
  --task viridian \
  --policy learned \
  --rl-name viridian_geo_clip \
  --episodes 100 \
  --max-steps 1000 \
  --seed 200 \
  --device cpu

echo "HEADLINE_RL_EXPERIMENT_DONE"
