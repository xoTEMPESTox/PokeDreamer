#!/usr/bin/env bash
# ==============================================================================
# Experiment: 5 Movement Legs x 3 Seeds (Grind & Heal Navigation)
#
# Reproduces:
#   - 5 movement legs (to_grass, to_viridian_pc, leave_viridian_pc, to_pewter_pc,
#     leave_pewter_pc) across 3 seeds (0, 1, 2)
#   - Trained purely in wm31 world-model imagination with potential-based --move-gate 2
#   - to_pewter_pc: trained to 1,800 iterations (initially 900 iterations, then copied
#     and continued to 1,800).
#     Resume mechanism: imagine_rl restores policy weights, Adam optimizer state, and
#     iteration count with a constant learning rate (3e-4), and its RNG is keyed on
#     [seed, iteration]. Continuing an existing run or a copy is mathematically
#     equivalent to running a single 1,800-iteration job without interruption.
#   - leave_pewter_pc: trained to 600 iterations with --merge-centers.
#     Reason: the probe confuses Pewter PC (map 58) frames on the exit mat with
#     Viridian PC (map 41) in ~23% of frames (see work/eval/door_warp_wm31.json).
#     With --merge-centers, probe reads of either Center count as the start Center,
#     preventing hallucinated warp penalties while stepping off the mat.
#   - Real-game evaluations: 100 episodes, seed 200, 1000 step cap on CPU
#
# Results Page & Tables:
#   - agent_notes/results/v3_gym.md: Table "Movement Legs (grind/heal)"
#   - lab/README.md: Section 3.3
#
# GPU Training Time (from work/agent_rl/<run>/train.log on RTX 4090):
#   - to_grass:          s0: 48.1 min (1800 it), s1: 43.5 min (1800 it), s2: 43.6 min (1800 it)
#   - to_viridian_pc:    s0: 43.1 min (1800 it), s1: 43.7 min (1800 it), s2: 43.4 min (1800 it)
#   - leave_viridian_pc: s0: 14.5 min (600 it),  s1: 14.5 min (600 it),  s2: 14.5 min (600 it)
#   - to_pewter_pc (900): s0: 21.8 min, s1: 21.8 min, s2: 21.8 min
#   - to_pewter_pc (1800): s0: +21.7 min, s1: +21.7 min, s2: +21.7 min (total ~43.5 min/seed)
#   - leave_pewter_pc (mc): s0: 14.5 min (600 it), s1: 14.6 min (600 it), s2: 14.5 min (600 it)
#   Total GPU time: ~483 min (~8.1 hours).
#
# Outputs Written:
#   - Policies:
#       work/agent_rl/to_grass_wm31_gate_s{S}/policy.pt
#       work/agent_rl/to_viridian_pc_wm31_gate_s{S}/policy.pt
#       work/agent_rl/leave_viridian_pc_wm31_gate_s{S}/policy.pt
#       work/agent_rl/to_pewter_pc_wm31_gate_s{S}/policy.pt (900 iters)
#       work/agent_rl/to_pewter_pc_wm31_gate_it1800_s{S}/policy.pt (1800 iters)
#       work/agent_rl/leave_pewter_pc_wm31_gate_mc_s{S}/policy.pt (600 iters, merge-centers)
#   - Training logs: work/agent_rl/<run>/train.log
#   - Eval JSONs: work/agent/{leg}_learned_{name}_s200.json
#   - Eval logs: work/agent/eval_{name}.log
# ==============================================================================
set -euo pipefail

PY=${PY:-python}
P="work/agent_rl"

mkdir -p "$P" work/agent

for S in 0 1 2; do
  # 1. to_grass (1,800 iterations)
  n="to_grass_wm31_gate_s${S}"
  echo "=== Training $n (1800 iters) ==="
  $PY -u -m lab.agent.imagine_rl \
    --task to_grass \
    --name "$n" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --minutes 120 \
    --max-iters 1800
  $PY -u -m lab.agent.mpc --task to_grass --policy learned --rl-name "$n" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n}.log" 2>&1

  # 2. to_viridian_pc (1,800 iterations)
  n="to_viridian_pc_wm31_gate_s${S}"
  echo "=== Training $n (1800 iters) ==="
  $PY -u -m lab.agent.imagine_rl \
    --task to_viridian_pc \
    --name "$n" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --minutes 120 \
    --max-iters 1800
  $PY -u -m lab.agent.mpc --task to_viridian_pc --policy learned --rl-name "$n" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n}.log" 2>&1

  # 3. leave_viridian_pc (600 iterations)
  n="leave_viridian_pc_wm31_gate_s${S}"
  echo "=== Training $n (600 iters) ==="
  $PY -u -m lab.agent.imagine_rl \
    --task leave_viridian_pc \
    --name "$n" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --minutes 120 \
    --max-iters 600
  $PY -u -m lab.agent.mpc --task leave_viridian_pc --policy learned --rl-name "$n" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n}.log" 2>&1

  # 4. to_pewter_pc (900 iterations initial, then copy and continue to 1,800 iterations)
  n_base="to_pewter_pc_wm31_gate_s${S}"
  echo "=== Training $n_base (900 iters) ==="
  $PY -u -m lab.agent.imagine_rl \
    --task to_pewter_pc \
    --name "$n_base" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --minutes 120 \
    --max-iters 900
  $PY -u -m lab.agent.mpc --task to_pewter_pc --policy learned --rl-name "$n_base" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n_base}.log" 2>&1

  n_1800="to_pewter_pc_wm31_gate_it1800_s${S}"
  echo "=== Continuing $n_1800 to 1,800 iters from $n_base ==="
  # Copy checkpoint directory so 900-iteration control remains intact.
  # Resume restores policy, optimizer, and iteration count; continuing a copy
  # is mathematically equivalent to running to 1,800 in one go.
  [ -d "$P/$n_1800" ] || cp -r "$P/$n_base" "$P/$n_1800"
  $PY -u -m lab.agent.imagine_rl \
    --task to_pewter_pc \
    --name "$n_1800" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --minutes 120 \
    --max-iters 1800
  $PY -u -m lab.agent.mpc --task to_pewter_pc --policy learned --rl-name "$n_1800" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n_1800}.log" 2>&1

  # 5. leave_pewter_pc with --merge-centers (600 iterations)
  n_mc="leave_pewter_pc_wm31_gate_mc_s${S}"
  echo "=== Training $n_mc (600 iters with --merge-centers) ==="
  $PY -u -m lab.agent.imagine_rl \
    --task leave_pewter_pc \
    --name "$n_mc" \
    --seed "$S" \
    --run wm31 \
    --probe work/probe_ext/probe.pt \
    --data ext \
    --geodesic \
    --move-gate 2 \
    --merge-centers \
    --minutes 120 \
    --max-iters 600
  $PY -u -m lab.agent.mpc --task leave_pewter_pc --policy learned --rl-name "$n_mc" --episodes 100 --max-steps 1000 --seed 200 --device cpu > "work/agent/eval_${n_mc}.log" 2>&1
done

echo "MOVEMENT_LEGS_EXPERIMENT_DONE"
