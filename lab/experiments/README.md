# PokeDreamer v3 Experiment Scripts (`lab/experiments/`)

This directory contains standalone, reproducible bash scripts for all experimental results in PokeDreamer v3. Each script is self-contained, adheres to `set -euo pipefail`, uses relative paths from the repository root, and is parameterized by `PY=${PY:-python}`.

All machine-specific hooks (such as `work/PAUSE` polling loops, background waits across separate bash chains, and hardcoded conda paths) have been removed.

---

## 1. Experiment Scripts Overview

| Script | What It Reproduces | Target Results / Tables | Approx. Time | Key Outputs |
|---|---|---|---|---|
| [`travel_legs.sh`](travel_legs.sh) | 5 travel legs (`to_route2`, `to_gate`, `forest`, `to_pewter`, `to_gym`) across 3 seeds (0, 1, 2) trained with potential-based `--move-gate 2`, plus 100-episode real-game evaluations and quest-start tile evaluations | [`agent_notes/results/v3_gym.md`](../../agent_notes/results/v3_gym.md) (Learned Navigation Legs); [`lab/README.md`](../README.md#32-brock-gym-quest-with-5-learned-world-model-travel-legs) §3.2, §3.4 | ~11.2 hr GPU (~44–48 min / leg on RTX 4090) | `work/agent_rl/{leg}_wm31_gate_s{0..2}/policy.pt`, `work/agent/eval_*.log` |
| [`movement_legs.sh`](movement_legs.sh) | 5 grind/heal movement legs (`to_grass`, `to_viridian_pc`, `leave_viridian_pc`, `to_pewter_pc`, `leave_pewter_pc`) across 3 seeds (0, 1, 2), including `to_pewter_pc` continued to 1,800 iterations and `leave_pewter_pc` with `--merge-centers` | [`agent_notes/results/v3_gym.md`](../../agent_notes/results/v3_gym.md) (Movement Legs); [`lab/README.md`](../README.md#33-brock-gym-quest-with-all-10-learned-legs-5-travel--5-movement) §3.3 | ~8.1 hr GPU on RTX 4090 | `work/agent_rl/{leg}_wm31_gate_*/policy.pt`, `work/agent/eval_*.log` |
| [`quests.sh`](quests.sh) | End-to-end Brock Gym Quests: privileged BFS baseline (10 runs, 10/10 badges), 5 learned travel legs (seeds 0–2, 10 runs each), and all 10 learned legs (seeds 0–2, 10 runs each with all 10 legs explicitly passed) | [`agent_notes/results/v3_gym.md`](../../agent_notes/results/v3_gym.md) (Tables 1–3); [`lab/README.md`](../README.md#31-brock-gym-quest-1010-badges-with-bfs-navigation) §3.1, §3.2, §3.3 | ~2 hr CPU (0 min GPU) | `work/gym_rng/summary.json`, `work/gym_learned_gateall_s{0..2}/summary.json`, `work/gym_learned_best_s{0..2}/summary.json` |
| [`battle_eval.sh`](battle_eval.sh) | Battle onset evaluation across tall-grass maps (12, 13, 51) and 1,024-window in-battle screen fidelity evaluation (rollout MSE k=1..32 vs copy-last, early battle termination measurement) | [`agent_notes/results/v3_battle_fidelity.md`](../../agent_notes/results/v3_battle_fidelity.md) (Tables 1 & 2); [`lab/README.md`](../README.md#310-battle-onset-evaluation-wm31-on-tall-grass) §3.10, §3.11 | ~18 min GPU on RTX 4090 | `work/eval/battle_onset_*.json`, `work/eval/battle_fidelity_wm31_ext.json`, `work/plots/battle_fidelity_wm31_ext.png` |
| [`door_warp.sh`](door_warp.sh) | Building entrance and Pokemon Center door/mat warp transition analysis comparing real vs imagined map-change rates per direction across 32 contexts × 4 samples per tile | [`agent_notes/PROGRESS.md`](../../agent_notes/PROGRESS.md#L496-L510); [`agent_notes/results/v3_gym.md`](../../agent_notes/results/v3_gym.md#L129-L140); [`lab/README.md`](../README.md#1-directory--module-layout) §1 | ~4 min CPU (~1 min GPU) | `work/eval/door_warp_wm31.json` |
| [`ppo_real.sh`](ppo_real.sh) | Model-free PPO baseline in the real Game Boy emulator: Viridian City task (geodesic vs Manhattan reward) and 5 travel legs across 3 seeds (42, 1, 2) with 12 parallel CPU workers | [`agent_notes/results/v3_agent.md`](../../agent_notes/results/v3_agent.md) (Table 3); [`lab/README.md`](../README.md#38-model-free-ppo-in-the-real-game-viridian-geodesic) §3.8, §5 note 2 | ~45–60 min CPU (0 min GPU) | `work/ppo_real/viridian_geo/eval.jsonl`, `work/ppo_real/{leg}_geo_s{seed}/eval.jsonl` |
| [`wm31.sh`](wm31.sh) | Dataset extended cache build, FrameProbe retraining, DIAMOND diffusion world model (wm31) 150-minute training, 32-step rollout evaluation, and position error decomposition | [`agent_notes/results/v3_dwm_eval.md`](../../agent_notes/results/v3_dwm_eval.md) (Tables 1 & 2); [`lab/README.md`](../README.md#39-diffusion-world-model-rollout-evaluation--error-decomposition) §3.9 | ~155 min GPU + ~10 min CPU | `work/runs/wm31/ckpt.pt`, `work/eval/wm31_test_ext_s3_a0_ext.npz`, `work/eval/wm31_test_ext_s3_a0_ext.json` |
| [`headline_rl.sh`](headline_rl.sh) | Headline imagination RL tasks: `leave_house` (100/100, median 28 presses), `route1` (100/100, median 22 presses), and `viridian` (100/100, median 284 presses) | [`agent_notes/results/v3_agent.md`](../../agent_notes/results/v3_agent.md) (Tables 1 & 2); [`lab/README.md`](../README.md#35-headline-imagination-rl-leave-house) §3.5, §3.6, §3.7 | ~111 min GPU on RTX 4090 | `work/agent_rl/{leave_house,route1,viridian_geo_clip}/policy.pt`, `work/agent/*.json` |

---

## 2. Usage Instructions

All scripts must be executed from the **repository root**:

```bash
# Set Python interpreter if not using system default 'python':
# export PY="/path/to/conda/envs/pokemon-rl/bin/python"
# or
export PY="python"  # with active pokemon-rl conda environment

# Run an experiment script:
bash lab/experiments/travel_legs.sh
bash lab/experiments/movement_legs.sh
bash lab/experiments/quests.sh
bash lab/experiments/battle_eval.sh
bash lab/experiments/door_warp.sh
bash lab/experiments/ppo_real.sh
bash lab/experiments/wm31.sh
bash lab/experiments/headline_rl.sh
```

---

## 3. Methodological Notes

1. **Resume Mechanism in `movement_legs.sh` (`to_pewter_pc`):**
   `to_pewter_pc` at 900 iterations scored 32/100 in the real game due to undertraining near the door. The script trains the 900-iteration baseline under `work/agent_rl/to_pewter_pc_wm31_gate_s$S`, copies it to `work/agent_rl/to_pewter_pc_wm31_gate_it1800_s$S`, and continues it to 1,800 iterations.
   Because `lab/agent/imagine_rl.py` restores the policy weights, Adam optimizer state, and iteration counter with a constant learning rate ($3 \times 10^{-4}$) and seeds its data RNG on `[seed, iteration]`, continuing from a checkpoint copy is mathematically identical to a single continuous 1,800-iteration run.

2. **Probe Confusion and `--merge-centers` (`leave_pewter_pc`):**
   `lab/dwm/door_warp.py` measured that the FrameProbe misreads Pewter Pokémon Center (map 58) frames on the exit mat as Viridian Pokémon Center (map 41) in 23% of cases. Passing `--merge-centers` treats map 41 reads as the current Center, preventing spurious $-2.0$ hallucinated warp penalties when stepping towards the door.

3. **Explicit 10-Leg Specification in `quests.sh`:**
   In `lab/agent/gym_quest.py`, `--leg-policies` parses key-value pairs for each leg task. Any task omitted defaults to privileged BFS. Passing all 10 legs explicitly (`to_route2`, `to_gate`, `forest`, `to_pewter`, `to_gym`, `to_grass`, `to_viridian_pc`, `leave_viridian_pc`, `to_pewter_pc`, `leave_pewter_pc`) ensures full learned-policy execution for both overworld travel and inner grind/heal pacing.
