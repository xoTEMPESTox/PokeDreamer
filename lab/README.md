# PokeDreamer v3 Lab (`lab/`)

This directory contains the experimental codebase for PokeDreamer v3: a DIAMOND-style diffusion world model of Pokémon Red, an agent whose navigation policies are trained entirely inside the model's imagination, and a dual System 1 / System 2 architecture tested in the real game on a Game Boy emulator (PyBoy).

---

## 1. Directory & Module Layout

| Path / Module | Type | Description |
|---|---|---|
| [`lab/`](../lab) | Package | Root experimental package for PokeDreamer v3. |
| [`lab/data.py`](../lab/data.py) | Module | Builds 72×80 greyscale frame caches, episode-level train/test splits, and dataset statistics from native-resolution transitions (`transitions_*.npz`). |
| [`lab/env.py`](../lab/env.py) | Module | Headless PyBoy emulator wrapper: observation downsampling (72×80 greyscale), RAM state extraction, save state generation, and fidelity checks against recorded frames. |
| [`lab/probe.py`](../lab/probe.py) | Module | CNN FrameProbe architecture and training: reads RAM labels (map ID, (x, y) coordinates, battle state) from a single 72×80 greyscale frame. |
| [`lab/upsampler.py`](../lab/upsampler.py) | Module | 0.7M-parameter CNN upsampler restoring 72×80 greyscale frames back to native 144×160 4-shade Game Boy frames (99.98% pixel accuracy). |
| [`lab/get_weights.py`](../lab/get_weights.py) | Script | Downloads published model weights, leg policies, and route save states from the Hugging Face Model Hub. |
| [`lab/make_starter_state.py`](../lab/make_starter_state.py) | Module | Deterministically boots the ROM through the intro and rival battle to generate `starter_pallet.state` and `starter_lab.state`. |
| [`lab/make_pokedex_state.py`](../lab/make_pokedex_state.py) | Module | Boots the ROM through Oak's Parcel quest to create `pokedex_viridian.state` (Charmander L8 in Viridian City with Pokédex and north road open). |
| [`lab/make_route_states.py`](../lab/make_route_states.py) | Module | Generates route save states across quest locations for evaluation and training. |
| [`lab/collect.py`](../lab/collect.py) | Module | Collects new exploratory gameplay trajectories in new areas (Route 2, Viridian Forest, Pewter City) into transition NPZ files. |
| [`lab/play.py`](../lab/play.py) | Module | Interactive PyBoy environment runner allowing manual keyboard play while logging actions and RAM states. |
| [`lab/bench_speed.py`](../lab/bench_speed.py) | Module | Benchmarks emulator stepping speed, observation downsampling, and world model inference latency. |
| [`lab/v1_reeval.py`](../lab/v1_reeval.py) | Module | Re-evaluates v1 VAE and dynamics baselines under the v3 evaluation protocol (test splits and no-change baseline). |
| [`lab/agent/`](../lab/agent) | Package | System 1 and System 2 agent algorithms, planning, navigation, and quests. |
| [`lab/agent/gym_quest.py`](../lab/agent/gym_quest.py) | Module | End-to-end Brock gym quest runner: orchestrates grind, heal, travel, and battle phases to earn the Boulder Badge in the real game. |
| [`lab/agent/imagine_rl.py`](../lab/agent/imagine_rl.py) | Module | Actor-critic RL policy training entirely inside the diffusion world model's imagination using probe-read progress and potential-based move gate rewards. |
| [`lab/agent/mpc.py`](../lab/agent/mpc.py) | Module | Model Predictive Control (MPC) planner in world-model imagination, and standard real-game policy evaluator (`--policy learned`). |
| [`lab/agent/ppo_real.py`](../lab/agent/ppo_real.py) | Module | Model-free PPO baseline trained directly in the real Game Boy emulator across multiple parallel CPU workers. |
| [`lab/agent/catch_quest.py`](../lab/agent/catch_quest.py) | Module | Evaluates Pokémon catching strategies in the real game (throw at full HP vs weaken first). |
| [`lab/agent/system12.py`](../lab/agent/system12.py) | Module | Integrated System 1/2 leveling and healing loop testing heal-threshold trade-offs in Route 1. |
| [`lab/agent/llm_planner.py`](../lab/agent/llm_planner.py) | Module | Local LLM System 2 planner integration (via Ollama) with strict RAM validation and fallback handling. |
| [`lab/agent/nav.py`](../lab/agent/nav.py) | Module | Privileged BFS navigation reading tile collision lists and map blocks directly from ROM and RAM. |
| [`lab/agent/tasks.py`](../lab/agent/tasks.py) | Module | Task definitions, target maps, exit coordinates, start states, and geodesic distance fields. |
| [`lab/agent/tile_sweep.py`](../lab/agent/tile_sweep.py) | Module | Start-tile robustness sweep evaluating leg policies across stratified reachable start coordinates. |
| [`lab/agent/route_exits.py`](../lab/agent/route_exits.py) | Module | Extracts map exit coordinates and transitions from recorded dataset episodes. |
| [`lab/agent/record.py`](../lab/agent/record.py) | Module | Records real-game policy rollout trajectories and actions. |
| [`lab/agent/make_move_states.py`](../lab/agent/make_move_states.py) | Module | Generates initial save states for movement legs (`to_grass`, `to_viridian_pc`, `to_pewter_pc`). |
| [`lab/dwm/`](../lab/dwm) | Package | DIAMOND-style Diffusion World Model implementation, training, and evaluations. |
| [`lab/dwm/model.py`](../lab/dwm/model.py) | Module | EDM denoiser U-Net architecture (16.7M params), multi-frame conditioning, and Euler sampler. |
| [`lab/dwm/train.py`](../lab/dwm/train.py) | Module | Diffusion world model training loop with EMA weight updates and GameNGen context-noise augmentation. |
| [`lab/dwm/eval_rollouts.py`](../lab/dwm/eval_rollouts.py) | Module | Evaluates open-loop imagined rollouts (1–32 steps) against real held-out frames (pixel MSE, position error, map accuracy). |
| [`lab/dwm/controllability.py`](../lab/dwm/controllability.py) | Module | Action controllability evaluation: measures signed displacement and orthogonal drift under commanded button sequences. |
| [`lab/dwm/real_control.py`](../lab/dwm/real_control.py) | Module | Directly compares imagined button rolls against identical real emulator button presses from the same 4 real context frames. |
| [`lab/dwm/battle_onset.py`](../lab/dwm/battle_onset.py) | Module | Measures whether wm31 imagines wild battle onset within 10 steps when walking on tall-grass routes. |
| [`lab/dwm/battle_fidelity.py`](../lab/dwm/battle_fidelity.py) | Module | Evaluates imagined battle screens: battle duration, early battle termination, and pixel MSE against copy-last baseline. |
| [`lab/dwm/hybrid.py`](../lab/dwm/hybrid.py) | Module | RAM-conditioned hybrid world model: symbolic RAM state dynamics (1.3M params) conditioning the diffusion renderer. |
| [`lab/dwm/train_hybrid.py`](../lab/dwm/train_hybrid.py) | Module | Training script for the symbolic state predictor and state-conditioned diffusion denoiser. |
| [`lab/dwm/eval_hybrid.py`](../lab/dwm/eval_hybrid.py) | Module | Evaluates symbolic state tracking and rendered frame quality of the hybrid model. |
| [`lab/dwm/door_warp.py`](../lab/dwm/door_warp.py) | Module | Analyzes warp transitions and door exit rendering in imagined rollouts. |
| [`lab/dwm/make_video.py`](../lab/dwm/make_video.py) | Module | Renders side-by-side MP4 comparison videos between real emulator play and imagined rollouts. |
| [`lab/dwm/dream_gif.py`](../lab/dwm/dream_gif.py) | Module | Generates looping animated GIFs of open-loop imagined world model dreams. |
| [`lab/eval/`](../lab/eval) | Package | Statistical evaluation metrics and error decomposition utilities. |
| [`lab/eval/metrics.py`](../lab/eval/metrics.py) | Module | Cluster bootstrap confidence intervals (over episodes), MSE, Manhattan tile error, and classification accuracy. |
| [`lab/eval/decompose.py`](../lab/eval/decompose.py) | Module | Decomposes imagined position error into wrong-map hallucinations vs drift on the correct map. |
| [`lab/viz/`](../lab/viz) | Package | Visualization scripts for report figures and 3D projection renders. |
| [`lab/tests/`](../lab/tests) | Package | Automated test suite (166 unit and integration tests across all modules). |

---

## 2. Setup

### 2.1 Python & Conda Environment

Activate the designated environment (`pokemon-rl`, Python 3.11):
```bash
conda activate pokemon-rl
```
All commands must be executed from the **repository root** (`PokeDreamer/`).

### 2.2 Pokémon Red ROM

Place a legally dumped, headered Pokémon Red (USA/Europe) Game Boy ROM in the repo root named:
```
Pokemon - Red Version (USA, Europe).gb
```
- Expected SHA-1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`
- Configured in [`lab/env.py:ROM_PATH`](../lab/env.py#L23).

Generate the initial save states from the ROM:
```bash
# Red's bedroom (saves/bedroom.state) and Pallet Town (saves/pallet.state)
python -m lab.env --make-states

# Starter states with Charmander after rival battle (saves/starter_pallet.state)
python -m lab.make_starter_state

# Viridian City with Pokédex (saves/pokedex_viridian.state)
python -m lab.make_pokedex_state
```

### 2.3 Dataset Cache Build

The world model and probe train on 72×80 greyscale transitions built from the Hugging Face dataset [`xxxTEMPESTxxx/PokeDreamer`](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer). If you have downloaded or collected transition NPZs into `work/`:
```bash
# Base dataset cache (139 train / 16 test_id / 45 test_ood episodes)
# Writes: work/cache_72x80.npz, work/splits.json, work/dataset_stats.md
python -m lab.data

# Extended dataset cache (including Route 2, Viridian Forest, Pewter City)
# Writes: work/cache_ext_72x80.npz, work/splits_ext.json
python -m lab.data --ext
```

### 2.4 Model Weights & Save States

Use [`lab/get_weights.py`](../lab/get_weights.py) to inspect and download published weights and route save states from Hugging Face ([`xxxTEMPESTxxx/PokeDreamer`](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer)):
```bash
# List all manifest items and check local status on disk:
python lab/get_weights.py --list

# Download all manifest files:
python lab/get_weights.py

# Or download specific categories/models:
python lab/get_weights.py --only wm31
python lab/get_weights.py --only legs
python lab/get_weights.py --only ppo
python lab/get_weights.py --only route_states
```
For a complete manifest of file sizes, paths, and result mappings, see [`lab/WEIGHTS.md`](../lab/WEIGHTS.md).

---

## 3. Reproduction Commands (Public Results)

Each command below is copied directly from the production chain scripts (`work/chain_*.sh`) and maps to a specific result table in `lab/results/`.

### 3.1 Brock Gym Quest: 10/10 Badges with BFS Navigation
- **Command:**
  ```bash
  python -m lab.agent.gym_quest --runs 10 --seed 100 --target-level 14 --nav bfs --work-dir work/gym_rng
  ```
- **Output Files:** `work/gym_rng/summary.json`, `work/gym_rng/run_seed_100.json` .. `109.json`
- **Result Table:** [`lab/results/v3_gym.md`](results/v3_gym.md) (10/10 badges, 8,791 ± 696 mean presses).

### 3.2 Brock Gym Quest with 5 Learned World-Model Travel Legs
- **Command (from [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L18-L21)):**
  ```bash
  python -m lab.agent.gym_quest --runs 10 --seed 100 --target-level 14 --nav learned \
    --leg-policies "to_route2=work/agent_rl/to_route2_wm31_gate_s0/policy.pt,to_gate=work/agent_rl/to_gate_wm31_gate_s0/policy.pt,forest=work/agent_rl/forest_wm31_gate_s0/policy.pt,to_pewter=work/agent_rl/to_pewter_wm31_gate_s0/policy.pt,to_gym=work/agent_rl/to_gym_wm31_gate_s0/policy.pt" \
    --work-dir work/gym_learned_gateall_s0
  ```
- **Output Files:** `work/gym_learned_gateall_s0/summary.json`
- **Result Table:** [`lab/results/v3_gym.md`](results/v3_gym.md) (10/10 badges; 9/10 runs with zero travel fallback).

### 3.3 Brock Gym Quest with All 10 Learned Legs (5 Travel + 5 Movement)
- **Command (from [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh#L27-L29)):**
  ```bash
  python -m lab.agent.gym_quest --runs 10 --seed 100 --target-level 14 --nav learned \
    --leg-policies "to_route2=work/agent_rl/to_route2_wm31_gate_s0/policy.pt,to_gate=work/agent_rl/to_gate_wm31_gate_s0/policy.pt,forest=work/agent_rl/forest_wm31_gate_s0/policy.pt,to_pewter=work/agent_rl/to_pewter_wm31_gate_s0/policy.pt,to_gym=work/agent_rl/to_gym_wm31_gate_s0/policy.pt,to_grass=work/agent_rl/to_grass_wm31_gate_s0/policy.pt,to_viridian_pc=work/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt,leave_viridian_pc=work/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt,to_pewter_pc=work/agent_rl/to_pewter_pc_wm31_gate_it1800_s0/policy.pt,leave_pewter_pc=work/agent_rl/leave_pewter_pc_wm31_gate_mc_s0/policy.pt" \
    --work-dir work/gym_learned_best_s0
  ```
- **Output Files:** `work/gym_learned_best_s0/summary.json`
- **Result Table:** [`lab/results/v3_gym.md`](results/v3_gym.md) & [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh).

### 3.4 Standalone Travel Leg Real-Game Evaluation (Viridian Forest)
- **Command (from [`work/chain_moves.sh`](../work/chain_moves.sh#L28)):**
  ```bash
  python -m lab.agent.mpc --task forest --policy learned --rl-name forest_wm31_gate_s0 --episodes 100 --max-steps 1000 --seed 200 --device cpu
  ```
- **Output Files:** `work/agent/forest_learned_forest_wm31_gate_s0_s200.json` (and `work/agent/eval_forest_wm31_gate_s0.log`)
- **Result Table:** [`lab/results/v3_gym.md`](results/v3_gym.md) (100/100 successes, median 560 presses).

### 3.5 Headline Imagination RL: Leave House
- **Command:**
  ```bash
  python -m lab.agent.mpc --task leave_house --policy learned --rl-name leave_house --episodes 100 --max-steps 60 --seed 200 --device cpu
  ```
- **Output Files:** `work/agent/leave_house_learned_leave_house_s200.json`
- **Result Table:** [`lab/results/v3_agent.md`](results/v3_agent.md) (100/100 successes, median 28 presses, zero-shot from 4 screen frames).

### 3.6 Harder Task Imagination RL: Route 1
- **Command:**
  ```bash
  python -m lab.agent.mpc --task route1 --policy learned --rl-name route1 --episodes 100 --max-steps 100 --seed 200 --device cpu
  ```
- **Output Files:** `work/agent/route1_learned_route1_s200.json`
- **Result Table:** [`lab/results/v3_agent.md`](results/v3_agent.md) (100/100 successes, median 22 presses).

### 3.7 Pallet Town → Route 1 → Viridian City (Geodesic + Clip)
- **Command:**
  ```bash
  python -m lab.agent.mpc --task viridian --policy learned --rl-name viridian_geo_clip --episodes 100 --max-steps 1000 --seed 200 --device cpu
  ```
- **Output Files:** `work/agent/viridian_learned_viridian_geo_clip_s200.json`
- **Result Table:** [`lab/results/v3_agent.md`](results/v3_agent.md,L120) (100/100 successes, median 284 presses).

### 3.8 Model-Free PPO in the Real Game (Viridian Geodesic)
- **Command (from [`work/chain_ppo.sh`](../work/chain_ppo.sh#L8)):**
  ```bash
  python -m lab.agent.ppo_real --task viridian --name viridian_geo --geodesic --cap 1000 --envs 12 --minutes 90 --max-env-steps 3000000
  ```
- **Output Files:** `work/ppo_real/viridian_geo/eval.jsonl`, `work/ppo_real/viridian_geo/policy.pt`
- **Result Table:** [`lab/results/v3_agent.md`](results/v3_agent.md) (95/100 successes, median 262 presses in 103k steps).

### 3.9 Diffusion World Model Rollout Evaluation & Error Decomposition
- **Commands:**
  ```bash
  # Generate 32-step rollouts on held-out test windows:
  python -m lab.dwm.eval_rollouts --run wm31 --data ext --split test_ext --n-windows 512 --k 32 --steps 3 --device cuda

  # Decompose position error into map hallucinations vs in-map drift:
  python -m lab.eval.decompose work/eval/wm31_test_ext_s3_a0_ext.npz
  ```
- **Output Files:** `work/eval/wm31_test_ext_s3_a0_ext.json`, `work/eval/wm31_test_ext_s3_a0_ext.npz`
- **Result Tables:** [`lab/results/v3_dwm_eval.md`](results/v3_dwm_eval.md); [`lab/results/v3_battle_fidelity.md`](results/v3_battle_fidelity.md).

### 3.10 Battle Onset Evaluation (wm31 on Tall Grass)
- **Command (from [`work/chain_battle_eval.sh`](../work/chain_battle_eval.sh#L10)):**
  ```bash
  python -m lab.dwm.battle_onset --run wm31 --data ext --map 13
  ```
- **Output Files:** `work/eval/battle_onset_wm31_map13_ext.json`
- **Result Table:** [`lab/results/v3_battle_fidelity.md`](results/v3_battle_fidelity.md).

### 3.11 Battle Fidelity Evaluation (Inside Battle Screens)
- **Command (from [`work/chain_battle_eval.sh`](../work/chain_battle_eval.sh#L11)):**
  ```bash
  python -m lab.dwm.battle_fidelity --run wm31 --data ext --max-windows 1024
  ```
- **Output Files:** `work/eval/battle_fidelity_wm31_ext.json`, `work/plots/battle_fidelity_wm31_ext.png`
- **Result Table:** [`lab/results/v3_battle_fidelity.md`](results/v3_battle_fidelity.md).

### 3.12 Catching Pokémon Strategy Comparison (Throw vs Weaken First)
- **Commands:**
  ```bash
  # Strategy 1: Throw Poké Ball at full HP
  python -m lab.agent.catch_quest --runs 10 --seed 100 --strategy throw --work-dir work/catch_throw

  # Strategy 2: Attack wild Pokémon until HP <= 60% before throwing
  python -m lab.agent.catch_quest --runs 10 --seed 100 --strategy weaken --work-dir work/catch_weaken
  ```
- **Output Files:** `work/catch_throw/summary.json`, `work/catch_weaken/summary.json`
- **Result Table:** [`lab/results/v3_catch.md`](results/v3_catch.md) (Weaken: 10/10 runs reached 10 catches, 85.5% catch rate per ball vs 31.1% for throw).

### 3.13 System 2 LLM Planner Bake-off
- **Command:**
  ```bash
  python -m lab.agent.gym_quest --runs 10 --seed 100 --target-level 14 --nav bfs --planner llm --llm-model gemma4:31b --work-dir work/sys2_bakeoff/gemma4_31b
  ```
- **Output Files:** `work/sys2_bakeoff/gemma4_31b/run_seed_100.json` .. `109.json`
- **Result Table:** [`lab/results/v3_system2.md`](results/v3_system2.md) (gemma4:31b: 10/10 badges, 0 rejected decisions, 7,239 median presses).

### 3.14 Start-Tile Robustness Sweep across 26 Reachable Tiles
- **Command:**
  ```bash
  python -m lab.agent.tile_sweep --task to_route2 --policies to_route2_wm31_gate_s0 --n-tiles 24 --episodes 30 --workers 8 --seed 0
  ```
- **Output Files:** `work/agent/sweep_to_route2/summary_1policies.csv` (and per-run JSONs)
- **Result Table:** [`lab/results/v3_tile_sweep.md`](results/v3_tile_sweep.md) (26/26 tiles ≥95% success; 780/780 = 100% successes).

---

## 4. Running the Tests

The lab includes 166 automated unit and integration tests under [`lab/tests/`](../lab/tests).

### 4.1 Run the Full Test Suite
```bash
python -m pytest lab/tests
```
All 166 tests pass synchronously in ~35 seconds on CPU.

### 4.2 Run Specific Test Suites
```bash
# Agent MPC, planning, and evaluation tests:
python -m pytest lab/tests/test_agent.py

# Gym quest end-to-end integration tests:
python -m pytest lab/tests/test_gym_quest.py

# PyBoy emulator environment and state generation:
python -m pytest lab/tests/test_env.py

# Battle fidelity and onset measurement tests:
python -m pytest lab/tests/test_battle_fidelity.py

# Tile sweep and start-state robustness tests:
python -m pytest lab/tests/test_tile_sweep.py

# Upsampler 144x160 reconstruction tests:
python -m pytest lab/tests/test_upsampler.py
```

> [!NOTE]
> Tests that require the ROM (`Pokemon - Red Version (USA, Europe).gb`) or route save states automatically inspect disk availability and gracefully skip with informative `@pytest.mark.skipif` notices if prerequisites are not present.

---

## 5. Honesty & Scope Notes

The findings in `lab/results/` are grounded in direct measurement against Game Boy RAM ground truth. Keep these explicit scope limits in mind when interpreting results:

1. **What is Learned vs What is Rules:**
   - **Learned in Imagination:** System 1 overworld navigation for travel legs (Viridian City → Route 2 → Gate → Forest → Pewter → Gym) and movement legs (grind pacing, PokéCenter healing) is driven by small CNN policies trained **purely inside the diffusion world model's imagination from pixels alone** (no RAM or emulator at test time).
   - **Rules / Scripted:** Battles (selecting FIGHT, choosing Ember/Scratch, buying Poké Balls) and high-level goal sequencing (grind to L14, heal if HP < 35%, travel to Pewter) are rule-based algorithms.
   - **Navigation Fallback:** If a learned navigation leg exceeds its 1,000-press budget without reaching the target, privileged BFS pathfinding takes over to complete the leg. This is tracked and reported per run: with the best leg set, 14 of 30 quests use no fallback anywhere (`work/gym_learned_best_s*`).

2. **The World Model is NOT a Sample-Efficiency Win Here:**
   - Model-free PPO trained directly in the real game reaches ≥95% success within 10,752-41,453 real steps on all three navigation tasks (leave_house, route1, viridian; Manhattan reward, 3 seeds; see [`lab/results/v3_agent.md`](results/v3_agent.md)). With the geodesic reward on the longest task it needed about 76k-103k steps, 4-5 minutes on CPU with 12 emulators.
   - The world model it competes with: wm31 trained for 150 minutes on one RTX 4090 (114,800 steps on 676,284 windows of the 800,000-frame extended dataset; `work/runs/wm31/train.log`), and each navigation leg then takes 15-45 minutes of imagination RL (600-1,800 iterations).
   - The scientific value of this project is not beating a fast emulator on sample count: it is **measuring what a learned pixel simulator gets right and wrong in grounded physical game units (tiles, maps, battle turns)**, developing safeguards against hallucinations, and proving zero-shot sim-to-real transfer from learned simulators for domains where real execution is slow, expensive, or unsafe.

3. **Simulating World Hallucinations Requires Safeguards:**
   - Unconstrained imagination rollouts suffer from map hallucinations (teleporting through walls) because 4 frames of visual context cannot preserve long-horizon map memory (41% wrong map at k=32 steps).
   - Naive planners or unconstrained RL exploit these hallucinations (e.g. walking into a bedroom wall for 60 steps).
   - Success requires anti-exploit safeguards: potential-based move gates (`--move-gate 2`), geodesic distance fields over observed transitions, and crediting map transitions only when adjacent to genuine exit tiles.

4. **Battles Cannot Currently Be Trained in the Diffusion World Model:**
   - As measured in [`lab/results/v3_battle_fidelity.md`](results/v3_battle_fidelity.md), wm31 suffers from **hallucinated early battle termination** in 72% of windows (battles end >3 steps early). An agent trained in this simulator would learn that battles resolve themselves regardless of pressed buttons.
   - Battle menus and text boxes degrade in pixel MSE beyond copy-last after 4 steps. Consequently, battle policies remain rule-based.
