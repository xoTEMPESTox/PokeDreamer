# PokeDreamer v3 Model & Weight Manifest

This manifest documents all model checkpoints, policies, and environment states required to reproduce the v3 experimental results published across `agent_notes/results/`. All file sizes are measured directly from the local filesystem.

> **Planned before upload:** the local checkpoints include optimizer state that inference never reads. `load_world_model` reads only `ckpt["ema"]` and the leg loaders only `ckpt["policy"]`, so the published files keep those keys (plus the step/iteration counters) and drop `opt` and the non-EMA `model`: wm31 about 67 MB instead of 268 MB, each policy about 6.3 MB instead of 18.9 MB (16.7M and 1.58M float32 parameters). The sizes in the tables below are the local files.

## Hugging Face Repositories

- **Model Repository:** [`xxxTEMPESTxxx/PokeDreamer`](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer) (cited in [README.md](../README.md#L3), [agent_notes/AUDIT.md](../agent_notes/AUDIT.md#L9), and [work/backup_hf.py](../work/backup_hf.py#L11))
- **Dataset Repository:** [`xxxTEMPESTxxx/PokeDreamer`](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer) (`repo_type="dataset"`, cited in [README.md](../README.md#L4), [agent_notes/AUDIT.md](../agent_notes/AUDIT.md#L6), and [work/backup_hf.py](../work/backup_hf.py#L11))
- **HF Directory Prefix:** All v3 assets are proposed under the `v3/` prefix (e.g. `v3/runs/wm31/ckpt.pt`) to keep existing archived v1 and v2 checkpoints intact without filename collisions (cited in [work/backup_hf.py](../work/backup_hf.py#L1-L26)).

---

## 1. Core Models

| Local Path | Size on Disk (Measured) | What It Is | Result Table / Module | Proposed HF Path |
|---|---|---|---|---|
| `work/runs/wm31/ckpt.pt` | 267,780,606 bytes<br>(255.38 MiB / 267.78 MB) | DIAMOND-style EDM denoiser U-Net (16.7M params) trained on extended dataset (114,800 steps); contains model state, EMA weights (`state["ema"]`), and optimizer | [`agent_notes/results/v3_battle_fidelity.md`](../agent_notes/results/v3_battle_fidelity.md#L5-L25); [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L40-L100); loaded by [`lab/dwm/controllability.py:load_world_model`](../lab/dwm/controllability.py#L213-L222) | `v3/runs/wm31/ckpt.pt` |
| `work/probe_ext/probe.pt` | 12,584,958 bytes<br>(12.00 MiB / 12.58 MB) | CNN FrameProbe trained on extended dataset frames; predicts map ID (256-way), player x/y coordinates, and battle flag from 72×80 greyscale | [`agent_notes/results/v3_battle_fidelity.md`](../agent_notes/results/v3_battle_fidelity.md#L23-L32); [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L27); loaded by [`lab/probe.py:load_probe`](../lab/probe.py#L47-L55) | `v3/probe_ext/probe.pt` |
| `work/upsampler/upsampler.pt` | 2,913,894 bytes<br>(2.78 MiB / 2.91 MB) | 0.7M-parameter CNN upsampler restoring 72×80 greyscale frames back to native 144×160 4-shade Game Boy frames (99.98% pixel accuracy) | [`agent_notes/results/upsampler.md`](../agent_notes/results/upsampler.md); [v3.md](../v3.md#L36); loaded by [`lab/upsampler.py:load_upsampler`](../lab/upsampler.py#L127-L142) | `v3/upsampler/upsampler.pt` |

**Subtotal:** 3 files, 283,279,458 bytes (270.16 MiB / 283.28 MB).

---

## 2. 10 Best Leg Policies (Gym Quest)

These policies drive the System 1 learned navigation in the Brock gym quest (`lab.agent.gym_quest`). Each checkpoint is saved in `policy.pt` format with dictionary keys `{"policy", "opt", "it", "elapsed"}` (as defined in [`lab/agent/imagine_rl.py`](../lab/agent/imagine_rl.py#L243)). All were trained entirely inside wm31's imagination using the potential-based move gate reward (`--move-gate 2`, `--data ext`, `--geodesic`).

| Local Path | Size on Disk (Measured) | What It Is | Result Table / Script | Proposed HF Path |
|---|---|---|---|---|
| `work/agent_rl/to_route2_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Travel Leg 1: Viridian City → Route 2 (seed 0; 100/100 across 26 start tiles) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L51); [`agent_notes/results/v3_tile_sweep.md`](../agent_notes/results/v3_tile_sweep.md#L27) | `v3/agent_rl/to_route2_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_gate_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Travel Leg 2: Route 2 → Viridian Forest Gate (seed 0; 100/100, median 94 presses) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L85); [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L19) | `v3/agent_rl/to_gate_wm31_gate_s0/policy.pt` |
| `work/agent_rl/forest_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Travel Leg 3: Viridian Forest south gate → north gate (seed 0; 100/100, median 560 presses) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L54,L71); [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L19) | `v3/agent_rl/forest_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_pewter_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Travel Leg 4: North Gate → Pewter City (seed 0; 100/100, median 28 presses) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L86); [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L19) | `v3/agent_rl/to_pewter_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_gym_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Travel Leg 5: Pewter City → Gym door (seed 0; 100/100, median 48 presses) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L87); [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L19) | `v3/agent_rl/to_gym_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_grass_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Movement Leg 1: Viridian City → Route 1 grinding grass (seed 0) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md); [`work/chain_moves.sh`](../work/chain_moves.sh#L18) | `v3/agent_rl/to_grass_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Movement Leg 2: Route 1 grass → Viridian PokéCenter (seed 0) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md); [`work/chain_moves.sh`](../work/chain_moves.sh#L18) | `v3/agent_rl/to_viridian_pc_wm31_gate_s0/policy.pt` |
| `work/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Movement Leg 3: Viridian PokéCenter interior → street exit (seed 0) | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md); [`work/chain_moves.sh`](../work/chain_moves.sh#L18) | `v3/agent_rl/leave_viridian_pc_wm31_gate_s0/policy.pt` |
| `work/agent_rl/to_pewter_pc_wm31_gate_it1800_s0/policy.pt` | 18,931,986 bytes<br>(18.05 MiB / 18.93 MB) | Movement Leg 4: Pewter City → Pewter PokéCenter (1800 it, seed 0; 100/100, median 15 presses) | [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh#L2-L26); [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md) | `v3/agent_rl/to_pewter_pc_wm31_gate_it1800_s0/policy.pt` |
| `work/agent_rl/leave_pewter_pc_wm31_gate_mc_s0/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Movement Leg 5: Pewter PokéCenter interior → street exit (merged centers, seed 0; 100/100) | [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh#L5,L26); [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md) | `v3/agent_rl/leave_pewter_pc_wm31_gate_mc_s0/policy.pt` |

**Subtotal:** 10 files, 189,318,132 bytes (180.55 MiB / 189.32 MB).

---

## 3. Headline v3 Agent Navigation Policies

These are the three primary System 1 policies trained in imagination for the headline navigation experiments in [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md).

| Local Path | Size on Disk (Measured) | What It Is | Result Table | Proposed HF Path |
|---|---|---|---|---|
| `work/agent_rl/leave_house/policy.pt` | 18,931,986 bytes<br>(18.05 MiB / 18.93 MB) | Final policy (it 640): bedroom stairs → 1st floor → Pallet Town (100/100 success, 28 median presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L43); [v3.md](../v3.md#L16) | `v3/agent_rl/leave_house/policy.pt` |
| `work/agent_rl/route1/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Policy (it 1440): Pallet Town → Route 1 (100/100 success, 22 median presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L63) | `v3/agent_rl/route1/policy.pt` |
| `work/agent_rl/viridian_geo_clip/policy.pt` | 18,931,794 bytes<br>(18.05 MiB / 18.93 MB) | Policy: Pallet Town → Route 1 → Viridian City with geodesic + clip (100/100 success, 284 median presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L104,L120) | `v3/agent_rl/viridian_geo_clip/policy.pt` |

**Subtotal:** 3 files, 56,795,574 bytes (54.16 MiB / 56.80 MB).

---

## 4. PPO Real-Game Baselines

Model-free PPO trained directly in the real game on the fast PyBoy emulator (12 parallel emulators on CPU). Named and evaluated across seeds in [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L124-L149) and [`work/chain_ppo2.sh`](../work/chain_ppo2.sh).

| Local Path | Size on Disk (Measured) | What It Is | Result Table | Proposed HF Path |
|---|---|---|---|---|
| `work/ppo_real/leave_house/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `leave_house` (seed 42; 100/100, median 24 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L146) | `v3/agent_rl/ppo_real_leave_house/policy.pt` |
| `work/ppo_real/leave_house_s1/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `leave_house` (seed 1; 100/100, median 24 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L146) | `v3/agent_rl/ppo_real_leave_house_s1/policy.pt` |
| `work/ppo_real/leave_house_s2/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `leave_house` (seed 2; 100/100, median 24 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L146) | `v3/agent_rl/ppo_real_leave_house_s2/policy.pt` |
| `work/ppo_real/route1/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `route1` (seed 42; 98/100, median 30 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L147) | `v3/agent_rl/ppo_real_route1/policy.pt` |
| `work/ppo_real/route1_s1/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `route1` (seed 1; 100/100, median 17 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L147) | `v3/agent_rl/ppo_real_route1_s1/policy.pt` |
| `work/ppo_real/route1_s2/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `route1` (seed 2; 100/100, median 17.5 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L147) | `v3/agent_rl/ppo_real_route1_s2/policy.pt` |
| `work/ppo_real/viridian/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `viridian` (Manhattan reward, seed 42; 100/100, median 249 presses in 54k steps) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L138,L148) | `v3/agent_rl/ppo_real_viridian/policy.pt` |
| `work/ppo_real/viridian_s1/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `viridian` (Manhattan reward, seed 1; 100/100, median 261 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L148) | `v3/agent_rl/ppo_real_viridian_s1/policy.pt` |
| `work/ppo_real/viridian_s2/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `viridian` (Manhattan reward, seed 2; 99/100, median 241 presses) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L148) | `v3/agent_rl/ppo_real_viridian_s2/policy.pt` |
| `work/ppo_real/viridian_geo/policy.pt` | 18,931,346 bytes<br>(18.05 MiB / 18.93 MB) | PPO real-game baseline: `viridian` (geodesic reward, seed 42; 95/100, median 262 presses in 103k steps) | [`agent_notes/results/v3_agent.md`](../agent_notes/results/v3_agent.md#L137) | `v3/agent_rl/ppo_real_viridian_geo/policy.pt` |

**Subtotal:** 10 files, 189,313,460 bytes (180.54 MiB / 189.31 MB).

---

## 5. Route Save States (`work/route_states/`)

PyBoy emulator save states captured across quest seeds (100, 101, 102) and locations along the Viridian → Pewter path, plus the movement leg targets. Each `.state` file is exactly 167,677 bytes.

| Local Path | Size on Disk (Measured) | What It Is | Result Table / Module | Proposed HF Path |
|---|---|---|---|---|
| `work/route_states/index.json` | 4,899 bytes (4.78 KiB) | JSON metadata indexing each save state: seed, map ID, player (x, y), party level, HP, max HP | [`lab/tests/test_collect.py`](../lab/tests/test_collect.py#L196); [`work/route_states/index.json`](../work/route_states/index.json#L1) | `v3/route_states/index.json` |
| `work/route_states/leave_pewter_pc.state` | 167,677 bytes (163.75 KiB) | Pewter PokéCenter interior (map 58) start state | [`lab/tests/test_move_tasks.py`](../lab/tests/test_move_tasks.py); [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh) | `v3/route_states/leave_pewter_pc.state` |
| `work/route_states/leave_viridian_pc.state` | 167,677 bytes (163.75 KiB) | Viridian PokéCenter interior (map 41) start state | [`lab/tests/test_move_tasks.py`](../lab/tests/test_move_tasks.py); [`work/chain_moves.sh`](../work/chain_moves.sh) | `v3/route_states/leave_viridian_pc.state` |
| `work/route_states/quest_to_gym_map2_13_26.state` | 167,677 bytes (163.75 KiB) | Pewter City gym arrival tile state (map 2, (13, 26)) | [`lab/tests/test_gym_quest.py`](../lab/tests/test_gym_quest.py); [`work/chain_gate_legs.sh`](../work/chain_gate_legs.sh#L14) | `v3/route_states/quest_to_gym_map2_13_26.state` |
| `work/route_states/quest_to_route2_map1_23_26.state` | 167,677 bytes (163.75 KiB) | Viridian City quest arrival tile state (map 1, (23, 26)) | [`agent_notes/results/v3_tile_sweep.md`](../agent_notes/results/v3_tile_sweep.md#L9); [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L3) | `v3/route_states/quest_to_route2_map1_23_26.state` |
| `work/route_states/seed100_map1.state` | 167,677 bytes (163.75 KiB) | Viridian City (map 1), Charmander L8, seed 100 | [`lab/tests/test_route_tasks.py`](../lab/tests/test_route_tasks.py#L85) | `v3/route_states/seed100_map1.state` |
| `work/route_states/seed100_map41.state` | 167,677 bytes (163.75 KiB) | Viridian PokéCenter (map 41), seed 100 | [`work/route_states/index.json`](../work/route_states/index.json#L13) | `v3/route_states/seed100_map41.state` |
| `work/route_states/seed100_map12.state` | 167,677 bytes (163.75 KiB) | Route 1 (map 12), seed 100 | [`work/route_states/index.json`](../work/route_states/index.json#L23) | `v3/route_states/seed100_map12.state` |
| `work/route_states/seed100_map13.state` | 167,677 bytes (163.75 KiB) | Route 2 (map 13), Charmander L14, seed 100 | [`lab/tests/test_learned_nav.py`](../lab/tests/test_learned_nav.py#L177) | `v3/route_states/seed100_map13.state` |
| `work/route_states/seed100_map50.state` | 167,677 bytes (163.75 KiB) | Route 2 Gate (map 50), seed 100 | [`lab/tests/test_learned_nav.py`](../lab/tests/test_learned_nav.py#L176) | `v3/route_states/seed100_map50.state` |
| `work/route_states/seed100_map51.state` | 167,677 bytes (163.75 KiB) | Viridian Forest (map 51), seed 100 | [`lab/tests/test_route_tasks.py`](../lab/tests/test_route_tasks.py#L87) | `v3/route_states/seed100_map51.state` |
| `work/route_states/seed100_map47.state` | 167,677 bytes (163.75 KiB) | North Gate (map 47), seed 100 | [`lab/tests/test_route_tasks.py`](../lab/tests/test_route_tasks.py#L88) | `v3/route_states/seed100_map47.state` |
| `work/route_states/seed100_map2.state` | 167,677 bytes (163.75 KiB) | Pewter City (map 2), seed 100 | [`lab/tests/test_route_tasks.py`](../lab/tests/test_route_tasks.py#L89) | `v3/route_states/seed100_map2.state` |
| `work/route_states/seed100_map58.state` | 167,677 bytes (163.75 KiB) | Pewter PokéCenter (map 58), seed 100 | [`work/route_states/index.json`](../work/route_states/index.json#L83) | `v3/route_states/seed100_map58.state` |
| `work/route_states/seed100_map54.state` | 167,677 bytes (163.75 KiB) | Pewter Gym (map 54), seed 100 | [`work/route_states/index.json`](../work/route_states/index.json#L93) | `v3/route_states/seed100_map54.state` |
| `work/route_states/seed101_map1.state` .. `map58.state` (10 files) | 1,676,770 bytes (1.60 MiB) | Seed 101 route checkpoints across maps 1, 12, 13, 2, 41, 47, 50, 51, 54, 58 | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L15); [`work/route_states/index.json`](../work/route_states/index.json#L101-L200) | `v3/route_states/seed101_*.state` |
| `work/route_states/seed102_map1.state` .. `map58.state` (10 files) | 1,676,770 bytes (1.60 MiB) | Seed 102 route checkpoints across maps 1, 12, 13, 2, 41, 47, 50, 51, 54, 58 | [`agent_notes/results/v3_gym.md`](../agent_notes/results/v3_gym.md#L15); [`work/route_states/index.json`](../work/route_states/index.json#L201-L300) | `v3/route_states/seed102_*.state` |
| `work/route_states/to_grass.state` | 167,677 bytes (163.75 KiB) | Movement leg start state: to Route 1 grass patch | [`lab/tests/test_move_tasks.py`](../lab/tests/test_move_tasks.py#L136); [`work/chain_moves.sh`](../work/chain_moves.sh) | `v3/route_states/to_grass.state` |
| `work/route_states/to_pewter_pc.state` | 167,677 bytes (163.75 KiB) | Movement leg start state: to Pewter PokéCenter | [`lab/tests/test_move_tasks.py`](../lab/tests/test_move_tasks.py#L158); [`work/chain_pewter_1800.sh`](../work/chain_pewter_1800.sh) | `v3/route_states/to_pewter_pc.state` |
| `work/route_states/to_viridian_pc.state` | 167,677 bytes (163.75 KiB) | Movement leg start state: to Viridian PokéCenter | [`lab/tests/test_move_tasks.py`](../lab/tests/test_move_tasks.py); [`work/chain_moves.sh`](../work/chain_moves.sh) | `v3/route_states/to_viridian_pc.state` |

**Subtotal:** 38 files (37 `.state` files + 1 `index.json`), 6,208,948 bytes (5.92 MiB / 6.21 MB).

> [!IMPORTANT]
> **Copyright Considerations Regarding Route Save States (Open Decision for Owner):**
> PyBoy `.state` files serialize the CPU registers, working RAM (WRAM), video RAM (VRAM), Object Attribute Memory (OAM), and cartridge battery RAM (SRAM) generated during execution of the Pokémon Red ROM. While they do not include the ROM executable itself, they hold in-memory snapshots of tilemaps and game state derived from the user's proprietary ROM session. 
> 
> *Owner Decision:* The owner must decide whether to publish these `.state` files directly to Hugging Face or instead publish deterministic generator scripts (such as [`lab/make_pokedex_state.py`](../lab/make_pokedex_state.py) and [`lab/env.py:make_states`](../lab/env.py#L109-L135)) that generate all required save states locally from the user's own legally dumped ROM.

---

## 6. Manifest Totals Summary

| Asset Group | File Count | Exact Total Bytes | Total Size (Binary MiB) | Total Size (Decimal MB) |
|---|---|---|---|---|
| **Core Models** (`wm31`, `probe_ext`, `upsampler`) | 3 | 283,279,458 | 270.16 MiB | 283.28 MB |
| **10 Best Leg Policies** (5 Travel + 5 Movement) | 10 | 189,318,132 | 180.55 MiB | 189.32 MB |
| **3 Headline v3 Agent Policies** (`leave_house`, `route1`, `viridian`) | 3 | 56,795,574 | 54.16 MiB | 56.80 MB |
| **10 PPO Real-Game Baselines** (leave_house, route1, viridian) | 10 | 189,313,460 | 180.54 MiB | 189.31 MB |
| **Route Save States** (37 `.state` + `index.json`) | 38 | 6,208,948 | 5.92 MiB | 6.21 MB |
| **TOTAL (Core Manifest: 61 files excluding headline policies)** | **61** | **668,120,000** | **637.17 MiB** | **668.12 MB** |
| **GRAND TOTAL (All 64 files including headline policies)** | **64** | **724,915,572** | **691.33 MiB** | **724.92 MB** |

---

## 7. What is Deliberately NOT Published

To keep the repository clean, compact, and reproducible, the following artifacts are deliberately excluded from Hugging Face:

1. **Intermediate Training Checkpoints:**
   - Periodic rollouts and snapshots in `work/runs/wm31/rollout_*.png` (25 image checkpoints, ~9 MB total).
   - Intermediate iteration checkpoints for imagination RL (e.g. `work/agent_rl/leave_house_it200/`, `_it400/`, `_it600/`, `to_pewter_pc_wm31_gate_wd0_s*`, `to_pewter_pc_wm31_gate_s0` 900-it checkpoint). Only the declared best final checkpoints are published.
2. **Raw Run Directories, Event Logs, and Trajectories:**
   - Thousands of per-episode raw JSON trace logs: `work/agent/*.json`, `work/gym_rng/run_seed_*.json`, `work/catch_sweep/`, `work/system12_sweep/`, and `work/sys2_bakeoff/`.
   - Large raw evaluation numpy archives: `work/eval/*.npz`.
   - Temporary stdout logs (`work/*.log`) and shell worker watchdogs (`work/PAUSE`).
3. **Rebuildable Dataset Caches:**
   - Downsampled 72×80 greyscale cache: `work/cache_72x80.npz` (1.13 GB) and `work/cache_ext_72x80.npz` (3.9 GB).
   - Upsampler native-resolution class cache: `work/native_classes.npy` (5.8 GB).
   - Rebuilding these caches takes ~3-10 minutes from the Hugging Face dataset [`xxxTEMPESTxxx/PokeDreamer`](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer) via `python -m lab.data` and `python -m lab.data --ext`.
