# PokéWorld — Model-Based RL in Pokémon Red

[![Hugging Face Models](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Repo-blue)](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20HuggingFace-Dataset%20Repo-green)](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer)
[![GitHub](https://img.shields.io/badge/GitHub-xoTEMPESTox%2FPokeDreamer-black?logo=github)](https://github.com/xoTEMPESTox/PokeDreamer)

An experimental research project applying **Dreamer-style model-based reinforcement learning** to play Pokémon Red on a Game Boy emulator (PyBoy). Built from scratch, iterating from a simple VAE + GRU dynamics model (v1) to a discrete Recurrent State-Space Model (RSSM) trained on native-resolution pixels (v2), with a roadmap toward a full dual-agent SOTA system (v3).

---

## 📜 Table of Contents

- [PokéWorld v1: VAE + GRU MPC Planner](#-pokeworld-v1-continuous-latents--mpc-planner)
- [PokéWorld v2: Discrete RSSM World Model](#-pokeworld-v2-discrete-rssm-world-model)
- [Key Findings & Metrics](#-key-findings--metrics)
- [Demo Videos](#-demo-videos)
- [Hugging Face Repositories](#-hugging-face-repositories)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [v3 Roadmap (Planned)](#-v3-roadmap-planned)

---

## 🎮 PokéWorld v1: Continuous Latents & MPC Planner

**Status: Completed & Archived** — [`v1/`](v1/)

### Architecture
| Component | Description |
|---|---|
| **Observation** | 40×36×3 pixels (PWhiddy downsampled) |
| **State Encoder** | Variational Autoencoder (VAE), z ∈ R^32 |
| **Dynamics** | Autoregressive GRU predicting z_{t+1} from (z_t, a_t) with scheduled sampling |
| **Controller** | Lookahead MPC planner using coordinate probes on latent space |
| **Probe** | Linear + MLP probes for player (x,y) and map_id from frozen latents |

### v1 Key Results
- **VAE Reconstruction**: Train/Val Loss ~1258/~1256 (MSE on pixel space)
- **Map ID Classification**: **98.7% accuracy** from 32-dim latent alone
- **Coordinate Probe Error**: **1.23 tiles** Manhattan distance (mean absolute error)
- **Dynamics Model** (Scheduled Sampling): Val AR Loss = **0.10314** vs. Pure TF Ablation = **0.72547**
- **Imagination Stability**: At 29 steps of free rollout, SS model drifts only **3.47 tiles** vs 10.44 tiles for pure teacher forcing

### Rollout Drift — Scheduled Sampling vs Teacher Forcing

| Rollout Step | SS Latent MSE | TF Latent MSE | SS Tile Error | TF Tile Error |
|---|---|---|---|---|
| Step 1 | 0.09268 | 0.12240 | 3.72 tiles | 4.06 tiles |
| Step 5 | 0.08751 | 0.23061 | 3.32 tiles | 5.08 tiles |
| Step 10 | 0.08842 | 0.40189 | **3.30 tiles** | **6.47 tiles** |
| Step 15 | 0.09238 | 0.59876 | **3.33 tiles** | **7.81 tiles** |
| Step 20 | 0.09830 | 0.78455 | **3.36 tiles** | **9.13 tiles** |
| Step 29 | **0.11197** | **1.04063** | **3.47 tiles** | **10.44 tiles** |

> **Finding**: Scheduled sampling is critical for imagination stability. The SS model's compounding drift stays flat under 3.5 tiles out to 29 steps, while the pure TF ablation exceeds 10.4 tiles — a 3× degradation.

---

## 🌟 PokéWorld v2: Discrete RSSM World Model

**Status: Completed & Archived** — [`v2/`](v2/) *(archiving in progress)*

### Architecture
| Component | Description |
|---|---|
| **Observation** | Native 160×144×3 pixels (4× higher resolution than v1) |
| **Encoder** | 4-layer Residual CNN → 512-dim embedding |
| **RSSM** | Deterministic GRU h_t (512-dim) + Stochastic Categorical s_t (32 classes × 32 categories = 1024-dim) |
| **Training** | Gumbel-Softmax straight-through + KL-balancing (80% prior / 20% posterior) |
| **Decoders** | Pixel decoder (ConvTranspose) + Reward predictor + Continue predictor |
| **Dataset** | 20 NPZ files × ~800 transitions = ~16,000 transition frames at native resolution |

### v2 Key Results

Training was run for **4 epochs** (50 min/epoch on RTX GPU) using batch size 64, sequence length 15:

| Epoch | Train Loss | Train Recon | Train KL | Val Loss | Val Recon | Val KL |
|---|---|---|---|---|---|---|
| 1 | 0.1476 | 0.1379 | 0.0078 | 0.1266 | 0.1256 | 0.0010 |
| 2 | 0.1207 | 0.1144 | 0.0063 | 0.1172 | 0.1110 | 0.0062 |
| 3 | 0.1490 | 0.1068 | 0.0422 | 0.1228 | 0.1142 | 0.0086 |
| 4 | 0.1021 | 0.1015 | 0.0005 | 0.1651 | 0.1003 | 0.0648 |

> **Finding**: Reconstruction loss steadily decreases across epochs, with the best model (epoch 4 by val recon = 0.1003) demonstrating pixel-level world modeling at native Game Boy resolution. KL divergence fluctuates as the discrete latent space balances expressiveness vs. compressibility — a known challenge with categorical RSSM training.

### v2 vs v1: Key Improvements
| Aspect | v1 | v2 |
|---|---|---|
| Resolution | 40×36 (downsampled) | 160×144 (native) |
| Latent space | Continuous R^32 | Discrete 32×32 categorical |
| Temporal modeling | BPTT GRU dynamics | Recurrent State-Space Model (RSSM) |
| Imagination | Prior-only rollout | Full posterior + prior with KL balancing |
| Gradient estimator | Reparameterization | Gumbel-Softmax straight-through |

---

## 🎬 Demo Videos

### v1: MPC Planner Navigation Demo

The v1 agent uses the GRU dynamics model to simulate future states, then selects actions by scoring imagined trajectories using a coordinate probe. This video shows the planner navigating Pallet Town.

> *Video: `v1/checkpoints/planner_navigation_demo.mp4`* (see v1/ folder)
> [![v1 Navigation Demo](https://img.shields.io/badge/▶%20v1%20Navigation-Demo%20Video-red)](v1/checkpoints/planner_navigation_demo.mp4)

### v2: Discrete RSSM Imagination vs. Real Emulator

The v2 agent (right panel) generates frames purely from the RSSM prior dynamics — no emulator steps. Starting from the same seed frame, it imaginines the visual consequences of each action autoregressively. The left panel shows the real emulator.

> *Video: `checkpoints/rssm_v2/side_by_side_demo_v2.mp4`*
> [![v2 RSSM Demo](https://img.shields.io/badge/▶%20v2%20RSSM-Imagination%20Demo-orange)](checkpoints/rssm_v2/side_by_side_demo_v2.mp4)

---

## 🤗 Hugging Face Repositories

All models, checkpoints, and datasets are backed up on Hugging Face:

| Repository | Type | Contents |
|---|---|---|
| [xxxTEMPESTxxx/PokeDreamer](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer) | 🤖 Model | RSSM v2 checkpoints (`best_world_model.pt`), v1 VAE/dynamics checkpoints |
| [xxxTEMPESTxxx/PokeDreamer](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer) | 📦 Dataset | Native-resolution transition NPZ files (20 files, ~340MB total) |

---

## 📂 Repository Structure

```
PokeDreamer/
├── README.md                   # This file
├── v2.md                       # v2 implementation plan
├── v3.md                       # v3 SOTA roadmap (planned)
├── environment.yml             # Conda environment spec
│
├── v1/                         # Archived v1: VAE + GRU MPC
│   ├── src/                    # v1 source code
│   ├── scripts/                # v1 training & demo scripts
│   ├── checkpoints/            # v1 model weights + demo videos
│   └── metrics_registry.md    # v1 experimental metrics
│
├── v2/                         # Archived v2: Discrete RSSM (populated after archiving)
│   ├── src/                    # v2 source code (RSSM, dataset, models)
│   ├── scripts/                # v2 training & demo scripts
│   └── checkpoints/            # v2 RSSM weights + demo video
│
├── src/                        # Active source (currently v2)
│   ├── models.py               # Encoder, Decoder, RSSMCell, predictors
│   ├── dataset.py              # PokémonDataset loader
│   ├── game_state.py           # PyBoy screen capture + RAM extraction
│   └── ram_addresses.py        # WRAM address constants for Pokémon Red
│
├── scripts/                    # Execution scripts
│   ├── collect_data.py         # Native-resolution data collection
│   ├── train_rssm.py           # RSSM world model trainer
│   ├── generate_demo_video_v2.py  # Side-by-side imagination demo
│   └── upload_to_hf.py         # Hugging Face backup utility
│
├── data/                       # Transition NPZ files (collected from emulator)
├── checkpoints/                # Model checkpoints by version
│   └── rssm_v2/                # v2 RSSM checkpoints + reconstruction grids
├── saves/                      # Game Boy save states
└── external/                   # External baseline code (PWhiddy PPO)
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/xoTEMPESTox/PokeDreamer.git
   cd PokeDreamer
   ```

2. **Create Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate pokemon-rl
   ```

3. **Place the ROM**:
   Copy your legally-obtained `Pokemon - Red Version (USA, Europe).gb` into the root directory.

4. **Collect Data** (optional, dataset available on HF):
   ```bash
   python scripts/collect_data.py --episodes 20 --out-dir data
   ```

5. **Train RSSM**:
   ```bash
   python scripts/train_rssm.py \
       --data-dir data \
       --epochs 12 \
       --batch-size 64 \
       --out-dir checkpoints/rssm_v2
   ```

6. **Generate Demo Video**:
   ```bash
   python scripts/generate_demo_video_v2.py \
       --checkpoint checkpoints/rssm_v2/best_world_model.pt \
       --save-state saves/intro_done.state \
       --out-video checkpoints/rssm_v2/side_by_side_demo_v2.mp4
   ```

---

## 🚀 v3 Roadmap (Planned)

See [`v3.md`](v3.md) for the full technical roadmap. High-level vision:

### System 1 (Fast Reactive Controller)
- **Behaviour Cloning** warm-start from 10–50h of curated human demonstrations
- **PPO + GRPO** online RL agent trained inside RSSM imagination rollouts
- Multi-task actor-critic conditioned on goal embeddings from System 2
- Reward from: Δbadges, Δ Pokédex, Δunique map IDs, battle outcomes, XP gain

### System 2 (Slow Strategic Planner)
- **Multimodal LLM** (Gemini 2.0 Flash / GPT-4o) receiving game screenshots + RAM state JSON
- Produces macro goals: "Reach Pewter City Gym", "Defeat Brock"
- Sub-goal queue injected as conditioning token into System 1 policy
- Replanning triggered on progress stalls (500+ steps without story flag advance)

### v3 Target Metrics
| Metric | v2 Achieved | v3 Target |
|---|---|---|
| Badges (no human intervention) | 0 (policy not trained) | ≥ 2 |
| Pokédex species caught | N/A | ≥ 10 |
| 50-step imagination drift | ~5–8 tiles (est.) | < 5 tiles |
| World model size | ~650MB | < 200MB |

---

## 📖 Citation & Credits

- **Dreamer / DreamerV3**: Hafner et al. (2021, 2023) — RSSM architecture inspiration
- **PWhiddy baseline PPO**: [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — used as data collection policy
- **PyBoy emulator**: [Baekalfen/PyBoy](https://github.com/Baekalfen/PyBoy)

---

*Project by [@xoTEMPESTox](https://github.com/xoTEMPESTox) — Research-grade, educational use only. No ROM files included.*
