# PokéWorld — Model-Based Reinforcement Learning in Pokémon Red

An experimental framework replicating and adapting state-of-the-art model-based reinforcement learning (Dreamer-style) to play and navigate Pokémon Red on the Game Boy emulator.

---

## 🚀 Project Overview

The project is developed in two sequential phases:

### 🎮 PokéWorld v1: Continuous Latents & MPC Planner (Completed)
*   **Observation Resolution**: $40 \times 36 \times 3$ pixels.
*   **State Representation**: Continuous 32-dimensional latent vector $z$ trained via a Variational Autoencoder (VAE).
*   **Dynamics Model**: Autoregressive GRU dynamics predicting $z_{t+1}$ given $(z_t, a_t)$ with scheduled sampling.
*   **Controller**: Lookahead Model Predictive Control (MPC) planner searching trajectories to navigate the overworld based on coordinates decoded from latent state probes.
*   **Archived Source**: All v1 source code, scripts, checkpoints, and demonstration videos are archived in the [v1/](file:///c:/Users/priya/Code/Pokemon-rl/v1) directory.

### 🌟 PokéWorld v2: Discrete RSSM & Imagination Policy (Active)
*   **Observation Resolution**: Native $160 \times 144 \times 3$ Game Boy pixels.
*   **State Representation**: Discrete Categorical Latents ($32 \times 32$ classes) trained end-to-end.
*   **Dynamics Model**: Recurrent State-Space Model (RSSM) featuring deterministic GRU updates ($h_t$) and stochastic categorical representations ($s_t$) using Gumbel-Softmax straight-through estimators.
*   **Policy Source**: Actor-Critic networks trained entirely inside the imagined rollouts ($H=15$) of the RSSM world model, executing zero-shot inside PyBoy.

---

## 📂 Repository Structure

```
├── data/                  # Collected high-resolution transition frames (.npz)
├── saves/                 # Game Boy save states (.state) for cycling starts
├── src/                   # Active v2 Source Code
│   ├── dataset.py         # PyTorch loader for sequence transition dataset
│   ├── game_state.py      # PyBoy screen capture and RAM extraction wrapper
│   ├── models.py          # Residual CNN, RSSM, Actor, Critic, and Predictor networks
│   └── ram_addresses.py   # WRAM addresses for coordinate evaluation probes
├── scripts/               # Utility and execution scripts
│   ├── collect_data.py    # Native resolution data collection script
│   ├── train_rssm.py      # World model sequence trainer with KL-balancing
│   ├── train_policy.py    # Policy imagination trainer & PyBoy evaluator
│   └── upload_to_hf.py    # Hugging Face Hub backup tool
├── v1/                    # Archived v1 code, scripts, and checkpoints
└── README.md              # Project documentation
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/xoTEMPESTox/PokeDreamer.git
    cd Pokemon-rl
    ```

2.  **Environment Setup**:
    Create the Conda environment using the provided `environment.yml` configuration:
    ```bash
    conda env create -f environment.yml
    conda activate pokemon-rl
    ```

3.  **Place the ROM**:
    Copy your legally obtained `Pokemon - Red Version (USA, Europe).gb` ROM into the root directory.

---

## 📈 Backups & Hugging Face Integrations

We use Hugging Face to backup and version large datasets:
*   **Dataset Repository**: [xxxTEMPESTxxx/PokeDreamer (Datasets)](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer)
*   **Model Repository**: [xxxTEMPESTxxx/PokeDreamer (Models)](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer)

To backup data local files:
```bash
# Decompress and login
pip install -U huggingface_hub
huggingface-cli login

# Backup dataset
python scripts/upload_to_hf.py \
    --repo-id "xxxTEMPESTxxx/PokeDreamer" \
    --folder "data" \
    --repo-type "dataset"
```
