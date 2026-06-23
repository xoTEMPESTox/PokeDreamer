# PokéWorld — Metrics Registry

This file records the training runs, hyperparameters, and final validation metrics for all models in the PokéWorld world model pipeline.

---

## 1. Variational Autoencoder (VAE)

Compresses grayscale observations matching PPO inputs $(36 \times 40 \times 3)$ to a low-dimensional latent space $z$.

- **Checkpoint File**: `checkpoints/vae/best_vae.pt`
- **Epochs Trained**: 15
- **Hyperparameters**:
  - `latent_dim`: 32
  - `beta` (KL weight): 1.0
  - `learning_rate`: 1e-3
  - `batch_size`: 128
- **Final Metrics**:
  - `Train Loss`: 1258.41 (Recon: 1227.76, KL: 30.66)
  - `Val Loss`: 1255.95 (Recon: 1225.83, KL: 30.13)

---

## 2. RAM State Probe

A linear/MLP probe trained on frozen latents $z$ to predict symbolic RAM states for evaluation.

- **Checkpoint File**: `checkpoints/probe/best_probe.pt`
- **Epochs Trained**: 10
- **Hyperparameters**:
  - `input_dim`: 32
  - `hidden_dim`: 64 (per task head)
  - `learning_rate`: 1e-3
  - `batch_size`: 128
- **Final Metrics**:
  - `Val Loss`: 8.3260
  - `Manhattan Coordinate Error (Pos MAE)`: **1.23 tiles**
  - `Map ID Classification Accuracy`: **98.7%**

---

## 3. Recurrent Latent Dynamics Model

Predicts future latents autoregressively: $z_{t+1} \approx \text{Dynamics}(z_t, a_t)$.

### 3.1 Scheduled Sampling Model (Primary)
- **Checkpoint File**: `checkpoints/dynamics/best_dynamics.pt`
- **Epochs Trained**: 20
- **Hyperparameters**:
  - `seq_len`: 30 (BPTT rollout steps)
  - `hidden_dim` (GRU): 256
  - `action_dim` (Embedding): 16
  - `decay_epochs`: 15 (linear decay of teacher forcing ratio)
  - `min_teacher_forcing`: 0.2
  - `learning_rate`: 1e-3
  - `batch_size`: 128
- **Final Metrics**:
  - `Val Loss (Autoregressive - AR)`: **0.10314**
  - `Val Loss (Teacher Forced - TF)`: **0.03675**

### 3.2 Pure Teacher Forcing Model (Ablation)
- **Checkpoint File**: `checkpoints/dynamics/best_dynamics_ablation.pt`
- **Epochs Trained**: 20
- **Hyperparameters**:
  - `seq_len`: 30
  - `hidden_dim` (GRU): 256
  - `action_dim` (Embedding): 16
  - `no_scheduled_sampling`: True (teacher forcing = 1.0 always)
  - `learning_rate`: 1e-3
  - `batch_size`: 128
- **Final Metrics**:
  - `Val Loss (Autoregressive - AR)`: **0.72547**
  - `Val Loss (Teacher Forced - TF)`: **0.01913**

---

## 4. Rollout Drift Evaluation (imagined vs. real)

Measured over 14,564 validation sequence trajectories of length 29. Compares Scheduled Sampling (SS) against the Pure Teacher Forcing (TF) ablation.

| Rollout Step ($k$) | SS Latent MSE | TF Latent MSE (Ablation) | SS Tile Error (Manhattan) | TF Tile Error (Ablation) |
|---|---|---|---|---|
| Step 1 | 0.09268 | 0.12240 | 3.72 tiles | 4.06 tiles |
| Step 5 | 0.08751 | 0.23061 | 3.32 tiles | 5.08 tiles |
| Step 10 | 0.08842 | 0.40189 | **3.30 tiles** | **6.47 tiles** |
| Step 15 | 0.09238 | 0.59876 | **3.33 tiles** | **7.81 tiles** |
| Step 20 | 0.09830 | 0.78455 | **3.36 tiles** | **9.13 tiles** |
| Step 25 | 0.10606 | 0.92376 | 3.42 tiles | 9.72 tiles |
| Step 29 | **0.11197** | **1.04063** | **3.47 tiles** | **10.44 tiles** |

> [!TIP]
> The scheduled sampling model demonstrates remarkable stability. Compounding drift for player coordinates remains flat under **3.5 tiles** out to 29 steps of lookahead imagination, whereas the pure teacher forcing ablation model degrades rapidly, exceeding **10.4 tiles** of drift by step 29.

