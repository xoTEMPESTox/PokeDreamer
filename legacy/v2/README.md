# PokéWorld v2 — Implementation Plan: RSSM & Dreamer-Style Imagination Policy

**Goal**: Implement a Recurrent State-Space Model (RSSM) with discrete categorical latents at a higher visual resolution ($80 \times 72$ or native $160 \times 144$), and train an Actor-Critic reinforcement learning agent entirely inside the imagined latent rollouts of this world model.

---

## 0. Architectural Shift (v1 continuous vs. v2 discrete)

| Component | v1 Model (Current) | v2 Model (Dreamer-style) |
|---|---|---|
| **Fidelity / Res** | $40 \times 36$ pixels | **$80 \times 72$ or $160 \times 144$ pixels** |
| **Latent State** | 32-dim continuous vector $z$ (VAE bottleneck) | **Discrete Categorical Latents** (32 variables, 32 classes each) |
| **Transition Model** | Continuous GRU: $(z_t, a_t) \rightarrow z_{t+1}$ | **RSSM** (Deterministic $h_t$ + Stochastic discrete $s_t$) |
| **Policy Source** | Frozen PPO checkpoint (reused baseline) | **Actor-Critic trained inside imagination** |
| **Evaluation** | Lookahead MPC Planner search | **Policy zero-shot execution on PyBoy** |

---

## 1. Phase 1: High-Resolution Data Collection (Day 1-2)

We need to capture higher-resolution observations to preserve retro boundaries and text boxes.

- [ ] Modify `collect_data.py` to capture and compress frames at $80 \times 72$ (or native $160 \times 144$).
- [ ] Maintain the WRAM extraction fields (position, map_id, HP) to serve as evaluation probes.
- [ ] Run PPO agent to collect 200k steps of high-resolution transitions.

---

## 2. Phase 2: Discrete RSSM World Model (Day 3-6)

### 2.1 ResNet-based CNN Encoder & Decoder
- [ ] **Encoder**: CNN with 2 residual skip connection layers to preserve high-frequency spatial text/sprite details.
- [ ] **Decoder**: Transpose CNN with residual links.

### 2.2 Recurrent State-Space Model (RSSM)
- [ ] **Deterministic path**: GRU hidden state $h_t$ (size 512).
- [ ] **Stochastic path**: Categorical distribution over $32 \times 32$ discrete latents. 
  - Representation/Posterior: $q(s_t | h_t, e_t)$ from encoder features.
  - Transition/Prior: $p(s_t | h_t)$ predicted from GRU.
- [ ] Train RSSM with multi-task loss:
  1. Reconstruction loss (Binary Cross-Entropy over pixels).
  2. KL-balancing loss (minimize KL divergence between prior and posterior).
  3. Reward prediction loss.

---

## 3. Phase 3: Actor-Critic Policy Training in Imagination (Day 7-10)

This is the core contribution: training the policy entirely inside the world model's imagined latent space.

### 3.1 Network Heads
- **Actor (Policy)**: $\pi(a_t | h_t, s_t)$ outputs action logits.
- **Critic (Value)**: $v(h_t, s_t)$ estimates expected discounted returns.

### 3.2 Imagined Rollouts & Policy Update
- [ ] From a starting state $s_0$, roll out the RSSM transition model autoregressively using actions sampled from the actor $\pi$.
- [ ] Horizon length: $H=15$ steps in imagination.
- [ ] Compute value targets using lambda-returns:
  $$V_t^\lambda = \hat{r}_t + \gamma (1 - \lambda) v(h_{t+1}, s_{t+1}) + \gamma \lambda V_{t+1}^\lambda$$
- [ ] **Actor Loss**: Maximize lambda-returns via policy gradient (or pathwise gradients through continuous reparameterized discrete distributions).
- [ ] **Critic Loss**: Minimize MSE between value estimates and lambda-returns.

---

## 4. Phase 4: Evaluation & Ablation (Day 11-12)

- **World Model Fidelity**: Compare visual reconstruction sharpness of $80 \times 72$ continuous VAE vs. discrete RSSM.
- **Imagination Training vs. Real Baseline**: Compare the success rate of the policy trained in imagination against the baseline PPO policy on PyBoy (overworld navigation trials).
- **Compounding Drift**: Map out coordinate drift of RSSM discrete rollouts out to $K=50$ steps.

---

## Key Risks & Mitigations

> [!WARNING]
> **Discrete Gradient Estimation**: Backpropagating gradients through discrete categorical distributions (for policy update) requires the Gumbel-Softmax reparameterization trick or straight-through estimators.
> *Mitigation*: Use PyTorch's `torch.nn.functional.gumbel_softmax` or straight-through estimation for Categorical variables.

> [!CAUTION]
> **Imagination Value Overfitting**: The critic can overfit to imagined rollouts, leading to poor policy updates.
> *Mitigation*: Update the world model and policy iteratively (collect fresh environment trajectories every $N$ training epochs).

---

## User Review Required

> [!IMPORTANT]
> Please review this v2 plan. It sets up the path to build a true model-based RL agent (DreamerV2 style) trained in discrete latent space imagination at $80 \times 72$ resolution.
