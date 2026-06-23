# Pokémon Red Hierarchical Agent — 3-Week Build Plan
**Architecture: Frozen PPO (System 1) + VLM Planner (System 2) + Stuck Predictor, with goal-conditioned BC as fallback**

---

## 0. Decision Tree (read this first)

```
Day 1-3: Test whether frozen PPO is steerable via macro-goal + controller
         |
         ├── YES, "go toward (x,y)" reliably moves the agent closer
         |     → Stay on Option A (zero PPO training). Proceed to Week 1 plan.
         |
         └── NO, PPO ignores macro-goals / wanders regardless of target
               → Fall back to Option B (goal-conditioned behavior cloning).
                 Budget 1-4 hrs of BC training on a 4090. Proceed to Week 1B.
```

Do not decide this in your head — decide it empirically by Day 3 with a logged test (see 1.3). Everything downstream depends on this branch.

---

## WEEK 1 — System 1 Wiring + Steerability Test

### Day 1: Environment + Checkpoint Setup
- [ ] Clone PWhiddy's PokemonRedExperiments (primary candidate — most documented, reaches Pewter/Cerulean reliably)
- [ ] Get the env running headless, confirm you can step the frozen policy and read RAM state (player x/y, map id, badges, party HP)
- [ ] Build a `GameState` extractor: `{map_id, x, y, facing, in_battle, dialog_open, party_hp, badges}` from PyBoy RAM hooks
- [ ] Confirm screen capture (for VLM) and RAM state are synchronized in the same step

**Deliverable:** a script that runs the frozen PPO for N steps and prints (x,y,map) trajectory to a log file.

### Day 2: Goal Translator + Controller Loop
- [ ] Define goal schema: `{"type": "goto", "map_id": ..., "xy": [x,y]}` and `{"type": "enter_building"}`, `{"type": "talk_to_npc"}`
- [ ] Write the **external controller** (this replaces any PPO retraining):
  ```
  loop:
      obs = get_observation()
      action = frozen_ppo(obs)
      step(action)
      dist = distance(current_xy, target_xy)
      if dist < threshold: goal_complete()
      if no_improvement_for(K steps): mark_stuck()
  ```
- [ ] No goal is ever fed into the PPO's input — it only sees the raw screen exactly as during its own training.

**Deliverable:** controller that can take a hardcoded `(map_id, x, y)` and run the PPO until arrival or timeout.

### Day 3: Steerability Test (the critical checkpoint)
- [ ] Run 20-30 trials: random current position → random nearby target within same map
- [ ] Log: success rate, average steps, stuck-rate (no progress for >K steps)
- [ ] **Decision point:** if success rate is reasonably high (PPO naturally biases toward unexplored/forward movement and the controller can just let it run and re-check distance), continue with Option A. If the PPO actively wanders away from target with no correlation, branch to Option B (BC fallback below).

**This is the single highest-risk step in the whole project — do it first, not last.**

### Day 4-5: Stuck/Progress Predictor (Part 3, Option C)
- [ ] Collect 50k-150k (state, action, next_state) transitions from frozen PPO rollouts (you already have these logs from Day 3)
- [ ] Label: `stuck=1` if position hasn't changed (or revisited a tile) for the last N steps, else `0`
- [ ] Train a small classifier (MLP on RAM-derived features, or a tiny CNN on downsampled screen) — this is genuinely a few hours, not days
- [ ] Wire it into the controller: if `stuck_probability > threshold`, trigger replan signal

**Deliverable:** working stuck-classifier integrated into the controller loop, with a logged precision/recall on held-out rollouts.

### [Branch] Week 1B — If Steerability Test Fails: Goal-Conditioned BC
- [ ] Auto-label existing PPO trajectories with goals (no humans needed):
  - Segments with consistent direction → `goal = move_{N/S/E/W}`
  - Segments ending at a building door → `goal = enter_building`
  - Segments where battle starts → `goal = engage_battle`
- [ ] Build dataset: `(screen, goal_onehot) → action_taken_by_PPO`
- [ ] Train via behavior cloning (cross-entropy vs. PPO's own actions) — 100k-500k transitions, 1-4 hrs on a 4090
- [ ] Replace the frozen PPO call with this new goal-conditioned cloned policy in the same controller loop from Day 2
- [ ] Re-run the Day 3 steerability test against this new policy

This adds ~1 day of work, not weeks, and keeps you on schedule.

---

## WEEK 2 — System 2: Vision-Language Planner

### Day 6-7: State Summarizer for the LLM
- [ ] Convert `GameState` + a downsampled/cropped screen (or ASCII/tile-map representation) into a short structured prompt context, e.g.:
  ```
  Location: Route 2, near Viridian Forest entrance
  Party: Charmander HP 18/20
  Badges: 0
  Nearby: building visible north, tall grass to the east
  Recent goals: [reached Pallet Town exit] [entered Oak's Lab] [completed]
  ```
- [ ] Decide: full screenshot to a multimodal model (e.g. via the Anthropic/OpenAI vision API) vs. RAM-derived structured text only. Structured text is cheaper and more reliable for v1; add screenshots as enrichment if time allows.

### Day 8-9: Planner → Goal Schema
- [ ] System prompt for the LLM: it sees current state + objective ("get the first badge") and must emit ONE next goal in your fixed JSON schema (`goto`, `enter_building`, `talk_to_npc`, `engage_battle`, `use_item`)
- [ ] Planner is called only at decision points (goal completion, stuck signal, or every K steps) — not every frame. This keeps API costs/latency sane.
- [ ] Implement the replan trigger: `stuck_predictor fires → call planner with "current approach failed" context`

**Deliverable:** end-to-end loop: Planner emits goal → Controller drives frozen PPO/BC policy toward it → Stuck predictor monitors → replan on failure → repeat.

### Day 10: Dialogue/Menu Handling
- [ ] Identify how much of "talk to NPC" / "navigate menu" is already handled by the base policy when dialog_open=True (PWhiddy's agent already presses A/B to advance text in some cases — verify this directly rather than assuming)
- [ ] If insufficient, add a trivial scripted handler (not RL): when `dialog_open`, just press A on a timer until it closes. This is legitimate engineering, not a gap — note it honestly in your writeup as a scripted micro-skill rather than claiming it's learned.

---

## WEEK 3 — Evaluation, Robustness, Polish, Writeup

### Day 11-12: Evaluation Harness
Define and log these metrics across N full runs (e.g. 10-20 runs from house start to first gym):
- **Instruction success rate**: % of planner-issued goals that the controller actually completes
- **Replan rate**: how often stuck-predictor correctly triggers a replan vs. false positives
- **Progress metrics**: towns reached, badges earned, steps-to-badge-1
- **Stuck-predictor precision/recall** on held-out trajectories
- **Latency**: planner calls per run, average wall-clock per goal

### Day 13: Stretch Goal — Skill Router (only if ahead of schedule)
- [ ] If Week 1-2 finished early, add a thin "skill selector" layer: Planner doesn't just emit a goto-goal, it picks among {Navigate, Battle-handoff, Dialogue-handoff} skill modules, even if Navigate is the only RL-backed one and the others are scripted/heuristic for now
- [ ] This is purely a demo-quality upgrade (Part 5, Option 4 lite) — do not let it threaten the Week 1-2 deliverables.

### Day 14-15: Demo + Writeup
- [ ] Record a video: planner's text goals overlaid on screen as the agent executes them, with stuck/replan events highlighted
- [ ] Write up honestly: what's learned (stuck predictor, optionally BC policy) vs. reused (frozen PPO) vs. scripted (dialogue advance)
- [ ] Report the metrics from Day 11-12 — recruiters will trust measured numbers over claims

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Frozen PPO ignores macro-goals entirely | Day 3 test catches this early; BC fallback costs ~1 extra day |
| LLM planner calls too slow/expensive for real-time play | Only call planner at decision points, not every frame; cache/batch where possible |
| PPO checkpoint quality varies by source (PWhiddy vs PokeRL vs neroRL) | Pick PWhiddy first (most reliable navigation + documented); keep PokeRL as backup if checkpoint loading breaks |
| Scope creep into full skill-policy architecture (Part 4) | Treat it as a stretch goal only, explicitly gated behind finishing Weeks 1-2 |
| Misty/Mt. Moon gaps in base policy make demo stall | Cap your "objective" at first badge (Brock) for the core demo; mention further-game limitations as known gaps, not failures |

---

## What You Can Honestly Claim on a Resume/Portfolio

- Hierarchical System 1 / System 2 agent design
- Reused open-source RL checkpoint with zero retraining (or minimal BC, clearly labeled)
- Built a goal-translation/control layer bridging symbolic planner output to a continuous-control policy
- Trained a lightweight learned world-model component (stuck/progress predictor) from rollout data
- Implemented closed-loop replanning based on predictive monitoring
- Quantitative evaluation: instruction success rate, replan accuracy, task completion metrics