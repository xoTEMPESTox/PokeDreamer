# PokéWorld — 3-Week Build Plan
**Core contribution: a learned RAM-state dynamics model (world model) for Pokémon Red, used for lookahead planning over imagined futures — not a VLM-wrapper-around-screenshots project.**

Repo name: `pokeworld` (or `redworld`)
Subtitle: *A Learned Dynamics Model of Pokémon Red for Lookahead Planning*

---

## 0. The Reframe (why this plan differs from v1)

Old framing (rejected by senior):
```
Screenshot → VLM → Action      (perception + reaction, no prediction = not a world model)
```

New framing:
```
State_t + Action_t → Dynamics Model → State_t+1   (this is the actual world model)
Planner imagines several action sequences, asks the world model what happens, picks the best imagined future, hands the winning macro-action to PPO to execute.
```

The world model is now the **star** of the project. The planner can be simple/even rule-based at first. PPO is just the executor. This is the architecture your senior would recognize as a real model-based system (Dreamer/PlaNet/MuZero lineage), just symbolic instead of pixel-latent.

**Test for whether you're done:** can you point at a component and say "this learned `(s_t, a_t) → s_t+1` from data, and the planner used its rollouts — not the real emulator — to decide"? If yes at every decision point, you have a world model project. If the planner ever needs to actually step the emulator to "see what happens," you've slipped back into reactive planning.

---

## 1. State & Action Representation (Day 1)

Use RAM-derived symbolic state, not pixels — this is the single biggest scope-control decision in the whole plan.

### State Schema
```
state = {
    map_id:        int
    x, y:          int
    direction:     {up,down,left,right}
    in_battle:     bool
    battle_turn:   int (0 if not in battle)
    party_hp:      vector or scalar (sum/avg)
    badges:        bitmask
    event_flags:   relevant subset (e.g. has_parcel, talked_to_oak)
    dialog_open:   bool
}
```

### Action Space
```
action ∈ {up, down, left, right, A, B, start}
```

- [ ] Write a RAM-hook extractor against PyBoy that pulls this struct every step.
- [ ] Confirm it's stable across map transitions and battle entry/exit.

**Deliverable:** a script that dumps `(state_t, action_t, state_t+1)` rows to disk while the frozen PPO plays.

---

## 2. Data Collection (Day 2-3)

- [ ] Run the frozen PPO (PWhiddy checkpoint) for enough episodes to collect **500k-1M transitions** covering: open-world walking, building entry/exit, battle entry/exit, menu/dialogue states.
- [ ] Make sure data isn't *only* successful runs — include stuck loops, wall-bumps, battle losses too. The dynamics model needs to learn what happens on failed actions, not just the happy path.
- [ ] Stratify/check coverage: are buildings, battles, and dialogue transitions represented in reasonable numbers, or do they need targeted collection (e.g. scripted episodes that walk straight at a door)?

**Deliverable:** a labeled dataset file (e.g. parquet/npz) of transitions, plus a short coverage report (counts per transition type).

---

## 3. The Dynamics Model — Core Deliverable (Day 4-7)

### 3.1 One-step model (baseline, Day 4-5)
```
MLP(state_t, action_t) → state_t+1
```
- [ ] Encode categorical fields (map_id, direction, action) as embeddings/one-hot, numeric fields (x,y,hp) as normalized scalars.
- [ ] Separate prediction heads per field (position regression, map_id classification, in_battle classification, etc.) rather than one flat output.
- [ ] Train with a simple supervised loss (cross-entropy for categorical heads, MSE for numeric heads).

**Eval immediately:** per-field accuracy on held-out transitions (position MAE, map-transition accuracy, battle-flag accuracy).

### 3.2 Multi-step rollout model (Day 6-7)
- [ ] Use the one-step model autoregressively: feed its own predicted `state_t+1` back in as input for `state_t+2`, etc., out to k=10-50 steps.
- [ ] Measure rollout drift: how far predicted (x,y) diverges from ground truth as k grows. Report this compounding error honestly.
- [ ] Optional: train directly on k-step targets (`state_t → state_t+k`) for a few fixed k to reduce compounding drift.

**Deliverable:** a `rollout(state, action_sequence, k)` function that returns imagined future states without touching the emulator, plus a drift-vs-k plot.

---

## 4. Planner: Imagination-Based Action Selection (Day 8-11)

```
for each candidate action-sequence (or single macro-action) in {go_north, go_east, go_west, go_south, enter_building, ...}:
    imagined_state = world_model.rollout(current_state, sequence, k)
    score = evaluate(imagined_state, objective)

choose argmax(score) sequence
hand winning sequence to PPO executor (or to a simple scripted move-toward-target controller)
```

- [ ] Day 8: define `evaluate(state, objective)` — start simple: distance-to-target-map, badge count, HP threshold, "reached unexplored map_id."
- [ ] Day 9-10: implement candidate generation (a small fixed set of macro-actions, e.g. "walk N in direction D," "approach nearest building," "approach nearest battle trigger") and the imagine-then-score loop.
- [ ] Day 11: wire the winning macro-action to an executor. Either:
  - (a) Frozen PPO with the simple controller-loop from v1 of the plan.
  - (b) A trivial scripted move-toward-(x,y) controller using A* / BFS over known map tiles.

---

## 5. Evaluation (Day 12-13)

- **Dynamics model accuracy**: per-field one-step accuracy/MAE; k-step rollout drift curve.
- **Planning ablation (the key experiment):** run the agent with planning (imagine-then-choose) vs. a no-imagination baseline (e.g. greedy/random action choice, or planner picks without consulting the world model). Compare task success rate, steps-to-goal, stuck-rate.
- **Counterfactual demo**: show concrete cases where the world model successfully predicted dead ends or bad paths and steered the planner away.
- **Task progress**: badges earned, towns reached.

---

## 6. Stretch Goals (Day 14)

- [ ] Latent version: replace symbolic RAM state with a learned encoder over a downsampled screen (`screen → z`), then train `z_t + a_t → z_t+1` (Dreamer-lite).
- [ ] Longer/deeper search: beam search or short MCTS over the imagined rollouts instead of single-shot greedy macro-action selection.
- [ ] VLM as a *front-end* only: use a vision-language model purely to translate a natural-language objective ("get me a badge") into the `evaluate()` scoring function's target.

---

## 7. Writeup + Demo (Day 15)

- [ ] Repo: `pokeworld`. README leads with the dynamics-model architecture diagram and the planning-ablation result.
- [ ] Be explicit and correct about what each piece is:
  - **World model**: learned `(s_t, a_t) → s_t+1` dynamics model trained on PPO rollout data.
  - **Planner**: imagination-based action scorer (hand-written objective function over imagined rollouts).
  - **Executor**: frozen PPO checkpoint (or scripted controller).
- [ ] Headline claim for resume: *"Trained a dynamics model of Pokémon Red from gameplay trajectories and used it to perform lookahead planning over imagined futures, with an ablation showing planning improves task success rate over reactive baselines."*

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Compounding rollout error makes k-step imagination unreliable | Report drift honestly; cap k to where accuracy is still useful; consider direct k-step training targets |
| RAM extraction for battle/dialogue state is buggy or incomplete | Validate against a handful of manually-verified episodes before scaling data collection |
| Planner's hand-written objective function feels too simple to be "real planning" | Frame this correctly — simplicity of the planner is fine and expected; the contribution is the world model + the ablation, not planner sophistication |
| Temptation to add VLM/LLM back in as the decision-maker under time pressure | Keep it as an optional thin translation layer only (Section 6); never let it choose actions directly |
| Running out of time before the ablation (Section 5) | This is the non-negotiable deliverable — protect Day 12-13 even if it means cutting the stretch goals in Section 6 |

---

## What Changed From the VLM-Planner Version

| | Old plan | New plan |
|---|---|---|
| Star of the project | VLM planner | Learned dynamics (world) model |
| What's learned | Frozen PPO + tiny stuck classifier | RAM dynamics model (one-step + multi-step rollout) |
| Planner role | Issues goals based on perception | Imagines futures via world model, scores them, picks best |
| PPO role | Primary executor, steered by goals | Boring executor only, can be replaced by scripted controller |
| Key evaluation | Instruction success rate | Planning ablation: with-imagination vs. without |
| Risk profile | Engineering-heavy, low research risk | Slightly more research risk, but directly answers the "that's not a world model" critique |