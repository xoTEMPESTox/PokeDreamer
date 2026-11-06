# Catching Pokémon with a System 1/2 loop in the real game (measured)

This page evaluates a closed-loop System 1/2 agent performing an autonomous Pokémon catching quest in the real Game Boy emulator. Starting from Viridian City with Charmander L8 and a fixed budget of 15 Poké Balls, two System 2 decision strategies are compared: throwing balls at full wild HP versus weakening the target to HP <= 60% before throwing. Weakening first nearly triples the catch rate per ball (85.5% vs. 31.1% in the 10-run protocol; 86.5% vs. 36.8% in a 50-run replication), enabling 10 / 10 runs to reach the 10-catch objective compared to 0 / 10 for throwing at full HP.

**Reproduce:** Reproducible via [lab/README.md §3.12](../README.md#312-catching-pokémon-strategy-comparison-throw-vs-weaken-first) (`python -m lab.agent.catch_quest --runs 10 --seed 100 --strategy {throw,weaken}`).

---

Code: `lab/agent/catch_quest.py`. Start: `saves/pokedex_viridian.state`, made by `python -m lab.make_pokedex_state` from the Pokémon Red ROM. It plays the starter sequence, then Oak's Parcel (Pokédex); Oak gives no Poké Balls in Red, so 5 are bought. State: Viridian City, Charmander L8, 5 balls, 2,175 money. Balls cost 200, so **15 balls is the hard budget per run**. No human input.

**Honest labels.**
- **System 2 v0:** a rule planner over RAM:
  - heal below 50% HP;
  - buy balls (Viridian Mart) while money allows;
  - catch until the target;
  - stop when out of balls and money.
- **System 1 skills:** scripted.
  - battle menu -> ITEM -> Poké Ball;
  - decline the nickname;
  - run when out of balls;
  - greedy navigation;
  - Route 1 grass patch.
- **Catch detection from RAM:** party + PC box count, ball count, battle ending without faint, Pokédex owned bits.

## Results: 10 runs per strategy (seeds 100-109, target 10 catches, budget 8,000 presses)

| System 2 strategy | catches mean ± sd [min, max] | catch rate per ball | balls thrown | knockouts / run | runs reaching 10 | presses |
|---|---|---|---|---|---|---|
| throw at full HP | 4.6 ± 1.4 [3, 7] | 31.1% | 14.8 | 0.0 | 0 / 10 | 1,468 |
| **weaken first** (attack while wild HP > 60%; throw if its HP <= 5) | **10.0 ± 0 [10, 10]** | **85.5%** | 11.9 | 0.2 | **10 / 10** | 1,911 |

Files: `work/catch_throw/summary.json`, `work/catch_weaken/summary.json`; first 5-run version `work/catch/`.

**Replication with 50 runs per strategy (fresh seeds 200-249; `work/catch_sweep/*/summary.json`, `work/sweep_cpu_summary.json`):**

| strategy | catches, mean [95% CI], min-max | runs reaching 10 | catch rate per ball |
|---|---|---|---|
| throw at full HP | 5.40 [4.86, 5.94], 1-10 | 1/50 | 36.8% |
| weaken first | 9.96 [9.90, 10.00], 9-10 | 48/50 | 86.5% |

This confirms the 10-run result.

**Independent verification:** Replaying seed 102 with the weaken strategy and reading RAM after the run confirms party 6 + PC box 5 = Charmander + 10 caught, matching the reported 10 catches with 10 balls thrown. An earlier throw-strategy check (seed 0) confirms party 1 -> 6, box 0 = 5 catches as reported.

**Reading:** With a fixed ball budget, the strategy, not the budget, decides whether "catch 10" is reachable. Weakening first nearly triples the catch rate per ball (Gen 1 catch odds rise as the wild Pokémon's HP falls). The cost is ~30% more presses and rare knockouts (2 in 10 runs). This is exactly the kind of decision a System 2 should make. Here it is a fixed rule; a reasoning System 2 would choose it from context (balls left, money, target).

**Limits:** Only Route 1 species (Rattata, Pidgey); distinct species owned rises from 1 to 3. More species need other routes (Route 22: Spearow, Nidoran; Route 2 / Viridian Forest: Caterpie, Weedle, Pikachu). GIF of one weaken run: `work/catch_showcase/run_102.gif`.
