# PokeDreamer

**A diffusion world model of Pokémon Red, measured against the game's own memory, and an agent whose navigation was learned entirely inside it that wins the first gym in the real game.**

[![tests](https://github.com/xoTEMPESTox/PokeDreamer/actions/workflows/tests.yml/badge.svg)](https://github.com/xoTEMPESTox/PokeDreamer/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
· [Website](https://pokedreamer.priyanshusah.com) · [Model weights](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer) · [Dataset](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer) · [Results](lab/results/README.md)

<p align="center"><img src="docs/assets/v3/opener_3d.gif" width="80%" alt="The PokeDreamer title rising over the pond in a 3D render of Viridian City built from the game's map data"></p>

## In 30 seconds
- **A world model of a real game.** A diffusion model learned to predict Pokémon Red frame by frame from recorded play: you press a button, it imagines the next screen.
- **An agent that learned inside the dream.** Ten navigation skills were trained purely in that imagination, with zero real-game steps, then dropped into the real game, where the agent wins the Boulder Badge in **30/30** runs.
- **Measured, not eyeballed.** The game's own memory says exactly where the agent and the imagined world really are, so every claim below is a number: where the dream drifts, which shortcuts the agent learned to exploit, and why battles can't be learned this way.
- **Seen in colour.** A neural upscaler and colorizer turn the 72x80 greyscale frames into full-resolution colour for display (99.27% per-pixel accuracy on held-out frames).

<table>
<tr>
<td width="38%"><img src="docs/assets/v3/gameplay_colour.gif" alt="The agent in the real game"></td>
<td width="62%"><img src="docs/assets/v3/real_vs_dream.gif" alt="Real game next to the world model's imagination"></td>
</tr>
<tr>
<td><sub>The real game: walking is a policy trained only in the world model; battles are rules.</sub></td>
<td><sub>Left, the real game. Right, the world model imagining the same moments from 4 real frames and the same buttons (re-anchored every 10 steps).</sub></td>
</tr>
</table>

## What this is
A research project on world models for game agents, built around one advantage a game gives you: the emulator's RAM is ground truth for every imagined frame. That makes it possible to measure, in game units (tiles, maps, battle turns), what a learned simulator gets right, where it drifts, and how a policy trained inside it exploits its mistakes.

- **World model:** wm31, a DIAMOND-style diffusion model of 72×80 frames, 16.7M parameters, conditioned on the last 4 frames and buttons.
- **Agent:** a System 1/2 design. Ten navigation legs (System 1) are small policies trained purely in the world model's imagination, with zero real-game steps. Battles and goal sequencing (System 2) are rules; a bake-off also tested local LLMs as System 2.
- **Measurement:** held-out episodes split by episode, a frame probe that reads map, tile and battle state from pixels, baselines (no-change, copy-last, greedy path-following, PPO in the real game), 3 seeds, bootstrap confidence intervals.

## Headline results (measured; sources in [`lab/results/`](lab/results/README.md))
| | result | page |
|---|---|---|
| Boulder Badge, real game, 10-leg agent | **30/30 runs** (3 seeds × 10) | [v3_gym](lab/results/v3_gym.md) |
| Runs where every navigation leg is a world-model policy (no fallback anywhere) | **14/30** | [v3_gym](lab/results/v3_gym.md) |
| Navigation legs at 100/100 on all 3 seeds | 7 of 10; to_pewter_pc 100/0/100, forest 100/69/97 | [v3_gym](lab/results/v3_gym.md) |
| Start-tile robustness after fixing a reward hack (move gate) | **2,339/2,340** episodes over 26 start tiles | [v3_tile_sweep](lab/results/v3_tile_sweep.md) |
| Can battles be trained inside this world model? | **No:** imagined battles end early in 74.6% of held-out windows (real-frame probe: 1.1%) | [v3_battle_fidelity](lab/results/v3_battle_fidelity.md) |
| Local LLMs as System 2 | 5 of 6 reach 10/10 badges | [v3_system2](lab/results/v3_system2.md) |

<p align="center"><img src="lab/results/figures/fig_door_warp.png" width="80%" alt="Imagined vs real map changes per button around Pokémon Center doors and mats"></p>

## What the measurements showed
- **Policies exploit the model's errors, and RAM is what reveals it.** A per-step reward clip let policies farm a ledge loop (+4 per loop); a potential-based "move gate" fixed it (2,339/2,340). The frame probe confuses the identical-looking Pokémon Center interiors, so pressing *right* on the Pewter Center mat looked like an exit in imagination; treating Center reads as one interior took leave_pewter_pc from 72/68/100 to 100/100/100 (figure above).
- **Drift depends on the map.** On the best-covered map (Viridian City) imagined position error at 8 steps beats the no-change baseline (2.71 vs 3.64 tiles); in Viridian Forest it is 15.65 tiles, far above the probe's own error on real forest frames (about 2), so the model imagines plausible forest somewhere else. See [v3_dwm_eval](lab/results/v3_dwm_eval.md).
- **Battles are out of reach for a 4-frame pixel model**: battle onset is roughly right, but imagined battles end early and their frames are worse than copying the last frame from 2 steps on.
- **Honest scope:** with a fast emulator the world model is *not* a sample-efficiency win: PPO in the real game reaches ≥95% within 10.8k-41.5k real steps on the same navigation tasks ([v3_agent](lab/results/v3_agent.md)). Its value here is as a measured object and a simulator you can train in without touching the game. Battles and planning are rules.

<p align="center"><img src="lab/results/figures/fig_drift_permap.png" width="80%" alt="Imagination drift per map"></p>

## Run it
You need your own legally obtained Pokémon Red ROM (not included) at `Pokemon - Red Version (USA, Europe).gb` in the repo root.

```bash
conda create -n pokedreamer python=3.11 && conda activate pokedreamer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # or .../whl/cpu
pip install -r lab/requirements.txt        # pinned versions used for v3 (environment.yml is the v1/v2 env)
python lab/get_weights.py                  # world model, probe, upsampler, policies from Hugging Face
python -m pytest -q lab/tests/             # tests that need the ROM or data skip if they're missing
```
Every public number has a reproduce command in [`lab/README.md`](lab/README.md) and a script in [`lab/experiments/`](lab/experiments/README.md); exact settings are in [`lab/configs/`](lab/configs/).

## Repository layout
| path | contents |
|---|---|
| [`lab/`](lab/README.md) | v3: environment, world model, probe, imagination RL, agents, evaluation, figures, video tools, tests |
| [`lab/results/`](lab/results/README.md) | results pages with every table and its source file |
| [`lab/experiments/`](lab/experiments/README.md), [`lab/configs/`](lab/configs/) | reproduce scripts and exact settings |
| [`legacy/`](legacy/README.md) | v1 (VAE + latent dynamics + MPC) and v2 (RSSM), kept as they were |
| [`docs/`](docs/) | the website |

## Citation
See [`CITATION.cff`](CITATION.cff). Code is MIT-licensed ([`LICENSE`](LICENSE)). No ROM, save state or other game asset is distributed. Pokémon is a trademark of Nintendo, Creatures Inc. and GAME FREAK inc.; this project is not affiliated with them.
