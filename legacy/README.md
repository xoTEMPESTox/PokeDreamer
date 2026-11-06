# Earlier versions (v1, v2) and the original v3 write-up

The current project lives in [`../lab/`](../lab/) and is described in the [top-level README](../README.md). This folder keeps the earlier work as it was, for reference and reproducibility. The project was called "PokéWorld" in these versions.

| path | what it is | write-up |
|---|---|---|
| [`v1/`](v1/) | Continuous-latent world model (VAE + latent dynamics) with an MPC planner | [`README_v1v2.md`](README_v1v2.md), [website v1 page](https://pokedreamer.priyanshusah.com/v1.html) |
| [`v2/`](v2/) | Discrete RSSM world model (DreamerV3-style) | [`v2.md`](v2.md), [`v2/README.md`](v2/README.md), [website v2 page](https://pokedreamer.priyanshusah.com/v2.html) |
| [`v3.md`](v3.md) | The original long-form v3 write-up; superseded by the top-level README and [`../lab/results/`](../lab/results/) | |
| [`README_v1v2.md`](README_v1v2.md) | The project README before the v3 release | |

## Running the old code
Each version's scripts put their own version folder on `sys.path`, so they run from the repo root, for example `python legacy/v2/scripts/train_rssm.py` (the website's commands use these paths). The v1 re-evaluation in v3 (`lab/v1_reeval.py`) loads the v1 model code from `legacy/v1/src`.

Weights for v1 and v2 are on the [Hugging Face model repo](https://huggingface.co/xxxTEMPESTxxx/PokeDreamer) (top-level files and the `dynamics/`, `probe/`, `vae/` folders); the data on the [dataset repo](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer). As before, no ROM is included: you need your own legally obtained copy of Pokémon Red.

