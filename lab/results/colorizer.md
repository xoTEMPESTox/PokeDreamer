# Colorizer for real and imagined frames (measured; display only)

The game and the world model are greyscale. For figures and the video, frames are coloured per tile, the way the Game Boy Color coloured games, with our own palettes. **No metric anywhere in the project uses the colorizer**; it only changes how frames look.

**Reproduce:** stage 1 `python -c "from lab.color.native import validate; validate()"`; stage 2 `python -m lab.color.learned --collect`, then `--train`, then `--eval` (see `lab/color/learned.py`).

## Stage 1: exact colour for real gameplay
`lab/color/native.py` reads, for every pixel, which tile it shows (background, window or sprite), the tile id and its 2-bit colour index, from PyBoy's video memory, tile maps, scroll and window registers and sprite table. `lab/color/rules.py` turns that into a colour class:
- 107 hand-labelled tiles (`lab/color/labels.json`), chosen as the most-used tiles of the overworld, Pokémon Center, gym, forest and interior tilesets;
- tile ids 128-255 (the font shared by every tileset) are text;
- sprites in the screen-centre 16x16 box are the player, other sprites are NPCs;
- in battle the tile memory holds battle graphics, so the overworld labels are not used.

Each class has a 4-colour ramp (`lab/color/palette.json`), indexed by the pixel's colour index.

**Check:** rebuilding the screen from the per-pixel colour indices and the palette registers matches PyBoy's own screen on **99.9412%** of pixels (5,065,820 / 5,068,800) over 220 frames that include walking, mid-scroll frames, text boxes, a Pokémon Center and battles (`work/color/native/validate.log`).

## Stage 2: learned colour for world-model frames
World-model frames have no tile ids. `lab/color/learned.py` trains a 1.9M-parameter U-Net to predict each pixel's colour class at 144x160 from a 72x80 frame (the upsampler's 4 shades plus the raw frame), with stage-1 classes as targets. Training: 108,000 frames from random walks over 27 save states, 25 minutes on one RTX 4090. Test: 40,000 frames from the 10 held-out `seed102_*` save states (`work/color/learned/metrics.json`).

| | pixel class accuracy |
|---|---|
| all test pixels | **99.27%** |
| outside battles | 99.43% |
| in battles | 98.82% |

Per class (lowest first): NPC sprites 94.5%, signs 97.1%, ledges 97.2%, text boxes 97.6%, tall grass 98.1%; all other present classes 98.8-99.9%. Water, blue roofs, carpet and HP bars do not occur in the test frames.

3.6% of test pixels end up a different final colour from stage 1. That is more than the class error because the colour index comes from the upsampler's shade, which is not always pixel-exact (`lab/results/upsampler.md`).

**On imagined frames** (`work/color/learned/demo.png`): coherent imagination colours cleanly; where the world model's frame has broken down, the colorizer paints that breakdown in colour rather than hiding it.

## Battle colours v2: per-species colours, attack effects, HP bars (measured 2026-11-07)

Stage 1 (exact, real frames) now colours battles from RAM instead of one flat battle class:
- **Each Pokémon gets its own colours.** The battle Pokémon's name is read as text from RAM (`0xCFDA` enemy, `0xD009` player; the game's font, no ID table) and looked up in our own ramps in `lab/color/species.json`, written by hand for the Pokémon this project meets. Other Pokémon fall back to their primary type (`0xCFEA` / `0xD019`).
- **Attack effects** (sprites drawn during a move) are coloured by the move's type (`0xCFD5`), e.g. Ember orange.
- **HP bars** are green above 50% HP, amber from 20% to 50%, and red below 20% (from the HP bytes).
- **Battle intro:** the trainer's back sprite gets a neutral trainer ramp until the player's HP bar appears.

Stage 2 (learned, for world-model frames) was retrained as `model_v2.pt` (82 classes, 1.93M parameters, 25 min) on the old data plus a battle-heavy collection:
- 366 battles with 18 species, including all three starters as the player's Pokémon;
- the held-out split is by battle: 293 train / 73 test, seed 0, `work/color/battle/split_v2.json`.

The published `model.pt` (99.27%) is unchanged and remains the default.

| held-out (by battle), `work/color/learned/metrics_v2.json` | `model_v2.pt` |
|---|---|
| overall pixel class accuracy | 98.88% |
| non-battle | 99.37% |
| battle | 96.33% |
| Pokémon sprite regions | 96.65% |
| test frames | 34,948 |

- The sprite-region figure is dominated by the player's Charmander back sprite.
- Enemy species on held-out battles range from 61.7% (Kakuna, 2 test battles) to 86.3% (Spearow, 3 test battles). Charmander is at 95.6%.
- Stage 2 cannot see the move type, so it colours attack effects by guess (Ember came out purple in the showcase). That matters only for imagined frames; real frames use stage 1.

Showcase: `work/color/battle/battle_colours_v2.png` (greyscale | stage 1 | stage 2):
- 10 species mid-battle;
- the HP bar at 100%, 33% and 11%;
- an Ember frame.

Scripts are in `work/color/battle/scripts/`. Tests: `lab/tests/test_battle_rich_color.py`.
