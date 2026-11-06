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
