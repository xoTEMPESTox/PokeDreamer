# 3D visuals: voxel-diorama prototype and plans (shelved 2026-11-04)

The goal was to make the project's demos easier to look at by showing real-game and world-model frames as 3D scenes. A
one-frame prototype works. Full automation and a live version were judged too much work for now, so this records what was
built and measured, and what a later attempt would need.

## What was built (one frame, offline)

A Game Boy frame (Pallet Town, 160x144) rendered as a voxel diorama in Blender 4.5 LTS with Cycles on the CPU.

| Step | Script | What it does |
|---|---|---|
| 1 | `prep_pair.py` | Crops one real/imagined pair from a rollout comparison video frame, scales back to 160x144, quantises to the 4 shades, assigns a tile id to each 8x8 cell, writes a tile contact sheet |
| 2 | `build_boxes.py` | Turns the frame into coloured boxes: ground relief, dome trees, standing house walls with the roof art on a gable, standing sprite and sign |
| 3 | `render_scene.py` | Builds one mesh in Blender and renders it: warm sun, sky fill, tilt-shift depth of field, shadow-catcher floor |
| 4 | `compose.py` | Background plus a before/after image next to the original frame |

**Hand-made part:** the tile labels in `build_boxes.py` were assigned by hand for this frame's 45 distinct tiles. This is not yet automatic.

**Measured** on an i9-14900K, 12 threads, below-normal priority, no GPU:
- 31,457 boxes.
- 1920x1080 at 128 samples with denoising: **38 s per frame**.
- 960x540 at 24 samples: 2.6 s.

**About the imagined frame:** this pair's world-model frame (step 36 of a 64-step rollout) differs from the real frame in 6 of 23,040 pixels. The player is standing still. Its diorama would be the same image, so it is not a meaningful test of the world model.

### Reproduce

The source frame is `work/viz/raw_frame_step36.png`, taken from `work/videos/main_test_id_k64_up.mp4`. Pair index 5 is the 6th row of that frame. Outputs go to `work/viz/`.

```
python lab/viz/prep_pair.py work/viz/raw_frame_step36.png 5 work/viz/p5
python lab/viz/build_boxes.py work/viz/p5_real
work/tools/blender-4.5.15-windows-x64/blender.exe -b --factory-startup -P lab/viz/render_scene.py -- work/viz/p5_real_boxes.npz work/viz/p5_real_render.png 1920 128 1.4
python lab/viz/compose.py work/viz/p5_real_render.png work/viz/p5_real.npy work/viz/p5_real
```

Blender 4.5.15 LTS is the portable zip from download.blender.org, with its sha256 checked against the official sums file.

Gotcha: give Blender an absolute output path. A relative one resolves against the drive root, which is why `render_scene.py` uses `abspath`.

## Plan A: automated offline clips (estimated ~1 day; not built)

1. **Tile labels without hand work.** Real frames take exact tile ids from the game. PyBoy 2.7 exposes `tilemap_background`, `get_sprite`, `game_area_collision` and `memory`; the screen tile buffer is at RAM `0xC3A0`. A one-time style table maps each tile or building block of the Pallet-to-Pewter tilesets to a 3D piece. Estimate: 2-3 h.
2. **Imagined frames.** World-model frames have no RAM, so each cell gets a nearest-tile match. Cells with low match confidence get "dream haze" (fog) and black areas become void. This would show where the imagination fails. Use a frame pair where the model actually drifts. Estimate: 1-2 h.
3. **Clip.** Real and imagined side by side, rendered on the CPU. Estimate: ~1 h of CPU per 100 frames at 1080p.

## Plan B: live, as the game runs, like a mod (estimated 3-4 days for outdoors; not built)

No hooking is needed: the emulator gives the tile map, scroll position, sprites and collision map every frame. All estimates below are in working days.

| Part | Estimate |
|---|---|
| Game state per frame from PyBoy | ~0.5 day |
| Style table, outdoors | 1 day |
| Style table, forest, Pokémon Centre, Mart, gym | +2-3 days |
| Real-time renderer (three.js preferred, since recorded runs could then replay in any browser without the ROM; Godot looks better but is harder to share) | 1-1.5 days |
| Sprites extruded automatically; battles and menus as a flat 2D screen in the scene | ~0.5 day |
| Live 3D view of the world model ("play inside the dream"), after the outdoor version | +1-2 days, needs a free GPU |

Real time loses Cycles' bounce light, so it will look flatter than the still.

## Realistic restyling, the "DLSS" wish (notes only; nothing tested here)

- **DLSS cannot do this.** It rebuilds the *same* image at a higher resolution; it does not change the style.
- **Offline restyle:** image-to-image diffusion with structure control, e.g. ControlNet on depth, edges or tiles, keeps the layout and changes the look. The input can be the diorama render or the game frame. It can run locally on the 4090 once training is free, or through a hosted image editor given the render as input. Grounding the AI image in a real frame keeps the post honest; label it as AI-restyled.
- **Real-time restyle:** few-step img2img (e.g. StreamDiffusion with SD-Turbo) can reach video rates on a 4090, but it flickers from frame to frame. Temporal consistency is the open problem.
- **Related research:** Intel Labs' "Enhancing Photorealism Enhancement" (2021) made GTA V photoreal, but used the game's G-buffers.

## References found during research

- Pokémon Red/Blue ported to PC with 3D voxel graphics (a recomp that hooks the running game, so it cannot take world-model pixels): https://www.tweaktown.com/news/112990/original-game-boy-versions-of-pok-mon-red-and-blue-ported-to-pc-with-3d-voxel-graphics/index.html
- 3DSen, an NES emulator with hand-made 3D profiles per game: https://www.tweaktown.com/news/105963/experience-classic-nes-games-from-the-1980s-in-full-3d-with-3dsen/index.html
- HD-2D Pallet Town and Vermilion fan demo in Unreal Engine 5: https://hypebeast.com/2023/4/pokemon-hd-2d-remake-unreal-engine-5-video
- Hyper-realistic Pallet Town in Unreal Engine 5: https://mirthico.pages.dev/posts/pokemon-fan-creates-hyper-realistic-pallet-town-in-unreal-engine-5
- Free CC0 3D props (e.g. Kenney packs) avoid modelling by hand.

**Risks:**
- Do not use 3D models ripped from newer Pokémon games.
- Nintendo is known for taking down Pokémon fan projects, so keep anything public to videos plus code that needs your own ROM.
