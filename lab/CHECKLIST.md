# v3 close-out checklist

Status of the work that turns the v3 lab results into something others can see, run and read. Updated as items
finish; each line names the file or run that proves it. Dates are 2026.

Legend: [x] done · [~] in progress · [ ] open · (P) needs Priyanshu

## 1. Close Path A: learned navigation, Boulder Badge
- [x] All 10 navigation legs trained in imagination, 3 seeds (`work/agent_rl/*_wm31_gate_*`)
- [x] 10-leg quests: 30/30 badges; best leg set 14/30 runs with no BFS fallback anywhere (`agent_notes/results/v3_gym.md`)
- [x] leave_pewter_pc fixed by `--merge-centers`: 100/100 on 3 seeds (commit 8825102)
- [x] to_pewter_pc: 1,800 iterations give 100 / 0 / 100. Seed 1 stays stuck; reported as 2 of 3 seeds, no further fix
- [x] Battle go/no-go: no-go for wm31 as is (`agent_notes/results/v3_battle_fidelity.md`)
- [~] Forest at 2x iterations, seeds 1-2 (`work/chain_forest_3560.sh`, 11-06)
- [ ] Final leg table and quest table consolidated; every number traceable to a file in `work/`

## 2. Display pipeline: upsampler + colorizer (no metric uses it)
- [x] Upsampler 72x80 -> 144x160, 91% of held-out frames pixel-exact (`agent_notes/results/upsampler.md`, `lab/upsampler.py`)
- [ ] Tile clustering and contact sheet of the ~500 tiles that cover 98% of pixels
- [ ] (P) Tile classes labelled; palettes chosen (GBC-like vs bolder: Priyanshu's call)
- [ ] Stage 1 colorizer: tile lookup on real frames, per-map palette variation
- [ ] Stage 2 colorizer: small CNN gray -> color trained on stage-1 output, works on world-model frames; held-out accuracy reported
- [ ] `lab/render.py`: one entry point, gray frame -> upsampled color frame; used by the video and the live-run recorder

## 3. Research visuals for the site (static PNG + SVG from `work/`)
- [ ] Agent paths per leg on the Kanto tile map, real vs imagined
- [ ] Imagination drift vs horizon with CIs, per leg
- [ ] Probe-read heatmaps and the door-warp diagram (`work/eval/door_warp_wm31.json`)
- [ ] GPU-hour ledger chart (planned / scope / rework)
- [ ] Battle-fidelity figure (`work/plots/battle_fidelity_wm31_ext.png` restyled)

## 4. The 60-second gameplay video
- [ ] (P) Storyboard approved: route, which legs are split real/dream, battle shown as a colorized 2D inset
- [ ] Three 5-second voxel candidates of different areas rendered for the look decision (`lab/viz/`)
- [ ] Full render (1080p, 24 fps; ~15 h CPU or ~2 h GPU)
- [ ] Captions, timing overlay, badge ending; export `work/videos/v3_hero.mp4`

## 5. Repo: clone and run
- [ ] `lab/WEIGHTS.md`: every published weight file, size, what it is, which table uses it
- [ ] Download script that pulls weights from the HF repos into `work/` (anonymous, `weights_only=True`)
- [ ] One reproduce command per public table, tested from a fresh clone (ROM supplied by the user)
- [ ] `lab/README.md`: layout of `lab/`, how the quests, evals and training are run
- [ ] (P) Upload final weights to the existing HF repos (wm31, probe, upsampler, colorizer, 10 leg policies, PPO baseline)
- [ ] (P) Proposed public README text reviewed (`agent_notes/README_draft.md`)

## 6. Launch (after 1-5; needs Priyanshu throughout)
- [ ] Portfolio case study page
- [ ] GitHub description and topics
- [ ] X / LinkedIn post with the hero video
- [ ] Reddit post (r/MachineLearning or r/reinforcementlearning; honest framing: measurement study, one badge, 3 seeds)
- [ ] Website paper: the v3 results as a single long-form page, no venue submission yet
- [ ] Interview answers: why a world model, what it gained, where it fails (`agent_notes/v3_roadmap.md`)

## Open decisions (P)
- Path B (learned battles): shelved with the measured no-go; revisit only as a RAM-state battle simulator
- Colorizer palette direction
- Video storyboard and area
- The "final" configuration for all public numbers (proposed: the best leg set, `work/gym_learned_best_s*`)
