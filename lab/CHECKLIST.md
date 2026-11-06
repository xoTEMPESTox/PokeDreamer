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
- [x] Forest at 2x iterations, seeds 1-2 (`work/chain_forest_3560.sh`): s1 69 -> 49/100 (more runs stall at the Antidote ball), s2 97 -> 99/100. Not a reliable gain; the 1,780-iteration policies stay
- [ ] Final leg table and quest table consolidated; every number traceable to a file in `work/`

## 2. Display pipeline: upsampler + colorizer (no metric uses it)
- [x] Upsampler 72x80 -> 144x160, 91% of held-out frames pixel-exact (`agent_notes/results/upsampler.md`, `lab/upsampler.py`)
- [x] 4x4-block inventory at 72x80 (`lab/color/tiles.py`, uncommitted): works, but the blocks are too small to label; superseded by native tiles
- [x] Native tile ids from the emulator per (tileset, tile): screen rebuilt at 99.94% pixel agreement (`lab/color/native.py`, 7c45f49)
- [x] Stage 1 colorizer: exact per-tile colour on real frames; 107 tiles labelled by Claude (95% of non-battle pixels), rules for text, player, NPCs, battles (`lab/color/rules.py`, `labels.json`, `palette.json`); preview `work/color/demo/demo_v2.png`
- [ ] (P) Review the colours (`work/color/demo/demo_v2.png`); palette direction
- [~] Stage 2 colorizer: CNN predicting each pixel's class from the 72x80 frame, trained on stage-1 targets; held-out accuracy (agy `color_learned`)
- [ ] `lab/render.py`: one entry point, gray frame -> upsampled color frame; used by the video and the live-run recorder

## 3. Research visuals for the site (static PNG + SVG from `work/`)
- [x] Seven figures, code in `lab/figures/` (4b62d2a), output `work/figures/` + captions: leg paths, leg failures, leg success vs greedy, door-warp misread heatmaps, battle fidelity, GPU-hour ledger, presses per quest set
- [x] Imagination drift per leg map (`fig_drift_permap`, b0ef9bf; `results/v3_dwm_eval.md` "Drift per leg map")
- [ ] A Kanto overview map with all ten legs on one canvas (needs the colorizer for the backdrop)

## 4. The 60-second gameplay video
- [x] Source run recorded in colour: seed 100, best set, badge in 8,718 presses, every navigation press by a policy (fd8d7ea; `work/video/`)
- [x] 4x colour preview + scene list (`work/video/preview_s100_color.mp4`, `scenes.json`)
- [ ] (P) Storyboard approved (draft: `agent_notes/video_storyboard.md`)
- [ ] Three 5-second voxel candidates of different areas rendered for the look decision (`lab/viz/`)
- [ ] Full render (1080p, 24 fps; ~15 h CPU or ~2 h GPU)
- [ ] Captions, timing overlay, badge ending; export `work/videos/v3_hero.mp4`

## 5. Repo: clone and run
- [x] `lab/WEIGHTS.md`, `lab/get_weights.py`, `lab/README.md` (4221ef3)
- [ ] Export step that drops optimizer state before upload (wm31 268 -> ~67 MB, each policy 18.9 -> ~6.3 MB)
- [ ] Point `lab/README.md` and `lab/WEIGHTS.md` at `lab/results/` instead of the working notes
- [ ] Reproduce commands tested from a fresh clone (ROM supplied by the user)
- [ ] (P) Upload final weights to the existing HF repos (wm31, probe, upsampler, colorizer, 10 leg policies, PPO baseline)
- [ ] (P) Route save states: publish or not (emulator states of a session on the user's own ROM)

## 6. Research-repo structure (before any launch work)
The repo must read like a research artifact on its own, since the website and posts will link to it.
Today: v3 code is in `lab/` (tracked), but its results, run scripts and notes live in git-excluded
`agent_notes/` and `work/`; v1/v2 sit at the top level beside `src/`, `scripts/` and `checkpoints/`.

Doable now, within the current rules (new files only):
- [x] Tracked results `lab/results/` (12 pages + index, numbers checked against the notes; ea372a9)
- [x] Experiment scripts `lab/experiments/`, configs `lab/configs/`, `lab/requirements.txt`, CI skip markers + GitHub Actions (95 pass / 79 skip without ROM; ec8913f)
- [~] Figures: code tracked in `lab/figures/`; rendered PNGs stay in `work/figures/` until the layout decision (copy into the repo or attach to the website)
- [ ] `.gitattributes` forcing LF for `*.sh` (they are checked out with CRLF on Windows)
- [x] Proposed full layout written as a tree with reasons (`agent_notes/repo_layout_proposal.md`; needs sign-off)

Needs Priyanshu's sign-off (the project rules keep these off-limits to agents):
- [ ] (P) Top-level layout: e.g. `v1/` + `v2/` + `src/` + `scripts/` + `checkpoints/` moved under `legacy/`, `lab/` renamed to `pokedreamer/` (or kept), `results/` and `paper/` at the top
- [ ] (P) New top-level `README.md`: abstract, headline results table, hero video, figure, how to run, honesty notes, citation (draft in `agent_notes/README_draft.md`)
- [ ] (P) `LICENSE` (code) and a statement that no ROM or game assets are distributed
- [ ] (P) `CITATION.cff`
- [ ] (P) `v3.md` and `docs/v3.html` updated to the final numbers, or folded into the website paper

## 7. Launch (after 1-6; needs Priyanshu throughout)
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
