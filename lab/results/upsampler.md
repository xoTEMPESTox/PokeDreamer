# Native-resolution upsampler (72x80 -> 144x160, 4 shades)

This page evaluates the CNN upsampler (`lab/upsampler.py`), which restores 72x80 greyscale frames back to native 144x160 4-shade Game Boy frames for screenshot and video display. On held-out in-distribution test episodes (`test_id`), the upsampler achieves 99.980% pixel accuracy, with 91.4% of frames rendered pixel-identical to the native Game Boy output. The module functions purely as a display post-processor and is not used by any evaluation metric in the world model evaluations.

**Reproduce:** No single Section 3 command reproduces this; trained via `python -m lab.upsampler`; verified in test suite via `pytest lab/tests/test_upsampler.py`; see [lab/README.md §1](../README.md#1-directory--module-layout) and [lab/README.md §4.2](../README.md#42-run-specific-test-suites).

---

Display-only post-processing for screenshots and videos. **No metric in `v3_dwm_eval.md` uses it.** Code: `lab/upsampler.py`.

An earlier run (Run 1: 25 min, 16,515 steps x batch 64, AdamW lr 1e-3, no clipping; archived as `work/upsampler/upsampler_run1.pt`, `metrics_run1.json`, `train_run1.log`) was replaced because the loss spiked twice without gradient clipping (0.023 -> 1.50 at step ~4.6k; 0.003 -> 1.24 at ~14k); Run 2 added gradient clipping 1.0, cosine decay, and lr 3e-4, resolving the instability.

## Final model (measured, current: `work/upsampler/upsampler.pt`)
Same model and data; lr 3e-4 + grad clip 1.0 + cosine decay; 25 min. Loss fell smoothly to 1e-4 with no spikes (`work/upsampler/train.log`). Metrics `work/upsampler/metrics.json` (2000 random frames per split):

| split | pixel accuracy | frames 100% exact |
|---|---|---|
| train | 99.999% | 98.6% |
| test_id | 99.980% | 91.4% |
| test_ood | 99.481% | 11.3% |

On held-out in-distribution episodes, 91.4% of upsampled real frames are pixel-identical to the native screen. OOD maps (Oak's Lab, Pallet; rare in training) still have small errors in most frames. Showcase re-rendered with it (`work/videos/*_up.mp4`, `work/videos/play_*.mp4`, `work/screenshots/shot_*`). Caveat: the upsampler makes imagined frames look like real Game Boy screens; it does not make their content correct (e.g. step 24 of `play_walk_square` shows imagination drift, rendered crisply).
