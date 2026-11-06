# Training throughput and GPU compute accounting (measured)

This page documents training throughput benchmarks for the 16.7M-parameter EDM diffusion world model and the CNN upsampler on an NVIDIA RTX 4090. It measures steps per second, sample throughput, and peak memory across batch sizes (64, 128, 192) and optimizer implementations (default foreach vs. fused AdamW). In addition, overall GPU compute accounting across the project is reported.

**Reproduce:** Benchmarked via `python -m lab.bench_speed`; see [lab/README.md §1](../README.md#1-directory--module-layout) (no single Section 3 command reproduces this benchmark).

**Figures:**
- `work/figures/fig_gpu_ledger.png`: Measured GPU compute accounting for the project: (top) 40.6 total hours broken down by component kind, dominated by imagination RL; (bottom) Path A stretch allocation across planned work (6.3 h), added scope (21.4 h), and debugging rework (13.5 h). This ledger is as of 2026-11-05; the current tally is in [`compute.md`](compute.md).

---

Raw: `work/bench_speed.log`. Script: `lab/bench_speed.py`, run at High priority while idle VRAM was allocated (~6 GB). RTX 4090, board limit 450 W, torch 2.5.1+cu121, bf16 autocast, 30 warmup + 200 timed steps per config. Speed only; no quality comparison.

## World model (16.7M-param EDM U-Net, 72x80, same step as `lab/dwm/train.py`)

| batch | AdamW | steps/s | samples/s | peak mem |
|---|---|---|---|---|
| 64 | default (foreach) | 11.38 | 729 | 5.6 GB |
| 64 | fused | **15.15** | **970** | 5.6 GB |
| 128 | default | 6.25 | 800 | 9.6 GB |
| 128 | fused | 7.35 | 941 | 9.6 GB |
| 192 | default | 4.30 | 826 | 13.7 GB |
| 192 | fused | 4.78 | 917 | 13.7 GB |

`cudnn.benchmark=True` changed every row by <1% (e.g. 11.24 / 15.19 steps/s at batch 64).

Reading: step time grows almost linearly with batch, so the GPU is saturated at batch 64; the spare VRAM is not a bottleneck. The fused optimizer removes ~22 ms of per-step overhead (+33% at batch 64). Recommendation for future runs: fused AdamW, batch 64 unless less gradient noise is wanted (then scale LR). The main and noaug runs used the default optimizer (kept for the matched ablation).

## Upsampler (batch 64, 144x160 output)

| data path | steps/s |
|---|---|
| current: numpy indexing on CPU, host->GPU copy, `loss.item()` every step | 11.55 |
| dataset on GPU, no per-step sync | 11.94 (+3%) |

`cudnn.benchmark` had no effect. GPU-bound; not worth changing the queued retrain.

## Hardware environment note
The RTX 4090 "tensor unlock" (FP32-accumulate rate, community Linux kernel module, Oct 2026) is Linux-only and needs a cold boot with firmware run before the driver loads. This machine runs Windows 11. The project's own BF16 GEMM gains are 0-20%, so this saturated small-conv workload would gain less (not measured).
