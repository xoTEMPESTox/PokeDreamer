# Frame->RAM probe (measured)

This page measures the accuracy of the Frame-to-RAM probe, a convolutional network instrument used to extract map ID, player tile coordinates, and battle status from single 72x80 greyscale frames. It establishes the baseline measurement error across training frames, held-out in-distribution test episodes (test_id, 16 episodes), and out-of-distribution Pallet-start episodes (test_ood, 45 episodes). On test_id, the probe achieves 99.36% map accuracy and 0.397 tiles mean Manhattan position error, ensuring that downstream evaluation errors on in-distribution episodes reflect world model dynamics rather than instrument distortion.

**Reproduce:** No single Section 3 command reproduces this; trained via `python -m lab.probe`; see [lab/README.md §1](../README.md#1-directory--module-layout) and [lab/README.md §2.4](../README.md#24-model-weights--save-states).

---

Instrument used to read map / player tile / battle off real and imagined 72x80 frames.
Code: `lab/probe.py`. Raw: `work/probe/probe_metrics.json`, log `work/probe/train.log`, weights `work/probe/probe.pt`.
Trained only on train-split frames (139 episodes), CPU 8 threads, 5,800 steps x batch 256 (638 s), Adam 1e-3. Map/x/y losses on non-battle frames only.

| split | map acc | majority-class map acc | x exact | y exact | mean Manhattan error (tiles) | battle acc |
|---|---|---|---|---|---|---|
| train (5k sample) | 99.53% | 61.17% | 98.61% | 98.66% | 0.217 | 99.64% |
| test_id (16 eps) | 99.36% | 43.88% | 96.68% | 97.44% | 0.397 | 99.64% |
| test_ood (45 Pallet-start eps) | 88.34% | 0.00% | 87.25% | 55.93% | 4.803 | 99.33% |

Map/x/y metrics on non-battle frames; battle accuracy on all frames.

Reading: on in-distribution episodes the probe reads position almost exactly, so imagined-frame errors are attributable to the world model. On the OOD split the probe itself is unreliable (Pallet Town and Oak's Lab are 3% and 9% of training frames), so OOD drift numbers carry up to ~4.8 tiles of instrument error. Use `pos_agree` (imagined vs real-frame probe reading) and the probe-ceiling curve when interpreting OOD.
