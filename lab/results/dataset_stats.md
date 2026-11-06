# Dataset statistics (measured)

This page reports dataset distribution and transition statistics measured across the base 72x80 greyscale frame cache. The dataset comprises 200000 steps partitioned across 200 episodes into train (139 episodes), in-distribution test (16 episodes), and out-of-distribution Pallet-start test (45 episodes) splits. It documents movement frequencies, map changes, battle occurrences, action frequencies, and per-map step distributions across each split.

**Reproduce:** Built via `python -m lab.data` from the base dataset transitions; see [lab/README.md §2.3](../README.md#23-dataset-cache-build).

---

Source: `work/cache_72x80.npz` built by `lab/data.py` from the HF dataset; splits in `work/splits.json`.

- Steps: 200000; episodes: 200 (all 1000 steps)
- Start maps: {38: 81, 0: 45, 40: 74}
- Split (episodes): train 139, test_id 16, test_ood (start map 0) 45
- Steps where (x, y) changes: 0.372 (also counts map transitions)
- Steps where map ID changes: 0.0024
- Steps in battle: 0.122
- `dialog_opens` is True on 95% of steps, so it is not a usable dialog flag (not used).
- Action counts: {0: 43557, 1: 36518, 2: 43690, 3: 33713, 4: 35621, 5: 5334, 6: 819, 7: 748}

| map ID | steps | fraction |
|---|---|---|
| 38 | 79548 | 0.398 |
| 40 | 35939 | 0.180 |
| 12 | 34489 | 0.172 |
| 0 | 25796 | 0.129 |
| 41 | 16491 | 0.082 |
| 1 | 3663 | 0.018 |
| 39 | 1563 | 0.008 |
| 37 | 1525 | 0.008 |
| 13 | 978 | 0.005 |
| 42 | 8 | 0.000 |

train map distribution: 38: 0.531, 12: 0.215, 41: 0.092, 40: 0.089, 0: 0.029, 1: 0.024, 37: 0.008, 13: 0.007, 39: 0.005, 42: 0.000

test_id map distribution: 38: 0.362, 12: 0.258, 41: 0.235, 40: 0.075, 0: 0.035, 1: 0.020, 37: 0.013, 39: 0.002

test_ood map distribution: 40: 0.498, 0: 0.470, 39: 0.018, 12: 0.010, 37: 0.004
