"""Split imagined position error into map hallucination vs drift on the correct map.

Usage: python -m lab.eval.decompose work/eval/main_test_id_s3_a0.npz [...]
"""
import sys

import numpy as np

from lab.eval.metrics import bootstrap_ci

KS = (1, 2, 4, 8, 16, 32)


def decompose(npz_path):
    d = np.load(npz_path)
    valid = ~np.isnan(d["pos_err_img"])                 # true frame overworld, true map unchanged
    right_map = valid & (d["img_map"] == d["true_map"])
    err_right = np.where(right_map, d["pos_err_img"], np.nan)
    nochange_right = np.where(right_map, d["pos_err_nochange"], np.nan)
    wrong_map = np.where(valid, (d["img_map"] != d["true_map"]).astype(float), np.nan)
    eps = d["episode"]
    out = {}
    for name, arr in [("wrong_map_rate", wrong_map), ("pos_err_img_right_map", err_right),
                      ("pos_err_nochange_same_windows", nochange_right)]:
        per_ep = np.stack([np.nanmean(arr[eps == e], 0) for e in np.unique(eps)])
        out[name] = bootstrap_ci(per_ep)
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\n{path}")
        res = decompose(path)
        print(f"{'metric':32s}" + "".join(f"{'k=' + str(k):>20s}" for k in KS))
        for name, (m, lo, hi) in res.items():
            print(f"{name:32s}" + "".join(f"{m[k - 1]:7.3f} [{lo[k - 1]:5.2f},{hi[k - 1]:5.2f}]" for k in KS))
