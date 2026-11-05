"""Tests for battle fidelity evaluation metrics (toy arrays, pure numpy/python)."""
import os
import sys

# Prevent OpenMP runtime collision between PyTorch, MKL, and OpenCV
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

from lab.dwm.battle_fidelity import (
    battle_end_step,
    classify_battle_end_pair,
    classify_battle_ends,
)


def test_battle_end_step():
    """Verify first future step index where battle is over (~in_battle)."""
    # 1. Never ends (all True)
    arr_all_true = np.ones(10, dtype=bool)
    assert battle_end_step(arr_all_true) is None

    # 2. Ends at step 0 (immediately over)
    arr_step0 = np.array([False, False, True, False])
    assert battle_end_step(arr_step0) == 0

    # 3. Ends at step 5
    arr_step5 = np.array([True, True, True, True, True, False, False, False])
    assert battle_end_step(arr_step5) == 5

    # 4. Ends at last step
    arr_last = np.array([True, True, True, False])
    assert battle_end_step(arr_last) == 3

    # 5. Empty array
    assert battle_end_step(np.empty(0, dtype=bool)) is None


def test_classify_battle_end_pair_cases_and_tolerance():
    """Verify all four cases and exact +-3 tolerance boundaries."""
    tol = 3
    r = 10

    # Case 1: real ends within H and imagined ends within +-3 steps
    assert classify_battle_end_pair(r, 10, tol) == "within_pm3"        # diff = 0
    assert classify_battle_end_pair(r, 13, tol) == "within_pm3"        # diff = +3 (upper bound)
    assert classify_battle_end_pair(r, 7, tol) == "within_pm3"         # diff = -3 (lower bound)
    assert classify_battle_end_pair(0, 3, tol) == "within_pm3"         # diff = +3
    assert classify_battle_end_pair(3, 0, tol) == "within_pm3"         # diff = -3

    # Case 2: hallucinated end (imagined earlier by >3 steps, or real never ended)
    assert classify_battle_end_pair(r, 6, tol) == "hallucinated_end"   # diff = -4 (just outside lower bound)
    assert classify_battle_end_pair(r, 0, tol) == "hallucinated_end"   # diff = -10
    assert classify_battle_end_pair(None, 5, tol) == "hallucinated_end"  # real never ended, imagined ended
    assert classify_battle_end_pair(None, 0, tol) == "hallucinated_end"
    assert classify_battle_end_pair(None, 31, tol) == "hallucinated_end"

    # Case 3: imagined ends later (>3 steps) or never
    assert classify_battle_end_pair(r, 14, tol) == "later_or_never"    # diff = +4 (just outside upper bound)
    assert classify_battle_end_pair(r, 25, tol) == "later_or_never"    # diff = +15
    assert classify_battle_end_pair(r, None, tol) == "later_or_never"  # real ended, imagined never ended
    assert classify_battle_end_pair(0, None, tol) == "later_or_never"

    # Case 4: neither ends within H
    assert classify_battle_end_pair(None, None, tol) == "neither"


def test_classify_battle_ends_batch():
    """Verify batch classification metrics, counts, fractions, and median absolute error."""
    # 8 toy windows spanning all 4 cases:
    # 3 within_pm3 (diffs: 0, +3, -3)
    # 2 hallucinated_end (diff -4, and real None with pred 5)
    # 2 later_or_never (diff +4, and real 10 with pred None)
    # 1 neither (real None, pred None)
    ram_ends = [10, 10, 10, 10, None, 10, 10, None]
    pred_ends = [10, 13, 7, 6, 5, 14, None, None]

    res = classify_battle_ends(ram_ends, pred_ends, tol=3)

    assert res["counts"]["within_pm3"] == 3
    assert res["counts"]["hallucinated_end"] == 2
    assert res["counts"]["later_or_never"] == 2
    assert res["counts"]["neither"] == 1

    assert sum(res["counts"].values()) == 8
    assert np.isclose(sum(res["fractions"].values()), 1.0)
    assert np.isclose(res["fractions"]["within_pm3"], 3 / 8)
    assert np.isclose(res["fractions"]["hallucinated_end"], 2 / 8)
    assert np.isclose(res["fractions"]["later_or_never"], 2 / 8)
    assert np.isclose(res["fractions"]["neither"], 1 / 8)

    # Errors where both end:
    # idx 0: |10-10| = 0
    # idx 1: |13-10| = 3
    # idx 2: |7-10| = 3
    # idx 3: |6-10| = 4
    # idx 5: |14-10| = 4
    # idx 4 has r=None, idx 6 has p=None, idx 7 has both None
    assert res["both_end_count"] == 5
    assert np.isclose(res["both_end_fraction"], 5 / 8)
    assert res["errors"] == [0, 3, 3, 4, 4]
    # Median of [0, 3, 3, 4, 4] is 3.0
    assert res["median_abs_error"] == 3.0


def test_classify_battle_ends_edge_cases():
    """Edge cases: empty inputs and no overlapping battle ends."""
    # Empty
    res_empty = classify_battle_ends([], [])
    assert res_empty["counts"] == {"within_pm3": 0, "hallucinated_end": 0, "later_or_never": 0, "neither": 0}
    assert res_empty["median_abs_error"] is None

    # No overlapping ends
    res_no_overlap = classify_battle_ends([None, None], [None, None])
    assert res_no_overlap["counts"]["neither"] == 2
    assert res_no_overlap["median_abs_error"] is None
