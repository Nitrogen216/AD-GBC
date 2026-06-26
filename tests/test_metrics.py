"""Unit tests for per-case metrics (A001).

Covers the advisor-mandated checklist (review 2/3):
  - pred == GT          -> Dice 1, IoU 1, HD95 0, ASSD 0, BF1 1
  - 1px translation     -> surface distance ~1px
  - batch-order invariant
  - batch-size invariant (batch of 1 == batch of N)
  - empty pred / nonempty GT -> overlap 0, surface NaN (or penalty), no error
  - both empty           -> Dice 1, HD95 0
  - resize/units (spacing) honoured

Run: ../../.venv/bin/python tests/test_metrics.py   (no pytest needed)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics_percase as M  # noqa: E402


def _square(h=32, w=32, top=8, left=8, size=12):
    m = np.zeros((h, w), dtype=bool)
    m[top:top + size, left:left + size] = True
    return m


def test_perfect_match():
    g = _square()
    r = M.evaluate_case(g, g)
    assert abs(r["dice"] - 1.0) < 1e-9
    assert abs(r["iou"] - 1.0) < 1e-9
    assert r["hd95"] == 0.0
    assert r["assd"] == 0.0
    assert abs(r["boundary_f1"] - 1.0) < 1e-9


def test_one_pixel_shift():
    g = _square()
    p = np.roll(g, 1, axis=1)  # shift right by 1px
    assert abs(M.case_hd95(p, g) - 1.0) < 1e-6, M.case_hd95(p, g)
    # ASSD <= 1 for a 1px rigid shift of a filled square
    assert M.case_assd(p, g) <= 1.0 + 1e-6
    assert M.case_dice(p, g) < 1.0


def test_batch_order_invariant():
    a, b, c = _square(left=4), _square(left=10), _square(left=16)
    out = np.stack([a, b, c])[:, None].astype(np.float32)
    gt = np.stack([a, b, c])[:, None].astype(np.float32)
    rows = M.evaluate_batch(out, gt, ids=["a", "b", "c"])
    perm = [2, 0, 1]
    rows_p = M.evaluate_batch(out[perm], gt[perm], ids=[["a", "b", "c"][i] for i in perm])
    by_id = {r["img_id"]: r for r in rows}
    by_id_p = {r["img_id"]: r for r in rows_p}
    for k in M.METRIC_CLUSTER:
        for i in ("a", "b", "c"):
            assert by_id[i][k] == by_id_p[i][k], (k, i)


def test_batch_size_invariant():
    imgs = [_square(left=4), _square(left=12), _square(left=18)]
    preds = [np.roll(m, 1, axis=0) for m in imgs]
    out = np.stack(preds)[:, None].astype(np.float32)
    gt = np.stack(imgs)[:, None].astype(np.float32)
    rows_full = M.evaluate_batch(out, gt, ids=[0, 1, 2])
    # same images evaluated one at a time
    rows_single = []
    for i in range(3):
        rows_single += M.evaluate_batch(out[i:i + 1], gt[i:i + 1], ids=[i])
    bid_f = {r["img_id"]: r for r in rows_full}
    bid_s = {r["img_id"]: r for r in rows_single}
    for k in M.METRIC_CLUSTER:
        for i in (0, 1, 2):
            assert bid_f[i][k] == bid_s[i][k], (k, i)


def test_empty_pred_nonempty_gt():
    g = _square()
    p = np.zeros_like(g)
    assert M.case_dice(p, g) == 0.0
    assert M.case_iou(p, g) == 0.0
    assert np.isnan(M.case_hd95(p, g))                       # default nan policy
    assert np.isnan(M.case_assd(p, g))
    # penalty policy returns a finite diagonal distance
    diag = M.case_hd95(p, g, hd_empty="diagonal")
    assert np.isfinite(diag) and diag > 0


def test_both_empty():
    z = np.zeros((32, 32), dtype=bool)
    r = M.evaluate_case(z, z)
    assert r["dice"] == 1.0 and r["iou"] == 1.0
    assert r["hd95"] == 0.0 and r["assd"] == 0.0
    assert r["boundary_f1"] == 1.0


def test_spacing_units():
    g = _square()
    p = np.roll(g, 1, axis=1)
    # doubling the column spacing doubles the horizontal surface distance
    d1 = M.case_hd95(p, g, spacing=(1.0, 1.0))
    d2 = M.case_hd95(p, g, spacing=(1.0, 2.0))
    assert abs(d2 - 2.0 * d1) < 1e-6, (d1, d2)


def test_aggregate_ignores_nan():
    g = _square()
    rows = [
        M.evaluate_case(g, g),                       # surface = 0
        M.evaluate_case(np.zeros_like(g), g),        # surface = nan
    ]
    summ = M.aggregate(rows)
    assert summ["hd95"]["n"] == 1            # nan excluded
    assert summ["dice"]["n"] == 2            # overlap defined for both
    assert abs(summ["dice"]["mean"] - 0.5) < 1e-9


def test_accepts_torch_tensors():
    try:
        import torch
    except ImportError:
        return
    g = _square()
    t_pred = torch.from_numpy(g.astype(np.float32))[None, None]
    t_gt = torch.from_numpy(g.astype(np.float32))[None, None]
    rows = M.evaluate_batch(t_pred, t_gt, ids=[0])
    assert abs(rows[0]["dice"] - 1.0) < 1e-9


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
