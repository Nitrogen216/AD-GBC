"""Per-case segmentation metrics (A001, advisor review 2/3).

Replaces the broken batch-level path in ``metrics.py`` where ``[B,C,H,W]`` arrays
were passed whole to medpy ``hd``/``hd95`` (the batch axis was treated as a
spatial dimension, so images bled into each other's surface distances and HD/HD95
depended on batch order and batch size).

Every metric here operates on a **single** 2D binary mask pair ``(H, W)``.
``evaluate_batch`` loops over the batch/channel and returns one row per case, so
results are invariant to batch order and batch size by construction. Aggregation
(macro-mean, median, IQR) is a separate explicit step.

Dependencies: numpy + scipy only (no medpy, no torch required). Torch tensors are
accepted and moved to cpu numpy, so this is backend-agnostic (cpu/mps/cuda).

Empty-case policy (frozen):
  - both empty (GT empty & pred empty): perfect overlap -> Dice/IoU/BF1 = 1,
    HD95/ASSD = 0.
  - exactly one empty (miss or false-positive): overlap metrics = 0; surface
    metrics (HD95/ASSD/BF1) are undefined -> returned as NaN by default
    (``hd_empty='nan'``) so they can be excluded from surface aggregation, or set
    to a penalty distance with ``hd_empty='diagonal'`` (image diagonal in px).
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

EPS = 1e-7


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #
def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor (any device)
        x = x.detach().to("cpu").numpy()
    return np.asarray(x)


def _as_bool_2d(mask, threshold=0.5):
    """Coerce a single mask to a boolean (H, W) array."""
    m = _to_numpy(mask)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"expected a single 2D mask, got shape {np.shape(mask)}")
    if m.dtype == bool:
        return m
    return m > threshold


# --------------------------------------------------------------------------- #
# overlap metrics (single case)
# --------------------------------------------------------------------------- #
def case_dice(pred, gt, threshold=0.5):
    p, g = _as_bool_2d(pred, threshold), _as_bool_2d(gt, threshold)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # both empty
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def case_iou(pred, gt, threshold=0.5):
    p, g = _as_bool_2d(pred, threshold), _as_bool_2d(gt, threshold)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0  # both empty
    return float(np.logical_and(p, g).sum() / union)


# --------------------------------------------------------------------------- #
# surface metrics (single case)
# --------------------------------------------------------------------------- #
def _boundary(mask):
    """Inner boundary pixels of a binary mask."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask))


def _directed_surface_distances(pred, gt, spacing):
    """Distances from every pred-boundary pixel to the nearest gt-boundary pixel."""
    gt_b = _boundary(gt)
    # distance to nearest gt-boundary pixel everywhere
    dt = distance_transform_edt(~gt_b, sampling=spacing)
    return dt[_boundary(pred)]


def _both_nonempty(p, g):
    return p.any() and g.any()


def _empty_surface_value(p, g, hd_empty, spacing, shape):
    """Surface-metric value when at least one mask is empty."""
    if not p.any() and not g.any():
        return 0.0  # both empty -> perfect
    if hd_empty == "nan":
        return float("nan")
    if hd_empty == "diagonal":
        h, w = shape
        return float(np.hypot(h * spacing[0], w * spacing[1]))
    raise ValueError(f"unknown hd_empty policy: {hd_empty}")


def case_hd95(pred, gt, threshold=0.5, spacing=(1.0, 1.0), hd_empty="nan"):
    p, g = _as_bool_2d(pred, threshold), _as_bool_2d(gt, threshold)
    if not _both_nonempty(p, g):
        return _empty_surface_value(p, g, hd_empty, spacing, p.shape)
    fwd = _directed_surface_distances(p, g, spacing)
    bwd = _directed_surface_distances(g, p, spacing)
    return float(np.percentile(np.hstack([fwd, bwd]), 95))


def case_assd(pred, gt, threshold=0.5, spacing=(1.0, 1.0), hd_empty="nan"):
    """Average symmetric surface distance."""
    p, g = _as_bool_2d(pred, threshold), _as_bool_2d(gt, threshold)
    if not _both_nonempty(p, g):
        return _empty_surface_value(p, g, hd_empty, spacing, p.shape)
    fwd = _directed_surface_distances(p, g, spacing)
    bwd = _directed_surface_distances(g, p, spacing)
    return float(np.hstack([fwd, bwd]).mean())


def case_boundary_f1(pred, gt, threshold=0.5, tolerance=2.0, spacing=(1.0, 1.0),
                     empty_value="nan"):
    """Boundary-F1 @ tolerance (pixels): pred/gt boundary agreement within tol."""
    p, g = _as_bool_2d(pred, threshold), _as_bool_2d(gt, threshold)
    if not p.any() and not g.any():
        return 1.0
    if not _both_nonempty(p, g):
        if empty_value == "nan":
            return float("nan")
        return 0.0
    pb, gb = _boundary(p), _boundary(g)
    dt_to_g = distance_transform_edt(~gb, sampling=spacing)
    dt_to_p = distance_transform_edt(~pb, sampling=spacing)
    precision = (dt_to_g[pb] <= tolerance).mean() if pb.any() else 0.0
    recall = (dt_to_p[gb] <= tolerance).mean() if gb.any() else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# --------------------------------------------------------------------------- #
# batch driver + aggregation
# --------------------------------------------------------------------------- #
METRIC_CLUSTER = ("dice", "iou", "hd95", "assd", "boundary_f1")


def evaluate_case(pred, gt, threshold=0.5, spacing=(1.0, 1.0), tolerance=2.0,
                  hd_empty="nan"):
    """All cluster metrics for one (H,W) case -> dict."""
    return {
        "dice": case_dice(pred, gt, threshold),
        "iou": case_iou(pred, gt, threshold),
        "hd95": case_hd95(pred, gt, threshold, spacing, hd_empty),
        "assd": case_assd(pred, gt, threshold, spacing, hd_empty),
        "boundary_f1": case_boundary_f1(pred, gt, threshold, tolerance, spacing,
                                        hd_empty),
    }


def evaluate_batch(output, target, ids=None, threshold=0.5, spacing=(1.0, 1.0),
                   tolerance=2.0, hd_empty="nan", apply_sigmoid=False):
    """Per-case rows for a batch.

    output/target: [B,C,H,W] or [B,H,W] (torch or numpy). If ``output`` is the
    training tuple ``(seg, intermediates)`` the seg tensor is used. Set
    ``apply_sigmoid=True`` if ``output`` is raw logits.

    Returns a list of dicts, one per (image, channel), each with the cluster
    metrics plus ``img_id`` and ``channel``. Order-/batch-size-invariant.
    """
    if isinstance(output, tuple):
        output = output[0]
    out = _to_numpy(output.sigmoid() if (apply_sigmoid and hasattr(output, "sigmoid"))
                    else output)
    if apply_sigmoid and not hasattr(output, "sigmoid"):
        out = 1.0 / (1.0 + np.exp(-out))
    tgt = _to_numpy(target)
    if out.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        out = out[:, None]
        tgt = tgt[:, None]
    B, C = out.shape[0], out.shape[1]
    rows = []
    for b in range(B):
        img_id = ids[b] if ids is not None else b
        for c in range(C):
            row = evaluate_case(out[b, c], tgt[b, c], threshold, spacing,
                                tolerance, hd_empty)
            row["img_id"] = img_id
            row["channel"] = c
            rows.append(row)
    return rows


def aggregate(rows, keys=METRIC_CLUSTER):
    """Macro mean / median / IQR per metric, ignoring NaN (e.g. surface metrics
    on empty cases). Returns {metric: {mean, median, iqr, n}}."""
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[k] = {"mean": float("nan"), "median": float("nan"),
                          "iqr": float("nan"), "n": 0}
            continue
        q75, q25 = np.percentile(vals, [75, 25])
        summary[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                      "iqr": float(q75 - q25), "n": int(vals.size)}
    return summary


def rows_to_csv(rows, path, keys=METRIC_CLUSTER):
    """Write per-case rows to CSV (img_id, channel, <metrics>)."""
    import csv
    fields = ["img_id", "channel", *keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
