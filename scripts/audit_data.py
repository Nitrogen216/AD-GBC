"""Data / split / near-duplicate / preprocessing audit (A002 + A003).

Non-training M-1 gate. Verifies, per dataset:
  - image/mask counts and one-to-one correspondence
  - split coverage (every split id resolvable) and train/val disjointness
  - near-duplicate images (8x8 average-hash, Hamming distance) within the
    dataset and, critically, ACROSS the train/val boundary (leakage)
  - preprocessing numeric ranges through the actual SimpleSegTransform + /255 path
    under preprocess_mode={legacy,corrected} (legacy here == current single /255;
    no albumentations Normalize in this fork, so no double-normalization)
  - GlaS: whether the current random split leaks official Test A/B cases into train

Writes refine-logs/DATA_AUDIT.md and prints a summary. Read-only on the data.

Usage: ../../.venv/bin/python scripts/audit_data.py [busi glas ...]
"""
import os
import sys
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SEG = os.path.join(ROOT, "Dataset", "segmentation")
SPLITS = os.path.join(ROOT, "Dataset", "splits", "segmentation")
OUT = os.path.join(ROOT, "refine-logs", "DATA_AUDIT.md")

# per-dataset (img_ext, mask_ext)
EXT = {
    "busi": (".png", "_mask.png"),
    "glas": (".png", ".png"),
    "cvc": (".png", ".png"),
    "ISIC17": (".jpg", "_segmentation.png"),
    "busi_smoke": (".png", "_mask.png"),
}
NEAR_DUP_HAMMING = 5


def ahash(path):
    im = Image.open(path).convert("L").resize((8, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def read_split(ds, seed=41):
    out = {}
    for part in ("train", "val"):
        p = os.path.join(SPLITS, ds, f"seed_{seed}_{part}.txt")
        if os.path.exists(p):
            with open(p) as f:
                out[part] = [ln.strip() for ln in f if ln.strip()]
    return out


def audit_dataset(ds, lines):
    img_ext, mask_ext = EXT[ds]
    img_dir = os.path.join(SEG, ds, "images")
    mask_dir = os.path.join(SEG, ds, "masks", "0")
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(img_ext))
    ids = [f[: -len(img_ext)] for f in imgs]
    lines.append(f"## {ds}\n")
    lines.append(f"- images: {len(ids)}; img_ext=`{img_ext}` mask_ext=`{mask_ext}`")

    # correspondence
    missing_mask = [i for i in ids if not os.path.exists(os.path.join(mask_dir, i + mask_ext))]
    masks = [f for f in os.listdir(mask_dir) if f.endswith(mask_ext)]
    mask_ids = {f[: -len(mask_ext)] for f in masks}
    orphan_mask = sorted(mask_ids - set(ids))
    lines.append(f"- missing masks: {len(missing_mask)}; orphan masks: {len(orphan_mask)}")
    if missing_mask[:5]:
        lines.append(f"  - e.g. missing: {missing_mask[:5]}")

    # splits
    sp = read_split(ds)
    if sp:
        tr, va = set(sp.get("train", [])), set(sp.get("val", []))
        inter = tr & va
        unresolved = [i for i in (tr | va) if i not in set(ids)]
        lines.append(f"- split seed_41: train={len(tr)} val={len(va)} "
                     f"overlap={len(inter)} unresolved_ids={len(unresolved)} "
                     f"coverage={len(tr | va)}/{len(ids)}")
        if inter:
            lines.append(f"  - ⚠️ TRAIN/VAL OVERLAP: {sorted(inter)[:5]}")
        if unresolved:
            lines.append(f"  - ⚠️ unresolved ids (no image): {unresolved[:5]}")
    else:
        tr, va = set(), set()
        lines.append("- split seed_41: NONE FOUND")

    # GlaS official test leakage
    if ds == "glas" and (tr or va):
        off_test = [i for i in ids if i.startswith("testA") or i.startswith("testB")]
        leaked = [i for i in off_test if i in tr]
        lines.append(f"- GlaS official testA/testB cases on disk: {len(off_test)}; "
                     f"leaked into train (seed_41): {len(leaked)} "
                     f"→ Protocol-S needs official splits, not this random split")

    # near-duplicates
    hashes = {i: ahash(os.path.join(img_dir, i + img_ext)) for i in ids}
    items = list(hashes.items())
    dup_pairs, cross = [], []
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            if hamming(items[x][1], items[y][1]) <= NEAR_DUP_HAMMING:
                ia, ib = items[x][0], items[y][0]
                dup_pairs.append((ia, ib))
                if (ia in tr and ib in va) or (ia in va and ib in tr):
                    cross.append((ia, ib))
    n_pairs = len(items) * (len(items) - 1) // 2
    frac = len(dup_pairs) / n_pairs if n_pairs else 0.0
    lines.append(f"- near-duplicate pairs (ahash≤{NEAR_DUP_HAMMING}): {len(dup_pairs)} "
                 f"({frac:.1%} of all pairs); cross train/val: {len(cross)}")
    if frac > 0.01:
        lines.append("  - ⚠️ CAVEAT: 8x8 average-hash is too coarse for low-contrast "
                     "modalities (e.g. ultrasound) — a high flag rate here is likely "
                     "false-positive, NOT confirmed leakage. Treat as an UPPER BOUND; "
                     "re-check suspects with a stronger phash/SSIM/pixel comparison "
                     "before acting.")
    if cross[:5]:
        lines.append(f"  - suspects (verify, not confirmed): {cross[:3]}")
    lines.append("")
    return {"ids": len(ids), "missing_mask": len(missing_mask),
            "overlap": len(tr & va), "cross_dup": len(cross)}


def audit_preprocessing(ds, lines):
    """Range check through the real transform path, both preprocess modes."""
    sys.path.insert(0, os.path.join(ROOT, "repos", "AD-GBC"))
    from simple_transforms import SimpleSegTransform
    img_ext, _ = EXT[ds]
    img_dir = os.path.join(SEG, ds, "images")
    sample = sorted(f for f in os.listdir(img_dir) if f.endswith(img_ext))[:8]
    raw_stats, post_stats = [], []
    tf = SimpleSegTransform(256, 256, training=False)
    for f in sample:
        arr = np.asarray(Image.open(os.path.join(img_dir, f)).convert("RGB"))
        raw_stats.append((arr.min(), arr.max(), arr.mean()))
        mask = np.zeros((*arr.shape[:2], 1), dtype=np.uint8)
        aug = tf(image=arr, mask=mask)["image"].astype("float32") / 255.0  # corrected==legacy here
        post_stats.append((aug.min(), aug.max(), aug.mean(), aug.std()))
    rs = np.array(raw_stats); ps = np.array(post_stats)
    lines.append(f"### {ds} preprocessing (n={len(sample)})")
    lines.append(f"- raw uint8: min={rs[:,0].min():.0f} max={rs[:,1].max():.0f} mean≈{rs[:,2].mean():.1f}")
    lines.append(f"- after Resize+/255: min={ps[:,0].min():.3f} max={ps[:,1].max():.3f} "
                 f"mean≈{ps[:,2].mean():.3f} std≈{ps[:,3].mean():.3f}")
    lines.append("- note: this fork uses SimpleSegTransform (NO albumentations "
                 "Normalize) → single /255, no double-normalization. legacy==corrected here.")
    lines.append("")


def main():
    datasets = sys.argv[1:] or ["busi", "glas"]
    lines = ["# Data / Split / Preprocessing Audit (A002 + A003)", ""]
    lines.append(f"Datasets: {datasets}. Near-dup ahash threshold={NEAR_DUP_HAMMING}.\n")
    summary = {}
    for ds in datasets:
        summary[ds] = audit_dataset(ds, lines)
    lines.append("# Preprocessing ranges\n")
    for ds in datasets:
        audit_preprocessing(ds, lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT}")
    # split-manifest hash for reproducibility tracking
    h = hashlib.sha256()
    for ds in datasets:
        for part in ("train", "val"):
            p = os.path.join(SPLITS, ds, f"seed_41_{part}.txt")
            if os.path.exists(p):
                h.update(open(p, "rb").read())
    print(f"split-manifest sha256[:12] = {h.hexdigest()[:12]}")


if __name__ == "__main__":
    main()
