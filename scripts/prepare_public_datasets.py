#!/usr/bin/env python3
"""Prepare public AD-GBC benchmark datasets into Dataset/segmentation/<dataset>.

Expected raw files are stored under Dataset/raw/:
  - Dataset_BUSI.zip
  - warwick_qu_dataset_released_2016_07_08.zip
  - ISIC-2017_Training_Data.zip
  - ISIC-2017_Training_Part1_GroundTruth.zip

The output layout is method-agnostic and can be reused by other algorithms:
  Dataset/segmentation/<dataset>/images
  Dataset/segmentation/<dataset>/masks/0
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import tarfile
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]


def find_workspace_root() -> Path:
    for candidate in [CODE_ROOT, *CODE_ROOT.parents]:
        if (candidate / "Dataset").exists() or (candidate / ".aris").exists():
            return candidate
    return CODE_ROOT.parent.parent


WORKSPACE_ROOT = find_workspace_root()
DATASET_ROOT = WORKSPACE_ROOT / "Dataset"
RAW = DATASET_ROOT / "raw"
EXTRACTED = RAW / "extracted"
INPUTS = DATASET_ROOT / "segmentation"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def clean_output(dataset: str) -> tuple[Path, Path]:
    out_root = INPUTS / dataset
    out_images = out_root / "images"
    out_masks = out_root / "masks" / "0"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    return out_images, out_masks


def safe_extract(zip_path: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        log(f"reuse extracted directory: {rel(dest)}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    dest.mkdir(parents=True, exist_ok=True)
    log(f"extract {zip_path.name} -> {rel(dest)}")
    try:
        with ZipFile(zip_path) as archive:
            dest_resolved = dest.resolve()
            for member in archive.infolist():
                target = (dest / member.filename).resolve()
                if dest_resolved not in target.parents and target != dest_resolved:
                    raise ValueError(f"unsafe zip member: {member.filename}")
            archive.extractall(dest)
    except BadZipFile as exc:
        raise RuntimeError(f"{zip_path} is not a complete zip file") from exc


def safe_extract_tar(tar_path: Path, dest: Path) -> None:
    img_dir = dest / "ds" / "img"
    ann_dir = dest / "ds" / "ann"
    if img_dir.exists() and ann_dir.exists():
        if len(list(img_dir.glob("*.png"))) == len(list(ann_dir.glob("*.json"))) >= 600:
            log(f"reuse extracted directory: {rel(dest)}")
            return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    log(f"extract {tar_path.name} -> {rel(dest)}")
    with tarfile.open(tar_path) as archive:
        dest_resolved = dest.resolve()
        for member in archive.getmembers():
            target = (dest / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise ValueError(f"unsafe tar member: {member.name}")
        archive.extractall(dest)


def first_dir(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"cannot find directory {name!r} under {root}")
    return matches[0]


def save_binary_mask(mask_paths: list[Path], output_path: Path) -> None:
    combined: np.ndarray | None = None
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        combined = mask if combined is None else np.logical_or(combined, mask)
    if combined is None:
        raise ValueError(f"no mask files for {output_path.name}")
    Image.fromarray((combined.astype(np.uint8) * 255)).save(output_path)


def prepare_busi() -> None:
    safe_extract(RAW / "Dataset_BUSI.zip", EXTRACTED / "busi")
    source = first_dir(EXTRACTED / "busi", "Dataset_BUSI_with_GT")
    out_images, out_masks = clean_output("busi")

    count = 0
    skipped = 0
    for class_name in ("benign", "malignant"):
        class_dir = source / class_name
        for image_path in sorted(class_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            mask_paths = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
            if not mask_paths:
                skipped += 1
                continue
            Image.open(image_path).convert("RGB").save(out_images / image_path.name)
            save_binary_mask(mask_paths, out_masks / f"{image_path.stem}_mask.png")
            count += 1

    log(f"BUSI prepared: {count} tumor images, {skipped} skipped")


def prepare_glas() -> None:
    safe_extract(
        RAW / "warwick_qu_dataset_released_2016_07_08.zip",
        EXTRACTED / "glas",
    )
    source_candidates = [
        p
        for p in (EXTRACTED / "glas").rglob("*")
        if p.is_dir() and "Warwick QU Dataset" in p.name
    ]
    if not source_candidates:
        raise FileNotFoundError("cannot find Warwick QU Dataset directory")
    source = source_candidates[0]
    out_images, out_masks = clean_output("glas")

    count = 0
    for image_path in sorted(source.glob("*.bmp")):
        if image_path.stem.endswith("_anno"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_anno.bmp")
        if not mask_path.exists():
            continue
        Image.open(image_path).convert("RGB").save(out_images / f"{image_path.stem}.png")
        save_binary_mask([mask_path], out_masks / f"{image_path.stem}.png")
        count += 1

    log(f"GlaS prepared: {count} images")


def prepare_isic17() -> None:
    safe_extract(RAW / "ISIC-2017_Training_Data.zip", EXTRACTED / "isic17_images")
    safe_extract(
        RAW / "ISIC-2017_Training_Part1_GroundTruth.zip",
        EXTRACTED / "isic17_masks",
    )
    image_map = {
        image_path.stem: image_path
        for image_path in (EXTRACTED / "isic17_images").rglob("*.jpg")
    }
    mask_paths = sorted((EXTRACTED / "isic17_masks").rglob("*_segmentation.png"))
    out_images, out_masks = clean_output("ISIC17")

    count = 0
    missing = 0
    for mask_path in mask_paths:
        image_id = mask_path.stem.replace("_segmentation", "")
        image_path = image_map.get(image_id)
        if image_path is None:
            missing += 1
            continue
        shutil.copy2(image_path, out_images / f"{image_id}.jpg")
        save_binary_mask([mask_path], out_masks / f"{image_id}_segmentation.png")
        count += 1

    log(f"ISIC17 prepared: {count} image/mask pairs, {missing} masks without image")


def locate_cvc_pair(root: Path) -> tuple[Path, Path] | None:
    image_names = {"original", "images", "image", "img"}
    mask_names = {"ground truth", "ground_truth", "masks", "mask", "gt"}
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    for image_dir in dirs:
        if image_dir.name.strip().lower() not in image_names:
            continue
        siblings = list(image_dir.parent.iterdir())
        for mask_dir in siblings:
            if mask_dir.is_dir() and mask_dir.name.strip().lower() in mask_names:
                return image_dir, mask_dir
    return None


def prepare_cvc_supervisely(source: Path) -> None:
    img_dir = source / "ds" / "img"
    ann_dir = source / "ds" / "ann"
    if not img_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"missing Supervisely ds/img or ds/ann under {source}")

    out_images, out_masks = clean_output("cvc")
    count = 0
    for image_path in sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem)):
        ann_path = ann_dir / f"{image_path.name}.json"
        if not ann_path.exists():
            continue
        ann = json.loads(ann_path.read_text())
        height = int(ann["size"]["height"])
        width = int(ann["size"]["width"])
        canvas = np.zeros((height, width), dtype=bool)
        for obj in ann.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue
            bitmap = obj.get("bitmap", {})
            origin_x, origin_y = bitmap["origin"]
            mask_bytes = zlib.decompress(base64.b64decode(bitmap["data"]))
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L")) > 0
            h, w = mask.shape
            canvas[origin_y : origin_y + h, origin_x : origin_x + w] |= mask

        Image.open(image_path).convert("RGB").save(out_images / image_path.name)
        Image.fromarray((canvas.astype(np.uint8) * 255)).save(
            out_masks / image_path.name
        )
        count += 1

    log(f"CVC prepared from Supervisely tar: {count} images")


def prepare_cvc(source: Path | None = None) -> None:
    cvc_tar = RAW / "cvc-clinicdb-DatasetNinja.tar"
    if source is None and cvc_tar.exists():
        extracted = EXTRACTED / "cvc"
        safe_extract_tar(cvc_tar, extracted)
        prepare_cvc_supervisely(extracted)
        return

    roots = [source] if source else [RAW / "cvc", EXTRACTED / "cvc", RAW]
    pair = None
    for root in roots:
        if root and root.exists():
            pair = locate_cvc_pair(root)
            if pair:
                break
    if not pair:
        raise FileNotFoundError(
            "cannot locate CVC Original/Ground Truth folders; pass --cvc-source"
        )

    image_dir, mask_dir = pair
    out_images, out_masks = clean_output("cvc")
    mask_map = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    }

    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        mask_path = mask_map.get(image_path.stem)
        if mask_path is None:
            continue
        Image.open(image_path).convert("RGB").save(out_images / f"{image_path.stem}.png")
        save_binary_mask([mask_path], out_masks / f"{image_path.stem}.png")
        count += 1

    log(f"CVC prepared: {count} images from {image_dir.parent}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["busi", "glas", "isic17", "cvc"],
        choices=["busi", "glas", "isic17", "cvc"],
    )
    parser.add_argument("--cvc-source", type=Path, default=None)
    args = parser.parse_args()

    for dataset in args.datasets:
        if dataset == "busi":
            prepare_busi()
        elif dataset == "glas":
            prepare_glas()
        elif dataset == "isic17":
            prepare_isic17()
        elif dataset == "cvc":
            prepare_cvc(args.cvc_source)


if __name__ == "__main__":
    main()
