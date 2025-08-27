#!/usr/bin/env python3
"""
Convert YOLO annotations to a TorchVision-friendly JSON
(no image copying; paths stored relative to --image_dir).

Works great for Faster R-CNN / RetinaNet / DETR loaders.

Example:
    python3 utils/data_format_conversions/yolo_to_torchvision.py \
        --yolo_dir path/to/yolo_labels \
        --image_dir path/to/images \
        --class_list path/to/classes.txt \
        --output_dir path/to/output \
        --allow_empty_images \
        --recursive
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_classes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise ValueError(f"No classes found in {path}")
    return names

def list_label_files(yolo_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in yolo_dir.rglob("*.txt") if p.is_file())
    return sorted(p for p in yolo_dir.iterdir() if p.is_file() and p.suffix == ".txt")

def find_image(image_dir: Path, stem: str, recursive: bool) -> Optional[Path]:
    if recursive:
        cands = [p for p in image_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem]
    else:
        cands = [p for p in image_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem]
    if not cands:
        return None
    # Prefer jpg/jpeg if multiple exist
    cands.sort(key=lambda p: (p.suffix.lower() not in {".jpg", ".jpeg"}, p.suffix))
    return cands[0]

def yolo_line_to_xyxy(parts: List[float], W: int, H: int) -> Tuple[List[float], int]:
    """
    YOLO: class x_center y_center w h  (all normalized)
    Returns: [x1,y1,x2,y2], class_id
    """
    cls, xc, yc, w, h = parts
    xc, yc, w, h = xc*W, yc*H, w*W, h*H
    x1 = xc - w/2.0
    y1 = yc - h/2.0
    x2 = xc + w/2.0
    y2 = yc + h/2.0
    return [x1, y1, x2, y2], int(cls)

def clip_xyxy(b: List[float], W: int, H: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(x1, W))
    y1 = max(0.0, min(y1, H))
    x2 = max(0.0, min(x2, W))
    y2 = max(0.0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def convert_yolo_to_torchvision(
    yolo_dir: str,
    image_dir: str,
    class_list_path: str,
    output_dir: str,
    allow_empty_images: bool = False,
    recursive: bool = False,
) -> None:
    yolo_dir = Path(yolo_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(class_list_path)
    labels = list_label_files(yolo_dir, recursive)
    if not labels:
        raise FileNotFoundError(f"No .txt files found under {yolo_dir}")

    records = []
    miss_img, malformed, out_of_range = 0, 0, 0

    for lbl in labels:
        stem = lbl.stem
        img_path = find_image(image_dir, stem, recursive)
        if img_path is None:
            print(f"[warn] no image for {lbl.name}; skipping")
            miss_img += 1
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        rel = str(img_path.relative_to(image_dir)) if recursive else img_path.name
        boxes_xyxy, y_labels = [], []

        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    vals = list(map(float, parts))
                except Exception:
                    malformed += 1
                    continue

                xyxy, cls_id = yolo_line_to_xyxy(vals, W, H)
                if not (0 <= cls_id < len(class_names)):
                    out_of_range += 1
                    continue
                xyxy = clip_xyxy(xyxy, W, H)
                if xyxy is None:
                    continue

                boxes_xyxy.append([round(v, 2) for v in xyxy])
                y_labels.append(cls_id)

        if boxes_xyxy or allow_empty_images:
            records.append({
                "file_name": rel,
                "width": W,
                "height": H,
                "boxes": boxes_xyxy,
                "labels": y_labels
            })

    with open(output_dir / "annotations_torchvision.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(output_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "image_root": str(image_dir)}, f, indent=2)

    print(f"[done] wrote {len(records)} images → {output_dir/'annotations_torchvision.json'}")
    if miss_img or malformed or out_of_range:
        print(f"[notes] missing images: {miss_img}, malformed lines: {malformed}, class-id out-of-range: {out_of_range}")

def main():
    ap = argparse.ArgumentParser(description="YOLO → TorchVision JSON (no image copying).")
    ap.add_argument("--yolo_dir", required=True, help="Directory with YOLO .txt files")
    ap.add_argument("--image_dir", required=True, help="Directory with images")
    ap.add_argument("--class_list", required=True, help="classes.txt (one per line)")
    ap.add_argument("--output_dir", required=True, help="Folder to write JSON files")
    ap.add_argument("--allow_empty_images", action="store_true", help="Keep images with 0 boxes")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders")
    args = ap.parse_args()

    convert_yolo_to_torchvision(
        yolo_dir=args.yolo_dir,
        image_dir=args.image_dir,
        class_list_path=args.class_list,
        output_dir=args.output_dir,
        allow_empty_images=args.allow_empty_images,
        recursive=args.recursive
    )

if __name__ == "__main__":
    main()
