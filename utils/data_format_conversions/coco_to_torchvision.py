#!/usr/bin/env python3
"""
Convert COCO instances.json to a TorchVision-friendly JSON
(no image copying; paths stored relative to --image_dir).

Example:
  python3 coco_to_torchvision.py \
    --instances_json /path/to/instances_train.json \
    --image_dir      /path/to/images/train \
    --output_dir     /path/to/annotations/train
"""
import json, argparse
from pathlib import Path

def coco_to_tv(instances_json, image_dir, output_dir):
    instances_json = Path(instances_json)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(instances_json, "r", encoding="utf-8"))
    # Map image_id -> image record
    img_map = {im["id"]: im for im in coco.get("images", [])}

    # Build category id -> contiguous label (0..K-1) and names
    cats = coco.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    class_names = [c["name"] for c in cats_sorted]

    # Gather anns per image
    per_image = {img_id: {"boxes": [], "labels": []} for img_id in img_map}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:  # skip crowd by default
            continue
        img_id = ann["image_id"]
        if img_id not in per_image: 
            continue
        x, y, w, h = ann["bbox"]
        # convert xywh -> xyxy
        x1, y1, x2, y2 = x, y, x + w, y + h
        W, H = img_map[img_id]["width"], img_map[img_id]["height"]
        # clip
        x1 = max(0.0, min(x1, W)); y1 = max(0.0, min(y1, H))
        x2 = max(0.0, min(x2, W)); y2 = max(0.0, min(y2, H))
        if x2 <= x1 or y2 <= y1: 
            continue
        per_image[img_id]["boxes"].append([round(x1,2), round(y1,2), round(x2,2), round(y2,2)])
        per_image[img_id]["labels"].append(cat_id_to_idx[ann["category_id"]])

    # Assemble TorchVision list
    records = []
    for img_id, im in img_map.items():
        file_name = im["file_name"]
        # store as relative to --image_dir if it already is, otherwise keep as-is
        # (torch dataloader can join with image_dir)
        rec = {
            "file_name": file_name,
            "width": im["width"],
            "height": im["height"],
            "boxes": per_image[img_id]["boxes"],
            "labels": per_image[img_id]["labels"],
        }
        records.append(rec)

    # write
    with open(output_dir / "annotations_torchvision.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(output_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "image_root": str(image_dir)}, f, indent=2)
    print(f"[done] {len(records)} images → {output_dir/'annotations_torchvision.json'}")

def main():
    ap = argparse.ArgumentParser(description="COCO → TorchVision JSON (no image copying).")
    ap.add_argument("--instances_json", required=True, help="Path to instances_{split}.json")
    ap.add_argument("--image_dir", required=True, help="Folder with images for the same split")
    ap.add_argument("--output_dir", required=True, help="Where to write annotations_torchvision.json")
    args = ap.parse_args()
    coco_to_tv(args.instances_json, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()
