import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from data.tv_json_dataset import TVJsonDetection, collate_fn
from models.rcnn_builder import build_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp import GradScaler, autocast  
from tqdm import tqdm
import time
import random
from torch.utils.data import Subset
from itertools import islice

# ------------------------------
# Helpers
# ------------------------------

def subset_dataset(ds, limit=None, seed=0, require_boxes=True):
    """
    Returns a deterministic subset of ds with up to `limit` items.
    If require_boxes=True, only keep images that have ≥1 bbox.
    Assumes your TVJsonDetection exposes ds.items with "boxes".
    """
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    if require_boxes:
        idxs = [i for i, r in enumerate(ds.items) if len(r["boxes"]) > 0]
    rng.shuffle(idxs)
    idxs = idxs[:limit]
    return Subset(ds, idxs)

# ------------------------------
# Training / Validation loops
# ------------------------------
def train_epoch(model, loader, opt, scaler, device, clip=1.0, desc=""):
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    for imgs, targets in pbar:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        opt.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=(device.type == "cuda")):
            losses = model(imgs, targets)       # dict
            loss = sum(losses.values())

        scaler.scale(loss).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        scaler.step(opt); scaler.update()

        running += loss.item(); n += 1
        # Optional: show a tiny postfix without spamming
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running / max(1, n)  # average train loss

@torch.no_grad()
def val_loss(model, loader, device):
    model.train()
    total, n = 0.0, 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast('cuda', enabled=(device.type == "cuda")):
            losses = model(imgs, targets)
            total += float(sum(losses.values()).item())
        n += 1
    return total / max(1, n)

# ------------------------------
# COCO evaluation
# ------------------------------
def tvjson_to_coco_dict(tv_items, classes):
    images, annotations, categories = [], [], []
    for i, name in enumerate(classes, start=1):
        categories.append({"id": i, "name": name, "supercategory": "object"})
    ann_id = 1
    for img_id, r in enumerate(tv_items, start=1):
        images.append({"id": img_id, "file_name": r["file_name"],
                       "width": r["width"], "height": r["height"]})
        for box, label in zip(r["boxes"], r["labels"]):
            x1,y1,x2,y2 = box
            w,h = max(0.0, x2-x1), max(0.0, y2-y1)
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(label)+1,  # +1: background reserved at 0
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w*h), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}

@torch.no_grad()
@torch.no_grad()
def evaluate_coco_map(model, val_loader, val_ds, classes, device, eval_limit=None):
    model.eval()

    # use the filtered items the loader is actually iterating over
    tv_items_all = val_ds.dataset.items if isinstance(val_ds, torch.utils.data.Subset) else val_ds.items
    tv_items = tv_items_all[:eval_limit] if (eval_limit is not None) else tv_items_all

    coco_gt_dict = tvjson_to_coco_dict(tv_items, classes)
    coco_gt = COCO(); coco_gt.dataset = coco_gt_dict; coco_gt.createIndex()

    dt, img_counter, taken = [], 0, 0

    for images, _ in val_loader:
        # stop if we’ve reached the eval_limit
        if eval_limit is not None and taken >= eval_limit:
            break

        # trim batch so we don’t overshoot
        if eval_limit is not None:
            remaining = eval_limit - taken
            if remaining <= 0:
                break
            images = images[:remaining]

        images = [im.to(device) for im in images]
        outputs = model(images)   # list[dict]

        taken += len(outputs)

        for out in outputs:
            img_counter += 1
            boxes  = out["boxes"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            labels = out["labels"].cpu().tolist()
            for (x1,y1,x2,y2), s, c in zip(boxes, scores, labels):
                w, h = max(0.0, x2-x1), max(0.0, y2-y1)
                if w <= 0 or h <= 0:
                    continue
                cat_id = int(c)  # torchvision: 1..K (bg filtered)
                if cat_id < 1:
                    continue
                dt.append({
                    "image_id": img_counter,   # 1..len(tv_items)
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(s),
                })

    if not dt:
        print("No detections; mAP is 0.0")
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}

    coco_dt = coco_gt.loadRes(dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = coco_eval.stats
    return {"mAP": stats[0], "AP50": stats[1], "AP75": stats[2], "AR100": stats[8]}

# ------------------------------
# Main
# ------------------------------
def main(cfg):
    import yaml
    conf = yaml.safe_load(open(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    os.makedirs(os.path.dirname(conf["ckpt_dir"]), exist_ok=True)
    os.makedirs(conf["ckpt_dir"], exist_ok=True)

    # classes
    meta = json.load(open(f'{conf["data_root"]}/annotations/train/dataset_meta.json'))
    classes = meta["classes"]
    num_classes = len(classes) + 1

    # datasets (make sure TVJsonDetection sets self.ann_json = ann_json)
    train_ds = TVJsonDetection(conf["train_images"], conf["train_ann"])
    val_ds   = TVJsonDetection(conf["val_images"],   conf["val_ann"])

    # Apply subsets for quick debugging
    train_ds = subset_dataset(train_ds, conf.get("train_limit", None), seed=42)
    val_ds   = subset_dataset(val_ds,   conf.get("val_limit", None),   seed=43)
    train_loader = DataLoader(train_ds, batch_size=conf["batch_size"], shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=conf["batch_size"], shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    # model
    model = build_model(
        model_id=conf["model_id"],
        num_classes=num_classes,
        anchor_sizes=conf["anchor_sizes"],
        aspect_ratios=conf["aspect_ratios"],
        min_size=conf["min_size"],
        max_size=conf["max_size"],
        device=device,
        dtype=dtype
    )

    # optim / sched / scaler  ✅ moved scaler here (device now defined)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=conf["lr"], weight_decay=conf["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=conf["epochs"])
    scaler = GradScaler('cuda', enabled=(device.type=="cuda"))

    best_val, best_map = float("inf"), -1.0
    EVAL_EVERY = 1

    for epoch in range(1, conf["epochs"] + 1):
        t0 = time.time()

        tr = train_epoch(
            model, train_loader, opt, scaler, device,
            clip=conf["grad_clip"],
            desc=f"Epoch {epoch}/{conf['epochs']}"
        )
        vl = val_loss(model, val_loader, device)
        sched.step()

        metrics = None
        if epoch % EVAL_EVERY == 0:
            metrics = evaluate_coco_map(
                model, val_loader,
                val_ds=val_ds,
                classes=classes,         # <-- pass classes here
                device=device
        else:
            metrics = {"mAP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "AR100": float("nan")}

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{conf['epochs']} | "
            f"train {tr:.4f} | val {vl:.4f} | "
            f"mAP {metrics['mAP']:.3f} | AP50 {metrics['AP50']:.3f} | AP75 {metrics['AP75']:.3f} | "
            f"AR100 {metrics['AR100']:.3f} | time {elapsed/60:.1f} min"
        )

        # Save bests
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_loss": vl, "metrics": metrics},
                    f"{conf['ckpt_dir']}/best_by_loss.pth")
        if (not np.isnan(metrics["mAP"])) and metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_loss": vl, "metrics": metrics},
                    f"{conf['ckpt_dir']}/best_by_map.pth")
            
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mocs.yaml")
    args = ap.parse_args()
    main(args.config)
