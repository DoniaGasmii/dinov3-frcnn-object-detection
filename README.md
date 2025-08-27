# DINOv3-FRCNN: Faster R-CNN with a Frozen DINOv3 Backbone

This repo turns **DINOv3** (Meta) into a **frozen backbone** for **Faster R-CNN** (torchvision), with small trainable heads (RPN + ROI heads). It supports both **HF Hub** and **local snapshots** of the DINOv3 checkpoints.

## Why this repo?

* Plug-and-play **frozen** DINOv3 backbone (ViT-B/16, ViT-S/16, or ViT-7B/16).
* Trains only the **detection head** → faster convergence, strong features out of the box.
* Works with **YOLO-style labels** via provided converters (`yolo → torchvision json`) or directly with **COCO JSON**.
* Per-epoch **COCO mAP** evaluation (mAP, AP50, AP75).

---

## Quickstart

### 0) Environment

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.56.0.dev0" huggingface_hub safetensors accelerate pycocotools
# optional HF transfer acceleration:
pip install -U hf-transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
```

> Colab users: enable GPU, then run the above cells.

### 1) Get DINOv3 weights (choose one)

**A) From Hugging Face (gated):**

```python
from huggingface_hub import login
login()  # paste the token that has access to the DINOv3 gating group
model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # or vits16 / vit7b16
```

**B) From your Drive/local snapshot:**
Download once with `snapshot_download` and set:

```yaml
model_id: "/path/to/dinov3-vitb16-pretrain-lvd1689m"  # folder that contains config.json, preprocessor_config.json, model-*.safetensors
```

### 2) Dataset layout (TorchVision JSON)

```
DATASET_ROOT/
  images/
    train/ ...  (images can be nested; file names are relative)
    val/   ...
    test/  ...
  annotations/
    train/annotations_torchvision.json
    val/  /annotations_torchvision.json
    test/ /annotations_torchvision.json
    */dataset_meta.json      # {"classes": ["Worker", "Tower crane", ...]}
```

Convert with our utilities:

```bash
# YOLO → TorchVision JSON
python utils/data_format_conversions/yolo_to_torchvision.py \
  --yolo_dir  DATASET_ROOT/labels/train \
  --image_dir DATASET_ROOT/images/train \
  --output    DATASET_ROOT/annotations/train/annotations_torchvision.json \
  --class_list classes.txt
```

*or* from COCO:

```bash
python utils/data_format_conversions/coco_to_torchvision.py \
  --instances_json DATASET_ROOT/coco_annotations/instances_train.json \
  --image_dir      DATASET_ROOT/images/train \
  --output_dir     DATASET_ROOT/annotations/train
```

### 3) Minimal config (`configs/mocs.yaml`)

```yaml
# backbone
model_id: "/content/drive/.../dinov3-vitb16-pretrain-lvd1689m"  # or HF repo id

# data
train_images: "/path/to/DATASET_ROOT/images/train"
val_images:   "/path/to/DATASET_ROOT/images/val"
train_ann:    "/path/to/DATASET_ROOT/annotations/train/annotations_torchvision.json"
val_ann:      "/path/to/DATASET_ROOT/annotations/val/annotations_torchvision.json"

# training
epochs: 10
batch_size: 2
lr: 0.0005
grad_clip: 1.0
min_size: 640
max_size: 960
ckpt_dir: "/path/to/checkpoints"

# anchors (reasonable defaults)
anchor_sizes: [[32,64,128],[64,128,256],[128,256,512]]
aspect_ratios: [[0.5,1.0,2.0],[0.5,1.0,2.0],[0.5,1.0,2.0]]
```

### 4) Train

```bash
python train_frcnn.py --config configs/mocs.yaml
```

You’ll see per-epoch logs like:

```
Epoch 1/10 | train 0.7421 | val 0.5312 | mAP 0.247 | AP50 0.511 | AP75 0.263
saved checkpoints/best_by_loss.pth
```

### 5) Evaluate a checkpoint

```bash
python eval_frcnn.py --config configs/mocs.yaml --ckpt /path/to/checkpoints/best_by_map.pth
```

This prints COCO metrics and can dump visualizations.

---

## Models & Backbones

* **Small / fast**: `facebook/dinov3-vits16-pretrain-lvd1689m`
* **Balanced**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
* **Huge (research)**: `facebook/dinov3-vit7b16-pretrain-sat493m`

  > Needs large VRAM. Use `batch_size=1`, `min_size/max_size` smaller (e.g., 512/896), AMP on.

Backbone is **frozen** by default; only the detection head (RPN + ROI heads/FPN) is trained. You can unfreeze part of the backbone by setting `requires_grad_(True)` on selected layers.

---

## Notes on access & gating

* Some DINOv3 checkpoints are **gated on Hugging Face**. You must apply and accept the license to use them.
* If you don’t want to authenticate in Colab, **use a local snapshot path** as `model_id`(no network calls).

---

## Results on MOCS construction dataset (not ready)

| Backbone           | Input (min/max) | Batch | mAP@\[.5:.95] | AP50 |
| ------------------ | --------------: | ----: | ------------: | ---: |
| ViT-B/16           |         640/960 |     1 |             … |    … |
| ViT-7B/16 (frozen) |         640/960 |     1 |             … |    … |

---

## Repo structure

```
dinov3_frcnn/
  configs/
  models/
    dino_backbone.py        # HF load (local or hub), frozen backbone, pyramid heads
    rcnn_builder.py         # Faster R-CNN assembly (anchor gen, ROI pooler p3/p4/p5)
  data/
    tv_json_dataset.py      # TorchVision JSON dataset + robust missing-file skip
  utils/
    data_format_conversions/
      yolo_to_torchvision.py
      coco_to_torchvision.py
  train_frcnn.py
  eval_frcnn.py
  README.md
```
