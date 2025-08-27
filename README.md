# DINOv3-FRCNN: Faster R-CNN with a Frozen DINOv3 Backbone

This repo turns **DINOv3** (Meta) into a **frozen backbone** for **Faster R-CNN** (torchvision), with small trainable heads (RPN + ROI heads).  
It supports both **HF Hub** and **local snapshots** of the DINOv3 checkpoints.

## Why this repo?
- Plug-and-play **frozen DINOv3 backbone** (ViT-B/16, ViT-S/16, or ViT-7B/16).  
- Trains only the **detection head** → faster convergence, strong features out of the box.  
- Works with **YOLO-style labels** via provided converters (`yolo → torchvision json`) or directly with **COCO JSON**.  
- Per-epoch **COCO mAP evaluation** (mAP, AP50, AP75).  

# Quickstart

## **0) Environment**

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.56.0.dev0" huggingface_hub safetensors accelerate pycocotools

# optional HF transfer acceleration:
pip install -U hf-transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
````

> **Note:** Colab users: enable GPU, then run the above cells.

---

## **1) Get DINOv3 weights (choose one)**

### **A) From Hugging Face (gated):**

```python
from huggingface_hub import login

login()  # paste the token that has access to the DINOv3 gating group

model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # or vits16 / vit7b16
```

### **B) From your Drive/local snapshot:**

Download once with `snapshot_download` and set:

```yaml
model_id: "/path/to/dinov3-vitb16-pretrain-lvd1689m"  # folder that contains config.json, preprocess...
```

---

## **2) Dataset layout (TorchVision JSON)**

```text
DATASET_ROOT/
  images/
    train/ ...   (images can be nested; file names are relative)
    val/   ...
    test/  ...
  annotations/
    train/annotations_torchvision.json
    val/annotations_torchvision.json
    test/annotations_torchvision.json
  */dataset_meta.json   # {"classes": ["Worker", "Tower crane", ...]}
```

---

## **3) Convert with utilities**

### **YOLO → TorchVision JSON**

```bash
python utils/data_format_conversions/yolo_to_torchvision.py \
  --yolo_dir  DATASET_ROOT/labels/train \
  --image_dir DATASET_ROOT/images/train \
  --output    DATASET_ROOT/annotations/train/annotations_torchvision.json \
  --class_list classes.txt
```

### **or from COCO:**

```bash
python utils/data_format_conversions/coco_to_torchvision.py \
  --instances_json DATASET_ROOT/coco_annotations/instances_train.json \
  --image_dir      DATASET_ROOT/images/train \
  --output_dir     DATASET_ROOT/annotations/train
```

````

