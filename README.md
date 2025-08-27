# DINOv3-FRCNN: Faster R-CNN with a Frozen DINOv3 Backbone

This repo turns **DINOv3** (Meta) into a **frozen backbone** for **Faster R-CNN** (torchvision), with small trainable heads (RPN + ROI heads).  
It supports both **HF Hub** and **local snapshots** of the DINOv3 checkpoints.

## Why this repo?
- Plug-and-play **frozen DINOv3 backbone** (ViT-B/16, ViT-S/16, or ViT-7B/16).  
- Trains only the **detection head** → faster convergence, strong features out of the box.  
- Works with **YOLO-style labels** via provided converters (`yolo → torchvision json`) or directly with **COCO JSON**.  
- Per-epoch **COCO mAP evaluation** (mAP, AP50, AP75).  

### Quickstart

#### 0) Environment

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.56.0.dev0" huggingface_hub safetensors accelerate pycocotools

# optional HF transfer acceleration:
pip install -U hf-transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
```
>**Note:** Colab users: enable GPU, then run the above cells.
