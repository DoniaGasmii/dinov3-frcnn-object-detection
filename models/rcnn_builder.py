import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .dino_backbone import DinoV3BackboneForRCNN, load_dino

import torchvision
from torchvision.ops import MultiScaleRoIAlign

def build_model(model_id, num_classes, anchor_sizes, aspect_ratios, min_size, max_size, device, dtype):
    proc, hf_model = load_dino(model_id, device, dtype)
    backbone = DinoV3BackboneForRCNN(hf_model)

    anchor_gen = AnchorGenerator(
        sizes=tuple(tuple(s) for s in anchor_sizes),
        aspect_ratios=tuple(tuple(a) for a in aspect_ratios)
    )

    # 🔧 Tell ROI heads which pyramid maps to pool from
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["p3", "p4", "p5"],  # must match your backbone dict keys
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pooler,                # <<< add this
        image_mean=getattr(proc, "image_mean", [0.485,0.456,0.406]),
        image_std=getattr(proc, "image_std",  [0.229,0.224,0.225]),
        min_size=min_size,
        max_size=max_size,
    ).to(device)

    for n, p in model.named_parameters():
        if "extract.m" in n:
            p.requires_grad_(False)
    return model
