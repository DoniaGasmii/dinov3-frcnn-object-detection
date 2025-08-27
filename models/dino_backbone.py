import torch, torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from torchvision.ops import FeaturePyramidNetwork
from typing import Dict
import math

class DinoV3Feature(nn.Module):
    def __init__(self, hf_model, patch_size=16):
        super().__init__()
        self.m = hf_model.eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.patch = patch_size
        self.C = hf_model.config.hidden_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(pixel_values=x, output_hidden_states=True).last_hidden_state
        B,T,C = out.shape
        H,W = x.shape[-2:]
        h,w = H//self.patch, W//self.patch
        tokens = out[:,1:,:][:, :h*w, :]
        f16 = tokens.view(B,h,w,C).permute(0,3,1,2).contiguous()  # [B,C,h,w]
        return f16

class DinoV3AdapterFPN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        self.p4 = nn.Conv2d(out_dim, out_dim, 3, 2, 1)  # stride 32
        self.p5 = nn.Conv2d(out_dim, out_dim, 3, 2, 1)  # stride 64
        self.fpn = FeaturePyramidNetwork([out_dim, out_dim, out_dim], out_dim)
        self.out_channels = out_dim

    def forward(self, f16: torch.Tensor) -> Dict[str, torch.Tensor]:
        p3 = self.proj(f16)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return self.fpn({"p3": p3, "p4": p4, "p5": p5})

class DinoV3BackboneForRCNN(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        patch = getattr(hf_model.config, "patch_size", 16)
        self.extract = DinoV3Feature(hf_model, patch)
        self.adapter = DinoV3AdapterFPN(self.extract.C, 256)
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f16 = self.extract(x)
        return self.adapter(f16)

def load_dino(model_id: str, device: torch.device, dtype):
    """
    Loads DINOv3 from either a local directory (preferred) or HF Hub.
    If `model_id` is a directory, we force local loading (no auth needed).
    """
    from pathlib import Path
    local = Path(model_id).is_dir()

    # Try to load the image processor; if it's missing locally,
    # fall back to ImageNet stats so torchvision's transform stays consistent.
    proc = None
    try:
        proc = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=local,   # <- no network when local
        )
    except Exception as e:
        print(f"⚠️ Could not load processor from '{model_id}' ({e}). "
              f"Falling back to ImageNet mean/std.")
        class _Fallback:
            image_mean = [0.485, 0.456, 0.406]
            image_std  = [0.229, 0.224, 0.225]
        proc = _Fallback()

    # Always load the model from the same place; if local, prevent downloads.
    hf_model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=local,      # <- no network when local
    ).to(device).eval()

    return proc, hf_model
