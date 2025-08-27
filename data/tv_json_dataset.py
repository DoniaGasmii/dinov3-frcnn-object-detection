# data/tv_json_dataset.py
import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

class TVJsonDetection(Dataset):
    def __init__(self, image_root: str, ann_json: str, allow_empty_images=True, verbose=True):
        self.root = image_root
        raw = json.load(open(ann_json, "r", encoding="utf-8"))
        self.items = []
        missing = mism = 0
        for r in raw:
            path = os.path.join(self.root, r["file_name"])
            if not os.path.exists(path):
                missing += 1
                continue
            boxes = r.get("boxes", [])
            labels = r.get("labels", [])
            if len(boxes) != len(labels):
                mism += 1
                continue
            if (len(boxes) == 0) and (not allow_empty_images):
                continue
            self.items.append(r)
        if verbose:
            print(f"[TVJsonDetection] {ann_json}: kept {len(self.items)} | missing {missing} | mismatch {mism}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = Image.open(os.path.join(self.root, r["file_name"])).convert("RGB")
        boxes  = torch.tensor(r["boxes"], dtype=torch.float32)
        labels = torch.tensor(r["labels"], dtype=torch.int64)
        return to_tensor(img), {"boxes": boxes, "labels": labels}

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
