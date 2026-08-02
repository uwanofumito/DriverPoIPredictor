"""Dataset for the JAAD pedestrian-state manifest (data/prepare_jaad_manifest.py)."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from models.pedestrian_state_model import LABEL_KEYS


class JAADPedestrianDataset(Dataset):
    def __init__(self, manifest_path, image_processor):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.processor = image_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["pedestrian_image"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            **{key: int(sample[key]) for key in LABEL_KEYS},
        }


def collate_jaad(features):
    batch = {"pixel_values": torch.stack([f["pixel_values"] for f in features])}
    for key in LABEL_KEYS:
        batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
    return batch
