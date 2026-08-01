"""Manifest-driven dataset (source-agnostic) + a synthetic mock for pipeline
smoke-testing.

Manifest schema (JSON, one list of samples) expected by ManifestDataset:

    [
      {
        "road_image": "relative/or/absolute/path/road.jpg",
        "driver_image": "relative/or/absolute/path/driver.jpg",
        "control": [[steering, throttle, brake, speed, accel_x, accel_y, accel_z], ...],
        "text": "highway, clear weather",   # optional; omit or "" if none
        "label": 0,
        "gaze_target": [gx, gy]   # optional; normalized windshield point in [0, 1], to supervise GazeHead
      },
      ...
    ]

`control` is a short time-series window (T timesteps); a single snapshot
(T=1) works too. No real dataset ships this exact JSON directly — write a
small preprocessing script that extracts frames + syncs telemetry into this
format. See data/prepare_ets2_manifest.py for a worked example (Brain4Cars'
own raw video is no longer retrievable from its official Dropbox link — see
project README — so the current preprocessing pipeline targets the Zenodo
"ETS2 simulator" driving dataset instead, which has the same
road-camera/driver-camera/telemetry shape).
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, image_processor, control_input_dim):
        self.data_root = Path(manifest_path).parent
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.processor = image_processor
        self.control_input_dim = control_input_dim

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path):
        path = Path(rel_path)
        if not path.is_absolute():
            path = self.data_root / path
        image = Image.open(path).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, index):
        sample = self.samples[index]
        control = torch.tensor(sample["control"], dtype=torch.float32)
        if control.ndim == 1:
            control = control.unsqueeze(0)  # [1, control_input_dim]

        item = {
            "road_pixel_values": self._load_image(sample["road_image"]),
            "driver_pixel_values": self._load_image(sample["driver_image"]),
            "control_series": control,
            "text": sample.get("text", "") or "",
            "label": int(sample["label"]),
        }
        if "gaze_target" in sample:
            item["gaze_target"] = torch.tensor(sample["gaze_target"], dtype=torch.float32)
        return item


class MockDriverStateDataset(Dataset):
    """Generates random data with the right shapes — no files, no download of
    a real dataset. Use this to verify the model/training loop run end-to-end
    before pointing at real (restricted) Brain4Cars data.
    """

    def __init__(
        self,
        num_samples,
        image_size,
        control_input_dim,
        num_classes,
        control_len_range=(1, 5),
        text_prob=0.5,
        has_gaze_target=True,
        seed=0,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.control_input_dim = control_input_dim
        self.num_classes = num_classes
        self.control_len_range = control_len_range
        self.text_prob = text_prob
        self.has_gaze_target = has_gaze_target
        self._rng = random.Random(seed)
        self._sample_texts = [
            "highway, clear weather",
            "urban intersection, heavy traffic",
            "night driving, rain",
            "",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(index)
        t = self._rng.randint(*self.control_len_range)

        item = {
            "road_pixel_values": torch.randn(*self.image_size, generator=g),
            "driver_pixel_values": torch.randn(*self.image_size, generator=g),
            "control_series": torch.randn(t, self.control_input_dim, generator=g),
            "text": self._rng.choice(self._sample_texts) if self._rng.random() < self.text_prob else "",
            "label": self._rng.randrange(self.num_classes),
        }
        if self.has_gaze_target:
            item["gaze_target"] = torch.rand(2, generator=g)
        return item


def collate_driver_state(features):
    max_t = max(f["control_series"].shape[0] for f in features)
    control_dim = features[0]["control_series"].shape[1]

    padded_control = torch.zeros(len(features), max_t, control_dim)
    for i, f in enumerate(features):
        t = f["control_series"].shape[0]
        padded_control[i, :t] = f["control_series"]

    batch = {
        "road_pixel_values": torch.stack([f["road_pixel_values"] for f in features]),
        "driver_pixel_values": torch.stack([f["driver_pixel_values"] for f in features]),
        "control_series": padded_control,
        "texts": [f["text"] for f in features],
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
    }
    if all("gaze_target" in f for f in features):
        batch["gaze_targets"] = torch.stack([f["gaze_target"] for f in features])
    return batch
