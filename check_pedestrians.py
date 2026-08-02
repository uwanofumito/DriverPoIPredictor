"""Scan the ETS2 road-camera frames for pedestrian detections, to check
whether the dataset even contains pedestrian-crossing scenarios before
building any hazard-response-mismatch labeling on top of it.

Uses a COCO-pretrained torchvision detector (no extra data/registration).
"""

import glob

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms import functional as F

COCO_PERSON_CLASS_ID = 1  # torchvision COCO detection label indices
SCORE_THRESHOLD = 0.5

weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
model.eval()

road_frames = sorted(glob.glob("data/raw/ets2/prepared/frames/*/*_road.jpg"))
print(f"Scanning {len(road_frames)} road frames for person detections (score > {SCORE_THRESHOLD})...")

hits = []
with torch.no_grad():
    for path in road_frames:
        image = Image.open(path).convert("RGB")
        tensor = F.to_tensor(image)
        output = model([tensor])[0]

        person_scores = [
            float(s) for label, s in zip(output["labels"], output["scores"])
            if int(label) == COCO_PERSON_CLASS_ID and float(s) > SCORE_THRESHOLD
        ]
        if person_scores:
            hits.append((path, max(person_scores)))

print(f"\n{len(hits)} / {len(road_frames)} frames had a person detection above {SCORE_THRESHOLD}:")
for path, score in sorted(hits, key=lambda x: -x[1]):
    print(f"  {score:.2f}  {path}")
