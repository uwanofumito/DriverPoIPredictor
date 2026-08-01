"""Gaze estimation via a Gaze360-pretrained network (yakhyo/gaze-estimation,
resnet18, MIT license — see models/gaze_resnet.py), replacing the earlier
head-pose-only proxy (data/head_pose.py, now retired).

Why this instead of head pose: head orientation is a poor stand-in for
where the eyes are actually pointed (you can hold your head still and look
sideways), and the earlier version had zero training on real gaze-labeled
data. This one was pretrained on Gaze360 — real gaze annotations from many
subjects — at the cost of ~11-13 degrees mean angular error per the
project's reported benchmarks. It's a real gaze estimator, but still not
calibrated to this dataset's specific camera mount, so the windshield
projection below (pose_to_gaze_point) remains a heuristic.

MediaPipe FaceLandmarker is kept around only as a face detector/cropper here
(the gaze model needs a face crop, not a full frame) — its head-pose output
is no longer used.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# so `from models.gaze_resnet import ...` resolves regardless of whether the
# caller was invoked as `python data/prepare_ets2_manifest.py` (which puts
# data/ on sys.path, not the project root) or as a package.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_landmarker = None
_gaze_model = None

FACE_MARGIN = 0.3  # expand the landmark bbox by this fraction on each side


def _get_landmarker(model_path="models_data/face_landmarker.task"):
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _get_gaze_model(weights_path="models_data/gaze_resnet18.pt"):
    global _gaze_model
    if _gaze_model is None:
        import torch

        from models.gaze_resnet import gaze_resnet18

        model = gaze_resnet18()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        _gaze_model = model
    return _gaze_model


def _detect_face_crop(image_path):
    import mediapipe as mp

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    mp_image = mp.Image.create_from_file(str(image_path))
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None

    xs = [lm.x for lm in result.face_landmarks[0]]
    ys = [lm.y for lm in result.face_landmarks[0]]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0.0, x_min - bw * FACE_MARGIN) * w
    x_max = min(1.0, x_max + bw * FACE_MARGIN) * w
    y_min = max(0.0, y_min - bh * FACE_MARGIN) * h
    y_max = min(1.0, y_max + bh * FACE_MARGIN) * h

    return image.crop((x_min, y_min, x_max, y_max))


def estimate_gaze_yaw_pitch(image_path):
    """Returns (yaw, pitch) in degrees, or None if no face detected."""
    import torch
    from torchvision import transforms

    from models.gaze_resnet import decode_bins

    face = _detect_face_crop(image_path)
    if face is None:
        return None

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    face_tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        yaw_logits, pitch_logits = _get_gaze_model()(face_tensor)
        yaw, pitch = decode_bins(yaw_logits, pitch_logits)

    return float(yaw.item()), float(pitch.item())


# Heuristic windshield field-of-view used to map a gaze angle -> a 2D point
# on the road image, purely for visualizing/supervising GazeHead. The Gaze360
# model itself is real (learned from human gaze data); this projection step
# is not — there's no calibrated mapping from gaze angle to this specific
# camera's road-image plane.
YAW_FOV_DEG = 35.0
PITCH_FOV_DEG = 25.0


def gaze_to_point(yaw, pitch):
    """(yaw, pitch) in degrees -> normalized (gx, gy) in [0, 1], clipped."""
    gx = 0.5 + (yaw / YAW_FOV_DEG) * 0.5
    gy = 0.5 + (pitch / PITCH_FOV_DEG) * 0.5
    return float(np.clip(gx, 0.0, 1.0)), float(np.clip(gy, 0.0, 1.0))
