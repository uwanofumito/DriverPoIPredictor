"""Head-pose estimation via MediaPipe FaceLandmarker.

Used as a weak-supervision proxy for two things we have no real ground
truth for (see data/prepare_ets2_manifest.py):
  1. a distraction/eyes-off-road label (derived from downward pitch —
     looking down at a phone/lap/dashboard, as opposed to sideways glances
     which are often legitimate mirror/blind-spot checks before a turn or
     lane change and would otherwise be misread as "distraction")
  2. a rough gaze-target point on the windshield, to supervise GazeHead
     instead of leaving it fully unsupervised.

This is a heuristic proxy, not measured gaze — there is no eye-tracking
ground truth in the ETS2 dataset (or Brain4Cars) to validate against.
"""

import numpy as np

_landmarker = None


def _get_landmarker(model_path="models_data/face_landmarker.task"):
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=False,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def estimate_yaw_pitch_roll(image_path):
    """Returns (yaw, pitch, roll) in degrees, or None if no face detected."""
    import mediapipe as mp

    landmarker = _get_landmarker()
    img = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(img)

    if not result.facial_transformation_matrixes:
        return None

    M = np.array(result.facial_transformation_matrixes[0])
    R = M[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    return yaw, pitch, roll


# Heuristic windshield field-of-view used to map head pose -> a 2D point on
# the road image, purely for visualizing/supervising GazeHead. Not measured.
YAW_FOV_DEG = 30.0
PITCH_FOV_DEG = 20.0


def pose_to_gaze_point(yaw, pitch):
    """(yaw, pitch) in degrees -> normalized (gx, gy) in [0, 1], clipped."""
    gx = 0.5 + (yaw / YAW_FOV_DEG) * 0.5
    gy = 0.5 + (pitch / PITCH_FOV_DEG) * 0.5
    return float(np.clip(gx, 0.0, 1.0)), float(np.clip(gy, 0.0, 1.0))
