"""Convert the Zenodo "ETS2 simulator" driving dataset (Tonutti & Ruffaldi,
2017; DOI 10.5281/zenodo.1009540) into the manifest JSON schema ManifestDataset
expects.

Why this dataset instead of Brain4Cars: Brain4Cars' own raw video is no
longer retrievable from its official Dropbox share (the folder tree is
still there, but every event folder is empty — the files were removed at
some point after the link was published). This ETS2 dataset has the same
shape (paired road-facing + driver-facing video, per-event vehicle
telemetry) and is actually downloadable, so it's used here to validate the
real-data path end-to-end.

Labels: a distraction/eyes-off-road proxy derived from head pose (see
head_pose.py), not the dataset's native maneuver labels (lchange/lturn/
rchange/rturn/straight) — those are kept per-event under "maneuver" for
reference but aren't used as the training target. There is no measured
alertness/distraction ground truth here or in Brain4Cars; this is a
heuristic stand-in, see the module docstrings for what it does and doesn't
capture.

Session-relative baseline: each recording session (one date+time, one
subject, one camera mounting) has its own neutral head pitch — e.g. one
subject's neutral posture reads as +8° pitch, another's as -1°, just from
how the camera happened to be mounted/aimed that day. An absolute pitch
threshold would mostly measure camera placement, not driver behavior. So
distraction is defined as pitch deviation *from that session's own median*,
not from zero.

Expected input layout (after extracting the three zips into one directory):

    <src_root>/
      face_camera/<class>/driver_<event_id>.mov
      road_camera/<class>/road_<event_id>.mov
      car_data/<class>/<event_id>/*.json   (many per-instant telemetry snapshots)

    classes = lchange, lturn, rchange, rturn, straight

Usage:
    python data/prepare_ets2_manifest.py \
        --src-root data/raw/ets2/extracted \
        --out-dir data/raw/ets2/prepared
"""

import argparse
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

from head_pose import estimate_yaw_pitch_roll, pose_to_gaze_point

CLASS_NAMES = ["lchange", "lturn", "rchange", "rturn", "straight"]

# Distraction proxy: looking DOWN (pitch above the session's own median, e.g.
# at a phone/lap/dashboard) beyond this threshold, in at least this fraction
# of sampled frames. Yaw (sideways) is deliberately excluded from the
# distraction criterion — before turns/lane changes a sideways glance is
# often a legitimate mirror/blind-spot check, not distraction, and this
# dataset has no independent ground truth to tell the two apart.
RELATIVE_PITCH_DEG = 10.0
DISTRACTION_RATIO_THRESHOLD = 0.3


def session_id_from_event_id(event_id):
    """'2017-07-05_13-19_140-600_lchange' -> '2017-07-05_13-19' (date+time)."""
    parts = event_id.split("_")
    return "_".join(parts[:2])


def extract_last_frame(video_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-sseof", "-0.5", "-i", str(video_path),
            "-update", "1", "-q:v", "2", str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def extract_sampled_frames(video_path, num_samples, tmp_dir, clip_seconds=5.0):
    """Evenly-spaced frames across the clip, for head-pose sampling only.

    Clips in this dataset are documented as "5 seconds before" each maneuver
    (see README), so a fixed fps filter yields ~num_samples frames without
    needing to probe each file's exact duration/frame count.
    """
    pattern = str(Path(tmp_dir) / "pose_%03d.jpg")
    fps = max(num_samples / clip_seconds, 1.0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "4", pattern,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sorted(Path(tmp_dir).glob("pose_*.jpg"))


def sample_event_poses(driver_video_path, num_pose_samples):
    """Returns list of (yaw, pitch, roll) tuples across the clip (face-detected frames only)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            frames = extract_sampled_frames(driver_video_path, num_pose_samples, tmp_dir)
        except subprocess.CalledProcessError:
            return []
        return [
            pose for f in frames
            if (pose := estimate_yaw_pitch_roll(f)) is not None
        ]


def control_vector_from_json(payload):
    truck = payload["truck"]
    accel = truck["acceleration"]
    return [
        truck["userSteer"],
        truck["userThrottle"],
        truck["userBrake"],
        truck["speed"],
        accel["x"],
        accel["y"],
        accel["z"],
    ]


def load_control_series(car_data_event_dir, max_samples):
    json_files = list(car_data_event_dir.glob("*.json"))

    def frame_key(p):
        m = re.search(r"_(\d+)\.json$", p.name)
        return int(m.group(1)) if m else 0

    json_files.sort(key=frame_key)
    if max_samples and len(json_files) > max_samples:
        json_files = json_files[-max_samples:]  # keep the samples closest to the maneuver

    series = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            payload = json.load(f)
        series.append(control_vector_from_json(payload))
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, help="Directory containing face_camera/road_camera/car_data")
    parser.add_argument("--out-dir", required=True, help="Where to write extracted frames + manifest.json")
    parser.add_argument("--max-control-samples", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=None, help="For a quick trial run")
    parser.add_argument("--num-pose-samples", type=int, default=15, help="Frames sampled per clip for the distraction proxy")
    parser.add_argument("--skip-gaze", action="store_true", help="Skip head-pose/distraction labeling (maneuver-only manifest)")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    # ---------- gather event file paths + control series (cheap) ----------
    events = []
    skipped = []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        road_files = sorted((src_root / "road_camera" / cls).glob("road_*.mov"))
        if args.limit_per_class:
            road_files = road_files[: args.limit_per_class]

        for road_path in road_files:
            event_id = road_path.stem[len("road_"):]  # strip "road_" prefix
            driver_path = src_root / "face_camera" / cls / f"driver_{event_id}.mov"
            car_data_dir = src_root / "car_data" / cls / event_id

            if not driver_path.exists() or not car_data_dir.is_dir():
                skipped.append(event_id)
                continue

            control = load_control_series(car_data_dir, args.max_control_samples)
            if not control:
                skipped.append(event_id)
                continue

            events.append({
                "cls": cls, "cls_idx": cls_idx, "event_id": event_id,
                "road_path": road_path, "driver_path": driver_path, "control": control,
            })

    # ---------- pass 1: sample head poses per event (the expensive step) ----------
    no_face_events = []
    if not args.skip_gaze:
        for ev in events:
            poses = sample_event_poses(ev["driver_path"], args.num_pose_samples)
            if not poses:
                no_face_events.append(ev["event_id"])
            ev["poses"] = poses
        events = [ev for ev in events if ev.get("poses")]

        # ---------- session-relative pitch baseline ----------
        session_pitches = {}
        for ev in events:
            sid = session_id_from_event_id(ev["event_id"])
            median_pitch = statistics.median(pt for (_, pt, _) in ev["poses"])
            session_pitches.setdefault(sid, []).append(median_pitch)
        session_baseline = {sid: statistics.median(vals) for sid, vals in session_pitches.items()}

    # ---------- pass 2: extract frames + assemble manifest ----------
    manifest = []
    for ev in events:
        cls, event_id = ev["cls"], ev["event_id"]
        road_frame = frames_dir / cls / f"{event_id}_road.jpg"
        driver_frame = frames_dir / cls / f"{event_id}_driver.jpg"
        try:
            extract_last_frame(ev["road_path"], road_frame)
            extract_last_frame(ev["driver_path"], driver_frame)
        except subprocess.CalledProcessError:
            skipped.append(event_id)
            continue

        entry = {
            "road_image": str(road_frame.relative_to(out_dir)),
            "driver_image": str(driver_frame.relative_to(out_dir)),
            "control": ev["control"],
            "text": "",
            "maneuver": cls,
        }

        if args.skip_gaze:
            entry["label"] = ev["cls_idx"]
        else:
            sid = session_id_from_event_id(event_id)
            baseline = session_baseline[sid]
            deviations = [pt - baseline for (_, pt, _) in ev["poses"]]
            down_ratio = sum(1 for d in deviations if d > RELATIVE_PITCH_DEG) / len(deviations)
            entry["label"] = int(down_ratio > DISTRACTION_RATIO_THRESHOLD)

            last_pose = estimate_yaw_pitch_roll(driver_frame)
            if last_pose is None:
                skipped.append(event_id)
                continue
            yaw, pitch, _ = last_pose
            entry["gaze_target"] = list(pose_to_gaze_point(yaw, pitch - baseline))
            entry["_down_ratio"] = round(down_ratio, 3)

        manifest.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    print(f"Wrote {len(manifest)} events to {manifest_path}")
    if skipped:
        print(f"Skipped {len(skipped)} events (missing pair/extraction failure): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    if no_face_events:
        print(f"Skipped {len(no_face_events)} events (no face detected in any sampled frame): {no_face_events[:10]}{'...' if len(no_face_events) > 10 else ''}")
    if args.skip_gaze:
        print(f"Class order (label index): {CLASS_NAMES}")
    else:
        n_distracted = sum(e["label"] for e in manifest)
        print(f"label = distraction proxy (session-relative pitch-down ratio > {DISTRACTION_RATIO_THRESHOLD}): "
              f"{n_distracted} distracted / {len(manifest) - n_distracted} attentive")
        print("'maneuver' field kept per-event for reference (not used as the training label)")


if __name__ == "__main__":
    main()
