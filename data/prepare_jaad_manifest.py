"""Convert JAAD (Rasouli et al., "Are They Going to Cross?", ICCVW 2017;
annotations from ykotseruba/JAAD, MIT license) into a pedestrian-state
manifest: one entry per sampled (pedestrian, frame), with real per-frame
attention/crossing/action ground truth (no proxy labels needed here, unlike
the driver side).

This is a STANDALONE manifest for a standalone pedestrian-state model — not
wired into DriverStateModel. See README's "Integrating pedestrian-state
estimation later" for how it's meant to attach eventually.

Expected input layout:

    <src_root>/
      annotations/video_0001.xml, ...       # from ykotseruba/JAAD
      clips/video_0001.mp4, ...             # from JAAD_clips.zip

Only `track label="pedestrian"` entries carry behavior annotations (JAAD
also has "ped"/"people" tracks for bystanders/groups — no attributes, skipped).

Usage:
    python data/prepare_jaad_manifest.py \
        --annotations-dir data/raw/jaad/annotations_repo/annotations \
        --clips-dir data/raw/jaad/clips \
        --out-dir data/raw/jaad/prepared \
        --frame-stride 15
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

LABEL_MAPS = {
    "look": {"not-looking": 0, "looking": 1},
    "cross": {"not-crossing": 0, "crossing": 1},
    "action": {"standing": 0, "walking": 1},
}


def parse_video_annotations(xml_path, frame_stride):
    """Returns {frame_idx: [ (pedestrian_id, bbox, attrs), ... ]}, subsampled
    every `frame_stride`-th annotated frame per pedestrian track."""
    root = ET.parse(xml_path).getroot()
    frames_needed = {}

    for track in root.findall(".//track[@label='pedestrian']"):
        boxes = track.findall("box")
        for i, box in enumerate(boxes):
            if i % frame_stride != 0:
                continue
            if box.get("occluded") == "2":  # fully occluded
                continue

            attrs = {a.get("name"): a.text for a in box.findall("attribute")}
            ped_id = attrs.get("id")
            bbox = (
                float(box.get("xtl")), float(box.get("ytl")),
                float(box.get("xbr")), float(box.get("ybr")),
            )
            frame_idx = int(box.get("frame"))

            labels = {}
            ok = True
            for key, mapping in LABEL_MAPS.items():
                val = attrs.get(key)
                if val not in mapping:
                    ok = False
                    break
                labels[key] = mapping[val]
            if not ok:
                continue

            frames_needed.setdefault(frame_idx, []).append((ped_id, bbox, labels))

    return frames_needed


def extract_crops(video_path, frames_needed, out_dir, video_id, margin=0.15):
    cap = cv2.VideoCapture(str(video_path))
    entries = []
    frame_idx = 0
    target_frames = set(frames_needed.keys())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_frames:
            h, w = frame.shape[:2]
            for ped_id, (xtl, ytl, xbr, ybr), labels in frames_needed[frame_idx]:
                bw, bh = xbr - xtl, ybr - ytl
                x0 = max(0, int(xtl - bw * margin))
                y0 = max(0, int(ytl - bh * margin))
                x1 = min(w, int(xbr + bw * margin))
                y1 = min(h, int(ybr + bh * margin))
                if x1 <= x0 or y1 <= y0:
                    continue

                crop = frame[y0:y1, x0:x1]
                out_path = out_dir / video_id / f"{ped_id}_{frame_idx:05d}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), crop)

                entries.append({
                    "pedestrian_image": str(out_path),
                    **labels,
                    "video": video_id,
                    "frame": frame_idx,
                    "pedestrian_id": ped_id,
                })
        frame_idx += 1

    cap.release()
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-stride", type=int, default=15, help="Sample every Nth annotated frame per pedestrian track")
    parser.add_argument("--limit-videos", type=int, default=None, help="For a quick trial run")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"

    xml_files = sorted(annotations_dir.glob("video_*.xml"))
    if args.limit_videos:
        xml_files = xml_files[: args.limit_videos]

    manifest = []
    missing_videos = []
    for xml_path in xml_files:
        video_id = xml_path.stem
        video_path = clips_dir / f"{video_id}.mp4"
        if not video_path.exists():
            missing_videos.append(video_id)
            continue

        frames_needed = parse_video_annotations(xml_path, args.frame_stride)
        if not frames_needed:
            continue

        entries = extract_crops(video_path, frames_needed, crops_dir, video_id)
        manifest.extend(entries)
        print(f"{video_id}: {len(entries)} crops")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    print(f"\nWrote {len(manifest)} pedestrian crops to {manifest_path}")
    if missing_videos:
        print(f"Skipped {len(missing_videos)} videos (clip file not found): {missing_videos[:10]}{'...' if len(missing_videos) > 10 else ''}")

    for key in LABEL_MAPS:
        pos = sum(e[key] for e in manifest)
        print(f"  {key}: {pos} positive / {len(manifest) - pos} negative")


if __name__ == "__main__":
    main()
