"""Run the trained model frame-by-frame over a few real ETS2 clips and
assemble an animated GIF of the predicted gaze heatmap over time, alongside
the driver's face over the same span — for showing the actual video motion
instead of a single static frame.
"""

import argparse
import base64
import io
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from config import DriverStateConfig
from models.driver_state_model import DriverStateModel

EVENT_IDS = [
    "2017-07-05_13-19_502-699_lchange",  # highest down-gaze ratio (0.467, distracted)
    "2017-07-05_13-19_455-198_lturn",    # borderline (0.267, attentive)
    "2017-07-05_13-19_157-704_rchange",  # clearly attentive (0.0)
]


def extract_frames(video_path, num_samples, tmp_dir, clip_seconds=5.0):
    pattern = str(Path(tmp_dir) / "f_%03d.jpg")
    fps = max(num_samples / clip_seconds, 1.0)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "3", pattern],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(Path(tmp_dir).glob("f_*.jpg"))


def overlay_heatmap(base_image, heatmap, alpha=0.55):
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(base_image.size, Image.BILINEAR)
    return Image.blend(base_image.convert("RGB"), colored_img, alpha)


def to_gif_data_uri(frames, duration_ms):
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{b64}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default="data/raw/ets2/extracted")
    parser.add_argument("--manifest", default="data/raw/ets2/prepared/manifest.json")
    parser.add_argument("--checkpoint", default="outputs/driver_state_model.pt")
    parser.add_argument("--out-json", default="outputs/animated_examples.json")
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--frame-size", type=int, default=320)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_event = {Path(e["road_image"]).stem.replace("_road", ""): e for e in manifest}

    config = DriverStateConfig(num_classes=2)
    model = DriverStateModel(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    results = []
    for event_id in EVENT_IDS:
        entry = by_event[event_id]
        cls = entry["maneuver"]
        road_video = src_root / "road_camera" / cls / f"road_{event_id}.mov"
        driver_video = src_root / "face_camera" / cls / f"driver_{event_id}.mov"
        control = torch.tensor(entry["control"], dtype=torch.float32).unsqueeze(0)

        with tempfile.TemporaryDirectory() as tmp_road, tempfile.TemporaryDirectory() as tmp_driver:
            road_frames = extract_frames(road_video, args.num_frames, tmp_road)
            driver_frames = extract_frames(driver_video, args.num_frames, tmp_driver)
            n = min(len(road_frames), len(driver_frames))
            print(f"{event_id}: {n} frame pairs")

            overlay_frames, driver_out_frames = [], []
            for i in range(n):
                road_pil = Image.open(road_frames[i]).convert("RGB")
                driver_pil = Image.open(driver_frames[i]).convert("RGB")

                road_pv = processor(images=road_pil, return_tensors="pt")["pixel_values"]
                driver_pv = processor(images=driver_pil, return_tensors="pt")["pixel_values"]

                with torch.no_grad():
                    out = model(road_pv, driver_pv, control, texts=[""])

                heatmap = out["gaze_heatmap"][0]
                small_road = road_pil.resize((args.frame_size, int(args.frame_size * road_pil.height / road_pil.width)))
                overlay = overlay_heatmap(small_road, heatmap)
                overlay_frames.append(overlay)

                small_driver = driver_pil.resize((args.frame_size, int(args.frame_size * driver_pil.height / driver_pil.width)))
                driver_out_frames.append(small_driver)

        results.append({
            "event_id": event_id,
            "maneuver": cls,
            "proxy_label": "distracted" if entry["label"] == 1 else "attentive",
            "down_ratio": entry.get("_down_ratio"),
            "road_gif_uri": to_gif_data_uri(overlay_frames, duration_ms=int(5000 / n)),
            "driver_gif_uri": to_gif_data_uri(driver_out_frames, duration_ms=int(5000 / n)),
        })
        print(f"  done: {event_id}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} animated examples to {args.out_json}")


if __name__ == "__main__":
    main()
