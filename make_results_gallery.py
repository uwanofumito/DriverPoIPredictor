"""Run the trained model over a handful of real ETS2 events and dump
per-example results (predicted class probs, predicted gaze heatmap overlay,
ground-truth proxy gaze point) as PNGs + a JSON summary, for building a
results artifact.
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from config import DriverStateConfig
from models.driver_state_model import DriverStateModel

CLASS_NAMES = ["attentive", "distracted"]


def overlay_heatmap(base_image, heatmap, alpha=0.55):
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(base_image.size, Image.BILINEAR)
    return Image.blend(base_image.convert("RGB"), colored_img, alpha)


def draw_marker(image, xy_norm, color=(0, 255, 0)):
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x, y = xy_norm[0] * w, xy_norm[1] * h
    r = 10
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)
    return img


def to_data_uri(image, fmt="JPEG", quality=85):
    import io

    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def pick_examples(manifest, max_per_maneuver=2):
    by_maneuver = {}
    for e in manifest:
        by_maneuver.setdefault(e["maneuver"], []).append(e)

    picked = []
    # always include every event with the distracted label (rare, interesting)
    picked += [e for e in manifest if e["label"] == 1]
    # plus a couple of attentive examples per maneuver class for variety
    for cls, events in by_maneuver.items():
        attentive = [e for e in events if e["label"] == 0]
        picked += attentive[:max_per_maneuver]

    # de-dup while preserving order
    seen = set()
    unique = []
    for e in picked:
        key = e["road_image"]
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw/ets2/prepared/manifest.json")
    parser.add_argument("--checkpoint", default="outputs/driver_state_model.pt")
    parser.add_argument("--out-json", default="outputs/results_gallery.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data_root = manifest_path.parent
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    examples = pick_examples(manifest)
    print(f"Selected {len(examples)} examples ({sum(e['label'] for e in examples)} distracted)")

    config = DriverStateConfig(num_classes=2)
    model = DriverStateModel(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    results = []
    for e in examples:
        road_path = data_root / e["road_image"]
        driver_path = data_root / e["driver_image"]
        road_pil = Image.open(road_path).convert("RGB")
        driver_pil = Image.open(driver_path).convert("RGB")

        road_pv = processor(images=road_pil, return_tensors="pt")["pixel_values"]
        driver_pv = processor(images=driver_pil, return_tensors="pt")["pixel_values"]
        control = torch.tensor(e["control"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = model(road_pv, driver_pv, control, texts=[""])

        probs = torch.softmax(out["logits"], dim=-1)[0].tolist()
        heatmap = out["gaze_heatmap"][0]

        overlay = overlay_heatmap(road_pil, heatmap)
        if "gaze_target" in e:
            overlay = draw_marker(overlay, e["gaze_target"])

        results.append({
            "event_id": Path(e["road_image"]).stem.replace("_road", ""),
            "maneuver": e["maneuver"],
            "proxy_label": CLASS_NAMES[e["label"]],
            "down_ratio": e.get("_down_ratio"),
            "pred_probs": {CLASS_NAMES[i]: round(p, 3) for i, p in enumerate(probs)},
            "pred_label": CLASS_NAMES[int(np.argmax(probs))],
            "road_overlay_uri": to_data_uri(overlay),
            "driver_uri": to_data_uri(driver_pil.resize((200, 200))),
        })
        print(f"  {results[-1]['event_id']}: proxy={results[-1]['proxy_label']} pred={results[-1]['pred_label']} probs={results[-1]['pred_probs']}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} examples to {args.out_json}")


if __name__ == "__main__":
    main()
