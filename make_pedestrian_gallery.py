"""Run the trained pedestrian-state checkpoint over a sample of real JAAD
crops and dump predictions + images for a results gallery."""

import argparse
import base64
import io
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor

from models.pedestrian_state_model import LABEL_KEYS, PedestrianStateModel

LABEL_NAMES = {
    "look": ["not-looking", "looking"],
    "cross": ["not-crossing", "crossing"],
    "action": ["standing", "walking"],
}


def to_data_uri(image, fmt="JPEG", quality=85):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw/jaad/prepared/manifest.json")
    parser.add_argument("--checkpoint", default="outputs/pedestrian_state_model.pt")
    parser.add_argument("--vision-model-path", default="facebook/dinov2-base")
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--out-json", default="outputs/pedestrian_gallery.json")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rng = random.Random(0)
    # bias sample toward the rarer "looking" class so the gallery isn't all not-looking
    looking = [e for e in manifest if e["look"] == 1]
    not_looking = [e for e in manifest if e["look"] == 0]
    n_look = min(len(looking), args.num_examples // 2)
    examples = rng.sample(looking, n_look) + rng.sample(not_looking, args.num_examples - n_look)
    rng.shuffle(examples)

    processor = AutoImageProcessor.from_pretrained(args.vision_model_path)
    model = PedestrianStateModel(vision_model_path=args.vision_model_path)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    results = []
    for e in examples:
        image = Image.open(e["pedestrian_image"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            out = model(pixel_values)

        preds = {}
        for key in LABEL_KEYS:
            probs = torch.softmax(out[key], dim=-1)[0]
            pred_idx = int(probs.argmax())
            preds[key] = {
                "true": LABEL_NAMES[key][e[key]],
                "pred": LABEL_NAMES[key][pred_idx],
                "prob": round(float(probs[pred_idx]), 3),
                "correct": pred_idx == e[key],
            }

        results.append({
            "pedestrian_id": e["pedestrian_id"],
            "video": e["video"],
            "frame": e["frame"],
            "image_uri": to_data_uri(image),
            "predictions": preds,
        })
        print(f"{e['pedestrian_id']} ({e['video']}, frame {e['frame']}): "
              + ", ".join(f"{k}={v['pred']}({'✓' if v['correct'] else '✗'})" for k, v in preds.items()))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} examples to {args.out_json}")


if __name__ == "__main__":
    main()
