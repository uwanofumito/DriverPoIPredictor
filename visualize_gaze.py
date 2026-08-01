"""Visualize the (untrained) model's gaze-attention heatmap on a synthetic
road scene, and its class prediction. Pipeline sanity-check only — weights
are randomly initialized except for the frozen pretrained backbones, so the
heatmap pattern reflects init, not learned driver behavior. Re-run this
after training on real data to see something meaningful.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

from config import DriverStateConfig
from models.driver_state_model import DriverStateModel

CLASS_NAMES = ["alert", "drowsy", "distracted", "cognitively_loaded"]


def make_synthetic_road_image(size=(512, 512)):
    """Simple hand-drawn road scene: sky, converging lane lines, a car ahead."""
    w, h = size
    img = Image.new("RGB", size, (135, 206, 235))  # sky
    draw = ImageDraw.Draw(img)
    horizon = h * 0.45
    draw.rectangle([0, horizon, w, h], fill=(90, 90, 95))  # road
    vp = (w / 2, horizon)
    draw.polygon([vp, (w * 0.05, h), (w * 0.35, h)], fill=(60, 60, 65))
    draw.polygon([vp, (w * 0.65, h), (w * 0.95, h)], fill=(60, 60, 65))
    for t in np.linspace(0.5, 0.95, 5):
        y = horizon + (h - horizon) * t
        x0 = vp[0] - 4 * t * 10
        x1 = vp[0] + 4 * t * 10
        draw.rectangle([x0, y, x1, y + 6], fill=(230, 230, 200))
    car_x, car_y, car_w, car_h = w * 0.44, h * 0.72, w * 0.12, h * 0.08
    draw.rectangle([car_x, car_y, car_x + car_w, car_y + car_h], fill=(180, 30, 30))
    return img


def make_synthetic_driver_image(size=(512, 512)):
    """Placeholder driver-facing frame: plain face-like oval, nothing trained on this."""
    img = Image.new("RGB", size, (40, 40, 45))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.ellipse([w * 0.3, h * 0.2, w * 0.7, h * 0.75], fill=(210, 170, 140))
    draw.ellipse([w * 0.38, h * 0.4, w * 0.46, h * 0.46], fill=(30, 30, 30))
    draw.ellipse([w * 0.54, h * 0.4, w * 0.62, h * 0.46], fill=(30, 30, 30))
    return img


def overlay_heatmap(base_image, heatmap, alpha=0.55):
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(base_image.size, Image.BILINEAR)
    return Image.blend(base_image.convert("RGB"), colored_img, alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a trained state_dict; omit to use randomly-initialized heads")
    parser.add_argument("--out", default="outputs/gaze_visualization.png")
    args = parser.parse_args()

    config = DriverStateConfig()
    model = DriverStateModel(config)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    road_pil = make_synthetic_road_image()
    driver_pil = make_synthetic_driver_image()

    road_pixel_values = processor(images=road_pil, return_tensors="pt")["pixel_values"]
    driver_pixel_values = processor(images=driver_pil, return_tensors="pt")["pixel_values"]
    control_series = torch.zeros(1, 1, config.control_input_dim)  # neutral control snapshot

    with torch.no_grad():
        out = model(road_pixel_values, driver_pixel_values, control_series, texts=[""])

    probs = torch.softmax(out["logits"], dim=-1)[0]
    pred_idx = probs.argmax().item()
    heatmap = out["gaze_heatmap"][0]  # [16, 16]

    overlay = overlay_heatmap(road_pil, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(road_pil)
    axes[0].set_title("Road (synthetic)")
    axes[1].imshow(driver_pil)
    axes[1].set_title("Driver (synthetic)")
    axes[2].imshow(overlay)
    axes[2].set_title("Gaze attention overlay (untrained)")
    for ax in axes:
        ax.axis("off")

    class_probs_str = "\n".join(
        f"{name}: {p:.3f}" for name, p in zip(CLASS_NAMES[: config.num_classes], probs.tolist())
    )
    fig.suptitle(
        f"predicted class: {CLASS_NAMES[pred_idx]}   |   probs:\n{class_probs_str}",
        fontsize=10,
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
