"""Training loop for the driver-state model.

Defaults to synthetic mock data so the pipeline can be verified without a
real dataset. Pass --manifest to train on real data once you have a manifest
(see data/dataset.py for the expected schema, and
data/prepare_ets2_manifest.py for a worked preprocessing example).
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DriverStateConfig
from data.dataset import ManifestDataset, MockDriverStateDataset, collate_driver_state
from models.driver_state_model import DriverStateModel
from models.gaze_head import gaze_target_heatmap


def build_dataloader(args, config):
    if args.manifest is None:
        dataset = MockDriverStateDataset(
            num_samples=args.mock_samples,
            image_size=(3, args.image_size, args.image_size),
            control_input_dim=config.control_input_dim,
            num_classes=config.num_classes,
        )
    else:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(config.vision_model_path)
        dataset = ManifestDataset(args.manifest, processor, config.control_input_dim)

    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_driver_state
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a manifest JSON (see data/dataset.py). Omit to train on synthetic mock data.",
    )
    parser.add_argument("--mock-samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=4, help="4 for the driver-state taxonomy; 5 for ETS2's maneuver labels")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-text", action="store_true", help="Disable the optional text branch")
    parser.add_argument("--gaze-loss-weight", type=float, default=0.5, help="Weight for the gaze_target auxiliary loss, when the manifest provides one")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = DriverStateConfig(use_text=not args.no_text, num_classes=args.num_classes)
    dataloader = build_dataloader(args, config)

    model = DriverStateModel(config).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss, total_cls_loss, total_gaze_loss, total_correct, total_n = 0.0, 0.0, 0.0, 0, 0
        for batch in tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            road = batch["road_pixel_values"].to(args.device)
            driver = batch["driver_pixel_values"].to(args.device)
            control = batch["control_series"].to(args.device)
            labels = batch["labels"].to(args.device)
            texts = batch["texts"] if config.use_text else None

            out = model(road, driver, control, texts=texts)
            cls_loss = criterion(out["logits"], labels)
            loss = cls_loss

            gaze_loss = None
            if config.use_gaze_head and "gaze_targets" in batch:
                gaze_targets = batch["gaze_targets"].to(args.device)
                side = out["gaze_heatmap"].shape[-1]
                target_heatmap = gaze_target_heatmap(gaze_targets, side)
                gaze_loss = -(target_heatmap * torch.log(out["gaze_heatmap"].clamp_min(1e-8))).sum(dim=(1, 2)).mean()
                loss = loss + args.gaze_loss_weight * gaze_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_cls_loss += cls_loss.item() * labels.size(0)
            if gaze_loss is not None:
                total_gaze_loss += gaze_loss.item() * labels.size(0)
            total_correct += (out["logits"].argmax(-1) == labels).sum().item()
            total_n += labels.size(0)

        msg = (
            f"epoch {epoch + 1}: loss={total_loss / total_n:.4f} "
            f"cls_loss={total_cls_loss / total_n:.4f} acc={total_correct / total_n:.4f}"
        )
        if total_gaze_loss:
            msg += f" gaze_loss={total_gaze_loss / total_n:.4f}"
        print(msg)


if __name__ == "__main__":
    main()
