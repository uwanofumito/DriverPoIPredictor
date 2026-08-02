"""Train the standalone pedestrian-state classifier on a JAAD manifest.

    python train_pedestrian.py --manifest data/raw/jaad/prepared/manifest.json
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from data.jaad_dataset import JAADPedestrianDataset, collate_jaad
from models.pedestrian_state_model import LABEL_KEYS, PedestrianStateModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vision-model-path", default="facebook/dinov2-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--save-checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.vision_model_path)
    dataset = JAADPedestrianDataset(args.manifest, processor)

    n_val = int(len(dataset) * args.val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"train={n_train} val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_jaad)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_jaad)

    model = PedestrianStateModel(vision_model_path=args.vision_model_path).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    def run_epoch(loader, train):
        model.train(train)
        totals = {key: [0, 0] for key in LABEL_KEYS}  # [correct, n]
        total_loss, total_n = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, desc="train" if train else "val"):
                pixel_values = batch["pixel_values"].to(args.device)
                out = model(pixel_values)

                loss = sum(criterion(out[key], batch[key].to(args.device)) for key in LABEL_KEYS)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                bsz = pixel_values.size(0)
                total_loss += loss.item() * bsz
                total_n += bsz
                for key in LABEL_KEYS:
                    labels = batch[key].to(args.device)
                    totals[key][0] += (out[key].argmax(-1) == labels).sum().item()
                    totals[key][1] += bsz

        acc_str = " ".join(f"{key}_acc={c / n:.4f}" for key, (c, n) in totals.items())
        print(f"{'train' if train else 'val  '}: loss={total_loss / total_n:.4f} {acc_str}")

    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        run_epoch(train_loader, train=True)
        run_epoch(val_loader, train=False)

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
