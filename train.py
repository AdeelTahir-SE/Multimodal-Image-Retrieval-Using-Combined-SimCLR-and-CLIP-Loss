import argparse
import json
import os
from dataclasses import dataclass

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_dataset import COCOPairDataset
from losses.clip_loss import clip_loss
from losses.simclr_loss import simclr_loss
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


@dataclass
class TrainingConfig:
    pairs_json: str
    batch_size: int
    num_workers: int
    embedding_dim: int
    image_backbone: str
    text_backbone: str
    image_pretrained: bool
    text_pretrained: bool
    alpha: float
    beta: float
    simclr_temperature: float
    clip_temperature: float
    epochs: int
    lr: float
    weight_decay: float
    checkpoint_dir: str
    device: str
    max_length: int



def load_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return TrainingConfig(
        pairs_json=cfg["data"]["pairs_json"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        embedding_dim=cfg["model"]["embedding_dim"],
        image_backbone=cfg["model"]["image_backbone"],
        text_backbone=cfg["model"]["text_backbone"],
        image_pretrained=cfg["model"].get("image_pretrained", False),
        text_pretrained=cfg["model"].get("text_pretrained", False),
        alpha=cfg["loss"]["alpha"],
        beta=cfg["loss"]["beta"],
        simclr_temperature=cfg["loss"].get("simclr_temperature", 0.07),
        clip_temperature=cfg["loss"].get("clip_temperature", 0.07),
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        device=cfg["training"].get("device", "cuda"),
        max_length=cfg["data"].get("max_length", 77),
    )


def save_checkpoint(path: str, epoch: int, image_encoder, text_encoder, optimizer, loss_value: float):
    payload = {
        "epoch": epoch,
        "image_encoder": image_encoder.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss_value,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimCLR+CLIP multimodal model")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = COCOPairDataset(
        pairs_json=cfg.pairs_json,
        max_length=cfg.max_length,
        tokenizer_name=cfg.text_backbone,
        train=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    image_encoder = ImageEncoder(
        embedding_dim=cfg.embedding_dim,
        backbone_name=cfg.image_backbone,
        use_pretrained=cfg.image_pretrained,
    ).to(device)

    text_encoder = TextEncoder(
        embedding_dim=cfg.embedding_dim,
        model_name=cfg.text_backbone,
        use_pretrained=cfg.text_pretrained,
    ).to(device)

    optimizer = AdamW(
        list(image_encoder.parameters()) + list(text_encoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        image_encoder.train()
        text_encoder.train()

        running_loss = 0.0
        running_simclr = 0.0
        running_clip = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in progress:
            aug1 = batch["image_aug1"].to(device)
            aug2 = batch["image_aug2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            z_i = image_encoder(aug1)
            z_j = image_encoder(aug2)
            z_txt = text_encoder(input_ids, attention_mask)

            l_simclr = simclr_loss(z_i, z_j, temperature=cfg.simclr_temperature)
            l_clip = clip_loss(z_i, z_txt, temperature=cfg.clip_temperature)
            loss = cfg.alpha * l_simclr + cfg.beta * l_clip

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_simclr += l_simclr.item()
            running_clip += l_clip.item()

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "simclr": f"{l_simclr.item():.4f}",
                    "clip": f"{l_clip.item():.4f}",
                }
            )

        epoch_loss = running_loss / len(dataloader)
        epoch_simclr = running_simclr / len(dataloader)
        epoch_clip = running_clip / len(dataloader)

        print(
            f"Epoch {epoch + 1}: loss={epoch_loss:.4f}, "
            f"simclr={epoch_simclr:.4f}, clip={epoch_clip:.4f}"
        )

        latest_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
        save_checkpoint(latest_path, epoch + 1, image_encoder, text_encoder, optimizer, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            save_checkpoint(best_path, epoch + 1, image_encoder, text_encoder, optimizer, epoch_loss)

        metrics_path = os.path.join(cfg.checkpoint_dir, "train_metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "simclr_loss": epoch_simclr,
                        "clip_loss": epoch_clip,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()