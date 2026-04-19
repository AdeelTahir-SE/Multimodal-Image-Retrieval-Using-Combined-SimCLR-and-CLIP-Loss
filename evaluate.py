import argparse
from typing import Iterable, List

import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_dataset import COCOPairDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


@torch.no_grad()
def encode_embeddings(image_encoder, text_encoder, dataloader, device):
    image_embeds: List[torch.Tensor] = []
    text_embeds: List[torch.Tensor] = []

    image_encoder.eval()
    text_encoder.eval()

    for batch in tqdm(dataloader, desc="Encoding"):
        images = batch["image_aug1"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        z_img = image_encoder(images)
        z_txt = text_encoder(input_ids, attention_mask)

        image_embeds.append(F.normalize(z_img, dim=1).cpu())
        text_embeds.append(F.normalize(z_txt, dim=1).cpu())

    return torch.cat(image_embeds, dim=0), torch.cat(text_embeds, dim=0)


def recall_at_k(similarity: torch.Tensor, ks: Iterable[int]) -> dict:
    n = similarity.size(0)
    labels = torch.arange(n)
    result = {}

    for k in ks:
        topk = similarity.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        result[f"R@{k}"] = correct

    return result


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal retrieval with Recall@K")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="Recall cutoffs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    dataset = COCOPairDataset(
        pairs_json=cfg["data"]["pairs_json"],
        max_length=cfg["data"].get("max_length", 77),
        tokenizer_name=cfg["model"]["text_backbone"],
        train=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    image_encoder = ImageEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        backbone_name=cfg["model"]["image_backbone"],
        use_pretrained=cfg["model"].get("image_pretrained", False),
    ).to(device)

    text_encoder = TextEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        model_name=cfg["model"]["text_backbone"],
        use_pretrained=cfg["model"].get("text_pretrained", False),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_encoder.load_state_dict(checkpoint["image_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])

    img_emb, txt_emb = encode_embeddings(image_encoder, text_encoder, dataloader, device)

    sim_i2i = img_emb @ img_emb.t()
    sim_t2i = txt_emb @ img_emb.t()

    # Exclude each image itself for image-to-image retrieval.
    sim_i2i.fill_diagonal_(-1e9)

    i2i_metrics = recall_at_k(sim_i2i, args.k)
    t2i_metrics = recall_at_k(sim_t2i, args.k)

    print("Image->Image retrieval")
    for key, value in i2i_metrics.items():
        print(f"  {key}: {value * 100:.2f}%")

    print("Text->Image retrieval")
    for key, value in t2i_metrics.items():
        print(f"  {key}: {value * 100:.2f}%")


if __name__ == "__main__":
    main()