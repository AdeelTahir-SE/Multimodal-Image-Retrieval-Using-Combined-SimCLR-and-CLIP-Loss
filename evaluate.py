import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco_dataset import COCOPairDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pairs(pairs_json: str) -> List[dict]:
    with open(pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"pairs_json must contain a non-empty list: {pairs_json}")
    return pairs


def build_positive_indices(pairs: Sequence[dict]) -> List[List[int]]:
    image_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(pairs):
        image_to_indices[item["image_path"]].append(idx)
    return [image_to_indices[item["image_path"]] for item in pairs]


@torch.no_grad()
def encode_embeddings(
    image_encoder: ImageEncoder,
    text_encoder: TextEncoder,
    dataloader: DataLoader,
    device: torch.device,
    pass_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_query_embeds: List[torch.Tensor] = []
    image_gallery_embeds: List[torch.Tensor] = []
    text_embeds: List[torch.Tensor] = []

    image_encoder.eval()
    text_encoder.eval()

    for batch in tqdm(dataloader, desc=f"Encoding pass {pass_id}"):
        images_q = batch["image_aug1"].to(device)
        images_g = batch["image_aug2"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        z_img_q = image_encoder(images_q)
        z_img_g = image_encoder(images_g)
        z_txt = text_encoder(input_ids, attention_mask)

        image_query_embeds.append(F.normalize(z_img_q, dim=1).cpu())
        image_gallery_embeds.append(F.normalize(z_img_g, dim=1).cpu())
        text_embeds.append(F.normalize(z_txt, dim=1).cpu())

    return (
        torch.cat(image_query_embeds, dim=0),
        torch.cat(image_gallery_embeds, dim=0),
        torch.cat(text_embeds, dim=0),
    )


def random_hit_probability(num_gallery: int, num_positive: int, k: int) -> float:
    if num_positive <= 0:
        return 0.0
    k = min(k, num_gallery)
    miss_prob = 1.0
    for i in range(k):
        miss_prob *= (num_gallery - num_positive - i) / (num_gallery - i)
    return 1.0 - miss_prob


def retrieval_metrics(
    similarity: torch.Tensor,
    positives_per_query: Sequence[Sequence[int]],
    ks: Iterable[int],
) -> dict:
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks:
        raise ValueError("At least one positive K is required")

    num_queries, num_gallery = similarity.shape
    max_k = min(max(ks), num_gallery)
    topk = similarity.topk(max_k, dim=1).indices

    hit_counts = {k: 0 for k in ks}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []
    random_expected = {k: 0.0 for k in ks}

    for q_idx, positive_indices in enumerate(positives_per_query):
        pos_set = set(positive_indices)
        if not pos_set:
            continue

        top_list = topk[q_idx].tolist()
        for k in ks:
            k_eff = min(k, len(top_list))
            if any(idx in pos_set for idx in top_list[:k_eff]):
                hit_counts[k] += 1
            random_expected[k] += random_hit_probability(num_gallery, len(pos_set), k)

        pos_tensor = torch.tensor(list(pos_set), dtype=torch.long)
        row = similarity[q_idx]
        best_positive_score = row[pos_tensor].max()
        rank = int((row > best_positive_score).sum().item()) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

    denom = max(1, len(ranks))
    metrics = {
        "MRR": sum(reciprocal_ranks) / denom,
        "MedR": float(statistics.median(ranks)) if ranks else float("inf"),
        "MeanR": sum(ranks) / denom,
    }
    for k in ks:
        metrics[f"R@{k}"] = hit_counts[k] / denom
        metrics[f"RndR@{k}"] = random_expected[k] / denom
        base = metrics[f"RndR@{k}"]
        metrics[f"Lift@{k}"] = (metrics[f"R@{k}"] / base) if base > 0 else float("inf")
    return metrics


def print_metric_block(title: str, metrics: dict, ks: Sequence[int]) -> None:
    print(title)
    for k in ks:
        print(
            f"  R@{k}: {metrics[f'R@{k}'] * 100:.2f}% | "
            f"Random~{metrics[f'RndR@{k}'] * 100:.2f}% | "
            f"Lift x{metrics[f'Lift@{k}']:.2f}"
        )
    print(f"  MRR: {metrics['MRR']:.4f}")
    print(f"  MedR: {metrics['MedR']:.1f}")
    print(f"  MeanR: {metrics['MeanR']:.2f}")


def summarize_quality(text_to_image_metrics: dict) -> str:
    r1 = text_to_image_metrics.get("R@1", 0.0)
    r5 = text_to_image_metrics.get("R@5", text_to_image_metrics.get("R@1", 0.0))
    lift1 = text_to_image_metrics.get("Lift@1", 0.0)

    if r1 >= 0.25 and r5 >= 0.55 and lift1 >= 8.0:
        return "Strong: model is retrieving relevant images reliably."
    if r1 >= 0.12 and r5 >= 0.35 and lift1 >= 4.0:
        return "Promising: model learned alignment, but still misses many exact matches."
    if r1 >= 0.05 and r5 >= 0.20:
        return "Weak to moderate: better than chance, but quality is not production-ready."
    return "Poor: retrieval is close to random for strict top ranks."


def average_metric_dicts(metric_dicts: Sequence[dict]) -> dict:
    keys = metric_dicts[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metric_dicts]
        result[key] = {
            "mean": float(sum(values) / len(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }
    return result


def print_qualitative_examples(
    pairs: Sequence[dict],
    similarity: torch.Tensor,
    top_k: int,
    num_examples: int,
) -> None:
    if num_examples <= 0:
        return
    print("\nQualitative text->image samples")
    example_indices = random.sample(range(len(pairs)), min(num_examples, len(pairs)))
    k_eff = min(top_k, similarity.size(1))

    for rank_idx, idx in enumerate(example_indices, start=1):
        top_idx = similarity[idx].topk(k_eff).indices.tolist()
        query_caption = pairs[idx]["caption"]
        gt_image = pairs[idx]["image_path"]
        print(f"  [{rank_idx}] caption: {query_caption}")
        print(f"      gt: {gt_image}")
        for k, pred_i in enumerate(top_idx[:3], start=1):
            pred_path = pairs[pred_i]["image_path"]
            marker = "*" if pred_path == gt_image else " "
            print(f"      {marker} top{k}: {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with robust diagnostics")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="Recall cutoffs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    parser.add_argument(
        "--eval_passes",
        type=int,
        default=3,
        help="Number of evaluation passes to average (helps with stochastic augmentations)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="How many qualitative text->image examples to print",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    pairs = load_pairs(cfg["data"]["pairs_json"])
    positives = build_positive_indices(pairs)

    requested_device = cfg.get("training", {}).get("device", "cpu")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pairs: {len(pairs)} from {cfg['data']['pairs_json']}")

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

    ks = sorted(set(k for k in args.k if k > 0))
    if not ks:
        raise ValueError("--k must contain at least one positive integer")

    i2i_runs: List[dict] = []
    t2i_runs: List[dict] = []
    i2t_runs: List[dict] = []
    final_t2i_similarity = None

    for pass_id in range(1, max(1, args.eval_passes) + 1):
        img_q_emb, img_g_emb, txt_emb = encode_embeddings(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dataloader=dataloader,
            device=device,
            pass_id=pass_id,
        )

        sim_i2i = img_q_emb @ img_g_emb.t()
        sim_t2i = txt_emb @ img_g_emb.t()
        sim_i2t = img_q_emb @ txt_emb.t()
        final_t2i_similarity = sim_t2i

        i2i_runs.append(retrieval_metrics(sim_i2i, positives, ks))
        t2i_runs.append(retrieval_metrics(sim_t2i, positives, ks))
        i2t_runs.append(retrieval_metrics(sim_i2t, positives, ks))

    i2i_agg = average_metric_dicts(i2i_runs)
    t2i_agg = average_metric_dicts(t2i_runs)
    i2t_agg = average_metric_dicts(i2t_runs)

    i2i_mean = {k: v["mean"] for k, v in i2i_agg.items()}
    t2i_mean = {k: v["mean"] for k, v in t2i_agg.items()}
    i2t_mean = {k: v["mean"] for k, v in i2t_agg.items()}

    print("\n=== Retrieval Metrics (mean across passes) ===")
    print_metric_block("Image->Image", i2i_mean, ks)
    print_metric_block("Text->Image", t2i_mean, ks)
    print_metric_block("Image->Text", i2t_mean, ks)

    print("\n=== Stability (std across passes) ===")
    std_line = ", ".join([f"R@{k} std={t2i_agg[f'R@{k}']['std'] * 100:.2f}%" for k in ks])
    print(f"Text->Image: {std_line}")

    verdict = summarize_quality(t2i_mean)
    print("\n=== Verdict ===")
    print(verdict)

    if final_t2i_similarity is not None:
        print_qualitative_examples(
            pairs=pairs,
            similarity=final_t2i_similarity,
            top_k=max(ks),
            num_examples=args.num_examples,
        )

    if args.save_json:
        report = {
            "checkpoint": str(Path(args.checkpoint)),
            "config": str(Path(args.config)),
            "num_pairs": len(pairs),
            "ks": ks,
            "eval_passes": int(args.eval_passes),
            "image_to_image": i2i_agg,
            "text_to_image": t2i_agg,
            "image_to_text": i2t_agg,
            "verdict": verdict,
            "runs": {
                "image_to_image": i2i_runs,
                "text_to_image": t2i_runs,
                "image_to_text": i2t_runs,
            },
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation report: {out_path}")


if __name__ == "__main__":
    main()