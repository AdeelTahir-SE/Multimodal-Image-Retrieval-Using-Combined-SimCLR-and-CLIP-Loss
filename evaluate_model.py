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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.coco_dataset import COCOPairDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_checkpoint_path(checkpoint_arg: str, base_dir: Path) -> Path:
    """Resolve checkpoint path with a fallback to ./checkpoints/<name>."""
    direct = resolve_path(checkpoint_arg, base_dir)
    if direct.exists():
        return direct

    checkpoint_name = Path(checkpoint_arg)
    if checkpoint_name.parent == Path("."):
        fallback = (base_dir / "checkpoints" / checkpoint_name.name).resolve()
        if fallback.exists():
            return fallback

    return direct


def load_pairs(pairs_json: Path) -> List[dict]:
    if not pairs_json.exists():
        raise FileNotFoundError(
            "Pairs file not found: "
            f"{pairs_json}\n"
            "Generate it with:\n"
            "python data/preprocess.py --output data/coco_pairs.json"
        )

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
    compute_text: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        z_txt = text_encoder(input_ids, attention_mask) if compute_text else None

        image_query_embeds.append(F.normalize(z_img_q, dim=1).cpu())
        image_gallery_embeds.append(F.normalize(z_img_g, dim=1).cpu())
        if z_txt is not None:
            text_embeds.append(F.normalize(z_txt, dim=1).cpu())

    return (
        torch.cat(image_query_embeds, dim=0),
        torch.cat(image_gallery_embeds, dim=0),
        torch.cat(text_embeds, dim=0) if text_embeds else None,
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


def _mean_view(metric_agg: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {key: value["mean"] for key, value in metric_agg.items()}


def _std_view(metric_agg: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {key: value["std"] for key, value in metric_agg.items()}


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    pairs_path = resolve_path(cfg["data"]["pairs_json"], config_path.parent)
    pairs = load_pairs(pairs_path)

    if args.max_pairs is not None and args.max_pairs > 0 and args.max_pairs < len(pairs):
        random.seed(args.seed)
        sampled_indices = sorted(random.sample(range(len(pairs)), args.max_pairs))
        pairs = [pairs[i] for i in sampled_indices]
    else:
        sampled_indices = None

    positives = build_positive_indices(pairs)

    requested_device = cfg.get("training", {}).get("device", "cpu")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset = COCOPairDataset(
        pairs_json=str(pairs_path),
        max_length=cfg["data"].get("max_length", 77),
        tokenizer_name=cfg["model"]["text_backbone"],
        train=False,
    )

    if sampled_indices is not None:
        dataset = Subset(dataset, sampled_indices)

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

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, Path.cwd())
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Tried:\n"
            f"- {resolve_path(args.checkpoint, Path.cwd())}\n"
            f"- {(Path.cwd() / 'checkpoints' / Path(args.checkpoint).name).resolve()}"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    image_encoder.load_state_dict(checkpoint["image_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])

    ks = sorted(set(k for k in args.k if k > 0))
    if not ks:
        raise ValueError("--k must include at least one positive integer")

    i2i_runs: List[dict] = []
    t2i_runs: List[dict] = []
    txt_emb_cached: torch.Tensor | None = None

    total_passes = max(1, args.eval_passes)
    for pass_id in range(1, total_passes + 1):
        img_q_emb, img_g_emb, txt_emb = encode_embeddings(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dataloader=dataloader,
            device=device,
            pass_id=pass_id,
            compute_text=(txt_emb_cached is None),
        )

        if txt_emb_cached is None:
            if txt_emb is None:
                raise RuntimeError("Text embeddings were not computed in the first pass")
            txt_emb_cached = txt_emb

        sim_i2i = img_q_emb @ img_g_emb.t()
        sim_t2i = txt_emb_cached @ img_g_emb.t()

        i2i_runs.append(retrieval_metrics(sim_i2i, positives, ks))
        t2i_runs.append(retrieval_metrics(sim_t2i, positives, ks))

    i2i_agg = average_metric_dicts(i2i_runs)
    t2i_agg = average_metric_dicts(t2i_runs)

    i2i_mean = _mean_view(i2i_agg)
    t2i_mean = _mean_view(t2i_agg)

    print("\n=== Model Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Pairs: {len(pairs)}")
    print(f"Passes: {total_passes}")

    print("\n=== Retrieval Metrics (mean across passes) ===")
    print_metric_block("Image->Image", i2i_mean, ks)
    print_metric_block("Text->Image", t2i_mean, ks)

    print("\n=== Stability (std across passes) ===")
    for name, agg in [("Image->Image", i2i_agg), ("Text->Image", t2i_agg)]:
        std_text = ", ".join([f"R@{k} std={agg[f'R@{k}']['std'] * 100:.2f}%" for k in ks])
        print(f"{name}: {std_text}")

    verdict = summarize_quality(t2i_mean)
    print("\n=== Verdict (Text->Image) ===")
    print(verdict)

    return {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(device),
        "num_pairs": len(pairs),
        "ks": ks,
        "eval_passes": total_passes,
        "metrics": {
            "image_to_image": i2i_agg,
            "text_to_image": t2i_agg,
        },
        "means": {
            "image_to_image": i2i_mean,
            "text_to_image": t2i_mean,
        },
        "std": {
            "image_to_image": _std_view(i2i_agg),
            "text_to_image": _std_view(t2i_agg),
        },
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal retrieval checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="Recall cutoffs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--eval_passes", type=int, default=3, help="Number of evaluation passes")
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Evaluate on a random subset of pairs for faster runs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast sanity eval: eval_passes=1 and max_pairs=5000 unless explicitly set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_json", type=str, default="checkpoints/eval_model_report.json", help="Output JSON path")
    args = parser.parse_args()

    if args.quick:
        args.eval_passes = 1
        if args.max_pairs is None:
            args.max_pairs = 5000

    report = evaluate_checkpoint(args)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
