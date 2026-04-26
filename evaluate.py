import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


EVAL_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Random view for image->image query so evaluation is not trivially identical to gallery vectors.
QUERY_IMAGE_AUGMENT = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@dataclass
class QuerySample:
    query_type: str
    query_text: str
    query_image_path: str
    target_image_path: str
    top_results: List[str]


class GalleryImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path]):
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return EVAL_IMAGE_TRANSFORM(image), idx


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_numpy_float32(tensor: torch.Tensor):
    return tensor.detach().cpu().contiguous().numpy().astype("float32", copy=False)


def image_id_from_name(name: str) -> int:
    # COCO files are like 000000123456.jpg -> 123456
    return int(Path(name).stem)


def load_coco_val(
    image_dir: Path,
    captions_json: Path,
    max_text_queries: int,
    max_image_queries: int,
    seed: int,
) -> Tuple[List[Path], Dict[int, int], List[Tuple[str, int]], List[int]]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not captions_json.exists() or not captions_json.is_file():
        raise FileNotFoundError(f"Captions json not found: {captions_json}")

    with captions_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    image_paths = sorted([p for p in image_dir.glob("*.jpg") if p.is_file()])
    if not image_paths:
        raise ValueError(f"No JPG images found in {image_dir}")

    image_id_to_gallery_idx: Dict[int, int] = {}
    for gallery_idx, path in enumerate(image_paths):
        image_id_to_gallery_idx[image_id_from_name(path.name)] = gallery_idx

    text_queries: List[Tuple[str, int]] = []
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id in image_id_to_gallery_idx:
            text_queries.append((ann["caption"], image_id_to_gallery_idx[image_id]))

    image_query_indices: List[int] = list(range(len(image_paths)))

    rng = random.Random(seed)
    if max_text_queries > 0 and len(text_queries) > max_text_queries:
        text_queries = rng.sample(text_queries, max_text_queries)
    if max_image_queries > 0 and len(image_query_indices) > max_image_queries:
        image_query_indices = rng.sample(image_query_indices, max_image_queries)

    return image_paths, image_id_to_gallery_idx, text_queries, image_query_indices


@torch.no_grad()
def encode_gallery(
    image_encoder: ImageEncoder,
    image_paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    image_encoder.eval()
    dataset = GalleryImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings: List[torch.Tensor] = []
    total = len(dataset)
    done = 0
    for batch_images, _ in loader:
        batch_images = batch_images.to(device)
        embed = F.normalize(image_encoder(batch_images), dim=1)
        all_embeddings.append(embed.cpu())
        done += batch_images.size(0)
        print(f"\rEncoding gallery: {done}/{total}", end="", flush=True)
    print()

    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def encode_text_queries(
    text_encoder: TextEncoder,
    tokenizer: DistilBertTokenizer,
    queries: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    text_encoder.eval()
    all_embeddings: List[torch.Tensor] = []

    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        tok = tokenizer(
            list(batch_queries),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        embed = F.normalize(text_encoder(input_ids, attention_mask), dim=1)
        all_embeddings.append(embed.cpu())

    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def encode_image_queries(
    image_encoder: ImageEncoder,
    image_paths: Sequence[Path],
    gallery_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    image_encoder.eval()
    query_tensors: List[torch.Tensor] = []

    for idx in gallery_indices:
        image = Image.open(image_paths[idx]).convert("RGB")
        query_tensors.append(QUERY_IMAGE_AUGMENT(image))

    all_embeddings: List[torch.Tensor] = []
    for start in range(0, len(query_tensors), batch_size):
        batch = torch.stack(query_tensors[start : start + batch_size], dim=0).to(device)
        embed = F.normalize(image_encoder(batch), dim=1)
        all_embeddings.append(embed.cpu())

    return torch.cat(all_embeddings, dim=0)


def build_faiss_index(embeddings: torch.Tensor) -> faiss.Index:
    vecs = to_numpy_float32(embeddings)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def compute_ranks(
    query_embeddings: torch.Tensor,
    target_gallery_indices: Sequence[int],
    gallery_index: faiss.Index,
    max_k: int,
) -> List[int]:
    q = to_numpy_float32(query_embeddings)
    faiss.normalize_L2(q)

    dists, indices = gallery_index.search(q, min(max_k, gallery_index.ntotal))
    _ = dists

    ranks: List[int] = []
    for row, target_idx in enumerate(target_gallery_indices):
        retrieved = indices[row].tolist()
        if target_idx in retrieved:
            ranks.append(retrieved.index(target_idx) + 1)
        else:
            ranks.append(gallery_index.ntotal + 1)
    return ranks


def metrics_from_ranks(ranks: Sequence[int], k_values: Sequence[int], gallery_size: int) -> Dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {}

    out: Dict[str, float] = {}
    for k in k_values:
        recall = sum(1 for r in ranks if r <= k) / n
        precision = sum((1.0 / k) for r in ranks if r <= k) / n
        out[f"R@{k}"] = recall
        out[f"P@{k}"] = precision

    out["MRR"] = sum((1.0 / r) if r <= gallery_size else 0.0 for r in ranks) / n
    out["mAP"] = out["MRR"]  # Single ground-truth target per query in this setup.
    return out


def pretty_metrics(title: str, metrics: Dict[str, float], k_values: Sequence[int]) -> str:
    lines = [f"\n{title}"]
    for k in k_values:
        lines.append(f"  R@{k}: {metrics[f'R@{k}']:.4f}")
    lines.append(f"  MRR : {metrics['MRR']:.4f}")
    lines.append(f"  mAP : {metrics['mAP']:.4f}")
    for k in k_values:
        lines.append(f"  P@{k}: {metrics[f'P@{k}']:.4f}")
    return "\n".join(lines)


def collect_samples_text_to_image(
    query_texts: Sequence[str],
    target_indices: Sequence[int],
    gallery_paths: Sequence[Path],
    query_embeddings: torch.Tensor,
    gallery_index: faiss.Index,
    num_samples: int,
    k: int,
    seed: int,
) -> List[QuerySample]:
    if num_samples <= 0 or len(query_texts) == 0:
        return []

    rng = random.Random(seed)
    picked = rng.sample(range(len(query_texts)), min(num_samples, len(query_texts)))

    q = to_numpy_float32(query_embeddings[picked])
    faiss.normalize_L2(q)
    _, indices = gallery_index.search(q, min(k, gallery_index.ntotal))

    samples: List[QuerySample] = []
    for row, q_idx in enumerate(picked):
        target_idx = target_indices[q_idx]
        top_results = [str(gallery_paths[i]) for i in indices[row].tolist()]
        samples.append(
            QuerySample(
                query_type="text_to_image",
                query_text=query_texts[q_idx],
                query_image_path="",
                target_image_path=str(gallery_paths[target_idx]),
                top_results=top_results,
            )
        )
    return samples


def collect_samples_image_to_image(
    query_gallery_indices: Sequence[int],
    gallery_paths: Sequence[Path],
    query_embeddings: torch.Tensor,
    gallery_index: faiss.Index,
    num_samples: int,
    k: int,
    seed: int,
) -> List[QuerySample]:
    if num_samples <= 0 or len(query_gallery_indices) == 0:
        return []

    rng = random.Random(seed + 101)
    picked = rng.sample(range(len(query_gallery_indices)), min(num_samples, len(query_gallery_indices)))

    q = to_numpy_float32(query_embeddings[picked])
    faiss.normalize_L2(q)
    _, indices = gallery_index.search(q, min(k, gallery_index.ntotal))

    samples: List[QuerySample] = []
    for row, local_idx in enumerate(picked):
        gallery_idx = query_gallery_indices[local_idx]
        top_results = [str(gallery_paths[i]) for i in indices[row].tolist()]
        samples.append(
            QuerySample(
                query_type="image_to_image",
                query_text="",
                query_image_path=str(gallery_paths[gallery_idx]),
                target_image_path=str(gallery_paths[gallery_idx]),
                top_results=top_results,
            )
        )
    return samples


def print_samples(samples: Sequence[QuerySample], title: str) -> None:
    if not samples:
        return

    print(f"\n{title}")
    for i, sample in enumerate(samples, start=1):
        print(f"\nSample {i} [{sample.query_type}]")
        if sample.query_text:
            print(f"  Query Text   : {sample.query_text}")
        if sample.query_image_path:
            print(f"  Query Image  : {sample.query_image_path}")
        print(f"  Target Image : {sample.target_image_path}")
        print("  Retrieved:")
        for rank, path in enumerate(sample.top_results, start=1):
            print(f"    {rank:>2}. {path}")


def save_json_report(
    out_path: Path,
    text_metrics: Dict[str, float],
    image_metrics: Dict[str, float],
    text_samples: Sequence[QuerySample],
    image_samples: Sequence[QuerySample],
) -> None:
    payload = {
        "text_to_image": text_metrics,
        "image_to_image": image_metrics,
        "samples": {
            "text_to_image": [sample.__dict__ for sample in text_samples],
            "image_to_image": [sample.__dict__ for sample in image_samples],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval model on COCO val2017")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--image_dir", default="data/coco/val2017", help="COCO val image directory")
    parser.add_argument(
        "--captions_json",
        default="data/coco/annotations/captions_val2017.json",
        help="COCO val captions annotations json",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="K values for Recall@K and Precision@K")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--max_text_queries", type=int, default=0, help="Limit text queries (0 means all)")
    parser.add_argument("--max_image_queries", type=int, default=0, help="Limit image queries (0 means all)")
    parser.add_argument("--num_samples", type=int, default=5, help="How many sample queries to print")
    parser.add_argument("--sample_top_k", type=int, default=10, help="Top-K to print for each sample query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_json", default="", help="Optional path to save metrics/samples json")
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    if any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive integers")

    cfg = load_config(Path(args.config))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    requested_device = cfg.get("training", {}).get("device", "cpu")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading model...")

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

    ckpt = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(ckpt["image_encoder"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    image_encoder.eval()
    text_encoder.eval()

    print("Loading COCO val metadata...")
    gallery_paths, _, text_queries, image_query_indices = load_coco_val(
        image_dir=Path(args.image_dir),
        captions_json=Path(args.captions_json),
        max_text_queries=args.max_text_queries,
        max_image_queries=args.max_image_queries,
        seed=args.seed,
    )

    print(f"Gallery size: {len(gallery_paths)}")
    print(f"Text queries: {len(text_queries)}")
    print(f"Image queries: {len(image_query_indices)}")

    print("Building gallery embeddings + FAISS index...")
    gallery_embeddings = encode_gallery(
        image_encoder=image_encoder,
        image_paths=gallery_paths,
        batch_size=args.batch_size,
        device=device,
    )
    gallery_index = build_faiss_index(gallery_embeddings)

    max_search_k = max(max(k_values), args.sample_top_k)

    print("Evaluating text -> image...")
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    query_texts = [t for t, _ in text_queries]
    text_targets = [idx for _, idx in text_queries]
    text_query_embeddings = encode_text_queries(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        queries=query_texts,
        max_length=cfg.get("data", {}).get("max_length", 77),
        batch_size=args.batch_size,
        device=device,
    )
    text_ranks = compute_ranks(
        query_embeddings=text_query_embeddings,
        target_gallery_indices=text_targets,
        gallery_index=gallery_index,
        max_k=max_search_k,
    )
    text_metrics = metrics_from_ranks(text_ranks, k_values, gallery_index.ntotal)

    print("Evaluating image -> image...")
    image_query_embeddings = encode_image_queries(
        image_encoder=image_encoder,
        image_paths=gallery_paths,
        gallery_indices=image_query_indices,
        batch_size=args.batch_size,
        device=device,
    )
    image_ranks = compute_ranks(
        query_embeddings=image_query_embeddings,
        target_gallery_indices=image_query_indices,
        gallery_index=gallery_index,
        max_k=max_search_k,
    )
    image_metrics = metrics_from_ranks(image_ranks, k_values, gallery_index.ntotal)

    print(pretty_metrics("Text -> Image", text_metrics, k_values))
    print(pretty_metrics("Image -> Image", image_metrics, k_values))

    text_samples = collect_samples_text_to_image(
        query_texts=query_texts,
        target_indices=text_targets,
        gallery_paths=gallery_paths,
        query_embeddings=text_query_embeddings,
        gallery_index=gallery_index,
        num_samples=args.num_samples,
        k=args.sample_top_k,
        seed=args.seed,
    )
    image_samples = collect_samples_image_to_image(
        query_gallery_indices=image_query_indices,
        gallery_paths=gallery_paths,
        query_embeddings=image_query_embeddings,
        gallery_index=gallery_index,
        num_samples=args.num_samples,
        k=args.sample_top_k,
        seed=args.seed,
    )

    print_samples(text_samples, "Manual Check Samples: Text -> Image")
    print_samples(image_samples, "Manual Check Samples: Image -> Image")

    if args.save_json:
        out_path = Path(args.save_json)
        save_json_report(
            out_path=out_path,
            text_metrics=text_metrics,
            image_metrics=image_metrics,
            text_samples=text_samples,
            image_samples=image_samples,
        )
        print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
