"""
evaluate.py  –  Semantic evaluation of the image-retrieval model on COCO val2017.

Metrics computed
----------------
1. Average CLIP Score @ K   – cosine similarity between text query and
                               retrieved images using a *frozen* openai/clip-vit-base-patch32.
2. Semantic NDCG @ K        – NDCG with graded relevance derived from
                               CLIP image-image similarity to the ground-truth image.
3. Soft Recall @ K          – fraction of queries where ≥1 retrieved image
                               has CLIP image similarity ≥ epsilon to the ground truth.
4. Standard Recall/Precision/MRR @ K (exact-match baseline).

Usage
-----
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --val_images  data/coco/val2017 \
    --captions    data/coco/annotations/captions_val2017.json \
    --config      config.yaml \
    --top_k       5 \
    --num_queries 1000 \
    --batch_size  64 \
    --out         checkpoints/eval_semantic.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

# ── project imports ──────────────────────────────────────────────────────────
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# CLIP evaluator uses its own normalisation
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Tiny dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
class PathImageDataset(Dataset):
    """Return (tensor, path_str) for a list of image paths."""
    def __init__(self, paths: List[Path], tfm):
        self.paths = paths
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().float().numpy()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXT)


def load_val_pairs(captions_json: Path, val_dir: Path,
                   num_queries: int, seed: int = 42) -> List[Dict]:
    """
    Returns a list of dicts: {image_id, file_name, caption}
    One caption per image (first caption encountered), sampled randomly.
    """
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    seen: set = set()
    pairs: List[Dict] = []

    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid in seen:
            continue
        fname = id2file.get(iid)
        if fname is None:
            continue
        fpath = val_dir / fname
        if not fpath.exists():
            continue
        pairs.append({"image_id": iid, "file_name": fname,
                       "path": fpath, "caption": ann["caption"]})
        seen.add(iid)

    random.seed(seed)
    random.shuffle(pairs)
    return pairs[:num_queries]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_retrieval_model(checkpoint: Path, cfg: dict, device: torch.device):
    img_enc = ImageEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        backbone_name=cfg["model"]["image_backbone"],
        use_pretrained=cfg["model"].get("image_pretrained", False),
    ).to(device)
    txt_enc = TextEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        model_name=cfg["model"]["text_backbone"],
        use_pretrained=cfg["model"].get("text_pretrained", False),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    img_enc.load_state_dict(payload["image_encoder"])
    txt_enc.load_state_dict(payload["text_encoder"])
    img_enc.eval()
    txt_enc.eval()
    return img_enc, txt_enc


def load_clip_evaluator(device: torch.device):
    """Load a frozen OpenAI CLIP (ViT-B/32) as the evaluator."""
    try:
        import clip  # openai/clip
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        print("[Evaluator] Loaded openai/clip ViT-B/32 via `clip` package.")
        return model, "openai_clip"
    except ImportError:
        pass

    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        print("[Evaluator] Loaded openai/clip-vit-base-patch32 via HuggingFace.")
        return (model, processor), "hf_clip"
    except Exception as e:
        raise RuntimeError(
            "Could not load CLIP evaluator. Install either `clip` (pip install git+https://github.com/openai/CLIP.git) "
            "or `transformers` with `openai/clip-vit-base-patch32`.\n" + str(e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# FAISS gallery builder
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_faiss_index(
    img_enc: ImageEncoder,
    image_paths: List[Path],
    batch_size: int,
    device: torch.device,
    cache_index: Path | None = None,
    cache_meta: Path | None = None,
) -> Tuple[faiss.Index, List[str]]:
    """Encode all val images and return a FAISS inner-product index."""
    # Try loading from cache
    if cache_index and cache_meta and cache_index.exists() and cache_meta.exists():
        print(f"[FAISS] Loading cached index from {cache_index}")
        index = faiss.read_index(str(cache_index))
        with open(cache_meta, "r", encoding="utf-8") as f:
            paths = json.load(f)["paths"]
        return index, paths

    print(f"[FAISS] Encoding {len(image_paths)} images …")
    dataset = PathImageDataset(image_paths, RETRIEVAL_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embs: List[np.ndarray] = []
    all_paths: List[str] = []
    img_enc.eval()

    for i, (imgs, paths) in enumerate(loader):
        imgs = imgs.to(device)
        emb  = F.normalize(img_enc(imgs), dim=1)
        all_embs.append(_to_f32(emb))
        all_paths.extend(paths)
        if (i + 1) % 20 == 0:
            print(f"  {len(all_paths)}/{len(image_paths)} images encoded", end="\r")

    print(f"  {len(all_paths)}/{len(image_paths)} images encoded")

    gallery = np.vstack(all_embs).astype("float32")
    faiss.normalize_L2(gallery)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)

    if cache_index and cache_meta:
        cache_index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(cache_index))
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"paths": all_paths}, f, indent=2)
        print(f"[FAISS] Index saved → {cache_index}")

    return index, all_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLIP evaluator helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def clip_image_embed(image_paths: List[Path], clip_pkg, clip_type: str,
                     device: torch.device) -> torch.Tensor:
    """Return L2-normalised CLIP image embeddings, shape (N, D)."""
    if clip_type == "openai_clip":
        import clip
        tensors = torch.stack([
            CLIP_TRANSFORM(Image.open(p).convert("RGB")) for p in image_paths
        ]).to(device)
        feats = clip_pkg.encode_image(tensors).float()
    else:
        model, processor = clip_pkg
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Use vision_model + visual_projection directly to always get a tensor
        vision_out = model.vision_model(**inputs)
        pooled = vision_out.pooler_output  # (N, hidden)
        feats = model.visual_projection(pooled).float()
    return F.normalize(feats, dim=1)


@torch.no_grad()
def clip_text_embed(texts: List[str], clip_pkg, clip_type: str,
                    device: torch.device) -> torch.Tensor:
    """Return L2-normalised CLIP text embeddings, shape (N, D)."""
    if clip_type == "openai_clip":
        import clip
        tokens = clip.tokenize(texts, truncate=True).to(device)
        feats = clip_pkg.encode_text(tokens).float()
    else:
        model, processor = clip_pkg
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        # Keep only text-relevant keys to avoid unexpected-argument errors
        text_keys = {"input_ids", "attention_mask"}
        inputs = {k: v.to(device) for k, v in inputs.items() if k in text_keys}
        # Use text_model + text_projection directly to always get a plain tensor
        text_out = model.text_model(**inputs)
        pooled = text_out.pooler_output  # (N, hidden)
        feats = model.text_projection(pooled).float()
    return F.normalize(feats, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Compute NDCG@K given a list of graded relevance scores (in rank order)."""
    top = relevances[:k]
    dcg  = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(top))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(hit_ranks: List[int | None], k: int) -> float:
    """Mean Reciprocal Rank: hit_ranks contains the 1-based rank of the hit, or None."""
    rrs = []
    for r in hit_ranks:
        if r is not None and r <= k:
            rrs.append(1.0 / r)
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(args):
    random.seed(42)
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. Load retrieval model ──────────────────────────────────────────────
    print("[Model] Loading retrieval model from", args.checkpoint)
    checkpoint = Path(args.checkpoint)
    img_enc, txt_enc = load_retrieval_model(checkpoint, cfg, device)
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    max_len   = cfg["data"].get("max_length", 77)

    # ── 2. Load CLIP evaluator ───────────────────────────────────────────────
    print("[Evaluator] Loading frozen CLIP …")
    clip_pkg, clip_type = load_clip_evaluator(device)

    # ── 3. Prepare val pairs ─────────────────────────────────────────────────
    val_dir      = Path(args.val_images)
    captions_ann = Path(args.captions)
    print(f"[Data] Loading captions from {captions_ann}")
    pairs = load_val_pairs(captions_ann, val_dir, args.num_queries)
    print(f"[Data] {len(pairs)} query pairs selected")

    # ── 4. Build FAISS gallery from ALL val images ───────────────────────────
    all_val_images = discover_images(val_dir)
    print(f"[Gallery] {len(all_val_images)} val images found")

    cache_dir   = Path(args.cache_dir)
    cache_index = cache_dir / "val2017_gallery.index"
    cache_meta  = cache_dir / "val2017_gallery.json"
    gallery_index, gallery_paths = build_faiss_index(
        img_enc, all_val_images, args.batch_size, device, cache_index, cache_meta
    )
    path2idx = {p: i for i, p in enumerate(gallery_paths)}

    # ── 5. Evaluation ────────────────────────────────────────────────────────
    K = args.top_k
    eps = args.soft_recall_eps

    # accumulators
    clip_scores_at_k:   List[float] = []   # Metric 1
    ndcg_at_k_scores:   List[float] = []   # Metric 2
    soft_recall_hits:   List[float] = []   # Metric 3
    exact_recall_at_k:  List[float] = []   # Metric 4 (standard)
    exact_precision_at_k: List[float] = []
    mrr_ranks:          List[int | None] = []

    print(f"\n[Eval] Running over {len(pairs)} queries (top_k={K}) …\n")

    # Process in small CLIP batches to avoid OOM
    CLIP_BATCH = 16

    for qi, pair in enumerate(pairs):
        caption   = pair["caption"]
        gt_path   = str(pair["path"])
        gt_path_p = Path(gt_path)

        # ── Text → FAISS retrieval ──
        tok = tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok["attention_mask"].to(device)
        txt_emb   = F.normalize(txt_enc(input_ids, attn_mask), dim=1)
        q_np      = _to_f32(txt_emb).reshape(1, -1)
        faiss.normalize_L2(q_np)
        _, topk_idxs = gallery_index.search(q_np, min(K, gallery_index.ntotal))
        topk_idxs = topk_idxs[0].tolist()

        retrieved_paths = [Path(gallery_paths[i]) for i in topk_idxs]

        # ── METRIC 4: Standard Exact Recall/Precision/MRR ──
        hit_rank = None
        exact_hits = 0
        for rank, rp in enumerate(retrieved_paths, 1):
            if str(rp) == gt_path or rp.name == gt_path_p.name:
                if hit_rank is None:
                    hit_rank = rank
                exact_hits += 1
        exact_recall_at_k.append(1.0 if hit_rank is not None else 0.0)
        exact_precision_at_k.append(exact_hits / K)
        mrr_ranks.append(hit_rank)

        # ── METRIC 1: Average CLIP Score @ K ──
        # Cosine similarity between text and each retrieved image via frozen CLIP
        txt_clip = clip_text_embed([caption], clip_pkg, clip_type, device)  # (1, D)
        imgs_clip = clip_image_embed(retrieved_paths, clip_pkg, clip_type, device)  # (K, D)
        sim_scores = (txt_clip @ imgs_clip.T).squeeze(0)  # (K,)
        avg_clip_score = sim_scores.mean().item()
        clip_scores_at_k.append(avg_clip_score)

        # ── METRIC 2: Semantic NDCG @ K ──
        # Relevance = CLIP image-image cosine similarity between retrieved and GT
        gt_clip = clip_image_embed([gt_path_p], clip_pkg, clip_type, device)  # (1, D)
        img_img_sim = (gt_clip @ imgs_clip.T).squeeze(0)  # (K,)
        # Shift from [-1,1] → [0,1] for non-negative relevance
        relevances = ((img_img_sim + 1.0) / 2.0).cpu().tolist()
        ndcg_at_k_scores.append(ndcg_at_k(relevances, K))

        # ── METRIC 3: Soft Recall @ K ──
        # Hit if any retrieved image is within eps cosine similarity of GT
        max_sim = img_img_sim.max().item()
        # Convert eps (threshold on [0,1] scale) back: sim in original [-1,1] scale
        # eps is on [0,1] scale after the shift, so threshold = 2*eps - 1
        threshold = 2.0 * eps - 1.0
        soft_recall_hits.append(1.0 if max_sim >= threshold else 0.0)

        if (qi + 1) % 50 == 0 or (qi + 1) == len(pairs):
            print(f"  [{qi+1:4d}/{len(pairs)}] "
                  f"CLIP@{K}={np.mean(clip_scores_at_k):.4f}  "
                  f"NDCG@{K}={np.mean(ndcg_at_k_scores):.4f}  "
                  f"SoftR@{K}={np.mean(soft_recall_hits):.4f}  "
                  f"R@{K}={np.mean(exact_recall_at_k):.4f}")

    # ── 6. Aggregate ─────────────────────────────────────────────────────────
    results = {
        "num_queries": len(pairs),
        "top_k": K,
        "soft_recall_epsilon": eps,
        "metrics": {
            f"avg_clip_score@{K}":  round(float(np.mean(clip_scores_at_k)),  6),
            f"semantic_ndcg@{K}":   round(float(np.mean(ndcg_at_k_scores)),  6),
            f"soft_recall@{K}":     round(float(np.mean(soft_recall_hits)),   6),
            f"exact_recall@{K}":    round(float(np.mean(exact_recall_at_k)),  6),
            f"exact_precision@{K}": round(float(np.mean(exact_precision_at_k)), 6),
            f"mrr@{K}":             round(mrr_at_k(mrr_ranks, K),             6),
        },
        "per_k_breakdown": {},
    }

    # Per-K breakdown for K=1,3,5,10 (up to top_k)
    for k_sub in [1, 3, 5, 10]:
        if k_sub > K:
            break
        sub_recalls = []
        sub_precs   = []
        for pair, hit_rank in zip(pairs, mrr_ranks):
            hit = hit_rank is not None and hit_rank <= k_sub
            sub_recalls.append(1.0 if hit else 0.0)
            sub_precs.append((1.0 / k_sub) if hit else 0.0)
        results["per_k_breakdown"][f"R@{k_sub}"] = round(float(np.mean(sub_recalls)), 6)
        results["per_k_breakdown"][f"P@{k_sub}"] = round(float(np.mean(sub_precs)),   6)

    # ── 7. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS  –  COCO val2017")
    print("=" * 60)
    print(f"  Queries evaluated : {results['num_queries']}")
    print(f"  Top-K             : {K}")
    print(f"  Soft-Recall ε     : {eps}  (on [0,1] cosine scale)")
    print("-" * 60)
    for name, val in results["metrics"].items():
        print(f"  {name:<28s}: {val:.6f}")
    print("-" * 60)
    print("  Per-K breakdown:")
    for name, val in results["per_k_breakdown"].items():
        print(f"    {name:<10s}: {val:.6f}")
    print("=" * 60)

    # ── 8. Save JSON report ───────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Semantic evaluation of image-retrieval model on COCO val2017"
    )
    p.add_argument("--checkpoint",  default="checkpoints/best.pt",
                   help="Path to best.pt checkpoint")
    p.add_argument("--config",      default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--val_images",  default="data/coco/val2017",
                   help="Directory of COCO val2017 images")
    p.add_argument("--captions",    default="data/coco/annotations/captions_val2017.json",
                   help="COCO captions_val2017.json annotation file")
    p.add_argument("--top_k",       type=int, default=5,
                   help="K for all @K metrics")
    p.add_argument("--num_queries", type=int, default=1000,
                   help="Number of val queries to evaluate (max ~5000)")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="Batch size for gallery encoding")
    p.add_argument("--cache_dir",   default=".cache/eval",
                   help="Directory to cache FAISS index")
    p.add_argument("--rebuild_cache", action="store_true",
                   help="Force rebuild of FAISS gallery cache")
    p.add_argument("--soft_recall_eps", type=float, default=0.80,
                   help="Cosine similarity threshold (0-1 scale) for Soft Recall")
    p.add_argument("--out",         default="checkpoints/eval_semantic.json",
                   help="Output JSON report path")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluate(args)
