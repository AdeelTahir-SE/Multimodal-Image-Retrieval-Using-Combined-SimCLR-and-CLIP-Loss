"""
evaluate_image.py  –  Image-to-Image retrieval evaluation on COCO val2017.

Why NOT CLIP-only for image-to-image?
--------------------------------------
CLIP was trained on image-text pairs; its image-image cosine similarity is
biased toward features that correlate with language, not pure visual content.
For image-to-image retrieval the ground truth is *visual/semantic category
overlap*, which COCO instance annotations provide directly.

Metrics used (in priority order)
---------------------------------
1. Category Precision@K          [PRIMARY]
       Fraction of top-K retrieved images sharing ≥1 COCO category with query.
       Ground truth from instances_val2017.json — the gold standard.

2. Category NDCG@K               [PRIMARY]
       Graded relevance = Jaccard similarity of category sets between
       retrieved image and query. Rewards retrieving the most category-similar
       images highest.

3. Category Recall@K             [PRIMARY]
       Did the model retrieve at least one image from the same category?

4. Self-Embedding Cosine@K       [MODEL QUALITY]
       Mean cosine similarity between query embedding and top-K retrieved
       embeddings — measured INSIDE the trained model's own 128-D space.
       Tells you how well the embedding space is organised.

5. Exact Self-Retrieval Rank     [SANITY CHECK]
       When the query image is kept in the gallery (--keep_query_in_gallery),
       what rank does it appear at? Should be rank 1.

Usage
-----
python evaluate_image.py \
    --checkpoint   checkpoints/best.pt \
    --val_images   data/coco/val2017 \
    --instances    data/coco/annotations/instances_val2017.json \
    --config       config.yaml \
    --top_k        5 \
    --num_queries  500 \
    --batch_size   32 \
    --out          checkpoints/eval_image2image.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder   # needed only to load checkpoint cleanly

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PathImageDataset(Dataset):
    def __init__(self, paths: List[Path], tfm):
        self.paths, self.tfm = paths, tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        return self.tfm(Image.open(p).convert("RGB")), str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().float().numpy()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXT)


# ─────────────────────────────────────────────────────────────────────────────
# COCO category annotation loader
# ─────────────────────────────────────────────────────────────────────────────
def load_coco_categories(instances_json: Path) -> Tuple[
    Dict[str, Set[int]],   # filename → set of category_ids
    Dict[int, str],        # category_id → category_name
]:
    """
    Parse instances_val2017.json to build:
      - image_categories: {filename: {cat_id, ...}}
      - id2name:          {cat_id: name}
    """
    print(f"[COCO] Loading instance annotations from {instances_json} …")
    with open(instances_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    id2name: Dict[int, str] = {cat["id"]: cat["name"] for cat in data["categories"]}

    image_categories: Dict[str, Set[int]] = {}
    for ann in data["annotations"]:
        fname = id2file.get(ann["image_id"])
        if fname is None:
            continue
        image_categories.setdefault(fname, set()).add(ann["category_id"])

    print(f"[COCO] {len(image_categories)} images with annotations · "
          f"{len(id2name)} categories")
    return image_categories, id2name


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_image_encoder(checkpoint: Path, cfg: dict, device: torch.device) -> ImageEncoder:
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
    del txt_enc
    return img_enc


# ─────────────────────────────────────────────────────────────────────────────
# FAISS gallery builder — returns index AND raw embeddings
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_faiss_index(
    img_enc: ImageEncoder,
    image_paths: List[Path],
    batch_size: int,
    device: torch.device,
    cache_index: Optional[Path] = None,
    cache_meta:  Optional[Path] = None,
    cache_emb:   Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[str], np.ndarray]:
    """
    Returns (faiss_index, path_list, gallery_embeddings_float32).
    gallery_embeddings shape: (N, embedding_dim) — L2 normalised.
    """
    if (not force_rebuild and cache_index and cache_meta and cache_emb
            and cache_index.exists() and cache_meta.exists() and cache_emb.exists()):
        print(f"[FAISS] Loading cached index from {cache_index}")
        index = faiss.read_index(str(cache_index))
        with open(cache_meta, "r", encoding="utf-8") as f:
            paths = json.load(f)["paths"]
        embs = np.load(str(cache_emb))
        return index, paths, embs

    print(f"[FAISS] Encoding {len(image_paths)} images …")
    dataset = PathImageDataset(image_paths, RETRIEVAL_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embs:  List[np.ndarray] = []
    all_paths: List[str]        = []

    for i, (imgs, paths) in enumerate(loader):
        emb = F.normalize(img_enc(imgs.to(device)), dim=1)
        all_embs.append(_to_f32(emb))
        all_paths.extend(paths)
        if (i + 1) % 20 == 0:
            print(f"  {len(all_paths)}/{len(image_paths)} encoded", end="\r")

    print(f"  {len(all_paths)}/{len(image_paths)} encoded")

    gallery = np.vstack(all_embs).astype("float32")
    faiss.normalize_L2(gallery)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)

    if cache_index and cache_meta and cache_emb:
        cache_index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(cache_index))
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"paths": all_paths}, f, indent=2)
        np.save(str(cache_emb), gallery)
        print(f"[FAISS] Index saved → {cache_index}")

    return index, all_paths, gallery


# ─────────────────────────────────────────────────────────────────────────────
# Semantic relevance helpers (COCO categories)
# ─────────────────────────────────────────────────────────────────────────────
def jaccard(a: Set[int], b: Set[int]) -> float:
    """Jaccard similarity between two category sets — used as graded relevance."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def category_overlap(a: Set[int], b: Set[int]) -> bool:
    """True if the two images share at least one COCO category."""
    return bool(a & b)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def ndcg_at_k(relevances: List[float], k: int) -> float:
    top   = relevances[:k]
    dcg   = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(hit_ranks: List[Optional[int]], k: int) -> float:
    vals = [1.0 / r if (r is not None and r <= k) else 0.0 for r in hit_ranks]
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. Load retrieval model ──────────────────────────────────────────────
    checkpoint = Path(args.checkpoint)
    print(f"[Model] Loading ImageEncoder from {checkpoint}")
    img_enc = load_image_encoder(checkpoint, cfg, device)

    # ── 2. Load COCO category annotations ───────────────────────────────────
    instances_json = Path(args.instances)
    image_categories, id2name = load_coco_categories(instances_json)

    # ── 3. Discover val images that have category annotations ────────────────
    val_dir    = Path(args.val_images)
    all_images = discover_images(val_dir)
    # Keep only images that have COCO instance annotations
    annotated  = [p for p in all_images if p.name in image_categories]
    print(f"[Gallery] {len(all_images)} total val images · "
          f"{len(annotated)} have category annotations")

    # ── 4. Sample queries ────────────────────────────────────────────────────
    query_pool = list(annotated)
    random.shuffle(query_pool)
    queries = query_pool[: args.num_queries]
    print(f"[Queries] {len(queries)} query images selected")

    # ── 5. Build gallery ─────────────────────────────────────────────────────
    cache_dir   = Path(args.cache_dir)
    cache_index = cache_dir / "val2017_img2img.index"
    cache_meta  = cache_dir / "val2017_img2img.json"
    cache_emb   = cache_dir / "val2017_img2img_emb.npy"

    if args.keep_query_in_gallery:
        gallery_images = annotated
        print(f"[Gallery] All {len(gallery_images)} annotated images in gallery "
              f"(self-retrieval mode).")
    else:
        query_names    = {p.name for p in queries}
        gallery_images = [p for p in annotated if p.name not in query_names]
        print(f"[Gallery] {len(gallery_images)} gallery images "
              f"(query images excluded — hard mode).")

    gallery_index, gallery_paths, gallery_embs = build_faiss_index(
        img_enc, gallery_images, args.batch_size, device,
        cache_index, cache_meta, cache_emb,
        force_rebuild=args.rebuild_cache,
    )

    K   = args.top_k
    emb_dim = gallery_embs.shape[1]

    # ── 6. Evaluation loop ───────────────────────────────────────────────────
    # Accumulators
    cat_precisions: List[float]         = []   # Metric 1
    cat_ndcg:       List[float]         = []   # Metric 2
    cat_recalls:    List[float]         = []   # Metric 3
    self_emb_sims:  List[float]         = []   # Metric 4
    exact_hit_ranks: List[Optional[int]] = []  # Metric 5 (sanity)

    print(f"\n[Eval] Running {len(queries)} queries (top_k={K}) …\n")

    for qi, qpath in enumerate(queries):
        q_cats = image_categories.get(qpath.name, set())

        # ── Encode query with retrieval model ──
        q_img = RETRIEVAL_TRANSFORM(Image.open(qpath).convert("RGB")).unsqueeze(0).to(device)
        q_emb = F.normalize(img_enc(q_img), dim=1)
        q_np  = _to_f32(q_emb).reshape(1, -1)
        faiss.normalize_L2(q_np)

        scores_np, topk_idxs_np = gallery_index.search(q_np, min(K, gallery_index.ntotal))
        topk_idxs  = topk_idxs_np[0].tolist()
        topk_scores = scores_np[0].tolist()          # cosine similarities from FAISS
        retrieved  = [Path(gallery_paths[i]) for i in topk_idxs]

        # ── METRIC 4: Self-embedding cosine similarity ──
        # Mean cosine similarity in the model's OWN embedding space
        self_emb_sims.append(float(np.mean(topk_scores)))

        # ── METRIC 5: Exact self-retrieval (only meaningful if keep_query_in_gallery) ──
        hit_rank = None
        for rank, rp in enumerate(retrieved, 1):
            if rp.name == qpath.name:
                hit_rank = rank
                break
        exact_hit_ranks.append(hit_rank)

        # ── Get categories for retrieved images ──
        ret_cats = [image_categories.get(rp.name, set()) for rp in retrieved]

        # ── METRIC 1: Category Precision@K ──
        hits = [1.0 if category_overlap(q_cats, rc) else 0.0 for rc in ret_cats]
        cat_precisions.append(float(np.mean(hits)))

        # ── METRIC 2: Category NDCG@K  (Jaccard = graded relevance) ──
        relevances = [jaccard(q_cats, rc) for rc in ret_cats]
        cat_ndcg.append(ndcg_at_k(relevances, K))

        # ── METRIC 3: Category Recall@K ──
        # Did we find at least one image sharing a category?
        cat_recalls.append(1.0 if any(category_overlap(q_cats, rc) for rc in ret_cats) else 0.0)

        if (qi + 1) % 50 == 0 or (qi + 1) == len(queries):
            print(f"  [{qi+1:4d}/{len(queries)}] "
                  f"CatP@{K}={np.mean(cat_precisions):.4f}  "
                  f"CatNDCG@{K}={np.mean(cat_ndcg):.4f}  "
                  f"CatR@{K}={np.mean(cat_recalls):.4f}  "
                  f"EmbSim@{K}={np.mean(self_emb_sims):.4f}")

    # ── 7. Per-K breakdown ───────────────────────────────────────────────────
    per_k: dict = {}
    for k_sub in [1, 3, 5, 10]:
        if k_sub > K:
            break
        # Re-compute by re-running per query is expensive; approximate from hit_ranks
        # For category metrics we'd need to redo, so report exact self-retrieval per-K
        sub_r = [1.0 if (r is not None and r <= k_sub) else 0.0
                 for r in exact_hit_ranks]
        per_k[f"exact_self_retrieval@{k_sub}"] = round(float(np.mean(sub_r)), 6)

    # ── 8. Aggregate results ─────────────────────────────────────────────────
    results = {
        "num_queries":           len(queries),
        "gallery_size":          len(gallery_paths),
        "top_k":                 K,
        "keep_query_in_gallery": args.keep_query_in_gallery,
        "metrics": {
            f"category_precision@{K}":     round(float(np.mean(cat_precisions)), 6),
            f"category_ndcg@{K}":          round(float(np.mean(cat_ndcg)),       6),
            f"category_recall@{K}":        round(float(np.mean(cat_recalls)),     6),
            f"self_embedding_cosine@{K}":  round(float(np.mean(self_emb_sims)),  6),
            f"exact_self_retrieval@{K}":   round(
                float(np.mean([1.0 if r is not None else 0.0 for r in exact_hit_ranks])), 6),
            f"mrr@{K}":                    round(mrr_at_k(exact_hit_ranks, K),   6),
        },
        "per_k_breakdown": per_k,
    }

    # ── 9. Print summary ─────────────────────────────────────────────────────
    mode = "self-retrieval" if args.keep_query_in_gallery else "hard (query excluded)"
    print("\n" + "=" * 64)
    print("  IMAGE-TO-IMAGE EVALUATION  –  COCO val2017")
    print("=" * 64)
    print(f"  Queries        : {results['num_queries']}")
    print(f"  Gallery size   : {results['gallery_size']}")
    print(f"  Top-K          : {K}")
    print(f"  Mode           : {mode}")
    print("-" * 64)
    desc = {
        f"category_precision@{K}":    "% of top-K sharing ≥1 COCO category  [PRIMARY]",
        f"category_ndcg@{K}":         "Jaccard-graded NDCG ranking score     [PRIMARY]",
        f"category_recall@{K}":       "Query answered by ≥1 same-cat image   [PRIMARY]",
        f"self_embedding_cosine@{K}": "Mean cosine sim in model's own space   [MODEL]",
        f"exact_self_retrieval@{K}":  "Exact image found in top-K (sanity)   [SANITY]",
        f"mrr@{K}":                   "Mean Reciprocal Rank of exact image    [SANITY]",
    }
    for name, val in results["metrics"].items():
        print(f"  {val:.6f}  {name:<35s}  {desc.get(name,'')}")
    if per_k:
        print("-" * 64)
        print("  Per-K self-retrieval breakdown:")
        for k, v in per_k.items():
            print(f"    {k:<35s}: {v:.6f}")
    print("=" * 64)

    # ── 10. Save report ───────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Image-to-image retrieval evaluation using COCO category semantics"
    )
    p.add_argument("--checkpoint",  default="checkpoints/best.pt")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--val_images",  default="data/coco/val2017")
    p.add_argument("--instances",   default="data/coco/annotations/instances_val2017.json",
                   help="COCO instances_val2017.json for category ground truth")
    p.add_argument("--top_k",       type=int,   default=5)
    p.add_argument("--num_queries", type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--cache_dir",   default=".cache/eval_img2img")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--keep_query_in_gallery", action="store_true",
                   help="Include query images in gallery for self-retrieval test.")
    p.add_argument("--out",         default="checkpoints/eval_image2image.json")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
