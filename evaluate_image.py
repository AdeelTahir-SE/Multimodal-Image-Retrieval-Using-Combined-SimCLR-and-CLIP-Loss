"""
evaluate_image.py  –  Image-to-Image retrieval evaluation on COCO val2017.

The query IS an image. The model encodes it with ImageEncoder, searches the
FAISS gallery, and we measure how well the retrieved images match the query
using a *frozen* CLIP (ViT-B/32) as the semantic evaluator.

Metrics
-------
1. Avg CLIP Image-Image Score @ K
       cos_sim( CLIP_img(query), CLIP_img(retrieved_i) )  averaged over K
2. Semantic NDCG @ K
       Graded relevance = CLIP image-image similarity → shifted to [0,1]
3. Soft Recall @ K
       ≥1 retrieved image has CLIP similarity ≥ ε to the query image
4. Exact Recall / Precision / MRR @ K   (self-retrieval baseline)
       Ground truth = the query image itself; it should appear as rank-1.
       The query image is EXCLUDED from the gallery to make this non-trivial,
       or kept in (flag --keep_query_in_gallery) to test rank-1 self-retrieval.

Usage
-----
python evaluate_image.py \
    --checkpoint   checkpoints/best.pt \
    --val_images   data/coco/val2017 \
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
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder   # needed only to load the checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PathImageDataset(Dataset):
    def __init__(self, paths: List[Path], tfm):
        self.paths = paths
        self.tfm   = tfm

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
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_image_encoder(checkpoint: Path, cfg: dict, device: torch.device) -> ImageEncoder:
    """Load only the ImageEncoder from the checkpoint."""
    img_enc = ImageEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        backbone_name=cfg["model"]["image_backbone"],
        use_pretrained=cfg["model"].get("image_pretrained", False),
    ).to(device)
    # Also instantiate TextEncoder so the checkpoint loads without KeyError
    txt_enc = TextEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        model_name=cfg["model"]["text_backbone"],
        use_pretrained=cfg["model"].get("text_pretrained", False),
    ).to(device)

    payload = torch.load(checkpoint, map_location=device)
    img_enc.load_state_dict(payload["image_encoder"])
    txt_enc.load_state_dict(payload["text_encoder"])   # discard after load
    img_enc.eval()
    del txt_enc
    return img_enc


def load_clip_evaluator(device: torch.device):
    """Load frozen CLIP ViT-B/32 — try openai/clip first, fall back to HF."""
    try:
        import clip
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        print("[Evaluator] Loaded openai/clip ViT-B/32 via `clip` package.")
        return model, "openai_clip"
    except ImportError:
        pass

    try:
        from transformers import CLIPModel, CLIPProcessor
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        print("[Evaluator] Loaded openai/clip-vit-base-patch32 via HuggingFace.")
        return (model, processor), "hf_clip"
    except Exception as e:
        raise RuntimeError(
            "Could not load CLIP evaluator.\n"
            "Install via:  pip install git+https://github.com/openai/CLIP.git\n"
            "or:           pip install transformers\n" + str(e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# FAISS gallery
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_faiss_index(
    img_enc: ImageEncoder,
    image_paths: List[Path],
    batch_size: int,
    device: torch.device,
    cache_index: Optional[Path] = None,
    cache_meta:  Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[str]]:

    if (not force_rebuild and cache_index and cache_meta
            and cache_index.exists() and cache_meta.exists()):
        print(f"[FAISS] Loading cached index from {cache_index}")
        index = faiss.read_index(str(cache_index))
        with open(cache_meta, "r", encoding="utf-8") as f:
            paths = json.load(f)["paths"]
        return index, paths

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

    if cache_index and cache_meta:
        cache_index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(cache_index))
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"paths": all_paths}, f, indent=2)
        print(f"[FAISS] Index saved → {cache_index}")

    return index, all_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLIP image encoder helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def clip_image_embed(
    image_paths: List[Path],
    clip_pkg,
    clip_type: str,
    device: torch.device,
) -> torch.Tensor:
    """Return L2-normalised CLIP image embeddings, shape (N, D)."""
    if clip_type == "openai_clip":
        import clip
        tensors = torch.stack(
            [CLIP_TRANSFORM(Image.open(p).convert("RGB")) for p in image_paths]
        ).to(device)
        feats = clip_pkg.encode_image(tensors).float()
    else:
        model, processor = clip_pkg
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Use vision_model + visual_projection to always get a plain tensor
        vision_out = model.vision_model(**inputs)
        feats = model.visual_projection(vision_out.pooler_output).float()
    return F.normalize(feats, dim=1)


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
# Main evaluation
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

    # ── 2. Load frozen CLIP evaluator ────────────────────────────────────────
    print("[Evaluator] Loading frozen CLIP …")
    clip_pkg, clip_type = load_clip_evaluator(device)

    # ── 3. Discover val images ───────────────────────────────────────────────
    val_dir = Path(args.val_images)
    all_images = discover_images(val_dir)
    print(f"[Gallery] {len(all_images)} val images found in {val_dir}")

    # ── 4. Sample queries ────────────────────────────────────────────────────
    query_pool = list(all_images)
    random.shuffle(query_pool)
    queries = query_pool[: args.num_queries]
    print(f"[Queries] {len(queries)} query images selected")

    # ── 5. Build FAISS gallery ───────────────────────────────────────────────
    cache_dir   = Path(args.cache_dir)
    cache_index = cache_dir / "val2017_img2img.index"
    cache_meta  = cache_dir / "val2017_img2img.json"

    if args.keep_query_in_gallery:
        gallery_images = all_images
        print("[Gallery] Query images KEPT in gallery (self-retrieval test).")
    else:
        # Build gallery from images NOT in the query set so rank-1 ≠ trivial self-match
        query_names = {p.name for p in queries}
        gallery_images = [p for p in all_images if p.name not in query_names]
        print(f"[Gallery] {len(gallery_images)} gallery images "
              f"(query images excluded — hard evaluation).")

    gallery_index, gallery_paths = build_faiss_index(
        img_enc, gallery_images, args.batch_size, device,
        cache_index, cache_meta, force_rebuild=args.rebuild_cache,
    )
    path2name = {Path(p).name: i for i, p in enumerate(gallery_paths)}

    K   = args.top_k
    eps = args.soft_recall_eps

    # ── 6. Per-query evaluation ──────────────────────────────────────────────
    clip_scores:   List[float]         = []
    ndcg_scores:   List[float]         = []
    soft_hits:     List[float]         = []
    exact_recalls: List[float]         = []
    exact_precs:   List[float]         = []
    hit_ranks:     List[Optional[int]] = []

    print(f"\n[Eval] Running {len(queries)} queries (top_k={K}) …\n")

    for qi, qpath in enumerate(queries):
        # ── Encode query with retrieval model ──
        q_img = RETRIEVAL_TRANSFORM(Image.open(qpath).convert("RGB")).unsqueeze(0).to(device)
        q_emb = F.normalize(img_enc(q_img), dim=1)
        q_np  = _to_f32(q_emb).reshape(1, -1)
        faiss.normalize_L2(q_np)

        _, topk_idxs = gallery_index.search(q_np, min(K, gallery_index.ntotal))
        topk_idxs    = topk_idxs[0].tolist()
        retrieved     = [Path(gallery_paths[i]) for i in topk_idxs]

        # ── METRIC 4: Exact self-retrieval ──
        # Ground truth: a gallery image with the same filename as the query.
        hit_rank   = None
        exact_hits = 0
        for rank, rp in enumerate(retrieved, 1):
            if rp.name == qpath.name:
                if hit_rank is None:
                    hit_rank = rank
                exact_hits += 1
        exact_recalls.append(1.0 if hit_rank is not None else 0.0)
        exact_precs.append(exact_hits / K)
        hit_ranks.append(hit_rank)

        # ── CLIP embeddings ──
        query_clip = clip_image_embed([qpath], clip_pkg, clip_type, device)      # (1, D)
        ret_clip   = clip_image_embed(retrieved, clip_pkg, clip_type, device)    # (K, D)
        sims       = (query_clip @ ret_clip.T).squeeze(0)                        # (K,)

        # ── METRIC 1: Avg CLIP Image-Image Score @ K ──
        clip_scores.append(sims.mean().item())

        # ── METRIC 2: Semantic NDCG @ K ──
        relevances = ((sims + 1.0) / 2.0).cpu().tolist()   # shift [-1,1] → [0,1]
        ndcg_scores.append(ndcg_at_k(relevances, K))

        # ── METRIC 3: Soft Recall @ K ──
        threshold = 2.0 * eps - 1.0                          # back to [-1,1] scale
        soft_hits.append(1.0 if sims.max().item() >= threshold else 0.0)

        if (qi + 1) % 50 == 0 or (qi + 1) == len(queries):
            print(f"  [{qi+1:4d}/{len(queries)}] "
                  f"CLIP@{K}={np.mean(clip_scores):.4f}  "
                  f"NDCG@{K}={np.mean(ndcg_scores):.4f}  "
                  f"SoftR@{K}={np.mean(soft_hits):.4f}  "
                  f"ExactR@{K}={np.mean(exact_recalls):.4f}")

    # ── 7. Aggregate ─────────────────────────────────────────────────────────
    results: dict = {
        "num_queries":         len(queries),
        "gallery_size":        len(gallery_paths),
        "top_k":               K,
        "soft_recall_epsilon": eps,
        "keep_query_in_gallery": args.keep_query_in_gallery,
        "metrics": {
            f"avg_clip_score@{K}":  round(float(np.mean(clip_scores)),   6),
            f"semantic_ndcg@{K}":   round(float(np.mean(ndcg_scores)),   6),
            f"soft_recall@{K}":     round(float(np.mean(soft_hits)),      6),
            f"exact_recall@{K}":    round(float(np.mean(exact_recalls)),  6),
            f"exact_precision@{K}": round(float(np.mean(exact_precs)),    6),
            f"mrr@{K}":             round(mrr_at_k(hit_ranks, K),         6),
        },
        "per_k_breakdown": {},
    }

    for k_sub in [1, 3, 5, 10]:
        if k_sub > K:
            break
        sub_r, sub_p = [], []
        for hr in hit_ranks:
            hit = hr is not None and hr <= k_sub
            sub_r.append(1.0 if hit else 0.0)
            sub_p.append((1.0 / k_sub) if hit else 0.0)
        results["per_k_breakdown"][f"R@{k_sub}"] = round(float(np.mean(sub_r)), 6)
        results["per_k_breakdown"][f"P@{k_sub}"] = round(float(np.mean(sub_p)), 6)

    # ── 8. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  IMAGE-TO-IMAGE EVALUATION  –  COCO val2017")
    print("=" * 62)
    print(f"  Queries       : {results['num_queries']}")
    print(f"  Gallery size  : {results['gallery_size']}")
    print(f"  Top-K         : {K}")
    print(f"  Soft-Recall ε : {eps}  (on [0,1] cosine scale)")
    keep_str = "YES (self-retrieval test)" if args.keep_query_in_gallery else "NO  (hard eval)"
    print(f"  Query in gallery : {keep_str}")
    print("-" * 62)
    for name, val in results["metrics"].items():
        print(f"  {name:<30s}: {val:.6f}")
    print("-" * 62)
    print("  Per-K breakdown:")
    for name, val in results["per_k_breakdown"].items():
        print(f"    {name:<10s}: {val:.6f}")
    print("=" * 62)

    # ── 9. Save report ────────────────────────────────────────────────────────
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
        description="Image-to-image retrieval evaluation on COCO val2017"
    )
    p.add_argument("--checkpoint",    default="checkpoints/best.pt")
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--val_images",    default="data/coco/val2017")
    p.add_argument("--top_k",         type=int,   default=5)
    p.add_argument("--num_queries",   type=int,   default=500)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--soft_recall_eps", type=float, default=0.80)
    p.add_argument("--cache_dir",     default=".cache/eval_img2img")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--keep_query_in_gallery", action="store_true",
                   help="Keep query images in the gallery (self-retrieval / rank-1 test).")
    p.add_argument("--out",           default="checkpoints/eval_image2image.json")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
