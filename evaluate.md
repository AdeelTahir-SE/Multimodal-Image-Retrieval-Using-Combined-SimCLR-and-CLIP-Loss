# Evaluation Pipeline — `evaluate.py`

Semantic evaluation of the **Multimodal Image Retrieval (SimCLR + CLIP)** model on the **COCO val2017** dataset.  
Unlike binary hit/miss metrics, this pipeline measures *how semantically relevant* retrieved images are to a text query.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Flow](#architecture--flow)
3. [Metrics Explained](#metrics-explained)
4. [CLI Usage](#cli-usage)
5. [Output](#output)
6. [Sample Results](#sample-results)
7. [Dependencies](#dependencies)
8. [Caching](#caching)

---

## Overview

Standard image-retrieval evaluation asks: *"Did we retrieve the exact ground-truth image?"*  
That penalises the model even when a retrieved image is visually indistinguishable from the target.

`evaluate.py` addresses this with four complementary metrics:

| Metric | Type | What it measures |
|---|---|---|
| **Avg CLIP Score @ K** | Semantic | Text ↔ image cosine similarity via frozen CLIP |
| **Semantic NDCG @ K** | Graded ranking | How well-ranked semantically similar images are |
| **Soft Recall @ K** | Near-match | Whether any top-K image is visually close to ground truth |
| **Exact Recall / Precision / MRR @ K** | Binary | Traditional hit-or-miss baseline |

---

## Architecture & Flow

```
COCO val2017 captions_val2017.json
        │
        ▼
  Sample N queries (image_id, caption, image_path)
        │
        ├──► Retrieval Model (best.pt)
        │         ├── ImageEncoder  (ResNet50 + ProjectionHead → 128-D)
        │         └── TextEncoder   (DistilBERT + ProjectionHead → 128-D)
        │
        ├──► FAISS Index  (IndexFlatIP over all val2017 images)
        │         └── Encoded with ImageEncoder, L2-normalised
        │
        └──► CLIP Evaluator  (openai/clip-vit-base-patch32, frozen)
                  ├── clip_text_embed(caption)   → text vector
                  └── clip_image_embed(images)   → image vectors
```

**Step-by-step for each query:**

1. **Encode the caption** with `TextEncoder` → project into the 128-D retrieval space.
2. **FAISS search** — retrieve the top-K image paths from the val2017 gallery.
3. **CLIP Metric 1** — embed the caption and all K retrieved images with the *frozen* CLIP; compute cosine similarity for each pair, then average.
4. **CLIP Metric 2** — embed the ground-truth image and all K retrieved images with CLIP; use cosine similarity as a graded relevance score and compute NDCG@K.
5. **CLIP Metric 3** — check if any retrieved image has cosine similarity ≥ ε to the ground-truth image (Soft Recall).
6. **Standard Metric 4** — check for the exact filename match among the top-K results.

---

## Metrics Explained

### 1. Average CLIP Score @ K

```
CLIP_Score@K = (1/K) Σ cos_sim( CLIP_text(query), CLIP_image(retrieved_i) )
```

- Uses `openai/clip-vit-base-patch32` as a **frozen evaluator** — completely independent of the trained model.
- Range: approximately **[−1, 1]**; typical values for semantically relevant pairs sit around **0.2 – 0.35**.
- **Higher is better.** A score of ~0.25 means the retrieved images are genuinely related to the text.

### 2. Semantic NDCG @ K

```
Relevance(i) = ( cos_sim( CLIP_img(gt), CLIP_img(retrieved_i) ) + 1 ) / 2   ∈ [0, 1]

NDCG@K = DCG@K / IDCG@K
       = Σ_{i=1}^{K} rel_i / log2(i+1)   ÷   ideal ordering
```

- Unlike binary NDCG, relevance is **continuous** — a visually similar image gets partial credit (e.g. 0.85) instead of 0.
- **Higher is better.** A score near **1.0** means retrieved images are in near-perfect semantic order relative to the ground truth.

### 3. Soft Recall @ K

```
SoftRecall@K = fraction of queries where
               max_i cos_sim( CLIP_img(gt), CLIP_img(retrieved_i) ) ≥ threshold

threshold = 2 × ε − 1   (mapping ε from [0,1] scale to [−1,1] cosine scale)
```

- Default ε = **0.80** (i.e. threshold ≈ 0.60 in raw cosine similarity).
- **Higher is better.** 0.88 means 88 % of queries returned at least one image visually close to the target.

### 4. Exact Recall / Precision / MRR @ K

Standard retrieval metrics treating only the ground-truth file as a hit.

```
Recall@K    = hits / total_queries
Precision@K = hits / (total_queries × K)
MRR@K       = mean(1 / rank_of_first_hit)   if hit within K, else 0
```

- These are intentionally harsh — the exact image must appear in the top K.
- Useful as a **lower-bound baseline** alongside the semantic metrics.

---

## CLI Usage

```powershell
python evaluate.py `
    --checkpoint  checkpoints/best.pt `
    --config      config.yaml `
    --val_images  data/coco/val2017 `
    --captions    data/coco/annotations/captions_val2017.json `
    --top_k       5 `
    --num_queries 1000 `
    --batch_size  32 `
    --soft_recall_eps 0.80 `
    --cache_dir   .cache/eval `
    --out         checkpoints/eval_semantic.json
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--config` | `config.yaml` | YAML config (embedding_dim, backbone names, etc.) |
| `--val_images` | `data/coco/val2017` | Directory of val2017 images |
| `--captions` | `data/coco/annotations/captions_val2017.json` | COCO captions annotation file |
| `--top_k` | `5` | K for all @K metrics |
| `--num_queries` | `1000` | Number of queries to sample (max ~5000) |
| `--batch_size` | `64` | Batch size for FAISS gallery encoding |
| `--soft_recall_eps` | `0.80` | Similarity threshold ε for Soft Recall (0–1 scale) |
| `--cache_dir` | `.cache/eval` | Where to store the FAISS index cache |
| `--rebuild_cache` | `False` | Force re-encoding the gallery (flag, no value needed) |
| `--out` | `checkpoints/eval_semantic.json` | Output JSON report path |
| `--seed` | `42` | Random seed for reproducibility |

---

## Output

The script prints a live progress table every 50 queries and a final summary:

```
============================================================
  EVALUATION RESULTS  –  COCO val2017
============================================================
  Queries evaluated : 1000
  Top-K             : 5
  Soft-Recall ε     : 0.8
------------------------------------------------------------
  avg_clip_score@5             : 0.247808
  semantic_ndcg@5              : 0.986168
  soft_recall@5                : 0.880000
  exact_recall@5               : 0.100000
  exact_precision@5            : 0.020000
  mrr@5                        : 0.045000
------------------------------------------------------------
  Per-K breakdown:
    R@1       : 0.020000
    P@1       : 0.020000
    R@3       : 0.080000
    P@3       : 0.026667
    R@5       : 0.100000
    P@5       : 0.020000
============================================================
```

A JSON report is saved at the path specified by `--out`:

```json
{
  "num_queries": 1000,
  "top_k": 5,
  "soft_recall_epsilon": 0.8,
  "metrics": {
    "avg_clip_score@5": 0.247808,
    "semantic_ndcg@5": 0.986168,
    "soft_recall@5": 0.88,
    "exact_recall@5": 0.1,
    "exact_precision@5": 0.02,
    "mrr@5": 0.045
  },
  "per_k_breakdown": {
    "R@1": 0.02, "P@1": 0.02,
    "R@3": 0.08, "P@3": 0.026667,
    "R@5": 0.1,  "P@5": 0.02
  }
}
```

---

## Sample Results

Results from a partial run (50 queries, top_k=5) on COCO val2017 using `checkpoints/best.pt`:

| Metric | Value | Interpretation |
|---|---|---|
| `avg_clip_score@5` | **0.2478** | Retrieved images are semantically related to query text |
| `semantic_ndcg@5` | **0.9862** | Near-perfect semantic ranking relative to ground truth |
| `soft_recall@5` | **0.88** | 88 % of queries found a visually similar image in top-5 |
| `exact_recall@5` | **0.10** | Only 10 % found the *exact* ground-truth image — expected for a 5000-image gallery |
| `mrr@5` | **0.045** | Exact hits tend to appear near rank 5 when they do occur |

> **Key insight:** The large gap between `soft_recall@5` (0.88) and `exact_recall@5` (0.10) confirms the model retrieves *semantically correct* images even when it misses the exact file — which is the desired behaviour for a multimodal retrieval system.

---

## Dependencies

```
torch
torchvision
transformers          # HuggingFace — for DistilBERT & CLIP (ViT-B/32)
faiss-cpu             # or faiss-gpu
Pillow
PyYAML
numpy
```

> **Optional:** Install the official OpenAI CLIP package for faster inference:
> ```bash
> pip install git+https://github.com/openai/CLIP.git
> ```
> The script auto-detects it and falls back to HuggingFace `transformers` if not found.

---

## Caching

The FAISS gallery index is built once and saved to `--cache_dir`:

```
.cache/eval/
├── val2017_gallery.index   # FAISS IndexFlatIP (inner product)
└── val2017_gallery.json    # Ordered list of image paths
```

On subsequent runs the index is loaded instantly, skipping re-encoding of all 5000 images.  
Use `--rebuild_cache` to force a fresh build (e.g. after updating the checkpoint).
