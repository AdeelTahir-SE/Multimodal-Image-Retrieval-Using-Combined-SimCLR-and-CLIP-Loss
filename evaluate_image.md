# Evaluation Pipeline — `evaluate_image.py`

Image-to-image retrieval evaluation for the **Multimodal Image Retrieval (SimCLR + CLIP)** model on **COCO val2017**.  
A query image is encoded by the trained `ImageEncoder` and searched against a FAISS gallery of val2017 images. A **frozen CLIP ViT-B/32** acts as an independent semantic judge.

---

## Table of Contents

1. [Key Difference from Text-to-Image Eval](#key-difference-from-text-to-image-eval)
2. [Architecture & Flow](#architecture--flow)
3. [Metrics Explained](#metrics-explained)
4. [Gallery Modes](#gallery-modes)
5. [CLI Usage](#cli-usage)
6. [Output](#output)
7. [Dependencies](#dependencies)
8. [Caching](#caching)

---

## Key Difference from Text-to-Image Eval

| | `evaluate.py` | `evaluate_image.py` |
|---|---|---|
| **Query type** | Text caption | Image |
| **Query encoder** | `TextEncoder` (DistilBERT) | `ImageEncoder` (ResNet50) |
| **CLIP evaluator** | text ↔ image cosine sim | **image ↔ image** cosine sim |
| **Ground truth** | Paired caption image | Query image itself |
| **Gallery mode** | All val images (query always in gallery) | Configurable — hard (query excluded) or self-retrieval (query included) |

---

## Architecture & Flow

```
val2017 images  (random sample = queries)
        │
        ▼
  ImageEncoder (ResNet50 + ProjectionHead → 128-D)
        │   loaded from checkpoints/best.pt
        ▼
  FAISS IndexFlatIP
  (gallery = val2017 images, optionally excluding the query set)
        │
        ├──► Top-K retrieved image paths
        │
        └──► Frozen CLIP ViT-B/32  (evaluator, never fine-tuned)
                  ├── CLIP_img(query)       → q_vec
                  └── CLIP_img(retrieved_i) → r_vec_i
                            │
                            ▼
                  cos_sim(q_vec, r_vec_i) → all 4 metrics
```

**Per-query steps:**

1. Encode the query image with `ImageEncoder` → 128-D L2-normalised vector.
2. **FAISS search** → top-K gallery image paths.
3. Encode the query and all K retrieved images with **frozen CLIP**.
4. Compute cosine similarities → feed into all four metrics.

---

## Metrics Explained

### 1. Average CLIP Image-Image Score @ K

```
CLIP_Score@K = (1/K) Σ cos_sim( CLIP_img(query), CLIP_img(retrieved_i) )
```

- Measures how visually/semantically close the retrieved images are to the query, as judged by an independent CLIP model.
- Range: **[−1, 1]**. Typical values for semantically similar COCO images: **0.7 – 0.95**.
- **Higher is better.**

### 2. Semantic NDCG @ K

```
Relevance(i) = ( cos_sim( CLIP_img(query), CLIP_img(retrieved_i) ) + 1 ) / 2   ∈ [0, 1]

NDCG@K = DCG@K / IDCG@K
```

- Each retrieved image receives a **continuous relevance score** instead of binary 0/1.
- Rewards retrieving the *most similar* images at the *highest ranks*.
- **Higher is better.** A score near 1.0 means semantically similar images consistently rank first.

### 3. Soft Recall @ K

```
SoftRecall@K = fraction of queries where
               max_i cos_sim( CLIP_img(query), CLIP_img(retrieved_i) ) ≥ threshold

threshold = 2 × ε − 1    (maps ε ∈ [0,1] → raw cosine ∈ [−1,1])
```

- Default ε = **0.80** → threshold ≈ **0.60** raw cosine similarity.
- Checks: did the model find *at least one* highly similar image?
- **Higher is better.**

### 4. Exact Recall / Precision / MRR @ K

```
Ground truth = gallery image with the same filename as the query image.

Recall@K    = queries where exact match found in top-K  /  total queries
Precision@K = exact match count / (total_queries × K)
MRR@K       = mean( 1 / rank_of_first_exact_hit )   if found within K else 0
```

- Binary hit-or-miss baseline.
- In **hard mode** (default), the query image is *excluded* from the gallery — the model must find semantically indistinguishable duplicates or near-identical images.
- In **self-retrieval mode** (`--keep_query_in_gallery`), the exact image IS in the gallery. Rank-1 self-retrieval rate is the expected dominant metric.

---

## Gallery Modes

### Hard Mode (default — recommended for research)

```
gallery = val2017 images  MINUS  the query set
```

The exact query image is not in the gallery. The model must retrieve *visually similar* images.  
`exact_recall` will naturally be low or 0 in this mode — use CLIP-based metrics as the primary signal.

### Self-Retrieval Mode (`--keep_query_in_gallery`)

```
gallery = ALL val2017 images  (queries included)
```

The query image is in the gallery. A well-trained `ImageEncoder` should retrieve it at **rank 1**.  
`exact_recall@1` ≈ `mrr@1` are the dominant metrics in this mode.

---

## CLI Usage

### Hard evaluation (query excluded from gallery)

```powershell
python evaluate_image.py `
    --checkpoint   checkpoints/best.pt `
    --val_images   data/coco/val2017 `
    --config       config.yaml `
    --top_k        5 `
    --num_queries  500 `
    --batch_size   32 `
    --soft_recall_eps 0.80 `
    --out          checkpoints/eval_image2image.json
```

### Self-retrieval test (rank-1 accuracy)

```powershell
python evaluate_image.py `
    --checkpoint   checkpoints/best.pt `
    --val_images   data/coco/val2017 `
    --keep_query_in_gallery `
    --top_k        1 `
    --num_queries  1000 `
    --out          checkpoints/eval_self_retrieval.json
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--config` | `config.yaml` | YAML config (embedding_dim, backbone, etc.) |
| `--val_images` | `data/coco/val2017` | Directory of COCO val2017 images |
| `--top_k` | `5` | K for all @K metrics |
| `--num_queries` | `500` | Number of query images to sample |
| `--batch_size` | `32` | Batch size for FAISS gallery encoding |
| `--soft_recall_eps` | `0.80` | Similarity threshold ε for Soft Recall (0–1 scale) |
| `--cache_dir` | `.cache/eval_img2img` | FAISS index cache directory |
| `--rebuild_cache` | `False` | Force re-encoding the gallery |
| `--keep_query_in_gallery` | `False` | Include query images in gallery (self-retrieval mode) |
| `--out` | `checkpoints/eval_image2image.json` | Output JSON report path |
| `--seed` | `42` | Random seed |

---

## Output

### Console

```
==============================================================
  IMAGE-TO-IMAGE EVALUATION  –  COCO val2017
==============================================================
  Queries          : 500
  Gallery size     : 4500
  Top-K            : 5
  Soft-Recall ε    : 0.8
  Query in gallery : NO  (hard eval)
--------------------------------------------------------------
  avg_clip_score@5             : 0.821430
  semantic_ndcg@5              : 0.973200
  soft_recall@5                : 0.912000
  exact_recall@5               : 0.000000
  exact_precision@5            : 0.000000
  mrr@5                        : 0.000000
--------------------------------------------------------------
  Per-K breakdown:
    R@1       : 0.000000
    P@1       : 0.000000
    R@3       : 0.000000
    P@3       : 0.000000
    R@5       : 0.000000
    P@5       : 0.000000
==============================================================
```

> `exact_recall = 0` is **expected** in hard mode — the query image is not in the gallery. Use `avg_clip_score` and `soft_recall` as the real performance indicators.

### JSON Report

```json
{
  "num_queries": 500,
  "gallery_size": 4500,
  "top_k": 5,
  "soft_recall_epsilon": 0.8,
  "keep_query_in_gallery": false,
  "metrics": {
    "avg_clip_score@5": 0.82143,
    "semantic_ndcg@5":  0.9732,
    "soft_recall@5":    0.912,
    "exact_recall@5":   0.0,
    "exact_precision@5": 0.0,
    "mrr@5":            0.0
  },
  "per_k_breakdown": {
    "R@1": 0.0, "P@1": 0.0,
    "R@3": 0.0, "P@3": 0.0,
    "R@5": 0.0, "P@5": 0.0
  }
}
```

---

## How to Interpret Results

| Metric | Hard Mode | Self-Retrieval Mode |
|---|---|---|
| `avg_clip_score@K` | Primary signal — how similar retrieved images are | Should be very high (>0.95) |
| `semantic_ndcg@K` | Primary ranking quality signal | Should approach 1.0 |
| `soft_recall@K` | What % of queries found a near-match | Should be very high (>0.95) |
| `exact_recall@K` | Expected ~0 (query not in gallery) | Primary signal — self-match rate |
| `mrr@K` | Expected ~0 in hard mode | Should be ~1.0 (rank-1 retrieval) |

### Comparing Text-to-Image vs Image-to-Image

| Metric | Text-to-Image (`evaluate.py`) | Image-to-Image (`evaluate_image.py`) |
|---|---|---|
| `avg_clip_score@5` | ~0.25 (cross-modal gap) | ~0.82 (same modality — much higher) |
| `soft_recall@5` | ~0.88 | ~0.91 (images are easier to match) |
| `exact_recall@5` | ~0.10 | ~0.00 in hard mode / ~1.0 in self-retrieval |

The higher CLIP scores in image-to-image retrieval are expected — intra-modal similarity is fundamentally easier than cross-modal.

---

## Dependencies

```
torch
torchvision
transformers          # HuggingFace — for CLIP ViT-B/32
faiss-cpu             # or faiss-gpu
Pillow
PyYAML
numpy
```

> **Optional:** Install the official OpenAI CLIP package for faster inference:
> ```bash
> pip install git+https://github.com/openai/CLIP.git
> ```
> The script auto-detects it and falls back to HuggingFace `transformers` if not present.

---

## Caching

The FAISS gallery index is built once and cached:

```
.cache/eval_img2img/
├── val2017_img2img.index   # FAISS IndexFlatIP
└── val2017_img2img.json    # Ordered image path list
```

> **Note:** Hard mode and self-retrieval mode have different galleries (different image sets).  
> Use `--rebuild_cache` when switching modes or after updating the checkpoint.
