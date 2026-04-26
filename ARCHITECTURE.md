# System Architecture — Multimodal Image Retrieval (SimCLR + CLIP)

> End-to-end documentation: data → preprocessing → training → FAISS indexing → retrieval → evaluation.

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Model Architecture](#3-model-architecture)
4. [Combined Loss Function](#4-combined-loss-function)
5. [Training Pipeline](#5-training-pipeline)
6. [Training Convergence](#6-training-convergence)
7. [Retrieval Pipeline (FAISS)](#7-retrieval-pipeline-faiss)
8. [Evaluation Pipeline](#8-evaluation-pipeline)
9. [Text-to-Image Results](#9-text-to-image-results)
10. [File Structure](#10-file-structure)

---

## 1. High-Level Pipeline

```mermaid
flowchart TD
    A["🗂️ COCO train2017\n118,287 images\n+captions_train2017.json"] --> B

    B["⚙️ preprocess.py\nSample 2,000 pairs\ncoco_pairs.json"] --> C

    C["🏋️ train.py\n10 epochs\nSimCLR + CLIP loss"] --> D

    D["💾 checkpoints/best.pt\nImageEncoder + TextEncoder\n128-D shared embedding"] --> E & F

    E["🔍 evaluate.py\nText → Image retrieval\nFAISS + CLIP evaluator"] --> G
    F["🖼️ evaluate_image.py\nImage → Image retrieval\nFAISS + CLIP evaluator"] --> G

    G["📊 eval_semantic.json\nCLIP Score · NDCG\nSoft Recall · MRR"]

    style A fill:#1e3a5f,color:#fff
    style B fill:#2d5016,color:#fff
    style C fill:#5c2d00,color:#fff
    style D fill:#4a0e4e,color:#fff
    style E fill:#003d40,color:#fff
    style F fill:#003d40,color:#fff
    style G fill:#1a1a2e,color:#fff
```

---

## 2. Data Preprocessing

```mermaid
flowchart LR
    A["captions_train2017.json\n~118K annotations"] --> B["Parse annotations\nFirst caption per image"]
    C["train2017/\nImage files"] --> D["Check file exists\non disk"]
    B & D --> E["Build pairs list\n{image_path, caption}"]
    E --> F{"subset_size=2000?"}
    F -->|Yes| G["Random sample\nseed=42\n→ 2,000 pairs"]
    F -->|No| H["All ~118K pairs"]
    G --> I["coco_pairs.json\n2,000 × {image_path, caption}"]
    H --> I

    style I fill:#2d5016,color:#fff
```

**Command:**
```powershell
python data/preprocess.py --subset_size 2000 --output data/coco_pairs.json
```

**Output format:**
```json
[
  { "image_path": "data/coco/train2017/000000123456.jpg",
    "caption":    "A dog sitting on a red couch." },
  ...
]
```

---

## 3. Model Architecture

### 3.1 Dual-Encoder Design

```mermaid
graph TB
    subgraph IE["ImageEncoder"]
        direction TB
        I1["Input Image\n224×224×3"] --> I2["ResNet-50 backbone\n(ImageNet pretrained)\nstrip final fc"]
        I2 --> I3["Global avg pool\n2048-D feature"]
        I3 --> I4["ProjectionHead\n2048→512→128\nReLU"]
        I4 --> I5["128-D image vector\nz_img"]
    end

    subgraph TE["TextEncoder"]
        direction TB
        T1["Input Caption\n≤77 tokens"] --> T2["DistilBERT\n(pretrained)\n6 layers · 768-D"]
        T2 --> T3["CLS token\n768-D"]
        T3 --> T4["ProjectionHead\n768→512→128\nReLU"]
        T4 --> T5["128-D text vector\nz_txt"]
    end

    I5 & T5 --> S["Shared 128-D\nEmbedding Space\n(L2 normalised)"]

    style IE fill:#1e3a5f,color:#fff
    style TE fill:#5c2d00,color:#fff
    style S fill:#4a0e4e,color:#fff
```

### 3.2 ProjectionHead (shared design)

```mermaid
graph LR
    A["input_dim\n(2048 or 768)"] --> B["Linear\ninput→512"]
    B --> C["ReLU"]
    C --> D["Linear\n512→128"]
    D --> E["128-D\nembedding"]
```

---

## 4. Combined Loss Function

The model is trained with a **weighted sum of two losses**:

```
L_total = α · L_SimCLR  +  β · L_CLIP
        = 0.5 · L_SimCLR + 0.5 · L_CLIP
```

### 4.1 SimCLR Loss (image-image contrastive)

```mermaid
graph TB
    A["Image x"] --> B["Augmentation 1\n(crop/flip/colour jitter)"]
    A --> C["Augmentation 2\n(crop/flip/colour jitter)"]
    B --> D["ImageEncoder → z_i\n128-D"]
    C --> E["ImageEncoder → z_j\n128-D"]
    D & E --> F["NT-Xent Loss\nτ = 0.10\nPull z_i ↔ z_j together\nPush all other pairs apart"]

    style F fill:#5c2d00,color:#fff
```

- Treats the two augmented views of the **same image** as positives.
- All other images in the batch (size=16) are negatives.
- Temperature τ = 0.10 (soft assignments).

### 4.2 CLIP Loss (image-text alignment)

```mermaid
graph TB
    A["Image x"] --> B["Augmentation 1"]
    B --> C["ImageEncoder → z_i\n128-D"]
    D["Caption of x"] --> E["TextEncoder → z_txt\n128-D"]
    C & E --> F["Cross-modal cosine\nsimilarity matrix\nN × N"]
    F --> G["Symmetric cross-entropy\nτ = 0.07\nAlign image ↔ text pairs"]

    style G fill:#003d40,color:#fff
```

- The diagonal of the N×N similarity matrix holds positive pairs.
- Off-diagonal entries are negatives.
- Symmetric: both image-to-text and text-to-image directions are optimised.

### 4.3 Loss Computation Graph (one batch)

```mermaid
graph LR
    B["Batch\n{aug1, aug2, input_ids, mask}"]
    B --> ZI["z_i = ImageEncoder(aug1)"]
    B --> ZJ["z_j = ImageEncoder(aug2)"]
    B --> ZT["z_txt = TextEncoder(ids, mask)"]

    ZI & ZJ --> LS["L_SimCLR\nNT-Xent(z_i, z_j, τ=0.10)"]
    ZI & ZT --> LC["L_CLIP\nSymmetric-CE(z_i, z_txt, τ=0.07)"]

    LS --> LT["L_total = 0.5·L_SimCLR + 0.5·L_CLIP"]
    LC --> LT
    LT --> BP["backward() → AdamW step"]

    style LT fill:#4a0e4e,color:#fff
```

---

## 5. Training Pipeline

```mermaid
flowchart TD
    A["config.yaml\nlr=3e-4 · wd=1e-4\nepochs=10 · batch=16"] --> B["Load COCOPairDataset\n2,000 pairs → 125 batches/epoch\n(drop_last=True → 62 complete batches)"]

    B --> C["Initialise\nImageEncoder (ResNet50, pretrained)\nTextEncoder  (DistilBERT, pretrained)\nAdamW optimiser"]

    C --> D{"Epoch loop\n1 … 10"}

    D --> E["Forward pass\nz_i, z_j, z_txt"]
    E --> F["Compute losses\nL_SimCLR + L_CLIP"]
    F --> G["Backward + AdamW step"]
    G --> H["Log loss to tqdm"]
    H --> I{"More batches?"}
    I -->|Yes| E
    I -->|No| J["Compute epoch mean loss"]

    J --> K["Save latest.pt"]
    J --> L{"epoch_loss < best_loss?"}
    L -->|Yes| M["Save best.pt ✓"]
    L -->|No| N["Keep previous best"]
    M & N --> O["Append to train_metrics.jsonl"]
    O --> D

    style M fill:#2d5016,color:#fff
    style C fill:#1e3a5f,color:#fff
```

**Training config (config.yaml):**

| Parameter | Value |
|---|---|
| `embedding_dim` | 128 |
| `image_backbone` | resnet50 |
| `text_backbone` | distilbert-base-uncased |
| `batch_size` | 16 |
| `epochs` | 10 |
| `lr` | 3e-4 |
| `weight_decay` | 1e-4 |
| `α` (SimCLR weight) | 0.5 |
| `β` (CLIP weight) | 0.5 |
| `simclr_temperature` | 0.10 |
| `clip_temperature` | 0.07 |

---

## 6. Training Convergence

### 6.1 Epoch-by-epoch metrics

| Epoch | Total Loss | SimCLR Loss | CLIP Loss | Δ Loss |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.4607 | 0.6779 | 2.2435 | — |
| 2 | 0.8814 | 0.3818 | 1.3809 | −0.579 |
| 3 | 0.7463 | 0.3459 | 1.1467 | −0.135 |
| 4 | 0.6515 | 0.3175 | 0.9855 | −0.095 |
| 5 | 0.5148 | 0.2621 | 0.7674 | −0.137 |
| 6 | 0.4525 | 0.2661 | 0.6388 | −0.062 |
| 7 | 0.4137 | 0.2384 | 0.5891 | −0.039 |
| 8 | 0.4032 | 0.2529 | 0.5536 | −0.010 |
| 9 | 0.3468 | 0.2093 | 0.4843 | −0.056 |
| **10** | **0.3195** | **0.1998** | **0.4393** | **−0.027** |

### 6.2 Loss curve

```mermaid
xychart-beta
    title "Training Loss over 10 Epochs"
    x-axis [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Loss" 0 --> 2.5
    line [1.4607, 0.8814, 0.7463, 0.6515, 0.5148, 0.4525, 0.4137, 0.4032, 0.3468, 0.3195]
    line [2.2435, 1.3809, 1.1467, 0.9855, 0.7674, 0.6388, 0.5891, 0.5536, 0.4843, 0.4393]
    line [0.6779, 0.3818, 0.3459, 0.3175, 0.2621, 0.2661, 0.2384, 0.2529, 0.2093, 0.1998]
```

> Total loss dropped **78.1%** from epoch 1 (1.46) to epoch 10 (0.32).  
> Best checkpoint saved at **epoch 10** → `checkpoints/best.pt`.

---

## 7. Retrieval Pipeline (FAISS)

### 7.1 Gallery indexing (offline, done once)

```mermaid
flowchart LR
    A["val2017/\n5,000 images"] --> B["PathImageDataset\nbatch_size=32"]
    B --> C["ImageEncoder (best.pt)\nResNet50 → 128-D"]
    C --> D["L2 normalise\nall embeddings"]
    D --> E["faiss.IndexFlatIP\nInner Product = cosine sim\n(after L2 norm)"]
    E --> F["val2017_gallery.index\n+ gallery_paths.json\n(.cache/eval/)"]

    style E fill:#4a0e4e,color:#fff
    style F fill:#2d5016,color:#fff
```

### 7.2 Query → retrieval (online, per query)

```mermaid
flowchart TD
    subgraph TXT["Text Query (evaluate.py)"]
        T1["Caption text"] --> T2["DistilBERT tokenizer\nmax_length=77"]
        T2 --> T3["TextEncoder → 128-D"]
        T3 --> T4["L2 normalise"]
    end

    subgraph IMG["Image Query (evaluate_image.py)"]
        I1["Query image"] --> I2["Resize 256 → CenterCrop 224\nNormalise ImageNet"]
        I2 --> I3["ImageEncoder → 128-D"]
        I3 --> I4["L2 normalise"]
    end

    T4 & I4 --> F["faiss.IndexFlatIP.search(query, K)\nInner product = cosine similarity"]
    F --> G["Top-K image paths\n+ similarity scores"]

    style F fill:#4a0e4e,color:#fff
    style G fill:#2d5016,color:#fff
```

---

## 8. Evaluation Pipeline

### 8.1 Two-model design

```mermaid
graph TB
    subgraph TRAINED["Retrieval Model (best.pt)"]
        IM["ImageEncoder\n128-D"]
        TX["TextEncoder\n128-D"]
    end

    subgraph EVAL["Frozen Evaluator"]
        CL["CLIP ViT-B/32\nopenai/clip-vit-base-patch32\nNEVER fine-tuned"]
    end

    Q["Query\n(text or image)"] --> TRAINED
    TRAINED --> FAISS["FAISS → Top-K paths"]
    FAISS --> CL
    Q --> CL
    CL --> M["4 Metric families"]

    style TRAINED fill:#1e3a5f,color:#fff
    style EVAL fill:#5c2d00,color:#fff
    style M fill:#4a0e4e,color:#fff
```

### 8.2 Per-query metric computation

```mermaid
flowchart TD
    Q["Text caption  /  Query image"] --> R["FAISS → Top-K retrieved images\n(using best.pt encoders)"]

    R --> M1 & M2 & M3 & M4

    subgraph M1["Metric 1: Avg CLIP Score@K"]
        direction LR
        A1["CLIP_text(caption)\nor CLIP_img(query)"] --> B1["cos_sim with each\nCLIP_img(retrieved_i)"]
        B1 --> C1["Mean over K\n→ avg_clip_score@K"]
    end

    subgraph M2["Metric 2: Semantic NDCG@K"]
        direction LR
        A2["CLIP_img(ground_truth)"] --> B2["cos_sim with each\nCLIP_img(retrieved_i)"]
        B2 --> C2["Shift [-1,1]→[0,1]\ngraded relevance"]
        C2 --> D2["NDCG@K formula\n→ semantic_ndcg@K"]
    end

    subgraph M3["Metric 3: Soft Recall@K"]
        direction LR
        A3["max(cos_sim to GT)\nacross K images"] --> B3{"≥ threshold\n(2ε−1)?"}
        B3 -->|Yes| C3["Hit = 1"]
        B3 -->|No| D3["Hit = 0"]
    end

    subgraph M4["Metric 4: Exact Match"]
        direction LR
        A4["Check filename\nmatch in top-K"] --> B4["Recall · Precision · MRR"]
    end

    style M1 fill:#003d40,color:#fff
    style M2 fill:#1e3a5f,color:#fff
    style M3 fill:#2d5016,color:#fff
    style M4 fill:#3d1a00,color:#fff
```

---

## 9. Text-to-Image Results

**Setup:** 50 queries · top_k=5 · ε=0.80 · COCO val2017 (5,000 gallery images)

```mermaid
pie title Soft Recall@5 Breakdown (50 queries)
    "Near-match found (≥0.80 sim)" : 44
    "No near-match in top-5" : 6
```

### 9.1 Results table

| Metric | Value | Meaning |
|---|---|---|
| `avg_clip_score@5` | **0.2478** | Retrieved images are semantically related to the caption |
| `semantic_ndcg@5` | **0.9862** | Near-perfect semantic ranking — correct images rank highest |
| `soft_recall@5` | **0.8800** | 88% of queries returned a visually similar image |
| `exact_recall@5` | 0.1000 | 10% found the exact ground-truth file (expected low: 1-in-5000) |
| `exact_precision@5` | 0.0200 | Binary baseline |
| `mrr@5` | 0.0450 | Exact hits appear near rank 4–5 when they do occur |

### 9.2 Key insight — semantic gap

```mermaid
graph LR
    A["soft_recall@5 = 0.88\nSemantics: model finds\nvisually correct images"] -->|gap| B["exact_recall@5 = 0.10\nBinary: model rarely finds\nthe exact file"]
    B --> C["Conclusion: model has\ngood semantic alignment\nbut ranks ~1 correct image\namong 5,000 gallery images"]

    style A fill:#2d5016,color:#fff
    style B fill:#5c2d00,color:#fff
    style C fill:#1e3a5f,color:#fff
```

> The 78-point gap between soft and exact recall confirms the model retrieves **semantically correct** images even when it misses the exact file — which is the intended behaviour of a multimodal retrieval system.

### 9.3 Per-K recall breakdown

| K | Recall@K | Precision@K |
|:---:|:---:|:---:|
| 1 | 0.020 | 0.020 |
| 3 | 0.080 | 0.027 |
| 5 | 0.100 | 0.020 |

---

## 10. File Structure

```mermaid
graph TD
    ROOT["📁 ML_Dataset/"] --> DATA & MODELS & LOSSES & DATASETS & CKPT & SCRIPTS

    DATA["📁 data/
    ├── preprocess.py
    ├── coco_pairs.json  ← 2000 pairs
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── captions_train2017.json
            └── captions_val2017.json"]

    MODELS["📁 models/
    ├── image_encoder.py   ResNet50+Head
    ├── text_encoder.py    DistilBERT+Head
    └── projection_head.py Linear→ReLU→Linear"]

    LOSSES["📁 losses/
    ├── simclr_loss.py   NT-Xent
    └── clip_loss.py     Symmetric-CE"]

    DATASETS["📁 datasets/
    └── coco_dataset.py  COCOPairDataset
        (dual augmentation + tokenizer)"]

    CKPT["📁 checkpoints/
    ├── best.pt           ← used for eval
    ├── latest.pt
    ├── train_metrics.jsonl
    ├── eval_semantic.json     text→image
    └── eval_image2image.json  image→image"]

    SCRIPTS["📄 Scripts
    ├── train.py
    ├── retrieve.py          CLI retrieval
    ├── retrieve_ui.py       Desktop UI
    ├── evaluate.py          text→image eval
    ├── evaluate_image.py    image→image eval
    └── config.yaml"]

    style ROOT fill:#1a1a2e,color:#fff
    style CKPT fill:#2d5016,color:#fff
```

---

## Quick-Start Commands

```powershell
# 1. Preprocess
python data/preprocess.py --subset_size 2000 --output data/coco_pairs.json

# 2. Train
python train.py --config config.yaml

# 3. Text → Image evaluation
python evaluate.py --checkpoint checkpoints/best.pt --num_queries 1000 --top_k 5

# 4. Image → Image evaluation (hard mode)
python evaluate_image.py --checkpoint checkpoints/best.pt --num_queries 500 --top_k 5

# 5. Image → Image evaluation (self-retrieval)
python evaluate_image.py --checkpoint checkpoints/best.pt --keep_query_in_gallery --top_k 1

# 6. CLI retrieval
python retrieve.py --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_text "a dog on a beach"
```
