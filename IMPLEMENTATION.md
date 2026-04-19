# IMPLEMENTATION.md

## Multimodal Image Retrieval Using Combined SimCLR and CLIP Loss

**Authors:** Adeel Tahir (456257) · Abdullah Sarafraz (455602)  
**Dataset:** MS-COCO 2017 (train split)  
**Framework:** PyTorch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Architecture](#5-architecture)
6. [Loss Functions](#6-loss-functions)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation](#8-evaluation)
9. [Configuration Reference](#9-configuration-reference)
10. [Known Issues & Decisions](#10-known-issues--decisions)

---

## 1. Project Overview

This project trains a unified multimodal retrieval model that handles both image and text queries within a single model. The training objective combines:

- **SimCLR Loss** — image-to-image contrastive learning across augmented view pairs
- **CLIP Loss** — image-to-text alignment via vision-language contrastive learning

```
Loss = α · L_SimCLR + β · L_CLIP
```

At inference time, the shared image encoder produces embeddings that are simultaneously view-invariant and semantically grounded in natural language.

---

## 2. Repository Structure

```
multimodal-retrieval/
├── data/
│   ├── download_coco.sh          # Script to download MS-COCO 2017
│   └── preprocess.py             # Caption pairing + dataset reduction
├── models/
│   ├── image_encoder.py          # Shared image encoder (ResNet/ViT backbone)
│   ├── text_encoder.py           # Text encoder (BERT/DistilBERT)
│   └── projection_head.py        # MLP projection heads for both encoders
├── losses/
│   ├── simclr_loss.py            # NT-Xent (normalized temperature-scaled cross-entropy)
│   └── clip_loss.py              # Symmetric cross-entropy over image-text similarity matrix
├── datasets/
│   └── coco_dataset.py           # PyTorch Dataset: returns (img_aug1, img_aug2, caption)
├── train.py                      # Main training loop
├── evaluate.py                   # Retrieval evaluation (Recall@K)
├── config.yaml                   # All hyperparameters
└── IMPLEMENTATION.md
```

---

## 3. Environment Setup

### Requirements

```bash
pip install torch torchvision transformers pycocotools pillow pyyaml tqdm
```

Tested with Python 3.10+, PyTorch 2.x, CUDA 11.8+.

### Quick Start

```bash
git clone <repo>
cd multimodal-retrieval
pip install -r requirements.txt

# Download and preprocess data
bash data/download_coco.sh
python data/preprocess.py --subset_size 20000  # reduced dataset

# Train
python train.py --config config.yaml
```

---

## 4. Dataset Preparation

### Download MS-COCO 2017

```bash
# data/download_coco.sh
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip -d data/coco/
unzip annotations_trainval2017.zip -d data/coco/
```

### Preprocessing (`data/preprocess.py`)

The script:
1. Loads `captions_train2017.json`
2. Builds an `image_id → first_caption` mapping
3. Produces `coco_pairs.json` — a flat list of `{ image_path, caption }` objects
4. Optionally samples a reduced subset via `--subset_size`

```python
# Output format: coco_pairs.json
[
  { "image_path": "data/coco/train2017/000000391895.jpg",
    "caption": "A man with a red helmet on a small moped on a dirt road." },
  ...
]
```

> **Why reduced?** Full MS-COCO (~118k images) was too slow for CLIP-only training in our initial run. A subset of ~20k images keeps epochs under 10 minutes on a single GPU.

---

## 5. Architecture

### Image Encoder

A ResNet-50 backbone (pretrained on ImageNet) with the final classification head removed. Outputs a 2048-d feature vector, projected to `embedding_dim` via an MLP head.

```python
# models/image_encoder.py
import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # strip fc
        self.projector = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.projector(feat)
```

### Text Encoder

DistilBERT with a linear projection head mapping `[CLS]` token output (768-d) to `embedding_dim`.

```python
# models/text_encoder.py
from transformers import DistilBertModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.projector = nn.Linear(768, embedding_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.projector(cls)
```

### Data Flow (single batch)

```
Image ──┬──[aug1]──► ImageEncoder ──► z_i  ─┐
        │                                    ├──► SimCLR Loss
        └──[aug2]──► ImageEncoder ──► z_j  ─┘

Image ──────[aug1]──► ImageEncoder ──► z_img ─┐
                                               ├──► CLIP Loss
Text  ──────────────► TextEncoder  ──► z_txt ─┘

Combined Loss = α · L_SimCLR + β · L_CLIP
```

---

## 6. Loss Functions

### SimCLR Loss (NT-Xent)

For a batch of `N` images, each image is augmented twice, yielding `2N` views. The positive pair for view `i` is its other augmented view. All other `2(N-1)` views in the batch are negatives.

```python
# losses/simclr_loss.py
import torch, torch.nn.functional as F

def simclr_loss(z_i, z_j, temperature=0.07):
    """
    z_i, z_j: (N, D) normalized embeddings of two augmented views
    """
    N = z_i.size(0)
    z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)   # (2N, D)
    sim = z @ z.T / temperature                              # (2N, 2N)

    # Mask out self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)
```

### CLIP Loss

Symmetric cross-entropy over the cosine similarity matrix of image and text embeddings. Each image is matched to its caption (positive) and all other captions in the batch (negatives).

```python
# losses/clip_loss.py
import torch, torch.nn.functional as F

def clip_loss(z_img, z_txt, temperature=0.07):
    """
    z_img, z_txt: (N, D) normalized embeddings
    """
    z_img = F.normalize(z_img, dim=1)
    z_txt = F.normalize(z_txt, dim=1)
    logits = z_img @ z_txt.T / temperature          # (N, N)
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

---

## 7. Training Pipeline

### Dataset (`datasets/coco_dataset.py`)

Each `__getitem__` returns `(aug1, aug2, input_ids, attention_mask)`.

```python
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image
import json, torch
from torch.utils.data import Dataset

AUGMENT = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class COCOPairDataset(Dataset):
    def __init__(self, pairs_json, max_length=77):
        self.pairs = json.load(open(pairs_json))
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        aug1, aug2 = AUGMENT(img), AUGMENT(img)
        tok = self.tokenizer(item["caption"], padding="max_length",
                             truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        return (aug1, aug2,
                tok["input_ids"].squeeze(0),
                tok["attention_mask"].squeeze(0))
```

### Training Loop (`train.py`)

```python
# Pseudocode — see train.py for full implementation
for epoch in range(cfg.epochs):
    for aug1, aug2, input_ids, attn_mask in dataloader:
        aug1, aug2 = aug1.to(device), aug2.to(device)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        z_i = image_enc(aug1)
        z_j = image_enc(aug2)
        z_txt = text_enc(input_ids, attn_mask)

        l_simclr = simclr_loss(z_i, z_j, temperature=cfg.temperature)
        l_clip   = clip_loss(z_i, z_txt, temperature=cfg.temperature)
        loss     = cfg.alpha * l_simclr + cfg.beta * l_clip

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 8. Evaluation

Retrieval is measured with **Recall@K** on a held-out validation split.

```bash
python evaluate.py --checkpoint checkpoints/best.pt --k 1 5 10
```

### Metrics

| Metric | Description |
|--------|-------------|
| Image→Image R@1 | % of image queries where the correct image is the top result |
| Image→Image R@5 | % of image queries where the correct image is in top-5 |
| Text→Image R@1 | % of text queries where the correct image is the top result |
| Text→Image R@5 | % of text queries where the correct image is in top-5 |

Baseline comparisons: CLIP-only model and SimCLR-only model trained under identical conditions.

---

## 9. Configuration Reference

```yaml
# config.yaml
data:
  pairs_json: data/coco_pairs.json
  subset_size: 20000
  batch_size: 128
  num_workers: 4

model:
  embedding_dim: 256
  image_backbone: resnet50     # or vit_b_16
  text_backbone: distilbert-base-uncased

loss:
  alpha: 0.5                   # SimCLR weight
  beta: 0.5                    # CLIP weight
  temperature: 0.07

training:
  epochs: 50
  lr: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 5
  checkpoint_dir: checkpoints/
```

Tune `alpha` and `beta` to shift emphasis between self-supervised visual invariance (`alpha`) and vision-language grounding (`beta`).

---

## 10. Known Issues & Decisions

**Full dataset training speed** — Initial CLIP-only training on all ~118k MS-COCO images was prohibitively slow. Dataset was reduced to ~20k samples for feasibility. Distributed training (`torch.nn.DataParallel` or DDP) is recommended for full-scale runs.

**Single caption per image** — MS-COCO provides 5 captions per image. Currently only the first caption is used. Using all 5 via random sampling during training could improve alignment.

**Temperature sensitivity** — SimCLR and CLIP both use `temperature=0.07` by default. The two losses may benefit from independent temperature values, especially when `alpha ≠ beta`.

**Augmentation overlap** — Both `z_i` (used in SimCLR) and `z_i` (used in CLIP) come from the same augmented view. An alternative is a dedicated, weaker augmentation for the CLIP branch to preserve semantic content for text alignment.

**Projection head sharing** — Image embeddings feed into both loss branches. Separate projection heads per loss (one for SimCLR, one for CLIP) can prevent gradient interference and is worth exploring.

---

*Last updated: April 2026*