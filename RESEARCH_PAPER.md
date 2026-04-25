# A Unified Multimodal Retrieval Framework for Image and Text Queries

**Authors:** [Author 1], [Author 2], [Author 3]  
**Affiliation:** [Department/University]  
**Email:** [author@domain.com]

## Abstract

Multimodal retrieval systems are often optimized for a single query modality, either text-to-image or image-to-image, which limits their practical deployment in mixed-query scenarios. In this work, we present a unified retrieval framework that learns a shared embedding space for both visual and textual inputs, enabling a single model to serve image and text queries. Our primary implemented approach combines image self-supervision and cross-modal alignment using a weighted objective: SimCLR loss for view-invariant visual representation learning and CLIP-style symmetric contrastive loss for image-text semantic grounding. The model is trained on MS-COCO image-caption pairs and evaluated using retrieval metrics including Recall at K, Mean Reciprocal Rank (MRR), Median Rank (MedR), and lift over random retrieval. We also define a second approach section as part of our research plan, reserved for future implementation and comparative analysis. This paper documents the motivation, architecture, training protocol, and evaluation methodology for robust multimodal retrieval.

**Keywords:** multimodal learning, image retrieval, text retrieval, contrastive learning, SimCLR, CLIP loss

## 1. Introduction

Modern retrieval applications require flexible query interfaces where users can search with natural language descriptions or with example images. Many existing systems specialize in one retrieval direction, creating fragmentation and maintenance overhead when deploying separate models for different query types.

Our objective is to train a single model that can process both image and text inputs while preserving retrieval quality across modalities. Specifically, we target:

1. A shared embedding space for image and text representations.
2. Robust semantic alignment for text-image matching.
3. Visual invariance to augmentations for stronger image-image similarity.
4. Improved retrieval accuracy and stability compared to random baselines.

To achieve this, we adopt a combined contrastive learning strategy and evaluate it in a unified retrieval setting.

## 2. Problem Formulation

Given a dataset of paired samples

$$
\mathcal{D} = \{(x_i, t_i)\}_{i=1}^{N},
$$

where $x_i$ is an image and $t_i$ is its caption, we learn two encoders:

$$
f_\theta: x \rightarrow \mathbb{R}^{d}, \quad g_\phi: t \rightarrow \mathbb{R}^{d},
$$

followed by projection into a normalized shared space. The system should support:

1. Text-to-Image retrieval.
2. Image-to-Text retrieval.
3. Image-to-Image retrieval.

for a single trained checkpoint.

## 3. Dataset and Preprocessing

We use the MS-COCO 2017 training split and build flattened image-caption pairs. Preprocessing performs:

1. Parsing COCO caption annotations.
2. Mapping each image id to a selected caption (first caption policy).
3. Generating JSON records of `{image_path, caption}`.
4. Optional subset sampling for resource-constrained training.

This enables efficient batching for multimodal contrastive learning.

## 4. Proposed Methods

### 4.1 Approach 1 (Implemented): Joint SimCLR + CLIP-Style Training

Our implemented method combines two complementary objectives.

**A. SimCLR objective (image-image):**

Two augmented views are generated per image, then optimized with NT-Xent:

$$
\mathcal{L}_{\text{SimCLR}} = -\frac{1}{2N}\sum_{i=1}^{2N}\log \frac{\exp(\text{sim}(z_i, z_{p(i)})/\tau_s)}{\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau_s)}
$$

where $p(i)$ denotes the positive pair index, and $\tau_s$ is temperature.

**B. CLIP-style objective (image-text):**

Given paired image and text embeddings in a batch:

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left[\text{CE}(S, y) + \text{CE}(S^\top, y)\right],
$$

where $S = Z_{img} Z_{txt}^{\top}/\tau_c$ is similarity logits, $y$ are identity labels, and $\tau_c$ is temperature.

**Combined loss:**

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{SimCLR}} + \beta \cdot \mathcal{L}_{\text{CLIP}}.
$$

#### 4.1.1 Architecture

1. Image encoder: configurable visual backbone (ResNet-50 or ViT-B/16) with projection head.
2. Text encoder: DistilBERT-based encoder with projection head.
3. Shared embedding dimensionality: configured in YAML (example: 128).
4. Embedding normalization before similarity computation.

#### 4.1.2 Data Augmentation

Training-time image augmentations include random resized crop, horizontal flip, color jitter, grayscale, blur, tensor conversion, and normalization.

#### 4.1.3 Training Pipeline

1. Load paired dataset and tokenize captions.
2. Generate two augmented image views per sample.
3. Compute image, augmented-image, and text embeddings.
4. Optimize weighted loss with AdamW.
5. Save latest and best checkpoints by epoch-level loss.

### 4.2 Approach 2 (Reserved for Comparative Study)

This section is intentionally left as research space for the second approach.

**Planned title:** [Approach 2 Name]  
**Planned objective:** [Describe objective]  
**Planned key idea:** [Describe method intuition]  
**Planned loss function:** [Add equations]  
**Planned architecture changes:** [Add modules/backbones]  
**Planned training differences:** [Add schedule, augmentations, mining, etc.]

#### 4.2.1 Method Description (To Be Added)

[Write detailed algorithmic steps and theoretical motivation here.]

#### 4.2.2 Expected Advantages and Trade-offs (To Be Added)

[Explain why Approach 2 may improve accuracy, efficiency, or robustness, and identify expected costs.]

## 5. Experimental Setup

### 5.1 Hardware and Software

1. Framework: PyTorch.
2. Dataset: MS-COCO processed pairs.
3. Config-driven hyperparameters from YAML files.

### 5.2 Hyperparameters (Approach 1)

Typical settings used in this repository include:

1. Batch size: 16-32.
2. Embedding dimension: 128.
3. SimCLR temperature: 0.1.
4. CLIP temperature: 0.07.
5. Loss weights: $\alpha = 0.5$, $\beta = 0.5$.
6. Optimizer: AdamW.

### 5.3 Evaluation Protocol

We evaluate:

1. Text-to-Image retrieval.
2. Image-to-Text retrieval.
3. Image-to-Image retrieval.

Metrics:

1. $R@K$ for $K \in \{1, 5, 10\}$.
2. Mean Reciprocal Rank (MRR).
3. Median Rank (MedR).
4. Mean Rank (MeanR).
5. Lift over random baseline.
6. Multi-pass stability (standard deviation across runs).

## 6. Results

### 6.1 Quantitative Results: Approach 1

| Task | R@1 | R@5 | R@10 | MRR | MedR | MeanR |
|---|---:|---:|---:|---:|---:|---:|
| Text->Image | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Image->Text | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Image->Image | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 6.2 Quantitative Results: Approach 2 (Reserved)

| Task | R@1 | R@5 | R@10 | MRR | MedR | MeanR |
|---|---:|---:|---:|---:|---:|---:|
| Text->Image | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] |
| Image->Text | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] |
| Image->Image | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] | [Space for Approach 2] |

### 6.3 Comparative Summary (Reserved)

| Metric | Approach 1 | Approach 2 | Delta |
|---|---:|---:|---:|
| Text->Image R@1 | [TBD] | [TBD] | [TBD] |
| Text->Image R@5 | [TBD] | [TBD] | [TBD] |
| Image->Text R@1 | [TBD] | [TBD] | [TBD] |
| Image->Image R@1 | [TBD] | [TBD] | [TBD] |
| Inference Cost | [TBD] | [TBD] | [TBD] |

## 7. Discussion

The combined objective is designed to solve two issues jointly: intra-modal visual robustness and cross-modal semantic alignment. SimCLR regularizes image representations to be augmentation-invariant, while CLIP-style alignment anchors those representations to language semantics. This supports mixed-query retrieval with one model.

Potential limitations include dependence on caption quality, computational cost for contrastive negatives at larger batch sizes, and sensitivity to loss balancing coefficients.

## 8. Conclusion and Future Work

We presented a unified multimodal retrieval framework that supports both image and text queries through a shared embedding space. The implemented Approach 1 combines SimCLR and CLIP-style objectives to improve retrieval relevance and robustness in a single training pipeline.

Future work will complete and benchmark Approach 2, perform ablation studies on loss weighting and backbone choices, and report full comparative results on standardized splits.

## 9. Reproducibility Checklist

1. Training script: available.
2. Evaluation script with multi-pass diagnostics: available.
3. Config files with hyperparameters: available.
4. Data preprocessing script: available.
5. Checkpoint loading and retrieval inference scripts: available.

## 10. Suggested Citation (Template)

```bibtex
@article{yourteam2026multimodal,
  title={A Unified Multimodal Retrieval Framework for Image and Text Queries},
  author={[Author 1] and [Author 2] and [Author 3]},
  journal={[Conference/Journal Name]},
  year={2026}
}
```
