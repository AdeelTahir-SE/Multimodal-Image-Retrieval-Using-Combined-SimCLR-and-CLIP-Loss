# Multimodal Image Retrieval (SimCLR + CLIP Loss)

This repository implements a unified multimodal retrieval model trained with a weighted combination of:

- SimCLR NT-Xent loss for image-image invariance
- CLIP-style symmetric contrastive loss for image-text alignment

## Project Structure

- data/download_coco.sh: Download MS-COCO 2017 train split and annotations
- data/preprocess.py: Build flattened image-caption pairs JSON
- datasets/coco_dataset.py: Dataset returning two image views + tokenized caption
- models/image_encoder.py: Vision encoder with projection head
- models/text_encoder.py: DistilBERT encoder with projection head
- losses/simclr_loss.py: NT-Xent implementation
- losses/clip_loss.py: Symmetric CLIP-style contrastive loss
- train.py: End-to-end training script
- evaluate.py: Retrieval evaluation with Recall@K
- config.yaml: Hyperparameters and runtime settings

## Install

```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
bash data/download_coco.sh
python data/preprocess.py --subset_size 20000 --output data/coco_pairs.json
```

## Train

```bash
python train.py --config config.yaml
```

## Evaluate

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --k 1 5 10
```

## Notes

- Defaults are non-pretrained (`image_pretrained: false`, `text_pretrained: false`).
- You can enable pretrained backbones in config for faster convergence.
- `subset_size` in config is informational; actual subset generation is controlled by `data/preprocess.py`.
