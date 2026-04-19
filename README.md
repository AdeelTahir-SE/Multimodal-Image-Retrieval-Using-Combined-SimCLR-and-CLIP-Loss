# Multimodal Image Retrieval (SimCLR + CLIP)

This project trains a shared embedding space for image and text retrieval using a weighted combination of SimCLR and CLIP losses.

## 1. Setup

Create and activate a virtual environment, then install dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download COCO Data

The provided script is bash-based.

### Linux or macOS

```bash
bash data/download_coco.sh
```

### Windows

Use Git Bash, WSL, or manually download these files into data/coco:

- http://images.cocodataset.org/zips/train2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After download, extract so these paths exist:

- data/coco/train2017
- data/coco/annotations/captions_train2017.json

## 3. Build Training Pairs JSON

Creates image-caption pairs from COCO captions and optionally samples a subset.

```bash
python data/preprocess.py --subset_size 20000 --output data/coco_pairs.json
```

If you want full train set pairs, omit subset_size:

```bash
python data/preprocess.py --output data/coco_pairs.json
```

## 4. Configure Training

Edit config.yaml as needed:

- data.pairs_json: path to generated pairs file
- data.batch_size: reduce if GPU memory is limited
- model.image_pretrained and model.text_pretrained: default false
- loss.alpha and loss.beta: weight SimCLR vs CLIP

## 5. Train

```bash
python train.py --config config.yaml
```

Outputs:

- checkpoints/latest.pt
- checkpoints/best.pt
- checkpoints/train_metrics.jsonl

## 6. Evaluate Retrieval

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --k 1 5 10
```

Printed metrics:

- Image->Image Recall@K
- Text->Image Recall@K

## 7. Quick Smoke Test (Optional)

For a quick sanity run, temporarily lower in config.yaml:

- training.epochs: 1
- data.batch_size: 8 or 16
- data.num_workers: 0 (helps on some Windows setups)

Then run train and evaluate commands above.

## Project Files

- data/preprocess.py: COCO caption pairing and subset sampling
- datasets/coco_dataset.py: two image augmentations + caption tokenization
- models/image_encoder.py: image backbone + projection
- models/text_encoder.py: DistilBERT + projection
- losses/simclr_loss.py: NT-Xent loss
- losses/clip_loss.py: CLIP-style symmetric contrastive loss
- train.py: training loop and checkpointing
- evaluate.py: Recall@K evaluation
