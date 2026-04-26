# Multimodal Image Retrieval (SimCLR + CLIP)

This project trains a shared embedding space for image and text retrieval with a weighted SimCLR + CLIP objective.

# Multimodal Image Retrieval (SimCLR + CLIP)

This project trains a shared embedding space for image and text retrieval, then serves retrieval via CLI and a desktop UI.

## 1. Environment Setup

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

The script downloads and extracts:

- train2017 images
- val2017 images
- annotations_trainval2017

Run:

```bash
bash data/download_coco.sh
```

After completion, you should have:

- data/coco/train2017
- data/coco/val2017
- data/coco/annotations/captions_train2017.json
- data/coco/annotations/captions_val2017.json

## 3. Build Training Pairs JSON

Build flattened image-caption pairs used by training/evaluation.

Small subset:

```bash
python data/preprocess.py --subset_size 20000 --output data/coco_pairs.json
```

Full train captions:

```bash
python data/preprocess.py --output data/coco_pairs.json
```

## 4. Configure

Edit config file (for example config.yaml or collab.config.yaml):

- data.pairs_json
- data.batch_size
- data.num_workers
- model settings
- training settings

## 5. Train

```bash
python train.py --config config.yaml
```

Outputs in checkpoint_dir (default checkpoints):

- latest.pt
- best.pt
- train_metrics.jsonl

## 6. Evaluate Model Quality

Use the rewritten evaluator for stronger diagnostics.

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --k 1 5 10 --eval_passes 3 --num_examples 5
```

For manual labeling of retrieved results, launch the UI:

```bash
python evaluate_ui.py
```

What it reports:

- Image->Image, Text->Image, and Image->Text metrics
- R@K, random baseline, and lift over random
- MRR, MedR, MeanR
- Stability across passes (std), useful because eval uses stochastic image views
- A plain-language verdict (Strong / Promising / Weak / Poor)
- Qualitative text->image examples

Optional JSON report:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --save_json checkpoints/eval_report.json
```

## 7. Retrieval from Command Line

Text query:

```bash
python retrieve.py --config collab.config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_text "an athlete playing tennis" --top_k 50
```

Image query:

```bash
python retrieve.py --config collab.config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_image data/coco/val2017/000000033368.jpg --top_k 50
```

Notes:

- Gallery indexing is cached under .cache/retrieval by default
- Use --rebuild_cache to force refresh
- Progress bar is shown while indexing images

## 8. Retrieval Desktop UI

Run:

```bash
python retrieve_ui.py
```

In the UI:

1. Set config/checkpoint/image_dir/cache_dir
2. Click Load Index
3. Choose Text or Image Path query
4. Click Search

The UI renders result images with score and path instead of only printing ranked lines.

## 9. Suggested End-to-End Flow

1. Download COCO data
2. Build pairs JSON
3. Train model
4. Evaluate model (check verdict + lift over random)
5. Run retrieve.py for quick checks
6. Use retrieve_ui.py for visual inspection

## Project Files

- data/preprocess.py: COCO caption pairing and subset sampling
- datasets/coco_dataset.py: image views + caption tokenization
- models/image_encoder.py: image encoder + projection
- models/text_encoder.py: text encoder + projection
- losses/simclr_loss.py: NT-Xent loss
- losses/clip_loss.py: CLIP-style symmetric contrastive loss
- train.py: training loop and checkpointing
- evaluate.py: robust retrieval evaluation
- retrieve.py: CLI retrieval (text/image query)
- retrieve_ui.py: desktop retrieval UI
```bash
python retrieve.py --config collab.config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_text "an athlete playing tennis" --top_k 20
```

Image query:

```bash
python retrieve.py --config collab.config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_image data/coco/val2017/000000033368.jpg --top_k 20
```

Use --rebuild_cache when you change image_dir contents or checkpoint and want to force refresh.

## 8. Retrieve with UI

```bash
python retrieve_ui.py
```

In the UI:

1. Click Load Index.
2. Choose Text or Image Path mode.
3. Enter query and click Search.
4. Scroll through ranked image cards.

## 9. Quick Smoke Test

For a fast sanity run, lower in config temporarily:

- training.epochs: 1
- data.batch_size: 8 or 16
- data.num_workers: 0

Then run train -> evaluate -> retrieve.

## Main Scripts

- train.py: training loop and checkpointing
- evaluate.py: robust retrieval evaluation with verdict and baselines
- retrieve.py: CLI retrieval against an image folder
- retrieve_ui.py: GUI retrieval viewer
- data/preprocess.py: COCO pair generation
