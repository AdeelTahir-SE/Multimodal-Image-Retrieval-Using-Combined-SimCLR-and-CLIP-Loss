# Model Evaluation Guide

This file explains how your model is evaluated in this project and how to run it.

## What is being evaluated

The evaluation checks whether image and text embeddings learned by your model are aligned in one shared embedding space.

Two retrieval tasks are evaluated:

1. Text to Image
- Query is a caption.
- Target is the correct image.
- Measures semantic alignment quality between text and image embeddings.

2. Image to Image
- Query is one augmented image view.
- Target is another view of the same image.
- Measures visual invariance and representation consistency.

## Metrics being reported

For each retrieval task, the evaluator reports:

1. Recall at K (R@1, R@5, R@10)
- Fraction of queries where at least one correct item appears in top K results.
- Higher is better.
- Practical meaning:
	- R@1: how often the top result is correct.
	- R@5 and R@10: how often a correct match appears within shortlists.

2. Mean Reciprocal Rank (MRR)
- Average of 1/rank of the first correct result.
- Higher is better.
- Practical meaning: rewards putting the first correct match very early.

3. Median Rank (MedR)
- Median rank position of the first correct result.
- Lower is better.
- Practical meaning: robust central rank that is less sensitive to outliers.

4. Mean Rank (MeanR)
- Average rank position of the first correct result.
- Lower is better.
- Practical meaning: overall ranking quality, but can be skewed by hard failures.

5. Random baseline and Lift
- Random expected recall (RndR@K) is estimated analytically.
- Lift@K = R@K / RndR@K.
- Lift greater than 1 means better than random retrieval.
- Practical meaning: shows whether gains are meaningful beyond chance.

6. Stability across passes
- Evaluation can run multiple passes because image augmentations are stochastic.
- Mean and standard deviation are reported across passes.
- Lower std means more stable evaluation behavior.

## How this evaluator works internally

1. Load config and image caption pairs.
2. Build positive index mapping (all captions that belong to the same image path are treated as positives).
3. Load model checkpoint:
- image_encoder state dict
- text_encoder state dict
4. Encode all samples:
- image query embeddings (aug view 1)
- image gallery embeddings (aug view 2)
- text embeddings
5. Build cosine like similarity matrices using normalized embeddings:
- Image to Image: S = Z_img_query * Z_img_gallery^T
- Text to Image: S = Z_txt * Z_img_gallery^T
6. Compute retrieval metrics from similarity ranking.
7. Repeat for multiple passes and aggregate mean/std.

## How to run evaluation

Run the new evaluator script:

python evaluate_model.py --config config.yaml --checkpoint checkpoints/best.pt --k 1 5 10 --eval_passes 3

Fast sanity run (much quicker):

python evaluate_model.py --config config.yaml --checkpoint checkpoints/best.pt --quick

Custom fast run with subset size:

python evaluate_model.py --config config.yaml --checkpoint checkpoints/best.pt --eval_passes 1 --max_pairs 3000

Save report JSON to custom path:

python evaluate_model.py --config config.yaml --checkpoint checkpoints/best.pt --save_json checkpoints/my_eval_report.json

## Output

Console output includes:

1. Per task retrieval metrics (mean across passes)
2. Per task recall stability (std across passes)
3. A text verdict based on Text to Image quality

JSON report includes:

1. Run metadata (checkpoint, config, device, passes)
2. Full per metric mean/std values for each task
3. Final verdict string

## Troubleshooting

1. FileNotFoundError for data/coco_pairs.json
- Cause: pair file was not generated yet.
- Fix:
python data/preprocess.py --output data/coco_pairs.json

2. Checkpoint not found
- Cause: wrong checkpoint path.
- Fix: use the real path, for example:
python evaluate_model.py --config config.yaml --checkpoint checkpoints/best.pt --k 1 5 10 --eval_passes 3

3. Config path issues
- The evaluator resolves pairs_json relative to your config file location.
- Keep data.pairs_json in config as a path relative to that config file (recommended).

4. Evaluation is too slow
- Use quick mode for one-pass subset evaluation:
python evaluate_model.py --config config.yaml --checkpoint best.pt --quick
- Increase batch size if memory allows:
python evaluate_model.py --config config.yaml --checkpoint best.pt --batch_size 64 --eval_passes 1
- Reduce eval passes and subset pairs:
python evaluate_model.py --config config.yaml --checkpoint best.pt --eval_passes 1 --max_pairs 5000

## Interpreting results quickly

1. Strong early indicator:
- Text to Image R@1 and R@5 improve clearly over random baseline.

2. Robust model behavior:
- Lift@1 and Lift@5 are high.
- Std across passes is small.

3. Representation quality:
- Image to Image should typically be easier and higher than cross modal tasks.

## Optional qualitative check

Use retrieval CLI to inspect top matches manually:

Text query:
python retrieve.py --config config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_text "a dog running in grass" --top_k 10

Image query:
python retrieve.py --config config.yaml --checkpoint checkpoints/best.pt --image_dir data/coco/val2017 --query_image data/coco/val2017/000000033368.jpg --top_k 10

This helps confirm whether numerical gains also look semantically correct.
