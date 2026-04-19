import argparse
import json
import os
import random
from typing import Dict, List


def build_image_caption_pairs(
    captions_path: str,
    images_root: str,
    subset_size: int | None = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Use the first caption per image id to match the implementation doc.
    id_to_caption: Dict[int, str] = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in id_to_caption:
            id_to_caption[image_id] = ann["caption"].strip()

    pairs: List[Dict[str, str]] = []
    for image_id, caption in id_to_caption.items():
        file_name = id_to_filename.get(image_id)
        if not file_name:
            continue

        image_path = os.path.join(images_root, file_name)
        if not os.path.exists(image_path):
            continue

        pairs.append({"image_path": image_path, "caption": caption})

    if subset_size is not None and 0 < subset_size < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, subset_size)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COCO image-caption pairs JSON")
    parser.add_argument(
        "--captions_json",
        default="data/coco/annotations/captions_train2017.json",
        help="Path to captions_train2017.json",
    )
    parser.add_argument(
        "--images_root",
        default="data/coco/train2017",
        help="Path to COCO train2017 image directory",
    )
    parser.add_argument(
        "--output",
        default="data/coco_pairs.json",
        help="Output path for flattened pairs JSON",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Optional number of pairs to sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pairs = build_image_caption_pairs(
        captions_path=args.captions_json,
        images_root=args.images_root,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()