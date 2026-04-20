import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import faiss
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_TRANSFORM = transforms.Compose(
	[
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	]
)


def _to_numpy_float32(tensor: torch.Tensor):
	return tensor.detach().cpu().contiguous().numpy().astype("float32", copy=False)


class FolderImageDataset(Dataset):
	def __init__(self, image_paths: Sequence[Path]):
		self.image_paths = list(image_paths)

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
		path = self.image_paths[idx]
		image = Image.open(path).convert("RGB")
		return IMAGE_TRANSFORM(image), str(path)


def load_config(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def discover_images(image_dir: Path, extensions: Iterable[str]) -> List[Path]:
	allowed = {ext.lower() for ext in extensions}
	return sorted(
		[p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
	)


def build_cache_key(image_dir: Path, checkpoint: Path, embedding_dim: int) -> str:
	image_paths = sorted(
		[p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
	)
	basis = {
		"image_dir": str(image_dir.resolve()),
		"checkpoint": str(checkpoint.resolve()),
		"checkpoint_mtime": checkpoint.stat().st_mtime,
		"embedding_dim": embedding_dim,
		"files": [
			{
				"path": str(path.relative_to(image_dir).as_posix()),
				"mtime": path.stat().st_mtime,
				"size": path.stat().st_size,
			}
			for path in image_paths
		],
	}
	return hashlib.md5(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()


def cache_prefix(cache_dir: Path, key: str) -> Path:
	cache_dir.mkdir(parents=True, exist_ok=True)
	return cache_dir / f"gallery_{key}"


@torch.no_grad()
def encode_gallery(
	image_encoder: ImageEncoder,
	image_paths: Sequence[Path],
	batch_size: int,
	device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
	dataset = FolderImageDataset(image_paths)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	total_images = len(dataset)
	processed = 0

	image_encoder.eval()
	embeddings: List[torch.Tensor] = []
	paths: List[str] = []

	for batch_images, batch_paths in loader:
		batch_images = batch_images.to(device)
		batch_embed = F.normalize(image_encoder(batch_images), dim=1)
		embeddings.append(batch_embed.cpu())
		paths.extend(batch_paths)
		processed += len(batch_paths)
		progress = processed / total_images
		bar_width = 30
		filled = int(bar_width * progress)
		bar = "=" * filled + "." * (bar_width - filled)
		print(f"\rIndexing images: [{bar}] {processed}/{total_images}", end="", flush=True)

	print()

	return torch.cat(embeddings, dim=0), paths


def load_or_build_gallery_cache(
	image_encoder: ImageEncoder,
	image_dir: Path,
	checkpoint: Path,
	embedding_dim: int,
	batch_size: int,
	cache_dir: Path,
	force_rebuild: bool,
	device: torch.device,
) -> Tuple[faiss.Index, List[str], Path]:
	image_paths = discover_images(image_dir, VALID_EXTENSIONS)
	if not image_paths:
		raise FileNotFoundError(f"No images found in folder: {image_dir}")

	key = build_cache_key(image_dir, checkpoint, embedding_dim)
	prefix = cache_prefix(cache_dir, key)
	index_path = prefix.with_suffix(".index")
	meta_path = prefix.with_suffix(".json")

	if index_path.exists() and meta_path.exists() and not force_rebuild:
		index = faiss.read_index(str(index_path))
		with open(meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
		return index, meta["paths"], prefix

	embeddings, paths = encode_gallery(
		image_encoder=image_encoder,
		image_paths=image_paths,
		batch_size=batch_size,
		device=device,
	)

	index = faiss.IndexFlatIP(embeddings.size(1))
	gallery_array = _to_numpy_float32(embeddings)
	faiss.normalize_L2(gallery_array)
	index.add(gallery_array)
	faiss.write_index(index, str(index_path))
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump({"paths": paths}, f, ensure_ascii=False, indent=2)
	return index, paths, prefix


@torch.no_grad()
def embed_text_query(
	query: str,
	text_encoder: TextEncoder,
	tokenizer: DistilBertTokenizer,
	max_length: int,
	device: torch.device,
) -> torch.Tensor:
	tok = tokenizer(
		query,
		padding="max_length",
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)
	input_ids = tok["input_ids"].to(device)
	attention_mask = tok["attention_mask"].to(device)
	vec = text_encoder(input_ids, attention_mask)
	return F.normalize(vec, dim=1).cpu().squeeze(0)


@torch.no_grad()
def embed_image_query(query_image: Path, image_encoder: ImageEncoder, device: torch.device) -> torch.Tensor:
	image = Image.open(query_image).convert("RGB")
	x = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
	vec = image_encoder(x)
	return F.normalize(vec, dim=1).cpu().squeeze(0)


def topk_matches(query_embed: torch.Tensor, gallery_index: faiss.Index, k: int) -> Tuple[List[float], List[int]]:
	query = _to_numpy_float32(query_embed).reshape(1, -1)
	faiss.normalize_L2(query)
	dists, indices = gallery_index.search(query, min(k, gallery_index.ntotal))
	return dists[0].tolist(), indices[0].tolist()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Fast retrieval: query by text or image against a folder of images"
	)
	parser.add_argument("--config", default="config.yaml", help="Path to config file")
	parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
	parser.add_argument("--image_dir", default="test_model", help="Folder with searchable images")
	parser.add_argument("--cache_dir", default=".cache/retrieval", help="Where to store gallery embeddings")
	parser.add_argument("--batch_size", type=int, default=64, help="Gallery encoding batch size")
	parser.add_argument("--top_k", type=int, default=5, help="Number of nearest images to return")
	parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuilding gallery cache")

	parser.add_argument("--query_text", type=str, default=None, help="Text query")
	parser.add_argument("--query_image", type=str, default=None, help="Image query path")

	args = parser.parse_args()

	if (args.query_text is None) == (args.query_image is None):
		raise ValueError("Provide exactly one of --query_text or --query_image")

	cfg = load_config(args.config)
	checkpoint = Path(args.checkpoint)
	image_dir = Path(args.image_dir)
	cache_dir = Path(args.cache_dir)

	if not checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
	if not image_dir.exists() or not image_dir.is_dir():
		raise FileNotFoundError(f"Image folder not found: {image_dir}")

	requested_device = cfg.get("training", {}).get("device", "cpu")
	device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

	image_encoder = ImageEncoder(
		embedding_dim=cfg["model"]["embedding_dim"],
		backbone_name=cfg["model"]["image_backbone"],
		use_pretrained=cfg["model"].get("image_pretrained", False),
	).to(device)

	text_encoder = TextEncoder(
		embedding_dim=cfg["model"]["embedding_dim"],
		model_name=cfg["model"]["text_backbone"],
		use_pretrained=cfg["model"].get("text_pretrained", False),
	).to(device)

	payload = torch.load(checkpoint, map_location=device)
	image_encoder.load_state_dict(payload["image_encoder"])
	text_encoder.load_state_dict(payload["text_encoder"])
	image_encoder.eval()
	text_encoder.eval()

	gallery_index, gallery_paths, used_cache_path = load_or_build_gallery_cache(
		image_encoder=image_encoder,
		image_dir=image_dir,
		checkpoint=checkpoint,
		embedding_dim=cfg["model"]["embedding_dim"],
		batch_size=args.batch_size,
		cache_dir=cache_dir,
		force_rebuild=args.rebuild_cache,
		device=device,
	)

	if args.query_text is not None:
		tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["text_backbone"])
		query_embed = embed_text_query(
			query=args.query_text,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			max_length=cfg["data"].get("max_length", 77),
			device=device,
		)
		query_label = f"text: {args.query_text}"
	else:
		query_image = Path(args.query_image)
		if not query_image.exists() or not query_image.is_file():
			raise FileNotFoundError(f"Query image not found: {query_image}")
		query_embed = embed_image_query(query_image=query_image, image_encoder=image_encoder, device=device)
		query_label = f"image: {query_image}"

	scores, indices = topk_matches(query_embed, gallery_index, args.top_k)

	print(f"Device: {device}")
	print(f"Gallery images: {len(gallery_paths)}")
	print(f"Gallery cache: {used_cache_path.with_suffix('.index')}")
	print(f"Query -> {query_label}")
	print("Top matches:")
	for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
		print(f"{rank:>2}. score={score:.4f} | {gallery_paths[idx]}")


if __name__ == "__main__":
	main()
