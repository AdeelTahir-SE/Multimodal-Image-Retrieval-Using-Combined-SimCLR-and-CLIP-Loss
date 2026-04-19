import json
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer


TRAIN_AUGMENT = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class COCOPairDataset(Dataset):
    def __init__(
        self,
        pairs_json: str,
        max_length: int = 77,
        tokenizer_name: str = "distilbert-base-uncased",
        train: bool = True,
    ):
        with open(pairs_json, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _tokenize(self, caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.pairs[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        if self.train:
            image_aug1 = TRAIN_AUGMENT(image)
            image_aug2 = TRAIN_AUGMENT(image)
        else:
            base = EVAL_TRANSFORM(image)
            image_aug1 = base
            image_aug2 = base

        input_ids, attention_mask = self._tokenize(item["caption"])

        return {
            "image_aug1": image_aug1,
            "image_aug2": image_aug2,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }