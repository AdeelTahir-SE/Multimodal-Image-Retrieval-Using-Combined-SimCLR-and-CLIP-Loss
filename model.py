"""
Backward-compatible entry module.

Use the concrete implementations from models/, datasets/, and losses/.
"""

from datasets.coco_dataset import COCOPairDataset
from losses.clip_loss import clip_loss
from losses.simclr_loss import simclr_loss
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "COCOPairDataset",
    "simclr_loss",
    "clip_loss",
]
