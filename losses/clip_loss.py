import torch
import torch.nn.functional as F


def clip_loss(z_img: torch.Tensor, z_txt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric CLIP-style contrastive loss over image-text similarities.

    z_img, z_txt: tensors of shape (N, D).
    """
    z_img = F.normalize(z_img, dim=1)
    z_txt = F.normalize(z_txt, dim=1)

    logits = z_img @ z_txt.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)