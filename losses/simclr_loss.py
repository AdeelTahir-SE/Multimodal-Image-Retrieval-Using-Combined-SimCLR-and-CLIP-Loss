import torch
import torch.nn.functional as F


def simclr_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss for SimCLR.

    z_i, z_j: tensors of shape (N, D) from two augmented views.
    """
    n = z_i.size(0)
    z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
    sim = z @ z.t() / temperature

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.cat(
        [
            torch.arange(n, 2 * n, device=z.device),
            torch.arange(0, n, device=z.device),
        ],
        dim=0,
    )
    return F.cross_entropy(sim, targets)