import torch.nn as nn
import torchvision.models as tv_models

from models.projection_head import ProjectionHead


class ImageEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        backbone_name: str = "resnet50",
        use_pretrained: bool = False,
    ):
        super().__init__()

        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "vit_b_16":
            weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
            backbone = tv_models.vit_b_16(weights=weights)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.encoder = backbone
        else:
            raise ValueError(f"Unsupported image backbone: {backbone_name}")

        self.projector = ProjectionHead(feature_dim, embedding_dim)

    def forward(self, x):
        feat = self.encoder(x)
        if feat.ndim == 4:
            feat = feat.squeeze(-1).squeeze(-1)
        return self.projector(feat)