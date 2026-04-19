import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel

from models.projection_head import ProjectionHead


class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        model_name: str = "distilbert-base-uncased",
        use_pretrained: bool = False,
    ):
        super().__init__()
        if use_pretrained:
            self.bert = DistilBertModel.from_pretrained(model_name)
        else:
            config = DistilBertConfig()
            self.bert = DistilBertModel(config)

        hidden_size = self.bert.config.dim
        self.projector = ProjectionHead(hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.projector(cls)