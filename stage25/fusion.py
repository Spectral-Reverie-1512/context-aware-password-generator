from __future__ import annotations

import torch
from torch import nn

from . import config


class ContextFusionNetwork(nn.Module):
    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        structured_dim: int = config.STRUCTURED_DIM,
        hidden_dim: int = config.MLP_HIDDEN,
        output_dim: int = config.CONTEXT_DIM,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + structured_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, semantic_vec: torch.Tensor, structured_vec: torch.Tensor) -> torch.Tensor:
        if semantic_vec.dim() == 1:
            semantic_vec = semantic_vec.unsqueeze(0)
        if structured_vec.dim() == 1:
            structured_vec = structured_vec.unsqueeze(0)
        combined = torch.cat([semantic_vec, structured_vec], dim=-1)
        return self.net(combined).squeeze(0)

