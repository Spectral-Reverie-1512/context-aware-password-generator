from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def label_smoothed_nll_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    ignore_index: int,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = targets.clone()
    targets_mask = targets != ignore_index
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - epsilon) * nll + epsilon * smooth
    loss = loss * targets_mask
    return loss.sum() / targets_mask.sum().clamp_min(1)


def contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class WarmupLinearDecay(nn.Module):
    def __init__(self, d_model: int, warmup_steps: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def forward(self, step: int) -> float:
        step = max(1, step)
        return (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )

