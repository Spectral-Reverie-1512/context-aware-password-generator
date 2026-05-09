from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(temperature, 1e-6)
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, top_k)
    min_val = values[-1]
    return torch.where(logits < min_val, torch.full_like(logits, -1e9), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0 or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    filtered = sorted_logits.masked_fill(mask, -1e9)
    restored = torch.full_like(logits, -1e9)
    restored.scatter_(0, sorted_indices, filtered)
    return restored


def sample_from_logits(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> int:
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))
