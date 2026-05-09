from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from . import config


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


@dataclass
class ModelOutput:
    logits_next: torch.Tensor
    logits_recon: torch.Tensor
    pooled: torch.Tensor


class PasswordEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = config.D_MODEL,
        ffn_hidden: int = config.FFN_HIDDEN,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        max_len: int = config.MAX_SEQ_LEN,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.next_token_head = nn.Linear(d_model, vocab_size)
        self.recon_head = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._masked_mean(x, attention_mask)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._masked_mean(x, attention_mask)
        logits_next = self.next_token_head(x)
        logits_recon = self.recon_head(x)
        return ModelOutput(logits_next=logits_next, logits_recon=logits_recon, pooled=pooled)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

