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


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.VAE_DROPOUT),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.VAE_DROPOUT),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_net(z_enc)
        logvar = self.logvar_net(z_enc)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


class ContextModulation(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DEC_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DEC_DROPOUT),
        )
        self.to_gamma = nn.Linear(hidden_dim, latent_dim)
        self.to_beta = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.net(context)
        gamma = self.to_gamma(h)
        beta = self.to_beta(h)
        return gamma * z + beta


class AutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        max_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, token_dim, padding_idx=pad_id)
        self.proj = nn.Linear(token_dim + config.LATENT_DIM, model_dim)
        self.positional = SinusoidalPositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, z_cond: torch.Tensor
    ) -> torch.Tensor:
        token_emb = self.token_embed(input_ids)
        z_expand = z_cond.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        x = torch.cat([token_emb, z_expand], dim=-1)
        x = self.proj(x)
        x = self.positional(x)
        x = self.dropout(x)

        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        key_padding_mask = attention_mask == 0

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        return logits


@dataclass
class Stage3Output:
    logits: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    z_cond: torch.Tensor


class Stage3Model(nn.Module):
    def __init__(
        self,
        stage2_encoder: nn.Module,
        vocab_size: int,
        pad_id: int,
        max_len: int = config.MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.stage2_encoder = stage2_encoder
        self.vae = VAEEncoder(config.ENC_DIM, config.VAE_HIDDEN, config.LATENT_DIM)
        self.context_mod = ContextModulation(config.LATENT_DIM, config.CONTEXT_HIDDEN)
        self.decoder = AutoregressiveDecoder(
            vocab_size=vocab_size,
            token_dim=config.DEC_TOKEN_DIM,
            model_dim=config.DEC_MODEL_DIM,
            num_heads=config.DEC_HEADS,
            num_layers=config.DEC_LAYERS,
            ffn_dim=config.DEC_FFN,
            dropout=config.DEC_DROPOUT,
            max_len=max_len,
            pad_id=pad_id,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_vec: torch.Tensor,
    ) -> Stage3Output:
        z_enc = self.stage2_encoder.encode(input_ids, attention_mask)
        mu, logvar = self.vae(z_enc)
        z = self.vae.reparameterize(mu, logvar)
        z_cond = self.context_mod(z, context_vec)
        logits = self.decoder(input_ids, attention_mask, z_cond)
        return Stage3Output(logits=logits, mu=mu, logvar=logvar, z=z, z_cond=z_cond)
