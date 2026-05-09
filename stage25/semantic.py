from __future__ import annotations

import torch

from . import config


class SemanticEncoder:
    def __init__(self, model_name: str = config.SEMANTIC_MODEL_NAME, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for Stage 2.5 semantic encoding. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(config.SEMANTIC_DIM)
        embedding = self.model.encode([text], convert_to_tensor=True, normalize_embeddings=False)
        return embedding.squeeze(0)

