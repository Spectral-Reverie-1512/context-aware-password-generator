from __future__ import annotations

from pathlib import Path

import torch

from . import config
from .extract import extract_context
from .fusion import ContextFusionNetwork
from .semantic import SemanticEncoder
from .structured import StructuredVectorizer


class ContextProcessor:
    def __init__(
        self,
        semantic_encoder: SemanticEncoder | None = None,
        structured_vectorizer: StructuredVectorizer | None = None,
        fusion_model: ContextFusionNetwork | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device
        self.semantic_encoder = semantic_encoder or SemanticEncoder(device=device)
        self.structured_vectorizer = structured_vectorizer or StructuredVectorizer()
        self.fusion_model = fusion_model or ContextFusionNetwork()
        if self.device:
            self.fusion_model.to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(config.CONTEXT_DIM)
        semantic_vec = self.semantic_encoder.encode(text)
        structured = extract_context(text)
        structured_vec = self.structured_vectorizer.encode(structured)

        if self.device:
            semantic_vec = semantic_vec.to(self.device)
            structured_vec = structured_vec.to(self.device)
        with torch.no_grad():
            context_vec = self.fusion_model(semantic_vec, structured_vec)
        return context_vec

    def encode_file(self, path: str | Path) -> torch.Tensor:
        path = Path(path)
        if not path.exists():
            return torch.zeros(config.CONTEXT_DIM)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.encode_text(text)

