from __future__ import annotations

import hashlib
import math

import torch

from . import config


def _stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16)


class StructuredVectorizer:
    def __init__(self, buckets: dict[str, int] | None = None) -> None:
        self.buckets = buckets or config.STRUCTURED_BUCKETS
        self.offsets = {}
        offset = 0
        for key, size in self.buckets.items():
            self.offsets[key] = (offset, size)
            offset += size
        self.dim = offset

    def _hash_bucket(self, key: str, token: str) -> int:
        offset, size = self.offsets[key]
        return offset + (_stable_hash(token) % size)

    def encode(self, features: dict[str, list[str]]) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)

        for key in ("names", "keywords", "dates"):
            items = features.get(key, [])
            for item in items:
                idx = self._hash_bucket(key, item)
                vec[idx] = min(1.0, vec[idx].item() + 0.2)

        for key in ("emails", "usernames"):
            items = features.get(key, [])
            for item in items:
                idx = self._hash_bucket(key, item)
                vec[idx] = 1.0

        numbers = features.get("numbers", [])
        if numbers:
            numeric_vals = [int(n) for n in numbers if n.isdigit()]
        else:
            numeric_vals = []

        offset, size = self.offsets["numbers"]
        stats_len = min(4, size)
        if numeric_vals:
            count = len(numeric_vals)
            mean = sum(numeric_vals) / count
            max_val = max(numeric_vals)
            avg_len = sum(len(str(n)) for n in numeric_vals) / count
            stats = [
                min(count / 10.0, 1.0),
                min(math.log1p(mean) / 10.0, 1.0),
                min(math.log1p(max_val) / 10.0, 1.0),
                min(avg_len / 10.0, 1.0),
            ]
            for i, value in enumerate(stats[:stats_len]):
                vec[offset + i] = value

        for token in numbers:
            idx = offset + stats_len + (_stable_hash(token) % max(1, size - stats_len))
            vec[idx] = min(1.0, vec[idx].item() + 0.25)

        return vec

