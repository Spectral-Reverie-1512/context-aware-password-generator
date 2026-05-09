from __future__ import annotations

from collections import Counter

from .segmenter import LEET_MAP, TAG_LEET, TAG_LETTERS, TAG_WORD


class VocabBuilder:
    def __init__(self, max_len: int = 12) -> None:
        self.max_len = max_len
        self.counts: Counter[str] = Counter()

    def _normalize_leet(self, token: str) -> str | None:
        chars: list[str] = []
        for ch in token:
            if ch.isalpha():
                chars.append(ch.lower())
            elif ch in LEET_MAP:
                chars.append(LEET_MAP[ch])
            else:
                return None
        return "".join(chars)

    def _accept(self, token: str) -> bool:
        if not token:
            return False
        if len(token) > self.max_len:
            return False
        return True

    def add_segments(self, segments: list[tuple[str, str]]) -> None:
        for segment, tag in segments:
            if tag == TAG_WORD:
                token = segment.lower()
            elif tag == TAG_LETTERS:
                token = segment.lower()
            elif tag == TAG_LEET:
                token = self._normalize_leet(segment)
                if token is None:
                    continue
            else:
                continue

            if self._accept(token):
                self.counts[token] += 1

