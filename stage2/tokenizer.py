from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import config

LEET_MAP = {
    "0": "o",
    "1": "i",
    "!": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "+": "t",
}


def _load_vocab_list(vocab_path: Path) -> list[str]:
    vocab: list[str] = []
    with vocab_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split("\t", 1)[0].strip().lower()
            if token:
                vocab.append(token)
    return vocab


def _load_segment_fields(line: str) -> list[tuple[str, str]]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 4:
        return []
    pairs = fields[2:]
    segments: list[tuple[str, str]] = []
    for i in range(0, len(pairs) - 1, 2):
        segment = pairs[i]
        tag = pairs[i + 1]
        segments.append((segment, tag))
    return segments


@dataclass
class TokenizedSample:
    input_ids: list[int]
    attention_mask: list[int]


class PasswordTokenizer:
    def __init__(
        self,
        vocab_path: str | Path,
        max_seq_len: int = config.MAX_SEQ_LEN,
        symbol_groups: dict[str, str] | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.symbol_groups = symbol_groups or config.SYMBOL_GROUPS
        self.symbol_lookup = self._build_symbol_lookup(self.symbol_groups)

        vocab_list = _load_vocab_list(Path(vocab_path))
        self.alpha_vocab = set(vocab_list)

        tokens = (
            config.SPECIAL_TOKENS
            + config.STRUCT_TOKENS
            + config.DIGIT_TOKENS
            + [f"<{name}>" for name in self.symbol_groups]
            + vocab_list
        )
        self.token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]

    @staticmethod
    def _build_symbol_lookup(symbol_groups: dict[str, str]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for name, chars in symbol_groups.items():
            token = f"<{name}>"
            for ch in chars:
                lookup[ch] = token
        return lookup

    def save_vocab(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
                    "symbol_groups": self.symbol_groups,
                    "max_seq_len": self.max_seq_len,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def tokens_from_segments(self, segments: Iterable[tuple[str, str]]) -> list[str]:
        tokens: list[str] = []
        for segment, tag in segments:
            struct = f"<S_{tag}>"
            if struct in self.token_to_id:
                tokens.append(struct)
            else:
                tokens.append("<S_symbols>")

            if tag in {"word", "letters", "keyboard"}:
                token = segment.lower()
                tokens.append(token if token in self.alpha_vocab else "<UNK>")
            elif tag == "leet":
                token = self._normalize_leet(segment)
                if token and token in self.alpha_vocab:
                    tokens.append(token)
                else:
                    tokens.append("<UNK>")
            elif tag in {"digits", "year"}:
                for ch in segment:
                    if ch.isdigit():
                        tokens.append(ch)
            elif tag == "symbols":
                for ch in segment:
                    tokens.append(self.symbol_lookup.get(ch, "<SYM_MISC>"))
            else:
                tokens.append("<UNK>")
        return tokens

    @staticmethod
    def _normalize_leet(segment: str) -> str | None:
        chars: list[str] = []
        for ch in segment:
            if ch.isalpha():
                chars.append(ch.lower())
            elif ch in LEET_MAP:
                chars.append(LEET_MAP[ch])
            else:
                return None
        return "".join(chars)

    def encode_tokens(self, tokens: list[str]) -> TokenizedSample:
        tokens = ["<BOS>"] + tokens + ["<EOS>"]
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            ids[-1] = self.eos_id
        else:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))

        attention_mask = [0 if idx == self.pad_id else 1 for idx in ids]
        return TokenizedSample(ids, attention_mask)

    def encode_segments(self, segments: Iterable[tuple[str, str]]) -> TokenizedSample:
        tokens = self.tokens_from_segments(segments)
        return self.encode_tokens(tokens)


class SegmentationDataset:
    def __init__(self, segmentations_path: str | Path, tokenizer: PasswordTokenizer) -> None:
        self.segmentations_path = Path(segmentations_path)
        self.tokenizer = tokenizer
        self.offsets = self._build_offsets()

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.segmentations_path.open("rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> TokenizedSample:
        offset = self.offsets[idx]
        with self.segmentations_path.open("rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8", errors="ignore")

        segments = _load_segment_fields(line)
        return self.tokenizer.encode_segments(segments)

