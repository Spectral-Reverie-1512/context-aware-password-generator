from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from stage1.segmenter import Segmenter
from stage25.fusion import ContextFusionNetwork
from stage25.semantic import SemanticEncoder
from stage25.structured import StructuredVectorizer

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


def _load_json_array(path: Path) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buf = ""
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == "[":
                    buf = buf[1:]
                    continue
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                yield obj
                buf = buf[idx:]
        buf = buf.strip()
        if buf and buf[0] == "]":
            return


@dataclass
class TokenizedSample:
    input_ids: list[int]
    attention_mask: list[int]
    context_vec: torch.Tensor
    structured_context: dict


class Stage3Tokenizer:
    def __init__(self, tokenizer_json: str | Path, wordlist_path: str | None = None) -> None:
        tokenizer_json = Path(tokenizer_json)
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        self.tokens = payload["tokens"]
        self.symbol_groups = payload.get("symbol_groups") or {}
        self.max_seq_len = int(payload.get("max_seq_len", config.MAX_SEQ_LEN))
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.unk_id = self.token_to_id.get("<UNK>", 1)
        self.bos_id = self.token_to_id.get("<BOS>", 2)
        self.eos_id = self.token_to_id.get("<EOS>", 3)

        self.symbol_lookup = self._build_symbol_lookup(self.symbol_groups)
        self.alpha_vocab = self._build_alpha_vocab()
        self.segmenter = Segmenter(wordlist_path=wordlist_path)

    def _build_alpha_vocab(self) -> set[str]:
        alpha = set()
        for tok in self.tokens:
            if tok.startswith("<") and tok.endswith(">"):
                continue
            if tok.isdigit():
                continue
            alpha.add(tok)
        return alpha

    @staticmethod
    def _build_symbol_lookup(symbol_groups: dict[str, str]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for name, chars in symbol_groups.items():
            token = f"<{name}>"
            for ch in chars:
                lookup[ch] = token
        return lookup

    def _normalize_leet(self, segment: str) -> str | None:
        chars: list[str] = []
        for ch in segment:
            if ch.isalpha():
                chars.append(ch.lower())
            elif ch in LEET_MAP:
                chars.append(LEET_MAP[ch])
            else:
                return None
        return "".join(chars)

    def tokens_from_segments(self, segments: Iterable[tuple[str, str]]) -> list[str]:
        tokens: list[str] = []
        for segment, tag in segments:
            struct = f"<S_{tag}>"
            tokens.append(struct if struct in self.token_to_id else "<S_symbols>")

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

    def encode_tokens(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        tokens = ["<BOS>"] + tokens + ["<EOS>"]
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            ids[-1] = self.eos_id
        else:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))

        mask = [0 if idx == self.pad_id else 1 for idx in ids]
        return ids, mask

    def encode_password(self, password: str) -> tuple[list[int], list[int]]:
        segments = self.segmenter.segment(password)
        tokens = self.tokens_from_segments(segments)
        return self.encode_tokens(tokens)


class ContextVectorizer:
    def __init__(self, fusion_state: str | None = None, device: str | None = None) -> None:
        self.device = device
        self.semantic = SemanticEncoder(device=device)
        self.structured = StructuredVectorizer()
        self.fusion = ContextFusionNetwork()
        if fusion_state:
            state = torch.load(fusion_state, map_location="cpu")
            self.fusion.load_state_dict(state)
        if self.device:
            self.fusion.to(self.device)

    def encode(self, raw_context: str, structured_context: dict[str, list[str]]) -> torch.Tensor:
        if not raw_context and not structured_context:
            return torch.zeros(config.CONTEXT_DIM)
        semantic_vec = self.semantic.encode(raw_context)
        structured_vec = self.structured.encode(structured_context)
        if self.device:
            semantic_vec = semantic_vec.to(self.device)
            structured_vec = structured_vec.to(self.device)
        with torch.no_grad():
            return self.fusion(semantic_vec, structured_vec)


class Stage3Dataset(Dataset):
    def __init__(
        self,
        cpm_path: str | Path,
        targets_path: str | Path,
        tokenizer: Stage3Tokenizer,
        context_vectorizer: ContextVectorizer | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.context_vectorizer = context_vectorizer
        self.samples: list[dict] = []

        targets = {}
        for obj in _load_json_array(Path(targets_path)):
            targets[obj.get("sample_id")] = obj.get("password", "")

        for obj in _load_json_array(Path(cpm_path)):
            sample_id = obj.get("sample_id")
            if sample_id not in targets:
                continue
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "raw_context": obj.get("raw_context", ""),
                    "structured_context": obj.get("structured_context", {}),
                    "password": targets[sample_id],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TokenizedSample:
        sample = self.samples[idx]
        input_ids, attention_mask = self.tokenizer.encode_password(sample["password"])
        if self.context_vectorizer:
            context_vec = self.context_vectorizer.encode(
                sample["raw_context"], sample["structured_context"]
            )
        else:
            context_vec = torch.zeros(config.CONTEXT_DIM)
        return TokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_vec=context_vec,
            structured_context=sample["structured_context"],
        )


def collate_batch(
    samples: list[TokenizedSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    input_ids = torch.tensor([s.input_ids for s in samples], dtype=torch.long)
    attention_mask = torch.tensor([s.attention_mask for s in samples], dtype=torch.long)
    context_vecs = torch.stack([s.context_vec for s in samples])
    structured = [s.structured_context for s in samples]
    return input_ids, attention_mask, context_vecs, structured
