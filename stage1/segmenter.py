from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KEYBOARD_SEQUENCES = [
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "1qaz",
    "2wsx",
    "3edc",
    "4rfv",
    "5tgb",
    "6yhn",
    "7ujm",
    "8ik",
    "9ol",
    "0p",
]

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

TAG_KEYBOARD = "keyboard"
TAG_LEET = "leet"
TAG_YEAR = "year"
TAG_WORD = "word"
TAG_LETTERS = "letters"
TAG_DIGITS = "digits"
TAG_SYMBOLS = "symbols"


def _load_wordlist(wordlist_path: str | None) -> dict[str, set[str]]:
    if wordlist_path:
        path = Path(wordlist_path)
    else:
        path = Path(__file__).with_name("wordlist.txt")

    word_index: dict[str, set[str]] = {}
    if not path.exists():
        return word_index

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if not word or not word.isalpha():
                continue
            word_index.setdefault(word[0], set()).add(word)
    return word_index


def _keyboard_walk_match(text: str, start: int) -> int:
    if start >= len(text):
        return 0
    lowered = text.lower()
    best = 0
    for seq in KEYBOARD_SEQUENCES:
        for variant in (seq, seq[::-1]):
            pos = variant.find(lowered[start])
            if pos == -1:
                continue
            idx = start
            seq_idx = pos
            while idx < len(lowered) and seq_idx < len(variant):
                if lowered[idx] != variant[seq_idx]:
                    break
                idx += 1
                seq_idx += 1
            best = max(best, idx - start)
    return best if best >= 3 else 0


def _leet_match(text: str, start: int) -> int:
    idx = start
    has_leet = False
    has_alpha = False
    while idx < len(text):
        ch = text[idx]
        if ch.isalpha():
            has_alpha = True
        elif ch in LEET_MAP:
            has_leet = True
        else:
            break
        idx += 1
    if idx - start >= 2 and has_leet and has_alpha:
        return idx - start
    return 0


def _year_match(text: str, start: int) -> int:
    if start + 4 > len(text):
        return 0
    chunk = text[start : start + 4]
    if not chunk.isdigit():
        return 0
    year = int(chunk)
    if 1900 <= year <= 2000:
        return 4
    return 0


def _word_match(
    text: str, start: int, word_index: dict[str, set[str]]
) -> int:
    if not word_index or start >= len(text):
        return 0
    if not text[start].isalpha():
        return 0
    lowered = text.lower()
    candidates = word_index.get(lowered[start], set())
    if not candidates:
        return 0

    best = 0
    max_len = max(len(w) for w in candidates)
    end_limit = min(len(text), start + max_len)
    for end in range(start + 1, end_limit + 1):
        chunk = lowered[start:end]
        if chunk in candidates and chunk.isalpha():
            best = max(best, end - start)
    return best


def _letters_match(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isalpha():
        idx += 1
    return idx - start if idx > start else 0


def _digits_match(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isdigit():
        idx += 1
    return idx - start if idx > start else 0


def _symbols_match(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and not text[idx].isalnum():
        idx += 1
    return idx - start if idx > start else 0


@dataclass
class Segmenter:
    wordlist_path: str | None = None

    def __post_init__(self) -> None:
        self.word_index = _load_wordlist(self.wordlist_path)

    def segment(self, password: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        idx = 0
        while idx < len(password):
            for length, tag in self._matchers(password, idx):
                if length:
                    segment = password[idx : idx + length]
                    segments.append((segment, tag))
                    idx += length
                    break
            else:
                segments.append((password[idx], TAG_SYMBOLS))
                idx += 1
        return segments

    def _matchers(self, password: str, idx: int) -> Iterable[tuple[int, str]]:
        yield _keyboard_walk_match(password, idx), TAG_KEYBOARD
        yield _leet_match(password, idx), TAG_LEET
        yield _year_match(password, idx), TAG_YEAR
        yield _word_match(password, idx, self.word_index), TAG_WORD
        yield _letters_match(password, idx), TAG_LETTERS
        yield _digits_match(password, idx), TAG_DIGITS
        yield _symbols_match(password, idx), TAG_SYMBOLS

