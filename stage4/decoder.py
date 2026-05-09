from __future__ import annotations

import random
import re


ALPHA_RE = re.compile(r"^[A-Za-z]+$")


def random_case(word: str) -> str:
    out = []
    for ch in word:
        if ch.isalpha():
            out.append(ch.upper() if random.random() < 0.5 else ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _extract_digits(context_struct: dict) -> list[str]:
    numbers = context_struct.get("numbers") or []
    digits = []
    for n in numbers:
        digits.extend(list(str(n)))
    return [d for d in digits if d.isdigit()]


def _extract_symbols(context_struct: dict) -> list[str]:
    symbols = context_struct.get("symbols") or []
    chars = []
    for s in symbols:
        chars.extend(list(str(s)))
    return [c for c in chars if not c.isalnum()]


def decode_tokens(
    tokens: list[str],
    context_struct: dict | None = None,
    symbol_pool: str = "!@#$%^&*()-_=+[]{};:,.?/|",
) -> str:
    context_struct = context_struct or {}
    out = []
    digit_hint = _extract_digits(context_struct)
    symbol_hint = _extract_symbols(context_struct)

    for tok in tokens:
        if tok.startswith("<S_"):
            continue
        if tok in {"<BOS>", "<EOS>", "<PAD>"}:
            continue
        if tok.startswith("<D") and tok.endswith(">"):
            length = int(tok[2:-1] or "1")
            digits = []
            for _ in range(length):
                if digit_hint:
                    digits.append(random.choice(digit_hint))
                else:
                    digits.append(str(random.randint(0, 9)))
            out.append("".join(digits))
            continue
        if tok.startswith("<S") and tok.endswith(">") and tok[2:-1].isdigit():
            length = int(tok[2:-1] or "1")
            symbols = []
            for _ in range(length):
                if symbol_hint:
                    symbols.append(random.choice(symbol_hint))
                else:
                    symbols.append(random.choice(symbol_pool))
            out.append("".join(symbols))
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        if ALPHA_RE.match(tok):
            out.append(random_case(tok))
        else:
            out.append(tok)

    return "".join(out)
