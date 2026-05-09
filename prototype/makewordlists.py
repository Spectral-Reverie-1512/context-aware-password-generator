"""
Build targeted wordlists using the same password generation logic as api_server_light.py,
but running entirely locally — no API server needed.

Usage:
  python make_wordlists.py
  python make_wordlists.py --context "College student loves anime" --num 2000
  python make_wordlists.py --scenarios-file scenarios.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Password generation logic (mirrored from api_server_light.py)
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = "!@#$%&_-"

DEFAULT_MIN_LENGTH = 5
DEFAULT_MAX_LENGTH = 32
DEFAULT_REQUIRE_SYMBOL = True
DEFAULT_REQUIRE_DIGIT = True


def generate_passwords(
    context: str = "",
    num_passwords: int = 10,
    seed: Optional[int] = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    require_upper: bool = True,
    require_lower: bool = True,
    require_digit: bool = True,
    require_symbol: bool = True,
    symbols: str = DEFAULT_SYMBOLS,
) -> List[str]:
    """Generate passwords using the same rules-based logic as api_server_light.py."""

    context_text = (context or "").strip()
    rng = random.Random(seed) if seed is not None else random.Random()
    symbols_pool = list(symbols) if symbols else list(DEFAULT_SYMBOLS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def enforce_strength(pwd: str) -> str | None:
        s = (pwd or "").strip()
        if not s or any(ch.isspace() for ch in s):
            return None
        if len(s) > max_length:
            s = s[:max_length]

        def has_upper(x: str) -> bool:
            return any(ch.isupper() for ch in x)

        def has_lower(x: str) -> bool:
            return any(ch.islower() for ch in x)

        def has_digit(x: str) -> bool:
            return any(ch.isdigit() for ch in x)

        def has_symbol(x: str) -> bool:
            return any(ch in symbols_pool for ch in x)

        chars = list(s)
        letter_idxs = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if require_upper and not has_upper(s) and letter_idxs:
            i = rng.choice(letter_idxs)
            chars[i] = chars[i].upper()
        if require_lower and not has_lower("".join(chars)) and letter_idxs:
            i = rng.choice(letter_idxs)
            chars[i] = chars[i].lower()

        out = "".join(chars)
        if require_digit and not has_digit(out):
            out += rng.choice("0123456789")
        if require_symbol and not has_symbol(out):
            out += rng.choice(symbols_pool)

        while len(out) < min_length:
            out += rng.choice("0123456789") if rng.random() < 0.55 else rng.choice(symbols_pool)

        if len(out) > max_length:
            out = out[:max_length]

        if require_upper and not has_upper(out):
            return None
        if require_lower and not has_lower(out):
            return None
        if require_digit and not has_digit(out):
            return None
        if require_symbol and not has_symbol(out):
            return None
        return out

    def extract_tokens(ctx: str) -> tuple[list[str], list[str]]:
        words = re.findall(r"[A-Za-z]+", ctx.lower())
        digits = re.findall(r"\d+", ctx)
        stop = {
            "and", "the", "uses", "use", "user", "named", "pet", "loves", "with",
            "likes", "works", "work", "at", "in", "on", "for", "from", "to", "of",
            "a", "an", "is", "are", "this", "that",
        }
        words = [w for w in words if w not in stop and len(w) >= 2]
        digits = [d for d in digits if 1 <= len(d) <= 6]
        return words, digits

    def random_case(word: str) -> str:
        if not word:
            return word
        mode = rng.choice(["title", "lower", "upper", "camel", "alt"])
        if mode == "title":
            return word[:1].upper() + word[1:].lower()
        if mode == "lower":
            return word.lower()
        if mode == "upper":
            return word.upper()
        if mode == "camel":
            return "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(word))
        return "".join(ch.upper() if rng.random() < 0.45 else ch.lower() for ch in word)

    def maybe_leet(word: str) -> str:
        mapping = {"a": "@", "i": "1", "e": "3", "o": "0", "s": "$", "t": "7"}
        out = []
        for ch in word:
            if ch.lower() in mapping and rng.random() < 0.18:
                out.append(mapping[ch.lower()])
            else:
                out.append(ch)
        return "".join(out)

    def build_word_part(words: list[str]) -> str:
        if not words:
            words = ["user", "context"]
        k = 2 if rng.random() < 0.62 else 1
        k = 3 if rng.random() < 0.18 else k
        k = min(k, max(1, len(words)))
        chosen = rng.sample(words, k=k) if len(words) >= k else [rng.choice(words)]
        sep = rng.choice(["", "", "", ".", "_", "-", ""])
        return sep.join(maybe_leet(random_case(w)) for w in chosen)

    def build_number_part(digits: list[str]) -> str:
        yearish = [d for d in digits if len(d) == 4 and d.startswith(("19", "20"))]
        pool = (yearish or digits) or ["2026", "123", "007", "99"]
        a = rng.choice(pool)
        b = rng.choice(pool) if rng.random() < 0.35 else ""
        extra = "".join(rng.choice("0123456789") for _ in range(rng.choice([0, 0, 1, 2, 3])))
        return f"{a}{b}{extra}"

    def build_symbol_part() -> str:
        count = rng.choice([1, 1, 2, 2, 3])
        return "".join(rng.choice(symbols_pool) for _ in range(count))

    # ------------------------------------------------------------------
    # WITH context
    # ------------------------------------------------------------------
    if context_text:
        words, digits = extract_tokens(context_text)

        def simple_word(words_list: list[str]) -> str:
            if not words_list:
                return "User"
            w = rng.choice(words_list)
            return w[:1].upper() + w[1:].lower()

        def simple_digits(digits_list: list[str]) -> str:
            yearish = [d for d in digits_list if len(d) == 4 and d.startswith(("19", "20"))]
            pool = (yearish or digits_list) or ["2026", "2025", "2000", "123"]
            return rng.choice(pool)

        def simple_symbol() -> str:
            common = [c for c in ["@", "!", "#", "$", "_"] if c in symbols_pool]
            pool = common or symbols_pool
            return rng.choice(pool)

        passwords: list[str] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = max(1000, num_passwords * 25)

        simple_target = min(int(num_passwords * 0.3), 1200)
        simple_attempts = 0
        while len(passwords) < simple_target and simple_attempts < max_attempts:
            simple_attempts += 1
            w1 = simple_word(words)
            w2 = simple_word([w for w in words if w != w1.lower()]) if rng.random() < 0.35 else ""
            num = simple_digits(digits)
            sym = simple_symbol()

            base = rng.choice(
                [
                    f"{w1}{sym}{num}",
                    f"{w1}{num}{sym}",
                    f"{w1}{w2}{sym}{num}" if w2 else f"{w1}{sym}{num}",
                    f"{w1}{w2}{num}{sym}" if w2 else f"{w1}{num}{sym}",
                    f"{w1}{sym}{w2}{num}" if w2 else f"{w1}{sym}{num}",
                    f"{w1}{num}",
                ]
            )
            cand = enforce_strength(base)
            if not cand or cand in seen:
                continue
            seen.add(cand)
            passwords.append(cand)

        while len(passwords) < num_passwords and attempts < max_attempts:
            attempts += 1
            w = build_word_part(words)
            num_part = build_number_part(digits)
            sym = build_symbol_part()

            pattern = rng.choice(
                [
                    f"{w}{num_part}{sym}",
                    f"{w}{sym}{num_part}",
                    f"{sym}{w}{num_part}",
                    f"{num_part}{w}{sym}",
                    f"{w}{sym}{num_part}{w[:max(1, len(w)//3)]}",
                ]
            )

            cand = enforce_strength(pattern)
            if not cand or cand in seen:
                continue
            seen.add(cand)
            passwords.append(cand)

        return passwords

    # ------------------------------------------------------------------
    # WITHOUT context: simple stems (no password_targets.json dependency)
    # ------------------------------------------------------------------
    stems = ["password", "dragon", "qwerty", "admin", "letmein"]
    numbers = ["123", "2026", "321", "007", "99"]
    passwords = []
    for i in range(num_passwords):
        word = stems[i % len(stems)]
        num_part = numbers[i % len(numbers)]
        sym = symbols_pool[i % len(symbols_pool)]
        cand = enforce_strength(f"{word.capitalize()}{num_part}{sym}") or f"{word.capitalize()}{num_part}{sym}"
        passwords.append(cand)
    return passwords


# ---------------------------------------------------------------------------
# Wordlist builder
# ---------------------------------------------------------------------------

WORDLISTS_DIR = Path(__file__).resolve().parent / "wordlists"

SCENARIOS = [
    {
        "context": "College student, uses Instagram and Netflix, loves anime and K-pop, pet cat named Momo",
        "num_passwords": 2000,
        "outfile": "momo.txt",
    }
]


def _clean_password(pwd: object) -> str | None:
    if not isinstance(pwd, str):
        return None
    s = pwd.strip()
    if not s or any(ch.isspace() for ch in s):
        return None
    return s


def make_wordlist(
    context: str,
    num: int,
    outfile: str,
    *,
    seed: int | None = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    require_symbol: bool = DEFAULT_REQUIRE_SYMBOL,
    require_digit: bool = DEFAULT_REQUIRE_DIGIT,
) -> None:
    out_path = WORDLISTS_DIR / outfile
    WORDLISTS_DIR.mkdir(parents=True, exist_ok=True)

    passwords_set: set[str] = set()
    remaining = num
    empty_rounds = 0
    max_empty_rounds = 20
    MAX_PER_BATCH = 5000

    while remaining > 0:
        batch_size = min(MAX_PER_BATCH, remaining)
        batch = generate_passwords(
            context=context,
            num_passwords=batch_size,
            seed=seed,
            min_length=min_length,
            max_length=max_length,
            require_symbol=require_symbol,
            require_digit=require_digit,
        )

        added_this_round = 0
        for raw in batch:
            cleaned = _clean_password(raw)
            if cleaned is None or cleaned in passwords_set:
                continue
            passwords_set.add(cleaned)
            added_this_round += 1

        remaining -= added_this_round

        if added_this_round == 0:
            empty_rounds += 1
            if empty_rounds >= max_empty_rounds:
                break
        else:
            empty_rounds = 0

        # Bump seed so repeated batches produce different outputs
        if seed is not None:
            seed += 1

    with out_path.open("w", encoding="utf-8") as f:
        for pwd in sorted(passwords_set):
            f.write(pwd + "\n")

    print(f"Wrote {len(passwords_set)} passwords to {out_path}")


def _scenario_options(s: dict) -> dict:
    out: dict = {}
    if "seed" in s and s["seed"] is not None:
        out["seed"] = int(s["seed"])
    if "min_length" in s:
        out["min_length"] = int(s["min_length"])
    if "max_length" in s:
        out["max_length"] = int(s["max_length"])
    if "require_symbol" in s:
        out["require_symbol"] = bool(s["require_symbol"])
    if "require_digit" in s:
        out["require_digit"] = bool(s["require_digit"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wordlists locally using the same logic as api_server_light.py."
    )
    parser.add_argument("--context", default=None, help="Context text (if omitted, will prompt).")
    parser.add_argument("--num", type=int, default=2000, help="Number of passwords (default: %(default)s)")
    parser.add_argument("--outfile", default="wordlist.txt", help="Output filename under wordlists/ (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for repeatable output.")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="min_length (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="max_length (default: %(default)s)")
    parser.add_argument("--no-require-symbol", action="store_true", help="Disable require_symbol.")
    parser.add_argument("--no-require-digit", action="store_true", help="Disable require_digit.")
    parser.add_argument(
        "--scenarios-file",
        default=None,
        help="JSON list of scenario objects (context, num_passwords, outfile, and optional seed/lengths/flags).",
    )
    args = parser.parse_args()

    gen_kw: dict = {
        "seed": args.seed,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "require_symbol": not args.no_require_symbol,
        "require_digit": not args.no_require_digit,
    }

    if args.scenarios_file:
        data = Path(args.scenarios_file).read_text(encoding="utf-8")
        scenarios = json.loads(data)
        if not isinstance(scenarios, list):
            raise SystemExit("--scenarios-file must contain a JSON list of scenarios.")
        for s in scenarios:
            if not isinstance(s, dict):
                continue
            merged = {**gen_kw, **_scenario_options(s)}
            make_wordlist(
                str(s.get("context", "")),
                int(s.get("num_passwords", 2000)),
                str(s.get("outfile", "wordlist.txt")),
                **merged,
            )
        print("Done.")
        return

    context = args.context
    if context is None:
        print("Enter context (press Enter to skip and use default scenarios):")
        context = input("> ").strip()
        if not context:
            for s in SCENARIOS:
                merged = {**gen_kw, **_scenario_options(s)}
                make_wordlist(s["context"], s["num_passwords"], s["outfile"], **merged)
            print("Done.")
            return

    make_wordlist(context, args.num, args.outfile, **gen_kw)
    print("Done.")


if __name__ == "__main__":
    main()