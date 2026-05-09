

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
import json
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

import io
import re


class GenerateRequest(BaseModel):
    context: str = Field("", description="Free-text context information")
    num_passwords: int = Field(
        10,
        ge=1,
        le=5000,
        description="Number of passwords to generate",
    )
    mode: Literal["auto", "ml", "rules"] = Field(
        "rules",
        description="Generation mode (light server uses rules).",
    )
    seed: Optional[int] = Field(None, description="Optional RNG seed for repeatable outputs.")
    min_length: int = Field(10, ge=5, le=64, description="Minimum password length.")
    max_length: int = Field(32, ge=5, le=128, description="Maximum password length.")
    require_upper: bool = Field(True, description="Require at least one uppercase letter.")
    require_lower: bool = Field(True, description="Require at least one lowercase letter.")
    require_digit: bool = Field(True, description="Require at least one digit.")
    require_symbol: bool = Field(True, description="Require at least one symbol.")
    symbols: str = Field("!@#$%&_-", description="Allowed symbols.")


class GenerateResponse(BaseModel):
    passwords: List[str]


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

app = FastAPI(title="Password Generator (Light)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


POPULAR_PASSWORDS: list[str] = []


def _load_popular_passwords() -> None:
    """
    Populate POPULAR_PASSWORDS from password_targets.json if not already loaded.
    Used when no context is provided to sample realistic passwords from data.
    """
    global POPULAR_PASSWORDS
    if POPULAR_PASSWORDS:
        return

    targets_path = PROJECT_DIR / "context" / "password_targets.json"
    if not targets_path.exists():
        return

    try:
        data = json.loads(targets_path.read_text(encoding="utf-8"))
    except Exception:
        return

    passwords: list[str] = []
    for item in data:
        pwd = item.get("password")
        if isinstance(pwd, str):
            passwords.append(pwd)
        if len(passwords) >= 100_000:
            break

    if passwords:
        POPULAR_PASSWORDS = passwords


@app.post("/api/generate-passwords", response_model=GenerateResponse)
def generate_passwords(request: GenerateRequest) -> GenerateResponse:
    context_text = (request.context or "").strip()
    rng = random.Random(request.seed) if request.seed is not None else random.Random()

    symbols_pool = list(request.symbols) if request.symbols else ["!", "@", "#", "$", "%", "&", "_", "-"]

    def enforce_strength(pwd: str) -> str | None:
        s = (pwd or "").strip()
        if not s or any(ch.isspace() for ch in s):
            return None
        if len(s) > request.max_length:
            s = s[: request.max_length]

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
        if request.require_upper and not has_upper(s) and letter_idxs:
            i = rng.choice(letter_idxs)
            chars[i] = chars[i].upper()
        if request.require_lower and not has_lower("".join(chars)) and letter_idxs:
            i = rng.choice(letter_idxs)
            chars[i] = chars[i].lower()

        out = "".join(chars)
        if request.require_digit and not has_digit(out):
            out += rng.choice("0123456789")
        if request.require_symbol and not has_symbol(out):
            out += rng.choice(symbols_pool)

        while len(out) < request.min_length:
            out += rng.choice("0123456789") if rng.random() < 0.55 else rng.choice(symbols_pool)

        if len(out) > request.max_length:
            out = out[: request.max_length]

        if request.require_upper and not has_upper(out):
            return None
        if request.require_lower and not has_lower(out):
            return None
        if request.require_digit and not has_digit(out):
            return None
        if request.require_symbol and not has_symbol(out):
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
        # bias to 2 words, sometimes 1 or 3
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

    # WITH context: derive richer, mixed passwords from context tokens.
    if context_text:
        words, digits = extract_tokens(context_text)

        # Add some "human-looking" candidates first (simple, predictable patterns),
        # then fill the rest with more diverse variants.
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
            # Bias toward the most common "human" symbols.
            common = [c for c in ["@", "!", "#", "$", "_"] if c in symbols_pool]
            pool = common or symbols_pool
            return rng.choice(pool)

        passwords: list[str] = []
        seen: set[str] = set()
        attempts = 0
        # Keep a practical ceiling for big requests so we don't spin forever
        # when the dedupe set gets saturated.
        max_attempts = max(1000, request.num_passwords * 25)

        # Roughly 30% simple patterns (capped) to improve "coverage check" realism.
        simple_target = min(int(request.num_passwords * 0.3), 1200)
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

        while len(passwords) < request.num_passwords and attempts < max_attempts:
            attempts += 1
            w = build_word_part(words)
            num = build_number_part(digits)
            sym = build_symbol_part()

            pattern = rng.choice(
                [
                    f"{w}{num}{sym}",
                    f"{w}{sym}{num}",
                    f"{sym}{w}{num}",
                    f"{num}{w}{sym}",
                    f"{w}{sym}{num}{w[: max(1, len(w)//3)]}",
                ]
            )

            cand = enforce_strength(pattern)
            if not cand or cand in seen:
                continue
            seen.add(cand)
            passwords.append(cand)

        return GenerateResponse(passwords=passwords)

    # WITHOUT context: sample directly from training targets (popular passwords).
    _load_popular_passwords()
    if not POPULAR_PASSWORDS:
        stems = ["password", "dragon", "qwerty", "admin", "letmein"]
        numbers = ["123", "2026", "321", "007", "99"]
        passwords: list[str] = []
        for i in range(request.num_passwords):
            word = stems[i % len(stems)]
            num = numbers[i % len(numbers)]
            sym = symbols_pool[i % len(symbols_pool)]
            cand = enforce_strength(f"{word.capitalize()}{num}{sym}") or f"{word.capitalize()}{num}{sym}"
            passwords.append(cand)
        return GenerateResponse(passwords=passwords)

    passwords: list[str] = []
    for _ in range(request.num_passwords):
        passwords.append(rng.choice(POPULAR_PASSWORDS))
    return GenerateResponse(passwords=passwords)


class CreatePdfRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=128, description="Password to encrypt the PDF.")
    content: str = Field(
        "This is a demo protected PDF generated by the API.",
        max_length=500,
        description="Short text stored as PDF metadata (demo purpose).",
    )
    filename: str = Field("demo_protected.pdf", max_length=120, description="Download filename.")


@app.post("/api/create-demo-pdf")
def create_demo_pdf(request: CreatePdfRequest):
    try:
        from pypdf import PdfWriter  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="pypdf is required on the server. Run: pip install pypdf",
        )

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Demo protected PDF", "/Subject": request.content[:200]})

    # RC4-128 works without extra deps; AES-256 needs cryptography.
    writer.encrypt(request.password, algorithm="RC4-128")

    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    filename = request.filename.strip() or "demo_protected.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


# Serve the static frontend if the directory exists.
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
