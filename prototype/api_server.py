from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
import json
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import io
import re
import random


DEFAULT_SYMBOLS = "!@#$%&_-"


def _extract_tokens(context: str) -> tuple[list[str], list[str]]:
    # Words: letters only; Digits: contiguous digit runs from text.
    words = re.findall(r"[A-Za-z]+", context.lower())
    digits = re.findall(r"\d+", context)

    stop_words = {
        "and",
        "the",
        "uses",
        "use",
        "user",
        "named",
        "pet",
        "loves",
        "with",
        "likes",
        "works",
        "work",
        "at",
        "in",
        "on",
        "for",
        "from",
        "to",
        "of",
        "a",
        "an",
        "is",
        "are",
        "this",
        "that",
    }
    words = [w for w in words if w not in stop_words and len(w) >= 2]

    # keep only reasonable digit chunks (avoid extremely long numbers)
    digits = [d for d in digits if 1 <= len(d) <= 6]

    return words, digits


def _clean_candidate(pwd: object) -> str | None:
    if not isinstance(pwd, str):
        return None
    s = pwd.strip()
    if not s:
        return None
    if any(ch.isspace() for ch in s):
        return None
    return s


def _enforce_policy(
    base: str,
    *,
    min_length: int,
    max_length: int,
    require_upper: bool,
    require_lower: bool,
    require_digit: bool,
    require_symbol: bool,
    symbols: str,
    rng: random.Random,
) -> str | None:
    s = base.strip()
    if not s or any(ch.isspace() for ch in s):
        return None

    # If too long, truncate but keep the tail (often contains digits/symbols).
    if len(s) > max_length:
        s = s[: max_length]

    def has_upper(x: str) -> bool:
        return any(ch.isupper() for ch in x)

    def has_lower(x: str) -> bool:
        return any(ch.islower() for ch in x)

    def has_digit(x: str) -> bool:
        return any(ch.isdigit() for ch in x)

    def has_symbol(x: str) -> bool:
        return any(ch in symbols for ch in x)

    chars = list(s)
    letter_idxs = [i for i, ch in enumerate(chars) if ch.isalpha()]

    if require_upper and not has_upper(s) and letter_idxs:
        i = rng.choice(letter_idxs)
        chars[i] = chars[i].upper()
    if require_lower and not has_lower("".join(chars)) and letter_idxs:
        i = rng.choice(letter_idxs)
        chars[i] = chars[i].lower()

    s2 = "".join(chars)

    # Append missing classes (keeps it simple and avoids whitespace)
    if require_digit and not has_digit(s2):
        s2 += rng.choice("0123456789")
    if require_symbol and not has_symbol(s2):
        s2 += rng.choice(symbols)

    # Pad to minimum length with a mix of digits/symbols (and occasional letters)
    while len(s2) < min_length:
        roll = rng.random()
        if roll < 0.45:
            s2 += rng.choice("0123456789")
        elif roll < 0.8:
            s2 += rng.choice(symbols)
        else:
            s2 += rng.choice("abcdefghijklmnopqrstuvwxyz").upper() if rng.random() < 0.5 else rng.choice("abcdefghijklmnopqrstuvwxyz")

    if len(s2) > max_length:
        s2 = s2[: max_length]

    if require_upper and not has_upper(s2):
        return None
    if require_lower and not has_lower(s2):
        return None
    if require_digit and not has_digit(s2):
        return None
    if require_symbol and not has_symbol(s2):
        return None
    if len(s2) < min_length:
        return None

    return s2


def generate_from_context(
    context: str,
    n: int,
    *,
    min_length: int = 10,
    max_length: int = 32,
    require_upper: bool = True,
    require_lower: bool = True,
    require_digit: bool = True,
    require_symbol: bool = True,
    symbols: str = DEFAULT_SYMBOLS,
    seed: int | None = None,
) -> list[str]:
    """
    Rule-based, context-aware generator that produces "robust" candidates:
    mixed case + digits + symbols with a minimum length, derived from context tokens.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    words, digits = _extract_tokens(context)
    if not words:
        words = ["user", "context"]

    # Prefer years / familiar fragments if present, else reasonable defaults.
    yearish = [d for d in digits if len(d) == 4 and d.startswith(("19", "20"))]
    digit_pool = (yearish or digits) or ["2026", "123", "007", "99"]

    separators = ["", "", "", ".", "_", "-", ""]
    symbol_pool = list(symbols) if symbols else list(DEFAULT_SYMBOLS)

    def maybe_leet(w: str) -> str:
        mapping = {"a": "@", "i": "1", "e": "3", "o": "0", "s": "$", "t": "7"}
        out = []
        for ch in w:
            if ch.lower() in mapping and rng.random() < 0.18:
                out.append(mapping[ch.lower()])
            else:
                out.append(ch)
        return "".join(out)

    def random_case(w: str) -> str:
        if not w:
            return w
        mode = rng.choice(["title", "lower", "upper", "camel", "alt"])
        if mode == "title":
            return w[:1].upper() + w[1:].lower()
        if mode == "lower":
            return w.lower()
        if mode == "upper":
            return w.upper()
        if mode == "camel":
            return "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(w))
        return "".join(ch.upper() if rng.random() < 0.45 else ch.lower() for ch in w)

    def build_word_part() -> str:
        k = 2 if rng.random() < 0.62 else 1
        k = 3 if rng.random() < 0.18 else k
        k = min(k, max(1, len(words)))
        chosen = rng.sample(words, k=k) if len(words) >= k else [rng.choice(words)]
        sep = rng.choice(separators)
        return sep.join(maybe_leet(random_case(w)) for w in chosen)

    def build_number_part() -> str:
        a = rng.choice(digit_pool)
        b = rng.choice(digit_pool) if rng.random() < 0.35 else ""
        extra = "".join(rng.choice("0123456789") for _ in range(rng.choice([0, 0, 1, 2, 3])))
        return f"{a}{b}{extra}"

    def build_symbol_part() -> str:
        count = rng.choice([1, 1, 2, 2, 3])
        return "".join(rng.choice(symbol_pool) for _ in range(count))

    candidates: set[str] = set()
    attempts = 0
    max_attempts = max(200, n * 50)

    def simple_word() -> str:
        w = rng.choice(words) if words else "user"
        return w[:1].upper() + w[1:].lower()

    def simple_digits() -> str:
        pool = digit_pool or ["2024", "2025", "2000", "123"]
        return rng.choice(pool)

    def simple_symbol() -> str:
        common = [c for c in ["@", "!", "#", "$", "_"] if c in symbol_pool]
        pool = common or symbol_pool
        return rng.choice(pool)

    # Seed with some "human-looking" patterns so output isn't purely "randomized strong".
    simple_target = min(int(n * 0.3), 1200)
    simple_attempts = 0
    while len(candidates) < simple_target and simple_attempts < max_attempts:
        simple_attempts += 1
        w1 = simple_word()
        # optionally add a second context word (simple capitalization)
        w2 = simple_word() if rng.random() < 0.35 else ""
        num = simple_digits()
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

        enforced = _enforce_policy(
            base,
            min_length=min_length,
            max_length=max_length,
            require_upper=require_upper,
            require_lower=require_lower,
            require_digit=require_digit,
            require_symbol=require_symbol,
            symbols=symbols or DEFAULT_SYMBOLS,
            rng=rng,
        )
        cleaned = _clean_candidate(enforced) if enforced is not None else None
        if cleaned:
            candidates.add(cleaned)

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        w = build_word_part()
        num = build_number_part()
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

        enforced = _enforce_policy(
            pattern,
            min_length=min_length,
            max_length=max_length,
            require_upper=require_upper,
            require_lower=require_lower,
            require_digit=require_digit,
            require_symbol=require_symbol,
            symbols=symbols or DEFAULT_SYMBOLS,
            rng=rng,
        )
        cleaned = _clean_candidate(enforced) if enforced is not None else None
        if cleaned:
            candidates.add(cleaned)

    return list(candidates)


class GenerateRequest(BaseModel):
    context: str = Field(..., description="Free-text context information")
    num_passwords: int = Field(
        10,
        ge=1,
        le=5000,
        description="Number of passwords to generate",
    )
    mode: Literal["auto", "ml", "rules"] = Field(
        "auto",
        description="Generation mode: auto (prefer ML), ml, or rules.",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional RNG seed for repeatable outputs (rule-based + postprocessing).",
    )
    min_length: int = Field(10, ge=5, le=64, description="Minimum password length.")
    max_length: int = Field(32, ge=5, le=128, description="Maximum password length.")
    require_upper: bool = Field(True, description="Require at least one uppercase letter.")
    require_lower: bool = Field(True, description="Require at least one lowercase letter.")
    require_digit: bool = Field(True, description="Require at least one digit.")
    require_symbol: bool = Field(True, description="Require at least one symbol.")
    symbols: str = Field(DEFAULT_SYMBOLS, description="Allowed symbols for enforcement.")


class GenerateResponse(BaseModel):
    passwords: List[str]


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Default model paths – adjust if your files live elsewhere.
TOKENIZER_PATH = PROJECT_DIR / "stage2_output" / "tokenizer.json"
ENCODER_PATH = PROJECT_DIR / "stage2_output" / "encoder_final.pt"
STAGE3_PATH = PROJECT_DIR / "stage3_output" / "stage3_final.pt"

app = FastAPI(title="Password Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state, populated on startup
DEVICE: str | None = None
TORCH_ERROR: str | None = None
tokenizer = None
stage2_encoder = None
stage3_model = None
context_vectorizer = None
stage3_config = None
gen_config = None
decode_tokens = None
sample_from_logits = None
extract_context = None
POPULAR_PASSWORDS: list[str] = []


@app.on_event("startup")
def load_models() -> None:
    """
    Lazily initialize global model objects used for generation.
    All heavy imports (torch + model code) are done here so the app
    can still start even if the ML stack is unavailable.
    """
    global tokenizer, stage2_encoder, stage3_model, context_vectorizer
    global DEVICE, TORCH_ERROR
    global stage3_config, gen_config, decode_tokens, sample_from_logits, extract_context

    try:
        import torch as _torch  # type: ignore
        from stage2.model import PasswordEncoder as _PasswordEncoder
        from stage3 import config as _stage3_config
        from stage3.data import ContextVectorizer as _ContextVectorizer, Stage3Tokenizer
        from stage3.model import Stage3Model as _Stage3Model
        from stage4 import config as _gen_config
        from stage4.decoder import decode_tokens as _decode_tokens
        from stage4.sampling import sample_from_logits as _sample_from_logits
        from stage25.extract import extract_context as _extract_context
    except Exception as exc:
        TORCH_ERROR = f"Model import failed: {exc}"
        tokenizer = None
        stage2_encoder = None
        stage3_model = None
        context_vectorizer = None
        DEVICE = "cpu"
        return

    # Expose imported modules / functions in globals for later use.
    globals()["torch"] = _torch
    stage3_config = _stage3_config
    gen_config = _gen_config
    decode_tokens = _decode_tokens
    sample_from_logits = _sample_from_logits
    extract_context = _extract_context

    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

    if not TOKENIZER_PATH.exists() or not ENCODER_PATH.exists() or not STAGE3_PATH.exists():
        # We don't raise here to keep the app start-up successful – errors surface on first request.
        tokenizer = None
        stage2_encoder = None
        stage3_model = None
        context_vectorizer = None
        return

    tokenizer = Stage3Tokenizer(TOKENIZER_PATH)

    stage2_encoder = _PasswordEncoder(
        vocab_size=len(tokenizer.token_to_id),
        d_model=stage3_config.ENC_DIM,
        ffn_hidden=512,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_len=tokenizer.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    stage2_encoder.load_state_dict(_torch.load(ENCODER_PATH, map_location="cpu"))
    stage2_encoder.to(DEVICE)
    stage2_encoder.eval()

    stage3_model = _Stage3Model(
        stage2_encoder=stage2_encoder,
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        max_len=tokenizer.max_seq_len,
    ).to(DEVICE)
    stage3_model.load_state_dict(_torch.load(STAGE3_PATH, map_location="cpu"))
    stage3_model.eval()

    context_vectorizer = _ContextVectorizer(device=DEVICE)


def _ensure_models_loaded() -> None:
    if any(obj is None for obj in (globals().get("tokenizer"), globals().get("stage3_model"))):
        raise HTTPException(
            status_code=500,
            detail=(
                "Model artifacts not found. Ensure tokenizer.json, encoder_final.pt, "
                "and stage3_final.pt are present under stage2_output/ and stage3_output/."
            ),
        )


def _load_popular_passwords() -> None:
    """
    Populate POPULAR_PASSWORDS from password_targets.json if not already loaded.
    This is used when the ML backend is unavailable, so we can still return
    realistic-looking passwords sampled from the training data.
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

    # Extract password strings; keep a reasonable subset to avoid huge memory use.
    passwords: list[str] = []
    for item in data:
        pwd = item.get("password")
        if isinstance(pwd, str):
            passwords.append(pwd)
        if len(passwords) >= 100_000:
            break

    if passwords:
        POPULAR_PASSWORDS = passwords


def _generate_one(context_vec: torch.Tensor, structured_context: dict) -> str:
    """
    Generate a single password string given a precomputed context vector.
    """
    _ensure_models_loaded()

    max_len = stage3_config.MAX_SEQ_LEN
    with torch.no_grad():
        z = torch.randn(1, stage3_config.LATENT_DIM, device=DEVICE)
        z_cond = stage3_model.context_mod(z, context_vec.unsqueeze(0))

        generated: list[int] = [tokenizer.bos_id]
        for _ in range(max_len - 1):
            ids = generated + [tokenizer.pad_id] * (max_len - len(generated))
            mask = [1] * len(generated) + [0] * (max_len - len(generated))
            input_ids = torch.tensor([ids], device=DEVICE)
            attention_mask = torch.tensor([mask], device=DEVICE)
            logits = stage3_model.decoder(input_ids, attention_mask, z_cond)
            next_logits = logits[0, len(generated) - 1]
            next_id = sample_from_logits(
                next_logits,
                temperature=gen_config.TEMPERATURE,
                top_k=gen_config.TOP_K,
                top_p=gen_config.TOP_P,
            )
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    token_strings = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in generated]
    password = decode_tokens(token_strings, context_struct=structured_context)

    # If the model produced an empty password, fall back to context generation
    if not password or password.strip() == "":
        fallback = generate_from_context(
            structured_context.get("raw_text", "user"), 1
        )[0]
        return fallback

    return password


@app.post("/api/generate-passwords", response_model=GenerateResponse)
def generate_passwords(request: GenerateRequest) -> GenerateResponse:
    context_text = request.context.strip()

    # If the ML stack failed to load (Torch DLL issue, etc.), fall back to
    # simple generators so the UI still works:
    # - WITH context: randomized, context-based patterns that try to build
    #   stronger passwords from multiple context hints.
    # - WITHOUT context: sample directly from the training targets.
    if TORCH_ERROR or request.mode == "rules":
        passwords = generate_from_context(
            context_text,
            request.num_passwords,
            min_length=request.min_length,
            max_length=request.max_length,
            require_upper=request.require_upper,
            require_lower=request.require_lower,
            require_digit=request.require_digit,
            require_symbol=request.require_symbol,
            symbols=request.symbols,
            seed=request.seed,
        )
        return GenerateResponse(passwords=passwords)
    # if TORCH_ERROR:
    #     if context_text:
    #         # Randomized, context-based patterns (mirrors api_server_light).
    #         raw_tokens = context_text.split()
    #         words = [
    #             "".join(filter(str.isalpha, w))
    #             for w in raw_tokens
    #             if any(ch.isalpha() for ch in w)
    #         ]
    #         digits = [
    #             "".join(filter(str.isdigit, w))
    #             for w in raw_tokens
    #             if any(ch.isdigit() for ch in w)
    #         ]
    #         words = [w.lower() for w in words if w] or ["context", "user", "sample"]
    #         digits = [d for d in digits if d] or ["123", "2026", "007"]
    #         symbols_pool = ["!", "@", "#", "$", "%", "&", "_", "-"]

    #         def random_case(word: str) -> str:
    #             return "".join(
    #                 ch.upper() if random.random() < 0.5 else ch.lower()
    #                 for ch in word
    #             )

    #         def maybe_leet(word: str) -> str:
    #             mapping = {"a": "@", "i": "1", "e": "3", "o": "0", "s": "$"}
    #             return "".join(
    #                 mapping.get(ch.lower(), ch) if random.random() < 0.25 else ch
    #                 for ch in word
    #             )

    #         def build_number_segment() -> str:
    #             """
    #             Build a numeric segment using 1–2 context-derived digit chunks,
    #             sometimes extended with extra random digits so overall length
    #             is more substantial and less predictable.
    #             """
    #             # If the context gave us explicit numbers (years, phone fragments, etc.)
    #             # bias toward using them, possibly in combination.
    #             base_chunks = digits or ["2026", "1234", "007", "99"]

    #             first = random.choice(base_chunks)
    #             use_second = random.random() < 0.4 and len(base_chunks) > 1
    #             second = random.choice(base_chunks) if use_second else ""

    #             segment = first + second

    #             # Occasionally add 1–3 extra random digits on top.
    #             extra_len = random.choice([0, 0, 1, 2, 3])
    #             if extra_len:
    #                 segment += "".join(random.choice("0123456789") for _ in range(extra_len))

    #             return segment

    #         def build_symbol_segment() -> str:
    #             """
    #             Build 1–3 symbols, allowing for repetition (e.g. '!!', '@#', '&!%').
    #             """
    #             count = random.choice([1, 1, 2, 2, 3])
    #             return "".join(random.choice(symbols_pool) for _ in range(count))

    #         def enforce_strength(pwd: str) -> str:
    #             """
    #             Apply simple heuristics so fallback passwords have:
    #             - mixed case
    #             - at least one digit
    #             - at least one symbol
    #             - minimum length of 10 characters
    #             """
    #             chars = list(pwd)

    #             has_upper = any(ch.isupper() for ch in chars)
    #             has_lower = any(ch.islower() for ch in chars)
    #             has_digit = any(ch.isdigit() for ch in chars)
    #             has_symbol = any(ch in symbols_pool for ch in chars)

    #             # Ensure at least one upper/lower by toggling a few letters if possible.
    #             letter_indices = [i for i, ch in enumerate(chars) if ch.isalpha()]
    #             if letter_indices and not has_upper:
    #                 idx = random.choice(letter_indices)
    #                 chars[idx] = chars[idx].upper()
    #                 has_upper = True
    #             if letter_indices and not has_lower:
    #                 idx = random.choice(letter_indices)
    #                 chars[idx] = chars[idx].lower()
    #                 has_lower = True

    #             # Ensure at least one digit and one symbol by appending if missing.
    #             if not has_digit:
    #                 chars.append(random.choice("0123456789"))
    #                 has_digit = True
    #             if not has_symbol:
    #                 chars.append(random.choice(symbols_pool))
    #                 has_symbol = True

    #             # Enforce minimum length by padding with digits/symbols.
    #             while len(chars) < 10:
    #                 if random.random() < 0.5:
    #                     chars.append(random.choice("0123456789"))
    #                 else:
    #                     chars.append(random.choice(symbols_pool))

    #             return "".join(chars)

    #         def gen_one() -> str:
    #             # Choose 1–3 context words (without replacement when possible).
    #             max_words = min(3, len(words))
    #             if max_words <= 1:
    #                 chosen = [random.choice(words)]
    #             else:
    #                 # Bias toward 2 words, sometimes 3.
    #                 k = random.choices(
    #                     population=[1, 2, 3][:max_words],
    #                     weights=[1, 4, 2][:max_words],
    #                     k=1,
    #                 )[0]
    #                 chosen = random.sample(words, k=k)

    #             word_parts = [maybe_leet(random_case(w)) for w in chosen]
    #             words_joined = "".join(word_parts)

    #             num_segment = build_number_segment()
    #             sym_segment = build_symbol_segment()

    #             # A richer set of patterns combining multiple words, digits, and symbols.
    #             pattern = random.choice(
    #                 [
    #                     "WNS",     # words + numbers + symbols
    #                     "WSN",     # words + symbols + numbers
    #                     "NW",      # numbers + words
    #                     "NSW",     # numbers + symbols + words
    #                     "SWN",     # symbols + words + numbers
    #                     "WNSW",    # words + numbers + symbols + words
    #                     "SWWN",    # symbols + words + words + numbers
    #                     "WNWS",    # words + numbers + words + symbols
    #                 ]
    #             )

    #             if pattern == "WNS":
    #                 base = f"{words_joined}{num_segment}{sym_segment}"
    #             elif pattern == "WSN":
    #                 base = f"{words_joined}{sym_segment}{num_segment}"
    #             elif pattern == "NW":
    #                 base = f"{num_segment}{words_joined}"
    #             elif pattern == "NSW":
    #                 base = f"{num_segment}{sym_segment}{words_joined}"
    #             elif pattern == "SWN":
    #                 base = f"{sym_segment}{words_joined}{num_segment}"
    #             elif pattern == "WNSW":
    #                 base = f"{words_joined}{num_segment}{sym_segment}{words_joined[: max(1, len(words_joined)//3)]}"
    #             elif pattern == "SWWN":
    #                 base = f"{sym_segment}{words_joined}{words_joined[: max(1, len(words_joined)//3)]}{num_segment}"
    #             elif pattern == "WNWS":
    #                 base = f"{words_joined}{num_segment}{words_joined[: max(1, len(words_joined)//3)]}{sym_segment}"
    #             else:
    #                 base = f"{words_joined}{num_segment}{sym_segment}"

    #             return enforce_strength(base)

    #         passwords: list[str] = [gen_one() for _ in range(request.num_passwords)]
    #         return GenerateResponse(passwords=passwords)

    #     # No context: sample from training targets.
    #     _load_popular_passwords()
    #     if not POPULAR_PASSWORDS:
    #         stems = ["password", "dragon", "qwerty", "admin", "letmein"]
    #         numbers = ["123", "2026", "321", "007", "99"]
    #         symbols = ["!", "@", "#", "$", "%"]
    #         passwords: list[str] = []
    #         for i in range(request.num_passwords):
    #             word = stems[i % len(stems)]
    #             num = numbers[i % len(numbers)]
    #             sym = symbols[i % len(symbols)]
    #             passwords.append(f"{word.capitalize()}{num}{sym}")
    #         return GenerateResponse(passwords=passwords)

    #     passwords: list[str] = []
    #     for _ in range(request.num_passwords):
    #         passwords.append(random.choice(POPULAR_PASSWORDS))
    #     return GenerateResponse(passwords=passwords)

    _ensure_models_loaded()

    # Extract structured hints from the raw context (numbers, symbols, etc.).
    if context_text:
        structured_context = extract_context(context_text)
        # Build context vector using both raw and structured views.
        context_vec = context_vectorizer.encode(context_text, structured_context).to(DEVICE)
    else:
        # No context provided: use an all-zero context vector so the model
        # falls back to its notion of globally likely passwords.
        structured_context = {}
        context_vec = torch.zeros(stage3_config.CONTEXT_DIM, device=DEVICE)

    passwords: set[str] = set()
    max_attempts = max(200, request.num_passwords * 50)
    attempts = 0

    while len(passwords) < request.num_passwords and attempts < max_attempts:
        attempts += 1
        pwd = _generate_one(context_vec, structured_context)
        cleaned = _clean_candidate(pwd)
        if not cleaned:
            continue

        # Postprocess ML output to meet policy; if it can't, skip it.
        enforced = _enforce_policy(
            cleaned,
            min_length=request.min_length,
            max_length=request.max_length,
            require_upper=request.require_upper,
            require_lower=request.require_lower,
            require_digit=request.require_digit,
            require_symbol=request.require_symbol,
            symbols=request.symbols,
            rng=random.Random(request.seed) if request.seed is not None else random.Random(),
        )
        final_pwd = _clean_candidate(enforced) if enforced is not None else None
        if final_pwd:
            passwords.add(final_pwd)

    # If ML didn't produce enough usable candidates, top-up with rule-based ones.
    if len(passwords) < request.num_passwords and request.mode in ("auto", "ml"):
        needed = request.num_passwords - len(passwords)
        topup = generate_from_context(
            context_text,
            needed,
            min_length=request.min_length,
            max_length=request.max_length,
            require_upper=request.require_upper,
            require_lower=request.require_lower,
            require_digit=request.require_digit,
            require_symbol=request.require_symbol,
            symbols=request.symbols,
            seed=request.seed,
        )
        passwords.update(topup)

    return GenerateResponse(passwords=sorted(passwords)[: request.num_passwords])


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

