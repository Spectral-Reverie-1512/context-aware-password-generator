from __future__ import annotations

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_DIM = 384
STRUCTURED_DIM = 64
CONTEXT_DIM = 128

MLP_HIDDEN = 256
DROPOUT = 0.1

SYMBOL_REPLACEMENTS = {
    "\n": " ",
    "\t": " ",
}

KEYWORD_TOP_K = 20

STRUCTURED_BUCKETS = {
    "names": 16,
    "dates": 8,
    "keywords": 16,
    "emails": 8,
    "usernames": 8,
    "numbers": 8,
}

