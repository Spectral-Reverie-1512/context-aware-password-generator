from __future__ import annotations

MAX_SEQ_LEN = 32

D_MODEL = 256
FFN_HIDDEN = 512
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

LR = 3e-4
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 4000
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05

CONTRASTIVE_TEMPERATURE = 0.1
CONTRASTIVE_DROPOUT = 0.15

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

STRUCT_TAGS = [
    "keyboard",
    "leet",
    "year",
    "word",
    "letters",
    "digits",
    "symbols",
]
STRUCT_TOKENS = [f"<S_{tag}>" for tag in STRUCT_TAGS]

DIGIT_TOKENS = [str(d) for d in range(10)]

SYMBOL_GROUPS = {
    "SYM_PUNCT": ".!?,;:",
    "SYM_QUOTES": "\"'",
    "SYM_BRACKETS": "()[]{}<>",
    "SYM_MATH": "+-*/=%^",
    "SYM_CURRENCY": "$€£₹¥",
    "SYM_MISC": "@#&_~|\\",
}
SYMBOL_TOKENS = [f"<{name}>" for name in SYMBOL_GROUPS]

