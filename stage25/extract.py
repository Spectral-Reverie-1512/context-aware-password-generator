from __future__ import annotations

import re
from collections import Counter

from . import config

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
AT_USERNAME_RE = re.compile(r"\B@\w{3,}\b")
USER_FIELD_RE = re.compile(r"\buser(?:name)?\s*[:=]\s*([A-Za-z0-9_.-]{3,})\b", re.IGNORECASE)

DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
DATE_ALT_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
MONTH_RE = re.compile(
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)

NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d{2,}\b")
WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "you",
    "your",
    "are",
    "was",
    "have",
    "has",
    "had",
    "but",
    "not",
    "can",
    "will",
    "all",
    "any",
    "our",
    "out",
    "into",
    "about",
    "more",
    "than",
    "just",
    "like",
    "they",
    "them",
    "their",
}


def normalize_text(text: str) -> str:
    for src, dst in config.SYMBOL_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.strip()


def extract_context(text: str) -> dict[str, list[str]]:
    text = normalize_text(text)
    if not text:
        return {"names": [], "dates": [], "keywords": [], "emails": [], "usernames": [], "numbers": []}

    emails = EMAIL_RE.findall(text)
    usernames = AT_USERNAME_RE.findall(text)
    usernames.extend(USER_FIELD_RE.findall(text))

    dates = DATE_RE.findall(text)
    dates.extend(DATE_ALT_RE.findall(text))
    dates.extend(MONTH_RE.findall(text))

    names = NAME_RE.findall(text)

    numbers = NUMBER_RE.findall(text)

    words = [w.lower() for w in WORD_RE.findall(text)]
    keywords = []
    if words:
        counts = Counter(w for w in words if w not in STOPWORDS)
        keywords = [w for w, _ in counts.most_common(config.KEYWORD_TOP_K)]

    return {
        "names": list(dict.fromkeys(names)),
        "dates": list(dict.fromkeys(dates)),
        "keywords": keywords,
        "emails": list(dict.fromkeys(emails)),
        "usernames": list(dict.fromkeys(usernames)),
        "numbers": list(dict.fromkeys(numbers)),
    }

