from __future__ import annotations

import random
import re


UPPER_RE = re.compile(r"[A-Z]")
LOWER_RE = re.compile(r"[a-z]")
DIGIT_RE = re.compile(r"\d")
SYMBOL_RE = re.compile(r"[^A-Za-z0-9]")


def validate(password: str, min_len: int, max_len: int, require_digit: bool,
             require_symbol: bool, require_upper: bool, require_lower: bool) -> bool:
    if len(password) < min_len or len(password) > max_len:
        return False
    if require_digit and not DIGIT_RE.search(password):
        return False
    if require_symbol and not SYMBOL_RE.search(password):
        return False
    if require_upper and not UPPER_RE.search(password):
        return False
    if require_lower and not LOWER_RE.search(password):
        return False
    return True


def repair(password: str, min_len: int, max_len: int, require_digit: bool,
           require_symbol: bool, require_upper: bool, require_lower: bool) -> str:
    if len(password) > max_len:
        password = password[:max_len]

    if require_digit and not DIGIT_RE.search(password):
        password += str(random.randint(0, 9))
    if require_symbol and not SYMBOL_RE.search(password):
        password += random.choice("!@#$%^&*")
    if require_upper and not UPPER_RE.search(password):
        password = password[:1].upper() + password[1:]
    if require_lower and not LOWER_RE.search(password):
        password = password[:1].lower() + password[1:]

    if len(password) < min_len:
        password += "a" * (min_len - len(password))
    return password
