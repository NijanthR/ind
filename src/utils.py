from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import math
import re
from typing import Iterable


WORD_RE = re.compile(r"[a-z0-9][a-z0-9+._&/-]{1,}", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def days_between(start: date | None, end: date | None) -> int | None:
    if not start or not end:
        return None
    return (end - start).days


def months_between(start: date | None, end: date | None) -> int | None:
    if not start or not end:
        return None
    return max(0, round((end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0))


def tokenise(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def hashed_vector(text: str, dimensions: int = 512) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenise(text)
    if not tokens:
        return vector
    for token in tokens:
        index = hash(token) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    list_a = list(vector_a)
    list_b = list(vector_b)
    length = min(len(list_a), len(list_b))
    if length == 0:
        return 0.0
    dot_product = sum(list_a[index] * list_b[index] for index in range(length))
    norm_a = math.sqrt(sum(value * value for value in list_a))
    norm_b = math.sqrt(sum(value * value for value in list_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot_product / (norm_a * norm_b)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def top_terms(text: str, limit: int = 10) -> list[str]:
    tokens = [token for token in tokenise(text) if len(token) > 2]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def format_sentence(parts: list[str]) -> str:
    filtered = [part.strip() for part in parts if part and part.strip()]
    if not filtered:
        return ""
    sentence = "; ".join(filtered)
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def string_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(left, right) / 100.0
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left.lower(), right.lower()).ratio()
