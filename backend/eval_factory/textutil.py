"""Token helpers shared by analysis, synthesis, and scoring."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """
    a an the and or of to for in on with from by as at is are was were be been being
    this that those these it its their his her your our they them we you i not no
    do does did can could should would will just than then into over under about
    what when where which who how why
    """.split()
)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in _STOP and len(token) > 2]


def recall(needles: list[str], haystack: list[str]) -> float:
    needed = set(needles)
    if not needed:
        return 0.0
    return len(needed & set(haystack)) / len(needed)


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clamp_score(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 4)


def excerpt(text: str, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
