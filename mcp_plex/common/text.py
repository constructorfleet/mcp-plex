"""Text normalization helpers shared across loader and server components."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["slugify", "strip_leading_article"]

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def strip_leading_article(text: str | None) -> str | None:
    """Remove a leading English article (The/A/An) while preserving fallbacks."""

    if not text:
        return text
    stripped = _LEADING_ARTICLE_RE.sub("", text).strip()
    return stripped or text


def slugify(value: str | None) -> str | None:
    """Return a lowercase ASCII slug derived from *value*, or ``None`` when empty."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    slug = _SLUG_CLEAN_RE.sub("-", lowered).strip("-")
    return slug or None
