"""Text cleaning, quality gating, and deduplication.

Quality-first ingest: strip boilerplate, drop junk, and remove duplicates
*before* anything reaches the vector store. Pure-Python and dependency-free so
it is fully unit-testable without torch/chromadb.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from fg.data.schema import Document

#: Lines that are almost always navigation / cookie / boilerplate noise.
_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(accept|manage)\s+cookies",
        r"^\s*(sign in|log in|subscribe|newsletter)\b",
        r"^\s*(share|tweet|pin it|copy link)\s*$",
        r"^\s*(advertisement|sponsored)\s*$",
        r"^\s*\d+\s*(min read|comments?)\s*$",
        r"^\s*(all rights reserved|©)",
    )
)

_WS_RE: re.Pattern[str] = re.compile(r"[ \t]+")
_MULTINEWLINE_RE: re.Pattern[str] = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapses runs of spaces/tabs and excess blank lines.

    Args:
        text: Raw text.

    Returns:
        Text with normalized whitespace and at most one blank line between
        paragraphs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return _MULTINEWLINE_RE.sub("\n\n", text).strip()


def strip_boilerplate(text: str) -> str:
    """Removes lines matching common boilerplate/navigation patterns.

    Args:
        text: Input text (ideally already whitespace-normalized).

    Returns:
        Text with boilerplate lines dropped.
    """
    kept: list[str] = [
        ln
        for ln in text.split("\n")
        if not any(p.search(ln) for p in _BOILERPLATE_PATTERNS)
    ]
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Full cleaning pass: unicode-normalize, de-boilerplate, tidy whitespace.

    Args:
        text: Raw text.

    Returns:
        Cleaned text (may be empty if everything was noise).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_whitespace(text)
    text = strip_boilerplate(text)
    return normalize_whitespace(text)


def content_hash(text: str) -> str:
    """Returns a stable hash of the *content* for dedup.

    Case- and whitespace-insensitive so trivially different copies collide.

    Args:
        text: Text to hash.

    Returns:
        Hex SHA-1 digest of the normalized content.
    """
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def is_low_quality(text: str, min_chars: int = 80, min_words: int = 15) -> bool:
    """Heuristic gate for junk / too-short documents.

    Args:
        text: Cleaned text.
        min_chars: Minimum character count to keep.
        min_words: Minimum word count to keep.

    Returns:
        ``True`` if the document should be dropped.
    """
    if not text or len(text) < min_chars:
        return True
    return len(text.split()) < min_words


def dedup_documents(docs: Iterable[Document]) -> list[Document]:
    """Removes exact / whitespace-and-case duplicate documents.

    Keeps the first occurrence of each unique content hash.

    Args:
        docs: Iterable of documents.

    Returns:
        Deduplicated list, order preserved.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        h = content_hash(doc.text)
        if h in seen:
            continue
        seen.add(h)
        unique.append(doc)
    return unique


def clean_documents(
    docs: Iterable[Document],
    *,
    min_chars: int = 80,
    min_words: int = 15,
) -> list[Document]:
    """Cleans, quality-gates, and deduplicates a document stream.

    Args:
        docs: Raw documents.
        min_chars: Minimum characters to keep a doc.
        min_words: Minimum words to keep a doc.

    Returns:
        Cleaned, filtered, deduplicated documents.
    """
    cleaned: list[Document] = []
    for doc in docs:
        text = clean_text(doc.text)
        if is_low_quality(text, min_chars=min_chars, min_words=min_words):
            continue
        cleaned.append(Document(text=text, metadata=dict(doc.metadata)))
    return dedup_documents(cleaned)
