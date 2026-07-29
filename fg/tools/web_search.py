"""Web search + self-feeding knowledge — the agentic 'eat fresh data' loop.

The knowledge graph and RAG are static snapshots. This tool lets the agent reach
the live web (via DuckDuckGo, no API key), and — crucially — **fold what it finds
back into its own memory**: results are chunked, extracted into KG triples, and
indexed into RAG. So the assistant answers current questions ("who's creative
director of X *now*?", "latest trends") *and* grows its knowledge base over time,
reusing the exact same `extract_triples` + RAG indexer built for books/Wikipedia.

Search backend: `ddgs` (DuckDuckGo) — `pip install ddgs`. Page text extraction
uses `trafilatura` if present, else BeautifulSoup, else a regex fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def web_search(query: str, k: int = 5, region: str = "wt-wt") -> list[dict[str, str]]:
    """Runs a DuckDuckGo text search.

    Args:
        query: Search query.
        k: Number of results.
        region: DDG region code.

    Returns:
        A list of ``{"title", "url", "snippet"}`` dicts (empty on failure).
    """
    try:
        try:
            from ddgs import DDGS            # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # older name
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Web search needs ddgs: pip install ddgs") from exc

    out: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=k):
                out.append({
                    "title": (r.get("title") or "").strip(),
                    "url": (r.get("href") or r.get("url") or "").strip(),
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed (%s).", exc)
    return out


# ---------------------------------------------------------------------------
# Page-text extraction
# ---------------------------------------------------------------------------

def _clean_html(html: str) -> str:
    """Strips tags/scripts to readable text (regex fallback)."""
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def fetch_text(url: str, timeout: int = 12, max_chars: int = 8000) -> str:
    """Fetches and extracts the main readable text of a page (best-effort).

    Tries trafilatura → BeautifulSoup → regex. Returns ``""`` on any failure so
    a single bad page never breaks the loop.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Page fetch needs requests: pip install requests") from exc
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0 (FashionGraph research)"})
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:  # noqa: BLE001
        logger.info("fetch_text: %s failed (%s)", url, exc)
        return ""

    try:  # best article extractor — purpose-built, actively maintained
        import trafilatura
        text = trafilatura.extract(html) or ""
        if text:
            return text[:max_chars]
    except Exception:  # noqa: BLE001
        pass
    try:  # fast modern HTML parser (lexbor); drops boilerplate tags
        from selectolax.parser import HTMLParser
        tree = HTMLParser(html)
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            for node in tree.css(tag):
                node.decompose()
        body = tree.body or tree
        return _WS_RE.sub(" ", body.text(separator=" ")).strip()[:max_chars]
    except Exception:  # noqa: BLE001
        return _clean_html(html)[:max_chars]


def _chunks(text: str, size: int = 350, max_chunks: int = 4) -> list[str]:
    """Splits text into up to *max_chunks* word-windows for extraction."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)][:max_chunks]


# ---------------------------------------------------------------------------
# The self-feeding loop
# ---------------------------------------------------------------------------

def ingest_results(
    results: list[dict],
    llm: Any | None = None,
    *,
    kg: Any | None = None,
    indexer: Any | None = None,
    fetch_pages: bool = True,
    max_chunks_per_page: int = 3,
) -> dict:
    """Folds already-gathered search *results* into the KG (+ RAG).

    Separated from :func:`learn_from_web` so the autonomous agent can grow its
    knowledge from the *same* results it already read while answering — no second
    search.

    Args:
        results: ``[{"title","url","snippet"}]`` (e.g. from :func:`web_search`).
        llm: LLM for triple extraction (required to grow the KG).
        kg: A ``KnowledgeGraph`` to add triples to (optional).
        indexer: A ``FashionKnowledgeIndexer`` to index passages into (optional).
        fetch_pages: Fetch full page text (richer) vs. use snippets only.
        max_chunks_per_page: Cap extraction windows per page (LLM-call budget).

    Returns:
        Stats: ``{"triples_added", "chunks_indexed", "sources"}``.
    """
    from fg.kg.extractor import extract_triples

    triples_added = 0
    chunks_indexed = 0
    sources: list[str] = []

    for r in results:
        url = r.get("url", "")
        text = r.get("snippet", "")
        if fetch_pages and url:
            page = fetch_text(url)
            if len(page.split()) > len(text.split()):
                text = page
        if len(text.split()) < 20:
            continue
        title = r.get("title") or ""

        if kg is not None and llm is not None:
            for chunk in _chunks(text, max_chunks=max_chunks_per_page):
                triples_added += kg.add_triples(extract_triples(chunk, llm, source=url or title))
        if indexer is not None:
            try:
                chunks_indexed += indexer.add_document(
                    text, {"source": "web", "source_type": "web",
                           "title": title, "url": url})
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG index failed for %s (%s).", url, exc)
        sources.append(url)

    return {"triples_added": triples_added, "chunks_indexed": chunks_indexed,
            "sources": sources}


def learn_from_web(
    query: str,
    llm: Any | None = None,
    *,
    kg: Any | None = None,
    indexer: Any | None = None,
    k: int = 5,
    fetch_pages: bool = True,
    max_chunks_per_page: int = 3,
) -> dict:
    """Searches the web and folds results into the KG (+ RAG).

    Returns:
        Stats: ``{"results", "triples_added", "chunks_indexed", "sources"}``.
    """
    results = web_search(query, k)
    stats = {"results": len(results), **ingest_results(
        results, llm, kg=kg, indexer=indexer,
        fetch_pages=fetch_pages, max_chunks_per_page=max_chunks_per_page)}
    logger.info("learn_from_web(%r): %s", query, stats)
    return stats
