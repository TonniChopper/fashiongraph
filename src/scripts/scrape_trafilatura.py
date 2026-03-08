"""Scrape fashion articles using Trafilatura's built-in boilerplate removal.

Provides ``scrape_fashion_article(url)`` which fetches a page, extracts
the main content and metadata via Trafilatura, and returns a clean dict.

No BeautifulSoup is used — all extraction relies on Trafilatura internals.

Usage::

    python -m src.scripts.scrape_trafilatura
"""

from __future__ import annotations

import json
import logging

import trafilatura

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------


def scrape_fashion_article(url: str) -> dict[str, str | None]:
    """Fetches and extracts the main content of a fashion article.

    Uses Trafilatura for both downloading and content extraction.
    Comments and embedded links are stripped automatically.

    Args:
        url: Full URL of the article to scrape.

    Returns:
        Dictionary with keys ``title``, ``author``, ``date``, and
        ``clean_text``.  Values are ``None`` when the field cannot
        be extracted.

    Raises:
        RuntimeError: If the URL cannot be fetched or content extraction
            fails entirely.
    """
    try:
        downloaded: str | None = trafilatura.fetch_url(url)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    if downloaded is None:
        raise RuntimeError(f"Trafilatura returned None for {url}")

    raw_json: str | None = trafilatura.extract(
        downloaded,
        output_format="json",
        include_comments=False,
        include_links=False,
        with_metadata=True,
        favor_recall=True,
    )

    if raw_json is None:
        raise RuntimeError(f"Content extraction returned None for {url}")

    data: dict = json.loads(raw_json)

    return {
        "title": data.get("title"),
        "author": data.get("author"),
        "date": data.get("date"),
        "clean_text": data.get("text"),
    }


# ------------------------------------------------------------------
# Test loop
# ------------------------------------------------------------------

TEST_URLS: list[str] = [
    "https://www.dazeddigital.com/fashion/article/65680/1/callon-the-new-label-breathing-celtic-spirit-into-sensual-spidery-knits",
    "https://www.dazeddigital.com/fashion/article/66341/1/prada-aw26-fw26-milan-fashion-week-mfw-miuccia-raf-simons-bella-hadid",
    "https://hypebeast.com/2026/3/bottega-veneta-winter-2026-milan-fashion-week-runway",
]


def main() -> None:
    """Scrapes a short list of test URLs and prints the results."""
    for url in TEST_URLS:
        logger.info("Scraping: %s", url)
        try:
            article: dict[str, str | None] = scrape_fashion_article(url)
        except RuntimeError as exc:
            logger.error("  ✗ %s", exc)
            continue

        logger.info("  Title : %s", article["title"])
        logger.info("  Author: %s", article["author"])
        logger.info("  Date  : %s", article["date"])

        text: str | None = article["clean_text"]
        preview: str = (text[:200] + "…") if text and len(text) > 200 else (text or "")
        logger.info("  Text  : %s", preview)
        logger.info("")


if __name__ == "__main__":
    main()

