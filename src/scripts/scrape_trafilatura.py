"""Scrape fashion articles from a URL list using Trafilatura.

Reads URLs from ``data/urls.txt`` (one per line), extracts the main
content via Trafilatura's built-in boilerplate removal, and saves each
article as a ``.txt`` file to ``data/raw/trafilatura/``.

No BeautifulSoup is used — all extraction relies on Trafilatura internals.

Usage::

    python -m src.scripts.scrape_trafilatura
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import trafilatura

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

URLS_FILE: Path = Path("data/urls.txt")
OUTPUT_DIR: Path = Path("data/raw/trafilatura")
DELAY_SECONDS: float = 1.0


# ------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------


def scrape_fashion_article(url: str) -> dict[str, str | None]:
    """Fetches and extracts the main content of a fashion article.

    Args:
        url: Full URL of the article to scrape.

    Returns:
        Dictionary with keys ``title``, ``author``, ``date``, and
        ``clean_text``.

    Raises:
        RuntimeError: If the URL cannot be fetched or extraction fails.
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
# Helpers
# ------------------------------------------------------------------


def _url_to_filename(url: str) -> str:
    """Converts a URL to a safe filename slug.

    Args:
        url: Article URL.

    Returns:
        Filename string ending in ``.txt``.
    """
    # Strip protocol and domain
    slug: str = re.sub(r"https?://(www\.)?", "", url)
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug)
    # Trim leading/trailing hyphens and limit length
    slug = slug.strip("-")[:120]
    return f"{slug}.txt"


def _load_urls(path: Path) -> list[str]:
    """Reads URLs from a text file, one per line.

    Skips empty lines and lines starting with ``#``.

    Args:
        path: Path to the URLs file.

    Returns:
        List of URL strings.
    """
    if not path.exists():
        logger.error("URL file not found: %s", path)
        return []

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def main() -> None:
    """Reads URLs from file, scrapes each, saves as .txt."""
    urls: list[str] = _load_urls(URLS_FILE)
    if not urls:
        logger.warning("No URLs to process. Add URLs to %s.", URLS_FILE)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %d URLs → %s", len(urls), OUTPUT_DIR)

    saved: int = 0
    skipped: int = 0

    for idx, url in enumerate(urls, start=1):
        logger.info("[%d/%d] %s", idx, len(urls), url)

        filename: str = _url_to_filename(url)
        dest: Path = OUTPUT_DIR / filename

        if dest.exists():
            logger.info("  Already exists, skipping: %s", filename)
            skipped += 1
            continue

        try:
            article: dict[str, str | None] = scrape_fashion_article(url)
        except RuntimeError as exc:
            logger.error("  ✗ %s", exc)
            skipped += 1
            time.sleep(DELAY_SECONDS)
            continue

        title: str = article["title"] or "Untitled"
        author: str = article["author"] or "Unknown"
        date: str = article["date"] or "Unknown"
        text: str = article["clean_text"] or ""

        if len(text.split()) < 30:
            logger.warning("  Too short after extraction (%d words), skipping.", len(text.split()))
            skipped += 1
            time.sleep(DELAY_SECONDS)
            continue

        content: str = (
            f"TITLE: {title}\n"
            f"AUTHOR: {author}\n"
            f"DATE: {date}\n"
            f"URL: {url}\n\n"
            f"{text}\n"
        )
        dest.write_text(content, encoding="utf-8")
        saved += 1
        logger.info("  ✅ Saved: %s (%d words)", filename, len(text.split()))

        time.sleep(DELAY_SECONDS)

    logger.info(
        "🏁 Done. Saved %d articles, skipped %d. Output: %s",
        saved, skipped, OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()

