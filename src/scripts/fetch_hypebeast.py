"""Scrape fashion articles from Hypebeast using Crawl4AI.

Fetches up to 100 article URLs from hypebeast.com/tags/fashion,
then crawls each article page and saves the extracted text to
``data/raw/hypebeast/<slug>.txt``.

Usage::

    python -m src.scripts.fetch_hypebeast
"""

import asyncio
import logging
import re
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

BASE_URL: str = "https://hypebeast.com/tags/fashion"
OUTPUT_DIR: Path = Path("data/raw/hypebeast")
MAX_ARTICLES: int = 100
DELAY_SECONDS: float = 1.0


def _slug_from_url(url: str) -> str:
    """Derives a filesystem-safe slug from an article URL.

    Args:
        url: Full article URL.

    Returns:
        A sanitised string usable as a filename (without extension).
    """
    last_segment: str = url.rstrip("/").split("/")[-1]
    slug: str = re.sub(r"[^\w\-]", "_", last_segment)
    return slug[:120] or "article"


def _extract_article_links(markdown: str) -> list[str]:
    """Extracts unique Hypebeast article links from crawled markdown.

    Looks for links matching the ``hypebeast.com/YYYY/...`` pattern which
    is the standard article URL format.

    Args:
        markdown: Raw markdown returned by Crawl4AI.

    Returns:
        Deduplicated list of article URLs.
    """
    pattern: str = r"https?://hypebeast\.com/\d{4}/\d{1,2}/[^\s\)\]\"\'>]+"
    raw_links: list[str] = re.findall(pattern, markdown)
    seen: set[str] = set()
    unique: list[str] = []
    for link in raw_links:
        clean: str = link.rstrip(".,;:!?)")
        if clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return unique


async def _fetch_listing_pages(
    crawler: AsyncWebCrawler,
    run_cfg: CrawlerRunConfig,
) -> list[str]:
    """Crawls paginated listing pages until MAX_ARTICLES links are collected.

    Args:
        crawler: An already-started ``AsyncWebCrawler`` instance.
        run_cfg: Shared crawler run configuration.

    Returns:
        List of article URLs (up to ``MAX_ARTICLES``).
    """
    all_links: list[str] = []
    page: int = 1

    while len(all_links) < MAX_ARTICLES:
        url: str = f"{BASE_URL}/page/{page}" if page > 1 else BASE_URL
        logger.info("Crawling listing page %d: %s", page, url)

        try:
            result = await crawler.arun(url=url, config=run_cfg)
        except Exception as exc:
            logger.error("Failed to crawl listing page %d: %s", page, exc)
            break

        if not result.success:
            logger.warning(
                "Listing page %d returned failure (status=%s). Stopping pagination.",
                page,
                getattr(result, "status_code", "unknown"),
            )
            break

        new_links: list[str] = _extract_article_links(result.markdown)
        if not new_links:
            logger.info("No more article links found on page %d. Stopping.", page)
            break

        all_links.extend(new_links)
        logger.info(
            "Page %d: found %d links (total so far: %d).",
            page,
            len(new_links),
            len(all_links),
        )

        page += 1
        await asyncio.sleep(DELAY_SECONDS)

    unique_links: list[str] = list(dict.fromkeys(all_links))[:MAX_ARTICLES]
    logger.info("Collected %d unique article URLs.", len(unique_links))
    return unique_links


async def _fetch_and_save_article(
    crawler: AsyncWebCrawler,
    run_cfg: CrawlerRunConfig,
    url: str,
    index: int,
) -> bool:
    """Crawls a single article and saves its text content.

    Args:
        crawler: An already-started ``AsyncWebCrawler`` instance.
        run_cfg: Shared crawler run configuration.
        url: Article URL to crawl.
        index: 1-based article number (for logging).

    Returns:
        ``True`` if the article was saved successfully, ``False`` otherwise.
    """
    slug: str = _slug_from_url(url)
    dest: Path = OUTPUT_DIR / f"{slug}.txt"

    if dest.exists():
        logger.info("[%d] Already exists, skipping: %s", index, dest.name)
        return True

    try:
        result = await crawler.arun(url=url, config=run_cfg)
    except Exception as exc:
        logger.error("[%d] Failed to crawl %s: %s", index, url, exc)
        return False

    if not result.success:
        logger.warning("[%d] Non-success response for %s", index, url)
        return False

    text: str = result.markdown.strip()
    if not text:
        logger.warning("[%d] Empty content for %s", index, url)
        return False

    try:
        dest.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.error("[%d] Could not write %s: %s", index, dest, exc)
        return False

    logger.info("[%d] Saved %s (%d chars).", index, dest.name, len(text))
    return True


async def main() -> None:
    """Entry point: collects article URLs and scrapes each one."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        article_urls: list[str] = await _fetch_listing_pages(crawler, run_cfg)

        if not article_urls:
            logger.warning("No article URLs collected. Exiting.")
            return

        saved: int = 0
        for idx, url in enumerate(article_urls, start=1):
            ok: bool = await _fetch_and_save_article(crawler, run_cfg, url, idx)
            if ok:
                saved += 1
            await asyncio.sleep(DELAY_SECONDS)

        logger.info(
            "Done. Saved %d / %d articles to %s.",
            saved,
            len(article_urls),
            OUTPUT_DIR,
        )


if __name__ == "__main__":
    asyncio.run(main())

