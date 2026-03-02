"""Scrape fashion articles from Vogue, Dazed Digital, and SSENSE.

Fetches up to 50 article URLs per source, then crawls each article
page and saves title + body text as ``.txt`` files.

Output directories::

    data/raw/vogue_articles/   — vogue.com/fashion/culture
    data/raw/dazed/            — dazeddigital.com/fashion
    data/raw/ssense/           — ssense.com/en-us/editorial

Usage::

    python -m src.scripts.fetch_bof
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

MAX_ARTICLES: int = 50
DELAY_SECONDS: float = 1.0


# ------------------------------------------------------------------
# Source configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SourceConfig:
    """Describes a single scraping target.

    Attributes:
        name: Human-readable source label (used in logs).
        base_url: Listing / hub page URL.
        output_dir: Local directory for saved ``.txt`` files.
        link_pattern: Regex that matches article URLs in the page markdown.
        min_parts: Minimum number of ``/``-separated URL parts for a link
            to be considered a real article (filters out bare category URLs).
        page_param: Query-string key used for pagination (e.g. ``page``).
            If ``None``, page number is appended as a path segment instead.
        path_pagination: If ``True``, pagination is ``/page/N`` instead of
            ``?page_param=N``.
    """

    name: str
    base_url: str
    output_dir: Path
    link_pattern: str
    min_parts: int = 5
    page_param: str | None = "page"
    path_pagination: bool = False


SOURCES: list[SourceConfig] = [
    # --- Vogue Culture ---------------------------------------------------
    SourceConfig(
        name="Vogue Culture",
        base_url="https://www.vogue.com/fashion/culture",
        output_dir=Path("data/raw/vogue_articles"),
        link_pattern=(
            r"https?://(?:www\.)?vogue\.com/article/[^\s\)\]\"\'>]+"
        ),
        min_parts=5,
        page_param="page",
    ),
    # --- Dazed Digital Fashion -------------------------------------------
    SourceConfig(
        name="Dazed Digital",
        base_url="https://www.dazeddigital.com/fashion",
        output_dir=Path("data/raw/dazed"),
        link_pattern=(
            r"https?://(?:www\.)?dazeddigital\.com/fashion/"
            r"(?:article|gallery)/[^\s\)\]\"\'>]+"
        ),
        min_parts=5,
        page_param=None,
        path_pagination=True,
    ),
    # --- SSENSE Editorial ------------------------------------------------
    SourceConfig(
        name="SSENSE",
        base_url="https://www.ssense.com/en-us/editorial",
        output_dir=Path("data/raw/ssense"),
        link_pattern=(
            r"https?://(?:www\.)?ssense\.com/en-us/editorial/[^\s\)\]\"\'>]+"
        ),
        min_parts=6,
        page_param="page",
    ),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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


def _extract_links(markdown: str, source: SourceConfig) -> list[str]:
    """Extracts unique article links for a given source.

    Args:
        markdown: Raw markdown returned by Crawl4AI.
        source: Source configuration with the link regex and filters.

    Returns:
        Deduplicated list of article URLs.
    """
    raw_links: list[str] = re.findall(source.link_pattern, markdown)
    seen: set[str] = set()
    unique: list[str] = []
    for link in raw_links:
        clean: str = link.rstrip(".,;:!?)")
        parts: list[str] = clean.rstrip("/").split("/")
        if len(parts) < source.min_parts:
            continue
        if clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return unique


def _extract_title(markdown: str) -> str:
    """Extracts the article title from crawled markdown.

    Looks for the first ``# Heading`` (H1).  Falls back to the first
    non-empty line.

    Args:
        markdown: Raw markdown of the article page.

    Returns:
        Extracted title string.
    """
    h1_match: re.Match[str] | None = re.search(
        r"^#\s+(.+)$", markdown, re.MULTILINE
    )
    if h1_match:
        return h1_match.group(1).strip()

    for line in markdown.splitlines():
        stripped: str = line.strip()
        if stripped:
            return stripped[:200]
    return "Untitled"


def _extract_body(markdown: str) -> str:
    """Extracts the main body text from crawled markdown.

    Takes content after the first H1 heading, trims at common footer
    markers.

    Args:
        markdown: Raw markdown of the article page.

    Returns:
        Cleaned body text.
    """
    h1_match: re.Match[str] | None = re.search(
        r"^#\s+.+$", markdown, re.MULTILINE
    )
    body: str = markdown[h1_match.end() :] if h1_match else markdown

    footer_patterns: list[str] = [
        r"(?m)^#{1,3}\s+(?:Related|More Stories|Newsletter|Sign Up|Footer"
        r"|You Might Also Like|Trending|Read More)",
        r"(?m)^---+\s*$",
        r"©\s*\d{4}",
    ]
    for fp in footer_patterns:
        footer_hit: re.Match[str] | None = re.search(fp, body)
        if footer_hit:
            body = body[: footer_hit.start()]

    return body.strip()


def _listing_url(source: SourceConfig, page: int) -> str:
    """Builds the listing page URL for a given page number.

    Args:
        source: Source configuration.
        page: 1-based page index.

    Returns:
        Full URL string for the listing page.
    """
    if page == 1:
        return source.base_url
    if source.path_pagination:
        return f"{source.base_url.rstrip('/')}/page/{page}"
    param: str = source.page_param or "page"
    sep: str = "&" if "?" in source.base_url else "?"
    return f"{source.base_url}{sep}{param}={page}"


# ------------------------------------------------------------------
# Crawl logic
# ------------------------------------------------------------------


async def _fetch_listing_pages(
    crawler: AsyncWebCrawler,
    run_cfg: CrawlerRunConfig,
    source: SourceConfig,
) -> list[str]:
    """Crawls paginated listing pages for *source* until enough links.

    Args:
        crawler: An already-started ``AsyncWebCrawler`` instance.
        run_cfg: Shared crawler run configuration.
        source: Source configuration.

    Returns:
        List of article URLs (up to ``MAX_ARTICLES``).
    """
    all_links: list[str] = []
    page: int = 1

    while len(all_links) < MAX_ARTICLES:
        url: str = _listing_url(source, page)
        logger.info(
            "[%s] Crawling listing page %d: %s", source.name, page, url
        )

        try:
            result = await crawler.arun(url=url, config=run_cfg)
        except Exception as exc:
            logger.error(
                "[%s] Failed to crawl listing page %d: %s",
                source.name, page, exc,
            )
            break

        if not result.success:
            logger.warning(
                "[%s] Listing page %d returned failure (status=%s). "
                "Stopping pagination.",
                source.name,
                page,
                getattr(result, "status_code", "unknown"),
            )
            break

        new_links: list[str] = _extract_links(result.markdown, source)
        if not new_links:
            logger.info(
                "[%s] No more article links on page %d. Stopping.",
                source.name, page,
            )
            break

        all_links.extend(new_links)
        logger.info(
            "[%s] Page %d: found %d links (total so far: %d).",
            source.name, page, len(new_links), len(all_links),
        )

        page += 1
        await asyncio.sleep(DELAY_SECONDS)

    unique: list[str] = list(dict.fromkeys(all_links))[:MAX_ARTICLES]
    logger.info(
        "[%s] Collected %d unique article URLs.", source.name, len(unique)
    )
    return unique


async def _fetch_and_save_article(
    crawler: AsyncWebCrawler,
    run_cfg: CrawlerRunConfig,
    source: SourceConfig,
    url: str,
    index: int,
) -> bool:
    """Crawls a single article and saves title + body as plain text.

    Output format::

        TITLE: <article title>
        SOURCE: <source name>
        URL: <original url>

        <body text>

    Args:
        crawler: An already-started ``AsyncWebCrawler`` instance.
        run_cfg: Shared crawler run configuration.
        source: Source configuration (for output dir and logging).
        url: Article URL to crawl.
        index: 1-based article number (for logging).

    Returns:
        ``True`` if the article was saved successfully, ``False`` otherwise.
    """
    slug: str = _slug_from_url(url)
    dest: Path = source.output_dir / f"{slug}.txt"

    if dest.exists():
        logger.info(
            "[%s][%d] Already exists, skipping: %s",
            source.name, index, dest.name,
        )
        return True

    try:
        result = await crawler.arun(url=url, config=run_cfg)
    except Exception as exc:
        logger.error(
            "[%s][%d] Failed to crawl %s: %s", source.name, index, url, exc
        )
        return False

    if not result.success:
        logger.warning(
            "[%s][%d] Non-success response for %s", source.name, index, url
        )
        return False

    markdown: str = result.markdown.strip()
    if not markdown:
        logger.warning(
            "[%s][%d] Empty content for %s", source.name, index, url
        )
        return False

    title: str = _extract_title(markdown)
    body: str = _extract_body(markdown)

    if not body:
        logger.warning(
            "[%s][%d] No body text extracted for %s",
            source.name, index, url,
        )
        return False

    content: str = (
        f"TITLE: {title}\n"
        f"SOURCE: {source.name}\n"
        f"URL: {url}\n\n"
        f"{body}\n"
    )

    try:
        dest.write_text(content, encoding="utf-8")
    except OSError as exc:
        logger.error(
            "[%s][%d] Could not write %s: %s",
            source.name, index, dest, exc,
        )
        return False

    logger.info(
        "[%s][%d] Saved %s (%d chars).",
        source.name, index, dest.name, len(content),
    )
    return True


async def _scrape_source(
    crawler: AsyncWebCrawler,
    run_cfg: CrawlerRunConfig,
    source: SourceConfig,
) -> int:
    """Runs the full scrape pipeline for a single source.

    Args:
        crawler: An already-started ``AsyncWebCrawler`` instance.
        run_cfg: Shared crawler run configuration.
        source: Source configuration.

    Returns:
        Number of articles saved.
    """
    source.output_dir.mkdir(parents=True, exist_ok=True)

    article_urls: list[str] = await _fetch_listing_pages(
        crawler, run_cfg, source
    )
    if not article_urls:
        logger.warning("[%s] No article URLs collected.", source.name)
        return 0

    saved: int = 0
    for idx, url in enumerate(article_urls, start=1):
        ok: bool = await _fetch_and_save_article(
            crawler, run_cfg, source, url, idx
        )
        if ok:
            saved += 1
        await asyncio.sleep(DELAY_SECONDS)

    logger.info(
        "[%s] Done. Saved %d / %d articles to %s.",
        source.name, saved, len(article_urls), source.output_dir,
    )
    return saved


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


async def main() -> None:
    """Entry point: scrapes all configured sources sequentially."""
    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig()

    grand_total: int = 0

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for source in SOURCES:
            logger.info("=" * 60)
            logger.info("Starting source: %s (%s)", source.name, source.base_url)
            logger.info("=" * 60)
            saved: int = await _scrape_source(crawler, run_cfg, source)
            grand_total += saved

    logger.info(
        "🏁 All sources done. %d total articles saved across %d sources.",
        grand_total, len(SOURCES),
    )


if __name__ == "__main__":
    asyncio.run(main())

