"""Auto-discover fashion article URLs from RSS feeds and sitemaps.

Fetches fresh article URLs from verified fashion sources (Hypebeast,
Dazed, WWD), validates each link, filters for fashion-relevant content,
and appends new URLs to ``data/urls.txt``.

Usage::

    python -m src.scripts.discover_urls
    python -m src.scripts.discover_urls --limit 50
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

URLS_FILE: Path = Path("data/urls.txt")
DELAY: float = 0.5
USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HEADERS: dict[str, str] = {"User-Agent": USER_AGENT}

# RSS / Atom feeds that reliably return article URLs
RSS_FEEDS: dict[str, str] = {
    # --- Streetwear / Hype ---
    "hypebeast": "https://hypebeast.com/feed",
    "highsnobiety": "https://www.highsnobiety.com/feed/",
    # --- Runway / Luxury ---
    "wwd": "https://wwd.com/feed/",
    "wwd_fashion": "https://wwd.com/fashion-news/feed/",
    "vogue": "https://www.vogue.com/feed/rss",
    "harpers_bazaar": "https://www.harpersbazaar.com/rss/all.xml/",
    "grazia": "https://graziamagazine.com/feed/",
    # --- Culture / Editorial ---
    "i_d": "https://i-d.co/feed/",
    "fashionista": "https://fashionista.com/.rss/full/",
    "fashion_gone_rogue": "https://www.fashiongonerogue.com/feed/",
    "coveteur": "https://coveteur.com/feed",
}

# Category pages to scrape for links (fallback for sites without RSS
# or where RSS mixes non-fashion content)
CATEGORY_PAGES: dict[str, str] = {
    # Main fashion sections
    "dazed_fashion": "https://www.dazeddigital.com/fashion",
    "hypebeast_fashion": "https://hypebeast.com/fashion",
    "vogue_fashion": "https://www.vogue.com/fashion",
    # WWD runway seasons (bridge articles: dense trends + brands + silhouettes)
    "wwd_fw26_paris": "https://wwd.com/runway/fall-2026/paris/",
    "wwd_fw26_milan": "https://wwd.com/runway/fall-2026/milan/",
    "wwd_fw26_london": "https://wwd.com/runway/fall-2026/london/",
    "wwd_fw26_nyc": "https://wwd.com/runway/fall-2026/new-york/",
    "wwd_ss26_paris": "https://wwd.com/runway/spring-2026/paris/",
    "wwd_ss26_milan": "https://wwd.com/runway/spring-2026/milan/",
    "wwd_fw25_paris": "https://wwd.com/runway/fall-2025/paris/",
    "wwd_fw25_milan": "https://wwd.com/runway/fall-2025/milan/",
    "wwd_ss25_paris": "https://wwd.com/runway/spring-2025/paris/",
    "wwd_ss25_milan": "https://wwd.com/runway/spring-2025/milan/",
}

# Only keep URLs that match these patterns (fashion-relevant)
FASHION_PATTERNS: list[re.Pattern] = [
    # Streetwear / Hype
    re.compile(r"hypebeast\.com/\d{4}/\d+/"),
    re.compile(r"highsnobiety\.com/p/"),
    # Runway / Luxury
    re.compile(r"wwd\.com/(runway|fashion-news|menswear-news|footwear-news)/"),
    re.compile(r"vogue\.com/(article|slideshow|fashion)/"),
    re.compile(r"harpersbazaar\.com/fashion/"),
    re.compile(r"graziamagazine\.com/articles/"),
    # Culture / Editorial
    re.compile(r"dazeddigital\.com/fashion/article/"),
    re.compile(r"i-d\.co/article/"),
    re.compile(r"fashionista\.com/\d{4}/"),
    re.compile(r"fashiongonerogue\.com/story/"),
    re.compile(r"coveteur\.com/[a-z]"),
    # One-off quality sources
    re.compile(r"fashionality\.nyc/"),
    re.compile(r"onclusive\.com/.+fashion"),
]

# Reject URLs matching these patterns (non-article pages)
REJECT_PATTERNS: list[re.Pattern] = [
    re.compile(r"/tag/"),
    re.compile(r"/tags/"),
    re.compile(r"/page/\d+"),
    re.compile(r"/author/"),
    re.compile(r"/category/"),
    re.compile(r"\.(jpg|png|gif|svg|css|js)$"),
    re.compile(r"_comments\.txt$"),
]

# Fashion keywords — at least one must appear in the URL slug
FASHION_KEYWORDS: set[str] = {
    "fashion", "runway", "collection", "wear", "spring", "fall",
    "winter", "summer", "fw", "ss", "aw", "rtw", "couture",
    "designer", "brand", "style", "lookbook", "campaign",
    "collaboration", "collab", "capsule", "sneaker", "shoe",
    "footwear", "apparel", "menswear", "womenswear", "streetwear",
    "luxury", "gucci", "prada", "balenciaga", "loewe", "fendi",
    "dior", "chanel", "bottega", "burberry", "givenchy", "hermes",
    "valentino", "versace", "celine", "ferragamo", "miu-miu",
    "off-white", "rick-owens", "yohji", "issey-miyake", "comme",
    "acne-studios", "jacquemus", "marni", "sacai", "kenzo",
    "nike", "adidas", "asics", "new-balance", "converse",
    "vans", "puma", "reebok", "jordan",
    "kith", "palace", "supreme", "stussy", "haven", "undercover",
    "dickies", "carhartt", "wtaps", "neighborhood",
    "milan-fashion-week", "paris-fashion-week", "london-fashion-week",
    "nyfw", "mfw", "pfw", "lfw",
    "trend", "outfit", "dress", "jacket", "coat", "pants", "knit",
    "denim", "leather", "silk", "wool", "tracktop", "tracksuit",
}

# Non-fashion slugs to reject outright (matched as exact word in slug)
REJECT_SLUGS: set[str] = {
    "anime", "manga", "gaming", "game", "trailer", "album",
    "stream", "movie", "film", "tv", "series", "episode",
    "nba", "nfl", "mlb", "ufc", "motorsport", "protro",
    "recipe", "restaurant", "food", "drink", "cocktail",
    "crypto", "bitcoin", "nft", "metaverse",
    "smartphone", "iphone", "samsung", "playstation", "xbox",
    "car", "automotive", "museum", "exhibit", "award", "winner",
    "album", "ep", "mixtape", "music", "song", "singer",
    "photography", "photographer", "hasselblad",
    "capcom", "resident", "fast", "furious",
}


# ------------------------------------------------------------------
# URL discovery methods
# ------------------------------------------------------------------


def _fetch_rss(feed_url: str) -> list[str]:
    """Parses an RSS/Atom feed and extracts article URLs.

    Args:
        feed_url: URL of the RSS or Atom feed.

    Returns:
        List of article URLs found in the feed.
    """
    try:
        resp = requests.get(feed_url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed for %s: %s", feed_url, exc)
        return []

    urls: list[str] = []

    # Standard RSS 2.0: <item><link>
    for link_el in root.findall(".//item/link"):
        if link_el.text:
            urls.append(link_el.text.strip())

    # Atom: <entry><link href="...">
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for link_el in root.findall(".//atom:entry/atom:link", ns):
        href: str | None = link_el.get("href")
        if href:
            urls.append(href.strip())

    logger.info("RSS %s → %d URLs", feed_url, len(urls))
    return urls


def _scrape_links_from_page(page_url: str) -> list[str]:
    """Extracts article links from a category/listing page.

    Uses trafilatura to download and regex to find article URLs.

    Args:
        page_url: URL of the listing page.

    Returns:
        List of article URLs found on the page.
    """
    try:
        resp = requests.get(page_url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Page fetch failed for %s: %s", page_url, exc)
        return []

    # Extract all href links
    hrefs: list[str] = re.findall(r'href=["\']([^"\']+)["\']', resp.text)

    urls: list[str] = []
    for href in hrefs:
        # Make absolute
        if href.startswith("/"):
            parsed = urlparse(page_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        if href.startswith("http"):
            urls.append(href)

    logger.info("Page %s → %d raw links", page_url, len(urls))
    return urls


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------


def _is_fashion_url(url: str) -> bool:
    """Checks if a URL is a fashion article worth scraping.

    For Dazed and WWD, the URL path structure already guarantees
    fashion relevance.  For Hypebeast (general feed), we additionally
    require at least one fashion keyword in the slug and reject known
    non-fashion topics.

    Args:
        url: URL to check.

    Returns:
        True if the URL looks like a fashion article.
    """
    # Hard reject patterns (tags, pagination, assets)
    if any(rp.search(url) for rp in REJECT_PATTERNS):
        return False

    # Must match at least one source pattern
    if not any(fp.search(url) for fp in FASHION_PATTERNS):
        return False

    # For Hypebeast & Highsnobiety URLs, apply keyword filtering
    # (their feeds mix fashion with music, gaming, food etc.)
    if "hypebeast.com" in url or "highsnobiety.com" in url:
        slug: str = url.rsplit("/", 1)[-1].lower()
        # Split slug into words on hyphens
        slug_words: set[str] = set(slug.split("-"))
        # Reject non-fashion topics
        if slug_words & REJECT_SLUGS:
            return False
        # Require at least one fashion keyword (exact word match)
        if not (slug_words & FASHION_KEYWORDS):
            return False

    # For Harper's Bazaar, only keep /fashion/ articles (skip celebrity gossip)
    if "harpersbazaar.com" in url and "/fashion/" not in url:
        return False

    # For Vogue, reject non-fashion content (celebrity, beauty, film, food)
    if "vogue.com" in url:
        slug = url.rsplit("/", 1)[-1].lower()
        vogue_reject = {
            "beauty", "interview", "wedding", "engagement", "ring",
            "recipe", "restaurant", "film", "movie", "book", "review",
            "broadway", "theater", "skincare", "hair", "moisture",
            "horoscope", "astrology", "wellness", "workout",
        }
        slug_words = set(slug.split("-"))
        if slug_words & vogue_reject:
            return False
        # Must have at least one fashion signal
        vogue_fashion = {
            "fashion", "runway", "collection", "backstage", "street",
            "style", "outfit", "trend", "wear", "show", "designer",
            "paris", "milan", "london", "nyfw", "couture", "look",
            "dress", "jacket", "coat", "spring", "fall", "winter",
        }
        if not (slug_words & vogue_fashion):
            return False

    return True


def _validate_url(url: str) -> bool:
    """Checks if a URL is reachable (HTTP 200).

    Args:
        url: URL to validate.

    Returns:
        True if the URL returns HTTP 200.
    """
    try:
        resp = requests.head(
            url, timeout=10, headers=HEADERS, allow_redirects=True,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def _load_existing_urls() -> set[str]:
    """Loads already-known URLs from data/urls.txt."""
    if not URLS_FILE.exists():
        return set()
    existing: set[str] = set()
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            existing.add(line)
    return existing


def main() -> None:
    """Discovers new fashion article URLs and appends them to urls.txt."""
    parser = argparse.ArgumentParser(description="Discover fashion article URLs")
    parser.add_argument("--limit", type=int, default=30, help="Max new URLs to add")
    parser.add_argument("--no-validate", action="store_true", help="Skip HTTP validation")
    args = parser.parse_args()

    existing: set[str] = _load_existing_urls()
    logger.info("Existing URLs in %s: %d", URLS_FILE, len(existing))

    # Collect candidate URLs from all sources
    candidates: list[str] = []

    # 1. RSS feeds
    for name, feed_url in RSS_FEEDS.items():
        logger.info("Fetching RSS: %s", name)
        candidates.extend(_fetch_rss(feed_url))
        time.sleep(DELAY)

    # 2. Category pages
    for name, page_url in CATEGORY_PAGES.items():
        logger.info("Scraping links: %s", name)
        candidates.extend(_scrape_links_from_page(page_url))
        time.sleep(DELAY)

    # Deduplicate and filter
    seen: set[str] = set()
    filtered: list[str] = []
    for url in candidates:
        url = url.split("?")[0].split("#")[0]  # strip query/fragment
        if url in seen or url in existing:
            continue
        seen.add(url)
        if _is_fashion_url(url):
            filtered.append(url)

    logger.info("Candidates after filtering: %d (from %d raw)", len(filtered), len(candidates))

    # Validate and collect
    new_urls: list[str] = []
    for idx, url in enumerate(filtered):
        if len(new_urls) >= args.limit:
            break

        if not args.no_validate:
            logger.info("[%d/%d] Validating: %s", idx + 1, len(filtered), url)
            if not _validate_url(url):
                logger.info("  ✗ Not reachable, skipping")
                time.sleep(DELAY)
                continue
            logger.info("  ✅ Valid")
            time.sleep(DELAY)
        else:
            logger.info("[%d/%d] Adding (no validation): %s", idx + 1, len(filtered), url)

        new_urls.append(url)

    if not new_urls:
        logger.info("No new URLs found.")
        return

    # Append to urls.txt
    URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with URLS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# --- Auto-discovered ({len(new_urls)} URLs) ---\n")
        for url in new_urls:
            fh.write(url + "\n")

    logger.info("✅ Added %d new URLs to %s", len(new_urls), URLS_FILE)
    for url in new_urls:
        logger.info("  + %s", url)


if __name__ == "__main__":
    main()

