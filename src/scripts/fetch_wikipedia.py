"""Fetch fashion-related Wikipedia articles and save as .txt files.

Downloads articles about fashion houses, designers, decades in fashion,
fashion movements, and key concepts.  Each article is saved as a plain
text file to ``data/raw/wikipedia/``.

Usage::

    python -m src.scripts.fetch_wikipedia
    python -m src.scripts.fetch_wikipedia --output data/raw/wikipedia
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import wikipediaapi

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Article lists — curated for maximum RAG value
# ------------------------------------------------------------------

# Major fashion houses & luxury brands
FASHION_HOUSES: list[str] = [
    "Chanel", "Dior", "Louis Vuitton", "Gucci", "Prada",
    "Balenciaga", "Givenchy", "Hermès", "Valentino (fashion house)",
    "Versace", "Fendi", "Burberry", "Celine (brand)",
    "Yves Saint Laurent (brand)", "Alexander McQueen (brand)",
    "Bottega Veneta", "Loewe (company)", "Maison Margiela",
    "Rick Owens", "Comme des Garçons", "Issey Miyake",
    "Vivienne Westwood", "Jean-Paul Gaultier", "Balmain",
    "Miu Miu", "Ferragamo", "Lanvin", "Schiaparelli (fashion house)",
    "Mugler (fashion brand)", "Acne Studios", "Jacquemus",
    "The Row (fashion label)", "Jil Sander", "Marni (fashion brand)",
    "Sacai (fashion brand)", "Kenzo (brand)",
    "Off-White (brand)", "Fear of God (clothing brand)",
]

# Iconic designers (personal articles)
DESIGNERS: list[str] = [
    "Coco Chanel", "Christian Dior", "Cristóbal Balenciaga",
    "Yves Saint Laurent (designer)", "Karl Lagerfeld",
    "Alexander McQueen", "Rei Kawakubo", "Miuccia Prada",
    "Tom Ford", "Marc Jacobs", "Phoebe Philo", "Demna",
    "Virgil Abloh", "Raf Simons", "Hedi Slimane",
    "Martin Margiela", "Ann Demeulemeester",
    "Azzedine Alaïa", "Hubert de Givenchy",
    "Manolo Blahnik", "Christian Louboutin",
    "Pierpaolo Piccioli", "Daniel Lee (fashion designer)",
    "Jonathan Anderson (designer)", "Alessandro Michele",
]

# Decades in fashion
DECADES: list[str] = [
    "1920s in Western fashion", "1930s in Western fashion",
    "1940s in Western fashion", "1950s in Western fashion",
    "1960s in Western fashion", "1970s in Western fashion",
    "1980s in Western fashion", "1990s in fashion",
    "2000s in fashion", "2010s in fashion", "2020s in fashion",
]

# Fashion concepts, movements, terminology
CONCEPTS: list[str] = [
    "Fashion design", "Haute couture", "Prêt-à-porter",
    "Fast fashion", "Sustainable fashion", "Quiet luxury",
    "Streetwear", "Avant-garde fashion",
    "Fashion week", "Milan Fashion Week", "Paris Fashion Week",
    "London Fashion Week", "New York Fashion Week",
    "Fashion journalism", "Fashion photography",
    "Costume Institute", "Met Gala",
    "LVMH", "Kering", "Richemont",
    "History of fashion design",
    "Little black dress", "Trench coat", "Sneaker collecting",
    "Normcore", "Athleisure", "Minimalism (fashion)",
    "Deconstructionism (fashion)",
]

# Textiles & materials
TEXTILES: list[str] = [
    "Silk", "Cashmere wool", "Denim", "Leather",
    "Tweed (cloth)", "Chiffon (fabric)", "Organza",
]

ALL_ARTICLES: list[str] = (
    FASHION_HOUSES + DESIGNERS + DECADES + CONCEPTS + TEXTILES
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _slugify(title: str) -> str:
    """Converts a Wikipedia title to a safe filename slug.

    Args:
        title: Wikipedia article title.

    Returns:
        Lowercase hyphenated slug.
    """
    slug: str = title.lower()
    slug = slug.replace(" ", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _clean_wikipedia_text(text: str) -> str:
    """Light cleanup of Wikipedia article text.

    Removes reference markers, empty sections, and excessive whitespace
    while preserving paragraph structure.

    Args:
        text: Raw Wikipedia article text.

    Returns:
        Cleaned text.
    """
    # Remove == References == and everything after
    for marker in ("== References ==", "== External links ==",
                    "== See also ==", "== Further reading ==",
                    "== Notes ==", "== Bibliography =="):
        idx: int = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # Remove empty section headers
    text = re.sub(r"^==+\s*==+\s*$", "", text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def fetch_wikipedia(
    output_dir: Path,
    articles: list[str] | None = None,
    delay: float = 0.3,
) -> int:
    """Fetches Wikipedia articles and saves them as .txt files.

    Args:
        output_dir: Directory to save .txt files.
        articles: List of Wikipedia article titles. Defaults to
            ``ALL_ARTICLES``.
        delay: Seconds to wait between API requests.

    Returns:
        Number of articles successfully saved.
    """
    if articles is None:
        articles = ALL_ARTICLES

    output_dir.mkdir(parents=True, exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        user_agent="FashionGraph/1.0 (fashion AI research project)",
        language="en",
    )

    saved: int = 0
    skipped: int = 0

    for idx, title in enumerate(articles):
        slug: str = _slugify(title)
        out_path: Path = output_dir / f"{slug}.txt"

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 500:
            logger.debug("Already exists: %s", out_path.name)
            skipped += 1
            continue

        page = wiki.page(title)

        if not page.exists():
            logger.warning("[%d/%d] ✗ Not found: %s", idx + 1, len(articles), title)
            time.sleep(delay)
            continue

        text: str = _clean_wikipedia_text(page.text)
        word_count: int = len(text.split())

        if word_count < 100:
            logger.warning(
                "[%d/%d] ✗ Too short (%d words): %s",
                idx + 1, len(articles), word_count, title,
            )
            time.sleep(delay)
            continue

        # Write with a header for provenance
        header: str = (
            f"TITLE: {page.title}\n"
            f"SOURCE: https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}\n"
            f"TYPE: wikipedia\n"
            f"\n"
        )
        out_path.write_text(header + text, encoding="utf-8")

        saved += 1
        logger.info(
            "[%d/%d] ✅ %s → %s (%d words)",
            idx + 1, len(articles), page.title, out_path.name, word_count,
        )

        time.sleep(delay)

    logger.info(
        "🏁 Done. Saved %d articles, skipped %d existing, %d not found/too short. "
        "Output: %s",
        saved, skipped, len(articles) - saved - skipped, output_dir,
    )
    return saved


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch fashion Wikipedia articles",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/raw/wikipedia"),
        help="Output directory (default: data/raw/wikipedia)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between requests in seconds (default: 0.3)",
    )
    args = parser.parse_args()

    fetch_wikipedia(output_dir=args.output, delay=args.delay)


if __name__ == "__main__":
    main()

