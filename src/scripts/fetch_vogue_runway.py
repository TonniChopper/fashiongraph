"""Download runway images for selected designers from Vogue Runway.

Uses the helper functions in ``src.scripts.vogue`` to:

1. Discover every show for each designer.
2. Download all runway images to ``data/raw/vogue_runway/{designer}/{show}/``.
3. Write a unified ``metadata.csv`` with columns:
   ``designer, show, look_index, image_url, local_path``.

A 1-second delay is added between HTTP requests to be polite.

Usage::

    python -m src.scripts.fetch_vogue_runway
"""

import asyncio
import csv
import json
import logging
import random
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from unidecode import unidecode

from src.scripts.vogue import designer_to_shows, extract_json_from_script

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DESIGNERS: list[str] = [
    "Alexander McQueen",
    "Rick Owens",
    "Gucci",
    "Prada",
    "Bottega Veneta",
    "Balenciaga",
    "Celine",
    "Loewe",
    "Acne Studios",
    "Marni",
    "Jacquemus",
    "The Row",
]

OUTPUT_DIR: Path = Path("data/raw/vogue_runway")
METADATA_CSV: Path = OUTPUT_DIR / "metadata.csv"
MAX_SHOWS: int = 4
RECENT_SHOWS: int = 3
OLD_TAIL_RATIO: float = 0.30
DELAY_SECONDS: float = 1.0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _slugify_designer(name: str) -> str:
    """Converts a designer display name to the Vogue URL slug.

    Args:
        name: Human-readable designer name (e.g. ``"Bottega Veneta"``).

    Returns:
        URL-safe slug matching Vogue's conventions.
    """
    slug: str = name.replace(" ", "-").replace(".", "-")
    slug = slug.replace("&", "").replace("+", "")
    slug = slug.replace("--", "-").lower()
    return unidecode(slug)


def _slugify_show(name: str) -> str:
    """Converts a show display name to the Vogue URL slug.

    Args:
        name: Human-readable show title (e.g. ``"Fall 2024 Ready-to-Wear"``).

    Returns:
        URL-safe slug matching Vogue's conventions.
    """
    return unidecode(name.replace(" ", "-").lower())


def _fetch_show_images(
    designer: str,
    show: str,
) -> list[dict[str, str]]:
    """Fetches the image gallery for a single designer × show.

    Args:
        designer: Designer display name.
        show: Show display name.

    Returns:
        List of dicts with keys ``image_url``, ``designer``, ``show``.
        Empty list on failure.
    """
    from bs4 import BeautifulSoup  # local import to keep top-level light

    show_slug: str = _slugify_show(show)
    designer_slug: str = _slugify_designer(designer)
    url: str = f"https://www.vogue.com/fashion-shows/{show_slug}/{designer_slug}"

    try:
        resp: requests.Response = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch show page %s: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.content, "html5lib")
    data = extract_json_from_script(
        soup.find_all("script", type="text/javascript"),
        "runwayShowGalleries",
    )
    if not data:
        logger.warning("No gallery JSON found for %s — %s", designer, show)
        return []

    try:
        items = data["transformed"]["runwayShowGalleries"]["galleries"][0]["items"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Could not parse gallery items for %s — %s: %s",
            designer, show, exc,
        )
        return []

    results: list[dict[str, str]] = []
    for item in items:
        try:
            img_url: str = item["image"]["sources"]["md"]["url"]
            results.append({
                "designer": designer,
                "show": show,
                "image_url": img_url,
            })
        except (KeyError, TypeError):
            continue

    return results


def _download_image(
    image_url: str,
    dest: Path,
) -> bool:
    """Downloads a single image and saves it as PNG.

    Args:
        image_url: Remote image URL.
        dest: Local file path for the saved image.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    try:
        resp: requests.Response = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        img: Image.Image = Image.open(BytesIO(resp.content))
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(dest))
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", image_url, exc)
        return False


def _parse_season_and_type(show: str) -> tuple[str, str]:
    """Splits a show title into season and collection type.

    Examples::

        "Fall 2026 Ready-to-Wear"  → ("Fall 2026", "Ready-to-Wear")
        "Spring 2025 Menswear"     → ("Spring 2025", "Menswear")
        "Resort 2024"              → ("Resort 2024", "")

    Args:
        show: Human-readable show title from Vogue.

    Returns:
        A ``(season, collection_type)`` tuple.  If the title cannot be
        parsed, *season* is the full title and *collection_type* is empty.
    """
    known_types: list[str] = [
        "Ready-to-Wear",
        "Menswear",
        "Couture",
        "Pre-Fall",
        "Resort",
    ]
    for ctype in known_types:
        if ctype.lower() in show.lower():
            season: str = show.lower().replace(ctype.lower(), "").strip()
            # Capitalise each word back ("fall 2026" → "Fall 2026")
            season = " ".join(w.capitalize() for w in season.split())
            return season, ctype
    return show, ""


def _write_image_sidecar(
    dest: Path,
    designer: str,
    show: str,
    season: str,
    collection_type: str,
) -> None:
    """Writes a JSON sidecar file next to a downloaded image.

    The sidecar has the same stem as the image but a ``.json`` extension.

    Args:
        dest: Path to the image file (e.g. ``…/img_001.png``).
        designer: Designer display name.
        show: Show display name.
        season: Parsed season string (e.g. ``"Fall 2026"``).
        collection_type: Parsed collection type (e.g. ``"Ready-to-Wear"``).
    """
    sidecar: Path = dest.with_suffix(".json")
    payload: dict[str, str] = {
        "designer": designer,
        "show": show,
        "season": season,
        "type": collection_type,
        "image_path": str(dest),
    }
    try:
        sidecar.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error("Failed to write sidecar %s: %s", sidecar, exc)


def _write_show_info(
    show_dir: Path,
    designer: str,
    show: str,
    season: str,
    collection_type: str,
    num_looks: int,
) -> None:
    """Writes a ``show_info.json`` file into the show directory.

    Args:
        show_dir: Path to the show folder.
        designer: Designer display name.
        show: Show display name.
        season: Parsed season string.
        collection_type: Parsed collection type.
        num_looks: Total number of look images in this show.
    """
    info_path: Path = show_dir / "show_info.json"
    payload: dict[str, str | int] = {
        "designer": designer,
        "show": show,
        "season": season,
        "type": collection_type,
        "num_looks": num_looks,
        "show_dir": str(show_dir),
    }
    try:
        show_dir.mkdir(parents=True, exist_ok=True)
        info_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("    📄 show_info.json written (%d looks).", num_looks)
    except OSError as exc:
        logger.error("Failed to write show_info.json in %s: %s", show_dir, exc)


def _select_shows(
    shows: list[str],
    recent: int = RECENT_SHOWS,
    tail_ratio: float = OLD_TAIL_RATIO,
) -> list[str]:
    """Picks the *recent* newest shows plus one random old show.

    Vogue returns shows newest-first.  The "old tail" is defined as the
    last ``tail_ratio`` fraction of the full list (i.e. the oldest 30 %
    of shows — typically pre-2018 for long-running designers).

    If the list is too short for the split, all shows are returned.

    Args:
        shows: Full show list, newest-first.
        recent: Number of most-recent shows to keep.
        tail_ratio: Fraction of the list that counts as "old tail".

    Returns:
        Deduplicated list of selected show names (order preserved).
    """
    if len(shows) <= recent:
        return list(shows)

    selected: list[str] = list(shows[:recent])

    tail_start: int = len(shows) - max(1, int(len(shows) * tail_ratio))
    old_pool: list[str] = shows[tail_start:]
    # Exclude any show already in the recent slice
    old_pool = [s for s in old_pool if s not in selected]

    if old_pool:
        old_pick: str = random.choice(old_pool)
        selected.append(old_pick)
        logger.info(
            "    Old show picked from tail [%d:] → %s", tail_start, old_pick,
        )
    else:
        logger.info("    No old shows available outside the recent slice.")

    return selected


# ------------------------------------------------------------------
# Core pipeline (sync, called from async wrapper)
# ------------------------------------------------------------------


def _process_designer(
    designer: str,
    metadata_rows: list[list[str]],
) -> int:
    """Processes selected shows for a single designer.

    Discovers all shows, selects the 3 most recent plus 1 random old
    show from the last 30 % of the catalogue, downloads images, and
    appends rows to *metadata_rows* (mutated in place).

    Args:
        designer: Designer display name.
        metadata_rows: Accumulator list; each element is
            ``[designer, show, look_index, image_url, local_path]``.

    Returns:
        Number of images successfully downloaded for this designer.
    """
    logger.info("🔍 Discovering shows for %s …", designer)
    try:
        all_shows: list[str] = designer_to_shows(designer)
    except Exception as exc:
        logger.error("Failed to fetch show list for %s: %s", designer, exc)
        return 0

    if not all_shows:
        logger.warning("No shows found for %s.", designer)
        return 0

    shows: list[str] = _select_shows(all_shows)

    logger.info(
        "Selected %d / %d show(s) for %s: %s",
        len(shows), len(all_shows), designer, shows,
    )

    designer_slug: str = _slugify_designer(designer)
    downloaded: int = 0

    for show in shows:
        show_slug: str = _slugify_show(show)
        show_dir: Path = OUTPUT_DIR / designer_slug / show_slug
        logger.info("  📸 %s — %s", designer, show)

        season, collection_type = _parse_season_and_type(show)

        time.sleep(DELAY_SECONDS)

        images: list[dict[str, str]] = _fetch_show_images(designer, show)
        if not images:
            logger.info("    No images for this show.")
            continue

        for idx, img_info in enumerate(images):
            filename: str = f"{designer_slug}-{show_slug}-{idx}.png"
            dest: Path = show_dir / filename

            if dest.exists():
                logger.debug("    Already exists: %s", filename)
                local_path: str = str(dest)
            else:
                time.sleep(DELAY_SECONDS)
                ok: bool = _download_image(img_info["image_url"], dest)
                if ok:
                    downloaded += 1
                    local_path = str(dest)
                    logger.debug("    ✅ %s", filename)
                else:
                    local_path = ""

            # Write per-image JSON sidecar
            if local_path:
                _write_image_sidecar(
                    dest, designer, show, season, collection_type,
                )

            metadata_rows.append([
                designer,
                show,
                str(idx),
                img_info["image_url"],
                local_path,
            ])

        # Write per-show show_info.json
        _write_show_info(
            show_dir, designer, show, season, collection_type, len(images),
        )

    return downloaded


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


async def main() -> None:
    """Entry point: iterates over designers and downloads all runway images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[list[str]] = []
    total_downloaded: int = 0

    for designer in DESIGNERS:
        # Run the blocking I/O in a thread so the event loop stays alive
        count: int = await asyncio.to_thread(
            _process_designer, designer, metadata_rows
        )
        total_downloaded += count
        logger.info(
            "✅ %s done — %d images downloaded this designer.",
            designer, count,
        )

    # ------------------------------------------------------------------
    # Write unified metadata CSV
    # ------------------------------------------------------------------
    try:
        with METADATA_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "designer", "show", "look_index", "image_url", "local_path",
            ])
            writer.writerows(metadata_rows)
        logger.info(
            "📄 Metadata CSV written to %s (%d rows).",
            METADATA_CSV, len(metadata_rows),
        )
    except OSError as exc:
        logger.error("Failed to write metadata CSV: %s", exc)

    logger.info(
        "🏁 All done. %d images downloaded across %d designers.",
        total_downloaded, len(DESIGNERS),
    )


if __name__ == "__main__":
    asyncio.run(main())

