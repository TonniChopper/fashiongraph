"""Extract each book (PDF/md) → a cleaned ``.txt`` sibling, once.

`kg build` and `data build` should read *text*, not re-parse PDFs on every run
(slow, and fragile if a PDF is malformed). This writes one ``<book>.txt`` next to
each source file, so the ``fashion_books`` source can then read plain text and
never touch a PDF at build time.

    python -m fg.training.extract_books                 # default data/raw/fashion_books
    python -m fg.training.extract_books --source <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fg.config import settings
from fg.training.build_corpus import read_document

logger: logging.Logger = logging.getLogger(__name__)


def extract_to_txt(source_dir: str | Path | None = None, overwrite: bool = False) -> dict:
    """Writes a cleaned ``.txt`` beside every PDF/md book that lacks one.

    Args:
        source_dir: Book folder (default ``data/raw/fashion_books``).
        overwrite: Re-extract even if the ``.txt`` already exists.

    Returns:
        Stats dict (also logged).
    """
    src = Path(source_dir) if source_dir else settings.data_dir / "raw" / "fashion_books"
    files = sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in {".pdf", ".md"} and p.is_file()
        and "figures" not in p.parts          # skip extracted images folder
    )
    written, skipped, empty = 0, 0, 0
    for fp in files:
        out = fp.with_suffix(".txt")
        if out.exists() and not overwrite:
            skipped += 1
            continue
        try:
            text = read_document(fp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s (%s) — skipping.", fp.name, exc)
            continue
        if len(text.split()) < 50:            # image-only scan / stub → needs OCR
            empty += 1
            logger.info("No extractable text in %s (scan?) — run OCR (ocr_books).", fp.name)
            continue
        out.write_text(text, encoding="utf-8")
        written += 1
        logger.info("%s → %s (%d words)", fp.name, out.name, len(text.split()))
    stats = {"written": written, "skipped_existing": skipped, "empty_or_scan": empty,
             "dir": str(src)}
    logger.info("extract_books: %s", stats)
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Extract books (PDF/md) → .txt siblings.")
    p.add_argument("--source", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    import json
    print(json.dumps(extract_to_txt(args.source, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
