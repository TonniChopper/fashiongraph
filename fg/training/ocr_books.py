"""OCR scanned fashion PDFs → text, so build_corpus can pick them up.

Some books are image-only scans (no text layer) — pypdf returns nothing. This
renders each page with PyMuPDF and runs Tesseract, writing cleaned ``.txt`` next
to a chosen output dir. Those ``.txt`` then feed ``build_corpus.py`` like any
other source.

    python -m fg.training.ocr_books --source data/raw/fashion_books \
        --out data/raw/fashion_books_ocr --max-pages 400
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

from fg.training.build_corpus import clean_page

logger: logging.Logger = logging.getLogger(__name__)


def is_scanned(pdf_path: Path, probe_pages: int = 5) -> bool:
    """True if the first pages carry essentially no extractable text (a scan)."""
    import fitz

    d = fitz.open(str(pdf_path))
    chars = sum(len(d[i].get_text()) for i in range(min(probe_pages, len(d))))
    d.close()
    return chars < 100


def ocr_pdf(pdf_path: Path, out_path: Path, dpi: int = 150,
            max_pages: int | None = None, lang: str = "eng") -> int:
    """OCRs a PDF page-by-page → cleaned text file. Returns character count."""
    import fitz
    import pytesseract
    from PIL import Image

    d = fitz.open(str(pdf_path))
    n = len(d) if max_pages is None else min(max_pages, len(d))
    parts, t0 = [], time.time()
    for i in range(n):
        pix = d[i].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        parts.append(clean_page(pytesseract.image_to_string(img, lang=lang)))
        if (i + 1) % 50 == 0:
            logger.info("  %s: %d/%d pages (%.0fs)", pdf_path.stem, i + 1, n, time.time() - t0)
    d.close()
    text = "\n".join(p for p in parts if p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("%s → %s (%d chars, %.0fs)", pdf_path.name, out_path.name,
                len(text), time.time() - t0)
    return len(text)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="OCR scanned fashion PDFs → text.")
    p.add_argument("--source", default="data/raw/fashion_books")
    p.add_argument("--out", default="data/raw/fashion_books_ocr")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--only", nargs="*", help="OCR only files whose name contains these substrings")
    p.add_argument("--skip", nargs="*", default=[], help="Skip files whose name contains these")
    args = p.parse_args()

    src, out = Path(args.source), Path(args.out)
    pdfs = sorted(src.glob("*.pdf"))
    for pdf in pdfs:
        name = pdf.name.lower()
        if args.only and not any(s.lower() in name for s in args.only):
            continue
        if any(s.lower() in name for s in args.skip):
            continue
        out_txt = out / (pdf.stem + ".txt")
        if out_txt.exists():
            logger.info("skip (exists): %s", out_txt.name)
            continue
        if not is_scanned(pdf):
            logger.info("skip (has text): %s", pdf.name)
            continue
        ocr_pdf(pdf, out_txt, dpi=args.dpi, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
