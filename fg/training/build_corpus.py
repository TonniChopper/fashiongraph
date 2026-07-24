"""Phase 5 Track A — build the domain-text corpus (the 'voice' half).

Turns your fashion books/articles (PDF / txt / md) into cleaned, chunked text for
continued-pretraining-style LoRA, so the model picks up fashion *prose, register
and vocabulary*. Pairs with ``build_instruction_data.py`` (the behaviour half).

Drop your files here (only text you're entitled to train on — not for redistribution):

    data/raw/fashion_books/*.pdf | *.txt | *.md

then::

    python -m fg.training.build_corpus

Output is MLX-LM-ready text JSONL (``{"text": "…"}``) at
``data/processed/corpus/{train,valid}.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from fg.config import settings

logger: logging.Logger = logging.getLogger(__name__)

_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_WS_RE = re.compile(r"[ \t]+")


def clean_page(text: str) -> str:
    """Cleans one page's raw text: de-hyphenate, drop page-number lines, collapse ws."""
    text = text.replace("-\n", "")                 # join hyphenated line breaks
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or _PAGE_NUM_RE.match(s) or len(s) <= 2:
            continue
        lines.append(s)
    return _WS_RE.sub(" ", " ".join(lines)).strip()


def read_document(path: Path) -> str:
    """Extracts cleaned text from a PDF / txt / md file (empty string on failure)."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return clean_page(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Need pypdf: pip install pypdf") from exc
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", path.name, exc)
            return ""
        parts = []
        for page in reader.pages:
            try:
                parts.append(clean_page(page.extract_text() or ""))
            except Exception:  # noqa: BLE001
                continue
        return " ".join(p for p in parts if p)
    return ""


def chunk_words(text: str, size: int = 350, overlap: int = 40) -> list[str]:
    """Splits text into overlapping ~``size``-word windows (≈512 tokens)."""
    words = text.split()
    if len(words) <= size:
        return [text] if len(words) >= 30 else []
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) >= 30:
            out.append(chunk)
        if i + size >= len(words):
            break
    return out


def build(
    source_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    size: int = 350,
    overlap: int = 40,
    val_frac: float = 0.03,
    seed: int = 42,
) -> dict:
    """Reads every document under *source_dir* → chunked ``{"text"}`` JSONL.

    Returns a stats dict (also logged). Returns ``{"documents": 0}`` (and writes
    nothing) if the source folder is missing/empty — so it's safe to call before
    you've added any books.
    """
    import random

    src = Path(source_dir) if source_dir else settings.data_dir / "raw" / "fashion_books"
    if not src.exists():
        logger.warning("No corpus folder at %s — add your PDFs there.", src)
        return {"documents": 0, "chunks": 0}

    files = [p for p in sorted(src.rglob("*"))
             if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    chunks: list[str] = []
    for fp in files:
        txt = read_document(fp)
        c = chunk_words(txt, size, overlap)
        logger.info("%s → %d chunks", fp.name, len(c))
        chunks.extend(c)

    if not chunks:
        logger.warning("No text extracted from %d files.", len(files))
        return {"documents": len(files), "chunks": 0}

    rng = random.Random(seed)
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * val_frac))
    valid, train = chunks[:n_val], chunks[n_val:]

    out = Path(out_dir) if out_dir else settings.data_dir / "processed" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for ch in rows:
                fh.write(json.dumps({"text": ch}, ensure_ascii=False) + "\n")

    stats = {"documents": len(files), "chunks": len(chunks),
             "train": len(train), "valid": len(valid), "out_dir": str(out)}
    logger.info("Corpus: %s", stats)
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build the fashion domain-text corpus.")
    p.add_argument("--source", default=None, help="Folder of PDFs/txt/md")
    p.add_argument("--out", default=None)
    p.add_argument("--size", type=int, default=350)
    p.add_argument("--overlap", type=int, default=40)
    args = p.parse_args()
    print(json.dumps(build(source_dir=args.source, out_dir=args.out,
                           size=args.size, overlap=args.overlap), indent=2))


if __name__ == "__main__":
    main()
