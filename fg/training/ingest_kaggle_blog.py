"""Ingest the Kaggle 'articles-extracted-from-a-fashion-blog' dataset → corpus text.

The dataset is a nested JSON crawl (``root.page[].record``); each record hides a
``title`` and ``post_text_content`` under a *variable* domain/category key, so we
recurse to find them. Thin posts (mostly image captions/tags) are dropped. Output
is a single ``.txt`` written into the corpus source folder so ``build_corpus``
chunks it like any other book.

    python -m fg.training.ingest_kaggle_blog \
        --json data/raw/fashion_books/wordpress_com_generic_crawler.json \
        --out data/raw/fashion_books/kaggle_fashion_blog.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _find(rec: dict, key: str) -> str | None:
    """Depth-first search for the first string value under *key*."""
    stack = [rec]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and isinstance(v, str):
                    return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def extract(json_path: Path, min_words: int = 40) -> list[str]:
    """Returns cleaned 'title + body' blocks for posts with real text."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pages = data.get("root", {}).get("page", []) if isinstance(data, dict) else []
    out: list[str] = []
    for pg in pages:
        rec = pg.get("record", pg) if isinstance(pg, dict) else pg
        body = _find(rec, "post_text_content") or ""
        title = _find(rec, "title") or ""
        body = _WS.sub(" ", body).strip()
        if len(body.split()) < min_words:
            continue
        out.append((title.strip() + ". " + body) if title else body)
    return out


def build(json_path: str | Path, out_path: str | Path, min_words: int = 40) -> dict:
    """Writes the extracted blog posts to a text file for build_corpus."""
    posts = extract(Path(json_path), min_words=min_words)
    text = "\n\n".join(posts)
    Path(out_path).write_text(text, encoding="utf-8")
    stats = {"posts_kept": len(posts), "words": len(text.split()), "out": str(out_path)}
    logger.info("Kaggle blog: %s", stats)
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Ingest the Kaggle fashion-blog JSON.")
    p.add_argument("--json", default="data/raw/fashion_books/wordpress_com_generic_crawler.json")
    p.add_argument("--out", default="data/raw/fashion_books/kaggle_fashion_blog.txt")
    p.add_argument("--min-words", type=int, default=40)
    args = p.parse_args()
    print(json.dumps(build(args.json, args.out, args.min_words), indent=2))


if __name__ == "__main__":
    main()
