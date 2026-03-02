"""Generate expert Q&A training pairs from scraped fashion articles.

Rule-based version — requires NO external API.  For each ``.txt`` file
under ``data/raw/``, the script cleans boilerplate, splits the text into
sentences, and produces 3 Q&A pairs using fixed templates.

Output is written to ``data/training/expert_pairs.jsonl`` (one JSON
object per line).

Usage::

    python -m src.scripts.generate_qa_pairs
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RAW_DIR: Path = Path("data/raw")
OUTPUT_DIR: Path = Path("data/training")
OUTPUT_FILE: Path = OUTPUT_DIR / "expert_pairs.jsonl"
LOG_EVERY: int = 10

NOISE_PATTERNS: list[str] = [
    "http",
    "Escape the algorithm",
    "SIGN UP",
    "policy",
    "cookie",
    "Privacy",
    "READ MORE",
    "copied",
    "mailto",
    "MediaAnother",
    "SSENSE MONTRAL",
]

MIN_LINE_LENGTH: int = 40

TEMPLATES: list[tuple[str, Callable[[list[str]], str]]] = [
    (
        "What fashion brands or designers are featured in this article?",
        lambda sents: " ".join(sents[0:3]),
    ),
    (
        "What specific style trends or clothing items are described?",
        lambda sents: " ".join(sents[2:5]),
    ),
    (
        "What is the cultural or industry significance discussed?",
        lambda sents: " ".join(sents[-4:-1]),
    ),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Removes boilerplate noise from scraped article text.

    Filtering rules applied line-by-line:

    * Lines containing any pattern from ``NOISE_PATTERNS`` are dropped.
    * Lines shorter than ``MIN_LINE_LENGTH`` characters are dropped.
    * Extra whitespace is collapsed.

    Args:
        text: Raw article text.

    Returns:
        Cleaned text with noisy / short lines removed.
    """
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped: str = line.strip()
        if not stripped:
            continue
        if len(stripped) < MIN_LINE_LENGTH:
            continue
        if any(pat in stripped for pat in NOISE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)


def _split_sentences(text: str) -> list[str]:
    """Splits text into sentences on ``. `` boundaries.

    Filters out empty / whitespace-only fragments.

    Args:
        text: Raw article text.

    Returns:
        List of non-empty sentence strings.
    """
    return [s.strip() for s in text.split(". ") if s.strip()]


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def main() -> None:
    """Entry point: reads articles, generates rule-based Q&A, writes JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files: list[Path] = sorted(RAW_DIR.rglob("*.txt"))
    logger.info("Found %d .txt files under %s.", len(txt_files), RAW_DIR)

    if not txt_files:
        logger.warning("No .txt files found. Exiting.")
        return

    total_pairs: int = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for idx, path in enumerate(txt_files, start=1):
            try:
                raw_text: str = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.error("Cannot read %s: %s", path, exc)
                continue

            if not raw_text.strip():
                continue

            text: str = clean_text(raw_text)
            if not text:
                continue

            # Skip files with too little useful content
            word_count: int = len(text.split())
            if word_count < 80:
                logger.info(
                    "Skipping %s — too short after cleaning (%d words)",
                    path.name, word_count,
                )
                continue

            sents: list[str] = _split_sentences(text)
            if len(sents) < 5:
                logger.debug("Skipping %s — fewer than 5 sentences.", path.name)
                continue

            source: str = path.name

            for question, answer_fn in TEMPLATES:
                answer: str = answer_fn(sents)
                if not answer.strip():
                    continue
                record: dict[str, str] = {
                    "question": question,
                    "answer": answer,
                    "source": source,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1

            if idx % LOG_EVERY == 0:
                logger.info(
                    "Progress: %d / %d files processed (%d pairs so far).",
                    idx, len(txt_files), total_pairs,
                )

    logger.info(
        "✅ Done. %d total pairs saved to %s from %d files.",
        total_pairs, OUTPUT_FILE, len(txt_files),
    )


if __name__ == "__main__":
    main()

