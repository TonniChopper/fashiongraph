"""Generate expert Q&A training pairs from scraped fashion articles.

Reads every ``.txt`` file under ``data/raw/`` (recursively), sends each
article to the OpenAI GPT-4o API asking for 3 expert Q&A pairs, and
writes them as JSONL to ``data/training/expert_pairs.jsonl``.

The ``OPENAI_API_KEY`` environment variable must be set (or present in a
``.env`` file in the project root).

Usage::

    python -m src.scripts.generate_qa_pairs
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

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
MODEL: str = "gpt-4o"
DELAY_SECONDS: float = 0.5
MAX_ARTICLE_CHARS: int = 12_000  # truncate very long articles to stay within context

SYSTEM_PROMPT: str = (
    "You are a senior fashion creative director. "
    "Based on the following article, generate 3 Q&A pairs where "
    "the question is something a fashion brand would ask "
    "and the answer is expert-level advice. "
    'Return ONLY a JSON array: [{"question": "...", "answer": "..."}]'
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _collect_txt_files(root: Path) -> list[Path]:
    """Recursively collects all ``.txt`` files under *root*.

    Args:
        root: Top-level directory to scan.

    Returns:
        Sorted list of ``.txt`` file paths.
    """
    files: list[Path] = sorted(root.rglob("*.txt"))
    logger.info("Found %d .txt files under %s.", len(files), root)
    return files


def _read_article(path: Path) -> str | None:
    """Reads and returns the text content of an article file.

    Returns ``None`` if the file is empty or unreadable.

    Args:
        path: Path to the ``.txt`` file.

    Returns:
        Article text (possibly truncated), or ``None`` on failure.
    """
    try:
        text: str = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return None

    if not text:
        logger.warning("Empty file, skipping: %s", path)
        return None

    if len(text) > MAX_ARTICLE_CHARS:
        logger.debug(
            "Truncating %s from %d to %d chars.",
            path.name, len(text), MAX_ARTICLE_CHARS,
        )
        text = text[:MAX_ARTICLE_CHARS]

    return text


def _call_openai(client: OpenAI, article_text: str) -> list[dict[str, str]]:
    """Sends an article to GPT-4o and parses the returned Q&A pairs.

    Args:
        client: Initialised OpenAI client.
        article_text: Plain-text article content.

    Returns:
        List of dicts with ``question`` and ``answer`` keys.

    Raises:
        ValueError: If the response cannot be parsed as the expected JSON.
        OpenAIError: On any API-level failure.
    """
    sys_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    usr_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": article_text,
    }
    messages: list[ChatCompletionMessageParam] = [sys_msg, usr_msg]
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=messages,
        response_format={"type": "json_object"},  # type: ignore[arg-type]
    )

    raw: str = response.choices[0].message.content or ""
    parsed = json.loads(raw)

    # The model may wrap the array in an object like {"pairs": [...]}
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
        else:
            raise ValueError(f"Unexpected JSON structure: {raw[:200]}")

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got: {type(parsed).__name__}")

    pairs: list[dict[str, str]] = []
    for item in parsed:
        q: str | None = item.get("question")
        a: str | None = item.get("answer")
        if q and a:
            pairs.append({"question": q.strip(), "answer": a.strip()})

    return pairs


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def main() -> None:
    """Entry point: reads articles, calls GPT-4o, writes JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files: list[Path] = _collect_txt_files(RAW_DIR)
    if not txt_files:
        logger.warning("No .txt files found under %s. Exiting.", RAW_DIR)
        return

    client = OpenAI()  # reads OPENAI_API_KEY from env

    total_pairs: int = 0
    processed: int = 0
    skipped: int = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for idx, path in enumerate(txt_files, start=1):
            rel: str = str(path.relative_to(RAW_DIR))
            logger.info("[%d/%d] Processing: %s", idx, len(txt_files), rel)

            article: str | None = _read_article(path)
            if article is None:
                skipped += 1
                continue

            try:
                pairs: list[dict[str, str]] = _call_openai(client, article)
            except (OpenAIError, ValueError, json.JSONDecodeError) as exc:
                logger.error(
                    "[%d/%d] API/parse error for %s: %s", idx, len(txt_files), rel, exc
                )
                skipped += 1
                time.sleep(DELAY_SECONDS)
                continue
            except Exception as exc:
                logger.error(
                    "[%d/%d] Unexpected error for %s: %s", idx, len(txt_files), rel, exc
                )
                skipped += 1
                time.sleep(DELAY_SECONDS)
                continue

            for pair in pairs:
                record: dict[str, str] = {
                    "instruction": pair["question"],
                    "output": pair["answer"],
                    "source_file": rel,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_pairs += len(pairs)
            processed += 1
            logger.info(
                "[%d/%d] Got %d pairs (running total: %d).",
                idx, len(txt_files), len(pairs), total_pairs,
            )

            time.sleep(DELAY_SECONDS)

    logger.info(
        "✅ Done. Processed %d files, skipped %d, wrote %d pairs to %s.",
        processed, skipped, total_pairs, OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

