"""Index expert_pairs.jsonl into ChromaDB via FashionKnowledgeIndexer."""

import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.indexer import FashionKnowledgeIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

PAIRS_PATH: Path = Path("data/training/expert_pairs.jsonl")
CHROMA_DIR: Path = Path("data/chroma")

SKIP_PHRASES: list[str] = [
    "Stay ahead of the curve",
    "Scan the QR code",
    "download the HYPE app",
    "Get must-see stories",
    "Expand your creative community",
    "Cookie Policy",
    "clicking Accept",
]


def main() -> None:
    """Reads expert Q&A pairs and indexes them into ChromaDB."""
    if not PAIRS_PATH.exists():
        logger.error("Pairs file not found: %s", PAIRS_PATH)
        sys.exit(1)

    # Clear existing ChromaDB for a clean re-index
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Cleared existing ChromaDB at %s", CHROMA_DIR)

    indexer = FashionKnowledgeIndexer(persist_dir=str(CHROMA_DIR))
    total: int = 0
    pairs: int = 0
    skipped: int = 0

    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair: dict = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line: %s", exc)
                continue

            # Skip boilerplate answers
            if any(phrase in pair["answer"] for phrase in SKIP_PHRASES):
                logger.debug("Skipping boilerplate pair from %s", pair.get("source"))
                skipped += 1
                continue

            text: str = f"Q: {pair['question']}\nA: {pair['answer']}"
            chunks: int = indexer.add_document(text, metadata={
                "source": pair.get("source", "unknown"),
                "type": "qa_pair",
            })
            total += chunks
            pairs += 1

    logger.info(
        "✅ Indexed %d chunks from %d pairs, skipped %d boilerplate (%s).",
        total, pairs, skipped, PAIRS_PATH.name,
    )
    logger.info("ChromaDB collection size: %d", indexer.size)


if __name__ == "__main__":
    main()

