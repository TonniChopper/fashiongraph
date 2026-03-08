"""Index Vogue Runway metadata into ChromaDB."""

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.indexer import FashionKnowledgeIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

CSV_PATH: Path = Path("data/raw/vogue_runway/metadata.csv")


def main() -> None:
    """Reads Vogue Runway metadata CSV and indexes each look into ChromaDB."""
    if not CSV_PATH.exists():
        logger.error("Metadata CSV not found: %s", CSV_PATH)
        sys.exit(1)

    indexer = FashionKnowledgeIndexer(persist_dir="data/chroma")
    total: int = 0
    rows: int = 0

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text: str = (
                f"Designer: {row['designer']}. "
                f"Collection: {row['show']}. "
                f"Look {row['look_index']} from {row['show']} by {row['designer']}."
            )
            chunks: int = indexer.add_document(text, metadata={
                "source": "vogue_runway",
                "designer": row["designer"],
                "show": row["show"],
                "look_index": row["look_index"],
                "image_path": row["local_path"],
                "type": "runway_look",
            })
            total += chunks
            rows += 1

    logger.info("✅ Indexed %d chunks from %d runway looks.", total, rows)
    logger.info("ChromaDB collection size: %d", indexer.size)


if __name__ == "__main__":
    main()

