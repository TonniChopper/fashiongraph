"""Fashion knowledge base indexer for ChromaDB."""

import logging
from pathlib import Path

import chromadb, hashlib
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger: logging.Logger = logging.getLogger(__name__)


class FashionKnowledgeIndexer:
    """Indexes fashion documents into ChromaDB for RAG retrieval.

    Takes raw text documents (Vogue articles, brand descriptions,
    trend analyses, your expert annotations) and stores them as
    vector embeddings for semantic search.
    """

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="fashion_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " "]
        )
        logger.info("FashionKnowledgeIndexer ready, collection size: %d",
                    self.collection.count())

    def add_document(self, text: str, metadata: dict) -> int:
        """Splits text into chunks and adds to ChromaDB.

        Args:
            text: Raw document text (article, annotation, description).
            metadata: Dict with keys like source, year, category, brand.

        Returns:
            Number of chunks added.
        """
        chunks = self.splitter.split_text(text)
        ids = [
            hashlib.md5(f"{metadata.get('source', 'doc')}_{chunk[:50]}".encode()).hexdigest()
            for chunk in chunks
        ]
        self.collection.add(
            documents=chunks,
            metadatas=[metadata] * len(chunks),
            ids=ids
        )
        logger.info("Added %d chunks from source: %s", len(chunks), metadata.get("source"))
        return len(chunks)

    def add_expert_annotation(self, element: str, season: str,
                               year: int, context: str) -> None:
        """Adds your expert trend annotation directly to the knowledge base."""
        text = (f"Fashion trend analysis: {element} in {season} {year}. "
                f"Context: {context}")
        self.add_document(text, metadata={
            "source": "expert_annotation",
            "element": element,
            "year": str(year),
            "season": season,
            "type": "trend_score"
        })

    @property
    def size(self) -> int:
        return self.collection.count()
