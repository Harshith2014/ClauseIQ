"""Qdrant vector store — upgrade path stub.

To activate:
    1. pip install qdrant-client langchain-qdrant
    2. Set QDRANT_URL and QDRANT_API_KEY in .env
    3. Implement the methods below following the FAISSStore interface.

All methods raise NotImplementedError until implemented, so any code that
depends on this class will fail loudly rather than silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document


class QdrantStore:
    """Qdrant vector store with the same interface as FAISSStore.

    Drop-in replacement: swap ``FAISSStore(embedder)`` for ``QdrantStore(embedder)``
    in retrieval/dense_retriever.py and everything downstream stays the same.
    """

    def __init__(self, embedder, collection_name: str = "legal-rag") -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        # TODO: initialise qdrant_client.QdrantClient here

    def build(self, chunks: List[Document]) -> None:
        """Upsert *chunks* into the Qdrant collection."""
        raise NotImplementedError(
            "QdrantStore.build() not yet implemented. "
            "See vectorstore/qdrant_store.py for instructions."
        )

    def save(self, save_dir: Path | None = None) -> None:
        """No-op for Qdrant (data lives in the server). Kept for interface parity."""
        raise NotImplementedError("QdrantStore.save() not applicable — data persists in Qdrant server.")

    def load(self, save_dir: Path | None = None) -> None:
        """Re-connect to an existing Qdrant collection."""
        raise NotImplementedError(
            "QdrantStore.load() not yet implemented. "
            "Connect to the Qdrant server and specify the collection name."
        )

    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Return top-*k* results from Qdrant."""
        raise NotImplementedError("QdrantStore.similarity_search() not yet implemented.")

    @property
    def size(self) -> int:
        raise NotImplementedError("QdrantStore.size not yet implemented.")
