"""Dense (semantic) retriever — wraps FAISSStore.

Returns (Document, cosine_score) pairs ranked highest-first.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from config import DEFAULT_TOP_K
from vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic retriever backed by a FAISSStore (or any store with similarity_search)."""

    def __init__(self, store: FAISSStore) -> None:
        """
        Args:
            store: A built (or loaded) FAISSStore instance.
        """
        self.store = store

    def retrieve(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """Return top-*k* semantically similar Documents.

        Args:
            query: Natural-language query.
            k:     Number of results.

        Returns:
            List of (Document, cosine_score) sorted descending by score.
        """
        results = self.store.similarity_search(query, k=k)
        logger.debug("Dense retrieval: %d results for query=%r", len(results), query[:60])
        return results
