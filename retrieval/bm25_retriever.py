"""BM25 sparse retriever using rank-bm25.

BM25Okapi is built in-memory from chunk texts at initialisation — fast and
deterministic, no serialisation needed.

Returns (Document, bm25_score) pairs ranked highest-first.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser."""
    return text.lower().split()


class BM25Retriever:
    """Sparse keyword retriever using BM25Okapi."""

    def __init__(self, chunks: List[Document]) -> None:
        """Build BM25 index from *chunks*.

        Args:
            chunks: The same chunk list used for the dense index (must match).
        """
        if not chunks:
            raise ValueError("chunks must be non-empty.")

        self._docs = list(chunks)
        tokenized_corpus = [_tokenize(doc.page_content) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d documents.", len(self._docs))

    def retrieve(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """Return top-*k* keyword-matched Documents.

        Args:
            query: Natural-language query string (tokenised internally).
            k:     Number of results to return.

        Returns:
            List of (Document, bm25_score) sorted descending by score.
            Documents with score 0 are still included if k exceeds non-zero matches.
        """
        tokens = _tokenize(query)
        scores: np.ndarray = self._bm25.get_scores(tokens)

        k = min(k, len(self._docs))
        top_indices = np.argsort(scores)[::-1][:k]

        results = [(self._docs[int(i)], float(scores[i])) for i in top_indices]
        logger.debug(
            "BM25 retrieval: top score=%.3f for query=%r",
            results[0][1] if results else 0,
            query[:60],
        )
        return results
