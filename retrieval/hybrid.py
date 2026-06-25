"""Hybrid retriever — Reciprocal Rank Fusion of dense + BM25 results.

RRF formula:  score(d) = sum_i  1 / (k + rank_i(d))

where rank_i(d) is the 1-based position of document d in ranked list i, and
k (default 60) controls how steeply rank differences are penalised.

Higher RRF score = higher combined rank. The constant k=60 is the standard
default from the original RRF paper (Cormack et al., 2009).

Usage:
    dense_results  = dense_retriever.retrieve(query, k=10)
    bm25_results   = bm25_retriever.retrieve(query, k=10)
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config import RRF_K, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def _doc_id(doc: Document) -> str:
    """Stable unique identifier for a Document based on its metadata."""
    source = doc.metadata.get("source", "unknown")
    chunk_idx = doc.metadata.get("chunk_index", id(doc))
    return f"{source}::{chunk_idx}"


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = RRF_K,
    top_n: int | None = None,
) -> List[Tuple[Document, float]]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results:  (Document, score) list from DenseRetriever, score descending.
        bm25_results:   (Document, score) list from BM25Retriever, score descending.
        k:              RRF constant (default 60). Higher = smoother blending.
        top_n:          If set, truncate output to top *top_n* results.
                        Defaults to config.DEFAULT_TOP_K.

    Returns:
        Deduplicated (Document, rrf_score) list sorted by rrf_score descending.
    """
    if top_n is None:
        top_n = DEFAULT_TOP_K

    rrf_scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results, start=1):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
        docs_by_id[did] = doc

    for rank, (doc, _) in enumerate(bm25_results, start=1):
        did = _doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
        docs_by_id[did] = doc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    merged = [(docs_by_id[did], rrf_scores[did]) for did in sorted_ids]
    result = merged[:top_n]

    logger.debug(
        "RRF: %d dense + %d bm25 → %d unique → returning top %d",
        len(dense_results),
        len(bm25_results),
        len(merged),
        len(result),
    )
    return result


class HybridRetriever:
    """Convenience wrapper that runs dense + BM25 and fuses results via RRF."""

    def __init__(self, dense_retriever, bm25_retriever, rrf_k: int = RRF_K) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """Run hybrid retrieval and return top-*k* fused results.

        Args:
            query: Natural-language query.
            k:     Number of final results (passed as top_n to RRF).

        Returns:
            (Document, rrf_score) list, highest score first.
        """
        # Fetch more candidates per retriever so RRF has sufficient pool to merge
        fetch_k = max(k * 2, 10)
        dense_results = self.dense.retrieve(query, k=fetch_k)
        bm25_results = self.bm25.retrieve(query, k=fetch_k)
        return reciprocal_rank_fusion(
            dense_results, bm25_results, k=self.rrf_k, top_n=k
        )
