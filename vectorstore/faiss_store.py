"""FAISS vector store — build, save, load, and search.

Uses IndexFlatIP (inner product) which equals cosine similarity when vectors
are L2-normalised (which BGE-small-en-v1.5 produces with normalize_embeddings=True).
Higher score = more similar.

Persistence layout (save_dir/):
    index.faiss  — raw FAISS binary index
    docs.pkl     — pickled list[Document] parallel to index vectors
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

from config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)


class FAISSStore:
    """Thin FAISS wrapper with LangChain Document support."""

    def __init__(self, embedder) -> None:
        """
        Args:
            embedder: HuggingFaceEmbeddings instance (from embeddings.embedder).
        """
        self.embedder = embedder
        self._index: faiss.Index | None = None
        self._docs: List[Document] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Document]) -> None:
        """Embed *chunks* and populate the FAISS index.

        Args:
            chunks: List of Documents (output of ingestion.chunker).
        """
        if not chunks:
            raise ValueError("chunks must be non-empty.")

        texts = [c.page_content for c in chunks]
        logger.info("Embedding %d chunks for FAISS index …", len(chunks))
        vectors = np.array(self.embedder.embed_documents(texts), dtype=np.float32)

        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product = cosine on normed vecs
        self._index.add(vectors)
        self._docs = list(chunks)

        logger.info(
            "FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: Path = VECTORSTORE_DIR) -> None:
        """Persist the index and documents to *save_dir*."""
        if self._index is None:
            raise RuntimeError("Call build() before save().")
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(save_dir / "index.faiss"))
        with open(save_dir / "docs.pkl", "wb") as fh:
            pickle.dump(self._docs, fh)
        logger.info("FAISS index saved to %s", save_dir)

    def load(self, save_dir: Path = VECTORSTORE_DIR) -> None:
        """Load a previously saved index from *save_dir*."""
        index_path = save_dir / "index.faiss"
        docs_path = save_dir / "docs.pkl"
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"No saved FAISS index found in '{save_dir}'. Run build() + save() first."
            )
        self._index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as fh:
            self._docs = pickle.load(fh)
        logger.info(
            "FAISS index loaded from %s (%d vectors)", save_dir, self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Return the top-*k* most similar Documents and their cosine scores.

        Args:
            query: Natural-language query string.
            k:     Number of results to return.

        Returns:
            List of (Document, cosine_score) tuples, highest score first.
        """
        if self._index is None:
            raise RuntimeError("Index not initialised. Call build() or load().")

        k = min(k, self._index.ntotal)
        vec = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        scores, indices = self._index.search(vec, k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS uses -1 for unfilled slots
                results.append((self._docs[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0
