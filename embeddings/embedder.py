"""Singleton wrapper around HuggingFace BGE-small-en-v1.5 embeddings.

BGE-small-en-v1.5 produces 384-dimensional L2-normalised embeddings.
Normalisation is enabled so cosine similarity == dot product (required for FAISS
IndexFlatIP inner-product search).

Usage:
    from embeddings.embedder import get_embedder
    embedder = get_embedder()
    vector = embedder.embed_query("What is the termination clause?")
"""
from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, MODELS_CACHE_DIR

logger = logging.getLogger(__name__)

_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (downloads model on first call).

    The model is downloaded once to MODELS_CACHE_DIR (.model_cache/) and reused
    across all subsequent calls within the same process.
    """
    global _embedder
    if _embedder is None:
        logger.info("Initialising embedder: %s", EMBED_MODEL)
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            cache_folder=str(MODELS_CACHE_DIR),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # L2 norm → cosine = dot
        )
        logger.info("Embedder ready.")
    return _embedder
