"""Split LangChain Documents into fixed-size overlapping chunks.

Chunk metadata carries everything from the parent Document plus:
  - chunk_index   : global sequential index across all chunks
  - chunk_size    : the chunk_size setting used to produce this chunk
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP_PCT

logger = logging.getLogger(__name__)


def chunk_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int | None = None,
) -> List[Document]:
    """Split *docs* into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        docs:       Source Documents (typically one per PDF page).
        chunk_size: Maximum number of characters per chunk.
        overlap:    Overlap in characters between adjacent chunks.
                    Defaults to ``int(chunk_size * CHUNK_OVERLAP_PCT)``.

    Returns:
        List of chunk Documents with full metadata from the parent Document
        plus ``chunk_index`` and ``chunk_size`` keys.
    """
    if overlap is None:
        overlap = int(chunk_size * CHUNK_OVERLAP_PCT)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Annotate each chunk with its global index and the chunk_size setting
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_size"] = chunk_size

    logger.info(
        "Chunked %d document(s) → %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
