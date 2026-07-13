"""Tests for ingestion.pdf_loader and ingestion.chunker.

All tests use the session-scoped sample PDFs created by tests/conftest.py.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from config import RAW_DIR, CHUNK_SIZES


# ---------------------------------------------------------------------------
# pdf_loader tests
# ---------------------------------------------------------------------------

class TestPDFLoader:
    def test_loads_at_least_one_document(self):
        docs = load_pdfs(RAW_DIR)
        assert len(docs) >= 1, "Expected at least one Document from PDF loading."

    def test_all_documents_are_langchain_type(self):
        docs = load_pdfs(RAW_DIR)
        for doc in docs:
            assert isinstance(doc, Document), f"Expected Document, got {type(doc)}"

    def test_source_metadata_present(self):
        docs = load_pdfs(RAW_DIR)
        for doc in docs:
            assert "source" in doc.metadata, "Missing 'source' in metadata."
            assert doc.metadata["source"].endswith(".pdf"), (
                f"'source' should be a .pdf filename, got: {doc.metadata['source']}"
            )

    def test_page_metadata_present(self):
        docs = load_pdfs(RAW_DIR)
        for doc in docs:
            assert "page" in doc.metadata, "Missing 'page' in metadata."
            assert isinstance(doc.metadata["page"], int), (
                f"'page' should be int, got {type(doc.metadata['page'])}"
            )

    def test_source_path_metadata_present(self):
        docs = load_pdfs(RAW_DIR)
        for doc in docs:
            assert "source_path" in doc.metadata, "Missing 'source_path' in metadata."

    def test_documents_have_non_empty_content(self):
        docs = load_pdfs(RAW_DIR)
        empty = [d for d in docs if not d.page_content.strip()]
        # Allow a small number of blank pages (e.g., cover pages) but not all
        assert len(empty) < len(docs), "All loaded documents have empty content."

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdfs(tmp_path)

    def test_multiple_pdf_files_loaded(self):
        docs = load_pdfs(RAW_DIR)
        sources = {d.metadata["source"] for d in docs}
        assert len(sources) >= 3, (
            f"Expected at least 3 source files, got: {sources}"
        )


# ---------------------------------------------------------------------------
# chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    @pytest.fixture(scope="class")
    def docs(self):
        return load_pdfs(RAW_DIR)

    def test_produces_chunks(self, docs):
        chunks = chunk_documents(docs, chunk_size=512)
        assert len(chunks) > 0

    def test_chunk_count_increases_with_smaller_size(self, docs):
        chunks_256 = chunk_documents(docs, chunk_size=256)
        chunks_512 = chunk_documents(docs, chunk_size=512)
        chunks_1024 = chunk_documents(docs, chunk_size=1024)
        assert len(chunks_256) >= len(chunks_512) >= len(chunks_1024), (
            "Smaller chunk sizes should produce more chunks."
        )

    @pytest.mark.parametrize("size", CHUNK_SIZES)
    def test_chunk_size_respected(self, docs, size):
        chunks = chunk_documents(docs, chunk_size=size)
        oversized = [c for c in chunks if len(c.page_content) > size * 1.1]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed chunk_size={size} by >10%. "
            "RecursiveCharacterTextSplitter may have an issue."
        )

    def test_chunk_index_present_and_sequential(self, docs):
        chunks = chunk_documents(docs, chunk_size=512)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), (
            "chunk_index should be 0-based sequential across all chunks."
        )

    def test_chunk_size_metadata_recorded(self, docs):
        chunks = chunk_documents(docs, chunk_size=256)
        for chunk in chunks:
            assert chunk.metadata.get("chunk_size") == 256

    def test_parent_metadata_preserved(self, docs):
        chunks = chunk_documents(docs, chunk_size=512)
        for chunk in chunks:
            assert "source" in chunk.metadata, "Parent 'source' metadata lost in chunk."
            assert "page" in chunk.metadata, "Parent 'page' metadata lost in chunk."

    def test_custom_overlap(self, docs):
        chunks = chunk_documents(docs, chunk_size=512, overlap=0)
        assert len(chunks) > 0

    def test_single_document_chunks(self):
        doc = Document(
            page_content="Section 1. This is clause one. " * 30,
            metadata={"source": "test.pdf", "page": 0},
        )
        chunks = chunk_documents([doc], chunk_size=100, overlap=10)
        assert len(chunks) >= 2, "Expected multiple chunks from a long single document."

    def test_short_document_not_split(self):
        doc = Document(
            page_content="Short text.",
            metadata={"source": "test.pdf", "page": 0},
        )
        chunks = chunk_documents([doc], chunk_size=512)
        assert len(chunks) == 1, "Short text should not be split into multiple chunks."
