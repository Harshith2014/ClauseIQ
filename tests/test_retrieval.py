"""Tests for vectorstore.faiss_store, retrieval.bm25_retriever, and retrieval.hybrid.

Uses a small in-memory corpus of 10 synthetic legal documents so tests are
fast and deterministic without hitting the full PDF pipeline.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from embeddings.embedder import get_embedder
from vectorstore.faiss_store import FAISSStore
from vectorstore.qdrant_store import QdrantStore
from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid import reciprocal_rank_fusion, HybridRetriever


# ---------------------------------------------------------------------------
# Shared test corpus — 10 docs covering distinct legal topics
# ---------------------------------------------------------------------------

CORPUS = [
    Document(page_content="The employee shall receive an annual base salary of $120,000 payable bi-weekly.",
             metadata={"source": "employment.pdf", "page": 0, "chunk_index": 0}),
    Document(page_content="The non-compete clause prevents employment in competing businesses for 12 months after termination.",
             metadata={"source": "employment.pdf", "page": 1, "chunk_index": 1}),
    Document(page_content="Provider guarantees 99.9% uptime per calendar month measured as availability percentage.",
             metadata={"source": "sla.pdf", "page": 0, "chunk_index": 2}),
    Document(page_content="Confidential information shall not be disclosed to any third party without prior written consent.",
             metadata={"source": "nda.pdf", "page": 0, "chunk_index": 3}),
    Document(page_content="Payment of $5,000 monthly fee is due within 30 days of invoice date.",
             metadata={"source": "sla.pdf", "page": 1, "chunk_index": 4}),
    Document(page_content="The employee is entitled to 20 days of paid time off accruing at 1.67 days per month.",
             metadata={"source": "employment.pdf", "page": 2, "chunk_index": 5}),
    Document(page_content="Governing law is the State of Delaware without regard to conflict of laws principles.",
             metadata={"source": "employment.pdf", "page": 3, "chunk_index": 6}),
    Document(page_content="SLA credits are issued when uptime falls below the guaranteed threshold in any calendar month.",
             metadata={"source": "sla.pdf", "page": 2, "chunk_index": 7}),
    Document(page_content="Either party may terminate the agreement with 30 days written notice to the other party.",
             metadata={"source": "employment.pdf", "page": 4, "chunk_index": 8}),
    Document(page_content="Employee benefits include health insurance, dental coverage, and vision care plans.",
             metadata={"source": "employment.pdf", "page": 5, "chunk_index": 9}),
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures (embedder + built FAISS store) — built once per run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


@pytest.fixture(scope="module")
def faiss_store(embedder):
    store = FAISSStore(embedder)
    store.build(CORPUS)
    return store


@pytest.fixture(scope="module")
def dense_retriever(faiss_store):
    return DenseRetriever(faiss_store)


@pytest.fixture(scope="module")
def bm25_retriever():
    return BM25Retriever(CORPUS)


@pytest.fixture(scope="module")
def hybrid_retriever(dense_retriever, bm25_retriever):
    return HybridRetriever(dense_retriever, bm25_retriever)


# ---------------------------------------------------------------------------
# FAISSStore tests
# ---------------------------------------------------------------------------

class TestFAISSStore:
    def test_build_populates_index(self, faiss_store):
        assert faiss_store.size == len(CORPUS)

    def test_similarity_search_returns_list(self, faiss_store):
        results = faiss_store.similarity_search("salary compensation", k=3)
        assert isinstance(results, list)

    def test_similarity_search_respects_k(self, faiss_store):
        for k in [1, 3, 5]:
            results = faiss_store.similarity_search("termination notice", k=k)
            assert len(results) == k

    def test_similarity_search_k_clamped_to_corpus_size(self, faiss_store):
        results = faiss_store.similarity_search("salary", k=100)
        assert len(results) == len(CORPUS)  # can't exceed corpus

    def test_similarity_search_returns_documents_and_floats(self, faiss_store):
        results = faiss_store.similarity_search("uptime guarantee", k=3)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_similarity_search_scores_in_valid_range(self, faiss_store):
        results = faiss_store.similarity_search("confidential information", k=5)
        for _, score in results:
            # Cosine similarity on normalised vecs is in [-1, 1]
            assert -1.1 <= score <= 1.1

    def test_similarity_search_top_result_is_relevant(self, faiss_store):
        results = faiss_store.similarity_search("annual salary compensation employee", k=3)
        top_doc = results[0][0]
        # The salary doc (chunk_index=0) or the paid time off doc should rank first
        assert any(
            kw in top_doc.page_content.lower()
            for kw in ["salary", "compensation", "paid", "benefit"]
        )

    def test_similarity_search_scores_descending(self, faiss_store):
        results = faiss_store.similarity_search("uptime SLA guarantee", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Scores should be descending."

    def test_metadata_preserved_after_search(self, faiss_store):
        results = faiss_store.similarity_search("uptime", k=3)
        for doc, _ in results:
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata

    def test_save_and_load(self, faiss_store, tmp_path, embedder):
        faiss_store.save(tmp_path)
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "docs.pkl").exists()

        new_store = FAISSStore(embedder)
        new_store.load(tmp_path)
        assert new_store.size == faiss_store.size

        # Results should match
        q = "termination notice period"
        original = faiss_store.similarity_search(q, k=3)
        loaded = new_store.similarity_search(q, k=3)
        orig_ids = [d.metadata["chunk_index"] for d, _ in original]
        load_ids = [d.metadata["chunk_index"] for d, _ in loaded]
        assert orig_ids == load_ids

    def test_build_raises_on_empty(self, embedder):
        store = FAISSStore(embedder)
        with pytest.raises(ValueError):
            store.build([])

    def test_search_raises_before_build(self, embedder):
        store = FAISSStore(embedder)
        with pytest.raises(RuntimeError):
            store.similarity_search("test")


# ---------------------------------------------------------------------------
# QdrantStore stub tests
# ---------------------------------------------------------------------------

class TestQdrantStoreStub:
    def test_build_raises_not_implemented(self, embedder):
        store = QdrantStore(embedder)
        with pytest.raises(NotImplementedError):
            store.build(CORPUS)

    def test_similarity_search_raises_not_implemented(self, embedder):
        store = QdrantStore(embedder)
        with pytest.raises(NotImplementedError):
            store.similarity_search("test")


# ---------------------------------------------------------------------------
# BM25Retriever tests
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_retrieve_returns_list(self, bm25_retriever):
        results = bm25_retriever.retrieve("salary employee", k=3)
        assert isinstance(results, list)

    def test_retrieve_respects_k(self, bm25_retriever):
        for k in [1, 3, 5]:
            results = bm25_retriever.retrieve("termination notice", k=k)
            assert len(results) == k

    def test_retrieve_returns_documents_and_floats(self, bm25_retriever):
        results = bm25_retriever.retrieve("confidential information", k=3)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_retrieve_scores_non_negative(self, bm25_retriever):
        results = bm25_retriever.retrieve("uptime guarantee sla", k=5)
        for _, score in results:
            assert score >= 0.0

    def test_retrieve_scores_descending(self, bm25_retriever):
        results = bm25_retriever.retrieve("uptime calendar month", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_keyword_match_ranks_first(self, bm25_retriever):
        # "uptime" appears only in the SLA-related docs
        results = bm25_retriever.retrieve("uptime guarantee 99.9 percent", k=3)
        top_doc = results[0][0]
        assert "uptime" in top_doc.page_content.lower()

    def test_retrieve_salary_doc_ranks_first(self, bm25_retriever):
        results = bm25_retriever.retrieve("annual salary 120000", k=3)
        top_doc = results[0][0]
        assert "salary" in top_doc.page_content.lower()

    def test_build_raises_on_empty(self):
        with pytest.raises(ValueError):
            BM25Retriever([])

    def test_metadata_preserved(self, bm25_retriever):
        results = bm25_retriever.retrieve("confidential", k=3)
        for doc, _ in results:
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata

    def test_k_clamped_to_corpus_size(self, bm25_retriever):
        results = bm25_retriever.retrieve("law", k=1000)
        assert len(results) == len(CORPUS)


# ---------------------------------------------------------------------------
# Hybrid (RRF) tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def _make_results(self, indices_scores):
        return [(CORPUS[i], s) for i, s in indices_scores]

    def test_rrf_no_duplicate_documents(self, dense_retriever, bm25_retriever):
        query = "confidential information disclosure"
        dense = dense_retriever.retrieve(query, k=8)
        bm25 = bm25_retriever.retrieve(query, k=8)
        merged = reciprocal_rank_fusion(dense, bm25)

        chunk_ids = [d.metadata["chunk_index"] for d, _ in merged]
        assert len(chunk_ids) == len(set(chunk_ids)), "RRF output contains duplicate documents."

    def test_rrf_scores_descending(self, dense_retriever, bm25_retriever):
        query = "termination notice 30 days"
        dense = dense_retriever.retrieve(query, k=8)
        bm25 = bm25_retriever.retrieve(query, k=8)
        merged = reciprocal_rank_fusion(dense, bm25)
        scores = [s for _, s in merged]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_doc_in_both_lists_ranks_higher(self):
        """A doc appearing in both ranked lists should outscore one appearing in only one."""
        # doc 0 appears at rank 1 in both lists → high RRF
        # doc 9 appears at rank 1 in dense only, rank 9 in bm25
        # doc 8 appears at rank 1 in bm25 only, not in dense
        dense = self._make_results([(0, 0.95), (9, 0.80), (1, 0.70)])
        bm25 = self._make_results([(0, 12.0), (8, 8.0), (1, 4.0)])

        merged = reciprocal_rank_fusion(dense, bm25, top_n=10)
        merged_ids = [d.metadata["chunk_index"] for d, _ in merged]

        # doc 0 should be first (appears rank-1 in both)
        assert merged_ids[0] == 0

    def test_rrf_top_n_truncates_output(self, dense_retriever, bm25_retriever):
        query = "salary benefits compensation"
        dense = dense_retriever.retrieve(query, k=10)
        bm25 = bm25_retriever.retrieve(query, k=10)
        for top_n in [1, 3, 5]:
            merged = reciprocal_rank_fusion(dense, bm25, top_n=top_n)
            assert len(merged) <= top_n

    def test_rrf_handles_empty_dense(self, bm25_retriever):
        bm25 = bm25_retriever.retrieve("uptime", k=5)
        merged = reciprocal_rank_fusion([], bm25)
        assert len(merged) > 0

    def test_rrf_handles_empty_bm25(self, dense_retriever):
        dense = dense_retriever.retrieve("uptime", k=5)
        merged = reciprocal_rank_fusion(dense, [])
        assert len(merged) > 0

    def test_rrf_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_all_scores_positive(self, dense_retriever, bm25_retriever):
        query = "governing law jurisdiction"
        dense = dense_retriever.retrieve(query, k=5)
        bm25 = bm25_retriever.retrieve(query, k=5)
        merged = reciprocal_rank_fusion(dense, bm25)
        for _, score in merged:
            assert score > 0.0


# ---------------------------------------------------------------------------
# HybridRetriever convenience wrapper tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_retrieve_returns_list(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("salary benefits", k=5)
        assert isinstance(results, list)

    def test_retrieve_respects_k(self, hybrid_retriever):
        for k in [1, 3, 5]:
            results = hybrid_retriever.retrieve("confidential disclosure", k=k)
            assert len(results) <= k

    def test_retrieve_no_duplicates(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("uptime sla credits payment", k=5)
        ids = [d.metadata["chunk_index"] for d, _ in results]
        assert len(ids) == len(set(ids))

    def test_retrieve_scores_descending(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("termination notice", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_returns_documents(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("employee salary", k=3)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
