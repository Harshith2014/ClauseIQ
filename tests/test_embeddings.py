"""Tests for embeddings.embedder.

NOTE: The first run downloads BAAI/bge-small-en-v1.5 (~130 MB) into .model_cache/.
      Subsequent runs use the cached model and are fast.
"""
from __future__ import annotations

import math
import pytest

from embeddings.embedder import get_embedder
from config import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    """Session-scoped embedder — model loaded once per test run."""
    return get_embedder()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_get_embedder_returns_instance(self, embedder):
        from langchain_huggingface import HuggingFaceEmbeddings
        assert isinstance(embedder, HuggingFaceEmbeddings)

    def test_singleton_behaviour(self):
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2, "get_embedder() should return the same cached instance."

    def test_embed_query_returns_list(self, embedder):
        result = embedder.embed_query("What is the termination clause?")
        assert isinstance(result, list), "embed_query should return a list."

    def test_embed_query_correct_dimension(self, embedder):
        result = embedder.embed_query("Non-compete agreement duration.")
        assert len(result) == EMBED_DIM, (
            f"Expected {EMBED_DIM}-dim vector, got {len(result)}."
        )

    def test_embed_documents_returns_list_of_lists(self, embedder):
        texts = [
            "The employee shall receive a base salary of $120,000.",
            "Confidential information shall not be disclosed.",
        ]
        results = embedder.embed_documents(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)
        for vec in results:
            assert len(vec) == EMBED_DIM

    def test_same_text_cosine_similarity_near_one(self, embedder):
        text = "The contract shall be governed by the laws of Delaware."
        v1 = embedder.embed_query(text)
        v2 = embedder.embed_query(text)
        sim = cosine_similarity(v1, v2)
        assert sim >= 0.999, f"Same text should have cosine ~1.0, got {sim:.4f}."

    def test_different_texts_lower_similarity(self, embedder):
        v1 = embedder.embed_query("The employee's annual salary is $120,000.")
        v2 = embedder.embed_query("The uptime guarantee is 99.9% per calendar month.")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.95, (
            f"Unrelated texts should have cosine < 0.95, got {sim:.4f}."
        )

    def test_semantically_similar_texts_higher_similarity(self, embedder):
        v1 = embedder.embed_query("Termination with 30 days notice.")
        v2 = embedder.embed_query("Either party may end the contract giving 30 days notice.")
        v3 = embedder.embed_query("Uptime SLA credits for service unavailability.")
        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated, (
            "Semantically related texts should have higher cosine than unrelated ones. "
            f"related={sim_related:.4f}, unrelated={sim_unrelated:.4f}"
        )

    def test_vectors_are_normalised(self, embedder):
        """With normalize_embeddings=True, L2 norm should be ~1.0."""
        vec = embedder.embed_query("Service level agreement penalty clause.")
        norm = math.sqrt(sum(x ** 2 for x in vec))
        assert abs(norm - 1.0) < 0.01, (
            f"Expected L2 norm ~1.0 (normalized embeddings), got {norm:.4f}."
        )

    def test_empty_string_embeds_without_error(self, embedder):
        result = embedder.embed_query("")
        assert len(result) == EMBED_DIM

    def test_batch_embed_consistent_with_single(self, embedder):
        texts = ["Governing law clause.", "Payment terms."]
        batch = embedder.embed_documents(texts)
        single_0 = embedder.embed_query(texts[0])
        sim = cosine_similarity(batch[0], single_0)
        assert sim >= 0.999, (
            "Batch and single embedding of the same text should be nearly identical."
        )
