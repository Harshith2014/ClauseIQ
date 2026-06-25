"""Tests for generation.prompts and generation.generator.

The LLM is mocked throughout — no Ollama required to run these tests.
Tests cover: context formatting, citation parsing, confidence parsing,
chain wiring, and structured output guarantees.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from generation.prompts import format_context, LEGAL_QA_PROMPT
from generation.generator import (
    Citation,
    GenerationResult,
    LegalQAChain,
    parse_citations,
    parse_confidence,
    clean_answer,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    (Document(
        page_content="2.1 Base Salary. Employer shall pay Employee an annual base salary of One Hundred Twenty Thousand Dollars ($120,000), payable bi-weekly.",
        metadata={"source": "employment_contract.pdf", "page": 0, "chunk_index": 2},
    ), 0.0325),
    (Document(
        page_content="5.1 Non-Compete. For a period of twelve (12) months following termination, Employee shall not engage in or be employed by any business that directly competes with Employer.",
        metadata={"source": "employment_contract.pdf", "page": 1, "chunk_index": 7},
    ), 0.0310),
    (Document(
        page_content="2.1 Uptime Commitment. Provider guarantees that the Services will be available ninety-nine point nine percent (99.9%) of each calendar month.",
        metadata={"source": "service_level_agreement.pdf", "page": 0, "chunk_index": 25},
    ), 0.0298),
]

# A well-formed LLM response with two citations and a confidence score
WELL_FORMED_RESPONSE = (
    'The employee\'s annual base salary is $120,000, paid bi-weekly. '
    '[Source: employment_contract.pdf, Page: 0, Clause: "2.1 Base Salary. Employer shall pay Employee an annual base salary of One Hundred Twenty Thousand Dollars ($120,000), payable bi-weekly."]\n\n'
    'Additionally, the employee is subject to a 12-month non-compete clause following termination. '
    '[Source: employment_contract.pdf, Page: 1, Clause: "For a period of twelve (12) months following termination, Employee shall not engage in or be employed by any business that directly competes with Employer."]\n\n'
    'Confidence: 0.95'
)

MULTI_SOURCE_RESPONSE = (
    'The SLA guarantees 99.9% uptime. '
    '[Source: service_level_agreement.pdf, Page: 0, Clause: "Provider guarantees that the Services will be available ninety-nine point nine percent (99.9%)"]\n'
    'Credits of 25% apply for uptime between 95%-99%. '
    '[Source: service_level_agreement.pdf, Page: 1, Clause: "95.0% - 99.0% uptime: Credit equal to twenty-five percent (25%) of monthly fee."]\n'
    'Confidence: 0.9'
)

NOT_FOUND_RESPONSE = "Not found in the provided documents.\n\nConfidence: 0.0"

MALFORMED_RESPONSE = "The salary is $120,000. This is a hallucinated claim with no citation."


# ---------------------------------------------------------------------------
# TestFormatContext
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_returns_string(self):
        result = format_context(SAMPLE_DOCS)
        assert isinstance(result, str)

    def test_includes_source(self):
        result = format_context(SAMPLE_DOCS)
        assert "employment_contract.pdf" in result

    def test_includes_page_number(self):
        result = format_context(SAMPLE_DOCS)
        assert "Page" in result

    def test_includes_content(self):
        result = format_context(SAMPLE_DOCS)
        assert "$120,000" in result

    def test_numbered_blocks(self):
        result = format_context(SAMPLE_DOCS)
        assert "[Block 1]" in result
        assert "[Block 2]" in result
        assert "[Block 3]" in result

    def test_empty_context_returns_placeholder(self):
        result = format_context([])
        assert "no context" in result.lower()

    def test_long_chunk_truncated(self):
        long_doc = Document(
            page_content="A" * 2000,
            metadata={"source": "test.pdf", "page": 0},
        )
        result = format_context([(long_doc, 0.5)])
        assert "truncated" in result.lower()
        assert len(result) < 2000 + 200  # significant compression

    def test_multiple_sources_all_present(self):
        result = format_context(SAMPLE_DOCS)
        assert "employment_contract.pdf" in result
        assert "service_level_agreement.pdf" in result


# ---------------------------------------------------------------------------
# TestParseCitations
# ---------------------------------------------------------------------------

class TestParseCitations:
    def test_single_citation_parsed(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        assert len(citations) >= 1

    def test_two_citations_parsed(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        assert len(citations) == 2

    def test_citation_has_correct_source(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        sources = [c.source for c in citations]
        assert "employment_contract.pdf" in sources

    def test_citation_has_correct_page(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        pages = [c.page for c in citations]
        assert 0 in pages
        assert 1 in pages

    def test_citation_page_is_int(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        for c in citations:
            assert isinstance(c.page, int)

    def test_citation_clause_non_empty(self):
        citations = parse_citations(WELL_FORMED_RESPONSE)
        for c in citations:
            assert len(c.clause) > 0

    def test_multi_source_citations(self):
        citations = parse_citations(MULTI_SOURCE_RESPONSE)
        sources = {c.source for c in citations}
        assert "service_level_agreement.pdf" in sources

    def test_deduplication(self):
        # Duplicate citation in text
        dup = WELL_FORMED_RESPONSE + (
            '\n[Source: employment_contract.pdf, Page: 0, Clause: "2.1 Base Salary. Employer shall pay Employee an annual base salary of One Hundred Twenty Thousand Dollars ($120,000), payable bi-weekly."]'
        )
        citations = parse_citations(dup)
        sources_pages = [(c.source, c.page) for c in citations]
        assert len(sources_pages) == len(set(sources_pages))

    def test_no_citations_returns_empty_list(self):
        citations = parse_citations(MALFORMED_RESPONSE)
        assert citations == []

    def test_not_found_response_no_citations(self):
        citations = parse_citations(NOT_FOUND_RESPONSE)
        assert citations == []

    def test_citation_str_repr(self):
        c = Citation(source="test.pdf", page=0, clause="some clause text")
        s = str(c)
        assert "test.pdf" in s
        assert "some clause text" in s

    def test_multi_source_both_pages_cited(self):
        citations = parse_citations(MULTI_SOURCE_RESPONSE)
        pages = [c.page for c in citations]
        assert 0 in pages
        assert 1 in pages


# ---------------------------------------------------------------------------
# TestParseConfidence
# ---------------------------------------------------------------------------

class TestParseConfidence:
    def test_high_confidence_parsed(self):
        conf = parse_confidence(WELL_FORMED_RESPONSE)
        assert abs(conf - 0.95) < 0.01

    def test_confidence_09(self):
        conf = parse_confidence(MULTI_SOURCE_RESPONSE)
        assert abs(conf - 0.9) < 0.01

    def test_zero_confidence(self):
        conf = parse_confidence(NOT_FOUND_RESPONSE)
        assert conf == 0.0

    def test_no_confidence_defaults_to_05(self):
        conf = parse_confidence(MALFORMED_RESPONSE)
        assert conf == 0.5

    def test_confidence_clamped_to_0_1(self):
        text = "Confidence: 1.5"
        conf = parse_confidence(text)
        assert conf <= 1.0

    def test_confidence_negative_clamped(self):
        text = "Confidence: -0.3"
        conf = parse_confidence(text)
        assert conf >= 0.0

    def test_confidence_exactly_one(self):
        conf = parse_confidence("Confidence: 1.0")
        assert conf == 1.0

    def test_confidence_exactly_zero(self):
        conf = parse_confidence("Confidence: 0.0")
        assert conf == 0.0


# ---------------------------------------------------------------------------
# TestCleanAnswer
# ---------------------------------------------------------------------------

class TestCleanAnswer:
    def test_removes_confidence_line(self):
        cleaned = clean_answer(WELL_FORMED_RESPONSE)
        assert "Confidence:" not in cleaned

    def test_answer_content_preserved(self):
        cleaned = clean_answer(WELL_FORMED_RESPONSE)
        assert "$120,000" in cleaned
        assert "non-compete" in cleaned

    def test_citations_preserved(self):
        cleaned = clean_answer(WELL_FORMED_RESPONSE)
        assert "[Source:" in cleaned

    def test_not_found_cleaned(self):
        cleaned = clean_answer(NOT_FOUND_RESPONSE)
        assert "Not found" in cleaned
        assert "Confidence:" not in cleaned


# ---------------------------------------------------------------------------
# TestLegalQAChain (mocked LLM)
# ---------------------------------------------------------------------------

def _make_chain(response_text: str) -> LegalQAChain:
    """Build a LegalQAChain whose LLM always returns *response_text*.

    Uses RunnableLambda as the fake LLM — a first-class LangChain Runnable
    that slots cleanly into the LCEL | chain without any patching.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from tests.test_retrieval import CORPUS
    from embeddings.embedder import get_embedder
    from vectorstore.faiss_store import FAISSStore
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.hybrid import HybridRetriever

    embedder  = get_embedder()
    store     = FAISSStore(embedder)
    store.build(CORPUS)
    retriever = HybridRetriever(DenseRetriever(store), BM25Retriever(CORPUS))

    # RunnableLambda ignores its input and always returns our scripted AIMessage.
    # StrOutputParser then extracts .content — exactly what the real LLM does.
    fake_llm = RunnableLambda(lambda _: AIMessage(content=response_text))

    return LegalQAChain(retriever=retriever, top_k=3, llm=fake_llm)


class TestLegalQAChain:
    @pytest.fixture(scope="class")
    def chain_well_formed(self):
        return _make_chain(WELL_FORMED_RESPONSE)

    @pytest.fixture(scope="class")
    def chain_not_found(self):
        return _make_chain(NOT_FOUND_RESPONSE)

    @pytest.fixture(scope="class")
    def chain_malformed(self):
        return _make_chain(MALFORMED_RESPONSE)

    # --- Return type ---
    def test_returns_generation_result(self, chain_well_formed):
        result = chain_well_formed.run("What is the salary?")
        assert isinstance(result, GenerationResult)

    # --- Fields present ---
    def test_result_has_query(self, chain_well_formed):
        q = "What is the annual salary?"
        result = chain_well_formed.run(q)
        assert result.query == q

    def test_result_has_non_empty_answer(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_result_has_citations_list(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert isinstance(result.citations, list)

    def test_result_has_confidence_float(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_raw_response(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert isinstance(result.raw_response, str)
        assert len(result.raw_response) > 0

    def test_result_has_num_context_chunks(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert isinstance(result.num_context_chunks, int)
        assert result.num_context_chunks > 0

    # --- Well-formed response correctness ---
    def test_well_formed_two_citations(self, chain_well_formed):
        result = chain_well_formed.run("Salary and non-compete?")
        assert len(result.citations) == 2

    def test_well_formed_confidence_095(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert abs(result.confidence - 0.95) < 0.01

    def test_well_formed_answer_no_confidence_line(self, chain_well_formed):
        result = chain_well_formed.run("Salary?")
        assert "Confidence:" not in result.answer

    # --- Not-found response ---
    def test_not_found_returns_zero_confidence(self, chain_not_found):
        result = chain_not_found.run("What is the governing planet?")
        assert result.confidence == 0.0

    def test_not_found_empty_citations(self, chain_not_found):
        result = chain_not_found.run("Obscure question?")
        assert result.citations == []

    # --- Malformed response (no citations) ---
    def test_malformed_no_citations(self, chain_malformed):
        result = chain_malformed.run("Salary?")
        assert result.citations == []

    def test_malformed_default_confidence(self, chain_malformed):
        result = chain_malformed.run("Salary?")
        assert result.confidence == 0.5

    # --- End-to-end wiring ---
    def test_chain_returns_result_for_any_query(self, chain_well_formed):
        result = chain_well_formed.run("Test query about the contract.")
        assert isinstance(result, GenerationResult)
        assert len(result.answer) > 0

    def test_chain_question_recorded_in_result(self, chain_well_formed):
        q = "Termination notice period?"
        result = chain_well_formed.run(q)
        assert result.query == q
