"""Tests for the eval harness.

Tests cover:
  - GoldenPair serialisation / round-trip
  - RAGAS metric functions (faithfulness, precision, recall, relevancy) with mocked LLM
  - LLM judge parsing (JSON + regex fallback)
  - Unanswerable detection
  - TOC artifact check
  - Multi-jurisdiction coverage check
  - run_eval aggregate() function
"""
from __future__ import annotations

import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.golden_dataset import (
    GoldenPair, CURATED_PAIRS, save_golden, load_golden,
)
from eval.ragas_scorer import (
    RAGASScores, score_faithfulness, score_context_precision,
    score_context_recall, _cosine, check_toc_artifact, check_multi_jurisdiction,
    _split_statements,
)
from eval.llm_judge import JudgeResult, DimensionScore, judge, _parse_dim, _regex_fallback
from eval.run_eval import is_not_found_answer, aggregate, EvalResult


# ---------------------------------------------------------------------------
# Golden dataset tests
# ---------------------------------------------------------------------------

class TestGoldenDataset(unittest.TestCase):

    def test_curated_count_meets_minimum(self):
        self.assertGreaterEqual(len(CURATED_PAIRS), 20,
                                "Expected at least 20 curated pairs")

    def test_unanswerable_pairs_present(self):
        unanswerable = [p for p in CURATED_PAIRS if p.unanswerable]
        self.assertGreaterEqual(len(unanswerable), 3,
                                "Must have at least 3 unanswerable pairs")

    def test_multi_jurisdiction_pair_present(self):
        mj = [p for p in CURATED_PAIRS if p.multi_jurisdiction]
        self.assertGreaterEqual(len(mj), 1)
        # Must list all 3 expected source files
        for p in mj:
            self.assertGreaterEqual(len(p.expected_sources), 3)

    def test_toc_artifact_pair_present(self):
        toc = [p for p in CURATED_PAIRS if p.toc_artifact_risk]
        self.assertGreaterEqual(len(toc), 1)

    def test_all_pairs_have_required_fields(self):
        for p in CURATED_PAIRS:
            self.assertIsInstance(p.id, str)
            self.assertTrue(p.id, f"Pair has empty id")
            self.assertIsInstance(p.question, str)
            self.assertTrue(p.question)
            self.assertIsInstance(p.ground_truth, str)
            self.assertTrue(p.ground_truth)
            self.assertIsInstance(p.category, str)

    def test_unique_ids(self):
        ids = [p.id for p in CURATED_PAIRS]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate pair IDs found")

    def test_save_and_load_roundtrip(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "golden_test.json"
            save_golden(CURATED_PAIRS, path=path)
            loaded = load_golden(path=path)
        self.assertEqual(len(loaded), len(CURATED_PAIRS))
        for orig, loaded_p in zip(CURATED_PAIRS, loaded):
            self.assertEqual(orig.id, loaded_p.id)
            self.assertEqual(orig.question, loaded_p.question)
            self.assertEqual(orig.unanswerable, loaded_p.unanswerable)
            self.assertEqual(orig.multi_jurisdiction, loaded_p.multi_jurisdiction)

    def test_load_golden_fallback_when_missing(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nonexistent.json"
            pairs = load_golden(path=path)
        self.assertEqual(pairs, list(CURATED_PAIRS))


# ---------------------------------------------------------------------------
# RAGAS metric tests (LLM calls mocked)
# ---------------------------------------------------------------------------

_FAITHFUL_LLM_RESPONSE = (
    "CLAIM: The salary is $175,000.\nSUPPORTED\n"
    "CLAIM: The term is 3 years.\nUNSUPPORTED\n"
    "SCORE: 1/2"
)
_PRECISION_YES = "YES"
_PRECISION_NO = "NO"
_RECALL_YES = "YES"


class TestRAGASScorer(unittest.TestCase):

    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.5]
        self.assertAlmostEqual(_cosine(v, v), 1.0, places=5)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(_cosine([1, 0], [0, 1]), 0.0, places=5)

    def test_cosine_zero_vector(self):
        self.assertEqual(_cosine([0, 0], [1, 1]), 0.0)

    def test_split_statements(self):
        text = "The salary is $175,000. The term is 3 years. Governing law is Texas."
        stmts = _split_statements(text)
        self.assertEqual(len(stmts), 3)

    def test_split_statements_single(self):
        stmts = _split_statements("Single statement without period")
        self.assertEqual(len(stmts), 1)

    @patch("eval.ragas_scorer._ollama", return_value=_FAITHFUL_LLM_RESPONSE)
    def test_faithfulness_parsed_score(self, mock_llm):
        score = score_faithfulness("The salary is $175,000.", ["Salary is $175,000 per year."])
        self.assertAlmostEqual(score, 0.5, places=2)

    @patch("eval.ragas_scorer._ollama", return_value="SCORE: 3/3")
    def test_faithfulness_perfect(self, mock_llm):
        score = score_faithfulness("All supported.", ["context"] * 3)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_faithfulness_empty_answer(self):
        score = score_faithfulness("", ["some context"])
        self.assertEqual(score, 0.0)

    def test_faithfulness_empty_context(self):
        score = score_faithfulness("Some answer", [])
        self.assertEqual(score, 0.0)

    @patch("eval.ragas_scorer._ollama", return_value=_PRECISION_YES)
    def test_context_precision_all_relevant(self, mock_llm):
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        score = score_context_precision("What is the salary?", chunks)
        self.assertAlmostEqual(score, 1.0, places=2)

    @patch("eval.ragas_scorer._ollama", return_value=_PRECISION_NO)
    def test_context_precision_none_relevant(self, mock_llm):
        chunks = ["irrelevant chunk"]
        score = score_context_precision("What is the salary?", chunks)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_context_precision_empty(self):
        score = score_context_precision("question", [])
        self.assertEqual(score, 0.0)

    @patch("eval.ragas_scorer._ollama", return_value=_RECALL_YES)
    def test_context_recall_full(self, mock_llm):
        score = score_context_recall("Salary is $175,000. Term is 3 years.", ["context chunk"])
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_context_recall_empty_ground_truth(self):
        score = score_context_recall("", ["context"])
        self.assertEqual(score, 0.0)

    def test_ragas_scores_mean(self):
        r = RAGASScores(faithfulness=0.8, answer_relevancy=0.6,
                        context_precision=0.7, context_recall=0.9)
        self.assertAlmostEqual(r.mean(), (0.8 + 0.6 + 0.7 + 0.9) / 4, places=5)

    def test_ragas_scores_as_dict(self):
        r = RAGASScores(faithfulness=0.75, answer_relevancy=0.5,
                        context_precision=0.6, context_recall=0.8)
        d = r.as_dict()
        self.assertIn("mean", d)
        self.assertEqual(d["faithfulness"], 0.75)

    def test_check_toc_artifact_detects_page0(self):
        class FakeCitation:
            page = 0
        self.assertTrue(check_toc_artifact([FakeCitation()]))

    def test_check_toc_artifact_no_false_positive(self):
        class FakeCitation:
            page = 5
        self.assertFalse(check_toc_artifact([FakeCitation()]))

    def test_check_toc_artifact_empty(self):
        self.assertFalse(check_toc_artifact([]))

    def test_check_multi_jurisdiction_all_found(self):
        answer = "Texas law applies to exhibit101. Jaipur India governs EMPLOYMENT-AGREEMENT. Victoria Australia governs the sample contract."
        sources = ["exhibit101.pdf", "EMPLOYMENT-AGREEMENT.pdf", "sample-standard_contract.pdf"]
        cov = check_multi_jurisdiction(answer, sources)
        self.assertTrue(all(cov.values()), f"Expected all found, got {cov}")

    def test_check_multi_jurisdiction_missing(self):
        answer = "Texas law applies. Jaipur India governs the contract."
        sources = ["exhibit101.pdf", "EMPLOYMENT-AGREEMENT.pdf", "sample-standard_contract.pdf"]
        cov = check_multi_jurisdiction(answer, sources)
        # Victoria/Australia not mentioned
        self.assertFalse(cov.get("sample-standard_contract.pdf", True))


# ---------------------------------------------------------------------------
# LLM judge tests
# ---------------------------------------------------------------------------

_GOOD_JUDGE_JSON = json.dumps({
    "correctness":     {"score": 4, "reasoning": "Mostly correct."},
    "faithfulness":    {"score": 5, "reasoning": "Fully grounded."},
    "helpfulness":     {"score": 3, "reasoning": "Adequate."},
    "citation_quality": {"score": 4, "reasoning": "Good citations."},
})

_BAD_JSON_JUDGE = (
    '"correctness": {"score": 4, "reasoning": "ok"}'
    '"faithfulness": {"score": 3}'
)


class TestLLMJudge(unittest.TestCase):

    def test_parse_dim_valid(self):
        obj = {"correctness": {"score": 4, "reasoning": "Good."}}
        d = _parse_dim(obj, "correctness")
        self.assertEqual(d.score, 4)
        self.assertEqual(d.reasoning, "Good.")

    def test_parse_dim_clamps_score(self):
        obj = {"correctness": {"score": 9}}
        d = _parse_dim(obj, "correctness")
        self.assertEqual(d.score, 5)

    def test_parse_dim_missing_key(self):
        d = _parse_dim({}, "correctness")
        self.assertEqual(d.score, 1)  # min clamped

    def test_regex_fallback(self):
        raw = '"correctness": {"score": 3, "reasoning": "test"}'
        d = _regex_fallback(raw, "correctness")
        self.assertEqual(d.score, 3)

    def test_regex_fallback_missing(self):
        d = _regex_fallback("no scores here", "correctness")
        self.assertEqual(d.score, 3)  # default middle

    @patch("eval.llm_judge._ollama", return_value=_GOOD_JUDGE_JSON)
    def test_judge_valid_json(self, mock_llm):
        r = judge("What is the salary?", "It is $175,000.", "$175,000.", ["The salary is $175,000."])
        self.assertEqual(r.correctness.score, 4)
        self.assertEqual(r.faithfulness.score, 5)
        self.assertAlmostEqual(r.overall, (4 + 5 + 3 + 4) / 4, places=3)
        self.assertIsNone(r.error)

    @patch("eval.llm_judge._ollama", return_value="no json at all")
    def test_judge_no_json_sets_error(self, mock_llm):
        r = judge("Q", "A", "GT", ["ctx"])
        self.assertIsNotNone(r.error)

    @patch("eval.llm_judge._ollama", side_effect=Exception("network down"))
    def test_judge_exception_safe(self, mock_llm):
        r = judge("Q", "A", "GT", ["ctx"])
        self.assertIsNotNone(r.error)
        self.assertEqual(r.overall, 0.0)


# ---------------------------------------------------------------------------
# Unanswerable detection tests
# ---------------------------------------------------------------------------

class TestUnanswerableDetection(unittest.TestCase):

    def test_detects_not_found(self):
        self.assertTrue(is_not_found_answer("The information is not found in the documents."))

    def test_detects_not_specified(self):
        self.assertTrue(is_not_found_answer("This is not specified in any of the agreements."))

    def test_detects_cannot_determine(self):
        self.assertTrue(is_not_found_answer("I cannot determine the answer from the context."))

    def test_detects_blank_template(self):
        self.assertTrue(is_not_found_answer("The job title is left blank in the template."))

    def test_hallucination_case(self):
        # A definite answer where none should exist
        self.assertFalse(is_not_found_answer("The strike price is $2.50 per share."))

    def test_real_answer_not_flagged(self):
        self.assertFalse(is_not_found_answer("The annual salary is $175,000."))


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------

def _make_eval_result(pair: GoldenPair, ragas_scores=None, judge_result=None, flags=None):
    """Helper to create a minimal EvalResult for aggregate() tests."""
    if ragas_scores is None:
        ragas_scores = RAGASScores(0.8, 0.7, 0.6, 0.9)
    return EvalResult(
        pair=pair,
        answer="test answer",
        confidence=0.75,
        context_chunks=["ctx"],
        citations=[],
        ragas=ragas_scores,
        judge_result=judge_result,
        elapsed=1.5,
        flags=flags or {},
    )


class TestAggregate(unittest.TestCase):

    def test_aggregate_basic(self):
        pairs = CURATED_PAIRS[:3]
        results = [_make_eval_result(p) for p in pairs]
        summary = aggregate(results)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["ragas_faithfulness"], 0.8, places=3)
        self.assertAlmostEqual(summary["ragas_context_recall"], 0.9, places=3)

    def test_aggregate_with_judge(self):
        jr = JudgeResult(
            question="Q", answer="A",
            correctness=DimensionScore(4, "ok"),
            faithfulness=DimensionScore(5, "ok"),
            helpfulness=DimensionScore(3, "ok"),
            citation_quality=DimensionScore(4, "ok"),
            overall=4.0,
        )
        pair = CURATED_PAIRS[0]
        results = [_make_eval_result(pair, judge_result=jr)]
        summary = aggregate(results)
        self.assertIn("judge_overall", summary)
        self.assertAlmostEqual(summary["judge_overall"], 4.0, places=3)

    def test_aggregate_unanswerable_tracking(self):
        u_pair = next(p for p in CURATED_PAIRS if p.unanswerable)
        r = _make_eval_result(u_pair, flags={"unanswerable_detected": True, "hallucination_risk": False})
        summary = aggregate([r])
        self.assertIn("unanswerable_detection_rate", summary)
        self.assertAlmostEqual(summary["unanswerable_detection_rate"], 1.0, places=3)

    def test_aggregate_unanswerable_hallucination(self):
        u_pair = next(p for p in CURATED_PAIRS if p.unanswerable)
        # unanswerable_detected=False means model hallucinated instead of saying "not found"
        r = _make_eval_result(u_pair, flags={"unanswerable_detected": False})
        summary = aggregate([r])
        self.assertEqual(summary["unanswerable_hallucination_count"], 1)

    def test_aggregate_toc_artifact(self):
        toc_pair = next(p for p in CURATED_PAIRS if p.toc_artifact_risk)
        r = _make_eval_result(toc_pair, flags={"toc_artifact": True})
        summary = aggregate([r])
        self.assertEqual(summary["toc_artifact_hits"], 1)

    def test_aggregate_empty(self):
        summary = aggregate([])
        self.assertEqual(summary["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
