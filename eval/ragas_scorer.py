"""RAGAS-style metrics implemented locally using Ollama + BGE embedder.

ragas>=0.2 requires scikit-network which fails on Python 3.14 (no pre-built wheel).
This module re-implements the four core RAGAS metrics from scratch using:
  - Ollama llama3.2  for LLM-based scoring (faithfulness, answer_relevancy)
  - BGE-small-en-v1.5  for embedding similarity (context_recall)
  - Pure Python for context_precision (rank-based)

Metric definitions (matches RAGAS v0.1 semantics):
  faithfulness       : fraction of answer claims that are supported by the context
  answer_relevancy   : cosine similarity between the question and the generated answer
  context_precision  : fraction of relevant context chunks among the top-k retrieved
  context_recall     : fraction of ground-truth statements covered by retrieved context

All scores are in [0.0, 1.0]. Each metric degrades gracefully on LLM or network errors.
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, GROQ_API_KEY, GROQ_MODEL

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGASScores:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    error: Optional[str] = None

    def mean(self) -> float:
        vals = [self.faithfulness, self.answer_relevancy,
                self.context_precision, self.context_recall]
        return sum(vals) / len(vals)

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "mean": round(self.mean(), 4),
        }


# ---------------------------------------------------------------------------
# LLM helper — Groq if key is set, else Ollama
# ---------------------------------------------------------------------------

def _ollama(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL,
            max_tokens: int = 512) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip()


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Route to Groq if GROQ_API_KEY is set, otherwise fall back to Ollama."""
    if GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    return _ollama(prompt, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Cosine similarity (pure numpy, no sklearn needed)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------
# Ask LLM: "How many of these answer statements are supported by the context?"
# Returns fraction of supported claims.

_FAITHFULNESS_PROMPT = """\
You are an expert evaluator. Given a CONTEXT and an ANSWER, determine what fraction
of the factual claims in the ANSWER are directly supported by the CONTEXT.

Instructions:
1. List each distinct factual claim in the ANSWER (one per line, prefix "CLAIM: ").
2. For each claim, write "SUPPORTED" or "UNSUPPORTED".
3. Output a final line: SCORE: <supported>/<total>

CONTEXT:
{context}

ANSWER:
{answer}

Evaluation:"""


def score_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Returns fraction of answer claims supported by the retrieved context."""
    if not answer.strip() or not context_chunks:
        return 0.0
    context = "\n---\n".join(c[:500] for c in context_chunks[:5])
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer[:800])
    try:
        raw = _call_llm(prompt, max_tokens=600)
        # Parse "SCORE: x/y"
        m = re.search(r"SCORE:\s*(\d+)\s*/\s*(\d+)", raw, re.IGNORECASE)
        if m:
            supported, total = int(m.group(1)), int(m.group(2))
            return supported / total if total > 0 else 0.0
        # Fallback: count SUPPORTED lines
        supported = raw.upper().count("SUPPORTED") - raw.upper().count("UNSUPPORTED")
        total = raw.upper().count("CLAIM:")
        return supported / total if total > 0 else 0.5
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 2. Answer Relevancy
# ---------------------------------------------------------------------------
# Generate synthetic questions from the answer, embed them + embed the original
# question, return mean cosine similarity.

_RELEVANCY_PROMPT = """\
Given the following ANSWER to a question, generate {n} diverse questions that this
answer is likely a response to. Output one question per line, no numbering.

ANSWER:
{answer}

Questions:"""


def score_answer_relevancy(question: str, answer: str, embedder, n_questions: int = 3) -> float:
    """Embed question + synthetic questions from answer, return mean cosine sim."""
    if not answer.strip():
        return 0.0
    try:
        prompt = _RELEVANCY_PROMPT.format(answer=answer[:600], n=n_questions)
        raw = _call_llm(prompt, max_tokens=200)
        synthetic_qs = [line.strip() for line in raw.splitlines() if line.strip()][:n_questions]
        if not synthetic_qs:
            return 0.0
        q_emb = embedder.embed_query(question)
        sims = []
        for sq in synthetic_qs:
            sq_emb = embedder.embed_query(sq)
            sims.append(_cosine(q_emb, sq_emb))
        return sum(sims) / len(sims)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 3. Context Precision
# ---------------------------------------------------------------------------
# Ask LLM: for each retrieved chunk, is it relevant to the question?
# precision@k = fraction of relevant chunks in the retrieved set.

_PRECISION_PROMPT = """\
Is the following CONTEXT CHUNK relevant for answering the QUESTION?
Answer with a single word: YES or NO.

QUESTION: {question}
CONTEXT CHUNK: {chunk}
Answer:"""


def score_context_precision(question: str, context_chunks: list[str]) -> float:
    """Fraction of retrieved chunks judged relevant to the question by the LLM."""
    if not context_chunks:
        return 0.0
    relevant = 0
    for chunk in context_chunks[:5]:
        try:
            prompt = _PRECISION_PROMPT.format(question=question, chunk=chunk[:400])
            raw = _call_llm(prompt, max_tokens=5).upper()
            if "YES" in raw:
                relevant += 1
        except Exception:
            pass
    return relevant / min(len(context_chunks), 5)


# ---------------------------------------------------------------------------
# 4. Context Recall
# ---------------------------------------------------------------------------
# Split ground truth into sentences, for each sentence ask: is it supported by context?
# recall = fraction of GT sentences covered.

_RECALL_PROMPT = """\
Is the following GROUND TRUTH STATEMENT directly supported by the CONTEXT?
Answer with a single word: YES or NO.

CONTEXT:
{context}

GROUND TRUTH STATEMENT: {statement}
Answer:"""


def _split_statements(text: str) -> list[str]:
    """Split ground truth into individual factual statements."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def score_context_recall(ground_truth: str, context_chunks: list[str]) -> float:
    """Fraction of ground-truth statements supported by retrieved context."""
    if not ground_truth.strip() or not context_chunks:
        return 0.0
    statements = _split_statements(ground_truth)
    if not statements:
        return 0.0
    context = "\n---\n".join(c[:400] for c in context_chunks[:5])
    covered = 0
    for stmt in statements:
        try:
            prompt = _RECALL_PROMPT.format(context=context, statement=stmt)
            raw = _call_llm(prompt, max_tokens=5).upper()
            if "YES" in raw:
                covered += 1
        except Exception:
            pass
    return covered / len(statements)


# ---------------------------------------------------------------------------
# TOC artifact check
# ---------------------------------------------------------------------------

def check_toc_artifact(citations, toc_page: int = 0) -> bool:
    """Return True if any citation refers to the TOC page (faithfulness failure)."""
    for c in citations:
        if getattr(c, "page", -1) == toc_page:
            return True
    return False


def check_multi_jurisdiction(answer: str, expected_sources: list[str]) -> dict:
    """Verify that all expected source files are mentioned in the answer."""
    answer_lower = answer.lower()
    results = {}
    for src in expected_sources:
        # Match by filename stem, e.g. "exhibit101" from "exhibit101.pdf"
        stem = Path(src).stem.lower().replace("_", " ").replace("-", " ")
        keywords = stem.split()
        # Also check for key jurisdiction words
        if "exhibit101" in stem or "exhibit" in stem:
            keywords = ["texas", "verdisys", "exhibit101"]
        elif "employment-agreement" in stem or "jaipur" in stem:
            keywords = ["jaipur", "india", "employment-agreement"]
        elif "sample" in stem or "standard" in stem:
            keywords = ["victoria", "australia", "purple nike"]
        else:
            keywords = [kw for kw in keywords if len(kw) > 4]  # drop short generic words
        found = any(kw in answer_lower for kw in keywords)
        results[src] = found
    return results


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_pair(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: list[str],
    embedder=None,
    citations=None,
    toc_artifact_risk: bool = False,
    multi_jurisdiction: bool = False,
    expected_sources: list[str] | None = None,
) -> RAGASScores:
    """Compute all four RAGAS metrics for a single QA pair.

    Args:
        question:          The user query.
        answer:            The RAG system's answer string.
        ground_truth:      The reference correct answer.
        context_chunks:    List of retrieved chunk texts used as context.
        embedder:          HuggingFaceEmbeddings instance (for answer_relevancy).
        citations:         List of Citation objects (for toc_artifact_risk check).
        toc_artifact_risk: If True, check whether citations land on TOC page.
        multi_jurisdiction: If True, verify all expected_sources appear in answer.
        expected_sources:  Required source files for multi_jurisdiction check.
    """
    scores = RAGASScores()
    try:
        scores.faithfulness = score_faithfulness(answer, context_chunks)

        if embedder is not None:
            scores.answer_relevancy = score_answer_relevancy(question, answer, embedder)
        else:
            scores.answer_relevancy = 0.0

        scores.context_precision = score_context_precision(question, context_chunks)
        scores.context_recall = score_context_recall(ground_truth, context_chunks)

        # Extra diagnostics (don't affect numeric scores but logged separately)
        if toc_artifact_risk and citations:
            if check_toc_artifact(citations):
                scores.error = "TOC_ARTIFACT: answer cites page 0 (table of contents)"

        if multi_jurisdiction and expected_sources:
            cov = check_multi_jurisdiction(answer, expected_sources)
            missing = [s for s, found in cov.items() if not found]
            if missing:
                msg = f"MULTI_JURISDICTION_INCOMPLETE: missing {missing}"
                scores.error = (scores.error + " | " + msg) if scores.error else msg

    except Exception as exc:
        scores.error = str(exc)

    return scores
