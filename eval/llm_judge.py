"""LLM-as-judge: structured 1-5 scoring of RAG answers using Ollama.

Dimensions scored (each 1-5):
  correctness   : factual accuracy vs. ground truth
  faithfulness  : answer is grounded in context (no hallucination)
  helpfulness   : usefulness / actionability of the answer
  citation_quality : citation format and traceability

Each dimension gets a score and brief reasoning. Output is a JudgeResult dataclass.
The LLM is asked to return valid JSON; a lenient fallback extracts scores via regex.
"""
from __future__ import annotations

import json
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
class DimensionScore:
    score: int         # 1-5
    reasoning: str


@dataclass
class JudgeResult:
    question: str
    answer: str
    correctness: DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    faithfulness: DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    helpfulness: DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    citation_quality: DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    overall: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "correctness": {"score": self.correctness.score, "reasoning": self.correctness.reasoning},
            "faithfulness": {"score": self.faithfulness.score, "reasoning": self.faithfulness.reasoning},
            "helpfulness": {"score": self.helpfulness.score, "reasoning": self.helpfulness.reasoning},
            "citation_quality": {"score": self.citation_quality.score, "reasoning": self.citation_quality.reasoning},
            "overall": round(self.overall, 3),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert judge evaluating a Retrieval-Augmented Generation (RAG) system \
for legal document Q&A.

Score the ANSWER on four dimensions, each from 1 (very poor) to 5 (excellent):

1. correctness    — Is the answer factually correct compared to the GROUND TRUTH?
2. faithfulness   — Is the answer grounded in the CONTEXT (no hallucination)?
3. helpfulness    — Is the answer clear, complete, and actionable?
4. citation_quality — Are citations present, specific (source+page+clause), and accurate?

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

CONTEXT (retrieved chunks):
{context}

ANSWER:
{answer}

Return ONLY valid JSON in this exact format (no extra text):
{{
  "correctness":     {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "faithfulness":    {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "helpfulness":     {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "citation_quality":{{"score": <1-5>, "reasoning": "<one sentence>"}}
}}
JSON:"""


# ---------------------------------------------------------------------------
# LLM helper — Groq if key is set, else Ollama
# ---------------------------------------------------------------------------

def _ollama(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip()


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
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
    return _ollama(prompt)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_dim(obj: dict, key: str) -> DimensionScore:
    d = obj.get(key, {})
    score = int(d.get("score", 0))
    score = max(1, min(5, score))
    reasoning = str(d.get("reasoning", "")).strip()
    return DimensionScore(score=score, reasoning=reasoning)


def _regex_fallback(raw: str, key: str) -> DimensionScore:
    """Extract score for a dimension via regex when JSON parse fails."""
    pattern = rf'"{key}".*?"score"\s*:\s*(\d)'
    m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
    score = int(m.group(1)) if m else 3  # default to middle
    return DimensionScore(score=score, reasoning="(parsed via regex fallback)")


# ---------------------------------------------------------------------------
# Main judge function
# ---------------------------------------------------------------------------

def judge(
    question: str,
    answer: str,
    ground_truth: str,
    context_chunks: list[str],
    model: str = OLLAMA_MODEL,
) -> JudgeResult:
    """Run the LLM judge and return a JudgeResult.

    Safe: all exceptions are caught; on failure a JudgeResult with score=0 is returned.
    """
    result = JudgeResult(question=question, answer=answer)

    context_str = "\n---\n".join(c[:400] for c in context_chunks[:4])
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth[:400],
        context=context_str,
        answer=answer[:600],
    )

    try:
        raw = _call_llm(prompt)

        # Try strict JSON parse
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(raw[start:end])
                result.correctness     = _parse_dim(obj, "correctness")
                result.faithfulness    = _parse_dim(obj, "faithfulness")
                result.helpfulness     = _parse_dim(obj, "helpfulness")
                result.citation_quality = _parse_dim(obj, "citation_quality")
            except json.JSONDecodeError:
                # Regex fallback per dimension
                result.correctness      = _regex_fallback(raw, "correctness")
                result.faithfulness     = _regex_fallback(raw, "faithfulness")
                result.helpfulness      = _regex_fallback(raw, "helpfulness")
                result.citation_quality = _regex_fallback(raw, "citation_quality")
        else:
            result.error = "LLM returned no JSON block"
            return result

        scores = [
            result.correctness.score,
            result.faithfulness.score,
            result.helpfulness.score,
            result.citation_quality.score,
        ]
        result.overall = sum(scores) / len(scores)

    except Exception as exc:
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Batch judge
# ---------------------------------------------------------------------------

def judge_batch(
    pairs: list[dict],  # list of {question, answer, ground_truth, context_chunks}
    model: str = OLLAMA_MODEL,
    verbose: bool = True,
) -> list[JudgeResult]:
    """Judge a list of QA pairs. Each dict must have keys: question, answer,
    ground_truth, context_chunks (list[str])."""
    results = []
    for i, p in enumerate(pairs, 1):
        if verbose:
            print(f"  Judging [{i}/{len(pairs)}]: {p['question'][:60]}...")
        r = judge(
            question=p["question"],
            answer=p["answer"],
            ground_truth=p["ground_truth"],
            context_chunks=p.get("context_chunks", []),
            model=model,
        )
        results.append(r)
    return results
