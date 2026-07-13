"""Legal Q&A generation chain with citation grounding.

Architecture:
    query
      -> HybridRetriever.retrieve()        [Phase 2]
      -> format_context()                  [prompts.py]
      -> LEGAL_QA_PROMPT | ChatOllama      [LCEL chain]
      -> StrOutputParser
      -> parse_citations() + parse_confidence()
      -> GenerationResult

Output contract:
    GenerationResult.answer      — full LLM answer with inline [Source:...] markers
    GenerationResult.citations   — parsed list of Citation dataclasses
    GenerationResult.confidence  — float 0.0 – 1.0
    GenerationResult.raw_response — unmodified LLM output (for debugging / logging)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from config import DEFAULT_TOP_K, OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL
from generation.prompts import LEGAL_QA_PROMPT, format_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """One grounded citation extracted from the LLM's response."""
    source: str
    page: int | str
    clause: str

    def __str__(self) -> str:
        return f'[Source: {self.source}, Page: {self.page}, Clause: "{self.clause}"]'


@dataclass
class GenerationResult:
    """Structured output from LegalQAChain.run()."""
    query:              str
    answer:             str          # LLM answer, citations stripped to end
    citations:          List[Citation]
    confidence:         float        # 0.0 – 1.0
    raw_response:       str          # unmodified LLM output
    num_context_chunks: int


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Tolerant regex: handles extra whitespace, single/double quotes around clause
_CITATION_RE = re.compile(
    r'\[Source:\s*([^\],]+?)\s*,\s*Page:\s*(\w+)\s*,\s*Clause:\s*["\u201c]([^"\u201d\]]+)["\u201d]\]',
    re.IGNORECASE,
)

_CONFIDENCE_RE = re.compile(
    r'Confidence:\s*([01](?:\.\d+)?)',
    re.IGNORECASE,
)

# Fallback lenient confidence: captures any decimal the model may write
_CONFIDENCE_LENIENT_RE = re.compile(
    r'[Cc]onfidence[:\s]+(\d(?:\.\d+)?)',
)


def parse_citations(text: str) -> List[Citation]:
    """Extract all [Source:..., Page:..., Clause:"..."] markers from *text*.

    Deduplicates by (source, page, first-50-chars-of-clause).
    """
    seen:    set  = set()
    results: List[Citation] = []

    for m in _CITATION_RE.finditer(text):
        source = m.group(1).strip()
        page_s = m.group(2).strip()
        clause = m.group(3).strip()
        key    = (source.lower(), page_s, clause[:50].lower())
        if key in seen:
            continue
        seen.add(key)
        try:
            page: int | str = int(page_s)
        except ValueError:
            page = page_s
        results.append(Citation(source=source, page=page, clause=clause))

    return results


def parse_confidence(text: str) -> float:
    """Extract the confidence score from the LLM's output.

    Falls back to 0.5 (neutral) if no score is found.
    """
    for pattern in (_CONFIDENCE_RE, _CONFIDENCE_LENIENT_RE):
        m = pattern.search(text)
        if m:
            try:
                return round(min(1.0, max(0.0, float(m.group(1)))), 2)
            except ValueError:
                pass
    return 0.5


def clean_answer(raw: str) -> str:
    """Return *raw* with the trailing 'Confidence: X.X' line removed."""
    cleaned = re.sub(
        r'\s*[Cc]onfidence:\s*[01](?:\.\d+)?\s*$',
        '',
        raw,
        flags=re.MULTILINE,
    ).rstrip()
    return cleaned


# ---------------------------------------------------------------------------
# Main chain
# ---------------------------------------------------------------------------

class LegalQAChain:
    """LCEL-based generation chain: retrieve → prompt → LLM → parse.

    Args:
        retriever:  HybridRetriever (or any object with .retrieve(query, k) method).
        top_k:      Number of context chunks to pass to the LLM.
        llm:        LangChain chat model. Defaults to ChatOllama(OLLAMA_MODEL).
                    Pass any compatible LLM (e.g. a mock) for testing.
    """

    def __init__(self, retriever, top_k: int = DEFAULT_TOP_K, llm=None) -> None:
        self.retriever = retriever
        self.top_k     = top_k

        if llm is None:
            if GROQ_API_KEY:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model=GROQ_MODEL,
                    api_key=GROQ_API_KEY,
                    temperature=0.0,
                    max_tokens=1024,
                )
            else:
                llm = ChatOllama(
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.0,      # fully deterministic — critical for legal Q&A
                    num_predict=1024,     # cap tokens to keep local model snappy
                )
        self.llm = llm
        # LCEL chain: prompt | llm | string parser
        self._chain = LEGAL_QA_PROMPT | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> GenerationResult:
        """Answer *query* with grounded citations.

        Returns:
            GenerationResult with structured answer, citations, and confidence.
        """
        logger.info("LegalQAChain.run — query=%r", query[:80])

        # 1. Hybrid retrieval
        retrieved = self.retriever.retrieve(query, k=self.top_k)
        logger.debug("Retrieved %d chunks.", len(retrieved))

        # 2. Format context for the prompt
        context_str = format_context(retrieved)

        # 3. LLM call via LCEL chain
        raw_response: str = self._chain.invoke({
            "context":  context_str,
            "question": query,
        })
        logger.debug("Raw LLM response length: %d chars.", len(raw_response))

        # 4. Parse structured output
        citations  = parse_citations(raw_response)
        confidence = parse_confidence(raw_response)
        answer     = clean_answer(raw_response)

        if not citations:
            logger.warning("No citations parsed from LLM response for query=%r", query[:60])

        return GenerationResult(
            query              = query,
            answer             = answer,
            citations          = citations,
            confidence         = confidence,
            raw_response       = raw_response,
            num_context_chunks = len(retrieved),
        )
