"""Prompt templates for the legal Q&A generation chain.

Design goals:
  1. Make it structurally impossible to hallucinate — every claim must cite context.
  2. Force a parseable citation format so the backend can extract structured metadata.
  3. Require a confidence score so the UI can colour-code answers.
  4. Include a worked example so smaller local LLMs (llama3.2) follow the format reliably.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 700  # truncate very long chunks to keep local LLM context tight


def format_context(retrieved: List[Tuple[Document, float]]) -> str:
    """Convert (Document, score) pairs into a numbered context string for the LLM.

    Each block clearly labels Source, Page, and Content so the LLM knows exactly
    what to cite.

    Args:
        retrieved: Output of HybridRetriever.retrieve() — (Document, rrf_score) list.

    Returns:
        Multi-block string ready to be injected into the prompt.
    """
    if not retrieved:
        return "(No context passages retrieved.)"

    blocks: List[str] = []
    for i, (doc, score) in enumerate(retrieved, 1):
        source = doc.metadata.get("source", "unknown_source")
        page   = doc.metadata.get("page", "?")
        text   = doc.page_content.strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + " [truncated]"
        blocks.append(
            f"[Block {i}]\n"
            f"Source : {source}\n"
            f"Page   : {page}\n"
            f"Content: {text}"
        )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# System prompt — the core anti-hallucination contract with the LLM
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a meticulous legal document analyst. Your sole function is to answer \
questions using ONLY the context passages provided to you.

=== ABSOLUTE RULES ===
1. NEVER introduce facts, figures, or interpretations that do not appear \
verbatim or near-verbatim in the provided context passages.
2. Every factual claim in your answer MUST be followed immediately by a \
citation in this EXACT format (preserve spacing and punctuation precisely):
   [Source: <filename>, Page: <page_number>, Clause: "<short exact quote>"]
3. If the answer requires information from multiple context blocks, include \
a separate citation for each block used.
4. If the question cannot be answered from the provided context, respond \
ONLY with:
   Not found in the provided documents.
5. Do not say "based on the context" or "according to the document" — \
just state the fact and cite it inline.

=== CONFIDENCE SCORE ===
After your answer (on its own line) write:
Confidence: <decimal between 0.0 and 1.0>
  1.0 = the answer is stated word-for-word in the context
  0.7 = the answer is clearly implied by the context
  0.4 = the answer requires reading between the lines
  0.0 = the context does not address the question

=== WORKED EXAMPLE ===
Question: What is the termination notice period?

Answer:
Either party must provide thirty (30) days written notice to terminate the \
agreement. [Source: employment_contract.pdf, Page: 1, Clause: "Either party \
wishing to terminate this Agreement shall provide thirty (30) days written \
notice to the other party."]

Confidence: 1.0
=== END EXAMPLE ===

Now answer the user's question using the context passages below.\
"""

# ---------------------------------------------------------------------------
# Human turn — injects context and question
# ---------------------------------------------------------------------------

_HUMAN = """\
=== CONTEXT PASSAGES ===
{context}

=== QUESTION ===
{question}

=== YOUR ANSWER ===\
"""

# ---------------------------------------------------------------------------
# Exported prompt
# ---------------------------------------------------------------------------

LEGAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
