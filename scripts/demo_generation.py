"""End-to-end generation demo with citation grounding.

Builds the full RAG pipeline on the real legal PDFs and runs the same
10 queries used in the Phase 2 retrieval demo, this time generating
grounded answers with source/page/clause citations.

Run:
    python scripts/demo_generation.py
    python scripts/demo_generation.py --chunk-size 256 --top-k 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from embeddings.embedder import get_embedder
from vectorstore.faiss_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid import HybridRetriever
from generation.generator import LegalQAChain
from config import RAW_DIR

QUERIES = [
    "What is the employee's annual salary?",
    "How many days notice is required to terminate the employment agreement?",
    "What is the uptime guarantee in the SLA?",
    "What are the SLA credits if uptime falls below 99%?",
    "What is the non-compete duration after termination?",
    "How long does the NDA remain in effect?",
    "What is the monthly fee under the service agreement?",
    "Which state's law governs the employment contract?",
    "What happens to confidential information upon termination?",
    "How many days of paid time off does the employee receive?",
]

SEP  = "-" * 72
SEP2 = "=" * 72

CONF_LABEL = {
    (0.8, 1.1): "HIGH",
    (0.5, 0.8): "MEDIUM",
    (0.0, 0.5): "LOW",
}


def conf_label(score: float) -> str:
    for (lo, hi), label in CONF_LABEL.items():
        if lo <= score < hi:
            return label
    return "?"


def print_result(idx: int, result) -> None:
    print(f"\n{SEP}")
    print(f"[{idx:02d}] QUERY: {result.query}")
    print(SEP)
    print(f"\nANSWER:\n{result.answer}")
    print(f"\nCONFIDENCE: {result.confidence:.2f}  ({conf_label(result.confidence)})")
    if result.citations:
        print(f"\nCITATIONS ({len(result.citations)}):")
        for i, c in enumerate(result.citations, 1):
            clause_preview = c.clause[:120] + ("..." if len(c.clause) > 120 else "")
            print(f"  [{i}] {c.source}  |  Page {c.page}")
            print(f"       \"{clause_preview}\"")
    else:
        print("\nCITATIONS: none parsed (LLM did not follow citation format)")
    print(f"\n  [context chunks used: {result.num_context_chunks}]")


def main(chunk_size: int, top_k: int) -> None:
    print(f"\n{SEP2}")
    print(" RAG Generation Demo  --  Legal Document Q&A with Citation Grounding")
    print(SEP2)

    # 1. Ingest
    t0 = time.time()
    print(f"\n[1/4] Loading PDFs ...")
    docs   = load_pdfs(RAW_DIR)
    chunks = chunk_documents(docs, chunk_size=chunk_size)
    print(f"      {len(docs)} pages -> {len(chunks)} chunks  (chunk_size={chunk_size})")

    # 2. Index
    print(f"\n[2/4] Building FAISS + BM25 index ...")
    embedder  = get_embedder()
    store     = FAISSStore(embedder)
    store.build(chunks)
    bm25      = BM25Retriever(chunks)
    dense     = DenseRetriever(store)
    retriever = HybridRetriever(dense, bm25)
    print(f"      FAISS: {store.size} vectors  |  BM25: {len(chunks)} docs  ({time.time()-t0:.1f}s)")

    # 3. Build generation chain
    print(f"\n[3/4] Connecting to Ollama LLM ...")
    chain = LegalQAChain(retriever=retriever, top_k=top_k)
    print(f"      Chain ready  (top_k={top_k})")

    # 4. Run queries
    print(f"\n[4/4] Running {len(QUERIES)} queries ...\n")
    print(SEP2)

    total_citations = 0
    for idx, query in enumerate(QUERIES, 1):
        t_q = time.time()
        result = chain.run(query)
        elapsed = time.time() - t_q
        total_citations += len(result.citations)
        print_result(idx, result)
        print(f"  [{elapsed:.1f}s]")

    print(f"\n{SEP2}")
    print(f" SUMMARY")
    print(f"  Queries run       : {len(QUERIES)}")
    print(f"  Total citations   : {total_citations}")
    print(f"  Avg citations/q   : {total_citations/len(QUERIES):.1f}")
    print(SEP2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG generation demo")
    parser.add_argument("--chunk-size", type=int, default=512, choices=[256, 512, 1024])
    parser.add_argument("--top-k",      type=int, default=4,
                        help="Context chunks passed to LLM (default: 4)")
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, top_k=args.top_k)
