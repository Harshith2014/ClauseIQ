"""End-to-end retrieval smoke test.

Loads the real PDFs → chunks → embeds → FAISS + BM25 → hybrid RRF.
Runs a set of sample legal queries and prints ranked results.

Run from project root:
    python scripts/demo_retrieval.py
    python scripts/demo_retrieval.py --chunk-size 256
    python scripts/demo_retrieval.py --chunk-size 1024 --top-k 3
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

SEP = "-" * 72


def print_results(query: str, results: list, chunk_size: int) -> None:
    print(f"\n{SEP}")
    print(f"QUERY  : {query}")
    print(f"RESULTS: (chunk_size={chunk_size})")
    print(SEP)
    for rank, (doc, score) in enumerate(results, 1):
        src   = doc.metadata.get("source", "?")
        page  = doc.metadata.get("page", "?")
        cidx  = doc.metadata.get("chunk_index", "?")
        text  = doc.page_content.strip().replace("\n", " ")
        snippet = text[:220] + ("…" if len(text) > 220 else "")
        print(f"  [{rank}] score={score:.4f}  |  {src}  p.{page}  chunk#{cidx}")
        print(f"       {snippet}")


def main(chunk_size: int, top_k: int) -> None:
    print(f"\n{'='*72}")
    print(" RAG Retrieval Demo — Legal Document Q&A")
    print(f"{'='*72}")

    # 1. Load PDFs
    t0 = time.time()
    print(f"\n[1/4] Loading PDFs from {RAW_DIR} ...")
    docs = load_pdfs(RAW_DIR)
    print(f"      Loaded {len(docs)} pages from {len({d.metadata['source'] for d in docs})} files  ({time.time()-t0:.1f}s)")

    # 2. Chunk
    t0 = time.time()
    print(f"\n[2/4] Chunking  (chunk_size={chunk_size}) ...")
    chunks = chunk_documents(docs, chunk_size=chunk_size)
    print(f"      {len(chunks)} chunks produced  ({time.time()-t0:.1f}s)")

    # 3. Embed + build FAISS
    t0 = time.time()
    print(f"\n[3/4] Building FAISS index (embedding {len(chunks)} chunks) ...")
    embedder = get_embedder()
    store = FAISSStore(embedder)
    store.build(chunks)
    print(f"      Index ready — {store.size} vectors, dim=384  ({time.time()-t0:.1f}s)")

    # 4. Build BM25 + HybridRetriever
    print(f"\n[4/4] Building BM25 index ...")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(store)
    hybrid = HybridRetriever(dense, bm25)
    print(f"      Ready.\n")

    # Run queries
    print(f"\n{'='*72}")
    print(f" Running {len(QUERIES)} sample queries  (top_k={top_k})")
    print(f"{'='*72}")

    for query in QUERIES:
        results = hybrid.retrieve(query, k=top_k)
        print_results(query, results, chunk_size)

    print(f"\n{SEP}")
    print(" Done. All queries complete.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG retrieval demo")
    parser.add_argument("--chunk-size", type=int, default=512,
                        choices=[256, 512, 1024], help="Chunk size (default: 512)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Results per query (default: 2)")
    args = parser.parse_args()
    main(chunk_size=args.chunk_size, top_k=args.top_k)
