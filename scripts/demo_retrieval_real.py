"""Retrieval demo on real documents — 10 cross-document queries."""
import sys
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
    "What is the annual salary?",
    "What is the termination notice period?",
    "What is the non-compete duration?",
    "What governing law applies?",
    "What is the severance pay?",
    "What are the confidentiality obligations after termination?",
    "What is the probationary period?",
    "What jurisdiction handles disputes?",
    "What are the vacation or leave entitlements?",
    "What happens to intellectual property created during employment?",
]

print("Building index on real documents...")
docs     = load_pdfs(RAW_DIR)
chunks   = chunk_documents(docs, chunk_size=512)
embedder = get_embedder()
store    = FAISSStore(embedder)
store.build(chunks)
hybrid   = HybridRetriever(DenseRetriever(store), BM25Retriever(chunks))

sources = sorted({d.metadata["source"] for d in docs})
print(f"\nIndexed {len(docs)} pages / {len(chunks)} chunks from {len(sources)} files:")
for s in sources:
    n = sum(1 for d in docs if d.metadata["source"] == s)
    print(f"  {s} ({n} pages)")

print("\n" + "="*72)
print(" RETRIEVAL RESULTS (top-3 per query, chunk_size=512)")
print("="*72)

for qi, query in enumerate(QUERIES, 1):
    results = hybrid.retrieve(query, k=3)
    print(f"\n[{qi:02d}] {query}")
    print("-"*72)
    for rank, (doc, score) in enumerate(results, 1):
        src   = doc.metadata["source"]
        page  = doc.metadata["page"]
        cidx  = doc.metadata["chunk_index"]
        text  = doc.page_content.strip().replace("\n", " ")
        snip  = text[:280] + ("..." if len(text) > 280 else "")
        print(f"  [{rank}] score={score:.4f} | {src} | p.{page} | chunk#{cidx}")
        print(f"       {snip}")

print("\n" + "="*72)
print(" Done.")
