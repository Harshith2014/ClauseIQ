"""Orchestrate a full evaluation run.

Usage:
    python eval/run_eval.py                  # full run (all golden pairs)
    python eval/run_eval.py --sample 5       # smoke test on 5 pairs
    python eval/run_eval.py --chunk-size 256 # different chunk size
    python eval/run_eval.py --no-judge       # skip LLM judge (faster)
    python eval/run_eval.py --save-csv       # append metrics to runs.csv

Pipeline per question:
  1. Retrieve context via HybridRetriever
  2. Generate answer via LegalQAChain
  3. RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  4. LLM judge (correctness, faithfulness, helpfulness, citation_quality)
  5. Special checks: TOC artifact, multi-jurisdiction completeness, unanswerable detection

Output: summary printed to console + optionally appended to runs.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K,
    GOLDEN_DATASET_PATH, RUNS_CSV_PATH, OLLAMA_MODEL,
)
from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from embeddings.embedder import get_embedder
from vectorstore.faiss_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid import HybridRetriever
from generation.generator import LegalQAChain
from eval.golden_dataset import load_golden, CURATED_PAIRS, GoldenPair
from eval.ragas_scorer import score_pair, RAGASScores, check_toc_artifact, check_multi_jurisdiction
from eval.llm_judge import judge, JudgeResult
from logging_.csv_logger import make_run_id, log_run
from logging_.langsmith_logger import LangSmithTracer, tracing_available


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class EvalResult:
    def __init__(
        self,
        pair: GoldenPair,
        answer: str,
        confidence: float,
        context_chunks: list[str],
        citations,
        ragas: RAGASScores,
        judge_result: JudgeResult | None,
        elapsed: float,
        flags: dict,
    ):
        self.pair = pair
        self.answer = answer
        self.confidence = confidence
        self.context_chunks = context_chunks
        self.citations = citations
        self.ragas = ragas
        self.judge_result = judge_result
        self.elapsed = elapsed
        self.flags = flags  # {"toc_artifact": bool, "unanswerable_detected": bool, ...}


# ---------------------------------------------------------------------------
# Unanswerable detection
# ---------------------------------------------------------------------------

_NOT_FOUND_PHRASES = [
    "not found", "not specified", "not mentioned", "not provided",
    "cannot determine", "no information", "does not contain",
    "not available", "not stated", "not indicated", "not addressed",
    "i cannot", "i don't", "not in the", "outside the scope",
    "blank", "template", "left blank",
]


def is_not_found_answer(answer: str) -> bool:
    al = answer.lower()
    return any(phrase in al for phrase in _NOT_FOUND_PHRASES)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_pipeline(chunk_size: int = DEFAULT_CHUNK_SIZE, top_k: int = DEFAULT_TOP_K):
    print(f"Building pipeline (chunk_size={chunk_size}, top_k={top_k})...")
    docs = load_pdfs(RAW_DIR)
    chunks = chunk_documents(docs, chunk_size=chunk_size)
    embedder = get_embedder()
    store = FAISSStore(embedder)
    store.build(chunks)
    retriever = HybridRetriever(DenseRetriever(store), BM25Retriever(chunks))
    chain = LegalQAChain(retriever=retriever, top_k=top_k)
    print(f"  {len(docs)} pages, {len(chunks)} chunks from {len({d.metadata['source'] for d in docs})} files")
    return chain, embedder


# ---------------------------------------------------------------------------
# Run single pair
# ---------------------------------------------------------------------------

def run_pair(
    pair: GoldenPair,
    chain: LegalQAChain,
    embedder,
    run_judge: bool = True,
) -> EvalResult:
    t0 = time.time()

    # --- Retrieve full chunks (used by RAGAS scorer) ---
    try:
        retrieved = chain.retriever.retrieve(pair.question, k=chain.top_k)
        context_chunks = [doc.page_content for doc, _ in retrieved]
    except Exception:
        context_chunks = []

    # --- Generate ---
    try:
        gen = chain.run(pair.question)
        answer = gen.answer
        confidence = gen.confidence
        citations = gen.citations
    except Exception as exc:
        answer = f"[ERROR: {exc}]"
        confidence = 0.0
        citations = []

    elapsed = time.time() - t0

    # --- RAGAS ---
    ragas = score_pair(
        question=pair.question,
        answer=answer,
        ground_truth=pair.ground_truth,
        context_chunks=context_chunks,
        embedder=embedder,
        citations=citations,
        toc_artifact_risk=pair.toc_artifact_risk,
        multi_jurisdiction=pair.multi_jurisdiction,
        expected_sources=pair.expected_sources if pair.multi_jurisdiction else None,
    )

    # --- LLM Judge ---
    judge_result = None
    if run_judge:
        judge_result = judge(
            question=pair.question,
            answer=answer,
            ground_truth=pair.ground_truth,
            context_chunks=context_chunks,
        )

    # --- Special flags ---
    flags: dict = {}

    if pair.unanswerable:
        # Mutually exclusive: only one flag, True = system correctly said "not found"
        flags["unanswerable_detected"] = is_not_found_answer(answer)

    if pair.toc_artifact_risk and citations:
        flags["toc_artifact"] = check_toc_artifact(citations)

    if pair.multi_jurisdiction and pair.expected_sources:
        cov = check_multi_jurisdiction(answer, pair.expected_sources)
        flags["multi_jurisdiction_coverage"] = cov
        flags["multi_jurisdiction_complete"] = all(cov.values())

    return EvalResult(
        pair=pair,
        answer=answer,
        confidence=confidence,
        context_chunks=context_chunks,
        citations=citations,
        ragas=ragas,
        judge_result=judge_result,
        elapsed=elapsed,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(results: list[EvalResult]) -> dict:
    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    ragas_fields = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    judge_fields = ["correctness", "faithfulness", "helpfulness", "citation_quality"]

    summary = {
        "n": len(results),
        "elapsed_avg": avg([r.elapsed for r in results]),
    }

    # RAGAS averages
    for f in ragas_fields:
        summary[f"ragas_{f}"] = avg([getattr(r.ragas, f) for r in results])

    # Judge averages
    judge_results = [r.judge_result for r in results if r.judge_result is not None]
    if judge_results:
        for f in judge_fields:
            vals = [getattr(jr, f).score for jr in judge_results if getattr(jr, f).score > 0]
            summary[f"judge_{f}"] = avg(vals)
        summary["judge_overall"] = avg([jr.overall for jr in judge_results if jr.overall > 0])

    # Special checks
    unanswerable = [r for r in results if r.pair.unanswerable]
    if unanswerable:
        detected = sum(1 for r in unanswerable if r.flags.get("unanswerable_detected"))
        summary["unanswerable_detection_rate"] = round(detected / len(unanswerable), 4)
        summary["unanswerable_hallucination_count"] = len(unanswerable) - detected

    toc_risk = [r for r in results if r.pair.toc_artifact_risk]
    if toc_risk:
        artifacts = sum(1 for r in toc_risk if r.flags.get("toc_artifact"))
        summary["toc_artifact_hits"] = artifacts

    multi_j = [r for r in results if r.pair.multi_jurisdiction]
    if multi_j:
        complete = sum(1 for r in multi_j if r.flags.get("multi_jurisdiction_complete"))
        summary["multi_jurisdiction_complete_rate"] = round(complete / len(multi_j), 4)

    return summary


# ---------------------------------------------------------------------------
# CSV logging — delegated to logging_.csv_logger
# ---------------------------------------------------------------------------

def append_to_csv(run_id: str, config: dict, summary: dict) -> None:
    log_run(run_id, config, summary)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

# Flags where True = good outcome (invert the [!]/[ok] indicator)
_POSITIVE_FLAGS = frozenset({"unanswerable_detected", "multi_jurisdiction_complete"})


def print_results(results: list[EvalResult], summary: dict) -> None:
    SEP = "-" * 72
    print("\n" + "=" * 72)
    print(" EVAL RESULTS")
    print("=" * 72)

    for r in results:
        flags_str = ""
        if r.flags:
            flag_parts = []
            for k, v in r.flags.items():
                if isinstance(v, bool):
                    bad = (not v) if k in _POSITIVE_FLAGS else v
                    flag_parts.append(f"{'[!]' if bad else '[ok]'} {k}")
                elif isinstance(v, dict):
                    missing = [s for s, found in v.items() if not found]
                    if missing:
                        flag_parts.append(f"[!] missing: {[Path(m).stem for m in missing]}")
            flags_str = "  FLAGS: " + " | ".join(flag_parts) if flag_parts else ""

        print(f"\n[{r.pair.id}] {r.pair.question[:65]}")
        print(f"  Answer: {r.answer[:120].replace(chr(10), ' ')}{'...' if len(r.answer) > 120 else ''}")
        ragas_d = r.ragas.as_dict()
        print(f"  RAGAS: faith={ragas_d['faithfulness']:.3f} | "
              f"rel={ragas_d['answer_relevancy']:.3f} | "
              f"prec={ragas_d['context_precision']:.3f} | "
              f"rec={ragas_d['context_recall']:.3f} | "
              f"mean={ragas_d['mean']:.3f}")
        if r.judge_result and r.judge_result.overall > 0:
            jr = r.judge_result
            print(f"  Judge: corr={jr.correctness.score} faith={jr.faithfulness.score} "
                  f"help={jr.helpfulness.score} cite={jr.citation_quality.score} "
                  f"overall={jr.overall:.2f}/5")
        if r.ragas.error:
            print(f"  RAGAS error: {r.ragas.error}")
        if flags_str:
            print(flags_str)

    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k:<40} {v}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument("--sample", type=int, default=0,
                        help="Evaluate only N pairs (0 = all)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (faster run)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Append metrics to runs.csv")
    parser.add_argument("--no-llm-golden", action="store_true",
                        help="Use only curated pairs (skip LLM-generated)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override run ID (default: timestamp-based)")
    args = parser.parse_args()

    run_id = make_run_id(args.run_id)
    config = {
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "model": OLLAMA_MODEL,
        "run_judge": not args.no_judge,
    }

    # Load golden dataset
    if GOLDEN_DATASET_PATH.exists():
        pairs = load_golden()
    else:
        pairs = list(CURATED_PAIRS)
        print(f"golden.json not found — using {len(pairs)} curated pairs")

    if args.no_llm_golden:
        pairs = [p for p in pairs if not p.id.startswith("llm")]

    if args.sample > 0:
        # Include at least one unanswerable + one multi-jurisdiction for coverage
        special = [p for p in pairs if p.unanswerable or p.multi_jurisdiction or p.toc_artifact_risk]
        normal = [p for p in pairs if p not in special]
        keep_special = special[:min(3, len(special))]
        keep_normal = normal[:max(0, args.sample - len(keep_special))]
        pairs = keep_special + keep_normal
        print(f"Sample mode: {len(pairs)} pairs ({len(keep_special)} special)")

    # Build pipeline
    chain, embedder = build_pipeline(args.chunk_size, args.top_k)

    # LangSmith tracing (no-op if key absent)
    if tracing_available():
        print(f"[LangSmith] Tracing available — will activate for run {run_id}")
    else:
        print("[LangSmith] No API key — tracing disabled (set LANGSMITH_API_KEY in .env to enable)")

    # Run evaluation (inside LangSmith context so all chain calls are traced)
    print(f"\nEvaluating {len(pairs)} pairs (run_id={run_id})...")
    results = []
    with LangSmithTracer(run_id=run_id) as tracer:
        for i, pair in enumerate(pairs, 1):
            print(f"  [{i}/{len(pairs)}] {pair.id}: {pair.question[:55]}...")
            r = run_pair(pair, chain, embedder, run_judge=not args.no_judge)
            results.append(r)

    # Aggregate
    summary = aggregate(results)
    summary["run_id"] = run_id

    # Print
    print_results(results, summary)

    # CSV
    if args.save_csv:
        append_to_csv(run_id, config, summary)

    # JSON dump for logging_/langsmith_logger.py
    out_path = Path(__file__).parent.parent / "data" / f"eval_{run_id}.json"
    out_data = []
    for r in results:
        out_data.append({
            "pair_id": r.pair.id,
            "question": r.pair.question,
            "answer": r.answer,
            "confidence": r.confidence,
            "ragas": r.ragas.as_dict(),
            "judge": r.judge_result.as_dict() if r.judge_result else None,
            "flags": r.flags,
            "elapsed": round(r.elapsed, 2),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "config": config, "summary": summary, "results": out_data},
                  f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
