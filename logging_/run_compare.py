"""Run comparison: print a metric delta table between two eval runs.

Usage:
    python logging_/run_compare.py run_001 run_002
    python logging_/run_compare.py                    # lists all runs

Delta format:
    metric          run_A     run_B    delta    change
    -------------------------------------------------------
    faithfulness    0.410     0.620   +0.210   +51.2%  (++)
    answer_relevancy 0.716    0.698   -0.018    -2.5%  (-)

Change symbols:
    (++) >= +5%    (+) < +5%    (--) <= -5%    (-) < -5%    (=) no change
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_.csv_logger import load_runs, list_runs

# Metrics to compare (in display order)
_NUMERIC_METRICS = [
    ("ragas_faithfulness",              "RAGAS faithfulness"),
    ("ragas_answer_relevancy",          "RAGAS answer relevancy"),
    ("ragas_context_precision",         "RAGAS context precision"),
    ("ragas_context_recall",            "RAGAS context recall"),
    ("judge_correctness",               "Judge correctness (1-5)"),
    ("judge_faithfulness",              "Judge faithfulness (1-5)"),
    ("judge_helpfulness",               "Judge helpfulness (1-5)"),
    ("judge_citation_quality",          "Judge citation quality (1-5)"),
    ("judge_overall",                   "Judge overall (1-5)"),
    ("unanswerable_detection_rate",     "Unanswerable detection rate"),
    ("toc_artifact_hits",               "TOC artifact hits"),
    ("multi_jurisdiction_complete_rate","Multi-jurisdiction complete rate"),
    ("elapsed_avg",                     "Avg latency (s)"),
]

_CONFIG_KEYS = ["chunk_size", "top_k", "model", "n"]


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _change_symbol(pct: float) -> str:
    if abs(pct) < 0.5:
        return "(=)"
    elif pct >= 5.0:
        return "(++)"
    elif pct > 0:
        return "(+)"
    elif pct <= -5.0:
        return "(--)"
    else:
        return "(-)"


def compare_runs(run_id_a: str, run_id_b: str) -> None:
    """Load two runs from runs.csv and print a side-by-side delta table."""
    rows = load_runs()
    if not rows:
        print("runs.csv is empty or does not exist.")
        return

    index = {r["run_id"]: r for r in rows}

    missing = [rid for rid in (run_id_a, run_id_b) if rid not in index]
    if missing:
        print(f"Run(s) not found in runs.csv: {missing}")
        print("Available runs:")
        list_runs()
        return

    a = index[run_id_a]
    b = index[run_id_b]

    # Header
    print("\n" + "=" * 72)
    print(f"  RUN COMPARISON:  {run_id_a}  vs  {run_id_b}")
    print("=" * 72)

    # Config section
    print(f"\n{'Config':<35} {'Run A':>10} {'Run B':>10}")
    print("-" * 57)
    for key in _CONFIG_KEYS:
        va = a.get(key, "—")
        vb = b.get(key, "—")
        flag = " <-- changed" if va != vb else ""
        print(f"  {key:<33} {str(va):>10} {str(vb):>10}{flag}")

    # Metrics delta section
    print(f"\n{'Metric':<35} {'Run A':>8} {'Run B':>8} {'Delta':>8} {'Change':>8}  Sig")
    print("-" * 72)

    for col, label in _NUMERIC_METRICS:
        va = _safe_float(a.get(col))
        vb = _safe_float(b.get(col))
        if va is None and vb is None:
            continue
        va_s = f"{va:.3f}" if va is not None else "—"
        vb_s = f"{vb:.3f}" if vb is not None else "—"

        if va is not None and vb is not None:
            delta = vb - va
            if va != 0:
                pct = (delta / abs(va)) * 100
            else:
                pct = 100.0 if delta > 0 else 0.0
            delta_s = f"{delta:+.3f}"
            pct_s   = f"{pct:+.1f}%"
            sym     = _change_symbol(pct)
        else:
            delta_s = "—"
            pct_s   = "—"
            sym     = ""

        print(f"  {label:<33} {va_s:>8} {vb_s:>8} {delta_s:>8} {pct_s:>8}  {sym}")

    print("\n" + "=" * 72)
    print("  Significance: (++) >=+5%  (+) >0%  (=) ~0%  (-) <0%  (--) <=-5%")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        compare_runs(args[0], args[1])
    elif len(args) == 0:
        print("All logged runs:")
        list_runs()
        print("\nUsage: python logging_/run_compare.py <run_id_a> <run_id_b>")
    else:
        print("Usage: python logging_/run_compare.py <run_id_a> <run_id_b>")
        sys.exit(1)
