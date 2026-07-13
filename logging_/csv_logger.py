"""CSV run logger.

Appends one row per eval run to runs.csv.
Each row: run_id, timestamp, config (chunk_size, top_k, model),
and all RAGAS + judge metric averages.

Usage:
    from logging_.csv_logger import make_run_id, log_run, load_runs
    run_id = make_run_id()           # "run_20260624_143022"
    run_id = make_run_id("run_001")  # explicit override
    log_run(run_id, config, summary)
    df_like = load_runs()            # list[dict] of all rows
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RUNS_CSV_PATH

# Ordered column groups — makes the CSV human-readable when opened in Excel
_FIXED_COLS = ["run_id", "timestamp", "chunk_size", "top_k", "model", "run_judge", "n", "elapsed_avg"]
_RAGAS_COLS = ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
               "ragas_context_recall"]
_JUDGE_COLS = ["judge_correctness", "judge_faithfulness", "judge_helpfulness",
               "judge_citation_quality", "judge_overall"]
_SPECIAL_COLS = ["unanswerable_detection_rate", "unanswerable_hallucination_count",
                 "toc_artifact_hits", "multi_jurisdiction_complete_rate"]

ALL_COLS = _FIXED_COLS + _RAGAS_COLS + _JUDGE_COLS + _SPECIAL_COLS


def make_run_id(override: Optional[str] = None) -> str:
    """Return a timestamp-based run_id, or *override* if provided."""
    if override:
        return override
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def log_run(run_id: str, config: dict, summary: dict) -> None:
    """Append one row to runs.csv.

    Creates the file with a header row on first call.
    Unknown columns in config/summary are appended after ALL_COLS.
    """
    # Build the row: known cols first, then any extras
    all_keys = dict.fromkeys(ALL_COLS)  # ordered set
    all_keys.update({k: None for k in config})
    all_keys.update({k: None for k in summary})
    fieldnames = list(all_keys.keys())

    row: dict = {"run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds")}
    row.update(config)
    row.update(summary)

    write_header = not RUNS_CSV_PATH.exists()

    with open(RUNS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[csv_logger] Logged run '{run_id}' to {RUNS_CSV_PATH}")


def load_runs() -> list[dict]:
    """Load all rows from runs.csv as a list of dicts.

    Returns [] if the file does not exist.
    """
    if not RUNS_CSV_PATH.exists():
        return []
    with open(RUNS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def list_runs() -> None:
    """Print a summary table of all logged runs."""
    rows = load_runs()
    if not rows:
        print("No runs logged yet. Run: python eval/run_eval.py --save-csv")
        return
    header = f"{'run_id':<28} {'timestamp':<22} {'n':>4} {'faith':>6} {'rel':>6} {'prec':>6} {'rec':>6} {'judge':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('run_id','?'):<28} "
            f"{r.get('timestamp','?'):<22} "
            f"{r.get('n','?'):>4} "
            f"{float(r.get('ragas_faithfulness', 0) or 0):>6.3f} "
            f"{float(r.get('ragas_answer_relevancy', 0) or 0):>6.3f} "
            f"{float(r.get('ragas_context_precision', 0) or 0):>6.3f} "
            f"{float(r.get('ragas_context_recall', 0) or 0):>6.3f} "
            f"{float(r.get('judge_overall', 0) or 0):>6.2f}"
        )


if __name__ == "__main__":
    list_runs()
