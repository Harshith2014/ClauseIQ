"""LangSmith tracing wrapper.

Activates LangSmith tracing for the duration of an eval run when
LANGSMITH_API_KEY is set in the environment.  Degrades to a no-op
silently if the key is absent or the SDK is not installed.

Usage:
    from logging_.langsmith_logger import LangSmithTracer

    with LangSmithTracer(run_id="run_001") as tracer:
        # all LangChain chain.invoke() calls inside this block
        # are automatically traced if LANGSMITH_API_KEY is set
        result = chain.run(query)

    # check whether tracing was active
    if tracer.active:
        print(f"Trace URL: {tracer.project_url}")
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGCHAIN_TRACING_V2


class LangSmithTracer:
    """Context manager that enables LangSmith tracing when a key is available.

    Tracing is activated by setting the three LangChain environment variables
    that the LangChain SDK reads automatically:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<key>
        LANGCHAIN_PROJECT=<project>

    On __exit__ the original env values are restored so the context manager
    is safe to nest and to use in test suites.
    """

    def __init__(self, run_id: Optional[str] = None, project: Optional[str] = None):
        self.run_id = run_id
        self.project = project or LANGSMITH_PROJECT
        self.active = False
        self.project_url: Optional[str] = None
        self._saved_env: dict[str, Optional[str]] = {}

    def __enter__(self) -> "LangSmithTracer":
        key = LANGSMITH_API_KEY or os.getenv("LANGCHAIN_API_KEY", "")
        if not key:
            # No key — tracing silently disabled
            return self

        try:
            # Verify the SDK is importable (langsmith is a dep of langchain)
            import langsmith  # noqa: F401
        except ImportError:
            print("[LangSmithTracer] langsmith SDK not installed — tracing disabled.")
            return self

        # Save current env state and inject tracing vars
        _vars = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": key,
            "LANGCHAIN_PROJECT": self.project,
        }
        for k, v in _vars.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v

        self.active = True
        self.project_url = f"https://smith.langchain.com/projects/{self.project}"
        print(f"[LangSmithTracer] Tracing ENABLED — project: {self.project}")
        if self.run_id:
            print(f"[LangSmithTracer] Run ID: {self.run_id}")
        return self

    def __exit__(self, *_) -> None:
        if not self.active:
            return
        # Restore original env values
        for k, original in self._saved_env.items():
            if original is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = original
        self.active = False


# ---------------------------------------------------------------------------
# Convenience: check-only helper (used by run_eval.py for status reporting)
# ---------------------------------------------------------------------------

def tracing_available() -> bool:
    """Return True if LangSmith credentials are configured."""
    key = LANGSMITH_API_KEY or os.getenv("LANGCHAIN_API_KEY", "")
    return bool(key)


if __name__ == "__main__":
    if tracing_available():
        print(f"LangSmith key found — project: {LANGSMITH_PROJECT}")
        print("Set LANGCHAIN_TRACING_V2=true in .env to enable tracing.")
    else:
        print("No LANGSMITH_API_KEY set. Add it to .env to enable tracing.")
        print("  LANGSMITH_API_KEY=ls__...")
        print("  LANGSMITH_PROJECT=clauseiq-rag")
