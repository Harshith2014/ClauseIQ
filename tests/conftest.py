"""Session-scoped fixtures shared across all test modules.

Ensures sample PDFs exist before any test that needs them runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root and scripts/ importable
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))

import pytest

from config import RAW_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_docs():
    """Create sample PDFs if data/raw/ is empty."""
    if not any(RAW_DIR.glob("*.pdf")):
        from create_sample_docs import create_all_sample_docs
        create_all_sample_docs()
    yield
