"""Root conftest.py — adds project root to sys.path for all test sessions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
