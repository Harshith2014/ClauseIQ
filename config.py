"""Central configuration for the RAG + Eval system.

All tuneable knobs live here. Import this module everywhere instead of
hard-coding values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
MODELS_CACHE_DIR = ROOT_DIR / ".model_cache"

# Create directories on import so nothing downstream needs to worry about it
for _d in [RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR, MODELS_CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM: int = 384  # fixed for BGE-small-en-v1.5

# ---------------------------------------------------------------------------
# LLM (Ollama — local dev default)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# ---------------------------------------------------------------------------
# LLM (Groq — used when GROQ_API_KEY is set; takes priority over Ollama)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZES: list[int] = [256, 512, 1024]
DEFAULT_CHUNK_SIZE: int = 512
CHUNK_OVERLAP_PCT: float = 0.10  # 10 % of chunk_size

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5
RRF_K: int = 60  # Reciprocal Rank Fusion constant (higher = smoother blending)

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH = DATA_DIR / "golden.json"
MIN_GOLDEN_PAIRS: int = 30
RUNS_CSV_PATH = ROOT_DIR / "runs.csv"

# ---------------------------------------------------------------------------
# LangSmith (optional — all features degrade gracefully if key is absent)
# ---------------------------------------------------------------------------
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rag-legal-qa")
LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
