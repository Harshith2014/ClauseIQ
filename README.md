# ClauseIQ — Legal Document Q&A with RAG + Eval

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://clauseiq-fhx6x3gwboaa8nbp29szxh.streamlit.app/)

ClauseIQ is a production-quality Retrieval-Augmented Generation (RAG) system
for querying legal documents. It answers natural-language questions about
employment agreements, NDAs, and service contracts by retrieving the most
relevant clauses and generating grounded answers with inline citations
(source file, page number, exact clause text, and confidence score).

Built as a portfolio project demonstrating the full LLM engineering stack:
ingestion, hybrid retrieval, generation, evaluation harness, and run logging.
**No paid APIs required** — everything runs locally via Ollama.

---

## Features

- **Hybrid retrieval** — BM25 + dense (FAISS) fused with Reciprocal Rank Fusion (k=60)
- **Citation grounding** — every answer includes `[Source, Page, Clause, Confidence]`
- **Eval harness** — RAGAS metrics + LLM-as-judge (1-5 scoring) on 23 golden QA pairs
- **Special checks** — TOC artifact detection, multi-jurisdiction coverage, unanswerable question handling
- **Run logging** — every eval appended to `runs.csv`; metric delta comparison across runs
- **LangSmith tracing** — optional, activates automatically when `LANGSMITH_API_KEY` is set
- **Streamlit UI** — interactive Q&A with source cards and session history

---

## Stack

| Concern | Library |
|---|---|
| PDF loading | `langchain-community` PyPDFLoader |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` `BAAI/bge-small-en-v1.5` (384-dim) |
| Vector DB | `faiss-cpu` (`IndexFlatIP`, cosine on normalised vectors) |
| Sparse retrieval | `rank-bm25` `BM25Okapi` |
| LLM | Ollama `llama3.2` (local, no API key) |
| UI | `streamlit` |
| Eval | Custom RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) |
| Tracing | `langsmith` (optional) |

---

## Project Structure

```
ClauseIQ/
├── config.py               # Central config (paths, model names, chunk sizes)
├── requirements.txt
├── .env.example
│
├── data/
│   └── raw/                # Drop your PDFs here (4 legal docs included)
│
├── ingestion/
│   ├── pdf_loader.py       # PyPDFLoader -> list[Document] with metadata
│   └── chunker.py          # RecursiveCharacterTextSplitter, adds chunk_index
│
├── embeddings/
│   └── embedder.py         # HuggingFaceEmbeddings singleton (BGE-small-en-v1.5)
│
├── vectorstore/
│   ├── faiss_store.py      # Build / save / load FAISS IndexFlatIP
│   └── qdrant_store.py     # Qdrant upgrade stub
│
├── retrieval/
│   ├── dense_retriever.py  # FAISS cosine top-k
│   ├── bm25_retriever.py   # BM25Okapi keyword retrieval
│   └── hybrid.py           # Reciprocal Rank Fusion combiner
│
├── generation/
│   ├── prompts.py          # LEGAL_QA_PROMPT + format_context()
│   └── generator.py        # LCEL chain -> GenerationResult (answer + citations)
│
├── ui/
│   └── app.py              # Streamlit Q&A interface
│
├── eval/
│   ├── golden_dataset.py   # 23 curated QA pairs + LLM generation
│   ├── ragas_scorer.py     # Faithfulness, answer relevancy, context precision/recall
│   ├── llm_judge.py        # LLM-as-judge 1-5 scoring (correctness, faithfulness, etc.)
│   └── run_eval.py         # Full eval orchestrator (CLI)
│
├── logging_/
│   ├── csv_logger.py       # Append run metrics to runs.csv
│   ├── langsmith_logger.py # LangSmith tracing context manager
│   └── run_compare.py      # Delta table between two runs
│
└── tests/                  # 164 unit tests (pytest), all green
```

---

## Setup

### Prerequisites

- Python 3.10+ (tested on 3.14)
- [Ollama](https://ollama.com) installed and running
- `llama3.2` model pulled: `ollama pull llama3.2`

### Install

```bash
git clone https://github.com/Harshith2014/ClauseIQ.git
cd ClauseIQ/Desktop/RAG
pip install -r requirements.txt
```

### Configure (optional)

```bash
cp .env.example .env
# Edit .env to set:
#   LANGSMITH_API_KEY=ls__...    (optional, for tracing)
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2
```

### Add your documents

Drop PDF files into `data/raw/`. Four sample legal documents are included:
- `exhibit101.pdf` — Executive employment agreement (Verdisys / California)
- `EMPLOYMENT-AGREEMENT.pdf` — Employment template (Jaipur, India)
- `EMPLOYMENT_AGREEMENT (3).pdf` — Employment template (LetsVenture, India)
- `sample-standard_contract.pdf` — Standard employment contract (Australia)

---

## Running Each Phase

### Phase 1-2: Ingestion + Retrieval tests

```bash
python -m pytest tests/test_ingestion.py tests/test_embeddings.py tests/test_retrieval.py -v
```

### Phase 3: Generation tests

```bash
python -m pytest tests/test_generation.py -v
```

### Phase 4: Streamlit UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501`. Use the sidebar to tune chunk size and top-k.
The pipeline is cached — changing settings rebuilds the index automatically.

### Phase 5: Retrieval demo on real documents

```bash
python scripts/demo_retrieval_real.py
```

### Phase 5: Eval harness

```bash
# Smoke test (5 pairs, no LLM judge — fast)
python eval/run_eval.py --sample 5 --no-judge

# Full run (all 23 pairs with judge, save baseline)
python eval/run_eval.py --run-id run_001 --save-csv

# Generate extra golden pairs via LLM
python eval/golden_dataset.py --llm-extra 10
```

### Phase 6: Logging + run comparison

```bash
# List all logged runs
python logging_/run_compare.py

# Compare two runs side-by-side
python logging_/run_compare.py run_001 run_002

# Re-run with different chunk size and compare
python eval/run_eval.py --chunk-size 256 --run-id run_002 --save-csv
python logging_/run_compare.py run_001 run_002
```

### Full test suite

```bash
python -m pytest tests/ -v
# Expected: 164 passed
```

---

## Eval Metrics

| Metric | Description |
|---|---|
| `ragas_faithfulness` | Fraction of answer claims supported by retrieved context |
| `ragas_answer_relevancy` | Cosine similarity between question and synthetic questions from the answer |
| `ragas_context_precision` | Fraction of retrieved chunks judged relevant by LLM |
| `ragas_context_recall` | Fraction of ground-truth statements covered by retrieved context |
| `judge_correctness` | Factual accuracy vs ground truth (1-5) |
| `judge_faithfulness` | Groundedness in context (1-5) |
| `judge_helpfulness` | Clarity and completeness (1-5) |
| `judge_citation_quality` | Citation format and traceability (1-5) |

### Special checks

- **TOC artifact** — flags answers that cite the table-of-contents page instead of the actual clause
- **Multi-jurisdiction** — verifies all expected governing-law sources are cited
- **Unanswerable** — detects whether the system says "not found" vs hallucinating on questions with no answer in the docs

---

## LangSmith Tracing

Set `LANGSMITH_API_KEY` in `.env`. All LangChain chain calls during eval runs
will be automatically traced to your LangSmith project (`clauseiq-rag` by default).

```env
LANGSMITH_API_KEY=ls__your_key_here
LANGSMITH_PROJECT=clauseiq-rag
```

---

## Architecture Notes

1. **Citation extraction** — the generator parses its own output with regex for `[Source:...]` and `Confidence:` lines, making it model-agnostic (works with any Ollama model)
2. **RAGAS without OpenAI** — all four RAGAS metrics implemented from scratch using Ollama + BGE embedder; no scikit-network dependency
3. **Chunk metadata continuity** — every split `Document` carries `{source, page, chunk_index, chunk_size}` end-to-end, so citations are always traceable
4. **BM25 rebuild on load** — BM25 index is rebuilt from chunk texts on startup (fast, deterministic, no serialisation needed)
5. **RRF k=60** — standard default; tunable via `config.py`
