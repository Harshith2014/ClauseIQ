"""Streamlit Q&A interface for the Legal Document RAG system.

Run:
    streamlit run ui/app.py

Features:
- Sidebar: chunk size, top-k, model info, document stats
- Query input with example questions
- Answer panel with inline citations highlighted
- Per-citation source cards (file, page, clause excerpt)
- Colour-coded confidence badge
- Session history of last 5 Q&A pairs
- Pipeline is built once and cached across reruns
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Make project root importable when launched from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, CHUNK_SIZES, DEFAULT_TOP_K, OLLAMA_MODEL
from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from embeddings.embedder import get_embedder
from vectorstore.faiss_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid import HybridRetriever
from generation.generator import LegalQAChain, GenerationResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Legal RAG — Document Q&A",
    page_icon="law",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.conf-high   { background:#1a7a4a; color:white; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:0.85rem; }
.conf-medium { background:#b07d0f; color:white; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:0.85rem; }
.conf-low    { background:#b03a2e; color:white; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:0.85rem; }
.citation-card {
    background: #1e1e2e;
    border-left: 4px solid #7c6af7;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
}
.history-item {
    border-left: 3px solid #444;
    padding-left: 10px;
    margin: 8px 0;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline builder (cached so it survives reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_pipeline(chunk_size: int, top_k: int):
    """Load PDFs, embed, index, and wire the full chain. Cached by (chunk_size, top_k)."""
    docs     = load_pdfs(RAW_DIR)
    chunks   = chunk_documents(docs, chunk_size=chunk_size)
    embedder = get_embedder()
    store    = FAISSStore(embedder)
    store.build(chunks)
    retriever = HybridRetriever(
        DenseRetriever(store),
        BM25Retriever(chunks),
    )
    chain = LegalQAChain(retriever=retriever, top_k=top_k)
    meta  = {
        "num_docs":   len({d.metadata["source"] for d in docs}),
        "num_pages":  len(docs),
        "num_chunks": len(chunks),
        "sources":    sorted({d.metadata["source"] for d in docs}),
    }
    return chain, meta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
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


def confidence_badge(score: float) -> str:
    if score >= 0.8:
        return f'<span class="conf-high">HIGH {score:.2f}</span>'
    elif score >= 0.5:
        return f'<span class="conf-medium">MEDIUM {score:.2f}</span>'
    else:
        return f'<span class="conf-low">LOW {score:.2f}</span>'


def render_result(result: GenerationResult) -> None:
    """Render answer, citations, and confidence in the main panel."""

    # Answer
    st.markdown("#### Answer")
    st.markdown(result.answer)

    # Confidence badge
    st.markdown(
        f"**Confidence:** {confidence_badge(result.confidence)}",
        unsafe_allow_html=True,
    )

    # Citations
    if result.citations:
        st.markdown(f"#### Citations &nbsp; `{len(result.citations)} source(s)`")
        for i, c in enumerate(result.citations, 1):
            clause_display = c.clause[:300] + ("..." if len(c.clause) > 300 else "")
            st.markdown(
                f"""<div class="citation-card">
                <b>[{i}]</b> &nbsp; <code>{c.source}</code> &nbsp;|&nbsp; Page&nbsp;{c.page}<br>
                <em>"{clause_display}"</em>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.warning(
            "No structured citations parsed. "
            "The LLM may not have followed the citation format — "
            "check the raw response below.",
            icon="⚠️",
        )

    # Raw response (collapsed)
    with st.expander("Raw LLM response", expanded=False):
        st.code(result.raw_response, language=None)

    st.caption(f"Context chunks used: {result.num_context_chunks}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Legal RAG")
    st.caption("Retrieval-Augmented Q&A on legal documents")
    st.divider()

    st.subheader("Pipeline settings")
    chunk_size = st.select_slider(
        "Chunk size (chars)",
        options=CHUNK_SIZES,
        value=512,
        help="Smaller = more precise retrieval. Larger = more context per chunk.",
    )
    top_k = st.slider(
        "Context chunks (top-k)",
        min_value=1, max_value=8, value=DEFAULT_TOP_K,
        help="How many chunks the LLM sees as context.",
    )

    st.divider()
    st.subheader("Model info")
    st.markdown(f"**LLM:** `{OLLAMA_MODEL}` via Ollama")
    st.markdown("**Embedder:** `BAAI/bge-small-en-v1.5` (384-dim)")
    st.markdown("**Retrieval:** Hybrid BM25 + FAISS (RRF k=60)")

    # Build pipeline
    st.divider()
    with st.spinner("Building index..."):
        try:
            chain, meta = build_pipeline(chunk_size, top_k)
            st.success("Pipeline ready")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.subheader("Indexed documents")
    st.markdown(f"**{meta['num_docs']} files** · {meta['num_pages']} pages · {meta['num_chunks']} chunks")
    for src in meta["sources"]:
        st.markdown(f"- `{src}`")

    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Legal Document Q&A")
st.caption(
    "Ask questions about the indexed legal documents. "
    "Every answer is grounded with exact source, page, and clause citations."
)

# Example query selector
with st.expander("Example questions", expanded=False):
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q[:30]}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

# Query input
prefill = st.session_state.pop("prefill", "")
query = st.text_input(
    "Your question",
    value=prefill,
    placeholder="e.g. What is the termination notice period?",
)

run_btn = st.button("Ask", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if run_btn and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        t0 = time.time()
        try:
            result = chain.run(query.strip())
            elapsed = time.time() - t0
        except Exception as e:
            st.error(f"Generation error: {e}")
            st.stop()

    st.divider()
    st.subheader(f"Result &nbsp; `{elapsed:.1f}s`")
    render_result(result)

    # Prepend to history (keep last 5)
    st.session_state.history = [result] + st.session_state.history
    st.session_state.history = st.session_state.history[:5]

elif run_btn:
    st.warning("Please enter a question first.")

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

if st.session_state.history:
    st.divider()
    st.subheader("Recent questions")
    # Skip first item (already shown above as current result)
    for prev in st.session_state.history[1:]:
        with st.container():
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(
                    f'<div class="history-item"><b>Q:</b> {prev.query}</div>',
                    unsafe_allow_html=True,
                )
                # Short answer preview (first 200 chars)
                preview = prev.answer[:200].replace("\n", " ")
                if len(prev.answer) > 200:
                    preview += "..."
                st.markdown(
                    f'<div class="history-item"><b>A:</b> {preview}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    confidence_badge(prev.confidence),
                    unsafe_allow_html=True,
                )
            st.markdown("")
