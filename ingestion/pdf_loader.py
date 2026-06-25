"""Load PDF files from a directory into LangChain Documents.

Each returned Document has the following metadata keys guaranteed:
  - source       : PDF filename (stem + .pdf)
  - source_path  : absolute path string
  - page         : 0-based page index (set by PyPDFLoader)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import RAW_DIR

logger = logging.getLogger(__name__)


def load_pdfs(data_dir: Path = RAW_DIR) -> List[Document]:
    """Load every *.pdf found in *data_dir* and return a flat list of Documents.

    Args:
        data_dir: Directory to scan for PDF files. Defaults to config.RAW_DIR.

    Returns:
        List of LangChain Documents, one per page, with normalised metadata.

    Raises:
        FileNotFoundError: If *data_dir* contains no PDF files.
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}'. "
            "Run 'python scripts/create_sample_docs.py' to generate sample docs."
        )

    all_docs: List[Document] = []
    for pdf_path in pdf_files:
        logger.info("Loading %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page_doc in pages:
            # Normalise metadata — PyPDFLoader already sets 'source' and 'page'
            # but we add a friendly filename and absolute path for clarity.
            page_doc.metadata["source"] = pdf_path.name
            page_doc.metadata["source_path"] = str(pdf_path)
            # 'page' is already 0-based int from PyPDFLoader

        all_docs.extend(pages)
        logger.info("  Loaded %d page(s) from %s", len(pages), pdf_path.name)

    logger.info("Total pages loaded: %d from %d file(s)", len(all_docs), len(pdf_files))
    return all_docs
