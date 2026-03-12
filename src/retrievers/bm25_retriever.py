"""
src/retrievers/bm25_retriever.py
─────────────────────────────────
BM25 sparse retrieval for keyword-heavy queries.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger


def build_bm25_index(
    documents: List[Document],
    persist_path: Optional[str] = None,
    k: int = 5,
) -> BM25Retriever:
    """Build a BM25 retriever from documents."""
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    if persist_path:
        Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(persist_path, "wb") as f:
            pickle.dump(retriever, f)
        logger.info(f"BM25 index saved to {persist_path}")

    return retriever


def load_bm25_index(persist_path: str, k: int = 5) -> BM25Retriever:
    """Load a persisted BM25 retriever."""
    with open(persist_path, "rb") as f:
        retriever = pickle.load(f)
    retriever.k = k
    logger.info(f"BM25 index loaded from {persist_path}")
    return retriever


def bm25_search(retriever: BM25Retriever, query: str) -> List[Document]:
    """Run BM25 search and return documents."""
    docs = retriever.get_relevant_documents(query)
    logger.debug(f"BM25: {len(docs)} results for query '{query[:60]}'")
    return docs
