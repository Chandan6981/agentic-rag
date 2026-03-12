"""
src/retrievers/hybrid_retriever.py
────────────────────────────────────
Ensemble retriever: FAISS (dense) + BM25 (sparse) with configurable alpha weighting.
Alpha = 1.0 → pure dense; Alpha = 0.0 → pure sparse.
"""

from __future__ import annotations

from typing import List, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger


def build_hybrid_retriever(
    faiss_store: FAISS,
    bm25_retriever,
    alpha: float = 0.7,
    k: int = 5,
) -> EnsembleRetriever:
    """
    Build an ensemble retriever combining FAISS and BM25.

    Args:
        alpha: weight for dense retrieval (1 - alpha goes to BM25)
    """
    faiss_retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[alpha, 1.0 - alpha],
    )
    logger.info(f"Hybrid retriever built: FAISS={alpha:.2f}, BM25={1-alpha:.2f}, k={k}")
    return ensemble


def retrieve_with_scores(
    retriever: EnsembleRetriever,
    query: str,
) -> List[Document]:
    """Run hybrid retrieval and return deduplicated documents."""
    docs = retriever.get_relevant_documents(query)
    # Deduplicate by page_content
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    logger.debug(f"Hybrid retriever: {len(unique)} unique docs for '{query[:60]}'")
    return unique
