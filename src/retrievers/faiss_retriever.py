"""
src/retrievers/faiss_retriever.py
──────────────────────────────────
FAISS dense vector store — build, load, and search.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.utils.embeddings import get_embedding_model


def build_faiss_index(
    documents: List[Document],
    persist_path: str,
) -> FAISS:
    """Build a FAISS index from documents and persist to disk."""
    Path(persist_path).mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model()

    logger.info(f"Building FAISS index from {len(documents)} documents …")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(persist_path)
    logger.info(f"FAISS index saved to {persist_path}")
    return vectorstore


def load_faiss_index(persist_path: str) -> FAISS:
    """Load a persisted FAISS index from disk."""
    embeddings = get_embedding_model()
    logger.info(f"Loading FAISS index from {persist_path} …")
    vectorstore = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    score_threshold: float = 0.0,
) -> List[Tuple[Document, float]]:
    """
    Return (document, score) pairs sorted by relevance.
    score_threshold: minimum similarity score to include.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]
    logger.debug(f"FAISS: {len(filtered)}/{len(results)} results above threshold {score_threshold}")
    return filtered


def add_documents(vectorstore: FAISS, documents: List[Document], persist_path: str) -> FAISS:
    """Incrementally add documents to an existing FAISS index."""
    vectorstore.add_documents(documents)
    vectorstore.save_local(persist_path)
    logger.info(f"Added {len(documents)} documents; index updated at {persist_path}")
    return vectorstore
