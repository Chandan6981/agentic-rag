"""
src/agents/retriever_agent.py
──────────────────────────────
Specialized sub-agent for document retrieval from the FAISS/BM25 knowledge base.
Returns grounded context with source attribution metadata.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import Tool
from loguru import logger

from src.retrievers.hybrid_retriever import build_hybrid_retriever, retrieve_with_scores
from src.retrievers.faiss_retriever import load_faiss_index
from src.retrievers.bm25_retriever import load_bm25_index


class RetrieverAgent:
    """
    Document retrieval agent backed by FAISS + BM25 hybrid retriever.
    Formats retrieved documents as numbered context blocks with source metadata.
    """

    def __init__(
        self,
        vectorstore_path: str | None = None,
        bm25_path: str | None = None,
        alpha: float = 0.7,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ):
        self.vectorstore_path = vectorstore_path or os.getenv(
            "VECTORSTORE_PATH", "data/vectorstore"
        )
        self.bm25_path = bm25_path or os.path.join(self.vectorstore_path, "bm25_index.pkl")
        self.alpha = alpha
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._retriever = None

    def _load_retriever(self):
        if self._retriever is not None:
            return
        try:
            faiss_store = load_faiss_index(self.vectorstore_path)
            bm25 = load_bm25_index(self.bm25_path, k=self.top_k)
            self._retriever = build_hybrid_retriever(
                faiss_store, bm25, alpha=self.alpha, k=self.top_k
            )
            logger.info("RetrieverAgent: hybrid retriever loaded.")
        except Exception as e:
            logger.error(f"RetrieverAgent: failed to load retriever — {e}")
            raise

    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant documents for a query.
        Returns formatted context string with source citations.
        """
        self._load_retriever()
        docs = retrieve_with_scores(self._retriever, query)

        if not docs:
            return "No relevant documents found in the knowledge base."

        formatted_chunks = []
        for i, doc in enumerate(docs[: self.top_k], start=1):
            source = doc.metadata.get("source", f"doc_{i}")
            doc_id = doc.metadata.get("doc_id", source)
            chunk_text = doc.page_content.strip()
            formatted_chunks.append(f"[Source {i}: {doc_id}]\n{chunk_text}")

        return "\n\n---\n\n".join(formatted_chunks)

    def as_tool(self) -> Tool:
        """Return a LangChain Tool wrapping this agent."""
        return Tool(
            name="retriever",
            func=self.retrieve,
            description=(
                "Search the domain-specific knowledge base for relevant documents. "
                "Input: a search query string. "
                "Output: relevant text passages with source citations."
            ),
        )
