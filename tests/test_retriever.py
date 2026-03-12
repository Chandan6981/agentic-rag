"""
tests/test_retriever.py
────────────────────────
Tests for FAISS, BM25, and hybrid retriever components.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Chunking ──────────────────────────────────────────────────────────────────

class TestDocumentChunking:

    def test_chunk_basic(self):
        from src.tools.document_tools import chunk_documents

        docs = [
            Document(
                page_content="This is a test document. " * 50,
                metadata={"source": "test.pdf"},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=128, chunk_overlap=16)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "doc_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_metadata_enrichment(self):
        from src.tools.document_tools import chunk_documents

        docs = [Document(page_content="Hello world.", metadata={"source": "doc.txt"})]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.metadata["doc_id"].startswith("doc_")
        assert isinstance(chunk.metadata["char_count"], int)


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def test_build_and_search(self):
        from src.retrievers.bm25_retriever import build_bm25_index, bm25_search

        docs = [
            Document(page_content="The cat sat on the mat.", metadata={"source": "a.txt"}),
            Document(page_content="Dogs are loyal animals.", metadata={"source": "b.txt"}),
            Document(page_content="Cats love sleeping.", metadata={"source": "c.txt"}),
        ]
        retriever = build_bm25_index(docs, k=2)
        results = bm25_search(retriever, "cat")
        assert len(results) >= 1
        texts = " ".join(d.page_content for d in results)
        assert "cat" in texts.lower() or "Cat" in texts

    def test_persist_and_reload(self):
        from src.retrievers.bm25_retriever import build_bm25_index, load_bm25_index, bm25_search

        docs = [
            Document(page_content="Quantum computing uses qubits.", metadata={"source": "q.txt"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bm25.pkl")
            build_bm25_index(docs, persist_path=path)
            loaded = load_bm25_index(path)
            results = bm25_search(loaded, "qubits")
            assert len(results) >= 1


# ── PII Scrubbing ─────────────────────────────────────────────────────────────

class TestPIIScrubbing:

    def test_scrub_email(self):
        from src.guardrails.constitutional_ai import scrub_pii

        text = "Contact me at alice@example.com for more info."
        scrubbed = scrub_pii(text)
        assert "alice@example.com" not in scrubbed
        assert "[EMAIL_REDACTED]" in scrubbed

    def test_scrub_phone(self):
        from src.guardrails.constitutional_ai import scrub_pii

        text = "Call us at 555-123-4567."
        scrubbed = scrub_pii(text)
        assert "555-123-4567" not in scrubbed

    def test_no_pii(self):
        from src.guardrails.constitutional_ai import scrub_pii

        text = "The sky is blue and the grass is green."
        scrubbed = scrub_pii(text)
        assert scrubbed == text
