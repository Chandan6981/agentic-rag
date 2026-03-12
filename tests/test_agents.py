"""
tests/test_agents.py
─────────────────────
Unit tests for orchestrator and sub-agents.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.calculator_agent import calculate, CalculatorAgent


# ── Calculator Agent ─────────────────────────────────────────────────────────

class TestCalculatorAgent:

    def test_basic_arithmetic(self):
        assert calculate("2 + 2") == "4"
        assert calculate("10 - 3") == "7"
        assert calculate("6 * 7") == "42"
        assert calculate("15 / 4") == "3.75"

    def test_power(self):
        assert calculate("2 ** 10") == "1024"

    def test_sqrt(self):
        assert calculate("sqrt(144)") == "12.0"

    def test_complex_expression(self):
        result = calculate("(100 * 1.23) / 4 + sqrt(16)")
        assert result == "34.75"

    def test_constants(self):
        result = float(calculate("pi"))
        assert abs(result - 3.14159265) < 1e-5

    def test_division_by_zero(self):
        result = calculate("1 / 0")
        assert "Error" in result
        assert "zero" in result.lower()

    def test_invalid_expression(self):
        result = calculate("import os; os.system('ls')")
        assert "Error" in result

    def test_unsafe_name(self):
        result = calculate("__import__('os').system('ls')")
        assert "Error" in result

    def test_as_tool(self):
        tool = CalculatorAgent.as_tool()
        assert tool.name == "calculator"
        assert callable(tool.func)


# ── Retriever Agent ───────────────────────────────────────────────────────────

class TestRetrieverAgent:

    @patch("src.agents.retriever_agent.load_faiss_index")
    @patch("src.agents.retriever_agent.load_bm25_index")
    @patch("src.agents.retriever_agent.build_hybrid_retriever")
    def test_retrieve_returns_formatted_context(
        self, mock_hybrid, mock_bm25, mock_faiss
    ):
        from langchain_core.documents import Document
        from src.agents.retriever_agent import RetrieverAgent

        mock_doc = Document(
            page_content="Photosynthesis uses sunlight, water, and CO2.",
            metadata={"doc_id": "bio_101", "source": "bio_101.pdf"},
        )

        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        mock_hybrid.return_value = mock_retriever

        agent = RetrieverAgent(vectorstore_path="/tmp/vs", bm25_path="/tmp/bm25.pkl")
        result = agent.retrieve("what is photosynthesis")

        assert "bio_101" in result
        assert "Photosynthesis" in result

    @patch("src.agents.retriever_agent.load_faiss_index")
    @patch("src.agents.retriever_agent.load_bm25_index")
    @patch("src.agents.retriever_agent.build_hybrid_retriever")
    def test_retrieve_no_docs(self, mock_hybrid, mock_bm25, mock_faiss):
        from src.agents.retriever_agent import RetrieverAgent

        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []
        mock_hybrid.return_value = mock_retriever

        agent = RetrieverAgent(vectorstore_path="/tmp/vs", bm25_path="/tmp/bm25.pkl")
        result = agent.retrieve("nonsense gibberish xyz")

        assert "No relevant" in result


# ── Web Search Agent ──────────────────────────────────────────────────────────

class TestWebSearchAgent:

    @patch("src.agents.web_search_agent.DuckDuckGoSearchRun")
    def test_search_returns_results(self, mock_ddg_cls):
        from src.agents.web_search_agent import WebSearchAgent

        mock_ddg = MagicMock()
        mock_ddg.run.return_value = "LangChain is a framework for LLM applications."
        mock_ddg_cls.return_value = mock_ddg

        agent = WebSearchAgent()
        result = agent.search("What is LangChain?")

        assert "LangChain" in result
        assert "Web Search Results" in result

    @patch("src.agents.web_search_agent.DuckDuckGoSearchRun")
    def test_search_empty_returns_message(self, mock_ddg_cls):
        from src.agents.web_search_agent import WebSearchAgent

        mock_ddg = MagicMock()
        mock_ddg.run.return_value = ""
        mock_ddg_cls.return_value = mock_ddg

        agent = WebSearchAgent()
        result = agent.search("xyznonexistent12345")
        assert "No relevant" in result or "results" in result.lower()
