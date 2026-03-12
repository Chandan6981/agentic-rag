"""
src/agents/web_search_agent.py
────────────────────────────────
Web search sub-agent using DuckDuckGo (free) with SerpAPI fallback.
Used only when the knowledge base lacks current information.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class WebSearchAgent:
    """
    Live web search sub-agent.
    Primary: DuckDuckGo (no API key needed).
    Fallback: SerpAPI (requires SERPAPI_KEY env var).
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self._ddg = DuckDuckGoSearchRun()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str) -> str:
        """
        Search the web and return a formatted results string.

        Args:
            query: natural language search query

        Returns:
            Formatted web search results with source URLs.
        """
        query = query.strip()
        logger.debug(f"WebSearch: '{query}'")

        try:
            result = self._ddg.run(query)
            if not result or len(result) < 20:
                return "No relevant web search results found."
            return f"[Web Search Results for: '{query}']\n{result}"
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}. Trying SerpAPI fallback …")
            return self._serpapi_search(query)

    def _serpapi_search(self, query: str) -> str:
        """SerpAPI fallback."""
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return "Web search unavailable: no SERPAPI_KEY configured."
        try:
            from langchain_community.utilities import SerpAPIWrapper
            serpapi = SerpAPIWrapper(serpapi_api_key=api_key)
            result = serpapi.run(query)
            return f"[Web Search Results for: '{query}']\n{result}"
        except Exception as e:
            logger.error(f"SerpAPI also failed: {e}")
            return f"Web search failed: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="web_search",
            func=self.search,
            description=(
                "Search the internet for current or external information NOT found "
                "in the knowledge base. Use only when the retriever returns insufficient context. "
                "Input: a concise search query string. "
                "Output: relevant web snippets."
            ),
        )
