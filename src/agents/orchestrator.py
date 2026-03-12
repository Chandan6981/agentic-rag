"""
src/agents/orchestrator.py
────────────────────────────
Main orchestrator agent.
- Decomposes multi-hop queries into sub-questions
- Routes to specialized sub-agents (retriever, calculator, web_search)
- Synthesizes final grounded answer with source attribution

Uses LangChain ReAct agent pattern with OpenAI GPT-4.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from loguru import logger

from src.agents.calculator_agent import CalculatorAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.web_search_agent import WebSearchAgent
from src.utils.prompt_templates import ORCHESTRATOR_PROMPT


class OrchestratorAgent:
    """
    Multi-agent orchestrator that decomposes queries and routes to sub-agents.
    Synthesizes a final grounded answer with citations.
    """

    def __init__(
        self,
        vectorstore_path: str | None = None,
        bm25_path: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_iterations: int = 6,
        verbose: bool = False,
        enable_web_search: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Sub-agents
        self._retriever_agent = RetrieverAgent(
            vectorstore_path=vectorstore_path,
            bm25_path=bm25_path,
        )
        self._calc_agent = CalculatorAgent()
        self._web_agent = WebSearchAgent() if enable_web_search else None

        # LLM
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=60,
            max_retries=3,
        )

        self._executor: Optional[AgentExecutor] = None

    def _build_tools(self) -> List[Tool]:
        tools = [
            self._retriever_agent.as_tool(),
            self._calc_agent.as_tool(),
        ]
        if self._web_agent:
            tools.append(self._web_agent.as_tool())
        return tools

    def _build_executor(self) -> AgentExecutor:
        tools = self._build_tools()
        agent = create_react_agent(
            llm=self._llm,
            tools=tools,
            prompt=ORCHESTRATOR_PROMPT,
        )
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question end-to-end and return a structured result.

        Returns:
            {
                "answer": str,
                "sources": List[dict],
                "agent_trace": List[dict],
                "latency_ms": int,
            }
        """
        if self._executor is None:
            self._executor = self._build_executor()

        start = time.monotonic()
        logger.info(f"Orchestrator processing: '{question[:80]}'")

        try:
            result = self._executor.invoke({"input": question})
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                "answer": f"An error occurred while processing your query: {e}",
                "sources": [],
                "agent_trace": [],
                "latency_ms": int((time.monotonic() - start) * 1000),
            }

        latency_ms = int((time.monotonic() - start) * 1000)
        answer = result.get("output", "")
        intermediate = result.get("intermediate_steps", [])

        # Extract source doc IDs mentioned in the answer
        sources = self._extract_sources(answer, intermediate)
        trace = self._format_trace(intermediate)

        logger.info(f"Orchestrator done in {latency_ms}ms. Sources: {[s['doc_id'] for s in sources]}")

        return {
            "answer": answer,
            "sources": sources,
            "agent_trace": trace,
            "latency_ms": latency_ms,
        }

    def _extract_sources(
        self,
        answer: str,
        intermediate_steps: List,
    ) -> List[Dict[str, Any]]:
        """Parse [Source N: doc_id] citations from the answer."""
        import re
        sources = []
        seen = set()

        # Extract from answer text
        pattern = r"\[Source\s*(?:\d+:\s*)?([^\]]+)\]"
        for match in re.finditer(pattern, answer):
            doc_id = match.group(1).strip()
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append({"doc_id": doc_id, "text": "", "score": None})

        # Also harvest raw docs from retriever tool observations
        for action, observation in intermediate_steps:
            if hasattr(action, "tool") and action.tool == "retriever":
                # Parse observation for source blocks
                for block in str(observation).split("---"):
                    header_match = re.match(r"\[Source \d+: ([^\]]+)\]", block.strip())
                    if header_match:
                        doc_id = header_match.group(1).strip()
                        snippet = block.strip()[len(header_match.group(0)):].strip()[:200]
                        if doc_id not in seen:
                            seen.add(doc_id)
                            sources.append({"doc_id": doc_id, "text": snippet, "score": None})

        return sources

    def _format_trace(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Format intermediate agent steps for API response."""
        trace = []
        for action, observation in intermediate_steps:
            trace.append({
                "tool": getattr(action, "tool", "unknown"),
                "tool_input": getattr(action, "tool_input", ""),
                "observation": str(observation)[:500],  # truncate long observations
            })
        return trace
