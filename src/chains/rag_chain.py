"""
src/chains/rag_chain.py
────────────────────────
Core RAG chain: CoT + few-shot prompting.
Used by the retriever agent for direct RAG queries (non-agentic path).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from loguru import logger

from src.utils.prompt_templates import COT_RAG_PROMPT


def format_docs(docs: List[Document]) -> str:
    """Format documents list into a single context string."""
    chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("doc_id", doc.metadata.get("source", f"doc_{i}"))
        chunks.append(f"[{source}]\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)


def build_rag_chain(retriever, model: str = "gpt-4o", temperature: float = 0.0):
    """
    Build a simple RAG chain using CoT + few-shot prompt template.

    Returns a LangChain LCEL chain:
        chain.invoke({"question": "..."}) → str answer
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | COT_RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info(f"RAG chain built with model={model}")
    return chain


def run_rag_query(chain, question: str) -> Dict[str, Any]:
    """Run the RAG chain and return answer + raw output."""
    logger.debug(f"RAG chain query: '{question[:80]}'")
    answer = chain.invoke(question)
    return {"answer": answer, "question": question}
