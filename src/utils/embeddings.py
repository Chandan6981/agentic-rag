"""
src/utils/embeddings.py
────────────────────────
Embedding model wrappers — OpenAI + local HuggingFace fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from langchain_openai import OpenAIEmbeddings
from loguru import logger


@lru_cache(maxsize=1)
def get_embedding_model(model: str | None = None) -> OpenAIEmbeddings:
    """Return a cached embedding model instance."""
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    logger.info(f"Loading embedding model: {model}")
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """Embed a list of strings and return vectors."""
    emb = get_embedding_model(model)
    return emb.embed_documents(texts)


def embed_query(query: str, model: str | None = None) -> List[float]:
    """Embed a single query string."""
    emb = get_embedding_model(model)
    return emb.embed_query(query)
