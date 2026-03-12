"""
src/tools/document_tools.py
─────────────────────────────
Document ingestion pipeline:
  - Load from local filesystem or AWS S3
  - Parse PDF, TXT, DOCX, Markdown
  - Chunk with configurable overlap
  - Attach metadata (source, doc_id, page, chunk_index)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import boto3
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ── Loaders ─────────────────────────────────────────────────────────────────

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_document(file_path: str) -> List[Document]:
    """Load a single document based on file extension."""
    ext = Path(file_path).suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        logger.warning(f"Unsupported file type: {ext} ({file_path})")
        return []
    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        logger.debug(f"Loaded {len(docs)} page(s) from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def load_directory(directory: str, glob: str = "**/*.*") -> List[Document]:
    """Recursively load all supported documents from a directory."""
    all_docs: List[Document] = []
    supported = set(_LOADER_MAP.keys())
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in supported:
            all_docs.extend(load_document(str(path)))
    logger.info(f"Loaded {len(all_docs)} total pages from {directory}")
    return all_docs


# ── S3 Ingestion ─────────────────────────────────────────────────────────────

def load_from_s3(
    bucket: str,
    prefix: str = "",
    local_tmp: str = "/tmp/rag_s3_docs",
) -> List[Document]:
    """
    Download documents from S3 and load them.

    Args:
        bucket: S3 bucket name
        prefix: key prefix to filter objects
        local_tmp: local directory to download files to
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    Path(local_tmp).mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    downloaded = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = Path(key).suffix.lower()
            if ext not in _LOADER_MAP:
                continue
            local_path = os.path.join(local_tmp, Path(key).name)
            try:
                s3.download_file(bucket, key, local_path)
                downloaded.append(local_path)
                logger.debug(f"Downloaded s3://{bucket}/{key} → {local_path}")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")

    all_docs: List[Document] = []
    for path in downloaded:
        all_docs.extend(load_document(path))
    logger.info(f"Loaded {len(all_docs)} documents from s3://{bucket}/{prefix}")
    return all_docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    Split documents into chunks with metadata enrichment.
    Each chunk gets: source, doc_id, chunk_index, char_count.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk.metadata["doc_id"] = f"{Path(source).stem}_{content_hash}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)

    logger.info(f"Chunked {len(documents)} documents → {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
