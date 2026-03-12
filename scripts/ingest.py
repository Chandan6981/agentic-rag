#!/usr/bin/env python3
"""
scripts/ingest.py
──────────────────
CLI script to ingest documents into the FAISS + BM25 vector store.

Usage:
    python scripts/ingest.py --input data/raw/ --output data/vectorstore/
    python scripts/ingest.py --s3-bucket my-bucket --s3-prefix documents/
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS + BM25 vector store.")
    parser.add_argument("--input", "-i", help="Local directory containing documents")
    parser.add_argument("--output", "-o", default="data/vectorstore", help="Vector store output path")
    parser.add_argument("--s3-bucket", help="S3 bucket name (optional)")
    parser.add_argument("--s3-prefix", default="", help="S3 key prefix (optional)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    from src.tools.document_tools import load_directory, load_from_s3, chunk_documents
    from src.retrievers.faiss_retriever import build_faiss_index
    from src.retrievers.bm25_retriever import build_bm25_index

    # Load documents
    if args.s3_bucket:
        logger.info(f"Loading from S3: s3://{args.s3_bucket}/{args.s3_prefix}")
        docs = load_from_s3(args.s3_bucket, args.s3_prefix)
    elif args.input:
        logger.info(f"Loading from local directory: {args.input}")
        docs = load_directory(args.input)
    else:
        logger.error("Provide --input or --s3-bucket.")
        sys.exit(1)

    if not docs:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded {len(docs)} document pages.")

    # Chunk
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    logger.info(f"Created {len(chunks)} chunks.")

    # Build indexes
    os.makedirs(args.output, exist_ok=True)
    build_faiss_index(chunks, args.output)
    build_bm25_index(chunks, os.path.join(args.output, "bm25_index.pkl"))

    logger.success(f"✅ Ingestion complete. Vector store saved to: {args.output}")
    logger.info(f"   Documents loaded : {len(docs)}")
    logger.info(f"   Chunks created   : {len(chunks)}")
    logger.info(f"   Chunk size       : {args.chunk_size}")
    logger.info(f"   Chunk overlap    : {args.chunk_overlap}")


if __name__ == "__main__":
    main()
