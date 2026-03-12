"""
src/api/routes.py
──────────────────
FastAPI route handlers.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from loguru import logger

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    AgentStep,
)
from src.agents.orchestrator import OrchestratorAgent
from src.guardrails.constitutional_ai import ConstitutionalAIFilter


router = APIRouter()

# ── Singletons (lazy init) ──────────────────────────────────────────────────
_orchestrator: Optional[OrchestratorAgent] = None
_guardrail: Optional[ConstitutionalAIFilter] = None


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent(
            vectorstore_path=os.getenv("VECTORSTORE_PATH", "data/vectorstore"),
            model="gpt-4o",
            temperature=0.0,
            verbose=os.getenv("ENVIRONMENT") == "development",
        )
    return _orchestrator


def get_guardrail() -> ConstitutionalAIFilter:
    global _guardrail
    if _guardrail is None:
        _guardrail = ConstitutionalAIFilter()
    return _guardrail


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vs_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    vs_loaded = os.path.exists(os.path.join(vs_path, "index.faiss"))
    return HealthResponse(vectorstore_loaded=vs_loaded)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    guardrail: ConstitutionalAIFilter = Depends(get_guardrail),
):
    """
    Submit a question to the agentic RAG system.
    Returns a grounded answer with source citations, agent trace, and guardrail metadata.
    """
    logger.info(f"[{request.session_id}] Query: {request.query[:80]}")

    try:
        result = orchestrator.query(request.query)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    answer = result["answer"]
    source_texts = [s.get("text", "") for s in result["sources"]]

    # Run constitutional AI guardrails
    guard_result = guardrail.filter(answer, source_texts)
    final_answer = guard_result.revised_answer if guard_result.revised_answer else answer

    return QueryResponse(
        answer=final_answer,
        sources=[SourceDocument(**s) for s in result["sources"]],
        agent_trace=[AgentStep(**step) for step in result["agent_trace"]],
        faithfulness_score=guard_result.faithfulness_score,
        toxicity_score=guard_result.toxicity_score,
        latency_ms=result["latency_ms"],
        session_id=request.session_id,
        guardrail_triggered=not guard_result.passed,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store.
    Accepts S3 bucket/prefix or local uploads.
    """
    from src.tools.document_tools import load_from_s3, chunk_documents
    from src.retrievers.faiss_retriever import build_faiss_index, load_faiss_index, add_documents
    from src.retrievers.bm25_retriever import build_bm25_index

    vs_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")

    if request.s3_bucket:
        docs = load_from_s3(request.s3_bucket, request.s3_prefix or "")
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide s3_bucket, or use the /ingest/upload endpoint for local files.",
        )

    chunks = chunk_documents(docs, request.chunk_size, request.chunk_overlap)
    build_faiss_index(chunks, vs_path)
    build_bm25_index(chunks, os.path.join(vs_path, "bm25_index.pkl"))

    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        vectorstore_path=vs_path,
    )


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: list[UploadFile] = File(...)):
    """Upload and ingest local files directly via multipart form."""
    import tempfile, shutil
    from src.tools.document_tools import load_document, chunk_documents
    from src.retrievers.faiss_retriever import build_faiss_index
    from src.retrievers.bm25_retriever import build_bm25_index

    vs_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    all_docs = []

    with tempfile.TemporaryDirectory() as tmp:
        for file in files:
            dest = os.path.join(tmp, file.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            all_docs.extend(load_document(dest))

    chunks = chunk_documents(all_docs)
    build_faiss_index(chunks, vs_path)
    build_bm25_index(chunks, os.path.join(vs_path, "bm25_index.pkl"))

    return IngestResponse(
        status="success",
        documents_loaded=len(all_docs),
        chunks_created=len(chunks),
        vectorstore_path=vs_path,
    )
