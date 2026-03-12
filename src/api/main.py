"""
src/api/main.py
────────────────
FastAPI application entrypoint.
Deployed on GCP Cloud Run (serverless, auto-scaling).
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import router

load_dotenv()

# ── Logging setup ────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
    colorize=True,
)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Agentic RAG API starting up …")
    yield
    logger.info("👋 Agentic RAG API shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG System",
    description=(
        "Multi-agent Retrieval-Augmented Generation system for domain-specific Q&A. "
        "Supports multi-hop query decomposition, hybrid FAISS+BM25 retrieval, "
        "GPT-4 synthesis, and Constitutional AI guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request logging + latency ─────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)")
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Agentic RAG API is running. Visit /docs for the API reference."}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8080)),
        reload=os.getenv("ENVIRONMENT") == "development",
        workers=1,
    )
