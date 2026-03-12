"""
src/api/schemas.py
───────────────────
Pydantic request/response models for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved documents")
    use_web_search: bool = Field(False, description="Allow web search fallback")
    use_lora_model: bool = Field(False, description="Use fine-tuned LoRA model for generation")
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Session identifier")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What are the main factors driving crop price volatility?",
            "top_k": 5,
            "use_web_search": False,
            "session_id": "abc-123",
        }
    }}


class SourceDocument(BaseModel):
    doc_id: str
    text: str = ""
    score: Optional[float] = None


class AgentStep(BaseModel):
    tool: str
    tool_input: str
    observation: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    agent_trace: List[AgentStep]
    faithfulness_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    latency_ms: int
    session_id: str
    guardrail_triggered: bool = False

    model_config = {"json_schema_extra": {
        "example": {
            "answer": "Crop prices are primarily driven by weather patterns, supply chain disruptions, and global demand shifts. [Source: agri_report_2023]",
            "sources": [{"doc_id": "agri_report_2023", "text": "Weather is the...", "score": 0.91}],
            "agent_trace": [{"tool": "retriever", "tool_input": "crop price factors", "observation": "..."}],
            "faithfulness_score": 0.87,
            "latency_ms": 1240,
            "session_id": "abc-123",
            "guardrail_triggered": False,
        }
    }}


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = ""
    chunk_size: int = Field(512, ge=128, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=256)


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    vectorstore_path: str


# ── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    vectorstore_loaded: bool = False
