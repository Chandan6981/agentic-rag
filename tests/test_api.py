"""
tests/test_api.py
──────────────────
FastAPI endpoint tests using httpx TestClient.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked orchestrator."""
    with patch("src.api.routes.get_orchestrator") as mock_orch_dep, \
         patch("src.api.routes.get_guardrail") as mock_guard_dep:

        mock_orchestrator = MagicMock()
        mock_orchestrator.query.return_value = {
            "answer": "Photosynthesis converts sunlight to glucose. [Source: bio_101]",
            "sources": [{"doc_id": "bio_101", "text": "Plants use sunlight...", "score": 0.92}],
            "agent_trace": [{"tool": "retriever", "tool_input": "photosynthesis", "observation": "..."}],
            "latency_ms": 850,
        }
        mock_orch_dep.return_value = mock_orchestrator

        from src.guardrails.constitutional_ai import GuardrailResult
        mock_guard = MagicMock()
        mock_guard.filter.return_value = GuardrailResult(
            passed=True, toxicity_score=0.01, faithfulness_score=0.88
        )
        mock_guard_dep.return_value = mock_guard

        from src.api.main import app
        yield TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vectorstore_loaded" in data


def test_query_endpoint_success(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is photosynthesis?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert data["guardrail_triggered"] is False


def test_query_validation_min_length(client):
    resp = client.post("/api/v1/query", json={"query": "Hi"})
    assert resp.status_code == 422  # validation error


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "running" in resp.json()["message"].lower()
