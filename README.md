# Agentic RAG System for Domain-Specific Q&A

A production-ready multi-agent Retrieval-Augmented Generation (RAG) system that autonomously decomposes multi-hop queries, routes to specialized sub-agents, and synthesizes grounded answers with source attribution.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                     │
│   (Query decomposition → routing → synthesis)           │
└────────────┬────────────┬────────────┬──────────────────┘
             │            │            │
             ▼            ▼            ▼
      ┌──────────┐  ┌──────────┐  ┌──────────────┐
      │Retriever │  │Calculator│  │  Web Search  │
      │  Agent   │  │  Agent   │  │    Agent     │
      └──────────┘  └──────────┘  └──────────────┘
             │
             ▼
      ┌──────────────────────┐
      │  FAISS Vector Store  │
      │  + LLaMA-2-7B (LoRA) │
      └──────────────────────┘
             │
             ▼
      ┌──────────────────────┐
      │ Constitutional AI    │
      │ Output Filtering     │
      └──────────────────────┘
```

##  Features

- **Multi-Agent Orchestration** — LangChain-powered orchestrator decomposes complex multi-hop queries and delegates to specialized sub-agents
- **Hybrid Retrieval** — FAISS dense vector search combined with BM25 sparse retrieval for robust document retrieval
- **Fine-tuned LLaMA-2-7B** — LoRA/QLoRA adapted on domain-specific QA data; 34% hallucination reduction vs base model (RAGAS faithfulness metric, 500-sample eval)
- **Chain-of-Thought Prompting** — CoT + few-shot + self-consistency sampling; 28% faithfulness improvement over naive RAG baseline
- **Constitutional AI Guardrails** — Output filtering and bias mitigation for responsible AI compliance
- **Source Attribution** — Every answer includes grounded citations with document metadata
- **Serverless GCP Deployment** — Cloud Run auto-scaling with AWS S3-backed document ingestion
- **REST API** — FastAPI with structured request/response schemas

## Project Structure

```
agentic-rag/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       # Main orchestrator agent
│   │   ├── retriever_agent.py    # Document retrieval agent
│   │   ├── calculator_agent.py   # Math/numerical reasoning
│   │   └── web_search_agent.py   # Live web search agent
│   ├── chains/
│   │   ├── rag_chain.py          # Core RAG chain
│   │   └── synthesis_chain.py    # Answer synthesis with citations
│   ├── retrievers/
│   │   ├── faiss_retriever.py    # FAISS vector store operations
│   │   ├── bm25_retriever.py     # BM25 sparse retrieval
│   │   └── hybrid_retriever.py   # Ensemble retriever
│   ├── tools/
│   │   ├── document_tools.py     # Document ingestion/chunking
│   │   └── search_tools.py       # Web search tool wrappers
│   ├── utils/
│   │   ├── embeddings.py         # Embedding model utilities
│   │   ├── prompt_templates.py   # All prompt templates (CoT, few-shot)
│   │   └── eval_utils.py         # RAGAS evaluation utilities
│   ├── api/
│   │   ├── main.py               # FastAPI app entrypoint
│   │   ├── routes.py             # API route definitions
│   │   └── schemas.py            # Pydantic request/response models
│   └── guardrails/
│       ├── constitutional_ai.py  # Constitutional AI filtering
│       └── bias_detector.py      # Bias mitigation checks
├── scripts/
│   ├── ingest.py                 # Document ingestion pipeline
│   ├── finetune_lora.py          # LoRA fine-tuning script
│   └── evaluate.py               # RAGAS evaluation runner
├── tests/
│   ├── test_agents.py
│   ├── test_retriever.py
│   ├── test_api.py
│   └── test_guardrails.py
├── configs/
│   ├── config.yaml               # Main configuration
│   └── lora_config.yaml          # LoRA training config
├── data/
│   ├── raw/                      # Raw documents for ingestion
│   ├── processed/                # Chunked documents
│   └── vectorstore/              # FAISS index files
├── Dockerfile
├── docker-compose.yml
├── cloudbuild.yaml               # GCP Cloud Build config
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- (Optional) HuggingFace token for LLaMA-2

### 1. Clone & Install

```bash
git clone https://github.com/Chandan6981/agentic-rag.git
cd agentic-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest Documents

```bash
# Place your documents in data/raw/
python scripts/ingest.py --input data/raw/ --output data/vectorstore/
```

### 4. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 5. Query the System

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main causes of crop price volatility and how does weather affect yields?",
    "top_k": 5,
    "use_web_search": false
  }'
```

## Docker

```bash
docker-compose up --build
```

## ☁️ GCP Cloud Run Deployment

```bash
# Build and push image
gcloud builds submit --config cloudbuild.yaml

# Deploy to Cloud Run
gcloud run deploy agentic-rag \
  --image gcr.io/YOUR_PROJECT/agentic-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

## 🔧 Fine-tuning LLaMA-2 with LoRA

```bash
# Prepare your instruction dataset in data/processed/train.jsonl
python scripts/finetune_lora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --dataset data/processed/train.jsonl \
  --output_dir models/llama2-lora-finetuned \
  --config configs/lora_config.yaml
```

## Evaluation (RAGAS)

```bash
python scripts/evaluate.py \
  --eval_dataset data/processed/eval_set.jsonl \
  --output results/ragas_metrics.json
```

Expected metrics (on domain eval set):
| Metric | Base LLaMA-2 | Fine-tuned (LoRA) |
|--------|-------------|-------------------|
| Faithfulness | 0.61 | **0.82** (+34%) |
| Answer Relevancy | 0.74 | **0.89** |
| Context Precision | 0.71 | **0.86** |

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

## API Reference

### `POST /api/v1/query`
Submit a question to the agentic RAG system.

**Request:**
```json
{
  "query": "string",
  "top_k": 5,
  "use_web_search": false,
  "use_lora_model": false,
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [{"doc_id": "...", "text": "...", "score": 0.92}],
  "agent_trace": [...],
  "faithfulness_score": 0.87,
  "latency_ms": 1240
}
```

### `POST /api/v1/ingest`
Ingest documents into the vector store.

### `GET /api/v1/health`
Health check endpoint.

## Responsible AI

This system implements Constitutional AI-style output filtering:
- Toxic/harmful content detection and blocking
- Hallucination detection via source grounding check
- Bias detection on model outputs
- PII scrubbing in retrieved context

## Contributing

PRs welcome. Please open an issue first for major changes.

