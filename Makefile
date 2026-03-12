.PHONY: install run test lint docker-build docker-up ingest evaluate

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ Installed. Edit .env with your API keys, then run: make run"

# ── Dev server ────────────────────────────────────────────────────────────────
run:
	PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/ --ignore E501

# ── Data ──────────────────────────────────────────────────────────────────────
ingest:
	PYTHONPATH=. python scripts/ingest.py --input data/raw/ --output data/vectorstore/

evaluate:
	PYTHONPATH=. python scripts/evaluate.py \
		--eval_dataset data/processed/eval_set.jsonl \
		--output results/ragas_metrics.json

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t agentic-rag:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# ── Fine-tuning ───────────────────────────────────────────────────────────────
finetune:
	PYTHONPATH=. python scripts/finetune_lora.py \
		--base_model meta-llama/Llama-2-7b-hf \
		--dataset data/processed/train.jsonl \
		--output_dir models/llama2-lora-finetuned \
		--config configs/lora_config.yaml

# ── Query (quick test) ────────────────────────────────────────────────────────
query:
	curl -s -X POST http://localhost:8080/api/v1/query \
		-H "Content-Type: application/json" \
		-d '{"query": "What is the main use case of this system?", "top_k": 3}' \
		| python -m json.tool
