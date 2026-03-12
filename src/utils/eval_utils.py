"""
src/utils/eval_utils.py
────────────────────────
RAGAS evaluation utilities.
Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("ragas not installed — evaluation will be skipped.")


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL eval dataset. Each line: {question, answer, contexts, ground_truth}."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} eval samples from {path}")
    return records


def run_ragas_eval(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Run RAGAS evaluation and return metric scores.

    Returns:
        {
            "faithfulness": 0.82,
            "answer_relevancy": 0.89,
            "context_precision": 0.86,
            "context_recall": 0.78,
        }
    """
    if not RAGAS_AVAILABLE:
        logger.error("ragas package is required for evaluation.")
        return {}

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }
    logger.info(f"RAGAS scores: {scores}")
    return scores


def save_eval_results(scores: Dict[str, float], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Eval results saved to {output_path}")
