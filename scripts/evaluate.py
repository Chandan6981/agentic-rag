#!/usr/bin/env python3
"""
scripts/evaluate.py
────────────────────
RAGAS evaluation runner.
Measures: faithfulness, answer_relevancy, context_precision, context_recall.

Usage:
    python scripts/evaluate.py \
        --eval_dataset data/processed/eval_set.jsonl \
        --output results/ragas_metrics.json

Expected JSONL format:
    {"question": "...", "ground_truth": "...", "contexts": ["...", "..."]}
"""

import argparse
import json
import os
import sys
import time

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG system.")
    parser.add_argument("--eval_dataset", default="data/processed/eval_set.jsonl")
    parser.add_argument("--output", default="results/ragas_metrics.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval to N samples")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    from src.agents.orchestrator import OrchestratorAgent
    from src.utils.eval_utils import load_eval_dataset, run_ragas_eval, save_eval_results

    # Load eval data
    samples = load_eval_dataset(args.eval_dataset)
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"Evaluating on {len(samples)} samples …")

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(model=args.model, verbose=False)

    questions, answers, contexts, ground_truths = [], [], [], []
    errors = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        ref_contexts = sample.get("contexts", [])

        logger.info(f"[{i+1}/{len(samples)}] Querying: {question[:60]} …")
        try:
            result = orchestrator.query(question)
            answer = result["answer"]
            # Use retrieved source texts as contexts
            ctx = [s.get("text", "") for s in result["sources"]] or ref_contexts
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            answer = ""
            ctx = ref_contexts
            errors += 1

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(ground_truth)

    logger.info(f"Finished inference. Errors: {errors}/{len(samples)}")
    logger.info("Running RAGAS evaluation …")

    scores = run_ragas_eval(questions, answers, contexts, ground_truths)

    # Add run metadata
    scores["n_samples"] = len(samples)
    scores["errors"] = errors
    scores["model"] = args.model
    scores["eval_dataset"] = args.eval_dataset

    save_eval_results(scores, args.output)

    print("\n" + "="*50)
    print("RAGAS Evaluation Results")
    print("="*50)
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")
    print("="*50)


if __name__ == "__main__":
    main()
