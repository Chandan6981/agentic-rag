"""
src/guardrails/constitutional_ai.py
─────────────────────────────────────
Constitutional AI-style output filtering.
- Toxicity detection (detoxify)
- Hallucination check (source grounding)
- Bias detection
- PII scrubbing

Reduces toxic/off-topic responses and aligns outputs with desired behavior.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from src.utils.prompt_templates import (
    CONSTITUTIONAL_CRITIQUE_PROMPT,
    CONSTITUTIONAL_REVISION_PROMPT,
)


@dataclass
class GuardrailResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    revised_answer: Optional[str] = None
    toxicity_score: float = 0.0
    faithfulness_score: float = 1.0


# ── PII Patterns ────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[CARD_REDACTED]"),
]


def scrub_pii(text: str) -> str:
    """Replace PII patterns with redaction placeholders."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Toxicity Check ───────────────────────────────────────────────────────────

def check_toxicity(text: str, threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Check for toxic content using detoxify.
    Returns (is_toxic: bool, toxicity_score: float).
    """
    try:
        from detoxify import Detoxify
        results = Detoxify("original").predict(text)
        score = results.get("toxicity", 0.0)
        is_toxic = score >= threshold
        if is_toxic:
            logger.warning(f"Toxicity detected: score={score:.3f}")
        return is_toxic, score
    except ImportError:
        logger.warning("detoxify not installed — skipping toxicity check.")
        return False, 0.0
    except Exception as e:
        logger.error(f"Toxicity check failed: {e}")
        return False, 0.0


# ── Hallucination / Faithfulness Check ──────────────────────────────────────

def check_faithfulness(answer: str, source_texts: List[str], threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Simple keyword-overlap faithfulness heuristic.
    A proper implementation uses RAGAS faithfulness metric at eval time.
    Returns (is_faithful: bool, score: float).
    """
    if not source_texts:
        return True, 1.0  # no sources to check against

    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
    source_words = set()
    for src in source_texts:
        source_words.update(re.findall(r"\b\w+\b", src.lower()))

    # Remove stop words for better signal
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "shall", "to", "of",
        "in", "on", "at", "by", "for", "with", "from", "that", "this",
        "it", "its", "and", "or", "but", "not", "no", "so", "if", "as",
    }
    answer_content = answer_words - stop_words
    if not answer_content:
        return True, 1.0

    overlap = answer_content & source_words
    score = len(overlap) / len(answer_content)
    is_faithful = score >= threshold
    if not is_faithful:
        logger.warning(f"Low faithfulness: score={score:.3f} (threshold={threshold})")
    return is_faithful, score


# ── Constitutional AI LLM Critique + Revision ───────────────────────────────

class ConstitutionalAIFilter:
    """
    LLM-based Constitutional AI output filter.
    Critiques outputs and rewrites them when issues are found.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        toxicity_threshold: float = 0.7,
        faithfulness_threshold: float = 0.5,
        enable_pii: bool = True,
    ):
        self.toxicity_threshold = toxicity_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.enable_pii = enable_pii
        self._llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._critique_chain = CONSTITUTIONAL_CRITIQUE_PROMPT | self._llm | StrOutputParser()
        self._revision_chain = CONSTITUTIONAL_REVISION_PROMPT | self._llm | StrOutputParser()

    def filter(
        self,
        answer: str,
        source_texts: Optional[List[str]] = None,
    ) -> GuardrailResult:
        """
        Run all guardrail checks on the answer.
        Returns a GuardrailResult with pass/fail and optionally a revised answer.
        """
        issues = []
        source_texts = source_texts or []

        # 1. PII scrub
        if self.enable_pii:
            answer = scrub_pii(answer)

        # 2. Toxicity check
        is_toxic, tox_score = check_toxicity(answer, self.toxicity_threshold)
        if is_toxic:
            issues.append(f"TOXICITY | high | toxicity_score={tox_score:.3f}")

        # 3. Faithfulness check
        is_faithful, faith_score = check_faithfulness(
            answer, source_texts, self.faithfulness_threshold
        )
        if not is_faithful:
            issues.append(f"HALLUCINATION | medium | faithfulness_score={faith_score:.3f}")

        # 4. LLM critique for bias and other issues
        sources_str = "\n---\n".join(source_texts[:3]) if source_texts else "No sources provided."
        try:
            critique = self._critique_chain.invoke({
                "answer": answer,
                "sources": sources_str,
            })
            if critique.strip().upper() != "PASS":
                for line in critique.strip().split("\n"):
                    line = line.strip()
                    if line and "|" in line:
                        issues.append(line)
        except Exception as e:
            logger.error(f"Constitutional critique failed: {e}")

        if not issues:
            return GuardrailResult(
                passed=True,
                toxicity_score=tox_score,
                faithfulness_score=faith_score,
            )

        # 5. Revise if issues found
        logger.info(f"Guardrail issues found: {issues}. Revising answer …")
        try:
            revised = self._revision_chain.invoke({
                "answer": answer,
                "issues": "\n".join(issues),
            })
        except Exception as e:
            logger.error(f"Revision failed: {e}")
            revised = answer  # return original if revision fails

        return GuardrailResult(
            passed=False,
            issues=issues,
            revised_answer=revised,
            toxicity_score=tox_score,
            faithfulness_score=faith_score,
        )
