"""
src/guardrails/bias_detector.py
─────────────────────────────────
Lightweight bias detection for model outputs.
Checks for demographic, gender, and racial bias indicators.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger


# Bias indicator word lists (extensible)
_GENDER_STEREOTYPES = [
    r"\bonly (men|women|males|females) can\b",
    r"\b(men|women) are (always|never|typically|naturally)\b",
    r"\b(he|she) should (stay|just)\b",
]

_RACIAL_INDICATORS = [
    r"\b(all|most|typical) (blacks|whites|asians|hispanics|latinos)\b",
    r"\b(race|ethnicity) (determines|affects) (intelligence|ability|character)\b",
]

_AGE_BIAS = [
    r"\b(old|elderly) people (can't|cannot|don't|shouldn't)\b",
    r"\bmillennials are (all|always|just)\b",
]

_ALL_PATTERNS = (
    [(p, "gender") for p in _GENDER_STEREOTYPES]
    + [(p, "racial") for p in _RACIAL_INDICATORS]
    + [(p, "age") for p in _AGE_BIAS]
)

_COMPILED = [(re.compile(p, re.IGNORECASE), category) for p, category in _ALL_PATTERNS]


def detect_bias(text: str) -> List[Tuple[str, str]]:
    """
    Scan text for bias indicators.

    Returns:
        List of (matched_text, category) tuples. Empty list = no bias detected.
    """
    findings = []
    for pattern, category in _COMPILED:
        match = pattern.search(text)
        if match:
            findings.append((match.group(0), category))
            logger.warning(f"Bias indicator [{category}]: '{match.group(0)}'")
    return findings


def is_biased(text: str) -> bool:
    """Quick boolean bias check."""
    return len(detect_bias(text)) > 0
