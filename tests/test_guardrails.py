"""
tests/test_guardrails.py
─────────────────────────
Tests for Constitutional AI guardrails.
"""

from __future__ import annotations

import pytest
from src.guardrails.constitutional_ai import check_faithfulness, scrub_pii
from src.guardrails.bias_detector import detect_bias, is_biased


class TestFaithfulness:

    def test_high_overlap_is_faithful(self):
        answer = "Photosynthesis uses sunlight water and carbon dioxide to produce glucose."
        sources = ["Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen."]
        faithful, score = check_faithfulness(answer, sources, threshold=0.3)
        assert faithful is True
        assert score > 0.3

    def test_low_overlap_not_faithful(self):
        answer = "The president declared war on neighboring countries yesterday morning."
        sources = ["Photosynthesis uses sunlight to produce glucose in plant cells."]
        faithful, score = check_faithfulness(answer, sources, threshold=0.3)
        assert faithful is False

    def test_empty_sources_always_faithful(self):
        faithful, score = check_faithfulness("any answer", [], threshold=0.5)
        assert faithful is True
        assert score == 1.0


class TestBiasDetector:

    def test_no_bias_clean_text(self):
        text = "The study examined crop yields across different regions over three years."
        assert is_biased(text) is False

    def test_gender_stereotype_detected(self):
        text = "Only men can lead engineering teams effectively."
        findings = detect_bias(text)
        assert len(findings) > 0
        categories = [f[1] for f in findings]
        assert "gender" in categories

    def test_clean_text_no_findings(self):
        text = "Machine learning models can improve agricultural forecasting accuracy."
        findings = detect_bias(text)
        assert len(findings) == 0


class TestPIIScrubbing:

    def test_ssn_scrubbed(self):
        text = "SSN: 123-45-6789"
        scrubbed = scrub_pii(text)
        assert "123-45-6789" not in scrubbed
        assert "[SSN_REDACTED]" in scrubbed

    def test_card_scrubbed(self):
        text = "Card: 4111 1111 1111 1111"
        scrubbed = scrub_pii(text)
        assert "4111" not in scrubbed or "[CARD_REDACTED]" in scrubbed

    def test_no_pii_unchanged(self):
        text = "Revenue grew 23% year-over-year in Q3 2023."
        assert scrub_pii(text) == text
