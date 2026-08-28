"""Tests for the structured recommendation explainer."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ml.recommendation_explainer import (
    _recent_incorrect_streak,
    _stale_days,
    _trend,
    build_explanation,
    render_card_html,
    render_inline_html,
)


def _attempts(*rows):
    """Build an attempts DataFrame from (correct, time_taken, days_ago) tuples."""
    base = datetime.now(timezone.utc)
    data = []
    for i, (correct, seconds, days_ago) in enumerate(rows):
        ts = (base - timedelta(days=days_ago)).isoformat()
        data.append({
            "student_id": "U_TEST",
            "question_id": f"Q{i:03d}",
            "subject": "DSA",
            "topic": "Arrays",
            "difficulty": "Medium",
            "is_correct": int(bool(correct)),
            "time_taken_seconds": float(seconds),
            "attempt_number": i + 1,
            "timestamp": ts,
            "session_id": "U_TEST_PRACTICE",
            "confidence_rating": 3,
            "is_synthetic": 0,
            "answer_text": None,
            "score": None,
        })
    return pd.DataFrame(data)


def test_no_history_emits_no_history_signal():
    expl = build_explanation("DSA", "Arrays", 0.5, pd.DataFrame(), ema_estimate=None)
    assert expl["evidence_attempts"] == 0
    assert expl["signals"][0]["kind"] == "no_history"
    assert 0.0 <= expl["confidence"] <= 0.6


def test_accuracy_drop_detected():
    hist = _attempts(*[(1, 30, 10), (1, 30, 9), (1, 30, 8), (1, 30, 7),
                       (0, 60, 6), (0, 60, 5), (0, 60, 4), (0, 60, 3),
                       (0, 60, 2), (0, 60, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    kinds = [s["kind"] for s in expl["signals"]]
    assert "accuracy_drop" in kinds, f"Expected accuracy_drop in signals, got {kinds}"


def test_wrong_streak_detected():
    hist = _attempts(*[(1, 30, 5), (0, 30, 4), (0, 30, 3), (0, 30, 2), (0, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    kinds = [s["kind"] for s in expl["signals"]]
    assert "wrong_streak" in kinds
    streak_signal = next(s for s in expl["signals"] if s["kind"] == "wrong_streak")
    assert "4 consecutive" in streak_signal["label"]


def test_stale_topic_detected():
    hist = _attempts(*[(1, 30, 10), (1, 30, 8)])  # last attempt 8 days ago
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    kinds = [s["kind"] for s in expl["signals"]]
    assert "stale_topic" in kinds
    stale = next(s for s in expl["signals"] if s["kind"] == "stale_topic")
    assert "8 day" in stale["label"]


def test_slow_response_detected():
    hist = _attempts(*[(1, 150, 5), (1, 140, 4), (1, 130, 3), (1, 160, 2), (1, 145, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    kinds = [s["kind"] for s in expl["signals"]]
    assert "slow_response" in kinds


def test_low_mastery_emits_signal():
    hist = _attempts(*[(1, 30, 5), (1, 30, 4), (1, 30, 3), (1, 30, 2), (1, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.40, hist)
    assert any(s["kind"] == "low_mastery" for s in expl["signals"])


def test_ema_delta_emits_signal():
    hist = _attempts(*[(1, 30, 5), (1, 30, 4), (1, 30, 3), (1, 30, 2), (1, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.50, hist, ema_estimate=0.30)
    assert any(s["kind"] == "ema_lower" for s in expl["signals"])


def test_confidence_increases_with_evidence():
    sparse = build_explanation("DSA", "Arrays", 0.4, _attempts((1, 30, 1)))
    rich = build_explanation("DSA", "Arrays", 0.4, _attempts(*[(1, 30, 10 - i) for i in range(10)]))
    assert rich["confidence"] > sparse["confidence"]


def test_explanation_is_json_serializable():
    hist = _attempts(*[(1, 30, 5), (0, 30, 4), (0, 30, 3), (0, 30, 2), (0, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    json.dumps(expl)


def test_render_card_html_contains_signals():
    hist = _attempts(*[(1, 30, 5), (0, 30, 4), (0, 30, 3), (0, 30, 2), (0, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    html = render_card_html(expl)
    assert "Recommendation confidence" in html
    assert "Why we recommend" in html
    for s in expl["signals"]:
        assert s["label"] in html or _html_escape(s["label"]) in html


def test_render_inline_html_is_compact():
    hist = _attempts(*[(1, 30, 5), (0, 30, 4), (0, 30, 3), (0, 30, 2), (0, 30, 1)])
    expl = build_explanation("DSA", "Arrays", 0.4, hist)
    html = render_inline_html(expl)
    assert "Why this question?" in html
    assert "Confidence" in html


def test_helpers_handle_empty():
    assert _trend(pd.DataFrame()) == 0.0
    assert _recent_incorrect_streak(pd.DataFrame()) == 0
    assert _stale_days(pd.DataFrame()) is None


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
