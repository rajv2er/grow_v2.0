"""Build a structured, human-readable explanation of why a topic is recommended.

The recommender already produces a one-line `reason` string. This module
adds a richer explanation that the UI can render as a card with
checkmark-style bullet points and a confidence score. The data sources
are all real: the global model's features, the per-user EMA, and the
topic's recent attempt history.

Output shape (JSON-serialisable):

    {
      "headline": "Why we recommend Arrays",
      "signals": [
        {"kind": "accuracy_drop",  "label": "Accuracy dropped from 78% to 54%", "weight": 0.9},
        {"kind": "wrong_streak",   "label": "3 recent incorrect attempts",      "weight": 0.7},
        {"kind": "slow_response",  "label": "Average response time increased",  "weight": 0.5},
        {"kind": "stale_topic",    "label": "Topic hasn't been practised for 8 days", "weight": 0.6},
        ...
      ],
      "confidence": 0.87,
      "model_mastery": 0.42,
      "ema_mastery": 0.38,
      "evidence_attempts": 17
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")


def _trend(hist: pd.DataFrame) -> float:
    """Difference in rolling accuracy: second half minus first half of attempts."""
    if len(hist) < 4:
        return 0.0
    mid = len(hist) // 2
    return float(hist.is_correct.iloc[mid:].mean() - hist.is_correct.iloc[:mid].mean())


def _stale_days(hist: pd.DataFrame) -> int | None:
    if hist.empty:
        return None
    last = _parse_timestamp(hist.timestamp).max()
    if pd.isna(last):
        return None
    return int((datetime.now(timezone.utc) - last).total_seconds() // 86400)


def _recent_incorrect_streak(hist: pd.DataFrame) -> int:
    """Count of consecutive wrong answers from the most recent attempt backwards."""
    if hist.empty:
        return 0
    sorted_hist = hist.sort_values("timestamp", ascending=False)
    streak = 0
    for v in sorted_hist.is_correct:
        if int(v) == 0:
            streak += 1
        else:
            break
    return streak


def build_explanation(
    subject: str,
    topic: str,
    mastery_probability: float,
    topic_history: pd.DataFrame,
    ema_estimate: float | None = None,
) -> dict[str, Any]:
    """Build a structured explanation for one recommended topic."""
    signals: list[dict[str, Any]] = []
    if topic_history is None or topic_history.empty:
        return {
            "headline": f"Recommended: {topic}",
            "signals": [
                {
                    "kind": "no_history",
                    "label": "No attempts on this topic yet — start with an easy item to establish a baseline.",
                    "weight": 0.5,
                }
            ],
            "confidence": 0.5,
            "model_mastery": float(mastery_probability),
            "ema_mastery": None,
            "evidence_attempts": 0,
        }

    sorted_hist = topic_history.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_hist)
    recent5 = sorted_hist.tail(5)
    recent5_acc = float(recent5.is_correct.mean()) if len(recent5) else 0.0
    earlier_acc = float(sorted_hist.head(max(n - 5, 1)).is_correct.mean()) if n > 5 else recent5_acc
    avg_time = float(sorted_hist["time_taken_seconds"].mean()) if "time_taken_seconds" in sorted_hist.columns and not sorted_hist.empty else 0.0
    trend = _trend(sorted_hist)
    streak = _recent_incorrect_streak(sorted_hist)
    stale_days = _stale_days(sorted_hist)
    diff_acc = float(sorted_hist[sorted_hist.difficulty == "Medium"].is_correct.mean()) if (sorted_hist.difficulty == "Medium").any() else 0.0

    if mastery_probability < 0.55:
        signals.append({
            "kind": "low_mastery",
            "label": f"Predicted mastery is only {mastery_probability:.0%}",
            "weight": 0.95,
        })

    if n >= 6 and (earlier_acc - recent5_acc) >= 0.10:
        signals.append({
            "kind": "accuracy_drop",
            "label": f"Accuracy dropped from {earlier_acc:.0%} to {recent5_acc:.0%} over the last {len(recent5)} attempts",
            "weight": 0.9,
        })

    if streak >= 2:
        signals.append({
            "kind": "wrong_streak",
            "label": f"{streak} consecutive incorrect answers on this topic",
            "weight": min(0.4 + 0.1 * streak, 0.95),
        })

    if trend < -0.10:
        signals.append({
            "kind": "trend",
            "label": f"Performance trend is declining ({trend:+.0%} across recent attempts)",
            "weight": 0.7,
        })
    elif trend > 0.10:
        signals.append({
            "kind": "trend",
            "label": f"Performance is improving ({trend:+.0%} across recent attempts)",
            "weight": 0.4,
        })

    if avg_time > 90:
        signals.append({
            "kind": "slow_response",
            "label": f"Average response time is {avg_time:.0f} seconds — taking longer than the band",
            "weight": 0.5,
        })

    if 0 < diff_acc < 0.55:
        signals.append({
            "kind": "medium_difficulty",
            "label": f"Medium-difficulty accuracy on this topic is {diff_acc:.0%}",
            "weight": 0.6,
        })

    if stale_days is not None and stale_days >= 5:
        signals.append({
            "kind": "stale_topic",
            "label": f"Topic hasn't been practised for {stale_days} day{'s' if stale_days != 1 else ''}",
            "weight": min(0.4 + 0.05 * stale_days, 0.8),
        })

    if ema_estimate is not None and abs(ema_estimate - mastery_probability) > 0.10:
        if ema_estimate < mastery_probability:
            signals.append({
                "kind": "ema_lower",
                "label": f"Your recent answers put mastery even lower at {ema_estimate:.0%}",
                "weight": 0.7,
            })
        else:
            signals.append({
                "kind": "ema_higher",
                "label": f"Your recent answers suggest mastery is {ema_estimate:.0%} — better than the model predicts",
                "weight": 0.5,
            })

    if not signals:
        signals.append({
            "kind": "exploration",
            "label": f"Limited strong evidence; continuing to gather signal at the {mastery_probability:.0%} mastery level",
            "weight": 0.3,
        })

    weights = [s["weight"] for s in signals]
    confidence = min(0.99, 0.4 + 0.6 * (sum(weights) / max(len(weights), 1)))
    if n < 3:
        confidence = min(confidence, 0.5)
    elif n < 6:
        confidence = min(confidence, 0.7)

    return {
        "headline": f"Why we recommend {topic}",
        "signals": sorted(signals, key=lambda s: -s["weight"]),
        "confidence": round(confidence, 2),
        "model_mastery": float(mastery_probability),
        "ema_mastery": float(ema_estimate) if ema_estimate is not None else None,
        "evidence_attempts": int(n),
    }


def render_card_html(expl: dict[str, Any]) -> str:
    """Render the explanation dict as a styled HTML card.

    The card shows the headline, one row per signal with a checkmark,
    a confidence bar, and a footer with the model/EMA mastery and the
    number of attempts the explanation is based on.
    """
    signals_html = "".join(
        f"<li style='margin:4px 0;color:#e6ecf7'><span style='color:#4ade80;margin-right:8px'>&#x2713;</span>{_escape(s['label'])}</li>"
        for s in expl["signals"]
    )
    confidence = float(expl["confidence"])
    mastery = expl["model_mastery"]
    ema = expl.get("ema_mastery")
    evidence = expl["evidence_attempts"]
    mastery_line = (
        f"Model mastery <b>{mastery:.0%}</b>"
        + (f" · EMA <b>{ema:.0%}</b>" if ema is not None else "")
        + f" · based on <b>{evidence}</b> attempt{'s' if evidence != 1 else ''}"
    )
    return (
        "<div class='ml-card-elev' style='border-left:3px solid #6366f1'>"
        f"<div style='font-weight:600;font-size:1.05rem'>{_escape(expl['headline'])}</div>"
        f"<ul style='list-style:none;padding-left:0;margin:10px 0 0 0'>{signals_html}</ul>"
        "<div style='margin-top:14px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;font-size:0.8rem;color:#9aa6c0'>"
        f"<span>Recommendation confidence</span><span><b>{confidence:.0%}</b></span></div>"
        f"<div class='ml-progress' style='margin-top:4px'><div style='width:{confidence*100:.1f}%'></div></div>"
        f"<div style='color:#9aa6c0;font-size:0.78rem;margin-top:8px'>{mastery_line}</div>"
        "</div></div>"
    )


def render_inline_html(expl: dict[str, Any]) -> str:
    """Compact one-line version for the pre-question card."""
    top = expl["signals"][:2]
    bullets = " · ".join(s["label"] for s in top)
    return (
        f"<div class='ml-card' style='border-left:3px solid #6366f1;padding:12px 14px'>"
        f"<div style='font-weight:600;font-size:0.92rem'>Why this question?</div>"
        f"<div style='color:#cbd5e1;font-size:0.85rem;margin-top:4px'>{_escape(bullets)}</div>"
        f"<div style='color:#9aa6c0;font-size:0.78rem;margin-top:6px'>"
        f"Confidence <b>{expl['confidence']:.0%}</b> · {expl['evidence_attempts']} attempt{'s' if expl['evidence_attempts'] != 1 else ''}"
        f"</div></div>"
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
