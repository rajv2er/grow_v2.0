"""Graders for the four new question types: TrueFalse, MultipleSelect, FillInBlank, Numerical.

All graders return a dict with at least:
  - is_correct: bool
  - score: float in [0, 1]
  - feedback: human-readable string
  - matched, missing: lists (where applicable)

TrueFalse and MCQ share a storage model (option_a/option_b + correct_answer),
so TrueFalse reuses the MCQ branch in `app/main.py` and is graded here
just for symmetry / direct testing.
"""
from __future__ import annotations

import json
from typing import Any

PASS_THRESHOLD = 0.6


def grade_true_false(question: dict, answer: str | None) -> dict[str, Any]:
    """Accept either an option letter ('A'/'B') or a True/False word."""
    raw = (answer or "").strip().lower()
    if raw in ("true", "t", "a"):
        chosen = "A"
    elif raw in ("false", "f", "b"):
        chosen = "B"
    else:
        chosen = ""
    correct = chosen == (question.get("correct_answer") or "").strip().upper()
    return {
        "is_correct": correct,
        "score": 1.0 if correct else 0.0,
        "feedback": ("Correct." if correct else f"Incorrect. The correct answer is {question.get('correct_answer')}."),
        "matched": ["true_false"] if correct else [],
        "missing": [] if correct else ["true_false"],
        "source": "true_false",
    }


def grade_multiple_select(question: dict, answer: str | None) -> dict[str, Any]:
    """answer is a list of option letters like ['A', 'C'] or a comma-separated string."""
    raw = answer or ""
    if isinstance(raw, str):
        given = {c.strip().upper() for c in raw.split(",") if c.strip()}
    else:
        given = {str(c).strip().upper() for c in raw}
    correct_raw = question.get("correct_answers_json") or "[]"
    try:
        correct = {c.strip().upper() for c in json.loads(correct_raw)}
    except (TypeError, ValueError):
        correct = set()
    if not correct:
        return {
            "is_correct": False, "score": 0.0, "feedback": "Question is missing a correct-answer set.",
            "matched": [], "missing": [], "source": "multiple_select",
        }
    matched = sorted(given & correct)
    missing_in_answer = sorted(correct - given)
    extra = sorted(given - correct)
    score = len(matched) / len(correct) if correct else 0.0
    # Selecting any wrong option fails the question — partial credit only
    # for partial selection of correct options.
    if extra:
        score = min(score, 0.5)
        if not missing_in_answer:
            score = min(score, 0.25)
    is_correct = score >= PASS_THRESHOLD and not extra
    feedback = (
        f"You selected {sorted(given) or 'nothing'}. Correct set: {sorted(correct)}."
    )
    return {
        "is_correct": is_correct, "score": round(score, 2),
        "feedback": feedback, "matched": matched, "missing": missing_in_answer,
        "extras": extra, "source": "multiple_select",
    }


def grade_fill_in_blank(question: dict, answer: str | None) -> dict[str, Any]:
    raw_blanks = question.get("blanks_json") or "[]"
    try:
        blanks = [str(b).strip().lower() for b in json.loads(raw_blanks)]
    except (TypeError, ValueError):
        blanks = []
    text = (answer or "").strip().lower()
    if not text:
        return {
            "is_correct": False, "score": 0.0,
            "feedback": "Empty answer.",
            "matched": [], "missing": blanks, "source": "fill_in_blank",
        }
    matched = [b for b in blanks if b in text]
    missing = [b for b in blanks if b not in text]
    score = (len(matched) / len(blanks)) if blanks else 0.0
    is_correct = score >= PASS_THRESHOLD and not missing
    if is_correct:
        feedback = "Correct."
    else:
        feedback = (
            f"Expected one of: {', '.join(blanks) or '(none)'}."
        )
    return {
        "is_correct": is_correct, "score": round(score, 2),
        "feedback": feedback, "matched": matched, "missing": missing,
        "source": "fill_in_blank",
    }


def grade_numerical(question: dict, answer: str | None) -> dict[str, Any]:
    try:
        expected = float(question.get("expected_value"))
    except (TypeError, ValueError):
        return {
            "is_correct": False, "score": 0.0, "feedback": "Question is missing expected_value.",
            "matched": [], "missing": [], "source": "numerical",
        }
    tolerance = float(question.get("tolerance") or 0.01)
    raw = (answer or "").strip().replace(",", "")
    try:
        given = float(raw)
    except ValueError:
        return {
            "is_correct": False, "score": 0.0,
            "feedback": f"Could not parse '{answer}' as a number. Expected: {expected}.",
            "matched": [], "missing": [str(expected)], "source": "numerical",
        }
    if expected == 0:
        rel_err = abs(given)
    else:
        rel_err = abs(given - expected) / abs(expected)
    is_correct = rel_err <= tolerance
    score = 1.0 if is_correct else max(0.0, 1.0 - rel_err)
    feedback = (
        f"Expected {expected} (±{tolerance:.0%} relative). You answered {given}."
    )
    return {
        "is_correct": is_correct, "score": round(score, 2),
        "feedback": feedback,
        "matched": [str(expected)] if is_correct else [],
        "missing": [] if is_correct else [str(expected)],
        "source": "numerical",
    }


def grade_any(question: dict, answer: Any) -> dict[str, Any]:
    """Dispatch by question_type. Returns the same dict shape as grade_subjective."""
    qtype = question.get("question_type")
    if qtype == "TrueFalse":
        return grade_true_false(question, answer)
    if qtype == "MultipleSelect":
        return grade_multiple_select(question, answer)
    if qtype == "FillInBlank":
        return grade_fill_in_blank(question, answer)
    if qtype == "Numerical":
        return grade_numerical(question, answer)
    raise ValueError(f"grade_any: unknown question_type {qtype!r}")
