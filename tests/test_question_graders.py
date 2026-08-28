"""Tests for the four new question-type graders."""
from __future__ import annotations

import json

import pytest

from ml.question_graders import (
    grade_any,
    grade_fill_in_blank,
    grade_multiple_select,
    grade_numerical,
    grade_true_false,
)


def _q(qtype, **kw) -> dict:
    base = {
        "question_id": "QTEST", "subject": "DSA", "topic": "Arrays",
        "question": "Q", "question_type": qtype, "difficulty": "Easy",
        "difficulty_rating": 0.25, "option_a": None, "option_b": None,
        "option_c": None, "option_d": None, "correct_answer": None,
        "model_answer": None, "explanation": "E",
        "blanks_json": None, "correct_answers_json": None,
        "expected_value": None, "tolerance": 0.01,
    }
    base.update(kw)
    return base


# ---- TrueFalse ----

def test_true_false_correct():
    q = _q("TrueFalse", option_a="True", option_b="False", correct_answer="A")
    assert grade_true_false(q, "A")["is_correct"] is True
    assert grade_true_false(q, "B")["is_correct"] is False
    assert grade_true_false(q, "true")["is_correct"] is True
    assert grade_true_false(q, None)["is_correct"] is False


# ---- MultipleSelect ----

def test_multiple_select_exact_match():
    q = _q("MultipleSelect", correct_answers_json=json.dumps(["A", "D"]))
    assert grade_multiple_select(q, ["A", "D"])["is_correct"] is True
    assert grade_multiple_select(q, "A,D")["is_correct"] is True
    assert grade_multiple_select(q, ["A", "B"])["is_correct"] is False


def test_multiple_select_penalises_extras():
    q = _q("MultipleSelect", correct_answers_json=json.dumps(["A", "D"]))
    res = grade_multiple_select(q, ["A", "B", "D"])
    assert res["is_correct"] is False
    assert res["score"] < 0.5
    assert "B" in res["extras"]


def test_multiple_select_partial_credit():
    q = _q("MultipleSelect", correct_answers_json=json.dumps(["A", "C", "D"]))
    res = grade_multiple_select(q, ["A"])
    assert res["score"] == pytest.approx(1 / 3, rel=0.01)
    assert res["is_correct"] is False


# ---- FillInBlank ----

def test_fill_in_blank_exact():
    q = _q("FillInBlank", blanks_json=json.dumps(["base"]))
    assert grade_fill_in_blank(q, "base")["is_correct"] is True
    assert grade_fill_in_blank(q, "Base")["is_correct"] is True
    assert grade_fill_in_blank(q, "I don't know")["is_correct"] is False
    assert grade_fill_in_blank(q, "")["is_correct"] is False


def test_fill_in_blank_within_sentence():
    q = _q("FillInBlank", blanks_json=json.dumps(["base"]))
    res = grade_fill_in_blank(q, "Every recursive function needs a base case to terminate.")
    assert res["is_correct"] is True


# ---- Numerical ----

def test_numerical_exact():
    q = _q("Numerical", expected_value=32.0, tolerance=0.0)
    assert grade_numerical(q, "32")["is_correct"] is True
    assert grade_numerical(q, "32.0")["is_correct"] is True
    assert grade_numerical(q, "33")["is_correct"] is False


def test_numerical_within_tolerance():
    q = _q("Numerical", expected_value=100.0, tolerance=0.05)
    assert grade_numerical(q, "104")["is_correct"] is True
    assert grade_numerical(q, "110")["is_correct"] is False


def test_numerical_rejects_non_numeric():
    q = _q("Numerical", expected_value=32.0, tolerance=0.01)
    res = grade_numerical(q, "thirty-two")
    assert res["is_correct"] is False
    assert res["score"] == 0.0


# ---- Dispatch ----

def test_grade_any_dispatches():
    q_tf = _q("TrueFalse", option_a="True", option_b="False", correct_answer="A")
    assert grade_any(q_tf, "A")["is_correct"] is True

    q_fb = _q("FillInBlank", blanks_json=json.dumps(["alpha"]))
    assert grade_any(q_fb, "alpha")["is_correct"] is True

    q_nu = _q("Numerical", expected_value=10.0, tolerance=0.01)
    assert grade_any(q_nu, "10")["is_correct"] is True

    with pytest.raises(ValueError):
        grade_any(_q("MCQ"), "A")
