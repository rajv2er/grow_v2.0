"""Transparent grading of free-text answers against a model answer.

The default grader is deterministic keyword-rubric coverage — inspectable and
reproducible for research. An external LLM grader can be plugged in via the
`llm` callable (answer, model_answer) -> score in [0, 1] without touching
callers; the rubric result is kept as the fallback whenever it returns nothing
usable.
"""
from __future__ import annotations

from typing import Callable

PASS_THRESHOLD = 0.6
MIN_CHARACTERS = 15

# Small built-in stopword set keeps grading dependency-free and deterministic.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "into", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "as", "not", "no", "can", "cannot", "could",
    "should", "would", "will", "shall", "may", "might", "must", "do", "does",
    "did", "have", "has", "had", "you", "your", "they", "their", "them", "we",
    "our", "i", "me", "my", "he", "she", "his", "her", "one", "two", "any",
    "all", "each", "every", "some", "such", "than", "so", "very", "just",
    "also", "only", "own", "same", "s", "t", "use", "used", "using", "example",
}


def _terms(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return {word for word in cleaned.split() if len(word) > 3 and word not in STOPWORDS}


def _matches(term: str, answer_terms: set[str]) -> bool:
    # Prefix match tolerates simple plurals/inflections ("tables" ~ "table").
    stem = term[:5] if len(term) > 5 else term
    return term in answer_terms or any(a.startswith(stem) for a in answer_terms)


def rubric_score(answer: str, model_answer: str) -> tuple[float, list[str], list[str]]:
    """Coverage of the model answer's key terms; short answers are capped at 0.5."""
    answer_text = (answer or "").strip()
    model_terms = _terms(model_answer)
    answer_terms = _terms(answer_text)
    matched = sorted(t for t in model_terms if _matches(t, answer_terms))
    missing = sorted(t for t in model_terms if not _matches(t, answer_terms))
    coverage = len(matched) / len(model_terms) if model_terms else 1.0
    if len(answer_text) < MIN_CHARACTERS:
        coverage = min(coverage, 0.5)
    return round(min(coverage, 1.0), 2), matched, missing


def grade_subjective(
    question: dict,
    answer: str,
    llm: Callable[[str, str], float] | None = None,
) -> dict:
    """Grade a free-text answer; returns score, pass flag, matched/missing terms and feedback."""
    model_answer = question.get("model_answer") or question.get("explanation", "")
    score, matched, missing = rubric_score(answer, model_answer)
    source = "rubric"
    if llm is not None:
        try:
            external = llm(answer or "", model_answer)
        except Exception:
            external = None
        if external is not None and 0.0 <= float(external) <= 1.0:
            score = round(float(external), 2)
            source = "llm"
    passed = score >= PASS_THRESHOLD
    if passed:
        feedback = f"Your answer covered the key ideas ({score:.0%} rubric coverage)."
        if missing:
            feedback += f" Consider also mentioning: {', '.join(missing[:4])}."
    else:
        detail = "the answer is too short" if len((answer or "").strip()) < MIN_CHARACTERS else f"these ideas were missing: {', '.join(missing[:4])}" if missing else "the key terms were not expressed"
        feedback = f"Review the topic — {detail}. Model answer: {model_answer}"
    return {"score": score, "is_correct": passed, "matched": matched, "missing": missing, "feedback": feedback, "source": source}
