"""Thin FastAPI layer over the same code paths the Streamlit UI uses.

This is not a separate frontend — the API and the UI share the service
layer. Endpoints here are intended to be called by:
  - external scripts (e.g. a Jupyter notebook analyzing a single learner's
    progress)
  - a future mobile app
  - CI / smoke tests
  - any third-party research consumer who wants to integrate the recommender
    without depending on Streamlit

Run with:
    uvicorn api.main:app --reload

Endpoints:
  GET  /healthz
  POST /students/{student_id}/mastery      — current mastery, EMA-overlaid
  GET  /students/{student_id}/recommendations?limit=5
  POST /students/{student_id}/attempts     — record an answer + EMA update
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import service
from data.questions.question_bank import build_question_bank

app = FastAPI(title="MasteryLab API", version="1.0")

_QUESTIONS = pd.DataFrame(build_question_bank())


class AnswerIn(BaseModel):
    question_id: str
    is_correct: bool
    seconds: float = Field(ge=0.0)
    confidence: int = Field(ge=1, le=5)
    answer_text: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


def _row_to_dict(row) -> dict[str, Any]:
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_row_to_dict(row) for row in frame.to_dict(orient="records")]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/students/{student_id}/mastery")
def get_mastery(student_id: str) -> dict[str, Any]:
    frame = service.current_mastery(student_id)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No mastery data for student {student_id}")
    return {"student_id": student_id, "topics": _frame_to_records(frame)}


@app.get("/students/{student_id}/recommendations")
def get_recommendations(student_id: str, limit: int = 5) -> dict[str, Any]:
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be 1..50")
    frame = service.next_recommendations(student_id, limit=limit)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations available for student {student_id}")
    return {"student_id": student_id, "limit": limit, "recommendations": _frame_to_records(frame)}


@app.post("/students/{student_id}/attempts", status_code=201)
def post_attempt(student_id: str, payload: AnswerIn) -> dict[str, Any]:
    q = _QUESTIONS[_QUESTIONS.question_id == payload.question_id]
    if q.empty:
        raise HTTPException(status_code=404, detail=f"Unknown question {payload.question_id}")
    service.record_answer(
        student_id,
        q.iloc[0],
        is_correct=payload.is_correct,
        seconds=payload.seconds,
        confidence=payload.confidence,
        answer_text=payload.answer_text,
        score=payload.score,
    )
    return {"status": "recorded", "student_id": student_id, "question_id": payload.question_id}
