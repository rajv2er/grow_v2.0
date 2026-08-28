"""Dataset summaries and learning-progression plots, labelled synthetic."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def dataset_statistics(attempts: pd.DataFrame) -> dict:
    return {
        "data_label": "SYNTHETIC / SIMULATED — NOT REAL STUDENT DATA",
        "attempts": int(len(attempts)), "students": int(attempts.student_id.nunique()),
        "subjects": int(attempts.subject.nunique()), "topics": int(attempts[["subject", "topic"]].drop_duplicates().shape[0]),
        "overall_accuracy": float(attempts.is_correct.mean()),
        "mean_response_seconds": float(attempts.time_taken_seconds.mean()),
    }


def performance_by_subject_topic(attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = attempts.groupby("subject", as_index=False).agg(accuracy=("is_correct", "mean"), mean_response_seconds=("time_taken_seconds", "mean"), attempts=("is_correct", "size"))
    topic = attempts.groupby(["subject", "topic"], as_index=False).agg(accuracy=("is_correct", "mean"), mean_response_seconds=("time_taken_seconds", "mean"), attempts=("is_correct", "size"))
    return subject, topic


def learning_curve(attempts: pd.DataFrame, output_path: Path) -> Path:
    """Plot observed (synthetic) rolling correctness over attempt sequence."""
    data = attempts.copy().sort_values(["student_id", "attempt_number"])
    grouped = data.groupby("attempt_number", as_index=False).is_correct.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped.attempt_number, grouped.is_correct.rolling(5, min_periods=1).mean(), color="#2563eb")
    ax.set(title="Synthetic learning curve", xlabel="Attempt number", ylabel="Mean rolling correctness", ylim=(0, 1))
    ax.text(0.99, 0.03, "Synthetic / simulated data — not real students", transform=ax.transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
