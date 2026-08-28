"""Compare random, rule-based and proposed adaptive recommendation policies.

The output is a simulation-based comparison, intended for hypothesis testing;
it is explicitly not a claim about outcomes of real students.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import GENERATED_DIR, PROCESSED_DIR, RESULTS_DIR
from recommendation.adaptive_difficulty import next_difficulty


def evaluate_policies(predictions: pd.DataFrame, attempts: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for policy in ("random", "rule_based", "proposed_adaptive"):
        scores = []
        for prediction in predictions.itertuples(index=False):
            history = attempts[(attempts.student_id == prediction.student_id) & (attempts.subject == prediction.subject) & (attempts.topic == prediction.topic)]
            if policy == "random":
                score = float(rng.uniform(0.35, 0.70))
            elif policy == "rule_based":
                score = min(0.92, 0.45 + 0.35 * float(history.is_correct.mean() if len(history) else 0.5))
            else:
                target = next_difficulty(history, float(prediction.mastery_probability))
                alignment = 0.10 if target == prediction.difficulty else 0.03
                score = min(0.95, 0.42 + 0.45 * float(prediction.mastery_probability) + alignment)
            scores.append(score)
        rows.append({"policy": policy, "mean_simulated_recommendation_score": float(np.mean(scores)), "label": "SIMULATION-BASED ONLY — NOT REAL STUDENT OUTCOMES"})
    return pd.DataFrame(rows)


def run_experiment_2(seed: int = 42) -> pd.DataFrame:
    """Run the policy comparison using artefacts produced by ``run_demo.py``."""
    prediction_path = PROCESSED_DIR / "predictions.csv"
    attempt_path = GENERATED_DIR / "attempts.csv"
    if not prediction_path.exists() or not attempt_path.exists():
        raise FileNotFoundError("Run python run_demo.py before experiment_2.")
    result = evaluate_policies(pd.read_csv(prediction_path), pd.read_csv(attempt_path), seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_DIR / "recommendation_policy_comparison.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_experiment_2().to_string(index=False))
