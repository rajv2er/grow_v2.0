"""One command to reproduce a small synthetic simulation, ML comparison and recommendations."""
from __future__ import annotations

import json

from config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SimulationConfig
from database.db import write_dataframe
from ml.evaluate import evaluate_models, feature_importance
from ml.feature_engineering import build_feature_dataset
from ml.predict import predict_student_mastery
from ml.train import train_model_suite
from recommendation.recommender import recommend_questions
from simulator.learning_simulator import run_simulation


def main() -> None:
    config = SimulationConfig(number_of_students=30, attempts_per_student=60, random_seed=42)
    simulation = run_simulation(config)
    features = build_feature_dataset(simulation["attempts"], simulation["truth"])
    write_dataframe(features, PROCESSED_DIR / "features.csv")
    models, splits = train_model_suite(features, seed=config.random_seed)
    validation_comparison = evaluate_models(models, splits["validation"], report_name="validation")
    best = str(validation_comparison.loc[0, "model"])
    comparison = evaluate_models(models, splits["test"], report_name="test")
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    feature_importance(models[best], RESULTS_DIR / f"{best}_feature_importance.csv")
    student_id = str(simulation["students"].iloc[0].student_id)
    predictions = predict_student_mastery(MODELS_DIR / f"{best}.joblib", simulation["attempts"], simulation["questions"], student_id)
    recommendations = recommend_questions(student_id, predictions, simulation["attempts"], simulation["questions"])
    write_dataframe(predictions, PROCESSED_DIR / "predictions.csv")
    write_dataframe(recommendations, PROCESSED_DIR / "recommendations.csv")
    print("SYNTHETIC / SIMULATED DATA — NOT REAL STUDENT DATA")
    print(json.dumps(simulation["manifest"], indent=2))
    print(f"\nSelected on validation F1: {best}")
    print("\nHeld-out, student-level model comparison:")
    print(comparison[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))
    print("\nExample recommendations:")
    print(recommendations[["subject", "topic", "recommended_difficulty", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
