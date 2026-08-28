"""Run reproducible ML model comparisons on a simulated cohort."""
from __future__ import annotations

import json

from analytics.reports import dataset_statistics, learning_curve, performance_by_subject_topic
from config import PROCESSED_DIR, RESULTS_DIR, SimulationConfig
from database.db import write_dataframe
from experiments.tracking import record as record_run
from ml.evaluate import evaluate_models, feature_importance
from ml.feature_engineering import build_feature_dataset
from ml.train import train_model_suite
from observability import log
from simulator.learning_simulator import run_simulation


def run_experiment_from_data(
    attempts,
    truth,
    seed: int = 42,
    simulation_manifest: dict | None = None,
):
    """Fit and evaluate exclusively from the supplied synthetic run.

    This keeps the training inputs connected to the cohort the researcher chose
    in the application, rather than replacing it with a hidden fixed cohort.
    """
    features = build_feature_dataset(attempts, truth)
    write_dataframe(features, PROCESSED_DIR / "features.csv")
    models, splits = train_model_suite(features, seed=seed)
    validation_comparison = evaluate_models(models, splits["validation"], report_name="validation")
    best = str(validation_comparison.loc[0, "model"])
    comparison = evaluate_models(models, splits["test"], report_name="test")
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    feature_importance(models[best], RESULTS_DIR / f"{best}_feature_importance.csv")
    subject, topic = performance_by_subject_topic(attempts)
    write_dataframe(subject, RESULTS_DIR / "subject_performance.csv")
    write_dataframe(topic, RESULTS_DIR / "topic_performance.csv")
    learning_curve(attempts, RESULTS_DIR / "learning_curve.png")
    report = {"simulation": simulation_manifest, "dataset_statistics": dataset_statistics(attempts), "best_model_by_validation_f1": best, "metrics_file": str(RESULTS_DIR / "model_comparison.csv")}
    (RESULTS_DIR / "experiment_1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        record_run(
            simulation_manifest=simulation_manifest,
            dataset_statistics=dataset_statistics(attempts),
            best_model=best,
            model_comparison=comparison,
            feature_names=list(features.columns),
        )
    except Exception as exc:  # tracking failure must not break the experiment
        log.warning("experiment tracking failed: %s", exc)
    return features, comparison, best


def run_experiment(config: SimulationConfig = SimulationConfig()):
    """Generate a reproducible synthetic cohort, then evaluate it."""
    simulation = run_simulation(config)
    features, comparison, best = run_experiment_from_data(
        simulation["attempts"], simulation["truth"], config.random_seed, simulation["manifest"]
    )
    return simulation, features, comparison, best


if __name__ == "__main__":
    simulation, features, comparison, best = run_experiment()
    print(f"Synthetic run {simulation['run_id']} complete. Best held-out F1 model: {best}")
    print(comparison[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))
