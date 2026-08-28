"""Student-level train/validation/test experiments for mastery prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MODELS_DIR
from ml.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from ml.preprocessing import build_preprocessor


def student_level_split(features: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Prevent a student's observations from appearing in both train and test."""
    group = features["student_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(outer.split(features, groups=group))
    train = features.iloc[train_idx].copy()
    holdout = features.iloc[holdout_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_idx, test_idx = next(inner.split(holdout, groups=holdout["student_id"]))
    return {"train": train, "validation": holdout.iloc[val_idx].copy(), "test": holdout.iloc[test_idx].copy()}


def train_model_suite(features: pd.DataFrame, output_dir: Path = MODELS_DIR, seed: int = 42) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Fit baseline and nonlinear models; evaluation happens on untouched test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features = features[features["prior_attempts"] >= 3].copy()
    if features.student_id.nunique() < 6:
        raise ValueError("At least six synthetic students are required for student-level evaluation.")
    splits = student_level_split(features, seed)
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", random_state=seed, n_jobs=-1,
        ),
    }
    trained = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipeline.fit(splits["train"][columns], splits["train"][TARGET_COLUMN])
        joblib.dump(pipeline, output_dir / f"{name}.joblib")
        trained[name] = pipeline
    metadata = {
        "label": "Models trained and evaluated on SYNTHETIC / SIMULATED data only.",
        "split": "GroupShuffleSplit by student: 70% train, 15% validation, 15% test",
        "feature_columns": columns,
        "target": "synthetic_mastered_label (latent mastery >= 0.70)",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return trained, splits
