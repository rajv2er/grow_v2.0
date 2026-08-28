"""Metrics, model selection and plots derived only from executed experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from config import RESULTS_DIR
from ml.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def _metrics(y_true: pd.Series, probability: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = (probability >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if y_true.nunique() > 1 else None,
    }
    return result, confusion_matrix(y_true, prediction, labels=[0, 1])


def evaluate_models(models: dict, test: pd.DataFrame, output_dir: Path = RESULTS_DIR, report_name: str = "test") -> pd.DataFrame:
    """Evaluate fitted models on a named split; metrics are never hard-coded."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET_COLUMN]
    rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    for name, model in models.items():
        probability = model.predict_proba(x_test)[:, 1]
        metric, matrix = _metrics(y_test, probability)
        metric["model"] = name
        metric["confusion_matrix"] = matrix.tolist()
        rows.append(metric)
        fig, ax = plt.subplots(figsize=(4, 3))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not mastered", "Mastered"], yticklabels=["Not mastered", "Mastered"], xlabel="Predicted", ylabel="Actual", title=f"{name} — synthetic test set")
        for i in range(2):
            for j in range(2): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        if y_test.nunique() > 1:
            fpr, tpr, _ = roc_curve(y_test, probability)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC {metric['roc_auc']:.3f})")
    summary = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    ax_roc.plot([0, 1], [0, 1], "--", color="#64748b", label="No-skill")
    ax_roc.set(title=f"ROC curves — {report_name} split", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / f"{report_name}_roc_curves.png", dpi=150)
    plt.close(fig_roc)
    summary.to_csv(output_dir / f"{report_name}_model_comparison.csv", index=False)
    (output_dir / f"{report_name}_model_comparison.json").write_text(json.dumps(summary.to_dict("records"), indent=2), encoding="utf-8")
    return summary


def feature_importance(model, output_path: Path) -> pd.DataFrame:
    """Export model-supported importances; coefficients use absolute magnitude."""
    names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = np.abs(estimator.coef_[0])
    result = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False)
    result.to_csv(output_path, index=False)
    return result
