# Model Card — MasteryLab Mastery Classifier

This card documents the trained mastery-classification model that powers the
"LearnAI" / "MasteryLab" adaptive learning system. It follows the
[model-card schema](https://arxiv.org/abs/1810.03993) introduced by Mitchell
et al. and is the single source of truth for what the model is, who it is
for, and where it should not be used.

---

## Model details

| Field | Value |
|---|---|
| **Model name** | `random_forest` (also produced: `logistic_regression`, `xgboost`, `gradient_boosting`) |
| **Version** | 1.0 — produced by `experiments/experiment_1.py` |
| **Type** | Tabular binary classifier (mastered vs. not-mastered) |
| **Selected on** | Held-out validation F1 score |
| **Reported on** | Held-out test students (GroupShuffleSplit, 70/15/15) |
| **Output** | `mastery_probability` ∈ [0, 1] per (subject, topic) |
| **Status thresholds** | `Strong` ≥ 0.75, `Developing` ≥ 0.55, `Weak` < 0.55 |
| **Framework** | scikit-learn 1.x, XGBoost 2.x, joblib 1.x |
| **Artifact** | `models/{model_name}.joblib` |
| **Reproducibility** | `python run_demo.py` → deterministic synthetic cohort + retrain |

---

## Intended use

### Primary use

- **Per-topic mastery estimation** for learners in a structured BTech CS
  curriculum spanning DSA, DBMS, Operating Systems, Computer Networks, and
  Software Engineering.
- **Adaptive question selection** for a single-learner practice loop:
  difficulty is targeted to `mastery ± 0.15`, novelty is preferred, and
  struggling topics are prioritised.
- **Research prototype** for a BTech major project; demonstrated live in
  the Streamlit UI under `app/main.py`.

### Out-of-scope use

- **Any deployment to real learners** without independent validation on
  ethically approved, real-world assessment data.
- **High-stakes decisions** (grading, course placement, certification).
- **Subjects outside the five CS domains** in the question bank.
- **Concurrent multi-learner or classroom-scale orchestration** — the
  recommender currently serves a single learner at a time.

---

## Training data

| Field | Value |
|---|---|
| **Source** | Synthetic — `simulator/learning_simulator.py` |
| **Scale** | 30 → 10,000 students (configurable; default 30 + 60 attempts/student) |
| **Labels** | `synthetic_mastered_label = (latent_mastery ≥ 0.70)` |
| **Split** | GroupShuffleSplit by student (no row-level leakage) |
| **Features (15)** | `prior_attempts`, `overall_accuracy`, `subject_accuracy`, `topic_accuracy`, `recent_accuracy_5`, `recent_accuracy_10`, `topic_avg_response_time`, `topic_response_time_std`, `difficulty_accuracy`, `previous_correct`, `improvement_trend`, `hours_since_topic_attempt`, `subject`, `topic`, `difficulty` |

**Important:** the labels come from a latent-skill simulator, not from real
student outcomes. The model is therefore a *model of synthetic behaviour*,
not of real learning.

---

## Evaluation

Reported by `experiments/experiment_1.py` on the held-out test students
of the most recent cohort (see `experiments/results/test_model_comparison.csv`):

- **Accuracy, Precision, Recall, F1, ROC-AUC** — per model
- **Confusion matrices** — per model (`experiments/results/*_confusion_matrix.png`)
- **ROC curves** — `experiments/results/test_roc_curves.png`
- **Feature importance** — per model (`experiments/results/*_feature_importance.csv`)

Metrics are computed and stored by the run; they are **never hard-coded**
in the UI. Every number rendered in Streamlit comes from the CSV or the
trained model artifact.

---

## Personalisation

The trained classifier is applied per-learner via two layers:

1. **Global inference** — `ml.predict.predict_student_mastery` rebuilds
   feature rows from the learner's attempt history and runs the model.
2. **Online mastery** — `ml/online_mastery.py` maintains a per-(student,
   subject, topic) logistic-domain EMA that updates after every answer,
   initialised from the global model's prior. When the EMA has ≥3
   observations on a topic, it overrides the global model's probability.

This means the system is **adaptive at inference time** even though the
classifier weights themselves are not retrained on the user. See
`tests/test_recommender_e2e.py` for the end-to-end test that exercises
this loop.

---

## Limitations and failure modes

| Failure mode | When it happens | Mitigation |
|---|---|---|
| **Cold start** | Brand-new learner, 0 attempts | Recommender falls back to `initial_diagnostic` (Easy MCQ in the requested subject) until ≥1 attempt exists |
| **Sparse history** | 1-2 attempts on a topic | Global model is used; EMA needs ≥3 attempts before overriding |
| **Noisy early predictions** | First few attempts have high variance | The recommender's `TOPIC_HOLD_QUESTIONS` setting keeps the user on one topic for 3 consecutive items to stabilise the signal |
| **Latency on first prediction** | Cold joblib load | `models/` is small (< 1 MB per artifact); load is one-time per session |
| **Subject/topic outside bank** | Recommender asked about unknown topic | `recommend_questions` returns an empty frame; UI shows empty state |
| **Same user, multiple devices** | Concurrent writes to the same `attempts` row | SQLite WAL mode is sufficient for single-user local use; multi-device would need a server-grade DB |

---

## Ethical considerations

- All training and test data is **synthetic**. No real student records
  are used at any point in the pipeline.
- The system must **never be presented as evidence about real learners**
  or used to make decisions about real students without independent
  validation on real, ethically approved data.
- The recommender and difficulty policy are inspectable
  (`recommendation/recommender.py`, `recommendation/adaptive_difficulty.py`).
  Every recommendation has a human-readable `reason` field that is shown
  in the UI.
- Subjective answers are graded by a transparent keyword rubric
  (`ml/subjective_grading.py`) with a documented `PASS_THRESHOLD = 0.6`,
  not by an opaque LLM.

---

## Versioning and updates

| Source of change | Process |
|---|---|
| Question bank | Edit `data/questions/question_bank.py`; re-run `run_demo.py` to refresh the seeded `questions` table |
| Simulator | Edit `simulator/learning_simulator.py`; re-run `run_demo.py` to regenerate the cohort |
| Features | Edit `ml/feature_engineering.py`; re-run `experiments/experiment_1.py` to retrain |
| Recommender policy | Edit `recommendation/recommender.py`; the new policy is picked up on the next `recommend_questions` call |
| UI | Edit `app/main.py`; Streamlit hot-reloads |

Every training run appends a record to `experiments/tracking/runs.jsonl`
with the cohort hash, hyperparameters, and metric snapshot, so that
historical comparison is auditable.

---

## Contact

This is a BTech major-project research prototype. For questions, raise
an issue in the project repository.
