# AGENTS.md

Operating notes for AI coding agents working in this repository.

## What this project is

A BTech major-project research prototype: an adaptive multi-subject learning
and weakness-detection system. The product surface is a Streamlit dashboard
("MasteryLab") backed by a SQLite database, a synthetic student simulator, an
ML mastery-prediction suite (LR / RF / XGBoost), a novelty-aware recommender,
and a spaced-practice queue. All student records in the seed dataset are
**synthetic / simulated** — see `README.md` for the research-integrity
caveats.

## Quick start

```bash
cd /Users/rajveer/IdeaProjects/grow_v2.0
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
streamlit run app/main.py
```

`run_demo.py` populates `data/generated/`, writes `database/learning_system.db`,
trains models, and produces experiment artefacts under `experiments/results/`.
The Streamlit app works against the existing DB; re-run `run_demo.py` whenever
config changes to refresh artefacts.

## Stack

- Python 3.12, Streamlit, SQLite, pandas, scikit-learn, XGBoost, matplotlib
- No PostgreSQL, no FastAPI, no JWT — the original spec mentioned those but
  this project is intentionally lightweight to stay reproducible for viva.
- Database: `database/learning_system.db` (SQLite, WAL mode)
- Models: `models/*.joblib` + `models/training_metadata.json`

## Layout

```
app/main.py                 # Streamlit front-end (single file)
data/questions/             # 150 MCQ + 50 subjective questions across 5 subjects
database/                   # schema, db helpers, SQLite file
ml/                         # features, preprocessing, training, evaluation, inference,
                            # subjective grading (keyword rubric)
recommendation/             # adaptive difficulty ladder + novelty-aware recommender
simulator/                  # latent-skill student + practice-history generator
experiments/                # experiment_1 runner, results CSVs, ROC + CM PNGs
analytics/                  # dataset summaries + learning curve
tests/                      # pytest suite (31 tests)
conftest.py                 # adds the repo root to sys.path for pytest
```

## Conventions

- **No fabricated metrics anywhere in the UI.** Every number rendered in
  Streamlit comes from the database, a trained model, or the simulator. If
  data is missing, show a clear empty state.
- **Synthetic data is always labelled.** `app/main.py` shows the synthetic
  notice banner on Simulation, ML Experiments, and Research Analytics pages.
- **Timestamps are ISO 8601 strings in the DB** (e.g.
  `2026-08-28T04:14:02.269102+00:00`). When parsing them in pandas use
  `pd.to_datetime(series, format="ISO8601", utc=True)` — the old
  `pd.to_datetime(..., utc=True)` default format rejects the `+00:00` suffix.
  See `ml/feature_engineering.py:103` and the call sites in `app/main.py`.
- **Charts:** use `st.bar_chart`, `st.line_chart`, and `st.dataframe` with
  `width="stretch"` (not the deprecated `use_container_width=True`).
- **UI primitives** live in `app/main.py`: `inject_styles`, `card`, `kpi`,
  `pill`, `section_label`, `empty_state`, `info_banner`. New pages should
  compose these instead of inlining HTML/CSS.
- **Streamlit AppTest** is the right tool for headless page-level smoke tests
  (catches `itertuples()` column-name mismatches and similar runtime errors
  that pure `import` cannot).

## Demo student

Real users live in `students` rows with `is_synthetic=0` and `student_id`
prefix `U` (`U00001`, `U00002`, `U00003`, …). The login page exposes a
**Use Demo Student** button that:

1. Looks up `U00003` (the first real account with prior attempts).
2. Calls `ensure_demo_warmup` which seeds up to 30 simulator-generated
   attempts if the user has fewer than that.
3. Signs the user in.

The seeder is **idempotent** — it only inserts the delta.

## Test commands

```bash
pytest -q                              # 31 tests
streamlit run app/main.py              # run the UI
python run_demo.py                     # regenerate synthetic cohort + retrain
python -m experiments.experiment_1     # model-only comparison
```

## Common pitfalls

- `pd.to_datetime(..., utc=True)` without a format string breaks on ISO 8601
  strings that include `+00:00`. Use `format="ISO8601"`.
- `simulator.learning_simulator.run_simulation` returns a `truth` DataFrame
  with column `synthetic_true_mastery` (not `mastery`).
- `recommendation.recommender.recommend_questions` writes a snapshot to the
  `recommendations` table — it deletes prior rows for the student first.
- `ml.predict.predict_student_mastery` also deletes prior `mastery_predictions`
  rows for the same student+model_name before inserting the new snapshot.
- `practice_queue` is FIFO due on `due_at`; `complete_queue_item` marks a row
  as `completed`. The Quiz page consults it before falling through to
  recommender-driven selection.

## Where to make changes

| Task | File |
|---|---|
| Add a new page | `app/main.py` — new function, register in `main()` and `PAGES` |
| Add a UI primitive | `app/main.py` — extend `inject_styles` CSS + helper |
| Add a new feature for the ML model | `ml/feature_engineering.py` |
| Add a new metric | `ml/evaluate.py` + extend `experiments/experiment_1.py` |
| Add a new recommender heuristic | `recommendation/recommender.py` |
| Add a new simulator parameter | `config.py` `SimulationConfig` + `simulator/learning_simulator.py` |
| Add a new schema column | `database/schema.py` (also add a migration in `database/db.py:initialise_database`) |
