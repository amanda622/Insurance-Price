# Insurance Price Prediction

Predicts annual health-insurance charges from a person's attributes (age, sex,
BMI, children, smoker, region). A scikit-learn regression pipeline is served
through a FastAPI JSON API and a small web UI, and every prediction is logged to
a database.

![CI](https://github.com/amanda622/Insurance-Price/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Prediction API and web UI** — `POST /predict` for JSON and an HTML form,
  both backed by the same model service.
- **Reproducible model selection** — the served pipeline was chosen by comparing
  candidates under a shared cross-validation split; the experiment is committed
  and byte-for-byte reproducible.
- **Output safeguard** — predictions are clamped to non-negative at the serving
  boundary, with the discovered failure locked in by a regression test.
- **Prediction logging** — each request and response is persisted (SQLite
  locally, PostgreSQL via Docker).

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m insurance_price.train               # writes models/model.joblib + metrics.json
uvicorn insurance_price.api.app:app --reload  # http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000/` for the web form or `/docs` for interactive API
docs. A `Makefile` wraps common commands: `make train`, `make test`, `make lint`,
`make run`.

## API

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"female","bmi":27.5,"children":2,"smoker":"no","region":"southwest"}'
```

```json
{ "predicted_charge": 6247.92, "currency": "USD" }
```

Invalid input (e.g. `age: 200` or `smoker: "maybe"`) returns HTTP `422`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/predict` | JSON prediction |
| `GET` | `/` | Web prediction form |
| `POST` | `/web/predict` | Form submission → result page |
| `GET` | `/predictions?limit=20` | Recent logged predictions |

## Model selection

The served model is `OneHotEncoder → PolynomialFeatures(degree=2) →
LinearRegression`.
[`experiments/compare_models.py`](experiments/compare_models.py) compares five
candidates under one shared `RepeatedKFold` split (5 folds × 5 repeats = 25
identical folds, so every candidate sees exactly the same data) and reports
mean / std / per-fold RMSE, MAE and R². Full numbers:
[`experiments/results/results.json`](experiments/results/results.json).

| Candidate | What it is | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| B1 | Mean baseline (`DummyRegressor`) | 12,100 | 9,099 | -0.004 |
| B2 | One-hot + `LinearRegression` (no polynomial) | 6,084 | 4,209 | 0.744 |
| B3 | Polynomial on numeric features only | 6,059 | 4,262 | 0.746 |
| **B4** | **Served model: polynomial on all features** | **4,867** | **2,938** | **0.835** |
| B5 | Random forest reference (fixed, untuned) | 4,836 | 2,674 | 0.837 |

*RMSE / MAE in USD, lower is better; mean over 25 folds.*

- B4 more than halves the mean-baseline RMSE (12,100 → 4,867).
- The polynomial step beats plain linear (B2) by ~20% RMSE on all 25 folds. The
  lift comes from category interactions (e.g. `smoker × bmi`, `smoker × age`),
  not numeric curvature — polynomial on numeric features only (B3) is
  indistinguishable from plain linear.
- The random forest (B5) has a marginally lower mean RMSE (4,836 vs 4,867) but
  wins on only 14 of 25 folds and adds non-determinism with no interpretable
  structure, so the simpler, deterministic B4 was kept.

Runs are byte-for-byte reproducible (fixed seed; library versions recorded in the
artifact): `python -m experiments.compare_models`.

## Output range and safeguard

A linear model is not constrained to non-negative outputs, but an insurance
charge cannot be negative.
[`experiments/investigate_negative_predictions.py`](experiments/investigate_negative_predictions.py)
grids the served model over its input space
([results](experiments/results/negative_predictions.json)):

- **Within the observed data range:** 28 of 6,144 grid points predict a negative
  charge (min −$3,414) — including plausible, non-extreme applicants (e.g. age
  18, bmi 53.13, 0 children, non-smoker, southeast).
- **Across the full schema-legal range:** 1,967 of 13,728 grid points go negative
  (min −$46,795), concentrated in extreme extrapolation.

So negative predictions are not only an extrapolation artifact — they also occur
for plausible, in-range inputs.
[`ModelService.predict`](src/insurance_price/model.py) clamps the output to
`>= 0` at the single serving boundary, and
[`tests/test_model.py`](tests/test_model.py) locks in the discovered failure as a
regression test. This contains the symptom (no caller sees a negative charge) but
does not remove the cause: the linear model form is still unbounded. A
non-negative model form (e.g. a log-target regressor) is future work.

> The grid rates above describe behavior over a synthetic grid of inputs, not the
> rate a real user population would encounter — that depends on the real input
> distribution.

## Architecture

```
            ┌──────────────┐      ┌──────────────────┐      ┌───────────────┐
  client →  │  FastAPI app │  →   │  ModelService    │  →   │ model.joblib  │
  (web/API) │  routes.py   │      │  (sklearn pipe)  │      │ (trained)     │
            └──────┬───────┘      └──────────────────┘      └───────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ repository   │  →   predictions table (SQLite / PostgreSQL)
            └──────────────┘
```

| Module | Responsibility |
| --- | --- |
| `features.py` | Build the scikit-learn pipeline (preprocessing + model) |
| `train.py` | Train, evaluate, save `model.joblib` + `metrics.json` |
| `model.py` | `ModelService`: load the artifact once, predict, guard output |
| `schemas.py` | Pydantic request/response models + validation |
| `config.py` | Typed settings from environment variables |
| `db/` | Engine/session (`base`), table (`models`), queries (`repository`) |
| `api/` | App factory (`app`), routes (`routes`), dependencies (`deps`) |

`experiments/` holds the model-comparison and negative-prediction scripts; they
only read production code and never change the served pipeline.

## Testing

```bash
pytest          # test suite
ruff check .    # lint
```

Tests cover the feature pipeline, the model service (including the non-negative
output regression test), the repository (against an isolated SQLite database),
and the API (valid requests, `422` validation, prediction logging).

## Limitations

- The output guard contains negative predictions, but the underlying linear model
  is unbounded; a non-negative model form is future work and has not been
  evaluated.
- Model comparison uses fixed, untuned candidates and descriptive cross-validation
  statistics (mean / std / per-fold) — no hyperparameter search or significance
  test. B4 is the best of the tested candidates, not a proven optimum.
- The API's legal input range (age 0–120, bmi up to 100) is wider than the
  training-data range (age 18–64, bmi ~16–53); inputs outside the observed range
  are extrapolation.
- The Docker image trains at build time, so the image *is* the model version;
  retraining means rebuilding the image.

## Docker

```bash
docker compose up --build
```

Builds the image (training the model at build time — see Limitations), starts
PostgreSQL, and serves the API on `http://localhost:8000`.

## License

MIT.
