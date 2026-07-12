# Insurance Price Prediction — ML API + Web App

Predict annual health-insurance charges from a person's attributes. A
scikit-learn regression model is served through a **FastAPI** JSON API and a
small web UI, with every prediction logged to a database.

What sets this project apart is **evidence**: instead of shipping a model on
faith, it compares the model against a trivial baseline and simpler
alternatives under cross-validation, investigates where the model misbehaves,
and makes a small, documented production decision from that evidence.

![CI](https://github.com/amanda622/Insurance-Price/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this model (evidence, not faith)

The served model is `OneHotEncoder -> PolynomialFeatures(degree=2) ->
LinearRegression`. To check that this design is actually justified,
[`experiments/compare_models.py`](experiments/compare_models.py) compares five
candidates under **one shared `RepeatedKFold` split** (5 folds x 5 repeats = 25
identical folds, so every candidate sees exactly the same data). It reports
mean / std / per-fold RMSE, MAE and R2. Full numbers:
[`experiments/results/results.json`](experiments/results/results.json).

| Candidate | What it is | RMSE | MAE | R2 |
| --- | --- | ---: | ---: | ---: |
| B1 | Mean baseline (`DummyRegressor`) | 12,100 | 9,099 | -0.004 |
| B2 | One-hot + `LinearRegression` (no polynomial) | 6,084 | 4,209 | 0.744 |
| B3 | Polynomial on numeric features only | 6,059 | 4,262 | 0.746 |
| **B4** | **Served model: polynomial on all features** | **4,867** | **2,938** | **0.835** |
| B5 | Random forest reference (fixed, untuned) | 4,836 | 2,674 | 0.837 |

*RMSE / MAE in USD, lower is better; mean over 25 folds.*

What the evidence shows:

- **It clearly beats a trivial baseline.** B4 more than halves the
  mean-baseline RMSE (12,100 -> 4,867).
- **The polynomial step earns its complexity.** B4 beats plain linear (B2) by
  about 20% RMSE — and does so on **all 25 folds**, not just on average.
- **The value is in category interactions, not numeric curvature.** Polynomial
  on numeric features only (B3) is indistinguishable from plain linear (B2, it
  wins on only 17/25 folds with an overlapping spread). The lift comes from
  interactions such as `smoker x bmi` and `smoker x age`.
- **A random forest was not worth switching to.** B5's mean RMSE is marginally
  lower (4,836 vs 4,867), but it wins on only **14 of 25 folds** and brings
  non-determinism and no interpretable structure. That ~0.6% average difference
  did not justify replacing a simpler, deterministic model, so **the current
  model (B4) was kept**.

Reruns are byte-for-byte reproducible (fixed seed; library versions recorded in
the artifact).

## Known failure: negative predictions

A linear model is not constrained to non-negative outputs, but an insurance
charge cannot be negative.
[`experiments/investigate_negative_predictions.py`](experiments/investigate_negative_predictions.py)
grids the served model over two input regions (full numbers in
[`experiments/results/negative_predictions.json`](experiments/results/negative_predictions.json)):

- **Within the observed data range:** 28 of 6,144 grid points predict a
  negative charge (min -$3,414). Example: age 18, bmi 53.13, 0 children,
  non-smoker, southeast — a plausible applicant, not an extreme input.
- **Across the full schema-legal range:** 1,967 of 13,728 grid points go
  negative (min -$46,795), concentrated in extreme extrapolation.

The key finding: negative predictions are **not** only an extreme-extrapolation
artifact — they also occur for plausible, in-range inputs.

**Safeguard.** [`ModelService.predict`](src/insurance_price/model.py) clamps the
output to `>= 0` at the single serving boundary, and
[`tests/test_model.py`](tests/test_model.py) locks in the exact discovered
failure as a regression test. This **contains the symptom** (no caller ever
sees a negative charge) but **does not remove the cause** (the linear model form
is still unbounded); a non-negative model form, such as a log-target regressor,
is left as documented future work.

> The grid rates above describe behavior over a synthetic grid of inputs, not
> the rate a real user population would encounter — that depends on the real
> input distribution.

## Architecture

```
            ┌──────────────┐      ┌──────────────────┐      ┌───────────────┐
  client →  │  FastAPI app │  →   │  ModelService    │  →   │ model.joblib  │
  (web/API) │  routes.py   │      │  (sklearn pipe)  │      │ (trained)     │
            └──────┬───────┘      └──────────────────┘      └───────────────┘
                   │
                   ▼
            ┌──────────────┐
            │ repository   │  →   predictions table (SQLite / Postgres)
            └──────────────┘
```

Each module has one job:

| Module | Responsibility |
| --- | --- |
| `features.py` | Build the scikit-learn pipeline (preprocessing + model) |
| `train.py` | Train, evaluate, save `model.joblib` + `metrics.json` |
| `model.py` | `ModelService`: load the artifact once, predict, guard output |
| `schemas.py` | Pydantic request/response models + validation |
| `config.py` | Typed settings from environment variables |
| `db/` | Engine/session (`base`), table (`models`), queries (`repository`) |
| `api/` | App factory (`app`), routes (`routes`), dependencies (`deps`) |

The `experiments/` directory holds the model comparison and negative-prediction
investigation described above; it only reads production code and never changes
the served pipeline.

## Getting started (local)

Requires Python 3.11+.

```bash
# 1. Create a virtual environment and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Train the model (writes models/model.joblib + models/metrics.json)
python -m insurance_price.train

# 3. Run the API (http://127.0.0.1:8000)
uvicorn insurance_price.api.app:app --reload
```

Then open `http://127.0.0.1:8000/` (web form) or `/docs` (interactive API docs).
A `Makefile` wraps the common commands: `make install`, `make train`,
`make test`, `make lint`, `make run`, `make docker-up`.

To reproduce the evidence:

```bash
python -m experiments.compare_models                    # -> results.json
python -m experiments.investigate_negative_predictions  # -> negative_predictions.json
```

## API reference

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
| `GET` | `/` | Web prediction form |
| `POST` | `/web/predict` | Form submission -> result page |
| `GET` | `/predictions?limit=20` | Recent logged predictions |
| `GET` | `/docs` | Swagger UI |

## Testing & quality

```bash
pytest          # run the test suite
ruff check .    # lint
```

Tests cover the feature pipeline, the model service (including the
non-negative-output regression test), the repository (against an isolated
SQLite database), and the API (valid requests, 422 validation, DB logging).

## Known limitations

- The output guard contains negative predictions, but the underlying linear
  model is unbounded; a non-negative model form (e.g. a log-target regressor)
  is future work and has not yet been evaluated.
- The model comparison uses fixed, untuned candidates and reports descriptive
  cross-validation statistics (mean / std / per-fold). No hyperparameter search
  or formal significance test was run — B4 is the best of the tested candidates,
  not a proven optimum.
- The API's legal input range (e.g. age 0-120, bmi up to 100) is much wider
  than the training data range (age 18-64, bmi about 16-53); predictions
  outside the observed range are extrapolation.
- The Docker image trains at build time, so the image *is* the model version;
  retraining means rebuilding the image.

## Docker & deployment

```bash
docker compose up --build
```

This builds the image (training the model at build time, see the trade-off
above), starts Postgres, and exposes the API on `http://localhost:8000`. The
image is self-contained, so any Docker host (Render, Railway, Fly.io) can serve
it directly from the `Dockerfile`.

## License

MIT.
