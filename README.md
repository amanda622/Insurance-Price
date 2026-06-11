# Insurance Price Prediction — ML API + Web App

Predict annual health-insurance charges from a person's attributes. A
scikit-learn regression model is served through a **FastAPI** JSON API and a
small web UI, with every prediction logged to a database.

![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Replace `USER/REPO` in the badge URL once the project is on GitHub.
> Add a live demo link here after deploying (see [Deployment](#deployment)).

---

## Features

- **JSON prediction API** (`POST /predict`) with automatic request validation
  and interactive Swagger docs at `/docs`.
- **Web UI** — a simple form for non-technical users.
- **Prediction logging** — every request + result is stored in a database
  (SQLite locally, Postgres in Docker) and exposed at `GET /predictions`.
- **Reproducible training** — one command regenerates the model and metrics.
- **Tested** — unit + API tests with pytest.
- **Containerized** — `docker compose up` brings up the API and a Postgres DB.
- **CI** — GitHub Actions runs lint, tests, and a Docker build on every push.

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
| `train.py` | Train → evaluate → save `model.joblib` + `metrics.json` |
| `model.py` | `ModelService`: load the artifact once, predict |
| `schemas.py` | Pydantic request/response models + validation |
| `config.py` | Typed settings from environment variables |
| `db/` | Engine/session (`base`), table (`models`), queries (`repository`) |
| `api/` | App factory (`app`), routes (`routes`), dependencies (`deps`) |

## Tech stack

Python 3.11+ · FastAPI · Pydantic · scikit-learn · pandas · SQLAlchemy 2 ·
SQLite/Postgres · pytest · Ruff · Docker · GitHub Actions.

## Project structure

```
src/insurance_price/
├── config.py          schemas.py      features.py     train.py     model.py
├── api/               app.py  routes.py  deps.py
├── db/                base.py  models.py  repository.py
├── templates/         index.html  result.html
└── static/            style.css
data/insurance.csv     notebooks/exploration.ipynb     tests/
Dockerfile  docker-compose.yml  Makefile  pyproject.toml  .github/workflows/ci.yml
```

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

Then open:

- `http://127.0.0.1:8000/` — the web form
- `http://127.0.0.1:8000/docs` — interactive API docs

A `Makefile` wraps the common commands: `make install`, `make train`,
`make test`, `make lint`, `make run`, `make docker-up`.

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

Invalid input (e.g. `age: 200` or `smoker: "maybe"`) returns HTTP `422` with a
description of the error.

### Other endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/` | Web prediction form |
| `POST` | `/web/predict` | Form submission → result page |
| `GET` | `/predictions?limit=20` | Recent logged predictions |
| `GET` | `/docs` | Swagger UI |

## Model & metrics

A `LinearRegression` over one-hot-encoded categoricals and degree-2 polynomial
features, trained on the [Medical Cost dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(1,338 rows). Held-out test performance (30% split):

| Metric | Value |
| --- | --- |
| R² | 0.836 |
| MAE | $2,903 |
| RMSE | $4,801 |

Metrics are regenerated into `models/metrics.json` on every training run.

## Testing & quality

```bash
pytest          # run the test suite
ruff check .    # lint
```

Tests cover the feature pipeline, the model service, the repository (against an
isolated SQLite database), and the API (valid requests, 422 validation, and DB
logging).

## Docker

```bash
docker compose up --build
```

This builds the image (training the model at build time so it ships ready to
serve), starts Postgres, and exposes the API on `http://localhost:8000`.

## Deployment

The Docker image is self-contained, so any container host works. Quickest path
to a public demo URL:

**Render (free tier):**
1. Push this repo to GitHub.
2. On [render.com](https://render.com), create a **Web Service** from the repo.
3. Choose **Docker** as the environment — Render uses the `Dockerfile`.
4. Add a Postgres instance and set `DATABASE_URL` to its connection string
   (use the `postgresql+psycopg://...` form).
5. Deploy, then put the live URL at the top of this README.

The same image deploys to **Railway** or **Fly.io** with their respective
"deploy from Dockerfile" flows.

## License

MIT.
