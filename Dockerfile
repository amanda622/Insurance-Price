# Self-contained image: installs the package, trains the model at build time,
# and serves the API with Uvicorn.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.joblib \
    DATA_PATH=/app/data/insurance.csv \
    DATABASE_URL=sqlite:////app/predictions.db

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train the model so the image ships ready to serve.
COPY data ./data
RUN python -m insurance_price.train

EXPOSE 8000

CMD ["uvicorn", "insurance_price.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
