.PHONY: install train test lint run docker-up docker-down

install:        ## Install the package with dev dependencies
	pip install -e ".[dev]"

train:          ## Train the model and write artifacts to models/
	python -m insurance_price.train

test:           ## Run the test suite
	pytest

lint:           ## Lint with ruff
	ruff check .

run:            ## Run the API locally with autoreload
	uvicorn insurance_price.api.app:app --reload

docker-up:      ## Build and start the app + Postgres stack
	docker compose up --build

docker-down:    ## Stop the stack and remove volumes
	docker compose down -v
