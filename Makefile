.PHONY: run run-debug install install-dev test test-verbose lint format \
        docker-build docker-push docker-run docker-stop docker-clean \
        docker-logs docker-restart docker-shell clean help

# ──────────────────────────────────────────────
#  Local Development
# ──────────────────────────────────────────────

## Run the Streamlit app locally
run:
	streamlit run app.py

## Run with debug logging enabled
run-debug:
	DEBUG_MODE=true streamlit run app.py

## Install production dependencies
install:
	pip install -r requirements.txt

## Install dev dependencies (adds pytest, black, ruff)
install-dev: install
	pip install pytest black ruff

## Run tests
test:
	python -m pytest backend/tests/ -v

## Run tests with coverage
test-verbose:
	python -m pytest backend/tests/ -v --tb=short --log-cli-level=INFO

## Lint with ruff
lint:
	ruff check . --fix

## Format with black
format:
	black .

# ──────────────────────────────────────────────
#  Docker
# ──────────────────────────────────────────────

## Build the Docker image
docker-build:
	docker build --target runtime -t finreport-rag:latest .

## Build and tag with git SHA
docker-build-tag:
	docker build --target runtime \
	  -t finreport-rag:latest \
	  -t finreport-rag:$(shell git rev-parse --short HEAD) .

## Push image to registry (set REGISTRY env var)
docker-push:
	docker push $(REGISTRY)/finreport-rag:latest

## Start the application with Docker Compose
docker-run:
	docker-compose up -d

## Stream container logs
docker-logs:
	docker-compose logs -f finreport-rag

## Stop the application
docker-stop:
	docker-compose down

## Stop and remove all volumes (destructive — wipes ChromaDB + logs)
docker-clean:
	docker-compose down -v
	docker rmi finreport-rag:latest 2>/dev/null || true

## Restart the container
docker-restart:
	docker-compose restart finreport-rag

## Open a shell inside the running container
docker-shell:
	docker exec -it finreport-rag-app /bin/bash

# ──────────────────────────────────────────────
#  Housekeeping
# ──────────────────────────────────────────────

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

## Show this help
help:
	@echo ""
	@echo "  FinReport RAG — Available Make Targets"
	@echo "  ─────────────────────────────────────────────────────────────"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
