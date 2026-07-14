.PHONY: install lint format typecheck test check run-api run-app clean

install:        ## Sync env from lockfile
	uv sync

lint:           ## Ruff lint
	uv run ruff check .

format:         ## Ruff format
	uv run ruff format .

typecheck:      ## Pyright
	uv run pyright

test:           ## Pytest + coverage
	uv run pytest

check: lint typecheck test  ## Full local quality gate

run-api:        ## Serve FastAPI locally (http://localhost:8000 redirects to /docs)
	uv run uvicorn defect_detector.api:app --reload

run-app:        ## Serve Streamlit locally
	uv run streamlit run src/defect_detector/app.py

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .coverage htmlcov
