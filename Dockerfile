# Serves the FastAPI app (src/ticket_triage/api.py). Not the same as
# docker-compose.yml, which is local-dev Postgres only -- this image is
# the actual deployable artifact.
#
# Before building: run `uv run python scripts/save_priority_model.py`
# locally to produce models/priority/ (gitignored -- a build artifact,
# not source, same as checkpoints/ and mlruns/). The DistilBERT Product
# model is NOT bundled here; it's pulled from Hugging Face Hub
# (PRODUCT_MODEL_REPO) at container startup instead, since it needs
# network access either way and baking 260MB into every image layer
# would be wasteful.
FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependency manifests first, for layer caching -- rebuilding the image
# after an app-code-only change shouldn't re-resolve/reinstall every
# dependency.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra serve

COPY README.md ./
COPY src/ ./src/
COPY models/priority/ ./models/priority/
RUN uv sync --frozen --no-dev --extra serve

# Hugging Face Spaces' Docker SDK expects the app on port 7860 and runs
# containers as a non-root user.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
EXPOSE 7860

# Not `uv run uvicorn ...` -- that re-checks/re-syncs the environment
# against the full lockfile on every container start, which pulled in
# the dev-only group (jupyter, matplotlib, ...) at runtime and defeated
# the point of a lean build. The venv was already fully assembled above;
# just run its uvicorn directly.
CMD [".venv/bin/uvicorn", "ticket_triage.api:app", "--host", "0.0.0.0", "--port", "7860"]
