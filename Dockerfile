FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies from lockfile (no dev deps, no cache to keep image lean)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# Copy application code
COPY main.py retrieve.py index.py ./

# Runtime directories (doc_store should be mounted as a volume in production)
RUN mkdir -p doc_store static/evidence

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
