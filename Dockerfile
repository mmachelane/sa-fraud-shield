FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency spec first (layer cache — only re-installs if pyproject.toml changes)
COPY pyproject.toml .

# Install project dependencies (no dev extras)
RUN uv pip install --system -e "."

# Copy source code
COPY shared/ shared/
COPY models/ models/
COPY api/ api/
COPY streaming/ streaming/
COPY explainability/ explainability/

# Expose API port
EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
