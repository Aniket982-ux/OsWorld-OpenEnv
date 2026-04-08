# Master Dockerfile - Optimized Multi-Stage Build
# This Dockerfile is fully compatible with the OpenEnv ecosystem.
# It uses a standard Python base for local build compatibility while 
# maintaining the same professional structure as the official templates.

# Stage 1: Builder
# Resolves dependencies and prepare the virtual environment
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install uv for lightning-fast, reproducible dependency management
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY pyproject.toml ./
COPY uv.lock ./

# Synchronize dependencies into a standalone virtual environment (.venv)
# This includes the openenv-core[core] package itself.
RUN uv sync --frozen --no-install-project --no-editable

# Copy the application source code
COPY . .

# Final sync to install the project package in the venv
RUN uv sync --frozen --no-editable

# Step 1.5: Automated OpenEnv Validation
# This ensures the environment is valid by OpenEnv standards before final image creation.
# If this fails, the build will stop, preventing invalid deployments.
RUN .venv/bin/openenv validate

# Stage 2: Runtime
# Minimal final image containing only the application and its dependencies
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
# Copy the cleaned source code
COPY --from=builder /app /app

# Point PATH and PYTHONPATH to our isolated virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# OpenEnv standard port
EXPOSE 8000

# Health Check to verify the FastAPI bridge is active
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the environment server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
