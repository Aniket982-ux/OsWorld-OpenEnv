FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy the entire workspace necessary for syncing
COPY . .

# Use uv to sync project dependencies (ignores dev dependencies by default if configured/omits them depending on uv lock options)
RUN uv sync

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn running the internal validation server
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
