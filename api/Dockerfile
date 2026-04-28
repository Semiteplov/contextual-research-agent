FROM python:3.12-slim

WORKDIR /app

# System dependencies (build tools for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: contextual_research_agent must be available in the image.
# Recommended: build from project root with this Dockerfile path.
COPY api/pyproject.toml /app/api/
COPY contextual_research_agent /app/contextual_research_agent

# Install API dependencies + the contextual_research_agent package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /app/api && \
    pip install --no-cache-dir -e /app

COPY api /app/api

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_LOG_LEVEL=info

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "api.main"]
