FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY ui/pyproject.toml /app/ui/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /app/ui

COPY ui /app/ui

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UI_HOST=0.0.0.0 \
    UI_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "ui.main"]
