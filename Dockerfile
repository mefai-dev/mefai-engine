FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[all]"

COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8080 9090

CMD ["python", "-m", "mefai_engine", "run", "--config", "configs/default.yaml"]
