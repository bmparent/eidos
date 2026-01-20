FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition
COPY pyproject.toml README.md ./

# Install dependencies
RUN pip install --no-cache-dir .

# Copy source
COPY src/Src ./src
COPY configs ./configs
COPY demo_data ./demo_data

# Install package in editable mode or build
RUN pip install --no-cache-dir .

# Create artifact dirs
RUN mkdir -p /data/eidos_artifacts

# Environment defaults
ENV EIDOS_ARTIFACT_ROOT=/data/eidos_artifacts
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8001 8501

# Entrypoint default (can be overridden)
CMD ["eidos-api"]
