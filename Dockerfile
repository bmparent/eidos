# Dockerfile
# ============================================================
# Eidos Brain - Production Container Image
# ============================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUNTIME_MODE=cloud \
    GCP_PROJECT_ID=hive-core-vertex-bmparent

# Create non-root user
RUN groupadd -r eidos && useradd -r -g eidos eidos

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure no secrets
RUN rm -rf secrets/ *.pem *.key || true

# Change ownership
RUN chown -R eidos:eidos /app

# Switch to non-root user
USER eidos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "eidos.EIDOS_BRAIN_UNIFIED_v0_4.7.02"]
