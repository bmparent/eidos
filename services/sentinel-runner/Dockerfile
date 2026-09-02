FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/eidos-package \
    EIDOS_ENGINE_PATH=/app/engine/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py \
    EIDOS_JOB_ROOT=/var/lib/eidos/jobs

WORKDIR /app

COPY services/sentinel-runner /app/runner
RUN pip install --no-cache-dir /app/runner

COPY eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py /app/engine/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py
COPY eidos/repo/src/eidos_brain /app/eidos-package/eidos_brain

RUN useradd --create-home --uid 10001 eidos \
    && mkdir -p /var/lib/eidos/jobs \
    && chown -R eidos:eidos /var/lib/eidos

USER eidos
EXPOSE 8000

CMD ["uvicorn", "sentinel_runner.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
