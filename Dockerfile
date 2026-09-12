# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MEDICAL_TRIAGE_MODEL_PATH=/app/artifacts/model.joblib \
    METRICS_PATH=/app/artifacts/metrics.json \
    KAGGLEHUB_CACHE=/home/appuser/.cache/kagglehub

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .

COPY scripts/container-entrypoint.sh /usr/local/bin/container-entrypoint
RUN chmod 0755 /usr/local/bin/container-entrypoint \
    && useradd --create-home --uid 10001 appuser \
    && install -d -o appuser -g appuser /app/artifacts /home/appuser/.cache/kagglehub

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5m --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

ENTRYPOINT ["container-entrypoint"]
CMD ["uvicorn", "medical_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
