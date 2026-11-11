# Multi-stage minimal container for standalone `wahs-engine`
FROM python:3.12-slim AS builder
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
WORKDIR /app

ARG WAHS_BUILD_REVISION=dev
LABEL org.opencontainers.image.revision=${WAHS_BUILD_REVISION}

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels wahs-engine \
    && rm -rf /wheels

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    WAHS_FHIR_AUTH_BASE_URL=http://fhir-auth:8080/fhir \
    LLM_BASE_URL=http://host.docker.internal:8079/v1

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]

USER 65532:65532
CMD ["python", "-m", "uvicorn", "wahs_engine.app:app", "--host", "0.0.0.0", "--port", "8080"]
