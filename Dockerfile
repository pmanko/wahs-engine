# Multi-stage minimal container for standalone `wahs-engine`
FROM python:3.12-slim AS builder
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels . \
    && pip install --no-cache-dir --no-index --find-links=/wheels tiktoken \
    && TIKTOKEN_CACHE_DIR=/tiktoken_cache python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')"

FROM python:3.12-slim
WORKDIR /app

ARG WAHS_BUILD_REVISION=dev
LABEL org.opencontainers.image.revision=${WAHS_BUILD_REVISION}

COPY --from=builder /wheels /wheels
COPY --from=builder /tiktoken_cache /opt/tiktoken_cache
RUN pip install --no-cache-dir --no-index --find-links=/wheels wahs-engine \
    && rm -rf /wheels

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache \
    WAHS_FHIR_AUTH_BASE_URL=http://fhir-auth:8080/fhir \
    LLM_BASE_URL=http://host.docker.internal:8079/v1

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]

USER 65532:65532
CMD ["python", "-m", "uvicorn", "wahs_engine.app:app", "--host", "0.0.0.0", "--port", "8080"]
