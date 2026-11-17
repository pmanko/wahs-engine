# Working in `wahs-engine`

`wahs-engine` is the standalone, zero-privilege OpenAI-compatible clinical FHIR R4 condensation, provenance, and `fhir-auth` middleware for Washington Health Summary (WAHS).

## Architecture & Invariants

- **Pure-Stdlib Core (`src/wahs_core/`):** `condenser.py`, `chart_serializer.py`, `clinical_graph.py`, `temporal_digest.py`, `wahs_tools.py`, and KB data (`kb_data/`) **must remain zero third-party dependencies** (verified in CI via `python -S`).
- **Service Boundary (`src/wahs_engine/`):** Lightweight FastAPI service (`app.py`, `assembler.py`, `upstream.py`, `seeder.py`; runtime deps: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `tokenizers`, `tiktoken`).
- **Security & Cache Isolation:**
  - Preserve bearer-token pass-through (`Authorization: Bearer <keycloak_jwt>`) to `fhir-auth` (`WAHSBundleAssembler`).
  - Keep strict `(verified_sub, datasets_fingerprint, scope_hash)` cache isolation—never leak patient state across subjects or scope filters.
  - Reject client-supplied `fhir_bundle` payloads whenever `WAHS_ALLOW_LOCAL_BUNDLE_MODE=false`.
- **Air-Gapped Tokenizers:** Keep vendored tokenizer assets (`src/wahs_engine/tokenizer_assets/*.gz`) offline-capable (`--network none` container smoke test runs in CI).

## Local Development & Verification

Requires Python `>=3.10`:

```bash
# Verify pure-stdlib wahs_core import (zero third-party deps)
PYTHONPATH=src python -S -c "import site; import wahs_core; assert wahs_core.__version__"

# Install editable with test deps and run unit + contract tests
pip install -e ".[dev]"
pytest -q
```

Run the standalone service or HAPI seeder CLI entrypoints:

```bash
WAHS_FHIR_AUTH_BASE_URL=http://localhost:8081/fhir \
LLM_BASE_URL=http://localhost:8079/v1 \
WAHS_DEFAULT_MODEL=gemma4:12b \
wahs-engine
```

## Git & PR Hygiene

- Keep changes scoped to `wahs_core`, `wahs_engine`, `tests/`, and `docs/`; never commit real PHI or local `.env` secrets.
- Push feature branches and open PRs on `pmanko/wahs-engine` early so CI (`unit-and-contract`, `docker-smoke`, and `live-hapi-jwt-proxy-integration`) runs right away.
