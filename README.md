# `wahs-engine`

Standalone, zero-privilege OpenAI-compatible clinical FHIR R4 condensation, knowledge enrichment, and `[N]` provenance middleware for **Washington Health Summary (`WAHS`)**.

## Package Architecture

- **`wahs_core`** (`src/wahs_core/` — **zero third-party Python dependencies**, pure stdlib):
  - **5-Tier Progressive Non-Destructive FHIR R4 Condenser** (`condenser.py`, `extractors.py`, `chart_serializer.py`) fitting arbitrary multi-decade US Core / IPS charts into any token budget (`512`–`32,768+` tokens) with an explicit `[Completeness Manifest]` and 100% `[N] -> resourceUuid` provenance.
  - **4-Layer Clinical Knowledge Base** (`kb.py`, `kb_lookup.py`, `normalizers.py`, `kb_data/`): CDC ACIP vaccine rules, RxNorm ingredient/brand-generic & therapeutic classes, KDIGO/ADA stage exclusivity, 42 CFR Part 2 / WA MHMDA SDOH privacy classification, and MedlinePlus/LOINC plain-language snippets.
  - **`ClinicalEntityGraph` & Tier-0 Overflow Tools** (`clinical_graph.py`, `wahs_tools.py`, `wahs_skills.py`, `wahs_analyzer.py`): Read-only chart drill-down (`search_chart_entries`, `get_lab_history`, `read_clinical_note`) when budget condensation omits historical records.
- **`wahs_engine`** (`src/wahs_engine/` — lightweight FastAPI service; runtime deps: `fastapi`, `uvicorn`, `httpx`, `pydantic` only):
  - **`WAHSBundleAssembler`** (`assembler.py`): Zero-privilege `Authorization: Bearer <keycloak_jwt>` pass-through to CIRG `fhir-auth` (`uwcirg/jwt-proxy`), traversing `masterPatient.link[]` -> live `Patient/{datasetId}` metadata -> paginated `Patient/{datasetId}/$everything` (including saved `patient-story` `Observation` `51855-5` & `Goal` resources), declarative `wahs.chart_scope`, `<5 KB` `wahs.overlay_resources`, and strict `(verified_sub, datasets_fingerprint, scope_hash)` caching.
  - **Dual-Boundary OpenAI `/v1` Wire Contract** (`app.py`, `upstream.py`):
    - **Northbound (`POST /v1/chat/completions`, `GET /v1/models`):** Drop-in compatible with Vercel AI SDK (`@ai-sdk/openai-compatible`) and standard OpenAI SDKs, attaching `x_wahs` (`citation_map`, `scope_applied`, `condensation`, `completeness_manifest`, `privacy_notices`, and HL7 `AI-Provenance`).
    - **Southbound (`LLM_BASE_URL`):** Connects plug-and-play to any OpenAI `/v1` inference server (`LiteLLM`, `vLLM`, `Ollama`, `llama-server`, `SGLang`).

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q

# Start standalone service
WAHS_FHIR_AUTH_BASE_URL=http://localhost:8081/fhir \
LLM_BASE_URL=http://localhost:8079/v1 \
WAHS_DEFAULT_MODEL=gemma4:12b \
wahs-engine
```
