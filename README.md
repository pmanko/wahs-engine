# `wahs-engine`

Standalone, zero-privilege OpenAI-compatible clinical FHIR R4 condensation, knowledge enrichment, and `[N]` provenance middleware for **Washington Health Summary (`WAHS`)**.

## Package Architecture

- **`wahs_core`** (`src/wahs_core/` — **zero third-party Python dependencies**, pure stdlib):
  - **Layer 0 Reference Packs & Normalizers** (`kb_data/*.json`, `kb.py`, `kb_lookup.py`, `normalizers.py`): RxNorm brand/ingredient/cross-reactivity tables, CDC ACIP vaccine rules, clinical stage exclusivity, and `meta.security`-first SDOH/privacy classification + lexical fallbacks.
  - **Layer 1 Faithful FHIR R4 Extractor** (`extractors.py`): Native `Observation.interpretation`/`referenceRange` extraction, faithful clinical note sections, and real undated semantics (`date=None`, `date_kind="unknown"`).
  - **Layer 2 Structural Condenser & Serializer** (`condenser.py`, `chart_serializer.py`): Pure 2-stage deterministic FHIR R4 condensation fitting arbitrary token budgets (`512`–`32,768+` tokens) with an explicit `[Completeness Manifest]`, unpacked inpatient/ED encounters, and 100% `[N] -> resourceUuid` provenance.
  - **Layer 3 `ClinicalEntityGraph`, `TemporalDigest` & Overflow Tools** (`clinical_graph.py`, `temporal_digest.py`, `wahs_tools.py`): Deterministic temporal anchor, clinical vs. administrative date separation, encounter horizon, and `n>=2` longitudinal trend digest (`enable_temporal_digest`; excluded from `chart_only`), typed entity-graph edge inference (`CROSS_REACTS_WITH`, `CONTRAINDICATED_OR_DOSE_ADJUST_IN`, `CONFLICTS_WITH`, `MONITORED_BY`, `TREATS`), Tier-0 overflow recovery, 4-layer KB lookup, and read-only chart/note/lab drill-down (`search_chart_entries`, `get_lab_history`, `read_clinical_note`, `retrieve_clinical_note_passages`).

See [`docs/architecture.md`](docs/architecture.md) for detailed layer contracts and privacy segmentation scope/limitations.
- **`wahs_engine`** (`src/wahs_engine/` — lightweight FastAPI service; runtime deps: `fastapi`, `uvicorn`, `httpx`, `pydantic` only):
  - **`WAHSBundleAssembler`** (`assembler.py`): Zero-privilege `Authorization: Bearer <keycloak_jwt>` pass-through to CIRG `fhir-auth` (`uwcirg/jwt-proxy`), traversing `masterPatient.link[]` -> live `Patient/{datasetId}` metadata -> paginated `Patient/{datasetId}/$everything` (including saved `patient-story` `Observation` `51855-5` & `Goal` resources), declarative `wahs.chart_scope`, `<5 KB` `wahs.overlay_resources`, and strict `(verified_sub, datasets_fingerprint, scope_hash)` caching.
  - **Dual-Boundary OpenAI `/v1` Wire Contract** (`app.py`, `upstream.py`):
    - **Northbound (`POST /v1/chat/completions`, `GET /v1/models`):** Drop-in compatible with Vercel AI SDK (`@ai-sdk/openai-compatible`) and standard OpenAI SDKs, attaching `x_wahs` (`citation_map`, `scope_applied`, `condensation`, `completeness_manifest`, `temporal_digest`, `privacy_notices`, and HL7 `AI-Provenance`).
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
