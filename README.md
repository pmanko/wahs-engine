# `wahs-engine`

Standalone, zero-privilege OpenAI-compatible clinical FHIR R4 condensation, symbolic clinical reconciliation, and `[N]` provenance middleware for **Washington Health Summary (`WAHS`)**.

Public documentation and evaluation readouts:
- Project overview: [openclinai.org/wahs/](https://openclinai.org/wahs/)
- AI strategy & model selection: [openclinai.org/wahs/llm-strategy/](https://openclinai.org/wahs/llm-strategy/)
- Developer wire contract & internals: [openclinai.org/wahs/engine-reference/](https://openclinai.org/wahs/engine-reference/)
- Architectural & functional audit: [openclinai.org/wahs/engine-audit/](https://openclinai.org/wahs/engine-audit/)

---

## Scope & Explicit Non-Goals

`wahs-engine` sits between patient-facing WAHS applications (`uwcirg/shl-ips` SvelteKit BFF and mobile companion clients), the per-user `fhir-auth` (`uwcirg/jwt-proxy`) security proxy over HAPI FHIR R4, and any stateless OpenAI-compatible model host (`LiteLLM`, `Ollama`, `vLLM`, `llama-server`, `SGLang`).

### What `wahs-engine` owns
1. **Zero-privilege multi-dataset FHIR R4 assembly (`WAHSBundleAssembler`):** Forwards the caller's `Authorization: Bearer <keycloak_jwt>` token to `fhir-auth`, resolves `MasterPatient.link[]`, checks dataset `Patient.meta.lastUpdated` freshness stamps on every turn, paginates `Patient/{datasetId}/$everything` (preserving provider imports alongside patient-authored *My Story* `LOINC 51855-5` notes and `Goal` resources), applies `wahs.chart_scope` filters and `<5 KB` `wahs.overlay_resources`, and enforces single-patient subject scoping.
2. **2-stage monotonic structural condensation (`wahs_core.condenser`):** Compresses multi-clinic FHIR R4 exports (`90%-99%` token reduction) into exact model budgets (`512` to `131,072` tokens; `16,384` default ceiling) using bundled offline Gemma 4 and `tiktoken` tokenizers, guaranteeing `covered_uuids(B1) <= covered_uuids(B2)` across budgets and emitting a machine-readable `[2] Completeness Manifest`.
3. **Deterministic clinical enrichment & data-layer privacy (`wahs_core` Layer 3):** Pre-computes a 7-edge `ClinicalEntityGraph` (allergy cross-reactivity, UCUM-normalized renal/electrolyte dosing hazards, cross-clinic stage/status conflicts, longitudinal note-vs-order holds), a tie-safe `TemporalDigest` (`n=1` and intraday-tie trend guards), 4-Layer Clinical KB guideline matching, conditional Tier-0 / BM25 Note-RAG overflow recovery, and fail-closed `external_share` sensitivity redaction.
4. **KV-cache-friendly prompt layout & `[N]` provenance (`SouthboundOpenAIClient`):** Keeps the condensed patient chart invariant at `messages[0]` for GPU prefix KV-cache reuse, attaches turn-specific enrichment at `messages[-1]`, strips internal `<think>` blocks (including truncated thoughts), and appends `x_wahs` (`citation_map`, `condensation`, `completeness_manifest`, `temporal_digest`, `privacy_notices`, and HL7 `AI-Provenance`).

### Explicit non-goals
- **Zero HAPI database admin credentials:** Never connects directly to `fhir-internal` or raw HAPI admin ports in runtime traffic; per-patient authorization is enforced by `fhir-auth` (`meta.security.code == jwt.sub`), optionally paired with local RS256 JWKS signature verification (`WAHS_JWT_JWKS_URL` / `WAHS_JWT_JWKS_JSON`).
- **Zero persistent PHI or chat database:** Conversation history lives in the caller's browser session (`messages[]`). Engine caches are ephemeral, byte- and entry-bounded in-memory LRUs keyed by `(verified_sub, datasets_fingerprint, scope_hash)`.
- **Zero FHIR write-backs or autonomous clinical decisions:** Strictly read-only over patient records; every response carries `human_review_required: true` in `x_wahs.ai_provenance`.
- **Not a substitute for certified institutional segmentation:** `external_share` code and lexical rules provide defense-in-depth alongside upstream HL7 DS4P `meta.security` labels, not standalone legal certification under 42 CFR Part 2 or WA MHMDA.

---

## Research, Interoperability & Clinical Foundations

| Pipeline Stage | Core Mechanism | Standards & Research Backing |
| :--- | :--- | :--- |
| **1. Fetch & Freshness** | Star-graph `Patient/$everything` assembly, DS4P security labels, optimistic `meta.lastUpdated` revalidation | HL7 US Core R4, ONC Cures Act Final Rule, SMART Health Links, HL7 FHIR DS4P Security Labels, HL7 FHIR RESTful Versioning |
| **2. Condense (90%-99%)** | 2-stage monotonic RLE + anti-starvation budget packing, IPS safety core pinning, BM25 passage lookup | Liu et al., *Lost in the Middle* (*TACL*, 2024); HL7 International Patient Summary (IPS / ISO 27269); Robertson & Zaragoza, *Okapi BM25* (2009) |
| **3. Clinical & Privacy Rules** | Symbolic 7-edge `ClinicalEntityGraph`, tie-safe `TemporalDigest`, fail-closed `external_share` redaction | HL7 CIRI, WHO SMART Guidelines, KDIGO 2024 CKD, ADA Diabetes Standards, NLM RxNorm, LOINC/UCUM, CDC ACIP, 42 CFR Part 2, WA MHMDA (RCW 19.373) |
| **4. KV Split & Provenance** | Invariant `messages[0]` / dynamic `messages[-1]`, inline `[1]..[N]` UUID citations, `AI-Provenance` | Kwon et al., *PagedAttention* (*SOSP*, 2023); Zheng et al., *RadixAttention* (*NeurIPS*, 2024); Gao et al., *ALCE* (*EMNLP*, 2023); PhysioNet ArchEHR-QA (2025); HL7 AI Transparency (`AI-Provenance`) |

---

## Package Structure

- **`src/wahs_core/` (zero third-party Python dependencies, `python -S` verified):**
  - `kb_data/*.json`, `kb_data/wahs_corpus.jsonl`, `kb_lookup.py`, `kb.py`, `normalizers.py`: Layer-0 clinical knowledge packs, RxNorm ingredient/salt normalizers, exclusivity rules, and `meta.security`-first sensitivity classifiers.
  - `extractors.py`: Layer-1 faithful FHIR R4 extractor across 16 resource types with multi-coding terminology preference, structured dosage/conclusion extraction, and invalidated-status filtering (`entered-in-error`, `refuted`, `not-done`, `cancelled`).
  - `condenser.py`, `chart_serializer.py`: Layer-2 2-stage monotonic structural condenser and `[1]..[N]` serializer.
  - `clinical_graph.py`, `temporal_digest.py`, `wahs_tools.py`: Layer-3 7-edge entity graph, longitudinal temporal digest, Tier-0 overflow recovery, and read-only lookup tools (`search_chart_entries`, `get_lab_history`, `read_clinical_note`, `retrieve_clinical_note_passages`).
- **`src/wahs_engine/` (FastAPI service; runtime dependencies: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `tokenizers`, `tiktoken`):**
  - `app.py`, `schemas.py`, `config.py`: FastAPI lifecycle, OpenAI `/v1` + `wahs` request validation, profile/budget parser, and settings.
  - `jwt_claims.py`: Bearer JWT decoding, subject binding, HS256 and pure-stdlib RS256 JWKS signature verification.
  - `assembler.py`: Zero-privilege `fhir-auth` star-graph assembler, single-flight dataset coalescing, `chart_scope` filtering, and byte/entry-bounded LRU caches.
  - `tokenizers_registry.py`, `tokenizer_assets/`: Air-gapped Gemma 4 and `o200k_base` token counters (`<0.01%` drift vs. GPU prompt counts).
  - `upstream.py`: Southbound OpenAI `/v1` client, history trimming, SSE pre-flight & lookahead streaming, and `x_wahs` envelope builder.
  - `seeder.py`: Idempotent 12-persona HAPI star-graph and Keycloak 26 demo seeder (`wahs-seed-hapi`).

See [`docs/architecture.md`](docs/architecture.md) for full layer contracts, budget math, wire schemas, and security details.

---

## Northbound API & Model Profiles

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health`, `GET /healthz` | Liveness and active configuration summary (`default_model`, `fhir_auth_base_url`, `llm_base_url`). |
| `GET /v1/models` | Lists configured WAHS pathway profiles merged with upstream southbound models. |
| `POST /v1/wahs/context/preview` | Returns assembled `chart_text`, `prefix_cache_key_hash`, `temporal_digest`, `security_audit`, and `x_wahs` without calling the LLM. |
| `POST /v1/chat/completions` | Standard OpenAI chat completions (sync JSON or `stream: true` SSE) with optional `wahs` scoping and `x_wahs` provenance. |

**Profile naming convention (`<stem>-<pathway>[@<budget>]`):**
- `-kb` (`condensation_kb`, **recommended default**, e.g. `wahs-26b-kb`, `wahs-12b-kb`, `wahs-e4b-kb`): `16k` condensed chart + `ClinicalEntityGraph` + `TemporalDigest` + 4-Layer KB + conditional Tier-0 overflow.
- `-rag` (`note_rag`, e.g. `wahs-26b-rag`): Adds single-patient BM25 passage retrieval over full clinical notes and reports.
- `-chart` / `-raw` (`chart_only`, e.g. `wahs-26b-chart`): Pure structural condensed chart baseline (disables KB, temporal digest, and overflow tools unless explicitly enabled).
- `-tools` (`dynamic_fhir_tools`, e.g. `wahs-26b-tools`): Compact `2k` IPS safety header + multi-round model-driven chart tool calls.
- Optional `@<budget>` suffix (e.g. `wahs-26b-kb@16k`, `wahs-26b-kb@32k`): Clamped safely between `512` tokens and `WAHS_MAX_CONTEXT_WINDOW // 2`.

---

## Quickstart & Verification

```bash
# 1. Verify pure-stdlib wahs_core import (zero third-party packages)
PYTHONPATH=src python3 -S -c "import sys, wahs_core; assert 'site' not in sys.modules; print(wahs_core.__version__)"

# 2. Install editable package with dev dependencies and run test suite
pip install -e ".[dev]"
pytest -q

# 3. Run standalone wahs-engine service (recommended H200 pilot model: wahs-26b-kb)
WAHS_FHIR_AUTH_BASE_URL=http://localhost:8081/fhir \
LLM_BASE_URL=http://localhost:8079/v1 \
WAHS_DEFAULT_MODEL=wahs-26b-kb \
WAHS_MODEL_MAP_JSON='{"wahs-26b-kb":"ollama/gemma4:26b","wahs-12b-kb":"ollama/gemma4:12b","wahs-e4b-kb":"ollama/gemma4:e4b"}' \
wahs-engine
```
