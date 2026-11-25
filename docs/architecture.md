# WAHS Engine Architecture & Design Reference

`wahs-engine` is organized into two strictly separated Python packages so pure clinical condensation and graph logic never depend on web or R&D frameworks:

1. **`wahs_core` (`src/wahs_core/`)**: 100% pure Python stdlib library (`python3 -S` verified in CI, zero third-party dependencies) implementing clinical normalization, FHIR R4 extraction, 2-stage monotonic token-budget condensation, the 7-edge `ClinicalEntityGraph`, tie-safe `TemporalDigest`, and privacy-enforcing chart lookup tools.
2. **`wahs_engine` (`src/wahs_engine/`)**: Lightweight FastAPI service (`fastapi`, `uvicorn`, `httpx`, `pydantic`, `tokenizers`, `tiktoken`) providing zero-privilege `fhir-auth` multi-dataset assembly, optional local RS256 JWKS verification, exact air-gapped BPE/SentencePiece token sizing, and dual-boundary OpenAI `/v1/chat/completions` streaming with `x_wahs` provenance.

---

## 1. System Boundaries & Explicit Non-Goals

```text
[WAHS Web / Mobile BFF]
   |  POST /v1/chat/completions
   |  Authorization: Bearer <keycloak_rs256_jwt>
   v
[wahs-engine]  <--->  [fhir-auth (uwcirg/jwt-proxy) + HAPI FHIR R4]
   |                     (enforces meta.security.code == jwt.sub)
   |  POST /v1/chat/completions
   |  messages[0] = invariant [1]..[N] chart (KV-cached)
   |  messages[-1] = user question + Layer-3 clinical/temporal suffix
   v
[Stateless OpenAI-Compatible GPU Host (LiteLLM / Ollama / vLLM / llama.cpp)]
```

- **Zero database credentials:** `wahs-engine` holds no HAPI admin keys and blocks `fhir-internal` URLs at startup. Every live chart turn checks `MasterPatient` and linked dataset `Patient` stubs through `fhir-auth` using the caller's own `Bearer` token.
- **Stateless conversations, bounded ephemeral caches:** Chat history stays in the client (`messages[]`). `wahs-engine` stores no transcripts on disk and enforces both entry-count and byte-size ceilings on in-memory dataset and prefix LRU caches.
- **Read-only clinical support:** `wahs-engine` never mutates FHIR resources in HAPI (`wahs-seed-hapi` is an explicit dev/demo CLI entrypoint) and marks every response `human_review_required: true` under HL7 `AI-Provenance`.

---

## 2. Strict 4-Layer `wahs_core` Pipeline

### Layer 0: Reference Knowledge Packs & Normalizers (`kb_data/`, `kb_lookup.py`, `normalizers.py`, `kb.py`)
- **Offline clinical packs (`src/wahs_core/kb_data/`):**
  - `rxnorm_lookup.json`: RxNorm active-ingredient normalization, brand-to-generic mappings, therapeutic class overlap rules, interaction hazards, and allergy cross-reactivity rules.
  - `clinical_exclusivity_and_privacy.json`: Mutually exclusive condition staging groups (`ckd_stage`, `glycemic_progression`, `heart_failure_stage`), HL7 `meta.security` restricted codes (`R`, `V`, `ETH`, `PSY`, `SDV`, `HIV`, `SUD`, `42CFRPART2`, `SEX`, `DOMVIO`, `BH`), sensitive screening LOINC codes, MOUD/HIV/WA-MHMDA lexical fallback patterns, and clinic-room device filters.
  - `clinical_graph_rules.json`: Drug-indication, UCUM-aware safety/contraindication, longitudinal lab-monitoring, and punctuation-safe note hold/resume directive rules.
  - `acip_rules.json`: CDC ACIP vaccine groups and schedule validation metadata.
  - `wahs_corpus.jsonl`: Curated clinical reference passages indexed in thread-safe in-memory SQLite FTS5 (`kb.py`) with pure-stdlib keyword fallback.
- **Electrolyte-safe drug normalization (`normalizers.py`):** Preserves primary mineral/electrolyte cations (`potassium chloride`, `potassium citrate`, `sodium bicarbonate`, `calcium carbonate`, `magnesium oxide`, `lithium carbonate`) while stripping pharmaceutical counter-ions (`losartan potassium` -> `losartan`, `atorvastatin calcium` -> `atorvastatin`).
- **Fail-closed posture validation:** `validate_sharing_posture()` accepts only `"patient_self"` and `"external_share"`, raising `ValueError` on unknown posture strings across condensation, graph construction, temporal digest, and lookup tools.

### Layer 1: Faithful 16-Resource FHIR R4 Extraction (`extractors.py`)
- Parses `Patient`, `AllergyIntolerance`, `Condition`, `MedicationRequest`, `MedicationAdministration`, `MedicationStatement`, `MedicationDispense`, `Immunization`, `DocumentReference`, `DiagnosticReport`, `Observation`, `Procedure`, `CarePlan`, `Goal`, `ImagingStudy`, `Device`, `Encounter`, and `Appointment` (while resolving `Organization`, `Practitioner`, `Medication`, and `Provenance.target[]` into `source_org`).
- **Multi-coding terminology preference:** `_coding_pair()` prefers standard clinical systems (`RxNorm`, `SNOMED CT`, `LOINC`, `CVX`) when `code.coding[]` lists billing or local hospital codes at `coding[0]`.
- **Invalidated status filtering:** Excludes `entered-in-error` allergies, medications, immunizations, procedures, notes, diagnostic reports, care plans, goals, and appointments; demotes `refuted`/`inactive`/`resolved` allergies from mandatory IPS status; skips invalidated child observations inside `DiagnosticReport.result[]`; excludes `not-done` vaccines from ACIP dose counts; and marks `cancelled`/`noshow` appointments `administrative` rather than `planned`.
- **Complete field & privacy coverage:** Reads `DiagnosticReport.conclusion` / `conclusionCode`, list-valued `MedicationStatement.dosage[]`, and `MedicationAdministration.dosage.dose`; classifies sensitivity across codes, display labels, clinical reasons, referenced sensitive `Condition`s, notes, and `source_org`.

### Layer 2: 2-Stage Monotonic Structural Condensation (`condenser.py`, `chart_serializer.py`)
- **Stage A (Lossless Normalization & RLE Folding):** Folds repeated routine outpatient visits, historical lab panels, stopped medications, resolved conditions, social history repeats, and `>3`-dose routine vaccine series while keeping inpatient/ED acuity encounters and `<=3`-dose series unpacked, carrying all constituent resource UUIDs in `member_uuids`.
- **Stage B (Anti-Starvation Prefix Packing & Progressive Note Expansion):**
  - Reserves mandatory IPS safety core items first (`Patient`, actionable `AllergyIntolerance`, active `Condition`, active `Medication*`, anomalous `Immunization`, active `Goal`, active patient `Device`).
  - Ranks optional items in two budget-independent passes (`_PRIMARY_SECTION_QUOTAS` per clinical section in Pass 0, remaining historical tail in Pass 1) and binary-searches the largest prefix fitting `target_tokens`, then expands excerpted notes newest-first.
  - Guarantees strict UUID prefix monotonicity (`covered_uuids(B1) <= covered_uuids(B2)` for all `B1 < B2`) and emits `[2] Completeness Manifest` (`omitted_clinical_items_count`, `omitted_by_section`, `excerpted_notes_count`).

### Layer 3: `ClinicalEntityGraph`, `TemporalDigest` & Overflow Retrieval (`clinical_graph.py`, `temporal_digest.py`, `wahs_tools.py`)
- **7-Edge `ClinicalEntityGraph`:** Constructs typed nodes and edges directly from `ExtractedChart` + `CondensedChartResult`:
  1. `CROSS_REACTS_WITH` (`AllergyIntolerance` -> active `Medication`)
  2. `CONTRAINDICATED_OR_DOSE_ADJUST_IN` (`Medication` -> `Condition` / UCUM-normalized `Observation` / directional glycemic & electrolyte thresholds)
  3. `CONFLICTS_WITH` (duplicate ingredients, overlapping therapeutic classes, mutually exclusive condition stages, active-vs-resolved contradictions, ACIP schedule anomalies)
  4. `CONTRADICTED_OR_MODIFIED_BY_NOTE` (word-boundary longitudinal note hold/stop vs. newer resume directives)
  5. `MONITORED_BY` (`Condition` / `Medication` -> unit-separated `Observation` trajectories)
  6. `TREATS` (`Medication` -> indicated `Condition`)
  7. `REPORTED_BY_ORG` (clinical entity -> `Organization`)
- **Deterministic `TemporalDigest`:** Anchors `reference_date` strictly on completed clinical events (never `Coverage`, `Device.manufactureDate`, or future planned encounters), separates `active`, `on-hold`, and `stopped` medication horizons, and summarizes `n>=2` numeric series while enforcing `n=1` and intraday-tie guards (`trend_supported: False`).
- **Overflow & Note Retrieval (`wahs_tools.py`):** Provides Tier-0 automatic overflow recovery plus read-only tools (`search_chart_entries`, `get_lab_history`, `read_clinical_note`, and Okapi BM25 `retrieve_clinical_note_passages`) with defense-in-depth `external_share` redaction.

---

## 3. `wahs_engine` Runtime, Security & Streaming Contracts

### Authentication & Subject Binding (`jwt_claims.py`, `assembler.py`)
- Extracts `Authorization: Bearer <jwt>` and enforces `exp`, `sub`, optional `WAHS_TRUSTED_SUB_HEADER`, and `wahs.user_sub` match.
- Supports local cryptographic verification via `WAHS_JWT_HS256_SECRET` or pure-stdlib RS256 JWKS verification (`WAHS_JWT_JWKS_URL` / `WAHS_JWT_JWKS_JSON`, `WAHS_JWT_ISSUER`, `WAHS_JWT_AUDIENCE`).
- Even when local JWKS is unset, `WAHSBundleAssembler` validates the token live against `fhir-auth` on every turn before reading `_prefix_cache` or `_dataset_cache`, and verifies `MasterPatient` ownership when `master_patient_id` is supplied explicitly.

### Caching, Single-Flight Coalescing & Event-Loop Offload (`assembler.py`)
- Dataset cache keys: `(verified_sub, dataset_id, lastUpdated)` with single-flight coalescing (`_inflight_datasets`) over a lifespan-scoped shared `httpx.AsyncClient`.
- Prefix cache keys: `(verified_sub, datasets_fingerprint, scope_hash)` bounded by both `max_entries` and `max_bytes`.
- CPU-intensive bundle extraction, condensation, and enrichment offload cleanly via `asyncio.to_thread()` so healthchecks and concurrent SSE streams remain non-blocking.
- Mode A (`wahs.fhir_bundle` when `WAHS_ALLOW_LOCAL_BUNDLE_MODE=true`) and Mode B (`fhir-auth`) apply identical `chart_scope` filters (`include/exclude_sources`, `include/exclude_categories`, `include/exclude_resource_types`, `include/exclude_resource_ids`, `date_from`, `date_to`).

### Token Budgeting, Pre-Flight SSE & `x_wahs` Provenance (`config.py`, `upstream.py`, `app.py`)
- **Effective chart budget:** `derive_effective_chart_budget()` subtracts exact system-prompt tokens, Layer-3 enrichment reserve (`min(1536, 0.125 * context_window)`), tool-schema reserve (`420`), output reserve, and conversation reserve from `spec.context_window`, then applies `spec.chart_budget_ceiling` (`16,384` default) and `wahs.target_budget_tokens` (`@<budget>` auto-lifts `context_window` up to `WAHS_MAX_CONTEXT_WINDOW`, default `131,072`).
- **Pre-flight validated streaming:** `POST /v1/chat/completions` validates authentication, chart assembly, `409` partial-external-share guards, `400` message validation, `422` context-window sizing, and initial upstream HTTP status **before** committing HTTP `200 text/event-stream` headers.
- **Clean output & provenance:** `_StreamingThinkStripper` and `clean_model_text()` strip both closed and length-truncated `<think>` / `<|channel>thought` blocks; defensive tool-argument coercion prevents malformed LLM JSON from raising `500`; and the terminal response/chunk attaches `x_wahs` (`citation_map`, `scope_applied`, `condensation`, `completeness_manifest`, `temporal_digest`, `privacy_notices`, `timings`, and HL7 `AI-Provenance`).

---

## 4. Privacy Segmentation Governance & Limitations

Under `sharing_posture == "external_share"`, `wahs-engine`:
1. Rejects truncated `$everything` fetches with HTTP `409` unless `allow_partial_external_share=true`.
2. Redacts items carrying HL7 restricted `meta.security` labels, curated sensitive screening/condition codes, or lexical MOUD/HIV/SUD/IPV/reproductive/behavioral-health markers across condensed charts, entity graphs, temporal digests, BM25 passages, and lookup tools.
3. Emits an explicit redaction count in `x_wahs.privacy_notices`.

**Important limitation:** Code and lexical fallbacks are a defense-in-depth engineering safeguard and do **not** replace certified institutional 42 CFR Part 2, WA My Health My Data Act (RCW 19.373), or clinician-reviewed DS4P segmentation workflows. Production deployments sharing records externally should apply authoritative `Resource.meta.security` labels upstream.
