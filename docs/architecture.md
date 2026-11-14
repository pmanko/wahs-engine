# WAHS Engine Architecture

`wahs-engine` is organized into two cleanly separated packages:
- **`wahs_core`** (`src/wahs_core/`): 100% pure Python stdlib clinical FHIR R4 normalization, structural condensation, entity graph enrichment, and chart retrieval (`python3 -S` compatible, zero third-party dependencies).
- **`wahs_engine`** (`src/wahs_engine/`): Thin FastAPI OpenAI `/v1` adapter and `fhir-auth` zero-privilege assembler (`fastapi`, `uvicorn`, `httpx`, `pydantic`).

## Strict 4-Layer `wahs_core` Pipeline

### Layer 0: Reference Knowledge & Normalizers (`kb_data/`, `kb_lookup.py`, `normalizers.py`, `kb.py`)
- Declarative JSON knowledge packs in `src/wahs_core/kb_data/`:
  - `rxnorm_lookup.json`: RxNorm ingredient normalization, brand-to-generic mappings, therapeutic classes, interaction hazards, and allergy cross-reactivity rules.
  - `clinical_exclusivity_and_privacy.json`: mutually exclusive condition stage rules, HL7 `meta.security` restricted confidentiality codes, non-medical/administrative SNOMED/LOINC exclusions, sensitive screening LOINC codes, lexical sensitivity fallbacks, and clinic-room device patterns.
  - `acip_rules.json`: CDC ACIP vaccine groups and schedule metadata.
  - `wahs_corpus.jsonl`: Clinical reference summaries indexed by SNOMED/RxNorm/LOINC/CVX codes and keywords.
- `normalizers.py` exposes stateless rules backed cleanly by cached `kb_lookup.py` loaders:
  - `classify_sensitivity_with_source` applies **HL7 `meta.security`-first** classification (`R`, `V`, `ETH`, `PSY`, `HIV`, `SDV`, `SEX`, `SUD`, `SCA`, `BH`) before code/lexical fallbacks.

### Layer 1: Faithful FHIR R4 Extraction (`extractors.py`)
- Converts raw FHIR R4 Bundles into typed `ExtractedChart`, `ClinicalItem`, and `NumericPoint` dataclasses.
- Reads native FHIR `Observation.interpretation` and `referenceRange` fields for structured `is_abnormal` tagging rather than guessing from text.
- Preserves genuine undated semantics (`date=None`, `date_kind="unknown"` on clinical items without clinical timestamps; `Patient` and `Device.manufactureDate` are excluded from the clinical `reference_date` anchor).
- Preserves clinical note sections verbatim (with SHA-256 deduplication of identical attachments) and never injects synthetic `[conflict]` or `[sensitivity]` suffixes into `ClinicalItem.text`.

### Layer 2: Pure Structural Budget Condensation (`condenser.py`, `chart_serializer.py`)
- `condense_bundle_to_budget(..., condense_mode="tiered")` runs a 2-stage deterministic packing pipeline:
  - **Stage A**: Run-length folds routine outpatient encounters, repeated panels, stopped meds, resolved conditions, vitals, and >3-dose vaccine series while preserving **inpatient/ICU/ED encounters** and **<=3-dose immunization series** unpacked.
  - **Stage B**: Strictly prioritizes mandatory IPS core items, trajectories, active care plans, and newest-first full clinical notes into the exact caller tokenizer budget (`chart_only` remains 100% free of pre-baked conflict/sensitivity tags).
- Assigns stable `[1]..[N]` citations mapped to `resourceUuid` / `memberUuids` and emits a typed `completeness_manifest` (`omitted_clinical_items_count`, `omitted_by_section`).

### Layer 3: Enrichment & Overflow Retrieval (`clinical_graph.py`, `wahs_tools.py`)
- Consumes typed `ExtractedChart` + `CondensedChartResult` directly (never regex-parsing rendered chart text):
  - **Tier-0 Overflow Recovery**: Surfaces high-relevance omitted items with sequential `[N]` mappings when `omitted_clinical_items_count > 0`.
  - **Patient Clinical Entity Graph**: Infers typed edges (`CROSS_REACTS_WITH`, `CONTRAINDICATED_OR_DOSE_ADJUST_IN`, `CONFLICTS_WITH`, `CONTRADICTED_OR_MODIFIED_BY_NOTE`, `MONITORED_BY`, `TREATS`) using Layer-0 rules.
  - **Clinical KB Lookup**: Attaches complete guideline summaries matching patient codes and the active question.
  - **On-Demand Tools (`wahs_tools.py`)**: `search_chart_entries`, `get_lab_history`, `read_clinical_note`, and BM25 `retrieve_clinical_note_passages` (`note_rag`).

## Model-Window Budgeting & Streaming (`wahs_engine/`)
- **`assembler.py`**: Derives effective chart budgets (`derive_effective_chart_budget`) from the model's `context_window` minus dynamic reserves (`ENRICHMENT_STRUCTURAL_RESERVE_TOKENS`, tool schema reserve, output/conversation reserves). `dynamic_fhir_tools` defaults to a `2048`-token IPS header only when `target_budget_tokens` is omitted.
- **`schemas.py` & `upstream.py`**: `max_tokens` defaults to `None` (no silent `1024` output truncation); enrichment receives remaining context-window headroom (`ctx_window - effective_max_tokens - current_toks - 128`); `stream=True` and `stream=False` share the same manifest-driven bounded overflow tool execution loop.

## Privacy Gate Scope & Limitations
- Privacy segmentation under `sharing_posture == "external_share"` prioritizes upstream FHIR `meta.security` labels, curated LOINC/SNOMED codes, and lexical fallbacks (`kb_data/clinical_exclusivity_and_privacy.json`).
- **Limitation**: Code/lexical fallback rules are a defense-in-depth safeguard and are **not** a substitute for certified institutional 42 CFR Part 2, Washington My Health My Data Act (WA MHMDA), or state reproductive/behavioral health segmentation pipelines. Production deployments handling sensitive records should attach authoritative `meta.security` labels upstream before calling `wahs-engine`.
