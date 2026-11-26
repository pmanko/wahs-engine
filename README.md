# `wahs-engine`

Minimal, zero-privilege OpenAI-compatible FHIR R4 condensation and citation middleware for **Washington Health Summary (WAHS)**.

## Architecture (3-Step Baseline + Unplugged Extension Shelf)

```mermaid
flowchart LR
  App["WAHS Web & Mobile BFFs<br/>(Bearer JWT + OpenAI Request)"]

  subgraph Engine["wahs-engine Core Baseline (Default)"]
    direction TB
    S1["1. Fetch & Scope<br/>Reads patient records via fhir-auth<br/>Applies explicit UI chart_scope"]
    S2["2. Verbatim-First Condense<br/>Verbatim [1]..[N] lines when chart <= budget<br/>Uniform RLE + water-filled notes if > budget<br/>Emits [2] Completeness Manifest"]
    S3["3. Extension Port (OFF by default)<br/>ContextEnricher Protocol hook<br/>Pass-through when WAHS_ENABLED_ENRICHERS=''"]
    S1 --> S2 --> S3
  end

  GPU["Stateless OpenAI GPU Host<br/>(wahs-26b / wahs-12b / wahs-e4b)"]

  subgraph Shelf["src/wahs_extensions/ Extracted Catalog (Unplugged by Default)"]
    direction TB
    X1["note_rag · Okapi BM25 Note Passages"]
    X2["overflow_lookup · Omitted Item Recovery"]
    X3["temporal_trends · Tie-Safe Lab Math"]
    X4["experimental/clinical_reconciliation · Sample Rules"]
  end

  App -->|"POST /v1/chat/completions"| Engine
  Engine -->|"messages[0]=Chart + User Turns"| GPU
  GPU -->|"Streamed Reply"| Engine
  Engine -->|"SSE + x_wahs.citation_map"| App
  Shelf -.->|"Future Opt-In via WAHS_ENABLED_ENRICHERS"| S3
```

## Package Layout

- `src/wahs_core/`: Pure-Python stdlib (`python -S` verified, zero third-party deps):
  - `extractors.py`: Domain-neutral FHIR R4 parser (`ClinicalItem`, `ClinicalNote`, `NumericPoint`).
  - `condenser.py`: 3-Tier Verbatim-First Monotonic Condenser (`Tier 0` Verbatim -> `Tier 1` Ontology-Free RLE -> `Tier 2` Budget-Packed).
  - `chart_serializer.py`: `[1]..[N]` chart renderer + citation traceability mappings.
  - `ports.py`: Hexagonal `ContextEnricher` `Protocol` + `EnrichmentDisclosure` schemas.
- `src/wahs_engine/`: Slim FastAPI runtime (`app.py`, `assembler.py`, `upstream.py`, `pipeline.py`, `profile_registry.py`, `prompt_loader.py`, `profiles.yaml`, `prompts/*.txt`, `config.py`, `schemas.py`, `jwt_claims.py`, `tokenizers_registry.py`).
- `src/wahs_extensions/`: Toggleable generalizable enricher catalog ([README](src/wahs_extensions/README.md)): `temporal`, `terminology_kb` (`CIEL v2026-04-28` + `RxNorm/ATC/CVX/LOINC` in-memory SQLite), `note_rag`, `overflow_lookup`, and `guideline_kb` (`[K1]..[KM]`), OFF by default (`default_profile: baseline` never imports `wahs_extensions`).
- `src/wahs_devtools/`: Synthetic HAPI/Keycloak demo persona seeder (`wahs-seed-hapi`).

## Wire Contract (`/v1/models`, `/v1/wahs/profiles`, `/v1/chat/completions`, `/v1/wahs/context/preview`)

- **Clean Model IDs:** `wahs-26b` (`ollama/gemma4:26b`, default), `wahs-12b` (`ollama/gemma4:12b`), `wahs-e4b` (`ollama/gemma4:e4b`).
- **Declarative Profiles & Live-Reloadable Prompts (`wahs`):** `profile` (`baseline` default, `temporal`, `terminology`, `temporal-terminology`, `rag-overflow`, `core-enriched`, `full-enriched`), `prompt` (`prompts/<id>.txt`, live-reloadable via `WAHS_PROMPTS_DIR`), and `enrichers` overrides, plus `master_patient_id`, `chart_scope`, `overlay_resources`, optional `target_budget_tokens`, `refresh`, `emit_stream_events`.
- **Non-Goals by Design:** Middleware privacy redaction/segmentation is intentionally unbuilt here pending UW CIRG selective-disclosure design; access isolation is enforced by `fhir-auth` and explicit UI `chart_scope` / `ips_bundle_id`.
