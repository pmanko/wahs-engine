# WAHS Extensions Catalog (`src/wahs_extensions/`)

Optional, unplugged R&D modules implementing [`ContextEnricher`](../wahs_core/ports.py).

## Why Core `wahs-engine` Keeps These Unplugged by Default

The production baseline of `wahs-engine` performs **strictly faithful FHIR R4 structural projection, verbatim-first condensation (`Tier 0` when chart fits available tokens, uniform RLE and note water-filling only when > budget), and `[1]..[N]` provenance citations** with zero embedded clinical rule packs and zero speculative middleware heuristics.

All enrichment modules below are opt-in via `WAHS_ENABLED_ENRICHERS` (comma-separated, empty `""` by default so `wahs_extensions` is never imported at startup or request time).

## Extension Catalog

| Enricher ID | Maturity | Scope & Mechanism | Disclosure `kind` | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| `temporal` | Generalizable (`QuantitativeTrendEnricher`, cap `450t`) | Emits Chart Anchor Date, `6m`/`12m` lookback cutoffs, visit vs administrative/device date guards, vaccine elapsed time, and tie-safe UCUM longitudinal lab deltas (`n == 1` guard). | `quantitative_summary` | Deterministic calendar and extracted `NumericPoint` arithmetic only; skips unparseable free-text results and incompatible units. |
| `terminology_kb` | Generalizable (`TerminologyConceptEnricher`, cap `550t`) | Dual-source in-memory SQLite (`CIEL v2026-04-28` + `RxNorm/ATC`, `CDC CVX/ACIP`, `LOINC`) resolving active drug ingredients/ATC class overlaps, adult vaccine series schedules, missing-range lab intervals, and OpenMRS `CIEL:<id>` concepts. | `terminology_grounding` | Descriptive concept normalization and standard intervals only; never overrides inline `Observation.referenceRange` or issues prescribing advice. |
| `note_rag` | Generalizable (`BM25NotePassageEnricher`, cap `1100t`) | Tier-2 single-patient Okapi BM25 passage retrieval over strictly unseen mid/tail note spans (`rendered_note_horizons`) with `terminology_kb` synonym query expansion and chunk caching. | `retrieval` | Self-gated to Tier 2 (inactive when all notes already fit verbatim in `messages[0]`). |
| `overflow_lookup` | Generalizable (`OmittedRecordRecoveryEnricher`, cap `550t`) | Tier-2 IDF-weighted Okapi BM25 + synonym/code recovery of omitted `ClinicalItem`s with sequential `[N+1]` citations. | `retrieval` | Self-gated to Tier 2 (inactive when `omitted_clinical_items_count == 0`). |
| `guideline_kb` | Generalizable (`PublicGuidelineEnricher`, cap `500t`) | In-memory SQLite FTS5 + chart-coding intersection over provenance-stamped public clinical reference snippets (`[K1]..[KM]` `KnowledgeReference`). | `knowledge_reference` | Educational public guideline snippets only; excluded from `core-enriched` so its effect is cleanly toggleable via `full-enriched`. |
| `experimental.clinical_reconciliation` | Quarantined Prototype (`PrototypeClinicalGraphEnricher`) | Sample chronic-care prototype (`sample_packs/`). Excluded from all `profiles.yaml` entries and rejected (`HTTP 422`) unless `WAHS_ALLOW_EXPERIMENTAL_ENRICHERS=true`. | `experimental_clinical_rules` | Prototype sample ontology only; NOT enabled in standard WAHS profiles. |

## Historical H200 Empirical Context

In our 900-cell H200 benchmark across `Gemma 4` (`E4B`, `12B`, `26B`), `Qwen 3` (`8B`, `14B`), and `Llama 3.1 8B`, single-patient `BM25` note retrieval (`note_rag`) improved Gemma 4 12B score at `16k` from `78.5` (`chart_only`) to `82.8/100` on large synthetic and MIMIC-IV charts (`18.4 MB`), while full `26B + KB` reached `85.1/100` (`0/75` harms). Every enricher here emits an `EnrichmentDisclosure` block surfaced in `x_wahs.enrichers` and `x_wahs.ai_provenance` whenever enabled.

## Enabling in Staging or Benchmarks

```bash
export WAHS_ENABLED_ENRICHERS="note_rag,temporal_trends"
```
