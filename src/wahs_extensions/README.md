# WAHS Extensions Catalog (`src/wahs_extensions/`)

Optional, unplugged R&D modules implementing [`ContextEnricher`](../wahs_core/ports.py).

## Why Core `wahs-engine` Keeps These Unplugged by Default

The production baseline of `wahs-engine` performs **strictly faithful FHIR R4 structural projection, verbatim-first condensation (`Tier 0` when chart fits available tokens, uniform RLE and note water-filling only when > budget), and `[1]..[N]` provenance citations** with zero embedded clinical rule packs and zero speculative middleware heuristics.

All enrichment modules below are opt-in via `WAHS_ENABLED_ENRICHERS` (comma-separated, empty `""` by default so `wahs_extensions` is never imported at startup or request time).

## Extension Catalog

| Enricher ID | Maturity | Scope & Mechanism | Disclosure `kind` | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| `note_rag` | General R&D (`BM25NotePassageEnricher`) | Pure-stdlib single-patient Okapi BM25 passage retrieval over excerpted or omitted `ClinicalNote`s / reports (`k1=1.5`, `b=0.75`). | `retrieval` | Lexical TF-IDF matching only; no synonym expansion or semantic embeddings; inactive in Tier 0 when full notes already fit. |
| `overflow_lookup` | General R&D (`OmittedRecordRecoveryEnricher`) | Recovers omitted `ClinicalItem`s matching query terms when `omitted_clinical_items_count > 0` and assigns sequential `[N]` citations. Includes read-only chart search helpers. | `retrieval` | Keyword matching over omitted sections; inactive when chart fits within `target_budget_tokens`. |
| `temporal_trends` | General R&D (`QuantitativeTrendEnricher`) | Longitudinal quantitative series math (UCUM-safe grouping, comparator preservation, `n == 1` no-trend guard, intraday tie safety) surfaced on trend questions. | `quantitative_summary` | Deterministic arithmetic over extracted `NumericPoint`s only; skips unparseable free-text results and incompatible units. |
| `experimental.clinical_reconciliation` | Experimental Prototype (`PrototypeClinicalGraphEnricher`) | Sample RxNorm/ACIP/CKD/diabetes rule packs (`sample_packs/`) + entity graph cross-checks for care-transition experiments. | `experimental_clinical_rules` | Prototype sample ontology covering selected chronic care scenarios only; NOT an exhaustive clinical knowledge base and NOT validated as standalone CDS. |

## Historical H200 Empirical Context

In our 900-cell H200 benchmark across `Gemma 4` (`E4B`, `12B`, `26B`), `Qwen 3` (`8B`, `14B`), and `Llama 3.1 8B`, single-patient `BM25` note retrieval (`note_rag`) improved Gemma 4 12B score at `16k` from `78.5` (`chart_only`) to `82.8/100` on large synthetic and MIMIC-IV charts (`18.4 MB`), while full `26B + KB` reached `85.1/100` (`0/75` harms). Every enricher here emits an `EnrichmentDisclosure` block surfaced in `x_wahs.enrichers` and `x_wahs.ai_provenance` whenever enabled.

## Enabling in Staging or Benchmarks

```bash
export WAHS_ENABLED_ENRICHERS="note_rag,temporal_trends"
```
