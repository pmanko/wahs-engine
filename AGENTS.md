# Engineering Guidelines (`wahs-engine`)

## Architectural Invariants
1. **`wahs_core` is pure Python stdlib:** Verify with `PYTHONPATH=src python -S -c "import sys, wahs_core; assert 'site' not in sys.modules and 'wahs_extensions' not in sys.modules"`. Zero third-party imports, zero network calls, zero embedded JSON knowledge packs.
2. **Zero-privilege FHIR isolation:** `wahs-engine` must always connect through `fhir-auth` (`uwcirg/jwt-proxy`), never directly to `fhir-internal`. `WahsEngineSettings` fails closed if `WAHS_FHIR_AUTH_BASE_URL` contains `fhir-internal`.
3. **Extensions unplugged by default:** `WAHS_ENABLED_ENRICHERS` defaults to `""`. When empty, `wahs_extensions` must never be imported at app startup or during `/v1/chat/completions` / `/v1/wahs/context/preview`.
4. **Domain neutrality in core:** Keep clinical rule packs and domain heuristics out of `src/wahs_core/` and `src/wahs_engine/`. Optional enrichment modules belong strictly under `src/wahs_extensions/` implementing `wahs_core.ports.ContextEnricher`.
5. **No speculative privacy regexes:** Selective disclosure is managed via `fhir-auth` security labels and user-controlled `chart_scope` / `ips_bundle_id`. Do not re-introduce lexical sensitivity classifiers into `wahs-engine`.
6. **Zero em dashes:** Do not introduce em dashes (`\u2014`) in documentation, prompts, or chart strings.
