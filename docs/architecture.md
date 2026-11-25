# WAHS Engine Runtime Architecture

## Overview

`wahs-engine` sits between stateless BFF clients (`shl-ips`, `wahs-mobile-frontend`) and two zero-privilege boundaries:

1. **Northbound FHIR Gateway (`fhir-auth`):** Enforces RS256 Bearer JWT authentication and patient compartment isolation (`meta.security.code == sub`). `wahs-engine` never stores patient data on disk and never talks directly to `fhir-internal`.
2. **Southbound OpenAI GPU Host (`LLM_BASE_URL`):** Stateless OpenAI-compatible inference for clean model aliases (`wahs-26b` default -> `ollama/gemma4:26b`, `wahs-12b`, `wahs-e4b`) with prefix-cache-friendly immutable `messages[0]` system framing.

## 3-Step Request Pipeline

1. **Fetch & Scope (`assembler.py`):** Resolves the authenticated Master Patient and linked dataset stubs, applies declarative `wahs.chart_scope` and unsaved `overlay_resources` (`[P1]..[PM]`), verifies single-patient references, and caches scoped snapshots in bounded in-memory LRU caches.
2. **Verbatim-First Condensation (`wahs_core/condenser.py`):**
   - **Tier 0 (Verbatim Complete Ledger):** When all extracted clinical items and full UTF-8 clinical notes fit inside `target_tokens` (`16,384` default ceiling), emits every entry verbatim (`folded_items_count=0`, `omitted_clinical_items_count=0`).
   - **Tier 1 (Ontology-Free RLE Folded Ledger):** Only when Tier 0 exceeds budget, run-length folds repeated identical observations, renewals, panels, resolved conditions, procedures, and routine visits by FHIR code/unit/setting while keeping the HL7 IPS Safety Core, acute encounters, and full notes verbatim.
   - **Tier 2 (Budget-Packed Ledger):** Preserves the IPS Safety Core intact, binary-searches a deterministic section-interleaved recency prefix (with clean 400-char head-only note previews), and expands newest notes back to full text while within budget.
3. **Extension Port (`pipeline.py` & `wahs_core/ports.py`):** Evaluates `ContextEnricher` plugins configured via `WAHS_ENABLED_ENRICHERS`. When empty (default), returns immediately without importing `wahs_extensions`, passing user turns byte-for-byte to the GPU host and streaming tokens immediately from Round 0.
