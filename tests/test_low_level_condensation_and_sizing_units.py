"""Low-level unit tests covering `wahs_core` & `wahs_engine` functions and boundaries."""

from __future__ import annotations

import base64
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
import httpx
import pytest

from wahs_core import (
    build_wahs_hub_enrichment_block,
    condense_bundle_to_budget,
    estimate_tokens_local,
    extract_bundle,
)
from wahs_core.wahs_patient_workspace import build_patient_skills_and_condensation_matrix
from wahs_engine.assembler import (
    AssembledWahsContext,
    WAHSBundleAssembler,
    _BoundedLRUCache,
    _clone_context_for_request,
    _parse_fhir_instant,
    _rewrite_next_link_to_fhir_auth,
    derive_effective_chart_budget,
)
from wahs_engine.config import (
    ModelSpec,
    WahsEngineSettings,
    _infer_provider_flavor,
    _infer_tokenizer_family,
)
from wahs_engine.jwt_claims import decode_and_bind_jwt
from wahs_engine.schemas import OpenAIChatCompletionRequest, WahsRequestOptions
from wahs_engine.tokenizers_registry import get_exact_token_counter
from wahs_engine.upstream import (
    SouthboundOpenAIClient,
    _apply_provider_southbound_params,
    _compute_suspected_truncation,
    _trim_history_to_fit,
)


def _clinical_uuids(res: Any) -> set[str]:
    out: set[str] = set()
    for m in res.mappings:
        ru = str(m.get("resourceUuid") or "")
        if ru and not ru.startswith("manifest-"):
            out.add(ru)
        for mu in m.get("memberUuids") or []:
            ms = str(mu or "")
            if ms and not ms.startswith("manifest-"):
                out.add(ms)
    return out


def _make_dense_safety_bundle() -> Dict[str, Any]:
    """Synthetic multi-conflict FHIR bundle exercising every mandatory & optional bucket."""
    long_note_body = (
        "Chief Complaint: Shortness of breath and leg edema. "
        "Assessment and Plan: Continue lisinopril 10 mg daily; monitor creatinine and potassium. "
        + ("Follow renal diet, daily weights, and home blood pressure log. " * 22)
    )
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-1",
                "name": [{"given": ["Alex"], "family": "Rivera"}],
                "birthDate": "1985-04-10",
                "gender": "male",
            }
        },
        {
            "resource": {
                "resourceType": "AllergyIntolerance",
                "id": "allergy-pen",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "verificationStatus": {"coding": [{"code": "confirmed"}]},
                "criticality": "high",
                "category": ["medication"],
                "recordedDate": "2010-05-01",
                "code": {"text": "Penicillin"},
                "reaction": [{"manifestation": [{"text": "Anaphylaxis"}], "severity": "severe"}],
            }
        },
        {
            "resource": {
                "resourceType": "Device",
                "id": "dev-cpap-1",
                "status": "active",
                "type": {"text": "Appliance for sleep apnea (physical object)"},
                "note": [{"time": "2024-02-01", "text": "CPAP humidifier active"}],
            }
        },
        {
            "resource": {
                "resourceType": "Goal",
                "id": "goal-active-1",
                "lifecycleStatus": "active",
                "startDate": "2025-10-01",
                "description": {"text": "Walk 30 minutes daily"},
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-ckd2",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2021-01-10",
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "431856006",
                            "display": "Chronic kidney disease stage 2 (disorder)",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-ckd3",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2025-01-10",
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "433144002",
                            "display": "Chronic kidney disease stage 3 (disorder)",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-sens-crim",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2019-06-01",
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "266948004",
                            "display": "Has a criminal record (finding)",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "medreq-lisinopril",
                "status": "active",
                "authoredOn": "2025-11-01",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "314076",
                            "display": "lisinopril 10 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Immunization",
                "id": "imm-covid-a",
                "status": "completed",
                "occurrenceDateTime": "2024-10-01",
                "vaccineCode": {"coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": "208"}], "text": "COVID-19, mRNA"},
            }
        },
        {
            "resource": {
                "resourceType": "Immunization",
                "id": "imm-covid-b",
                "status": "completed",
                "occurrenceDateTime": "2024-10-05",
                "vaccineCode": {"coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": "208"}], "text": "COVID-19, mRNA"},
            }
        },
    ]
    for i in range(60):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-egfr-{i}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"2025-{1 + (i % 10):02d}-{1 + (i % 25):02d}",
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "33914-3", "display": "eGFR"}]
                    },
                    "valueQuantity": {"value": 62 - (i % 15), "unit": "mL/min/1.73_m2"},
                }
            }
        )
    for idx, dt in enumerate(("2024-01-10", "2025-01-10", "2026-01-10")):
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-note-{idx}",
                    "status": "current",
                    "date": f"{dt}T10:00:00Z",
                    "type": {"text": f"Progress Note {dt}"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(
                                    f"NOTE_DATE_{dt}: {long_note_body}".encode("utf-8")
                                ).decode("ascii"),
                            }
                        }
                    ],
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_estimate_tokens_local_calibration_and_edge_cases() -> None:
    assert estimate_tokens_local("") == 0
    # Per-digit splitting: digits never get merged into 4-char blocks
    assert estimate_tokens_local("1234567890") >= 10
    assert estimate_tokens_local("HbA1c 7.4% (2026-10-15) >= 100") > 10

    gemma_cnt = get_exact_token_counter("gemma4")
    o200k_cnt = get_exact_token_counter("o200k_base")
    bundle = _make_dense_safety_bundle()
    rendered = condense_bundle_to_budget(bundle, target_tokens=8192, token_counter=gemma_cnt)
    est = estimate_tokens_local(rendered.chart_text)
    g_toks = gemma_cnt(rendered.chart_text)
    o_toks = o200k_cnt(rendered.chart_text)
    assert 0.90 <= (est / max(1, g_toks)) <= 1.15
    assert est >= o_toks

    fixture_paths = sorted(glob.glob("/Users/pmanko/code/hq/wahs/bundles/*.json"))
    if fixture_paths:
        first_bundle = json.loads(Path(fixture_paths[0]).read_text(encoding="utf-8"))
        res_8k = condense_bundle_to_budget(first_bundle, target_tokens=8192, token_counter=gemma_cnt)
        est_8k = estimate_tokens_local(res_8k.chart_text)
        g_8k = gemma_cnt(res_8k.chart_text)
        assert 0.90 <= (est_8k / max(1, g_8k)) <= 1.15


def test_pure_stdlib_condenser_512_safety_core_and_monotonicity() -> None:
    bundle = _make_dense_safety_bundle()
    required_uuids = {
        "pat-1",
        "allergy-pen",
        "dev-cpap-1",
        "goal-active-1",
        "cond-ckd2",
        "cond-ckd3",
        "cond-sens-crim",
        "medreq-lisinopril",
        "imm-covid-a",
        "imm-covid-b",
    }

    prev_uuids: set[str] = set()
    for budget in (512, 1024, 2048, 4096, 8192, 16384):
        res = condense_bundle_to_budget(bundle, target_tokens=budget, token_counter=None)
        assert res.actual_tokens <= budget
        assert res.completeness_manifest["omitted_clinical_items_count"] >= 0
        present = _clinical_uuids(res)
        missing = required_uuids - present
        assert not missing, f"Pure-stdlib budget={budget} dropped safety/goal UUIDs: {missing}"
        assert not (prev_uuids - present), f"Dropped clinical UUIDs at budget={budget}: {prev_uuids - present}"
        prev_uuids = present

    res_512 = condense_bundle_to_budget(bundle, target_tokens=512, token_counter=None)
    assert "Allergy: Penicillin" in res_512.chart_text
    assert res_512.actual_tokens <= 512

    # Force compact_details=True ultra folding and verify duplicate conflict_note deduplication
    import copy
    tight_bundle = copy.deepcopy(bundle)
    for m_idx in range(1, 9):
        tight_bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": f"tight-med-{m_idx}",
                    "status": "active",
                    "medicationCodeableConcept": {
                        "text": f"Long Extended Release Antihypertensive Regimen Compound {m_idx} 50mg Oral Tablet"
                    },
                    "dosageInstruction": [
                        {"text": "Take one tablet by mouth twice daily with meals for blood pressure"}
                    ],
                }
            }
        )
    res_compact = condense_bundle_to_budget(tight_bundle, target_tokens=620, token_counter=None)
    cond_lines = [ln for ln in res_compact.chart_text.splitlines() if "Condition (" in ln]
    assert cond_lines
    assert cond_lines[0].count("[stage_conflict:") == 1

    # Argument validation guards
    for bad_budget in (0, -100, True):
        with pytest.raises(ValueError):
            condense_bundle_to_budget(bundle, target_tokens=bad_budget)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        condense_bundle_to_budget(bundle, target_tokens=50)
    with pytest.raises(ValueError):
        condense_bundle_to_budget(bundle, target_tokens=2048, mode="bogus")
    with pytest.raises(ValueError):
        condense_bundle_to_budget(bundle, target_tokens=2048, sharing_posture="bogus")


def test_stage_a_unbounded_categories_vs_stage_b_newest_first_note_expansion() -> None:
    gemma_cnt = get_exact_token_counter("gemma4")
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-unbounded",
                "name": [{"given": ["Jordan"], "family": "Kim"}],
                "birthDate": "1972-07-04",
            }
        }
    ]
    # 45 distinct lab analytes (previously truncated by hardcoded [:20] / [:32] caps)
    for i in range(45):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-uniq-{i:02d}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"2026-01-{(1 + (i % 25)):02d}",
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"99{i:03d}-1", "display": f"UniqueAnalyte{i:02d}"}]},
                    "valueQuantity": {"value": 10 + i, "unit": "mg/dL"},
                }
            }
        )
    # Three dated notes of ~3.5k chars each (~1.9k tokens total full vs ~240 excerpted)
    filler = "Detailed outpatient nephrology interval assessment and medication reconciliation plan. " * 42
    for idx, dt in enumerate(("2024-01-10", "2025-01-10", "2026-01-10")):
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-date-{idx}",
                    "status": "current",
                    "date": f"{dt}T09:00:00Z",
                    "type": {"text": f"Clinic Note {dt}"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(f"MARKER_{dt}_START {filler} MARKER_{dt}_END".encode()).decode("ascii"),
                            }
                        }
                    ],
                }
            }
        )
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}

    # Stage A fast-path fits all 45 distinct labs and all 3 notes in full
    res_full = condense_bundle_to_budget(bundle, target_tokens=16384, token_counter=gemma_cnt)
    assert res_full.tier == 1
    assert res_full.completeness_manifest["omitted_clinical_items_count"] == 0
    assert res_full.completeness_manifest["excerpted_notes_count"] == 0
    for i in range(45):
        assert f"UniqueAnalyte{i:02d}" in res_full.chart_text
    assert "MARKER_2024-01-10_END" in res_full.chart_text
    assert "MARKER_2026-01-10_END" in res_full.chart_text

    # Pick a tight intermediate budget where structured items + 1 full note fit, leaving older notes excerpted
    res_all_excerpted = condense_bundle_to_budget(bundle, target_tokens=2048, token_counter=gemma_cnt)
    if res_all_excerpted.completeness_manifest["omitted_clinical_items_count"] == 0:
        assert res_all_excerpted.tier == 2
        assert res_all_excerpted.completeness_manifest["excerpted_notes_count"] >= 1

    # Sweep tight step sizes to catch an intermediate Tier 2 state (omitted=0, some excerpted)
    intermediate = None
    for b in range(1600, 6500, 150):
        probe = condense_bundle_to_budget(bundle, target_tokens=b, token_counter=gemma_cnt)
        if (
            probe.completeness_manifest["omitted_clinical_items_count"] == 0
            and 0 < probe.completeness_manifest["excerpted_notes_count"] < 3
        ):
            intermediate = probe
            break
    assert intermediate is not None, "Expected Tier 2 partial full-note expansion state"
    assert intermediate.tier == 2
    assert "MARKER_2026-01-10_END" in intermediate.chart_text
    assert "MARKER_2024-01-10_END" not in intermediate.chart_text


def test_workspace_canonical_zero_omitted_budget_waits_for_full_notes() -> None:
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-ws",
                "name": [{"given": ["Dana"], "family": "Lin"}],
                "birthDate": "1978-03-11",
            }
        }
    ]
    # Add five distinct clinical notes large enough (~5k tokens full) to stay excerpted at 2048 but fit completely by 8192
    for idx, dt in enumerate(("2025-09-01", "2025-10-01", "2025-11-01", "2026-01-01", "2026-02-01")):
        note_text = (
            f"Interval History visit {idx} on {dt}: Blood pressure reviewed on lisinopril. "
            + " ".join(f"finding_series_{idx}_item_{w} monitored stably" for w in range(95))
        )
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-ws-{idx}",
                    "status": "current",
                    "date": f"{dt}T10:00:00Z",
                    "type": {"text": f"Progress Note {dt}"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(note_text.encode("utf-8")).decode("ascii"),
                            }
                        }
                    ],
                }
            }
        )
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}

    ws = build_patient_skills_and_condensation_matrix(bundle)
    hier = ws["condensation_matrix"]["hierarchical"]
    assert hier["2048"]["omitted_records_count"] == 0
    assert hier["2048"]["tier"] == 2
    assert "alias_of_budget" not in hier["2048"]

    # 8192 has full un-excerpted notes (tier 1), so 16384+ becomes alias_of_budget == "4096" or "8192"
    first_alias_target = hier["32768"].get("alias_of_budget")
    assert first_alias_target in {"4096", "8192"}
    assert hier[first_alias_target]["tier"] == 1
    assert hier[first_alias_target]["omitted_records_count"] == 0


def test_hub_enrichment_binary_search_respects_exact_bpe_counters() -> None:
    bundle = _make_dense_safety_bundle()
    extracted = extract_bundle(bundle)
    condensed_512 = condense_bundle_to_budget(extracted, target_tokens=512)
    msgs = [
        {
            "role": "user",
            "content": (
                f"{condensed_512.chart_text}\n\n"
                "USER_QUESTION: Summarize my CKD stages, eGFR labs, progress notes, and immunization schedule."
            ),
        }
    ]
    for fam in ("gemma4", "o200k_base"):
        cnt = get_exact_token_counter(fam)
        for max_toks in (120, 240, 500):
            res = build_wahs_hub_enrichment_block(
                msgs,
                extracted=extracted,
                max_tokens=max_toks,
                token_counter=cnt,
            )
            assert cnt(res.enrichment_text) <= max_toks
            assert res.enrichment_tokens == cnt(res.enrichment_text)


def test_wahs_engine_settings_from_env_matrix_and_fhir_internal_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _infer_tokenizer_family("wahs-12b-kb", "llama-3") == "gemma4"
    assert _infer_tokenizer_family("gpt-4o", "gpt-4o") == "o200k_base"
    assert _infer_provider_flavor("gpt-5-mini", "litellm") == "openai"
    assert _infer_provider_flavor("ollama/gemma4:12b", "ollama") == "ollama"

    monkeypatch.setenv("WAHS_FHIR_AUTH_BASE_URL", "http://fhir-auth:8080/fhir")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm-gw:4000")
    monkeypatch.setenv("WAHS_CONTEXT_WINDOW", "65536")
    monkeypatch.setenv("WAHS_CHART_BUDGET_CEILING", "none")
    monkeypatch.setenv("WAHS_PROVIDER_FLAVOR", "llamacpp")
    monkeypatch.setenv("WAHS_CACHE_TTL_SECONDS", "900")
    monkeypatch.setenv("WAHS_MAX_EVERYTHING_RESOURCES", "15000")
    monkeypatch.setenv("WAHS_ASSEMBLY_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv(
        "WAHS_MODEL_SPECS_JSON",
        json.dumps(
            {
                "frontier-1m": {
                    "upstream_model": "gpt-5",
                    "context_window": 1048576,
                    "chart_budget_ceiling": None,
                    "default_max_output_tokens": 2048,
                    "tokenizer_family": "o200k_base",
                    "provider_flavor": "openai",
                    "disable_thinking": False,
                },
                "custom-local": {
                    "context_window": 16384,
                    "chart_budget_ceiling": 8192,
                },
            }
        ),
    )
    cfg = WahsEngineSettings.from_env()
    assert cfg.default_context_window == 65536
    assert cfg.default_chart_budget_ceiling is None
    assert cfg.default_provider_flavor == "llamacpp"
    assert cfg.cache_ttl_seconds == 900.0
    assert cfg.max_everything_resources == 15000
    assert cfg.assembly_timeout_seconds == 30.0
    assert cfg.upstream_chat_completions_url == "http://llm-gw:4000/v1/chat/completions"
    assert cfg.upstream_models_url == "http://llm-gw:4000/v1/models"

    sp_1m = cfg.resolve_model_spec("frontier-1m")
    assert sp_1m.context_window == 1048576
    assert sp_1m.chart_budget_ceiling is None
    assert sp_1m.default_max_output_tokens == 2048
    assert sp_1m.tokenizer_family == "o200k_base"
    assert sp_1m.provider_flavor == "openai"
    assert sp_1m.disable_thinking is False

    sp_fallback = cfg.resolve_model_spec("gemma4:e4b")
    assert sp_fallback.context_window == 65536
    assert sp_fallback.chart_budget_ceiling is None
    assert sp_fallback.provider_flavor == "llamacpp"

    monkeypatch.setenv("WAHS_FHIR_AUTH_BASE_URL", "http://fhir-internal:8080/fhir")
    with pytest.raises(ValueError):
        WahsEngineSettings.from_env()


def test_derive_effective_chart_budget_all_branches() -> None:
    spec_capped = ModelSpec(
        upstream_model="ollama/gemma4:12b",
        context_window=32768,
        chart_budget_ceiling=16384,
        default_max_output_tokens=1024,
        tokenizer_family="gemma4",
    )
    spec_uncapped = ModelSpec(
        upstream_model="gpt-5",
        context_window=131072,
        chart_budget_ceiling=None,
        default_max_output_tokens=1024,
        tokenizer_family="o200k_base",
    )

    b1, r1 = derive_effective_chart_budget(spec_capped, WahsRequestOptions())
    assert (b1, r1) == (16384, "chart_budget_ceiling")

    b2, r2 = derive_effective_chart_budget(spec_capped, WahsRequestOptions(target_budget_tokens=4096))
    assert (b2, r2) == (4096, "caller_target_budget_tokens")

    b3, r3 = derive_effective_chart_budget(spec_capped, WahsRequestOptions(target_budget_tokens=200000))
    assert (b3, r3) == (16384, "clamped_to_chart_budget_ceiling")

    b4, r4 = derive_effective_chart_budget(spec_uncapped, WahsRequestOptions())
    assert b4 > 65536 and r4 == "context_window_derived"

    b5, r5 = derive_effective_chart_budget(spec_uncapped, WahsRequestOptions(target_budget_tokens=200000))
    assert b5 == b4 and r5 == "clamped_to_context_window"

    b6, r6 = derive_effective_chart_budget(
        spec_capped, WahsRequestOptions(retrieval_strategy="dynamic_fhir_tools")
    )
    assert (b6, r6) == (2048, "dynamic_fhir_tools_header")

    b7, r7 = derive_effective_chart_budget(
        spec_capped,
        WahsRequestOptions(retrieval_strategy="dynamic_fhir_tools", target_budget_tokens=8000),
    )
    assert (b7, r7) == (2048, "dynamic_fhir_tools_header")

    b8, r8 = derive_effective_chart_budget(
        spec_capped,
        WahsRequestOptions(retrieval_strategy="dynamic_fhir_tools", target_budget_tokens=1024),
    )
    assert (b8, r8) == (1024, "caller_target_budget_tokens")

    # KB (+800) + overflow tools (+420) deductions reduce uncapped window budget cleanly
    b_chart_only, _ = derive_effective_chart_budget(
        spec_uncapped, WahsRequestOptions(retrieval_strategy="chart_only", enable_overflow_tools=False)
    )
    b_kb, _ = derive_effective_chart_budget(spec_uncapped, WahsRequestOptions(enable_kb=True, enable_overflow_tools=False))
    b_kb_tools, _ = derive_effective_chart_budget(spec_uncapped, WahsRequestOptions(enable_kb=True, enable_overflow_tools=True))
    assert b_chart_only - b_kb == 800
    assert b_kb - b_kb_tools == 420

    tiny_bad = ModelSpec(
        upstream_model="too-small",
        context_window=1024,
        chart_budget_ceiling=None,
        default_max_output_tokens=256,
    )
    with pytest.raises(HTTPException) as exc:
        derive_effective_chart_budget(tiny_bad, WahsRequestOptions())
    assert exc.value.status_code == 422


def test_bounded_lru_cache_eviction_and_monotonic_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("wahs_engine.assembler.time.monotonic", lambda: clock[0])

    cache: _BoundedLRUCache[str, int] = _BoundedLRUCache(max_entries=2, ttl_seconds=10.0)
    cache.put("a", 1)
    clock[0] = 101.0
    cache.put("b", 2)
    # Touch "a" so "b" is LRU
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

    # Advance past TTL for "a" and "c"
    clock[0] = 112.0
    assert cache.get("a") is None
    assert cache.get("c") is None
    assert len(cache) == 0

    no_ttl: _BoundedLRUCache[str, int] = _BoundedLRUCache(max_entries=2, ttl_seconds=0.0)
    no_ttl.put("k", 9)
    clock[0] = 999999.0
    assert no_ttl.get("k") == 9


@pytest.mark.asyncio
async def test_assembler_helpers_instant_next_link_clone_and_max_resources() -> None:
    assert _parse_fhir_instant("")[:2] == (0, 0.0)
    assert _parse_fhir_instant("not-a-timestamp")[:2] == (0, 0.0)
    t_utc = _parse_fhir_instant("2026-11-10T12:00:00Z")
    t_ist = _parse_fhir_instant("2026-11-10T17:30:00+05:30")
    t_naive = _parse_fhir_instant("2026-11-10T12:00:00")
    assert t_utc[0] == 1 and t_utc[1] == t_ist[1] == t_naive[1]

    # Critical trailing slash for jwt-proxy /fhir/?_getpages=... routing
    assert (
        _rewrite_next_link_to_fhir_auth("http://fhir-auth:8008/fhir", "https://fhir-internal:8080/fhir?_getpages=abc&_count=50")
        == "http://fhir-auth:8008/fhir/?_getpages=abc&_count=50"
    )
    assert (
        _rewrite_next_link_to_fhir_auth("http://fhir-auth:8008/fhir/", "http://fhir-internal:8080/fhir/Patient/p1/$everything?_count=50")
        == "http://fhir-auth:8008/fhir/Patient/p1/$everything?_count=50"
    )

    # _clone_context_for_request must isolate mutable fields
    bundle = _make_dense_safety_bundle()
    extracted = extract_bundle(bundle)
    condensed = condense_bundle_to_budget(extracted, target_tokens=2048)
    orig = AssembledWahsContext(
        verified_sub="sub-1",
        patient_id="pat-1",
        patient_name="Alex Rivera",
        birth_date="1985-04-10",
        system_prompt="sys",
        prefix_cache_key_hash="hash-1",
        datasets_fingerprint="fp-1",
        scope_hash="sh-1",
        datasets_included=[],
        scope_applied={},
        security_audit={},
        extracted=extracted,
        condensed=condensed,
        citation_map={"1": {"id": "pat-1"}},
        privacy_notices=["n1"],
        partial_datasets=[{"dataset_id": "ds-1"}],
    )
    cloned = _clone_context_for_request(
        orig, cache_hit=True, budget_requested=1000, clamp_reason="caller_target_budget_tokens"
    )
    cloned.citation_map["99"] = {"id": "mutated"}
    cloned.privacy_notices.append("n2")
    cloned.partial_datasets[0]["dataset_id"] = "mutated"
    assert "99" not in orig.citation_map
    assert orig.privacy_notices == ["n1"]
    assert orig.partial_datasets[0]["dataset_id"] == "ds-1"
    assert orig.cache_hit is False

    # Mid-page max_everything_resources truncation check
    def fhir_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "resourceType": "Bundle",
                "type": "searchset",
                "link": [{"relation": "next", "url": "http://fhir-internal:8080/fhir?_getpages=page2"}],
                "entry": [
                    {"resource": {"resourceType": "Observation", "id": f"obs-m-{i}"}}
                    for i in range(10)
                ],
            },
        )

    settings = WahsEngineSettings(
        fhir_auth_base_url="http://fhir-auth:8080/fhir",
        everything_page_size=10,
        max_everything_pages=5,
        max_everything_resources=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(fhir_handler)) as client:
        asm = WAHSBundleAssembler(settings, http_client=client)
        collected, is_partial, reason = await asm._fetch_paginated_everything(
            client, settings.fhir_auth_base_url, "ds-1", "dummy-token"
        )
    assert is_partial is True
    assert reason == "max_everything_resources"
    assert len(collected) == 4


@pytest.mark.asyncio
async def test_upstream_helpers_flavors_truncation_trimming_and_tool_round_422() -> None:
    p_litellm: Dict[str, Any] = {}
    _apply_provider_southbound_params(
        p_litellm, ModelSpec("ollama/gemma4:12b", context_window=32768, provider_flavor="litellm", disable_thinking=True)
    )
    assert p_litellm == {"num_ctx": 32768, "reasoning_effort": "none"}

    p_ollama: Dict[str, Any] = {}
    _apply_provider_southbound_params(
        p_ollama, ModelSpec("gemma4:12b", context_window=16384, provider_flavor="ollama", disable_thinking=False)
    )
    assert p_ollama == {"num_ctx": 16384}

    p_llama: Dict[str, Any] = {}
    _apply_provider_southbound_params(
        p_llama, ModelSpec("gemma4-q4", context_window=32768, provider_flavor="llamacpp", disable_thinking=True)
    )
    assert p_llama == {"chat_template_kwargs": {"enable_thinking": False}}

    p_openai: Dict[str, Any] = {}
    _apply_provider_southbound_params(
        p_openai, ModelSpec("gpt-5", context_window=1048576, provider_flavor="openai", disable_thinking=True)
    )
    assert p_openai == {}

    # Truncation boundary checks (`exact_prompt_tokens >= 1024` and `< 0.85 * exact`)
    assert _compute_suspected_truncation(llm_rounds=1, upstream_prompt_tokens=800, exact_prompt_tokens=1000) is False
    assert _compute_suspected_truncation(llm_rounds=1, upstream_prompt_tokens=0, exact_prompt_tokens=2000) is False
    assert _compute_suspected_truncation(llm_rounds=2, upstream_prompt_tokens=500, exact_prompt_tokens=2000) is False
    assert _compute_suspected_truncation(llm_rounds=1, upstream_prompt_tokens=849, exact_prompt_tokens=1024) is True
    assert _compute_suspected_truncation(llm_rounds=1, upstream_prompt_tokens=871, exact_prompt_tokens=1024) is False

    # _trim_history_to_fit preserves messages[0] and latest user turn while dropping oldest pairs
    msgs = [
        {"role": "system", "content": "SYSTEM_CHART_FIXED"},
        {"role": "user", "content": "Old Q " + ("alpha " * 500)},
        {"role": "assistant", "content": "Old A " + ("beta " * 500)},
        {"role": "user", "content": "Latest Q"},
    ]
    toks, trimmed = _trim_history_to_fit(
        msgs, tools=None, tokenizer_family="gemma4", context_window=1024, max_output_tokens=256
    )
    assert trimmed == 2
    assert toks <= 768
    assert msgs == [
        {"role": "system", "content": "SYSTEM_CHART_FIXED"},
        {"role": "user", "content": "Latest Q"},
    ]

    with pytest.raises(HTTPException) as exc_trim:
        _trim_history_to_fit(
            [{"role": "system", "content": "word " * 1000}, {"role": "user", "content": "hi"}],
            tools=None,
            tokenizer_family="gemma4",
            context_window=512,
            max_output_tokens=128,
        )
    assert exc_trim.value.status_code == 422

    # Round-2 tool-loop context-window overflow raises HTTP 422 BEFORE sending a 2nd upstream POST
    bundle = _make_dense_safety_bundle()
    extracted = extract_bundle(bundle)
    condensed_512 = condense_bundle_to_budget(extracted, target_tokens=512)
    ctx_tiny = AssembledWahsContext(
        verified_sub="sub-t",
        patient_id="pat-1",
        patient_name="Alex Rivera",
        birth_date="1985-04-10",
        system_prompt=condensed_512.chart_text,
        prefix_cache_key_hash="h-t",
        datasets_fingerprint="f-t",
        scope_hash="s-t",
        datasets_included=[],
        scope_applied={},
        security_audit={},
        extracted=extracted,
        condensed=condensed_512,
        citation_map={},
        privacy_notices=[],
        context_window=1400,
        chart_budget_ceiling=512,
        budget_effective=512,
    )
    calls: List[Dict[str, Any]] = []

    def tool_llm_handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-tool-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tc-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_clinical_note",
                                        "arguments": json.dumps({"note_id_or_query": "Progress", "max_chars": 8000}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 20},
            },
        )

    settings_t = WahsEngineSettings(
        llm_base_url="http://llm-backend:4000/v1",
        default_model="tiny-tool",
        model_specs={
            "tiny-tool": ModelSpec(
                upstream_model="ollama/gemma4:12b",
                context_window=1600,
                chart_budget_ceiling=512,
                default_max_output_tokens=128,
            )
        },
    )
    req_t = OpenAIChatCompletionRequest(
        model="tiny-tool",
        stream=False,
        max_tokens=128,
        messages=[{"role": "user", "content": "Read my notes"}],
    )
    opts_t = WahsRequestOptions(enable_kb=False, enable_overflow_tools=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(tool_llm_handler)) as llm_client:
        southbound = SouthboundOpenAIClient(settings_t, http_client=llm_client)
        with pytest.raises(HTTPException) as exc_round2:
            await southbound.complete(ctx=ctx_tiny, req=req_t, options=opts_t)
    assert exc_round2.value.status_code == 422
    assert len(calls) == 1
