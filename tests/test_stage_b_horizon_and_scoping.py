"""Regression tests for Stage-B primary clinical horizon, hold-clause preservation, Rule 5 privacy, and prompt scoping."""

from __future__ import annotations

import base64
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from wahs_core import build_wahs_hub_enrichment_block
from wahs_core.condenser import condense_bundle_to_budget
from wahs_engine.app import create_app
from wahs_engine.assembler import WAHS_SYSTEM_PROMPT
from wahs_engine.config import ModelSpec, WahsEngineSettings
from wahs_engine.tokenizers_registry import get_exact_token_counter


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


def _build_starvation_stress_bundle() -> Dict[str, Any]:
    """Synthetic chart with hundreds of routine historical labs before clinical notes, vitals, and imaging."""
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-starve",
                "name": [{"given": ["Jordan"], "family": "Mercer"}],
                "birthDate": "1958-02-14",
                "gender": "male",
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-ckd3",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2023-01-01",
                "code": {"text": "Chronic kidney disease stage 3B"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-lis-starve",
                "status": "active",
                "authoredOn": "2025-10-01",
                "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
            }
        },
    ]
    # 350 distinct routine numeric lab observations (would starve later sections under flat section_rank)
    for i in range(350):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-routine-{i:03d}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"2022-{(i % 12) + 1:02d}-{(i % 25) + 1:02d}",
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"88{i:03d}-1", "display": f"RoutineAnalyte{i:03d}"}]},
                    "valueQuantity": {"value": 10 + (i % 40), "unit": "mg/dL"},
                }
            }
        )
    # Urgent microbiology observation
    entries.append(
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-urine-culture",
                "status": "final",
                "category": [{"coding": [{"code": "laboratory"}]}],
                "effectiveDateTime": "2026-01-25",
                "code": {"text": "Microbiology: URINE CULTURE"},
                "valueString": "Escherichia coli > 100,000 CFU/mL",
            }
        }
    )
    # Recent vital signs panel
    entries.append(
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-vital-bp",
                "status": "final",
                "category": [{"coding": [{"code": "vital-signs"}]}],
                "effectiveDateTime": "2026-01-25",
                "code": {"text": "Blood pressure panel"},
                "component": [
                    {"code": {"text": "Systolic blood pressure"}, "valueQuantity": {"value": 98, "unit": "mm[Hg]"}},
                    {"code": {"text": "Diastolic blood pressure"}, "valueQuantity": {"value": 60, "unit": "mm[Hg]"}},
                ],
            }
        }
    )
    # Clinical note where hold instruction appears past char 400 (without conflict_note pre-tagging)
    long_filler = "Hospital course stable on telemetry monitoring and IV fluid repletion with close nephrology follow-up. " * 6
    note_body = f"Discharge Summary (2026-01-26): {long_filler} Assessment and Plan: Hold lisinopril and spironolactone immediately at hospital discharge due to hyperkalemia and BP 98/60."
    entries.append(
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-discharge-hold",
                "status": "current",
                "date": "2026-01-26T12:00:00Z",
                "type": {"text": "Discharge Summary"},
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": base64.b64encode(note_body.encode("utf-8")).decode("ascii"),
                        }
                    }
                ],
            }
        }
    )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_stage_b_primary_horizon_prevents_section_starvation_and_preserves_hold_clauses() -> None:
    gemma_cnt = get_exact_token_counter("gemma4")
    bundle = _build_starvation_stress_bundle()

    prev_uuids: set[str] = set()
    for budget in (1024, 2048, 4096, 8192, 16384):
        res = condense_bundle_to_budget(bundle, target_tokens=budget, token_counter=gemma_cnt)
        assert res.actual_tokens <= budget
        cur = _clinical_uuids(res)
        assert not (prev_uuids - cur), f"Non-monotonic UUID drop at {budget}: {prev_uuids - cur}"
        prev_uuids = cur

    res_4k = condense_bundle_to_budget(bundle, target_tokens=4096, token_counter=gemma_cnt)
    lower_4k = res_4k.chart_text.lower()
    assert "doc-discharge-hold" in _clinical_uuids(res_4k)
    assert "obs-vital-bp" in _clinical_uuids(res_4k)
    assert "obs-urine-culture" in _clinical_uuids(res_4k)
    assert "hold lisinopril" in lower_4k
    assert "98" in res_4k.chart_text


def test_large_hq_fixtures_keep_notes_vitals_imaging_and_holds_at_16k_32k() -> None:
    bundles_dir = Path("/Users/pmanko/code/hq/wahs/bundles")
    if not bundles_dir.exists():
        pytest.skip("Fixture bundles under /Users/pmanko/code/hq/wahs/bundles not present")
    gemma_cnt = get_exact_token_counter("gemma4")

    marcus = json.loads((bundles_dir / "Marcus_Vance_Oncology_Immunotherapy.json").read_text(encoding="utf-8"))
    for budget in (8192, 16384, 32768):
        res = condense_bundle_to_budget(marcus, target_tokens=budget, token_counter=gemma_cnt)
        docs = sum(1 for m in res.mappings if m.get("resourceType") == "DocumentReference")
        bp_lines = sum(1 for rec in res.records if "blood pressure" in (rec.get("text") or "").lower())
        assert docs >= 10, f"Expected >=10 clinical notes for Marcus_Vance at {budget}, got {docs}"
        assert bp_lines >= 1, f"Expected blood pressure vitals for Marcus_Vance at {budget}"
        assert "stop naproxen" in res.chart_text.lower()

    mimic = json.loads((bundles_dir / "MIMIC_10014354_ICU_Heme_Cardiorenal.json").read_text(encoding="utf-8"))
    for budget in (8192, 16384, 32768):
        res = condense_bundle_to_budget(mimic, target_tokens=budget, token_counter=gemma_cnt)
        docs = sum(1 for m in res.mappings if m.get("resourceType") == "DocumentReference")
        lower_txt = res.chart_text.lower()
        assert docs >= 10, f"Expected >=10 clinical notes for MIMIC_10014354 at {budget}, got {docs}"
        assert "hold lisinopril" in lower_txt
        assert "stop simvastatin" in lower_txt
        assert "urine culture" in lower_txt

    maurice = json.loads((bundles_dir / "Maurice742_Auer97.json").read_text(encoding="utf-8"))
    res_m16 = condense_bundle_to_budget(maurice, target_tokens=16384, token_counter=gemma_cnt)
    assert any(m.get("resourceType") == "ImagingStudy" for m in res_m16.mappings)
    assert any(m.get("resourceType") == "DiagnosticReport" for m in res_m16.mappings)
    assert any(m.get("resourceType") == "CarePlan" for m in res_m16.mappings)

    # Full monotonic matrix check including dense budgets
    for path_str in sorted(glob.glob(str(bundles_dir / "*.json"))):
        b = json.loads(Path(path_str).read_text(encoding="utf-8"))
        prev: set[str] = set()
        for bdg in (1024, 2048, 2800, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144):
            r = condense_bundle_to_budget(b, target_tokens=bdg, token_counter=gemma_cnt)
            cur = _clinical_uuids(r)
            assert not (prev - cur), f"{Path(path_str).name} non-monotonic at {bdg}: {prev - cur}"
            prev = cur


def test_rule5_privacy_and_kb_note_rag_scoping_headers() -> None:
    assert "MINIMUM-NECESSARY CONVERSATIONAL PRIVACY" in WAHS_SYSTEM_PROMPT
    assert "third party" in WAHS_SYSTEM_PROMPT.lower()
    enr = build_wahs_hub_enrichment_block("USER_QUESTION: What is my potassium?")
    assert "Use ONLY subgraph links directly relevant" in enr.enrichment_text


@pytest.mark.asyncio
async def test_upstream_kb_and_note_rag_scoping_headers() -> None:
    bundle = _build_starvation_stress_bundle()
    captured: List[Dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            captured.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-scope-1",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hold lisinopril [1]."}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 12, "total_tokens": 512},
                },
            )
        return httpx.Response(404)

    settings = WahsEngineSettings(
        fhir_auth_base_url="http://fhir-auth:8080/fhir",
        llm_base_url="http://llm-backend:4000/v1",
        allow_local_bundle_mode=True,
        default_model="wahs-12b-kb",
        model_specs={
            "wahs-12b-kb": ModelSpec(
                upstream_model="ollama/gemma4:12b",
                context_window=32768,
                chart_budget_ceiling=16384,
                tokenizer_family="gemma4",
                provider_flavor="litellm",
                disable_thinking=True,
            )
        },
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as client_http:
        app = create_app(settings, llm_http_client=client_http)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What should I do with lisinopril after discharge?"}],
                    "wahs": {
                        "fhir_bundle": bundle,
                        "retrieval_strategy": "note_rag",
                        "enable_kb": True,
                    },
                },
            )
            assert resp.status_code == 200
            last_user = captured[-1]["messages"][-1]["content"]
            assert "Note Precedence:" in last_user
            assert "Use ONLY subgraph links directly relevant" in last_user
