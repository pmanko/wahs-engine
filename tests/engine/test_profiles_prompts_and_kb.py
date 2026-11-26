"""Stage 2 tests for declarative `profiles.yaml`, live `.txt` prompt reload, zero-import baseline,
prefix-hash invariance, Dual-Source `CIEL v2026-04-28` + Public Standards `terminology_kb`,
`temporal`, unseen-span `note_rag`, `overflow_lookup`, and `[K1]` `guideline_kb`.
"""

from __future__ import annotations

import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import pytest

from wahs_core import condense_bundle_to_budget, estimate_tokens_local, extract_bundle
from wahs_core.ports import EnrichmentInput
from wahs_engine.app import create_app
from wahs_engine.config import ModelSpec, WahsEngineSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
MAURICE_PATH = Path("/Users/pmanko/code/hq/wahs/bundles/Maurice742_Auer97.json")


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _clinical_safety_bundle() -> Dict[str, Any]:
    long_head = "Routine clinic preamble documenting vitals, stable diet, and home exercise log. " * 45
    unseen_tail = (
        "CRITICAL UNSEEN TAIL DETAIL: Cardiology consult specifically recommended holding OTC "
        "naproxen and confirming clopidogrel duration following drug-eluting stent placement."
    )
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-stage2",
                "name": [{"given": ["Margaret"], "family": "Test"}],
                "birthDate": "1956-04-12",
                "gender": "female",
            }
        },
        {
            "resource": {
                "resourceType": "Encounter",
                "id": "enc-visit-1",
                "status": "finished",
                "class": {"code": "AMB"},
                "type": [{"text": "Outpatient Primary Care Follow-Up"}],
                "period": {"start": "2026-10-15", "end": "2026-10-15"},
            }
        },
        {
            "resource": {
                "resourceType": "Appointment",
                "id": "appt-future-1",
                "status": "booked",
                "start": "2026-12-05",
                "description": "Cardiology Lipid & Medication Review",
            }
        },
        {
            "resource": {
                "resourceType": "Device",
                "id": "dev-admin-date",
                "status": "active",
                "manufactureDate": "2026-11-01",
                "type": {"text": "Home Blood Pressure Monitor"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-lipitor",
                "status": "active",
                "authoredOn": "2026-10-01",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "617311",
                            "display": "Lipitor 40 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-simva",
                "status": "active",
                "authoredOn": "2026-10-02",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "312961",
                            "display": "Simvastatin 20 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-plavix",
                "status": "active",
                "authoredOn": "2026-10-03",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "213169",
                            "display": "Plavix 75 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-ativan",
                "status": "active",
                "authoredOn": "2026-10-04",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "197901",
                            "display": "Ativan 1 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-ambien",
                "status": "active",
                "authoredOn": "2026-10-05",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "854873",
                            "display": "Ambien 10 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "79651AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2024-03-10",
                "code": {
                    "coding": [
                        {
                            "system": "https://openconceptlab.org/orgs/CIEL/sources/CIEL",
                            "code": "119481",
                            "display": "Diabetes mellitus",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Immunization",
                "id": "imm-shingrix-1",
                "status": "completed",
                "occurrenceDateTime": "2026-07-15",
                "vaccineCode": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/cvx",
                            "code": "187",
                            "display": "Zoster vaccine recombinant, adjuvanted (Shingrix)",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Immunization",
                "id": "imm-covid-old",
                "status": "completed",
                "occurrenceDateTime": "2021-02-14",
                "vaccineCode": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/cvx",
                            "code": "208",
                            "display": "COVID-19, mRNA, LNP-S, PF",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-hba1c-low",
                "status": "final",
                "category": [{"coding": [{"code": "laboratory"}]}],
                "effectiveDateTime": "2026-10-15",
                "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                "valueQuantity": {"value": 3.0, "unit": "%"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-potassium-inline-ref",
                "status": "final",
                "category": [{"coding": [{"code": "laboratory"}]}],
                "effectiveDateTime": "2026-10-15",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3", "display": "Potassium"}]},
                "valueQuantity": {"value": 5.3, "unit": "mmol/L"},
                "referenceRange": [{"low": {"value": 3.6, "unit": "mmol/L"}, "high": {"value": 5.2, "unit": "mmol/L"}}],
            }
        },
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-tail-1",
                "status": "current",
                "date": "2026-10-15",
                "type": {"text": "Cardiology Note"},
                "content": [{"attachment": {"data": _b64(long_head + unseen_tail)}}],
            }
        },
    ]
    for idx in range(25):
        entries.append(
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": f"proc-hist-{idx:02d}",
                    "status": "completed",
                    "performedDateTime": f"2019-0{(idx % 9) + 1}-10",
                    "code": {"text": "Colonoscopy mucosal biopsy" if idx == 12 else f"Routine foot exam #{idx}"},
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_zero_multiline_system_prompts_in_python_source() -> None:
    for py_path in sorted((SRC_DIR / "wahs_engine").glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "PROMPT" in target.id.upper():
                        assert not isinstance(node.value, ast.Constant), f"Hardcoded prompt constant in {py_path}"


def test_subprocess_baseline_never_imports_wahs_extensions() -> None:
    code = (
        "import json, sys, httpx; "
        "from fastapi.testclient import TestClient; "
        "from wahs_engine.app import create_app; "
        "from wahs_engine.config import WahsEngineSettings; "
        "h = lambda r: httpx.Response(200, json={'id':'c1','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':'ok [1].'},'finish_reason':'stop'}],'usage':{'prompt_tokens':50,'completion_tokens':2}}); "
        "app = create_app(WahsEngineSettings(allow_local_bundle_mode=True), llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(h))); "
        "assert 'wahs_extensions' not in sys.modules; "
        "b = {'resourceType':'Bundle','type':'collection','entry':[{'resource':{'resourceType':'Patient','id':'p1'}}]}; "
        "c = TestClient(app); "
        "assert c.get('/healthz').json()['default_profile'] == 'baseline'; "
        "assert c.get('/v1/wahs/profiles').status_code == 200; "
        "assert c.get('/v1/models').status_code == 200; "
        "assert c.post('/v1/wahs/context/preview', json={'model':'wahs-26b','wahs':{'fhir_bundle':b}}).status_code == 200; "
        "assert c.post('/v1/chat/completions', json={'model':'wahs-26b','messages':[{'role':'user','content':'hi'}],'wahs':{'fhir_bundle':b}}).status_code == 200; "
        "assert 'wahs_extensions' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(SRC_DIR), "WAHS_ENABLED_ENRICHERS": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_profiles_endpoint_live_prompt_reload_prefix_invariance_and_422_gates(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "wahs-default.txt"
    prompt_file.write_text("System Prompt Version ONE: cite [1]..[N].", encoding="utf-8")

    captured: List[Dict[str, Any]] = []

    def llm_handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-prof",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Grounded [1] [K1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 4},
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        prompts_dir=str(prompts_dir),
        default_context_window=131072,
        model_specs={
            "wahs-26b": ModelSpec(
                upstream_model="ollama/gemma4:26b",
                context_window=131072,
                preferred_chart_tokens=None,
            )
        },
    )
    app = create_app(settings, llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)))
    bundle = _clinical_safety_bundle()

    with TestClient(app) as client:
        cat = client.get("/v1/wahs/profiles")
        assert cat.status_code == 200
        cdata = cat.json()
        assert cdata["default_profile"] == "baseline"
        assert set(cdata["profiles"]) == {
            "baseline",
            "temporal",
            "terminology",
            "temporal-terminology",
            "rag-overflow",
            "core-enriched",
            "full-enriched",
        }
        sha_v1 = cdata["profiles"]["baseline"]["prompt_sha256"]
        assert len(sha_v1) == 64

        # 1. Prefix-hash invariance across enricher toggles at fixed 32k budget and prompt inside 128k window
        hashes: List[str] = []
        for prof_name in ("baseline", "temporal", "terminology", "core-enriched", "full-enriched"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-26b",
                    "messages": [{"role": "user", "content": "Review my medications, Shingrix vaccine, and HbA1c."}],
                    "wahs": {"fhir_bundle": bundle, "profile": prof_name, "target_budget_tokens": 32768},
                },
            )
            assert r.status_code == 200, r.text
            hashes.append(r.headers["X-WAHS-Prefix-Hash"])
            xw = r.json()["x_wahs"]
            assert xw["condensation"]["budget_effective"] == 32768
            assert xw["condensation"]["profile_id"] == prof_name
            assert xw["condensation"]["prompt_id"] == "wahs-default"
            assert xw["condensation"]["prompt_sha256"] == sha_v1[:12]
            assert xw["ai_provenance"]["prompt_sha256"] == sha_v1
        assert len(set(hashes)) == 1, f"Prefix hash drifted across enricher toggles: {hashes}"
        assert "System Prompt Version ONE" in captured[-1]["messages"][0]["content"]

        # 2. Live .txt prompt edit updates messages[0], prompt_sha256, and X-WAHS-Prefix-Hash without restart
        prompt_file.write_text("System Prompt Version TWO LIVE RELOADED: cite [1]..[N] and [K1].", encoding="utf-8")
        cat2 = client.get("/v1/wahs/profiles").json()
        sha_v2 = cat2["profiles"]["baseline"]["prompt_sha256"]
        assert sha_v2 != sha_v1

        r_reloaded = client.post(
            "/v1/chat/completions",
            json={
                "model": "wahs-26b",
                "messages": [{"role": "user", "content": "Check active meds."}],
                "wahs": {"fhir_bundle": bundle, "profile": "baseline", "target_budget_tokens": 32768},
            },
        )
        assert r_reloaded.status_code == 200
        assert r_reloaded.headers["X-WAHS-Prefix-Hash"] != hashes[0]
        assert "System Prompt Version TWO LIVE RELOADED" in captured[-1]["messages"][0]["content"]
        assert r_reloaded.json()["x_wahs"]["ai_provenance"]["prompt_sha256"] == sha_v2

        # 3. Validation HTTP 422 gates (experimental.*, missing prompt, unknown profile, unknown enricher)
        r_exp = client.post(
            "/v1/wahs/context/preview",
            json={
                "model": "wahs-26b",
                "wahs": {"fhir_bundle": bundle, "enrichers": ["experimental.clinical_reconciliation"]},
            },
        )
        assert r_exp.status_code == 422

        r_bad_prompt = client.post(
            "/v1/wahs/context/preview",
            json={"model": "wahs-26b", "wahs": {"fhir_bundle": bundle, "prompt": "nonexistent-prompt"}},
        )
        assert r_bad_prompt.status_code == 422

        r_bad_prof = client.post(
            "/v1/wahs/context/preview",
            json={"model": "wahs-26b", "wahs": {"fhir_bundle": bundle, "profile": "nonexistent-profile"}},
        )
        assert r_bad_prof.status_code == 422

        r_bad_enr = client.post(
            "/v1/wahs/context/preview",
            json={"model": "wahs-26b", "wahs": {"fhir_bundle": bundle, "enrichers": ["bogus_enricher"]}},
        )
        assert r_bad_enr.status_code == 422

        # Preset expansion ("all_generalizable") works cleanly
        r_preset = client.post(
            "/v1/wahs/context/preview",
            json={
                "model": "wahs-26b",
                "messages": [{"role": "user", "content": "Review statin duplicate and Shingrix series."}],
                "wahs": {"fhir_bundle": bundle, "enrichers": ["all_generalizable"]},
            },
        )
        assert r_preset.status_code == 200
        enr_ids = [e["enricher_id"] for e in r_preset.json()["x_wahs"]["enrichers"]]
        assert enr_ids == ["temporal", "terminology_kb", "note_rag", "overflow_lookup", "guideline_kb"]


def test_terminology_kb_temporal_rag_overflow_and_guideline_kb_clinical_accuracy() -> None:
    bundle = _clinical_safety_bundle()
    settings = WahsEngineSettings(allow_local_bundle_mode=True, default_context_window=131072)
    app = create_app(settings)

    with TestClient(app) as client:
        prev = client.post(
            "/v1/wahs/context/preview",
            json={
                "model": "wahs-26b",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Are any of my active medications duplicated or sedating? "
                            "Am I up to date on Shingrix and COVID vaccines, and what does my HbA1c 3.0% mean?"
                        ),
                    }
                ],
                "wahs": {"fhir_bundle": bundle, "profile": "full-enriched", "target_budget_tokens": 32768},
            },
        )
        assert prev.status_code == 200, prev.text
        body = prev.json()
        chart_text = body["chart_text"]
        xw = body["x_wahs"]
        enr_txt = xw["enrichment_text"]
        enr_low = enr_txt.lower()

        # Immunization dose_status: administered in core chart
        assert "dose_status: administered" in chart_text
        assert "status: completed" not in chart_text

        # Temporal anchor: anchor date, 6m/12m cutoffs, visits, admin/device date warning, vaccine elapsed time
        assert "Chart Anchor Date (latest realized clinical event): 2026-10-15" in enr_txt
        assert "past 6 months >= 2026-04-15 | past 12 months >= 2025-10-15" in enr_txt
        assert "Last Completed Clinical Visit: 2026-10-15" in enr_txt
        assert "Upcoming Scheduled Visits/Appointments: 2026-12-05" in enr_txt
        assert "Administrative/Device Date Warning:" in enr_txt
        assert "COVID-19" in enr_txt and "years before Chart Anchor" in enr_txt

        # Terminology KB: Lipitor (atorvastatin) vs Simvastatin (ATC C10AA), Plavix (clopidogrel), Ativan + Ambien, Shingrix 2-dose, HbA1c 3.0%, CIEL
        assert "atorvastatin" in enr_low and "83367" in enr_txt
        assert "simvastatin" in enr_low and "36567" in enr_txt
        assert "c10aa" in enr_low
        assert "clopidogrel" in enr_low
        assert "lorazepam" in enr_low and "zolpidem" in enr_low
        assert "shingrix" in enr_low and "2-dose" in enr_low
        assert "4.0-5.6" in enr_txt and "below normal reference range" in enr_low
        # Potassium had inline Observation.referenceRange -> must NOT be backfilled in Missing-Range Lab
        assert "Missing-Range Lab (Potassium" not in enr_txt
        assert "CIEL:119481" in enr_txt or "CIEL:79651" in enr_txt

        # Guideline KB: [K1] KnowledgeReference registered in citation_map
        assert "K1" in xw["citation_map"]
        assert xw["citation_map"]["K1"]["resourceType"] == "KnowledgeReference"
        assert "[K1]" in enr_txt

        # Tier-2 tighten budget: verify unseen-span note_rag and overflow_lookup fire and register [N+1]
        prev_t2 = client.post(
            "/v1/wahs/context/preview",
            json={
                "model": "wahs-26b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Did cardiology mention holding naproxen or stent antiplatelet duration, and did I ever have a colonoscopy biopsy?",
                    }
                ],
                "wahs": {"fhir_bundle": bundle, "profile": "rag-overflow", "target_budget_tokens": 900},
            },
        )
        assert prev_t2.status_code == 200, prev_t2.text
        xw2 = prev_t2.json()["x_wahs"]
        assert xw2["condensation"]["tier"] == 2
        assert "CRITICAL UNSEEN TAIL DETAIL" in xw2["enrichment_text"]
        assert "Colonoscopy mucosal biopsy" in xw2["enrichment_text"]
        by_id = {e["enricher_id"]: e for e in xw2["enrichers"]}
        assert by_id["note_rag"]["fired"] is True
        assert by_id["overflow_lookup"]["fired"] is True


@pytest.mark.skipif(not MAURICE_PATH.is_file(), reason="Local Maurice742_Auer97.json fixture not present")
def test_maurice_auer_unseen_note_rag_latency_under_25ms() -> None:
    from wahs_extensions.note_rag import BM25NotePassageEnricher

    bundle = json.loads(MAURICE_PATH.read_text(encoding="utf-8"))
    extracted = extract_bundle(bundle)
    condensed = condense_bundle_to_budget(extracted, target_tokens=16384, token_counter=estimate_tokens_local)
    assert condensed.tier == 2
    cit_map = {str(m["index"]): dict(m, id=m["resourceUuid"]) for m in condensed.mappings}

    enricher = BM25NotePassageEnricher()
    inp = EnrichmentInput(
        extracted=extracted,
        condensed=condensed,
        citation_map=cit_map,
        question="What did the cardiology and hospital discharge notes say about heart failure diuretics and follow-up?",
        conversation=[],
        max_tokens=1100,
        token_counter=estimate_tokens_local,
    )
    # Warm chunk index once, then verify cached query turn runs well under 25 ms (and check first call too)
    t_cold = time.perf_counter()
    out1 = enricher.enrich(inp)
    cold_ms = (time.perf_counter() - t_cold) * 1000.0
    assert out1 is not None and out1.token_count <= 1100

    inp2 = EnrichmentInput(
        extracted=extracted,
        condensed=condensed,
        citation_map=cit_map,
        question="Any mention of anticoagulation bleeding or renal creatinine monitoring in notes?",
        conversation=[],
        max_tokens=1100,
        token_counter=estimate_tokens_local,
    )
    t_warm = time.perf_counter()
    out2 = enricher.enrich(inp2)
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    assert out2 is not None and out2.token_count <= 1100
    assert warm_ms < 25.0, f"Expected cached note_rag < 25 ms on Maurice742_Auer97, got {warm_ms:.2f} ms (cold={cold_ms:.2f} ms)"
