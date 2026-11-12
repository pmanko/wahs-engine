"""Stage 4 tests: BM25 clinical note RAG, dynamic FHIR tools, seed forwarding, and timings telemetry."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import pytest

from wahs_core import retrieve_clinical_note_passages
from wahs_engine.app import create_app
from wahs_engine.config import WahsEngineSettings


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _sample_long_note_bundle() -> Dict[str, Any]:
    adm18_text = (
        "BIDMC Admission #18 (2025-04-12 to 2025-04-17): Patient admitted with complicated "
        "Escherichia coli urinary tract infection and transient acute kidney injury (creatinine 1.9 mg/dL). "
        "Treated with IV ceftriaxone, transitioned to oral cefpodoxime with full resolution."
    )
    adm19_text = (
        "BIDMC Admission #19 (2025-05-03 to 2025-05-08): Readmitted for volume overload and "
        "acute decompensated heart failure. Diuresed 4.2 liters with IV furosemide; potassium peaked "
        "at 5.6 mmol/L so spironolactone 25 mg was held at discharge."
    )
    sdoh_note_text = (
        " confidential Social Work Note (2025-06-01): Screening positive for intimate partner violence "
        "and domestic abuse safety plan established under confidential SDOH protocol."
    )
    return {
        "resourceType": "Bundle",
        "id": "bundle-stage4-test",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-s4",
                    "name": [{"given": ["Elena"], "family": "Rostova"}],
                    "birthDate": "1958-04-10",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-spiro",
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": "Patient/pat-s4"},
                    "authoredOn": "2026-01-10",
                    "medicationCodeableConcept": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "313096",
                                "display": "Spironolactone 25 MG Oral Tablet",
                            }
                        ],
                        "text": "Spironolactone 25 MG Oral Tablet",
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-k-56",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {
                        "coding": [
                            {"system": "http://loinc.org", "code": "2823-3", "display": "Potassium"}
                        ],
                        "text": "Potassium",
                    },
                    "subject": {"reference": "Patient/pat-s4"},
                    "effectiveDateTime": "2026-01-18",
                    "valueQuantity": {"value": 5.6, "unit": "mmol/L"},
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-adm-18",
                    "status": "current",
                    "type": {"text": "Discharge Summary Admission #18"},
                    "subject": {"reference": "Patient/pat-s4"},
                    "date": "2025-04-17T15:00:00Z",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64(adm18_text),
                            }
                        }
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-adm-19",
                    "status": "current",
                    "type": {"text": "Discharge Summary Admission #19"},
                    "subject": {"reference": "Patient/pat-s4"},
                    "date": "2025-05-08T14:00:00Z",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64(adm19_text),
                            }
                        }
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-sdoh-ipv",
                    "status": "current",
                    "type": {"text": "Confidential Social Work Assessment"},
                    "subject": {"reference": "Patient/pat-s4"},
                    "date": "2025-06-01T10:00:00Z",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64(sdoh_note_text),
                            }
                        }
                    ],
                }
            },
        ],
    }


def test_retrieve_clinical_note_passages_bm25_and_privacy_redaction() -> None:
    bundle = _sample_long_note_bundle()
    res = retrieve_clinical_note_passages(
        bundle,
        "Summarize my April 2025 urinary tract infection hospital admission #18",
        sharing_posture="patient_self",
        top_k=3,
    )
    assert res["total_chunks_indexed"] >= 3
    assert len(res["results"]) >= 1
    assert res["results"][0]["resource_uuid"] == "doc-adm-18"
    assert "urinary tract infection" in res["results"][0]["passage_text"].lower()

    # External share must redact SDOH/IPV confidential note
    ext = retrieve_clinical_note_passages(
        bundle,
        "intimate partner violence domestic abuse safety plan",
        sharing_posture="external_share",
        top_k=5,
    )
    assert ext["redacted_for_privacy_count"] >= 1
    returned_uuids = {r["resource_uuid"] for r in ext["results"]}
    assert "doc-sdoh-ipv" not in returned_uuids


def test_note_rag_and_dynamic_fhir_tools_with_seed_and_timings() -> None:
    bundle = _sample_long_note_bundle()
    captured_payloads: List[Dict[str, Any]] = []

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        msgs = body.get("messages") or []
        has_tool_result = any(m.get("role") == "tool" for m in msgs)

        if body.get("tools") and not has_tool_result and "dynamic tool test" in msgs[-1]["content"].lower():
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-tool-1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_note_18",
                                        "type": "function",
                                        "function": {
                                            "name": "read_clinical_note",
                                            "arguments": json.dumps({"note_id_or_query": "doc-adm-18"}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "timings": {
                        "prompt_n": 320,
                        "prompt_ms": 140.0,
                        "predicted_n": 22,
                        "predicted_ms": 85.0,
                        "cache_n": 100,
                    },
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "cmpl-final",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Admission #18 treated E. coli urinary tract infection [4].",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 410, "completion_tokens": 18, "total_tokens": 428},
                "timings": {
                    "prompt_n": 410,
                    "prompt_ms": 160.0,
                    "predicted_n": 18,
                    "predicted_ms": 95.0,
                    "cache_n": 280,
                },
            },
        )

    settings = WahsEngineSettings(
        fhir_auth_base_url="http://mock-fhir",
        llm_base_url="http://mock-llm/v1",
        default_model="gemma-e4b",
        allow_local_bundle_mode=True,
    )
    llm_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_llm_handler))
    app = create_app(settings=settings, llm_http_client=llm_client)
    client = TestClient(app)

    # 1. Test `note_rag` + `seed` forwarding + `timings`
    resp_rag = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "seed": 43,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": "What happened during my April 2025 urinary tract infection admission #18?",
                }
            ],
            "wahs": {
                "fhir_bundle": bundle,
                "target_budget_tokens": 4096,
                "retrieval_strategy": "note_rag",
            },
        },
    )
    assert resp_rag.status_code == 200
    data_rag = resp_rag.json()
    assert captured_payloads[-1]["seed"] == 43
    x_rag = data_rag["x_wahs"]
    assert x_rag["retrieval_strategy"] == "note_rag"
    assert x_rag["note_rag"]["passages_retrieved"] >= 1
    cited_uuids = {v["id"] for v in x_rag["citation_map"].values() if isinstance(v, dict)}
    assert "doc-adm-18" in cited_uuids
    assert x_rag["timings"]["prompt_ms"] == pytest.approx(160.0)
    assert x_rag["timings"]["predicted_ms"] == pytest.approx(95.0)

    # 2. Test `dynamic_fhir_tools` multi-round tool execution + cumulative timings
    resp_tools = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "seed": 44,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": "Dynamic tool test: look up my April 2025 admission #18 clinical note.",
                }
            ],
            "wahs": {
                "fhir_bundle": bundle,
                "target_budget_tokens": 8192,
                "retrieval_strategy": "dynamic_fhir_tools",
            },
        },
    )
    assert resp_tools.status_code == 200
    data_tools = resp_tools.json()
    x_tools = data_tools["x_wahs"]
    assert x_tools["retrieval_strategy"] == "dynamic_fhir_tools"
    # Compact IPS header budget <= 2048 even when requested_budget_tokens == 8192
    assert x_tools["condensation"]["budget_tokens"] == 2048
    assert x_tools["condensation"]["requested_budget_tokens"] == 8192
    assert len(x_tools["tool_calls"]) == 1
    assert x_tools["tool_calls"][0]["name"] == "read_clinical_note"
    assert x_tools["timings"]["llm_rounds"] == 2
    assert x_tools["timings"]["prompt_ms"] == pytest.approx(300.0)
    assert x_tools["timings"]["predicted_ms"] == pytest.approx(180.0)


def test_external_share_blocks_tier0_overflow_recovery_across_all_strategies() -> None:
    """Regression test: when `sharing_posture == 'external_share'` and condensation drops items
    (`omitted_clinical_items > 0`), Tier-0 overflow recovery (`build_wahs_hub_enrichment_block`)
    and KB lookup must NOT re-inject redacted `SDOH/PRIVACY-SENSITIVE` active Conditions or notes.
    """
    bundle = _sample_long_note_bundle()
    # Add active SDOH/IPV and social isolation Conditions plus historical labs so omitted_clinical_items > 0 at small budget
    bundle["entry"].extend(
        [
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "smart-cond-sdoh-ipv",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "706893006",
                                "display": "Victim of intimate partner abuse (finding)",
                            }
                        ],
                        "text": "Victim of intimate partner abuse",
                    },
                    "subject": {"reference": "Patient/pat-long-1"},
                    "onsetDateTime": "2025-05-02",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "margaret-cond-sdoh-iso",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "55600006",
                                "display": "Social isolation (finding)",
                            }
                        ],
                        "text": "Social isolation",
                    },
                    "subject": {"reference": "Patient/pat-long-1"},
                    "onsetDateTime": "2025-08-19",
                }
            },
            *[
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-hist-{idx}",
                        "status": "final",
                        "category": [{"coding": [{"code": "laboratory"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]},
                        "subject": {"reference": "Patient/pat-long-1"},
                        "effectiveDateTime": f"2022-0{(idx % 9) + 1}-10",
                        "valueQuantity": {"value": 1.1 + idx * 0.01, "unit": "mg/dL"},
                    }
                }
                for idx in range(25)
            ],
        ]
    )

    captured_payloads: List[Dict[str, Any]] = []

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-ext-1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "External referral summary limited to active medical care [1].",
                        },
                    }
                ],
                "timings": {"prompt_n": 240, "prompt_ms": 90.0, "predicted_n": 18, "predicted_ms": 50.0},
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="gemma-e4b",
    )
    llm_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_llm_handler))
    client = TestClient(create_app(settings=settings, llm_http_client=llm_client))

    forbidden = (
        "smart-cond-sdoh-ipv",
        "margaret-cond-sdoh-iso",
        "doc-sdoh-ipv",
        "intimate partner abuse",
        "intimate partner violence",
        "social isolation",
    )
    for strat in ("condensation_kb", "note_rag", "dynamic_fhir_tools"):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-e4b",
                "seed": 42,
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate an external referral summary of active medical conditions and hospitalizations.",
                    }
                ],
                "wahs": {
                    "fhir_bundle": bundle,
                    "target_budget_tokens": 512,
                    "sharing_posture": "external_share",
                    "retrieval_strategy": strat,
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        sent_blob = json.dumps(captured_payloads[-1]).lower()
        resp_blob = json.dumps(body).lower()
        for term in forbidden:
            assert term not in sent_blob, f"{strat} leaked {term!r} into upstream LLM prompt"
            assert term not in resp_blob, f"{strat} leaked {term!r} into response / citation_map"


def test_cache_miss_citation_enrichment_does_not_pollute_subsequent_requests() -> None:
    """Regression test for PR#2-C1: cache-miss requests adding Tier-0 overflow or note_rag citations
    must not mutate the prefix-cached `citation_map` seen by subsequent independent requests."""
    bundle = _sample_long_note_bundle()
    bundle["entry"].extend(
        [
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-old-endo",
                    "status": "current",
                    "type": {"text": "Infectious Disease Consult 2017"},
                    "subject": {"reference": "Patient/pat-s4"},
                    "date": "2017-03-11T10:00:00Z",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64("Historical workup in 2017 ruled out infective endocarditis and valvular vegetation."),
                            }
                        }
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-old-pneumo",
                    "status": "current",
                    "type": {"text": "Pulmonary Consult 2019"},
                    "subject": {"reference": "Patient/pat-s4"},
                    "date": "2019-06-15T10:00:00Z",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64("Chest radiograph in 2019 evaluated for small apical pneumothorax without effusion."),
                            }
                        }
                    ],
                }
            },
            *[
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-recent-filler-{idx}",
                        "status": "final",
                        "category": [{"coding": [{"code": "laboratory"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": f"99{idx:03d}-1", "display": f"Panel Marker {idx}"}]},
                        "subject": {"reference": "Patient/pat-s4"},
                        "effectiveDateTime": f"2026-02-{(idx % 20) + 1:02d}",
                        "valueQuantity": {"value": 10 + idx, "unit": "U/L"},
                    }
                }
                for idx in range(25)
            ],
            *[
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-hist-creat-{idx}",
                        "status": "final",
                        "category": [{"coding": [{"code": "laboratory"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]},
                        "subject": {"reference": "Patient/pat-s4"},
                        "effectiveDateTime": f"2021-0{(idx % 9) + 1}-05",
                        "valueQuantity": {"value": 1.1 + idx * 0.02, "unit": "mg/dL"},
                    }
                }
                for idx in range(12)
            ],
            *[
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-hist-ferritin-{idx}",
                        "status": "final",
                        "category": [{"coding": [{"code": "laboratory"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": "2276-4", "display": "Ferritin"}]},
                        "subject": {"reference": "Patient/pat-s4"},
                        "effectiveDateTime": f"2020-0{(idx % 9) + 1}-12",
                        "valueQuantity": {"value": 45 + idx, "unit": "ng/mL"},
                    }
                }
                for idx in range(12)
            ],
        ]
    )

    def mock_llm_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl-cache-iso",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Reviewed chart entry [1]."},
                    }
                ],
                "timings": {"prompt_n": 200, "prompt_ms": 50.0, "predicted_n": 10, "predicted_ms": 20.0},
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="gemma-e4b",
    )
    client = TestClient(
        create_app(
            settings=settings,
            llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock_llm_handler)),
        )
    )

    # 1. Note RAG: Q1 (cache miss) cites doc-old-endo; Q2 (cache hit) cites doc-old-pneumo without inheriting doc-old-endo
    r_rag1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "messages": [{"role": "user", "content": "Did my consult find infective endocarditis or vegetation?"}],
            "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512, "retrieval_strategy": "note_rag"},
        },
    )
    assert r_rag1.status_code == 200
    xw_rag1 = r_rag1.json()["x_wahs"]
    assert xw_rag1["condensation"]["cache_hit"] is False
    ids_rag1 = {v["id"] for v in xw_rag1["citation_map"].values() if isinstance(v, dict)}
    assert "doc-old-endo" in ids_rag1

    r_rag2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "messages": [{"role": "user", "content": "Did my chest radiograph show an apical pneumothorax?"}],
            "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512, "retrieval_strategy": "note_rag"},
        },
    )
    assert r_rag2.status_code == 200
    xw_rag2 = r_rag2.json()["x_wahs"]
    assert xw_rag2["condensation"]["cache_hit"] is True
    ids_rag2 = {v["id"] for v in xw_rag2["citation_map"].values() if isinstance(v, dict)}
    assert "doc-old-pneumo" in ids_rag2
    assert "doc-old-endo" not in ids_rag2

    # 2. Tier-0 overflow recovery (`condensation_kb`) at a fresh budget (520 -> cache miss on Q1, hit on Q2)
    r_kb1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "messages": [{"role": "user", "content": "What did my 2017 consult say about endocarditis or vegetation?"}],
            "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 520, "retrieval_strategy": "condensation_kb"},
        },
    )
    assert r_kb1.status_code == 200
    xw_kb1 = r_kb1.json()["x_wahs"]
    assert xw_kb1["condensation"]["cache_hit"] is False
    recovered_kb1 = {
        k: v
        for k, v in xw_kb1["citation_map"].items()
        if isinstance(v, dict) and v.get("source") == "Tier-0 Overflow Recovery"
    }
    assert recovered_kb1
    assert any(v.get("id") == "doc-old-endo" for v in recovered_kb1.values())

    r_kb2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "messages": [{"role": "user", "content": "What did my 2019 chest radiograph say about apical pneumothorax?"}],
            "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 520, "retrieval_strategy": "condensation_kb"},
        },
    )
    assert r_kb2.status_code == 200
    xw_kb2 = r_kb2.json()["x_wahs"]
    assert xw_kb2["condensation"]["cache_hit"] is True
    recovered_kb2 = {
        k: v
        for k, v in xw_kb2["citation_map"].items()
        if isinstance(v, dict) and v.get("source") == "Tier-0 Overflow Recovery"
    }
    assert recovered_kb2
    assert any(v.get("id") == "doc-old-pneumo" for v in recovered_kb2.values())
    assert not any(v.get("id") == "doc-old-endo" for v in recovered_kb2.values())


def test_streaming_dynamic_fhir_tools_and_long_note_pagination() -> None:
    """Regression test for PR#2-C2 & PR#2-C3:
    1. `stream=True` + `retrieval_strategy='dynamic_fhir_tools'` executes multi-round tool calls and emits SSE chunks + `x_wahs`.
    2. `read_clinical_note` preserves full >3,200-char text on single-UUID lookup and supports `offset`/`max_chars` pagination.
    """
    bundle = _sample_long_note_bundle()
    long_note_body = (
        ("Hospital course summary: patient admitted for acute decompensated heart failure and bacteremiaworkup. " * 55)
        + "Final discharge pearl: vancomycin trough 14.2 mcg/mL on 2025-05-08."
    )
    assert len(long_note_body) > 5000
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-long-discharge",
                "status": "current",
                "type": {"text": "Long Inpatient Discharge Summary"},
                "subject": {"reference": "Patient/pat-s4"},
                "date": "2025-05-09T11:00:00Z",
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": _b64(long_note_body),
                        }
                    }
                ],
            }
        }
    )

    observed_tool_payloads: List[Dict[str, Any]] = []

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        msgs = body.get("messages") or []
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        for tm in tool_msgs[len(observed_tool_payloads) :]:
            observed_tool_payloads.append(json.loads(tm["content"]))

        assert body.get("stream") is False
        step = len(tool_msgs)
        if step == 0:
            assert body.get("tools")
            tc = {
                "id": "call_uuid_full",
                "type": "function",
                "function": {
                    "name": "read_clinical_note",
                    "arguments": json.dumps({"note_id_or_query": "doc-long-discharge"}),
                },
            }
        elif step == 1:
            tc = {
                "id": "call_page_1",
                "type": "function",
                "function": {
                    "name": "read_clinical_note",
                    "arguments": json.dumps(
                        {"note_id_or_query": "Hospital course summary", "offset": 0, "max_chars": 3200}
                    ),
                },
            }
        elif step == 2:
            tc = {
                "id": "call_page_2",
                "type": "function",
                "function": {
                    "name": "read_clinical_note",
                    "arguments": json.dumps(
                        {"note_id_or_query": "Hospital course summary", "offset": 3200, "max_chars": 3200}
                    ),
                },
            }
        else:
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-stream-tools-final",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Vancomycin trough was 14.2 mcg/mL [4].",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 450, "completion_tokens": 14, "total_tokens": 464},
                    "timings": {"prompt_n": 450, "prompt_ms": 80.0, "predicted_n": 14, "predicted_ms": 30.0},
                },
            )

        return httpx.Response(
            200,
            json={
                "id": f"cmpl-tool-step-{step}",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "", "tool_calls": [tc]},
                    }
                ],
                "timings": {"prompt_n": 250, "prompt_ms": 40.0, "predicted_n": 15, "predicted_ms": 25.0},
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="gemma-e4b",
    )
    client = TestClient(
        create_app(
            settings=settings,
            llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock_llm_handler)),
        )
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-e4b",
            "stream": True,
            "messages": [{"role": "user", "content": "What was the final vancomycin trough in doc-long-discharge?"}],
            "wahs": {
                "fhir_bundle": bundle,
                "target_budget_tokens": 8192,
                "retrieval_strategy": "dynamic_fhir_tools",
            },
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    chunks = []
    saw_done = False
    for line in resp.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(payload))

    assert saw_done is True
    full_text = "".join((c["choices"][0].get("delta") or {}).get("content") or "" for c in chunks)
    assert "Vancomycin trough was 14.2 mcg/mL" in full_text

    # Tool result 1: single-UUID lookup returned full note untruncated without duplicate full_text
    r_uuid = observed_tool_payloads[0]["results"][0]
    assert "full_text" not in r_uuid
    assert r_uuid["has_more"] is False
    assert r_uuid["total_chars"] == len(long_note_body)
    assert "vancomycin trough 14.2 mcg/mL" in r_uuid["text"]

    # Tool results 2 & 3: windowed pagination (`max_chars=3200`, then `offset=3200`)
    r_p1 = observed_tool_payloads[1]["results"][0]
    assert r_p1["has_more"] is True
    assert r_p1["next_offset"] == 3200
    r_p2 = observed_tool_payloads[2]["results"][0]
    assert r_p2["has_more"] is False
    assert "vancomycin trough 14.2 mcg/mL" in r_p2["text"]

    x_wahs = chunks[-1]["x_wahs"]
    assert len(x_wahs["tool_calls"]) == 3
    cited_ids = {v["id"] for v in x_wahs["citation_map"].values() if isinstance(v, dict)}
    assert "doc-long-discharge" in cited_ids

