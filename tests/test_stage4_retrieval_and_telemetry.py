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
