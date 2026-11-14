"""Clean-layer architecture, pure-stdlib, and Layer-3 Temporal Digest verification tests."""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import pytest

import wahs_core
from wahs_core import (
    ClinicalEntityGraph,
    build_temporal_digest,
    build_wahs_hub_enrichment_block,
    condense_bundle_to_budget,
    extract_bundle,
)
from wahs_engine.app import create_app
from wahs_engine.config import ModelSpec, WahsEngineSettings


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _temporal_and_clinical_bundle() -> Dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-temp-1",
                    "name": [{"given": ["Elena"], "family": "Vance"}],
                    "birthDate": "1968-04-12",
                    "gender": "female",
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "allergy-pcn",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": "Penicillin G"},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ckd3",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"coding": [{"code": "433144002", "display": "Chronic kidney disease stage 3"}], "text": "Chronic kidney disease stage 3"},
                    "onsetDateTime": "2024-03-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ckd4",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"coding": [{"code": "431857002", "display": "Chronic kidney disease stage 4"}], "text": "Chronic kidney disease stage 4"},
                    "onsetDateTime": "2026-09-10",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-amox",
                    "status": "active",
                    "medicationCodeableConcept": {"text": "Amoxicillin 500 mg oral capsule"},
                    "authoredOn": "2026-10-01",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-old-nsaid",
                    "status": "stopped",
                    "medicationCodeableConcept": {"text": "Ibuprofen 600 mg tablet"},
                    "authoredOn": "2025-04-10",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-completed",
                    "status": "finished",
                    "class": {"code": "AMB"},
                    "type": [{"text": "Nephrology Follow-Up"}],
                    "period": {"start": "2026-10-15T09:00:00Z", "end": "2026-10-15T09:45:00Z"},
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-future-booked",
                    "status": "booked",
                    "class": {"code": "AMB"},
                    "type": [{"text": "Scheduled Nephrology Lab Review"}],
                    "period": {"start": "2027-01-20T10:00:00Z", "end": "2027-01-20T10:30:00Z"},
                }
            },
            {
                "resource": {
                    "resourceType": "Device",
                    "id": "dev-admin-later",
                    "status": "active",
                    "manufactureDate": "2026-12-01",
                    "type": {"text": "Home Blood Pressure Cuff"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-cr-1",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "2160-0", "display": "Creatinine"}]},
                    "effectiveDateTime": "2025-10-15",
                    "valueQuantity": {"value": 1.1, "unit": "mg/dL"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-cr-2",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "2160-0", "display": "Creatinine"}]},
                    "effectiveDateTime": "2026-10-15",
                    "interpretation": [{"coding": [{"code": "H"}]}],
                    "valueQuantity": {"value": 1.65, "unit": "mg/dL"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-a1c-single",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "effectiveDateTime": "2026-10-15",
                    "valueQuantity": {"value": 7.1, "unit": "%"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-hiv-sens",
                    "status": "final",
                    "meta": {"security": [{"code": "HIV"}]},
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "25836-8", "display": "HIV 1 RNA"}]},
                    "effectiveDateTime": "2026-10-18",
                    "valueQuantity": {"value": 20, "unit": "copies/mL"},
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-short-none",
                    "status": "current",
                    "type": {"text": "Medication Allergy Check"},
                    "date": "2026-10-15",
                    "content": [{"attachment": {"contentType": "text/plain", "data": _b64("No active allergies.")}}],
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-neph-note",
                    "status": "current",
                    "type": {"text": "Nephrology Progress Note"},
                    "date": "2026-10-15",
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": _b64("Creatinine rose to 1.65 mg/dL; avoid nephrotoxic NSAIDs such as ibuprofen."),
                            }
                        }
                    ],
                }
            },
        ],
    }


def test_pure_stdlib_and_removed_legacy_modules() -> None:
    src_dir = str(Path(wahs_core.__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys, wahs_core; "
                "assert 'site' not in sys.modules; "
                "assert hasattr(wahs_core, 'build_temporal_digest'); "
                "assert hasattr(wahs_core, 'ClinicalEntityGraph'); "
                "print(wahs_core.__version__)"
            ),
        ],
        env={"PYTHONPATH": src_dir},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "0.1.0" in proc.stdout

    for dead_mod in (
        "wahs_core.wahs_analyzer",
        "wahs_core.wahs_patient_workspace",
        "wahs_core.wahs_skills",
        "wahs_core.fhir_browser",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(dead_mod)

    assert not hasattr(wahs_core, "extract_bundle_lossless")


def test_layer_purity_typed_graph_and_temporal_digest() -> None:
    bundle = _temporal_and_clinical_bundle()
    extracted = extract_bundle(bundle)
    condensed = condense_bundle_to_budget(bundle, target_tokens=8192)

    assert "[conflict]" not in condensed.chart_text.lower()
    assert "[sensitivity]" not in condensed.chart_text.lower()
    assert "(None)" not in condensed.chart_text
    assert any(
        "No active allergies." in it.text
        for it in extracted.items
        if it.resource_uuid == "doc-short-none"
    )

    graph = ClinicalEntityGraph.from_extracted_chart(extracted, sharing_posture="patient_self")
    assert any(e.edge_type == "CROSS_REACTS_WITH" for e in graph.edges)
    assert any(e.edge_type == "CONFLICTS_WITH" for e in graph.edges)

    ext_graph = ClinicalEntityGraph.from_extracted_chart(extracted, sharing_posture="external_share")
    assert "HIV 1 RNA" not in "\n".join(n.label for n in ext_graph.nodes.values())

    enr_low = build_wahs_hub_enrichment_block(
        "What is my Penicillin allergy status?",
        extracted=extracted,
        condensed=condense_bundle_to_budget(bundle, target_tokens=512),
    )
    assert "(None)" not in enr_low.enrichment_text
    assert "[UUID:" not in enr_low.enrichment_text

    # Temporal digest: clinical anchor (2026-10-18) vs admin (2026-12-01) vs future booked (2027-01-20)
    digest_self = build_temporal_digest(
        extracted,
        condensed,
        question="How is my creatinine changing and when is my next appointment?",
        sharing_posture="patient_self",
    )
    assert "(None)" not in digest_self.digest_text
    assert "[UUID:" not in digest_self.digest_text
    assert digest_self.reference_date == "2026-10-18"
    assert digest_self.summary["last_clinical_encounter_date"] == "2026-10-15"
    assert digest_self.summary["upcoming_appointments_count"] == 1
    assert "Home Blood Pressure Cuff (2026-12-01; NOT a clinical encounter)" in digest_self.digest_text
    assert "Scheduled Nephrology Lab Review (2027-01-20, status: booked)" in digest_self.digest_text
    assert "n=2, increasing +0.55; range 1.1..1.65" in digest_self.digest_text
    assert "n=1, single point — trend not supported" in digest_self.digest_text

    # External share removes HIV 1 RNA (2026-10-18) so reference_date reverts cleanly to 2026-10-15
    digest_ext = build_temporal_digest(
        extracted,
        condensed,
        question="What are my labs?",
        sharing_posture="external_share",
    )
    assert "HIV 1 RNA" not in digest_ext.digest_text
    assert digest_ext.reference_date == "2026-10-15"


@pytest.mark.asyncio
async def test_engine_temporal_digest_strategies_prefix_hash_and_sse_tools() -> None:
    bundle = _temporal_and_clinical_bundle()
    # Remove future encounter to also verify explicit none-documented line in engine output
    bundle["entry"] = [
        e for e in bundle["entry"] if e["resource"].get("id") != "enc-future-booked"
    ]
    captured: List[Dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404)
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        if body.get("stream"):
            msgs = body.get("messages") or []
            has_tool_reply = any(m.get("role") == "tool" for m in msgs)
            if body.get("tools") and not has_tool_reply:
                sse_tc = "".join(
                    [
                        'data: {"id":"tc-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_sse_1","type":"function","function":{"name":"get_lab_history","arguments":"{\\"analyte\\":"}}]},"finish_reason":null}]}\n\n',
                        'data: {"id":"tc-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Creatinine\\",\\"limit\\":5}"}}]},"finish_reason":"tool_calls"}]}\n\n',
                        "data: [DONE]\n\n",
                    ]
                )
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=sse_tc.encode("utf-8"),
                )
            sse_ok = "".join(
                [
                    'data: {"id":"ok-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Creatinine rose from 1.1 to 1.65 mg/dL."},"finish_reason":"stop"}],"usage":{"prompt_tokens":310,"completion_tokens":12}}\n\n',
                    "data: [DONE]\n\n",
                ]
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_ok.encode("utf-8"),
            )
        return httpx.Response(
            200,
            json={
                "id": "cmpl-td-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Creatinine increased to 1.65 mg/dL."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 300, "completion_tokens": 10, "total_tokens": 310},
            },
        )

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

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        with TestClient(app) as client:
            # Preview excludes digest on chart_only and enables it on condensation_kb
            p_chart = client.post(
                "/v1/wahs/context/preview",
                json={
                    "model": "wahs-12b-kb",
                    "messages": [{"role": "user", "content": "How is creatinine?"}],
                    "wahs": {"fhir_bundle": bundle, "retrieval_strategy": "chart_only"},
                },
            )
            assert p_chart.status_code == 200
            assert p_chart.json()["temporal_digest"] == {"enabled": False}
            assert p_chart.json()["x_wahs"]["temporal_digest"] == {"enabled": False}

            p_kb = client.post(
                "/v1/wahs/context/preview",
                json={
                    "model": "wahs-12b-kb",
                    "messages": [{"role": "user", "content": "How is creatinine?"}],
                    "wahs": {"fhir_bundle": bundle, "retrieval_strategy": "condensation_kb"},
                },
            )
            assert p_kb.status_code == 200
            assert p_kb.json()["temporal_digest"]["enabled"] is True
            assert p_kb.json()["temporal_digest"]["multi_point_series_count"] >= 1

            # Sync chart_only strictly excludes digest & KB
            r_chart = client.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "How is my creatinine?"}],
                    "wahs": {
                        "fhir_bundle": bundle,
                        "retrieval_strategy": "chart_only",
                        "target_budget_tokens": 4096,
                    },
                },
            )
            assert r_chart.status_code == 200
            assert r_chart.json()["x_wahs"]["temporal_digest"] == {"enabled": False}
            assert "DETERMINISTIC TEMPORAL DIGEST" not in captured[-1]["messages"][-1]["content"]
            sys_chart_only = captured[-1]["messages"][0]["content"]

            # Sync enriched arms include digest on messages[-1] while keeping messages[0] identical at same budget
            for strat in ("condensation_kb", "note_rag", "dynamic_fhir_tools"):
                r_enr = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-12b-kb",
                        "stream": False,
                        "messages": [{"role": "user", "content": "How is my creatinine?"}],
                        "wahs": {
                            "fhir_bundle": bundle,
                            "retrieval_strategy": strat,
                            "target_budget_tokens": 4096,
                        },
                    },
                )
                assert r_enr.status_code == 200
                td = r_enr.json()["x_wahs"]["temporal_digest"]
                assert td["enabled"] is True
                assert td["reference_date"] == "2026-10-18"
                assert captured[-1]["messages"][0]["content"] == sys_chart_only
                assert "DETERMINISTIC TEMPORAL DIGEST" in captured[-1]["messages"][-1]["content"]
                assert "Upcoming/Scheduled Visits: none documented in chart." in captured[-1]["messages"][-1]["content"]

            # Native SSE stream with Tier-0 overflow tool call accumulation (tight 512 budget)
            r_sse_tools = client.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-12b-kb",
                    "stream": True,
                    "messages": [{"role": "user", "content": "What is my creatinine trajectory?"}],
                    "wahs": {
                        "fhir_bundle": bundle,
                        "retrieval_strategy": "condensation_kb",
                        "target_budget_tokens": 512,
                    },
                },
            )
            assert r_sse_tools.status_code == 200
            chunks = [
                json.loads(line[5:].strip())
                for line in r_sse_tools.text.splitlines()
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
            assert any(
                "Creatinine rose from 1.1 to 1.65 mg/dL." in ((c["choices"][0].get("delta") or {}).get("content") or "")
                for c in chunks
            )
            final_xw = chunks[-1]["x_wahs"]
            assert final_xw["temporal_digest"]["enabled"] is True
            assert final_xw["condensation"]["overflow_recovered"] >= 1
            assert any(tc["name"] == "get_lab_history" for tc in final_xw.get("tool_calls") or [])
