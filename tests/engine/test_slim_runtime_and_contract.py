"""Runtime, wire contract, SSE, and lazy-extension tests for slim `wahs_engine`."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import pytest

from wahs_engine.app import create_app
from wahs_engine.assembler import WAHS_SYSTEM_PROMPT, enforce_single_patient_scope
from wahs_engine.config import WahsEngineSettings


def _bundle_with_two_records() -> Dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-rt-1", "name": [{"given": ["Sam"], "family": "Patel"}], "birthDate": "1970-03-04"}},
            {"resource": {"resourceType": "Condition", "id": "c-1", "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": "2023-05-01", "code": {"text": "Type 2 Diabetes"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "m-1", "status": "active", "authoredOn": "2026-10-01", "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"}}},
        ],
    }


def test_default_runtime_never_imports_wahs_extensions_and_forwards_verbatim_prompt() -> None:
    captured_payloads: List[Dict[str, Any]] = []

    def llm_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        if req.url.path.endswith("/chat/completions"):
            body = json.loads(req.content.decode("utf-8"))
            captured_payloads.append(body)
            if body.get("stream"):
                sse = (
                    'data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<think>hidden</think>Take Metformin [1]."},"finish_reason":null}]}\n\n'
                    'data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":5}}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-2",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "<think>internal</think>Metformin 500 mg is active [1]."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 115, "completion_tokens": 8},
                },
            )
        return httpx.Response(404)

    settings = WahsEngineSettings(
        fhir_auth_base_url="http://fhir-auth:8080/fhir",
        llm_base_url="http://llm-mock:8079/v1",
        allow_local_bundle_mode=True,
    )
    llm_client = httpx.AsyncClient(transport=httpx.MockTransport(llm_handler))
    app = create_app(settings, llm_http_client=llm_client)

    with TestClient(app) as client:
        models_resp = client.get("/v1/models").json()
        assert [m["id"] for m in models_resp["data"]] == ["wahs-26b", "wahs-12b", "wahs-e4b"]

        prev = client.post(
            "/v1/wahs/context/preview",
            json={"model": "wahs-26b", "wahs": {"fhir_bundle": _bundle_with_two_records()}},
        )
        assert prev.status_code == 200
        pdata = prev.json()
        assert pdata["x_wahs"]["enrichers"] == []
        assert pdata["x_wahs"]["condensation"]["tier"] == 0

        exact_prompt = "   What diabetes medication am I taking? Please preserve spacing.   "
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "wahs-26b",
                "stream": False,
                "messages": [{"role": "user", "content": exact_prompt}],
                "wahs": {"fhir_bundle": _bundle_with_two_records(), "sharing_posture": "ignored_extra"},
            },
        )
        assert chat.status_code == 200
        cdata = chat.json()
        assert cdata["choices"][0]["message"]["content"] == "Metformin 500 mg is active [1]."
        assert captured_payloads[-1]["model"] == "ollama/gemma4:26b"
        assert captured_payloads[-1]["messages"][-1]["content"] == exact_prompt
        assert cdata["x_wahs"]["ai_provenance"]["enrichers"] == []

        sse_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "wahs-12b",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
                "wahs": {"fhir_bundle": _bundle_with_two_records()},
            },
        )
        assert sse_resp.status_code == 200
        assert "hidden" not in sse_resp.text
        assert "Take Metformin [1]." in sse_resp.text
        assert "reset_content" not in sse_resp.text

    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    iso = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys, httpx; from fastapi.testclient import TestClient; "
            "from wahs_engine.app import create_app; from wahs_engine.config import WahsEngineSettings; "
            "h = lambda r: httpx.Response(200, json={'object':'list','data':[]}); "
            "app = create_app(WahsEngineSettings(allow_local_bundle_mode=True), llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(h))); "
            "c = TestClient(app); assert c.get('/v1/models').status_code == 200; "
            "b = {'resourceType':'Bundle','type':'collection','entry':[{'resource':{'resourceType':'Patient','id':'p1'}}]}; "
            "assert c.post('/v1/wahs/context/preview', json={'model':'wahs-26b','wahs':{'fhir_bundle':b}}).status_code == 200; "
            "assert 'wahs_extensions' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": src_dir},
        capture_output=True,
        text=True,
        check=False,
    )
    assert iso.returncode == 0, iso.stderr
    assert "CONVERSATIONAL PRIVACY" not in WAHS_SYSTEM_PROMPT
    assert "sharing_posture" not in WAHS_SYSTEM_PROMPT


def test_security_invariants_and_single_patient_scope() -> None:
    with pytest.raises(ValueError, match="fhir-internal"):
        WahsEngineSettings(fhir_auth_base_url="http://fhir-internal:8080/fhir")

    mixed_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-a"}},
            {"resource": {"resourceType": "Patient", "id": "pat-foreign"}},
            {"resource": {"resourceType": "Observation", "id": "obs-ok", "subject": {"reference": "Patient/pat-a"}}},
            {"resource": {"resourceType": "Observation", "id": "obs-bad", "subject": {"reference": "Patient/pat-foreign"}}},
        ],
    }
    guarded, audit = enforce_single_patient_scope(mixed_bundle, authorized_patient_id="pat-a", allowed_patient_refs={"pat-a"})
    assert audit["stripped_foreign_count"] == 2
    assert len(guarded["entry"]) == 2


def test_wahs_devtools_cli_help() -> None:
    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "wahs_devtools.seeder", "--help"],
        env={**os.environ, "PYTHONPATH": src_dir},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Seed multi-source WAHS FHIR R4 star-graph personas into HAPI / fhir-auth." in proc.stdout
