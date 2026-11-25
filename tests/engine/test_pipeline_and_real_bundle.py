"""Opt-in `EnricherPipeline` and multi-decade chart verification."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import pytest

from wahs_core import condense_bundle_to_budget
from wahs_engine.app import create_app
from wahs_engine.config import WahsEngineSettings


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_operator_enabled_enrichers_wire_into_x_wahs_and_ai_provenance() -> None:
    captured: List[Dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/chat/completions"):
            body = json.loads(req.content.decode("utf-8"))
            captured.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-ext",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Checked [1]."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 3},
                },
            )
        return httpx.Response(404)

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-opt", "name": [{"given": ["Dana"], "family": "Kim"}]}},
            {"resource": {"resourceType": "Observation", "id": "o1", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2024-01-01", "code": {"text": "Creatinine"}, "valueQuantity": {"value": 1.0, "unit": "mg/dL"}}},
            {"resource": {"resourceType": "Observation", "id": "o2", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2025-01-01", "code": {"text": "Creatinine"}, "valueQuantity": {"value": 1.6, "unit": "mg/dL"}}},
            {"resource": {"resourceType": "DocumentReference", "id": "d1", "status": "current", "date": "2025-01-02", "type": {"text": "Cardiology Note"}, "content": [{"attachment": {"data": _b64("Cardiology follow-up note discussing daily blood pressure and fluid status. " * 120 + "Advise holding naproxen due to rising creatinine.")}}]}},
        ],
    }

    settings = WahsEngineSettings(
        fhir_auth_base_url="http://fhir-auth:8080/fhir",
        llm_base_url="http://llm-mock:8079/v1",
        allow_local_bundle_mode=True,
        enabled_enrichers=("note_rag", "temporal_trends"),
    )
    app = create_app(settings, llm_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "wahs-26b",
                "messages": [{"role": "user", "content": "What is my creatinine trend and cardiology naproxen advice?"}],
                "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512},
            },
        )
        assert resp.status_code == 200
        xw = resp.json()["x_wahs"]
        ids = {e["enricher_id"] for e in xw["enrichers"]}
        assert "temporal_trends" in ids
        assert "note_rag" in ids
        assert xw["ai_provenance"]["enrichers"] == xw["enrichers"]
        assert "LONGITUDINAL QUANTITATIVE TRENDS" in captured[-1]["messages"][-1]["content"]


MAURICE_PATH = Path("/Users/pmanko/code/hq/wahs/bundles/Maurice742_Auer97.json")


@pytest.mark.skipif(not MAURICE_PATH.is_file(), reason="Local Maurice742_Auer97.json fixture not present")
def test_maurice_auer_multidecade_monotonic_condensation() -> None:
    bundle = json.loads(MAURICE_PATH.read_text(encoding="utf-8"))
    prev_cov: set[str] | None = None
    for budget in (2048, 4096, 8192, 16384):
        res = condense_bundle_to_budget(bundle, target_tokens=budget)
        assert res.actual_tokens <= budget
        assert res.completeness_manifest["tier"] in {0, 1, 2}
        cov = {
            str(u)
            for m in res.mappings
            for u in ([m.get("resourceUuid")] + list(m.get("memberUuids") or []))
            if u
        }
        if prev_cov is not None:
            assert prev_cov.issubset(cov)
        prev_cov = cov
