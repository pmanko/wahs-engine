"""Comprehensive tests for exact offline tokenization, monotonic 2-stage condensation,
model-derived sizing, southbound parameter contract, and freshness/partial-fetch hygiene."""

from __future__ import annotations

import asyncio
import base64
import copy
import glob
import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest

from wahs_core.condenser import condense_bundle_to_budget
from wahs_engine.app import create_app
from wahs_engine.assembler import (
    CATEGORY_SYSTEM,
    IDENTIFIER_SYSTEM,
    METHOD_SYSTEM,
    SECURITY_SYSTEM,
    SOURCE_NAME_SYSTEM,
    WAHSBundleAssembler,
    _parse_fhir_instant,
)
from wahs_engine.config import ModelSpec, WahsEngineSettings
from wahs_engine.jwt_claims import decode_and_bind_jwt
from wahs_engine.schemas import WahsRequestOptions
from wahs_engine.tokenizers_registry import (
    count_chat_messages_tokens,
    get_exact_token_counter,
)

TEST_SECRET = "test-sizing-secret-key"


def _make_jwt(sub: str, secret: str = TEST_SECRET, exp_offset: int = 3600) -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": sub, "email": f"{sub}@example.org", "exp": int(time.time()) + exp_offset}).encode()
    ).rstrip(b"=").decode()
    sig_input = f"{header}.{payload}".encode("ascii")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


def _build_rich_synthetic_bundle(num_notes: int = 8, num_labs: int = 60) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-mono-1",
                "name": [{"given": ["Taylor"], "family": "Brooks"}],
                "birthDate": "1965-09-14",
                "gender": "female",
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-htn-1",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "onsetDateTime": "2020-01-10",
                "code": {"text": "Essential hypertension"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-lis-1",
                "status": "active",
                "authoredOn": "2024-05-01",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "314076",
                            "display": "Lisinopril 10 MG Oral Tablet",
                        }
                    ]
                },
            }
        },
    ]
    for i in range(num_labs):
        yr = 2021 + (i // 12)
        mo = 1 + (i % 12)
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-creat-{i}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"{yr:04d}-{mo:02d}-{(1 + (i % 25)):02d}",
                    "code": {
                        "coding": [
                            {"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}
                        ]
                    },
                    "valueQuantity": {"value": round(1.0 + 0.01 * (i % 15), 2), "unit": "mg/dL"},
                }
            }
        )
    for i in range(num_notes):
        body_text = (
            f"Clinical Progress Note #{i}: Patient discussed home blood pressure log and medication adherence. "
            + ("Reviewed renal function trends, electrolytes, dietary sodium restriction, and follow-up schedule. " * 25)
        )
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-note-{i}",
                    "status": "current",
                    "date": f"2025-0{1 + (i % 9)}-15T10:00:00Z",
                    "type": {"text": f"Primary Care Visit Note {i}"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(body_text.encode("utf-8")).decode("ascii"),
                            }
                        }
                    ],
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


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


def test_offline_tokenizers_and_monotonic_synthetic_condenser() -> None:
    gemma_cnt = get_exact_token_counter("gemma4")
    o200k_cnt = get_exact_token_counter("o200k_base")
    sample = "Lisinopril 10 MG Oral Tablet (2026-10-15)"
    assert gemma_cnt(sample) > 0
    assert o200k_cnt(sample) > 0
    # Chat framing matches calibrated `8 + 5 * len(messages)` formula
    msgs = [
        {"role": "system", "content": sample},
        {"role": "user", "content": "Summarize medications"},
    ]
    assert count_chat_messages_tokens(msgs, tokenizer_family="gemma4") == (
        gemma_cnt(sample) + gemma_cnt("Summarize medications") + 8 + 5 * 2
    )

    bundle = _build_rich_synthetic_bundle()
    prev_uuids: set[str] = set()
    for budget in [1024, 2048, 2800, 4096, 8192, 16384, 32768]:
        res = condense_bundle_to_budget(bundle, target_tokens=budget, token_counter=gemma_cnt)
        assert res.actual_tokens <= budget
        cur = _clinical_uuids(res)
        assert not (prev_uuids - cur), f"Dropped clinical UUIDs at budget={budget}: {prev_uuids - cur}"
        prev_uuids = cur


def test_all_hq_fixture_bundles_strict_monotonicity_across_262k() -> None:
    bundle_paths = sorted(glob.glob("/Users/pmanko/code/hq/wahs/bundles/*.json"))
    if not bundle_paths:
        pytest.skip("Fixture bundles under /Users/pmanko/code/hq/wahs/bundles not present in environment")
    gemma_cnt = get_exact_token_counter("gemma4")
    budgets = [1024, 2048, 2800, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    for path in bundle_paths:
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        prev_uuids: set[str] = set()
        for b in budgets:
            res = condense_bundle_to_budget(bundle, target_tokens=b, token_counter=gemma_cnt)
            assert res.actual_tokens <= b
            cur = _clinical_uuids(res)
            assert not (prev_uuids - cur), f"{Path(path).name} dropped UUIDs at {b}: {prev_uuids - cur}"
            prev_uuids = cur
        assert res.completeness_manifest["omitted_clinical_items_count"] == 0
        assert res.completeness_manifest["excerpted_notes_count"] == 0


@pytest.mark.asyncio
async def test_model_derived_sizing_at_most_clamp_and_history_trim_and_422() -> None:
    bundle = _build_rich_synthetic_bundle(num_notes=6, num_labs=40)
    captured_payloads: List[Dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(payload)
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-test-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Length capped answer [1]."},
                            "finish_reason": "length",
                        }
                    ],
                    # Simulate severe silent upstream truncation (100 reported vs >2000 sent)
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
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
            ),
            "frontier-1m": ModelSpec(
                upstream_model="gpt-5-frontier",
                context_window=1048576,
                chart_budget_ceiling=None,
                tokenizer_family="o200k_base",
                provider_flavor="openai",
                disable_thinking=False,
            ),
            "tiny-ctx": ModelSpec(
                upstream_model="tiny-local",
                context_window=3072,
                chart_budget_ceiling=2048,
                default_max_output_tokens=256,
                tokenizer_family="gemma4",
                provider_flavor="llamacpp",
                disable_thinking=True,
            ),
        },
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        with TestClient(app) as client:
            # 1. Omit target_budget_tokens on 32k model -> derived chart_budget_ceiling=16384; preview + chat share prefix hash
            prev_resp = client.post(
                "/v1/wahs/context/preview",
                json={"model": "wahs-12b-kb", "wahs": {"fhir_bundle": bundle}},
            )
            assert prev_resp.status_code == 200
            prev_cond = prev_resp.json()["x_wahs"]["condensation"]
            assert prev_cond["budget_effective"] == 16384
            assert prev_cond["budget_requested"] is None
            assert prev_cond["context_window"] == 32768
            assert prev_cond["chart_budget_ceiling"] == 16384
            assert prev_cond["clamp_reason"] == "chart_budget_ceiling"

            # 2. Requesting 200,000 on 32k model clamps to 16,384 (same prefix_cache_key_hash -> cache_hit!)
            chat_resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What is my creatinine?"}],
                    "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 200000},
                },
            )
            assert chat_resp.status_code == 200
            c_body = chat_resp.json()
            assert c_body["choices"][0]["finish_reason"] == "length"
            assert c_body["x_wahs"]["condensation"]["cache_hit"] is True
            assert c_body["x_wahs"]["condensation"]["budget_effective"] == 16384
            assert c_body["x_wahs"]["condensation"]["clamp_reason"] == "clamped_to_chart_budget_ceiling"
            assert c_body["x_wahs"]["condensation"]["suspected_upstream_truncation"] is True
            assert captured_payloads[-1]["num_ctx"] == 32768
            assert captured_payloads[-1]["reasoning_effort"] == "none"

            # 3. 1M frontier model with ceiling=None unlocks > 131,072 derived budget & omits num_ctx for openai flavor
            prev_1m = client.post(
                "/v1/wahs/context/preview",
                json={"model": "frontier-1m", "wahs": {"fhir_bundle": bundle}},
            )
            assert prev_1m.status_code == 200
            assert prev_1m.json()["x_wahs"]["condensation"]["budget_effective"] > 131072
            assert prev_1m.json()["x_wahs"]["condensation"]["clamp_reason"] == "context_window_derived"
            chat_1m = client.post(
                "/v1/chat/completions",
                json={
                    "model": "frontier-1m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Check history."}],
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert chat_1m.status_code == 200
            assert "num_ctx" not in captured_payloads[-1]
            assert "chat_template_kwargs" not in captured_payloads[-1]

            # 4. Multi-turn history trimming on tiny-ctx keeps messages[0] byte-identical
            long_turns = []
            for idx in range(6):
                long_turns.append({"role": "user", "content": f"Turn {idx} question " + ("details " * 250)})
                long_turns.append({"role": "assistant", "content": f"Turn {idx} answer " + ("notes " * 250)})
            long_turns.append({"role": "user", "content": "Final question?"})
            prev_tiny_resp = client.post(
                "/v1/wahs/context/preview",
                json={"model": "tiny-ctx", "wahs": {"fhir_bundle": bundle}},
            )
            assert prev_tiny_resp.status_code == 200, prev_tiny_resp.text
            sys_before = prev_tiny_resp.json()["chart_text"]
            trim_resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny-ctx",
                    "stream": False,
                    "max_tokens": 256,
                    "messages": long_turns,
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert trim_resp.status_code == 200
            t_cond = trim_resp.json()["x_wahs"]["condensation"]
            assert t_cond["history_trimmed_turns"] >= 2
            assert sys_before in captured_payloads[-1]["messages"][0]["content"]
            assert captured_payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False}

            # 5. Un-fittable single user turn overflows context_window -> HTTP 422
            huge_msg = "question " * 4000
            overflow_resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny-ctx",
                    "stream": False,
                    "messages": [{"role": "user", "content": huge_msg}],
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert overflow_resp.status_code == 422


@pytest.mark.asyncio
async def test_no_http_400_retry_mutation_and_sse_finish_reason_passthrough() -> None:
    bundle = _build_rich_synthetic_bundle(num_notes=2, num_labs=30)
    calls: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content.decode("utf-8"))
            calls.append(body)
            if body.get("stream"):
                sse = "".join(
                    [
                        'data: {"id":"s-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Partial SSE"},"finish_reason":null}]}\n\n',
                        'data: {"id":"s-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":350,"completion_tokens":3}}\n\n',
                        "data: [DONE]\n\n",
                    ]
                )
                return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
            return httpx.Response(400, json={"error": {"message": "Bad param"}})
        return httpx.Response(404)

    settings = WahsEngineSettings(
        llm_base_url="http://llm-backend:4000/v1",
        allow_local_bundle_mode=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        with pytest.raises(httpx.HTTPStatusError):
            # Sync call returning 400 MUST NOT retry with mutated payload
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
                await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-12b-kb",
                        "stream": False,
                        "messages": [{"role": "user", "content": "Latest labs?"}],
                        "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512},
                    },
                )
        assert len(calls) == 1

        # SSE stream passes upstream finish_reason="length" through instead of hardcoding "stop"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            sse_resp = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-12b-kb",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert sse_resp.status_code == 200
            events = [
                json.loads(line[5:].strip())
                for line in sse_resp.text.splitlines()
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
            assert events[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_partial_fetch_detection_missing_timestamp_ttl_and_instant_sort() -> None:
    class FreshnessMockServer:
        def __init__(self) -> None:
            self.token = _make_jwt("user-fresh")
            self.everything_calls = 0
            self.omit_story_timestamp = False

        def handler(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            query = request.url.query.decode()
            if path == "/fhir/Patient":
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "id": "master-f",
                                    "meta": {
                                        "lastUpdated": "2026-11-10T18:00:00Z",
                                        "security": [{"system": SECURITY_SYSTEM, "code": "user-fresh"}],
                                    },
                                    "name": [{"given": ["Fresh"], "family": "Tester"}],
                                    "link": (
                                        [{"other": {"reference": "Patient/ds-tz-new"}, "type": "seealso"}]
                                        if self.omit_story_timestamp
                                        else [
                                            {"other": {"reference": "Patient/ds-tz-old"}, "type": "seealso"},
                                            {"other": {"reference": "Patient/ds-tz-new"}, "type": "seealso"},
                                        ]
                                    ),
                                }
                            }
                        ],
                    },
                )
            if path == "/fhir/Patient/ds-tz-old":
                # 12:00 UTC
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Patient",
                        "id": "ds-tz-old",
                        "meta": {
                            "lastUpdated": "2026-11-10T12:00:00Z",
                            "source": "urn:wahealthsummary:patient-story",
                            "tag": [
                                {"system": CATEGORY_SYSTEM, "code": "patient-story"},
                                {"system": METHOD_SYSTEM, "code": "patient-story-form"},
                                {"system": SOURCE_NAME_SYSTEM, "code": "Patient Story"},
                            ],
                        },
                    },
                )
            if path == "/fhir/Patient/ds-tz-new":
                # 10:00-08:00 == 18:00Z (MUST beat 12:00Z even though '10' < '12' lexicographically!)
                meta: Dict[str, Any] = {
                    "source": "urn:wahealthsummary:patient-story",
                    "tag": [
                        {"system": CATEGORY_SYSTEM, "code": "patient-story"},
                        {"system": METHOD_SYSTEM, "code": "patient-story-form"},
                        {"system": SOURCE_NAME_SYSTEM, "code": "Patient Story"},
                    ],
                }
                if not self.omit_story_timestamp:
                    meta["lastUpdated"] = "2026-11-10T10:00:00-08:00"
                return httpx.Response(200, json={"resourceType": "Patient", "id": "ds-tz-new", "meta": meta})
            if path == "/fhir/Patient/ds-tz-new/$everything":
                self.everything_calls += 1
                # Simulated jwt-proxy overwrite: Bundle.total == len(entry) == 1 even though page 2 exists!
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "total": 1,
                        "link": [{"relation": "next", "url": "http://fhir-internal:8080/fhir?_getpages=p2"}],
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Goal",
                                    "id": "goal-tz-new",
                                    "lifecycleStatus": "active",
                                    "startDate": "2026-11-10",
                                    "subject": {"reference": "Patient/ds-tz-new"},
                                    "description": {"text": "Newest UTC goal wins"},
                                }
                            }
                        ],
                    },
                )
            if path in {"/fhir/", "/fhir"} and "_getpages=p2" in query:
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "total": 1,
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Observation",
                                    "id": "obs-p2",
                                    "status": "final",
                                    "effectiveDateTime": "2026-11-10",
                                    "subject": {"reference": "Patient/ds-tz-new"},
                                    "code": {"coding": [{"system": "http://loinc.org", "code": "51855-5", "display": "Patient Note"}]},
                                    "valueString": "Second page clinical note",
                                }
                            }
                        ],
                    },
                )
            return httpx.Response(404)

    # UTC-offset instant helper check
    assert _parse_fhir_instant("2026-11-10T10:00:00-08:00") > _parse_fhir_instant("2026-11-10T12:00:00Z")

    srv = FreshnessMockServer()
    async with httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)) as client:
        # 1. max_everything_pages=1 -> truncated via pending next link; uncached; external_share -> 409
        settings_partial = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            jwt_hs256_secret=TEST_SECRET,
            max_everything_pages=1,
        )
        asm_partial = WAHSBundleAssembler(settings_partial, http_client=client)
        ident = decode_and_bind_jwt(srv.token, hs256_secret=TEST_SECRET)
        ctx_p = await asm_partial.assemble(options=WahsRequestOptions(), identity=ident)
        assert ctx_p.datasets_included[0]["dataset_id"] == "ds-tz-new"
        assert ctx_p.partial_fetch is True
        assert ctx_p.condensed.completeness_manifest["partial_fetch"] is True
        assert "[PARTIAL RECORD WARNING:" in ctx_p.condensed.chart_text
        assert len(asm_partial._dataset_cache) == 0
        assert len(asm_partial._prefix_cache) == 0

        with pytest.raises(HTTPException) as exc_409:
            await asm_partial.assemble(
                options=WahsRequestOptions(sharing_posture="external_share"),
                identity=ident,
            )
        assert exc_409.value.status_code == 409

        ctx_ext_ok = await asm_partial.assemble(
            options=WahsRequestOptions(
                sharing_posture="external_share",
                allow_partial_external_share=True,
            ),
            identity=ident,
        )
        assert ctx_ext_ok.partial_fetch is True

        # 2. Missing meta.lastUpdated / versionId never caches as `"v1"`
        srv.everything_calls = 0
        srv.omit_story_timestamp = True
        settings_full = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            jwt_hs256_secret=TEST_SECRET,
            max_everything_pages=10,
            cache_ttl_seconds=0.05,
        )
        asm_full = WAHSBundleAssembler(settings_full, http_client=client)
        await asm_full.assemble(options=WahsRequestOptions(), identity=ident)
        await asm_full.assemble(options=WahsRequestOptions(), identity=ident)
        assert srv.everything_calls == 2  # Never reused unstamped cache entry!

        # 3. With timestamp present: TTL expiration & wahs.refresh=True both force re-fetch
        srv.everything_calls = 0
        srv.omit_story_timestamp = False
        c1 = await asm_full.assemble(options=WahsRequestOptions(), identity=ident)
        c2 = await asm_full.assemble(options=WahsRequestOptions(), identity=ident)
        assert c1.cache_hit is False
        assert c2.cache_hit is True
        assert srv.everything_calls == 1

        c_ref = await asm_full.assemble(options=WahsRequestOptions(refresh=True), identity=ident)
        assert c_ref.cache_hit is False
        assert srv.everything_calls == 2

        await asyncio.sleep(0.08)
        c_ttl = await asm_full.assemble(options=WahsRequestOptions(), identity=ident)
        assert c_ttl.cache_hit is False
        assert srv.everything_calls == 3
