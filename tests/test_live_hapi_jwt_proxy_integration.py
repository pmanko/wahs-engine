"""Live integration test against real `HAPI FHIR v8.0.0` + `uwcirg/jwt-proxy`.

Gated on `WAHS_LIVE_FHIR_AUTH_URL` and `WAHS_LIVE_HAPI_INTERNAL_URL`.
Tests real `RS256` JWKS verification via `jwt-proxy`, first-page `_count=1000`
page-size behavior, multi-page `$everything` pagination across `jwt-proxy`
(`05_allow_patient_summary.py` + `51_fhir_response_security.py`), `max_everything_pages=1`
partial-fetch detection, dataset replacement (`ds-v1 -> ds-v2`), in-place write/delete
inside an existing dataset (`wahs.refresh=True`), and unchanged MasterPatient re-save.
"""

from __future__ import annotations

import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid

import httpx
import pytest

from wahs_engine.app import create_app
from wahs_engine.assembler import (
    CATEGORY_SYSTEM,
    IDENTIFIER_SYSTEM,
    METHOD_SYSTEM,
    SECURITY_SYSTEM,
    SOURCE_NAME_SYSTEM,
    WAHSBundleAssembler,
)
from wahs_engine.config import ModelSpec, WahsEngineSettings
from wahs_engine.jwt_claims import decode_and_bind_jwt
from wahs_engine.schemas import WahsRequestOptions
from wahs_engine.tokenizers_registry import count_chat_messages_tokens

LIVE_FHIR_AUTH_URL = os.environ.get("WAHS_LIVE_FHIR_AUTH_URL", "").strip()
LIVE_HAPI_INTERNAL_URL = os.environ.get("WAHS_LIVE_HAPI_INTERNAL_URL", "").strip()
LIVE_JWKS_PORT = int(os.environ.get("WAHS_LIVE_JWKS_PORT", "18991").strip() or "18991")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_rsa_jwks_and_signer():
    """Generate an ephemeral 2048-bit RSA keypair and JWKS payload using stdlib + openssl."""
    tmpdir = tempfile.TemporaryDirectory(prefix="wahs-live-jwks-")
    priv_path = os.path.join(tmpdir.name, "priv.pem")
    subprocess.run(
        ["openssl", "genrsa", "-out", priv_path, "2048"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    mod_out = subprocess.check_output(
        ["openssl", "rsa", "-in", priv_path, "-modulus", "-noout"],
        text=True,
    ).strip()
    assert mod_out.startswith("Modulus=")
    n_bytes = bytes.fromhex(mod_out.split("=", 1)[1].strip())
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "wahs-live-k1",
                "use": "sig",
                "alg": "RS256",
                "n": _b64url(n_bytes),
                "e": "AQAB",
            }
        ]
    }

    def sign_jwt(sub: str) -> str:
        header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT", "kid": "wahs-live-k1"}).encode())
        payload = _b64url(
            json.dumps(
                {
                    "sub": sub,
                    "email": f"{sub}@example.org",
                    "iat": int(time.time()) - 5,
                    "exp": int(time.time()) + 3600,
                }
            ).encode()
        )
        signing_input = f"{header}.{payload}".encode("ascii")
        sig = subprocess.check_output(
            ["openssl", "dgst", "-sha256", "-sign", priv_path],
            input=signing_input,
        )
        return f"{header}.{payload}.{_b64url(sig)}"

    return tmpdir, jwks, sign_jwt


def _start_jwks_server(jwks: dict, port: int):
    body = json.dumps(jwks).encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture(scope="module")
def live_jwks_signer():
    """Share one ephemeral RS256 keypair and JWKS endpoint across live integration tests."""
    tmpdir, jwks, sign_jwt = _generate_rsa_jwks_and_signer()
    jwks_server = _start_jwks_server(jwks, LIVE_JWKS_PORT)
    try:
        yield sign_jwt
    finally:
        jwks_server.shutdown()
        jwks_server.server_close()
        tmpdir.cleanup()


def _make_patient_resource(
    pid: str,
    *,
    sub: str,
    is_master: bool = False,
    links: list[str] | None = None,
    given: str = "Live",
    category: str = "patient-story",
    method: str = "patient-story-form",
    source_name: str = "Patient Story",
) -> dict:
    res: dict = {
        "resourceType": "Patient",
        "id": pid,
        "meta": {
            "security": [{"system": SECURITY_SYSTEM, "code": sub}],
        },
        "name": [{"given": [given], "family": "Tester"}],
        "birthDate": "1975-05-05",
    }
    if is_master:
        res["identifier"] = [{"system": IDENTIFIER_SYSTEM, "value": sub}]
        res["link"] = [
            {"other": {"reference": f"Patient/{lid}"}, "type": "seealso"}
            for lid in (links or [])
        ]
    else:
        res["meta"]["tag"] = [
            {"system": CATEGORY_SYSTEM, "code": category},
            {"system": METHOD_SYSTEM, "code": method},
            {"system": SOURCE_NAME_SYSTEM, "code": source_name},
        ]
    return res


@pytest.mark.skipif(
    not (LIVE_FHIR_AUTH_URL and LIVE_HAPI_INTERNAL_URL),
    reason="WAHS_LIVE_FHIR_AUTH_URL and WAHS_LIVE_HAPI_INTERNAL_URL not set",
)
@pytest.mark.asyncio
async def test_live_hapi_and_jwt_proxy_pagination_and_freshness(live_jwks_signer) -> None:
    sub = f"live-user-{uuid.uuid4().hex[:8]}"
    master_id = f"master-{sub}"
    ds_v1_id = f"ds-story-v1-{uuid.uuid4().hex[:6]}"
    ds_v2_id = f"ds-story-v2-{uuid.uuid4().hex[:6]}"
    token = live_jwks_signer(sub)
    # Empty hs256_secret mirrors production (`jwt-proxy` verifies RS256 against JWKS_URL)
    ident = decode_and_bind_jwt(token, hs256_secret="")

    async with httpx.AsyncClient(timeout=30.0) as hapi:
        # 1. Seed ds-story-v1 with 250 Observations so >200 threshold and pagination can both be measured
        for res in [
            _make_patient_resource(ds_v1_id, sub=sub),
            _make_patient_resource(ds_v2_id, sub=sub),
            _make_patient_resource(master_id, sub=sub, is_master=True, links=[ds_v1_id]),
        ]:
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{res['id']}", json=res)
            assert r.status_code in (200, 201), r.text

        sem = asyncio.Semaphore(16)

        async def _put_obs(idx: int) -> None:
            obs = {
                "resourceType": "Observation",
                "id": f"obs-live-{ds_v1_id}-{idx:03d}",
                "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
                "status": "final",
                "effectiveDateTime": "2026-11-01",
                "subject": {"reference": f"Patient/{ds_v1_id}"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
                "valueQuantity": {"value": 70 + idx * 0.1, "unit": "kg"},
            }
            async with sem:
                resp = await hapi.put(
                    f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Observation/{obs['id']}",
                    json=obs,
                )
                assert resp.status_code in (200, 201), resp.text

        await asyncio.gather(*(_put_obs(i) for i in range(250)))

        goal_v2 = {
            "resourceType": "Goal",
            "id": f"goal-live-{ds_v2_id}",
            "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
            "lifecycleStatus": "active",
            "startDate": "2026-11-11",
            "subject": {"reference": f"Patient/{ds_v2_id}"},
            "description": {"text": "Walk 35 minutes daily after dataset replacement"},
        }
        r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Goal/{goal_v2['id']}", json=goal_v2)
        assert r.status_code in (200, 201), r.text

    async with httpx.AsyncClient(timeout=30.0) as auth_client:
        # 2a. Probe actual first-page size HAPI + jwt-proxy returns when `_count=1000` is requested
        probe_resp = await auth_client.get(
            f"{LIVE_FHIR_AUTH_URL.rstrip('/')}/Patient/{ds_v1_id}/$everything",
            params={"_count": "1000"},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
        )
        assert probe_resp.status_code == 200, probe_resp.text
        probe_bundle = probe_resp.json()
        first_page_entries_for_1000 = len(probe_bundle.get("entry") or [])
        probe_next_links = [
            str(l.get("url") or "")
            for l in (probe_bundle.get("link") or [])
            if isinstance(l, dict) and l.get("relation") == "next"
        ]

        # 2b. Verify multi-page `$everything` through jwt-proxy (`_count=50` -> 6 pages, Page 2+ via 51_fhir_response_security.py)
        settings_paginated = WahsEngineSettings(
            fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
            everything_page_size=50,
            max_everything_pages=10,
        )
        asm = WAHSBundleAssembler(settings_paginated, http_client=auth_client)
        ctx_full = await asm.assemble(options=WahsRequestOptions(), identity=ident)
        assert ctx_full.partial_fetch is False
        assert ctx_full.extracted.total_bundle_entries >= 250

        # 3. Verify `max_everything_pages=1` with `_count=50` triggers `partial_fetch=True` cleanly!
        settings_capped = WahsEngineSettings(
            fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
            everything_page_size=50,
            max_everything_pages=1,
        )
        asm_capped = WAHSBundleAssembler(settings_capped, http_client=auth_client)
        ctx_part = await asm_capped.assemble(options=WahsRequestOptions(), identity=ident)
        assert ctx_part.partial_fetch is True
        assert ctx_part.partial_datasets[0]["reason"] == "max_everything_pages"
        assert "[PARTIAL RECORD WARNING:" in ctx_part.condensed.chart_text
        assert len(asm_capped._dataset_cache) == 0

        # 4. Dataset replacement (`ds_v1 -> ds_v2` via MasterPatient update) automatically invalidates cache
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            updated_master = _make_patient_resource(master_id, sub=sub, is_master=True, links=[ds_v2_id])
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{master_id}", json=updated_master)
            assert r.status_code in (200, 201), r.text

        ctx_replaced = await asm.assemble(options=WahsRequestOptions(), identity=ident)
        assert ctx_replaced.cache_hit is False
        assert "Walk 35 minutes daily after dataset replacement" in ctx_replaced.condensed.chart_text

        # 5. Unchanged re-save of MasterPatient: measure whether HAPI moves `meta.lastUpdated`
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            master_before = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{master_id}")).json()
            await asyncio.sleep(0.05)
            r_same = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{master_id}", json=updated_master)
            assert r_same.status_code in (200, 201), r_same.text
            master_after = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{master_id}")).json()
        unchanged_master_put_moved_last_updated = (
            master_before.get("meta", {}).get("lastUpdated")
            != master_after.get("meta", {}).get("lastUpdated")
        )

        # 6. In-place Goal write & delete inside `ds_v2` WITHOUT touching `Patient/ds_v2`:
        #    record whether HAPI moved Patient/ds_v2 `meta.lastUpdated`, and prove `refresh=True` sees both!
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            before_pat = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{ds_v2_id}")).json()
            in_place_id = f"goal-inplace-{uuid.uuid4().hex[:6]}"
            in_place_goal = {
                "resourceType": "Goal",
                "id": in_place_id,
                "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
                "lifecycleStatus": "active",
                "startDate": "2026-11-12",
                "subject": {"reference": f"Patient/{ds_v2_id}"},
                "description": {"text": "In-place appended hydration goal"},
            }
            await asyncio.sleep(0.05)
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Goal/{in_place_id}", json=in_place_goal)
            assert r.status_code in (200, 201), r.text
            after_write_pat = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{ds_v2_id}")).json()

        inplace_write_moved_patient_last_updated = (
            before_pat.get("meta", {}).get("lastUpdated")
            != after_write_pat.get("meta", {}).get("lastUpdated")
        )

        ctx_refreshed = await asm.assemble(options=WahsRequestOptions(refresh=True), identity=ident)
        assert ctx_refreshed.cache_hit is False
        assert "In-place appended hydration goal" in ctx_refreshed.condensed.chart_text

        # Delete that Goal inside ds_v2 and confirm refresh=True removes it
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            await asyncio.sleep(0.05)
            r_del = await hapi.delete(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Goal/{in_place_id}")
            assert r_del.status_code in (200, 204), r_del.text
            after_del_pat = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{ds_v2_id}")).json()

        inplace_delete_moved_patient_last_updated = (
            after_write_pat.get("meta", {}).get("lastUpdated")
            != after_del_pat.get("meta", {}).get("lastUpdated")
        )

        ctx_after_del = await asm.assemble(options=WahsRequestOptions(refresh=True), identity=ident)
        assert ctx_after_del.cache_hit is False
        assert "In-place appended hydration goal" not in ctx_after_del.condensed.chart_text

        telemetry = {
            "seeded_observations": 250,
            "first_page_entries_for_count_1000": first_page_entries_for_1000,
            "probe_next_link_present": bool(probe_next_links),
            "paginated_total_entries_at_count_50": ctx_full.extracted.total_bundle_entries,
            "unchanged_master_put_moved_last_updated": unchanged_master_put_moved_last_updated,
            "inplace_write_moved_patient_last_updated": inplace_write_moved_patient_last_updated,
            "inplace_delete_moved_patient_last_updated": inplace_delete_moved_patient_last_updated,
        }
        print("LIVE_HAPI_JWT_PROXY_TELEMETRY=" + json.dumps(telemetry, sort_keys=True))


@pytest.mark.skipif(
    not (LIVE_FHIR_AUTH_URL and LIVE_HAPI_INTERNAL_URL),
    reason="WAHS_LIVE_FHIR_AUTH_URL and WAHS_LIVE_HAPI_INTERNAL_URL not set",
)
@pytest.mark.asyncio
async def test_live_http_service_preview_sse_cache_and_partial_409_over_jwt_proxy(
    live_jwks_signer,
) -> None:
    """Exercise full FastAPI HTTP/SSE `/v1/wahs/context/preview` & `/v1/chat/completions` against HAPI + jwt-proxy."""
    sub = f"svc-user-{uuid.uuid4().hex[:8]}"
    master_id = f"master-{sub}"
    ds_clinic_id = f"ds-clinic-{uuid.uuid4().hex[:6]}"
    ds_story_v1_id = f"ds-story-v1-{uuid.uuid4().hex[:6]}"
    ds_story_v2_id = f"ds-story-v2-{uuid.uuid4().hex[:6]}"
    token = live_jwks_signer(sub)

    async with httpx.AsyncClient(timeout=30.0) as hapi:
        for res in [
            _make_patient_resource(
                ds_clinic_id,
                sub=sub,
                category="provider-health-record",
                method="provider-health-record-sof",
                source_name="UW Medicine",
            ),
            _make_patient_resource(ds_story_v1_id, sub=sub),
            _make_patient_resource(ds_story_v2_id, sub=sub),
            _make_patient_resource(master_id, sub=sub, is_master=True, links=[ds_clinic_id, ds_story_v1_id]),
        ]:
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{res['id']}", json=res)
            assert r.status_code in (200, 201), r.text

        warfarin = {
            "resourceType": "MedicationRequest",
            "id": f"med-warf-{ds_clinic_id}",
            "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
            "status": "active",
            "intent": "order",
            "authoredOn": "2026-10-10",
            "subject": {"reference": f"Patient/{ds_clinic_id}"},
            "medicationCodeableConcept": {"text": "Warfarin sodium 5 MG Oral Tablet"},
        }
        story_obs_v1 = {
            "resourceType": "Observation",
            "id": f"obs-story-v1-{ds_story_v1_id}",
            "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
            "status": "final",
            "effectiveDateTime": "2026-11-10",
            "subject": {"reference": f"Patient/{ds_story_v1_id}"},
            "code": {"coding": [{"system": "http://loinc.org", "code": "51855-5", "display": "Patient Note"}]},
            "valueString": "Taking OTC ibuprofen for knee soreness",
        }
        story_goal_v2 = {
            "resourceType": "Goal",
            "id": f"goal-story-v2-{ds_story_v2_id}",
            "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
            "lifecycleStatus": "active",
            "startDate": "2026-11-12",
            "subject": {"reference": f"Patient/{ds_story_v2_id}"},
            "description": {"text": "Switched off NSAIDs to protect INR target"},
        }
        for res in (warfarin, story_obs_v1, story_goal_v2):
            rt = res["resourceType"]
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/{rt}/{res['id']}", json=res)
            assert r.status_code in (200, 201), r.text

        sem = asyncio.Semaphore(16)

        async def _put_clinic_obs(idx: int) -> None:
            obs = {
                "resourceType": "Observation",
                "id": f"obs-clinic-{ds_clinic_id}-{idx:03d}",
                "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
                "status": "final",
                "category": [{"coding": [{"code": "laboratory"}]}],
                "effectiveDateTime": f"2026-{(idx % 10) + 1:02d}-{(idx % 25) + 1:02d}",
                "subject": {"reference": f"Patient/{ds_clinic_id}"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "6301-6", "display": "INR in Platelet poor plasma"}]},
                "valueQuantity": {"value": round(2.1 + (idx % 15) * 0.1, 1), "unit": "{INR}"},
            }
            async with sem:
                resp = await hapi.put(
                    f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Observation/{obs['id']}",
                    json=obs,
                )
                assert resp.status_code in (200, 201), resp.text

        await asyncio.gather(*(_put_clinic_obs(i) for i in range(120)))

    southbound_calls: list[dict] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content.decode("utf-8"))
            southbound_calls.append(body)
            exact_p = count_chat_messages_tokens(body["messages"], tokenizer_family="gemma4")
            if body.get("stream"):
                sse = "".join(
                    [
                        'data: {"id":"cmpl-live-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Avoid ibuprofen while taking warfarin [1]."},"finish_reason":null}]}\n\n',
                        f'data: {{"id":"cmpl-live-1","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{exact_p},"completion_tokens":8}}}}\n\n',
                        "data: [DONE]\n\n",
                    ]
                )
                return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-live-2",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Goal updated to protect INR [1]."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": exact_p, "completion_tokens": 7, "total_tokens": exact_p + 7},
                },
            )
        return httpx.Response(404)

    settings = WahsEngineSettings(
        fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
        llm_base_url="http://llm-mock:4000/v1",
        allow_local_bundle_mode=False,
        default_model="wahs-12b-kb",
        everything_page_size=50,
        max_everything_pages=10,
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

    async with (
        httpx.AsyncClient(timeout=30.0) as fhir_client,
        httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client,
    ):
        app = create_app(settings, fhir_http_client=fhir_client, llm_http_client=llm_client)
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://wahs-test") as api:
            # 1. Production zero-privilege guard rejects inline bundle even with valid JWT
            bad_bundle = await api.post(
                "/v1/wahs/context/preview",
                headers=headers,
                json={"model": "wahs-12b-kb", "wahs": {"fhir_bundle": {"resourceType": "Bundle", "entry": []}}},
            )
            assert bad_bundle.status_code == 403

            # 2. Preview over real jwt-proxy + HAPI v8.0.0 without target_budget_tokens
            prev_resp = await api.post(
                "/v1/wahs/context/preview",
                headers=headers,
                json={"model": "wahs-12b-kb", "wahs": {}},
            )
            assert prev_resp.status_code == 200, prev_resp.text
            prev_data = prev_resp.json()
            prev_cond = prev_data["x_wahs"]["condensation"]
            assert prev_cond["budget_effective"] == 16384
            assert prev_cond["clamp_reason"] == "chart_budget_ceiling"
            assert prev_cond["cache_hit"] is False
            assert prev_cond["dataset_fetches"] == 2
            assert prev_cond["partial_fetch"] is False
            assert "Warfarin sodium" in prev_data["chart_text"]
            assert "OTC ibuprofen" in prev_data["chart_text"]

            # 3. SSE chat immediately hits warm prefix cache (`cache_hit=True`, `dataset_fetches=0`)
            sse_resp = await api.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "wahs-12b-kb",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Are my knee meds safe with warfarin?"}],
                    "wahs": {},
                },
            )
            assert sse_resp.status_code == 200, sse_resp.text
            assert sse_resp.headers["X-WAHS-Prefix-Hash"] == prev_data["prefix_cache_key_hash"]
            sse_events = [
                json.loads(line[5:].strip())
                for line in sse_resp.text.splitlines()
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
            final_cond = sse_events[-1]["x_wahs"]["condensation"]
            assert final_cond["cache_hit"] is True
            assert final_cond["dataset_fetches"] == 0
            assert final_cond["suspected_upstream_truncation"] is False
            assert southbound_calls[-1]["num_ctx"] == 32768

            # 4. Replace story dataset on MasterPatient in HAPI -> clinic dataset reused from cache!
            async with httpx.AsyncClient(timeout=30.0) as hapi:
                updated_master = _make_patient_resource(
                    master_id, sub=sub, is_master=True, links=[ds_clinic_id, ds_story_v2_id]
                )
                r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{master_id}", json=updated_master)
                assert r.status_code in (200, 201), r.text

            sync_resp = await api.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What changed in my story?"}],
                    "wahs": {},
                },
            )
            assert sync_resp.status_code == 200, sync_resp.text
            sync_cond = sync_resp.json()["x_wahs"]["condensation"]
            assert sync_cond["cache_hit"] is False
            assert sync_cond["dataset_cache_hits"] == 1
            assert sync_cond["dataset_fetches"] == 1
            assert "Switched off NSAIDs to protect INR target" in southbound_calls[-1]["messages"][0]["content"]

            # 5. Partial fetch over real HAPI + jwt-proxy returns 409 for external_share unless opted in
            capped_settings = WahsEngineSettings(
                fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
                llm_base_url="http://llm-mock:4000/v1",
                allow_local_bundle_mode=False,
                default_model="wahs-12b-kb",
                everything_page_size=50,
                max_everything_pages=1,
            )
            capped_app = create_app(capped_settings, fhir_http_client=fhir_client, llm_http_client=llm_client)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=capped_app), base_url="http://wahs-capped"
            ) as capped_api:
                resp_409 = await capped_api.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "wahs-12b-kb",
                        "stream": False,
                        "messages": [{"role": "user", "content": "External summary"}],
                        "wahs": {"sharing_posture": "external_share"},
                    },
                )
                assert resp_409.status_code == 409

                resp_ok_partial = await capped_api.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "wahs-12b-kb",
                        "stream": False,
                        "messages": [{"role": "user", "content": "External summary"}],
                        "wahs": {"sharing_posture": "external_share", "allow_partial_external_share": True},
                    },
                )
                assert resp_ok_partial.status_code == 200
                assert resp_ok_partial.json()["x_wahs"]["condensation"]["partial_fetch"] is True
