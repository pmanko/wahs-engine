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

from wahs_engine.assembler import (
    CATEGORY_SYSTEM,
    IDENTIFIER_SYSTEM,
    METHOD_SYSTEM,
    SECURITY_SYSTEM,
    SOURCE_NAME_SYSTEM,
    WAHSBundleAssembler,
)
from wahs_engine.config import WahsEngineSettings
from wahs_engine.jwt_claims import decode_and_bind_jwt
from wahs_engine.schemas import WahsRequestOptions

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


@pytest.mark.skipif(
    not (LIVE_FHIR_AUTH_URL and LIVE_HAPI_INTERNAL_URL),
    reason="WAHS_LIVE_FHIR_AUTH_URL and WAHS_LIVE_HAPI_INTERNAL_URL not set",
)
@pytest.mark.asyncio
async def test_live_hapi_and_jwt_proxy_pagination_and_freshness() -> None:
    tmpdir, jwks, sign_jwt = _generate_rsa_jwks_and_signer()
    jwks_server = _start_jwks_server(jwks, LIVE_JWKS_PORT)

    sub = f"live-user-{uuid.uuid4().hex[:8]}"
    master_id = f"master-{sub}"
    ds_v1_id = f"ds-story-v1-{uuid.uuid4().hex[:6]}"
    ds_v2_id = f"ds-story-v2-{uuid.uuid4().hex[:6]}"
    token = sign_jwt(sub)
    # Empty hs256_secret mirrors production (`jwt-proxy` verifies RS256 against JWKS_URL)
    ident = decode_and_bind_jwt(token, hs256_secret="")

    def _patient_resource(
        pid: str,
        *,
        is_master: bool = False,
        links: list[str] | None = None,
        given: str = "Live",
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
                {"system": CATEGORY_SYSTEM, "code": "patient-story"},
                {"system": METHOD_SYSTEM, "code": "patient-story-form"},
                {"system": SOURCE_NAME_SYSTEM, "code": "Patient Story"},
            ]
        return res

    try:
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            # 1. Seed ds-story-v1 with 250 Observations so >200 threshold and pagination can both be measured
            for res in [
                _patient_resource(ds_v1_id),
                _patient_resource(ds_v2_id),
                _patient_resource(master_id, is_master=True, links=[ds_v1_id]),
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
                updated_master = _patient_resource(master_id, is_master=True, links=[ds_v2_id])
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
    finally:
        jwks_server.shutdown()
        jwks_server.server_close()
        tmpdir.cleanup()
