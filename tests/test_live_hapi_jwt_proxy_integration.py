"""Live integration test against real `HAPI FHIR v8.0.0` + `uwcirg/jwt-proxy`.

Gated on `WAHS_LIVE_FHIR_AUTH_URL` and `WAHS_LIVE_HAPI_INTERNAL_URL`.
Tests multi-page `$everything` pagination through `jwt-proxy`, `max_everything_pages=1`
truncation detection, dataset replacement (`ds-v1 -> ds-v2`), and in-place writes
inside an existing dataset (`wahs.refresh=True`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
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
LIVE_JWT_SECRET = os.environ.get("WAHS_LIVE_JWT_SECRET", "live-hapi-test-secret").strip()


def _make_jwt(sub: str, secret: str) -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": sub,
                "email": f"{sub}@example.org",
                "iat": int(time.time()) - 5,
                "exp": int(time.time()) + 3600,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig_input = f"{header}.{payload}".encode("ascii")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


@pytest.mark.skipif(
    not (LIVE_FHIR_AUTH_URL and LIVE_HAPI_INTERNAL_URL),
    reason="WAHS_LIVE_FHIR_AUTH_URL and WAHS_LIVE_HAPI_INTERNAL_URL not set",
)
@pytest.mark.asyncio
async def test_live_hapi_and_jwt_proxy_pagination_and_freshness() -> None:
    sub = f"live-user-{uuid.uuid4().hex[:8]}"
    master_id = f"master-{sub}"
    ds_v1_id = f"ds-story-v1-{uuid.uuid4().hex[:6]}"
    ds_v2_id = f"ds-story-v2-{uuid.uuid4().hex[:6]}"
    token = _make_jwt(sub, LIVE_JWT_SECRET)
    ident = decode_and_bind_jwt(token, hs256_secret=LIVE_JWT_SECRET)

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

    async with httpx.AsyncClient(timeout=30.0) as hapi:
        # 1. Seed ds-story-v1 with 25 Observations so `_count=10` guarantees multi-page `$everything`
        for res in [
            _patient_resource(ds_v1_id),
            _patient_resource(ds_v2_id),
            _patient_resource(master_id, is_master=True, links=[ds_v1_id]),
        ]:
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{res['id']}", json=res)
            assert r.status_code in (200, 201), r.text

        for i in range(25):
            obs = {
                "resourceType": "Observation",
                "id": f"obs-live-{ds_v1_id}-{i:02d}",
                "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
                "status": "final",
                "effectiveDateTime": "2026-11-01",
                "subject": {"reference": f"Patient/{ds_v1_id}"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
                "valueQuantity": {"value": 70 + i * 0.1, "unit": "kg"},
            }
            r = await hapi.put(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Observation/{obs['id']}", json=obs)
            assert r.status_code in (200, 201), r.text

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

    # 2. Verify multi-page `$everything` through jwt-proxy
    async with httpx.AsyncClient(timeout=30.0) as auth_client:
        settings_paginated = WahsEngineSettings(
            fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
            jwt_hs256_secret=LIVE_JWT_SECRET,
            everything_page_size=10,
            max_everything_pages=10,
        )
        asm = WAHSBundleAssembler(settings_paginated, http_client=auth_client)
        ctx_full = await asm.assemble(options=WahsRequestOptions(), identity=ident)
        assert ctx_full.partial_fetch is False
        # Patient + 25 observations across 3 pages (>= 25 non-master resources collected)
        assert ctx_full.extracted.total_bundle_entries >= 25

        # 3. Verify `max_everything_pages=1` with `_count=10` triggers `partial_fetch=True` cleanly!
        settings_capped = WahsEngineSettings(
            fhir_auth_base_url=LIVE_FHIR_AUTH_URL,
            jwt_hs256_secret=LIVE_JWT_SECRET,
            everything_page_size=10,
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

        # 5. In-place Observation added inside `ds_v2` WITHOUT touching `Patient/ds_v2`:
        #    record whether HAPI moved Patient/ds_v2 `meta.lastUpdated`, and prove `refresh=True` sees it!
        async with httpx.AsyncClient(timeout=30.0) as hapi:
            before_pat = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{ds_v2_id}")).json()
            in_place_goal = {
                "resourceType": "Goal",
                "id": f"goal-inplace-{uuid.uuid4().hex[:6]}",
                "meta": {"security": [{"system": SECURITY_SYSTEM, "code": sub}]},
                "lifecycleStatus": "active",
                "startDate": "2026-11-12",
                "subject": {"reference": f"Patient/{ds_v2_id}"},
                "description": {"text": "In-place appended hydration goal"},
            }
            r = await hapi.put(
                f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Goal/{in_place_goal['id']}", json=in_place_goal
            )
            assert r.status_code in (200, 201), r.text
            after_pat = (await hapi.get(f"{LIVE_HAPI_INTERNAL_URL.rstrip('/')}/Patient/{ds_v2_id}")).json()

        patient_last_updated_moved = (
            before_pat.get("meta", {}).get("lastUpdated")
            != after_pat.get("meta", {}).get("lastUpdated")
        )
        print(f"HAPI v8.0.0 in-place edit moved Patient.meta.lastUpdated={patient_last_updated_moved}")

        ctx_refreshed = await asm.assemble(
            options=WahsRequestOptions(refresh=True),
            identity=ident,
        )
        assert ctx_refreshed.cache_hit is False
        assert "In-place appended hydration goal" in ctx_refreshed.condensed.chart_text
