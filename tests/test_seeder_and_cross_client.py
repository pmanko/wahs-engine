"""Stage 3 (`AC-3.2` & `AC-3.3`) integration tests for multi-source HAPI seeding and cross-client cache invalidation."""

from __future__ import annotations

import base64
import copy
import json
from typing import Any, Dict, List
import uuid

from fastapi import FastAPI, Request
import httpx
import pytest

from wahs_engine.app import create_app
from wahs_engine.assembler import (
    CATEGORY_SYSTEM,
    METHOD_SYSTEM,
    SECURITY_SYSTEM,
    SOURCE_NAME_SYSTEM,
)
from wahs_engine.config import WahsEngineSettings
from wahs_engine.seeder import OBSERVATION_CATEGORY_SYSTEM, seed_persona_to_hapi


def _make_jwt(sub: str) -> str:
    hdr = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).decode().rstrip("=")
    return f"{hdr}.{body}.sig"


def _build_in_memory_hapi_and_jwt_proxy() -> FastAPI:
    """In-memory HAPI FHIR R4 + CIRG `jwt-proxy` enforcing Keycloak `sub` security labels."""
    hapi = FastAPI()
    store: Dict[str, Dict[str, Any]] = {}
    state = {"seq": 0, "tick": 0}

    def _next_ts() -> str:
        state["tick"] += 1
        return f"2026-11-11T18:00:{state['tick']:02d}Z"

    def _caller_sub(req: Request) -> str:
        auth = req.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            tok = auth.split(" ", 1)[1]
            parts = tok.split(".")
            if len(parts) == 3:
                padded = parts[1] + "=" * (-len(parts[1]) % 4)
                return json.loads(base64.urlsafe_b64decode(padded)).get("sub", "")
        return ""

    def _visible_to_sub(res: Dict[str, Any], sub: str) -> bool:
        if not sub:
            return True  # Admin HAPI seeder port
        secs = (res.get("meta") or {}).get("security") or []
        return any(s.get("system") == SECURITY_SYSTEM and s.get("code") == sub for s in secs)

    @hapi.get("/fhir/Patient")
    async def search_patient(identifier: str, request: Request) -> Dict[str, Any]:
        sub = _caller_sub(request)
        sys_uri, val = identifier.split("|", 1)
        matches = []
        for r in store.values():
            if r.get("resourceType") == "Patient" and _visible_to_sub(r, sub):
                if any(i.get("system") == sys_uri and i.get("value") == val for i in r.get("identifier") or []):
                    matches.append({"resource": copy.deepcopy(r)})
        return {"resourceType": "Bundle", "type": "searchset", "total": len(matches), "entry": matches}

    @hapi.post("/fhir/Patient")
    async def create_patient(request: Request) -> Dict[str, Any]:
        body = await request.json()
        sub = _caller_sub(request)
        state["seq"] += 1
        pid = f"pat-{state['seq']}"
        saved = copy.deepcopy(body)
        saved["id"] = pid
        meta = dict(saved.get("meta") or {})
        meta["lastUpdated"] = _next_ts()
        if sub and not meta.get("security"):
            meta["security"] = [{"system": SECURITY_SYSTEM, "code": sub}]
        saved["meta"] = meta
        store[f"Patient/{pid}"] = saved
        return saved

    @hapi.get("/fhir/Patient/{pid}")
    async def get_patient(pid: str, request: Request) -> Dict[str, Any]:
        from fastapi import HTTPException

        sub = _caller_sub(request)
        pat = store.get(f"Patient/{pid}")
        if not pat or not _visible_to_sub(pat, sub):
            raise HTTPException(status_code=404, detail="Not found")
        return copy.deepcopy(pat)

    @hapi.put("/fhir/Patient/{pid}")
    async def update_patient(pid: str, request: Request) -> Dict[str, Any]:
        body = await request.json()
        sub = _caller_sub(request)
        saved = copy.deepcopy(body)
        saved["id"] = pid
        meta = dict(saved.get("meta") or {})
        meta["lastUpdated"] = _next_ts()
        if sub and not meta.get("security"):
            meta["security"] = [{"system": SECURITY_SYSTEM, "code": sub}]
        saved["meta"] = meta
        store[f"Patient/{pid}"] = saved
        return saved

    @hapi.get("/fhir/Patient/{pid}/$everything")
    async def patient_everything(pid: str, request: Request) -> Dict[str, Any]:
        sub = _caller_sub(request)
        pat_ref = f"Patient/{pid}"
        entries: List[Dict[str, Any]] = []
        ds_pat = store.get(pat_ref)
        if ds_pat and _visible_to_sub(ds_pat, sub):
            entries.append({"resource": copy.deepcopy(ds_pat)})
        for r in store.values():
            if r.get("resourceType") != "Patient" and _visible_to_sub(r, sub):
                if (r.get("subject") or {}).get("reference") == pat_ref or (r.get("patient") or {}).get("reference") == pat_ref:
                    entries.append({"resource": copy.deepcopy(r)})
        return {"resourceType": "Bundle", "type": "searchset", "entry": entries}

    @hapi.post("/fhir")
    async def post_transaction(request: Request) -> Dict[str, Any]:
        body = await request.json()
        sub = _caller_sub(request)
        urn_map: Dict[str, str] = {}
        staged = []
        for entry in body.get("entry") or []:
            rt = entry["resource"]["resourceType"]
            state["seq"] += 1
            rid = f"{rt.lower()}-{state['seq']}"
            if entry.get("fullUrl"):
                urn_map[entry["fullUrl"]] = f"{rt}/{rid}"
            staged.append((entry, rt, rid))

        resp_entries = []
        for entry, rt, rid in staged:
            raw_str = json.dumps(entry["resource"])
            for urn, ref in urn_map.items():
                raw_str = raw_str.replace(urn, ref)
            rewritten = json.loads(raw_str)
            rewritten["id"] = rid
            meta = dict(rewritten.get("meta") or {})
            meta["lastUpdated"] = _next_ts()
            if sub and not meta.get("security"):
                meta["security"] = [{"system": SECURITY_SYSTEM, "code": sub}]
            rewritten["meta"] = meta
            store[f"{rt}/{rid}"] = rewritten
            resp_entries.append({"response": {"status": "201 Created", "location": f"{rt}/{rid}/_history/1"}})

        return {"resourceType": "Bundle", "type": "transaction-response", "entry": resp_entries}

    return hapi


def _build_mock_llm_app(captured_prompts: List[str]) -> FastAPI:
    llm = FastAPI()

    @llm.post("/v1/chat/completions")
    async def completions(request: Request) -> Dict[str, Any]:
        body = await request.json()
        sys_msg = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        captured_prompts.append(sys_msg)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": 1760000000,
            "model": body.get("model", "gemma4:12b"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Grounded review across pooled clinics and My Story [1] [2].",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return llm


@pytest.mark.asyncio
async def test_seeder_and_cross_client_write_invalidates_only_updated_dataset_cache() -> None:
    """Verifies AC-3.2 + AC-3.3 end-to-end:
    1. `seed_persona_to_hapi` seeds multi-source Alex (4 datasets) + Leo (3 datasets) with Keycloak `sub` security labels.
    2. `wahs-engine` assembles Alex's 4-source record via `fhir-auth` and caches datasets by `(verified_sub, dataset_id, lastUpdated)`.
    3. A mobile companion (`wahs-mobile-frontend`) writes a new `LOINC 51855-5` `Observation` + `Goal` into HAPI via `fhir-auth` and updates `MasterPatient.link[]`.
    4. On the very next `/v1/chat/completions` turn (zero manual cache flush), `wahs-engine` keeps the 3 clinic datasets cached (`dataset_cache_hits == 3`, `dataset_fetches == 1`), fetches ONLY the newly updated `My Story` dataset, and includes the new story note in the condensed prompt & `x_wahs.citation_map`!
    """
    hapi_app = _build_in_memory_hapi_and_jwt_proxy()
    captured_prompts: List[str] = []
    llm_app = _build_mock_llm_app(captured_prompts)

    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=hapi_app), base_url="http://fhir-auth:8080") as fhir_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=llm_app), base_url="http://gpu:8079") as llm_client,
    ):
        alex_seed = await seed_persona_to_hapi(
            fhir_base_url="http://fhir-auth:8080/fhir",
            keycloak_sub="kc-alex-sub-1",
            persona="alex",
            http_client=fhir_client,
        )
        leo_seed = await seed_persona_to_hapi(
            fhir_base_url="http://fhir-auth:8080/fhir",
            keycloak_sub="kc-leo-sub-2",
            persona="leo",
            http_client=fhir_client,
        )
        assert len(alex_seed["dataset_references"]) == 4
        assert len(leo_seed["dataset_references"]) == 3

        engine_app = create_app(
            WahsEngineSettings(
                fhir_auth_base_url="http://fhir-auth:8080/fhir",
                llm_base_url="http://gpu:8079/v1",
                allow_local_bundle_mode=True,
            ),
            fhir_http_client=fhir_client,
            llm_http_client=llm_client,
        )

        alex_jwt = _make_jwt("kc-alex-sub-1")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engine_app), base_url="http://wahs-engine") as engine_client:
            # Turn 1: initial 4-source fetch
            r1 = await engine_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {alex_jwt}"},
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Review my medications and notes."}],
                },
            )
            assert r1.status_code == 200
            xw1 = r1.json()["x_wahs"]
            assert xw1["condensation"]["datasets_included"] == 4
            assert xw1["condensation"]["dataset_fetches"] == 4
            assert "Warfarin" in captured_prompts[-1]

            # Cross-client write from `wahs-mobile-frontend`: post updated LOINC 51855-5 Observation + Goal and update MasterPatient.link[]
            new_story_tx = await fhir_client.post(
                "/fhir",
                headers={"Authorization": f"Bearer {alex_jwt}"},
                json={
                    "resourceType": "Bundle",
                    "type": "transaction",
                    "entry": [
                        {
                            "fullUrl": "urn:uuid:new-story-pat",
                            "request": {"method": "POST", "url": "Patient"},
                            "resource": {
                                "resourceType": "Patient",
                                "meta": {
                                    "source": "http://localhost:4000",
                                    "tag": [
                                        {"system": CATEGORY_SYSTEM, "code": "patient-story"},
                                        {"system": METHOD_SYSTEM, "code": "patient-story-form"},
                                        {"system": SOURCE_NAME_SYSTEM, "code": "My Story"},
                                    ],
                                },
                                "name": [{"given": ["Alex"], "family": "Johnson"}],
                                "birthDate": "1962-03-14",
                            },
                        },
                        {
                            "fullUrl": "urn:uuid:new-story-obs",
                            "request": {"method": "POST", "url": "Observation"},
                            "resource": {
                                "resourceType": "Observation",
                                "status": "final",
                                "category": [
                                    {
                                        "coding": [
                                            {
                                                "system": OBSERVATION_CATEGORY_SYSTEM,
                                                "code": "patient-story",
                                                "display": "Patient Story",
                                            }
                                        ]
                                    }
                                ],
                                "code": {
                                    "coding": [
                                        {"system": "http://loinc.org", "code": "51855-5", "display": "Patient Note"}
                                    ]
                                },
                                "subject": {"reference": "urn:uuid:new-story-pat"},
                                "effectiveDateTime": "2026-11-11",
                                "valueString": "Switched from urgent care ibuprofen to topical diclofenac gel per cardiology pharmacist.",
                            },
                        },
                    ],
                },
            )
            assert new_story_tx.status_code == 200
            new_story_ref = new_story_tx.json()["entry"][0]["response"]["location"].split("/_history")[0]

            # Replace the 4th dataset link (old My Story) on MasterPatient with new_story_ref
            updated_links = [
                {"other": {"reference": ref}, "type": "seealso"}
                for ref in [*alex_seed["dataset_references"][:3], new_story_ref]
            ]
            master_id = alex_seed["master_patient_id"]
            put_master = await fhir_client.put(
                f"/fhir/Patient/{master_id}",
                headers={"Authorization": f"Bearer {alex_jwt}"},
                json={
                    "resourceType": "Patient",
                    "id": master_id,
                    "identifier": [{"system": "http://keycloak.cirg.uw.edu", "value": "kc-alex-sub-1"}],
                    "name": [{"given": ["Alex"], "family": "Johnson"}],
                    "birthDate": "1962-03-14",
                    "link": updated_links,
                },
            )
            assert put_master.status_code == 200

            # Turn 2: immediate next chat turn — 3 clinic datasets hit cache, 1 updated My Story dataset fetched!
            r2 = await engine_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {alex_jwt}"},
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What changed in my medication note?"}],
                },
            )
            assert r2.status_code == 200
            xw2 = r2.json()["x_wahs"]
            assert xw2["condensation"]["dataset_cache_hits"] == 3
            assert xw2["condensation"]["dataset_fetches"] == 1
            assert "diclofenac gel" in captured_prompts[-1]
            assert any("diclofenac gel" in v.get("display", "") for v in xw2["citation_map"].values())

            # Also verify `local-demo-token` works cleanly when allow_local_bundle_mode=True
            r_demo = await engine_client.post(
                "/v1/wahs/context/preview",
                headers={"Authorization": "Bearer local-demo-token"},
                json={
                    "model": "wahs-12b-kb",
                    "wahs": {
                        "fhir_bundle": {
                            "resourceType": "Bundle",
                            "entry": [
                                {
                                    "resource": {
                                        "resourceType": "Patient",
                                        "id": "demo-p1",
                                        "name": [{"given": ["Alex"], "family": "Johnson"}],
                                    }
                                }
                            ],
                        }
                    },
                },
            )
            assert r_demo.status_code == 200
