"""Security, timeliness, pagination, and `chart_scope` tests for `WAHSBundleAssembler` (AC-2.2)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List

from fastapi import HTTPException
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
from wahs_engine.schemas import ChartScope, WahsRequestOptions

TEST_SECRET = "test-wahs-hmac-secret-key"


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


class MockFhirAuthServer:
    """In-memory mock of CIRG `fhir-auth` (`uwcirg/jwt-proxy`) + HAPI FHIR R4."""

    def __init__(self) -> None:
        self.request_log: List[str] = []
        self.story_version = "2026-11-10T10:00:00Z"
        self.story_goal_text = "Walk 20 minutes daily"
        self.valid_tokens = {
            "user-alice": _make_jwt("user-alice"),
            "user-bob": _make_jwt("user-bob"),
        }
        self._token_to_sub = {tok: sub for sub, tok in self.valid_tokens.items()}

    def handler(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        sub = self._token_to_sub.get(token)
        self.request_log.append(f"{request.method} {request.url.path}?{request.url.query.decode()}")

        if not sub:
            return httpx.Response(401, json={"detail": "jwt-proxy: invalid or unauthorized token"})

        path = request.url.path
        query = request.url.query.decode()

        # MasterPatient search
        if path == "/fhir/Patient" and f"{IDENTIFIER_SYSTEM}|" in request.url.params.get("identifier", ""):
            q_sub = request.url.params["identifier"].split("|", 1)[1]
            if q_sub != sub:
                return httpx.Response(403, json={"detail": "jwt-proxy: forbidden cross-user query"})
            if sub == "user-alice":
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "id": "master-alice",
                                    "meta": {
                                        "lastUpdated": "2026-11-10T09:00:00Z",
                                        "security": [{"system": SECURITY_SYSTEM, "code": "user-alice"}],
                                    },
                                    "name": [{"given": ["Alice"], "family": "Nguyen"}],
                                    "birthDate": "1974-06-18",
                                    "link": [
                                        {"other": {"reference": "Patient/ds-uw-old"}, "type": "seealso"},
                                        {"other": {"reference": "Patient/ds-uw-new"}, "type": "seealso"},
                                        {"other": {"reference": "Patient/ds-story"}, "type": "seealso"},
                                    ],
                                }
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Patient",
                                "id": "master-bob",
                                "meta": {
                                    "lastUpdated": "2026-11-10T09:00:00Z",
                                    "security": [{"system": SECURITY_SYSTEM, "code": "user-bob"}],
                                },
                                "name": [{"given": ["Bob"], "family": "Patel"}],
                                "birthDate": "1982-01-22",
                                "link": [{"other": {"reference": "Patient/ds-bob-mc"}, "type": "seealso"}],
                            }
                        }
                    ],
                },
            )

        # Dataset Patient metadata stubs (including duplicate UW Medicine collection to test dedup!)
        if path == "/fhir/Patient/ds-uw-old":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "ds-uw-old",
                    "meta": {
                        "lastUpdated": "2026-09-01T00:00:00Z",
                        "source": "https://epic.uwmedicine.org/fhir",
                        "tag": [
                            {"system": CATEGORY_SYSTEM, "code": "provider-health-record"},
                            {"system": METHOD_SYSTEM, "code": "provider-health-record-sof"},
                            {"system": SOURCE_NAME_SYSTEM, "code": "UW Medicine"},
                        ],
                    },
                },
            )
        if path == "/fhir/Patient/ds-uw-new":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "ds-uw-new",
                    "meta": {
                        "lastUpdated": "2026-11-01T00:00:00Z",
                        "source": "https://epic.uwmedicine.org/fhir",
                        "tag": [
                            {"system": CATEGORY_SYSTEM, "code": "provider-health-record"},
                            {"system": METHOD_SYSTEM, "code": "provider-health-record-sof"},
                            {"system": SOURCE_NAME_SYSTEM, "code": "UW Medicine"},
                        ],
                    },
                },
            )
        if path == "/fhir/Patient/ds-story":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "ds-story",
                    "meta": {
                        "lastUpdated": self.story_version,
                        "source": "urn:wahealthsummary:patient-story",
                        "tag": [
                            {"system": CATEGORY_SYSTEM, "code": "patient-story"},
                            {"system": METHOD_SYSTEM, "code": "patient-story-form"},
                            {"system": SOURCE_NAME_SYSTEM, "code": "Patient Story"},
                        ],
                    },
                },
            )
        if path == "/fhir/Patient/ds-bob-mc":
            if sub != "user-bob":
                return httpx.Response(403, json={"detail": "Forbidden"})
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "ds-bob-mc",
                    "meta": {
                        "lastUpdated": "2026-11-01T00:00:00Z",
                        "tag": [
                            {"system": CATEGORY_SYSTEM, "code": "provider-health-record"},
                            {"system": METHOD_SYSTEM, "code": "provider-health-record-sof"},
                            {"system": SOURCE_NAME_SYSTEM, "code": "MultiCare"},
                        ],
                    },
                },
            )

        # Multi-page (>1,000 entries!) `$everything` with HAPI internal URL rewrite test
        if path == "/fhir/Patient/ds-uw-new/$everything":
            entries = [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-lisinopril",
                        "status": "active",
                        "authoredOn": "2026-10-15",
                        "subject": {"reference": "Patient/ds-uw-new"},
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
                # Foreign patient resource accidentally returned in bundle -> must be stripped!
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-foreign-intruder",
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "subject": {"reference": "Patient/someone-else-999"},
                        "code": {"text": "Unauthorized Intruder Diagnosis"},
                    }
                },
            ]
            for i in range(600):
                entries.append(
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "id": f"obs-p1-{i}",
                            "status": "final",
                            "category": [{"coding": [{"code": "laboratory"}]}],
                            "effectiveDateTime": "2025-05-10",
                            "subject": {"reference": "Patient/ds-uw-new"},
                            "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7", "display": "Glucose"}]},
                            "valueQuantity": {"value": 98, "unit": "mg/dL"},
                        }
                    }
                )
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    # Note internal HAPI URL (`https://fhir-internal:8080/fhir?_getpages=page2`) that MUST be rewritten to fhir-auth!
                    "link": [{"relation": "next", "url": "https://fhir-internal:8080/fhir?_getpages=uw_page_2"}],
                    "entry": entries,
                },
            )

        if path in {"/fhir/", "/fhir"} and "_getpages=uw_page_2" in query:
            entries_p2 = []
            for i in range(500):
                entries_p2.append(
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "id": f"obs-p2-{i}",
                            "status": "final",
                            "category": [{"coding": [{"code": "laboratory"}]}],
                            "effectiveDateTime": "2026-10-20",
                            "subject": {"reference": "Patient/ds-uw-new"},
                            "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]},
                            "valueQuantity": {"value": 1.35, "unit": "mg/dL"},
                        }
                    }
                )
            return httpx.Response(200, json={"resourceType": "Bundle", "type": "searchset", "entry": entries_p2})

        if path == "/fhir/Patient/ds-story/$everything":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Observation",
                                "id": "obs-alice-story",
                                "status": "final",
                                "effectiveDateTime": "2026-11-09",
                                "subject": {"reference": "Patient/ds-story"},
                                "code": {"coding": [{"system": "http://loinc.org", "code": "51855-5", "display": "Patient Note"}]},
                                "valueString": "Cough started two weeks after starting lisinopril.",
                            }
                        },
                        {
                            "resource": {
                                "resourceType": "Goal",
                                "id": "goal-alice-1",
                                "lifecycleStatus": "active",
                                "startDate": "2026-11-09",
                                "subject": {"reference": "Patient/ds-story"},
                                "description": {"text": self.story_goal_text},
                            }
                        },
                    ],
                },
            )

        if path == "/fhir/Patient/ds-bob-mc/$everything":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Condition",
                                "id": "cond-bob-asthma",
                                "clinicalStatus": {"coding": [{"code": "active"}]},
                                "onsetDateTime": "2026-04-01",
                                "subject": {"reference": "Patient/ds-bob-mc"},
                                "code": {"text": "Mild intermittent asthma"},
                            }
                        }
                    ],
                },
            )

        if path == "/fhir/Bundle/ips-share-1":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "id": "ips-share-1",
                    "meta": {"lastUpdated": "2026-11-10T11:00:00Z"},
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Composition",
                                "id": "comp-1",
                                "section": [{"entry": [{"reference": "MedicationRequest/med-lisinopril"}]}],
                            }
                        }
                    ],
                },
            )

        return httpx.Response(404, json={"detail": f"Unhandled mock path {path}?{query}"})


def test_fhir_internal_url_is_forbidden() -> None:
    with pytest.raises(ValueError, match="fhir-internal"):
        WahsEngineSettings(fhir_auth_base_url="http://fhir-internal:8080/fhir")


@pytest.mark.asyncio
async def test_star_graph_pagination_dedup_timeliness_and_cross_sub_isolation() -> None:
    mock_server = MockFhirAuthServer()
    transport = httpx.MockTransport(mock_server.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        settings = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            jwt_hs256_secret=TEST_SECRET,
        )
        assembler = WAHSBundleAssembler(settings, http_client=client)

        alice_id = decode_and_bind_jwt(mock_server.valid_tokens["user-alice"], hs256_secret=TEST_SECRET)
        bob_id = decode_and_bind_jwt(mock_server.valid_tokens["user-bob"], hs256_secret=TEST_SECRET)

        # 1. Initial Alice assembly: deduplicates ds-uw-old vs ds-uw-new, paginates >1,100 entries across 2 pages,
        #    rewrites https://fhir-internal:8080 onto http://fhir-auth:8080/fhir/, and strips foreign intruder!
        ctx1 = await assembler.assemble(options=WahsRequestOptions(), identity=alice_id)
        assert ctx1.patient_name == "Alice Nguyen"
        assert ctx1.cache_hit is False
        assert ctx1.dataset_fetches == 2  # ds-uw-new + ds-story (ds-uw-old deduplicated!)
        assert not any("ds-uw-old/$everything" in req for req in mock_server.request_log)
        assert "Creatinine" in ctx1.condensed.chart_text  # From page 2 (>1,000 entries)!
        assert "Walk 20 minutes daily" in ctx1.condensed.chart_text
        assert "Unauthorized Intruder Diagnosis" not in ctx1.condensed.chart_text
        assert ctx1.security_audit["stripped_foreign_count"] == 1

        # 2. Second identical Alice call -> hits prefix cache with 0 new $everything fetches!
        log_len_before = len(mock_server.request_log)
        ctx2 = await assembler.assemble(options=WahsRequestOptions(), identity=alice_id)
        assert ctx2.cache_hit is True
        new_reqs = mock_server.request_log[log_len_before:]
        assert not any("$everything" in r for r in new_reqs)

        # 3. Forged token claiming `sub="user-alice"` (even if HS256 secret is disabled) MUST hit fhir-auth and fail 401!
        no_secret_assembler = WAHSBundleAssembler(
            WahsEngineSettings(fhir_auth_base_url="http://fhir-auth:8080/fhir"),
            http_client=client,
        )
        # Seed cache first with valid Alice token
        alice_unverified = decode_and_bind_jwt(mock_server.valid_tokens["user-alice"])
        await no_secret_assembler.assemble(options=WahsRequestOptions(), identity=alice_unverified)
        forged_alice_tok = _make_jwt("user-alice", secret="attacker-wrong-secret")
        forged_id = decode_and_bind_jwt(forged_alice_tok)
        with pytest.raises(HTTPException) as exc_forged:
            await no_secret_assembler.assemble(options=WahsRequestOptions(), identity=forged_id)
        assert exc_forged.value.status_code == 401

        # 4. Caller user_sub mismatch spoofing -> HTTP 403
        with pytest.raises(HTTPException) as exc_mismatch:
            decode_and_bind_jwt(
                mock_server.valid_tokens["user-alice"],
                hs256_secret=TEST_SECRET,
                expected_user_sub="user-bob",
            )
        assert exc_mismatch.value.status_code == 403

        # 5. Cross-user isolation: Bob never sees Alice's meds/notes
        ctx_bob = await assembler.assemble(options=WahsRequestOptions(), identity=bob_id)
        assert ctx_bob.patient_name == "Bob Patel"
        assert "Lisinopril" not in ctx_bob.condensed.chart_text
        assert "Mild intermittent asthma" in ctx_bob.condensed.chart_text

        # 6. Timeliness (`meta.lastUpdated` advances on `ds-story` after saving an updated Goal):
        #    `ds-uw-new` is reused from `_dataset_cache` (`dataset_cache_hits == 1`), only `ds-story` is re-fetched!
        mock_server.story_version = "2026-11-11T12:30:00Z"
        mock_server.story_goal_text = "Walk 45 minutes daily and track home BP"
        ctx3 = await assembler.assemble(options=WahsRequestOptions(), identity=alice_id)
        assert ctx3.cache_hit is False
        assert ctx3.dataset_cache_hits == 1
        assert ctx3.dataset_fetches == 1
        assert "Walk 45 minutes daily and track home BP" in ctx3.condensed.chart_text


@pytest.mark.asyncio
async def test_chart_scope_ips_bundle_and_overlay_cap() -> None:
    mock_server = MockFhirAuthServer()
    transport = httpx.MockTransport(mock_server.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        settings = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            jwt_hs256_secret=TEST_SECRET,
        )
        assembler = WAHSBundleAssembler(settings, http_client=client)
        alice_id = decode_and_bind_jwt(mock_server.valid_tokens["user-alice"], hs256_secret=TEST_SECRET)

        # Exclude `patient-story` category -> only UW Medicine included
        ctx_no_story = await assembler.assemble(
            options=WahsRequestOptions(
                chart_scope=ChartScope(exclude_categories=["patient-story"]),
                overlay_resources=[{"type": "Flag", "item": "Lisinopril", "text": "Dry cough question"}],
            ),
            identity=alice_id,
        )
        assert "Walk 20 minutes daily" not in ctx_no_story.condensed.chart_text
        assert "Lisinopril" in ctx_no_story.condensed.chart_text
        assert "P1" in ctx_no_story.citation_map
        assert "Dry cough question" in ctx_no_story.system_prompt

        # IPS bundle scoping (`ips_bundle_id`) -> restricts to MedicationRequest/med-lisinopril
        ctx_ips = await assembler.assemble(
            options=WahsRequestOptions(chart_scope=ChartScope(ips_bundle_id="ips-share-1")),
            identity=alice_id,
        )
        assert "Lisinopril" in ctx_ips.condensed.chart_text
        assert "Walk 20 minutes daily" not in ctx_ips.condensed.chart_text

        # >5 KB overlay_resources -> HTTP 413
        oversize = [{"type": "Note", "text": "x" * 6000}]
        with pytest.raises(HTTPException) as exc_413:
            await assembler.assemble(
                options=WahsRequestOptions(overlay_resources=oversize),
                identity=alice_id,
            )
        assert exc_413.value.status_code == 413
