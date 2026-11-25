"""Comprehensive regression suite for the pristine `wahs-engine` baseline (`C1`-`C10`, `E4`-`E11`)."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException
import httpx
import pytest

from wahs_core import (
    ClinicalEntityGraph,
    build_temporal_digest,
    build_wahs_hub_enrichment_block,
    condense_bundle_to_budget,
    extract_bundle,
    get_lab_history,
    retrieve_clinical_note_passages,
    search_chart_entries,
    search_kb,
)
from wahs_core.normalizers import (
    classify_sensitivity,
    detect_condition_conflicts,
    normalize_drug_ingredient,
    validate_sharing_posture,
)
from wahs_engine.app import create_app
from wahs_engine.assembler import (
    CATEGORY_SYSTEM,
    IDENTIFIER_SYSTEM,
    SECURITY_SYSTEM,
    WAHSBundleAssembler,
    _BoundedLRUCache,
)
from wahs_engine.config import ModelSpec, WahsEngineSettings
from wahs_engine.jwt_claims import VerifiedCallerIdentity, decode_and_bind_jwt
from wahs_engine.schemas import ChartScope, WahsRequestOptions
from wahs_engine.upstream import clean_model_text


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_test_rsa_2048_keypair() -> Tuple[int, int, int, Dict[str, str]]:
    """Deterministic 2048-bit RSA test keypair using two fixed 1024-bit RFC 3526 / safe primes."""
    # Two distinct 1024-bit primes (p != q, gcd(e, (p-1)*(q-1)) == 1) for offline RS256 unit testing
    p = int(
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
        "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
        "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
        "FFFFFFFFFFFFFFFF",
        16,
    )
    # Next 1024-bit prime derived deterministically
    q_cand = int(
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
        "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
        "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
        "FFFFFFFFFFFFFC2F",
        16,
    )

    def _is_probable_prime(num: int) -> bool:
        if num % 2 == 0:
            return False
        d_val = num - 1
        r_val = 0
        while d_val % 2 == 0:
            d_val //= 2
            r_val += 1
        for a in (2, 3, 5, 7, 11, 13, 17):
            x = pow(a, d_val, num)
            if x in (1, num - 1):
                continue
            for _ in range(r_val - 1):
                x = pow(x, 2, num)
                if x == num - 1:
                    break
            else:
                return False
        return True

    e = 65537
    while not _is_probable_prime(q_cand) or math.gcd(e, q_cand - 1) != 1 or q_cand == p:
        q_cand -= 2
    q = q_cand
    n = p * q
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)
    k = (n.bit_length() + 7) // 8
    jwk = {
        "kty": "RSA",
        "kid": "wahs-test-rs256-1",
        "use": "sig",
        "alg": "RS256",
        "n": _b64url(n.to_bytes(k, "big")),
        "e": _b64url(e.to_bytes(3, "big")),
    }
    return n, e, d, jwk


def _sign_rs256_jwt(claims: Dict[str, Any], n: int, d: int, kid: str = "wahs-test-rs256-1") -> str:
    k = (n.bit_length() + 7) // 8
    hdr_b64 = _b64url(json.dumps({"alg": "RS256", "typ": "JWT", "kid": kid}).encode("utf-8"))
    pay_b64 = _b64url(json.dumps(claims).encode("utf-8"))
    signing_input = f"{hdr_b64}.{pay_b64}".encode("ascii")
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(signing_input).digest()
    ps_len = k - len(digest_info) - 3
    em = b"\x00\x01" + (b"\xff" * ps_len) + b"\x00" + digest_info
    sig_int = pow(int.from_bytes(em, "big"), d, n)
    sig_b64 = _b64url(sig_int.to_bytes(k, "big"))
    return f"{hdr_b64}.{pay_b64}.{sig_b64}"


def test_c1_c2_privacy_reasons_orgs_moud_arv_and_fail_closed_posture() -> None:
    """C1 + C2: Sensitivity classifies reasons/orgs/notes, covers MOUD/ARV/MHMDA, and fails closed on bad posture."""
    for med_name in (
        "Buprenorphine 8 MG / Naloxone 2 MG Sublingual Film",
        "Biktarvy 50-200-25 MG Oral Tablet",
        "Mifepristone 200 MG Oral Tablet",
    ):
        assert classify_sensitivity("", med_name) == "SDOH/PRIVACY-SENSITIVE", med_name

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-priv-1", "birthDate": "1985-03-10"}},
            {
                "resource": {
                    "resourceType": "Organization",
                    "id": "org-sud",
                    "name": "Evergreen Addiction Recovery Clinic",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-hiv",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": "HIV infection"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-benign-with-sens-reason",
                    "status": "active",
                    "medicationCodeableConcept": {"text": "Gabapentin 300 MG Oral Capsule"},
                    "reasonCode": [{"text": "Alcohol withdrawal management"}],
                    "requester": {"reference": "Organization/org-sud"},
                }
            },
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-sens-ref",
                    "status": "completed",
                    "code": {"text": "Outpatient counseling session"},
                    "reasonReference": [{"reference": "Condition/cond-hiv"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Goal",
                    "id": "goal-sens-note",
                    "lifecycleStatus": "active",
                    "description": {"text": "Maintain daily adherence"},
                    "note": [{"text": "Continue addiction treatment peer group weekly"}],
                }
            },
        ],
    }
    ext = extract_bundle(bundle)
    by_id = {it.resource_uuid: it for it in ext.items}
    assert by_id["med-benign-with-sens-reason"].sensitivity == "SDOH/PRIVACY-SENSITIVE"
    assert by_id["proc-sens-ref"].sensitivity == "SDOH/PRIVACY-SENSITIVE"
    assert by_id["goal-sens-note"].sensitivity == "SDOH/PRIVACY-SENSITIVE"

    res_search = search_chart_entries(ext, query="gabapentin counseling adherence", sharing_posture="external_share")
    assert res_search["results"] == []
    assert res_search["redacted_for_privacy_count"] >= 3

    for bad_posture in ("external", "public", "", "PATIENT_SELF"):
        with pytest.raises(ValueError, match="sharing_posture"):
            validate_sharing_posture(bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            search_chart_entries(ext, query="gabapentin", sharing_posture=bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            get_lab_history(ext, "creatinine", sharing_posture=bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            retrieve_clinical_note_passages(ext, "counseling", sharing_posture=bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            ClinicalEntityGraph.from_extracted_chart(ext, sharing_posture=bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            build_temporal_digest(ext, condense_bundle_to_budget(ext, target_tokens=2048), sharing_posture=bad_posture)
        with pytest.raises(ValueError, match="sharing_posture"):
            build_wahs_hub_enrichment_block("What meds?", extracted=ext, sharing_posture=bad_posture)


def test_c3_c4_c9_fhir_statuses_conclusions_dosages_and_coding_priority() -> None:
    """C3 + C4 + C9: Filter invalidated statuses, preserve conclusions & list dosages, and prefer standard codings."""
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-c349", "birthDate": "1970-06-15"}},
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-eie",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "verificationStatus": {"coding": [{"code": "entered-in-error"}]},
                    "code": {"text": "Sulfa antibiotics"},
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-refuted",
                    "clinicalStatus": {"coding": [{"code": "inactive"}]},
                    "verificationStatus": {"coding": [{"code": "refuted"}]},
                    "code": {"text": "Latex"},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-dual-coded",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [
                            {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "N18.32", "display": "CKD 3b"},
                            {"system": "http://snomed.info/sct", "code": "700379002", "display": "Chronic kidney disease stage 3B"},
                        ]
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationStatement",
                    "id": "medstmt-list-dosage",
                    "status": "active",
                    "dateAsserted": "2026-02-10",
                    "medicationCodeableConcept": {"text": "Apixaban 5 MG Oral Tablet"},
                    "dosage": [{"text": "5 mg twice daily"}],
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "id": "medadm-dose-qty",
                    "status": "completed",
                    "effectiveDateTime": "2026-02-09T08:00:00Z",
                    "medicationCodeableConcept": {"text": "Furosemide IV"},
                    "dosage": {"dose": {"value": 40, "unit": "mg"}},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-valid-k",
                    "status": "final",
                    "effectiveDateTime": "2026-02-10",
                    "code": {"text": "Potassium"},
                    "valueQuantity": {"value": 4.4, "unit": "mmol/L"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-eie-k",
                    "status": "entered-in-error",
                    "effectiveDateTime": "2026-02-10",
                    "code": {"text": "Potassium"},
                    "valueQuantity": {"value": 9.1, "unit": "mmol/L"},
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr-panel-mixed",
                    "status": "final",
                    "effectiveDateTime": "2026-02-10",
                    "code": {"text": "Electrolyte Panel"},
                    "result": [
                        {"reference": "Observation/obs-valid-k"},
                        {"reference": "Observation/obs-eie-k"},
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr-echo-conclusion",
                    "status": "final",
                    "effectiveDateTime": "2026-02-11",
                    "code": {"text": "Transthoracic Echocardiogram"},
                    "conclusion": "LVEF 42% with mild mitral regurgitation.",
                }
            },
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-not-done",
                    "status": "not-done",
                    "performedDateTime": "2026-02-08",
                    "code": {"text": "Colonoscopy"},
                }
            },
            {
                "resource": {
                    "resourceType": "Appointment",
                    "id": "appt-cancelled",
                    "status": "cancelled",
                    "start": "2026-05-01T10:00:00Z",
                    "description": "Cancelled Dermatology Visit",
                }
            },
        ],
    }
    ext = extract_bundle(bundle)
    by_id = {it.resource_uuid: it for it in ext.items}

    # C3: entered-in-error allergy excluded; refuted allergy not mandatory
    assert "alg-eie" not in by_id
    assert by_id["alg-refuted"].mandatory is False

    # C9: SNOMED CT preferred over ICD-10-CM at coding[0]
    assert by_id["cond-dual-coded"].code == "700379002"

    # C4: MedicationStatement list dosage + MedicationAdministration dose quantity + DiagnosticReport conclusion
    assert "5 mg twice daily" in by_id["medstmt-list-dosage"].text
    assert "40 mg" in by_id["medadm-dose-qty"].text
    assert "LVEF 42%" in by_id["dr-echo-conclusion"].text

    # C3: Invalidated child observation omitted from DiagnosticReport panel text
    assert "4.4 mmol/L" in by_id["dr-panel-mixed"].text
    assert "9.1" not in by_id["dr-panel-mixed"].text

    # C3: not-done procedure shows status and cancelled appointment never renders as upcoming
    assert "status: not-done" in by_id["proc-not-done"].text
    assert by_id["proc-not-done"].clinical_date is None
    assert by_id["appt-cancelled"].date_kind == "administrative"
    cond = condense_bundle_to_budget(ext, target_tokens=4096)
    assert "Cancelled Dermatology Visit" not in cond.chart_text
    dig = build_temporal_digest(ext, cond)
    assert dig.summary["upcoming_appointments_count"] == 0


def test_c5_c6_c7_c8_c10_electrolytes_cues_hba1c_units_hf_and_fts_threads() -> None:
    """C5 + C6 + C7 + C8 + C10: Salt preservation, clause-final cues, HbA1c direction, unit split, HF stage, FTS threads."""
    # C5: Electrolyte supplements stay distinct while drug counter-ion salts normalize to base drug
    assert normalize_drug_ingredient("Potassium Citrate 10 MEQ Extended Release Oral Tablet") == "potassium citrate"
    assert normalize_drug_ingredient("Potassium Bicarbonate 25 MEQ") == "potassium bicarbonate"
    assert normalize_drug_ingredient("Sodium Bicarbonate 650 MG Oral Tablet") == "sodium bicarbonate"
    assert normalize_drug_ingredient("Calcium Carbonate 500 MG Chewable Tablet") == "calcium carbonate"
    assert normalize_drug_ingredient("Losartan Potassium 50 MG Oral Tablet") == "losartan"

    # C8: NYHA Heart Failure stage conflict detection
    hf_conflicts = detect_condition_conflicts(
        [
            {"resourceUuid": "hf-2", "code": "", "display": "Heart failure NYHA class II"},
            {"resourceUuid": "hf-4", "code": "", "display": "Congestive heart failure NYHA class IV"},
        ]
    )
    assert "hf-2" in hf_conflicts and "hf-4" in hf_conflicts

    # C6 + C7: Sentence-final "resumed.", non-alerting "outside pharmacy", elevated HbA1c non-hypoglycemia,
    # unit-separated trajectories, mixed-TZ multi-month trend, and on-hold med separation
    note_hold = base64.b64encode(b"Hold lisinopril temporarily during AKI.").decode("ascii")
    note_resumed = base64.b64encode(b"Renal function recovered; lisinopril was resumed. Filled at outside pharmacy.").decode("ascii")
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-g1", "birthDate": "1962-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-lis",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-glargine",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Insulin Glargine 100 UNT/ML Pen Injector"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-spiro-hold",
                    "status": "on-hold",
                    "authoredOn": "2026-02-01",
                    "medicationCodeableConcept": {"text": "Spironolactone 25 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-1",
                    "status": "current",
                    "date": "2026-01-10",
                    "content": [{"attachment": {"contentType": "text/plain", "data": note_hold}}],
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-2",
                    "status": "current",
                    "date": "2026-01-20",
                    "content": [{"attachment": {"contentType": "text/plain", "data": note_resumed}}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-a1c-high",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": "2025-06-01",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "interpretation": [{"coding": [{"code": "H"}]}],
                    "valueQuantity": {"value": 9.2, "unit": "%"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-a1c-high-2",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": "2026-02-15T10:30:00Z",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "interpretation": [{"coding": [{"code": "H"}]}],
                    "valueQuantity": {"value": 8.4, "unit": "%"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-glu-mgdl",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": "2026-02-10",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7", "display": "Glucose"}]},
                    "valueQuantity": {"value": 110, "unit": "mg/dL"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-glu-mmol",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": "2026-02-12",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7", "display": "Glucose"}]},
                    "valueQuantity": {"value": 6.1, "unit": "mmol/L"},
                }
            },
        ],
    }
    ext = extract_bundle(bundle)
    graph = ClinicalEntityGraph.from_extracted_chart(ext)
    # Clause-final "resumed." cancels prior hold, and "outside pharmacy" does not emit false hold edge
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in graph.edges)
    # High HbA1c (8.4% [H]) must NOT trigger a false hypoglycemia contraindication edge on insulin glargine
    assert not any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" and "hypoglycemia" in e.rationale.lower()
        for e in graph.edges
    )
    # Glucose mg/dL and mmol/L stay in separate Observation trajectory nodes
    glu_nodes = [n for n in graph.nodes.values() if n.category == "Observation" and "glucose" in n.label.lower()]
    assert len(glu_nodes) == 2

    # Mixed date-only (2025-06-01) + UTC instant (2026-02-15T10:30:00Z) separated by >2 days supports trend,
    # and on-hold medication is reported separately from stopped medications
    cond = condense_bundle_to_budget(ext, target_tokens=4096)
    dig = build_temporal_digest(ext, cond, question="How is my Hemoglobin A1c changing?")
    assert dig.summary["multi_point_series_count"] >= 1
    assert "decreasing -0.8" in dig.digest_text
    assert "1 on-hold" in dig.digest_text

    # C10: Concurrent thread-pool FTS5 searches execute without SQLite cursor/thread errors
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda q: search_kb(q, k=3, corpus="wahs"),
                ["hyperkalemia lisinopril", "ckd stage egfr", "zoster vaccine", "clopidogrel plavix"] * 6,
            )
        )
    assert all(isinstance(r, list) and len(r) >= 1 for r in results)


def test_e4_rs256_jwks_verification_and_master_patient_ownership_guard() -> None:
    """E4: Pure-stdlib RS256 JWKS verification + iss/aud checks + MasterPatient ownership check."""
    n, _e, d, jwk = _generate_test_rsa_2048_keypair()
    jwks_str = json.dumps({"keys": [jwk]})
    now = int(time.time())
    valid_tok = _sign_rs256_jwt(
        {
            "sub": "kc-user-42",
            "iss": "https://keycloak.uw.edu/realms/wahs",
            "aud": ["wahs-client", "account"],
            "exp": now + 600,
        },
        n,
        d,
    )
    ident = decode_and_bind_jwt(
        valid_tok,
        jwks_json=jwks_str,
        expected_issuer="https://keycloak.uw.edu/realms/wahs",
        expected_audience="wahs-client",
    )
    assert ident.sub == "kc-user-42"

    # Tampered payload or wrong issuer/audience must fail 401
    tampered = valid_tok.rsplit(".", 1)[0] + "." + _b64url(b"\x01" * 256)
    with pytest.raises(HTTPException) as exc_sig:
        decode_and_bind_jwt(tampered, jwks_json=jwks_str)
    assert exc_sig.value.status_code == 401

    with pytest.raises(HTTPException) as exc_iss:
        decode_and_bind_jwt(valid_tok, jwks_json=jwks_str, expected_issuer="https://wrong-issuer")
    assert exc_iss.value.status_code == 401

    # MasterPatient explicit ID ownership check blocks cross-subject access
    async def _check_mp_guard() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "master-bob",
                    "identifier": [{"system": IDENTIFIER_SYSTEM, "value": "user-bob"}],
                    "meta": {"security": [{"system": SECURITY_SYSTEM, "code": "user-bob"}]},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            asm = WAHSBundleAssembler(WahsEngineSettings(), http_client=client)
            alice = VerifiedCallerIdentity(
                sub="user-alice", raw_token="t", token_fingerprint="fp", claims={"sub": "user-alice"}
            )
            with pytest.raises(HTTPException) as exc_mp:
                await asm.assemble(options=WahsRequestOptions(master_patient_id="master-bob"), identity=alice)
            assert exc_mp.value.status_code == 403

    asyncio.run(_check_mp_guard())


def test_e5_e6_e7_e8_e9_e11_runtime_streaming_preflight_think_tools_lru_and_scope() -> None:
    """E5-E9 + E11: Unclosed <think>, SSE preflight 422/502, emit_stream_events=False, malformed tool JSON, byte LRU, Mode-A category scope, and OpenAI params."""
    # E5: Unclosed `<think>` stripped cleanly when finish_reason == "length"
    assert clean_model_text("Visible start <think>cutoff internal reasoning without close tag") == "Visible start"
    assert clean_model_text("<think>purely internal cutoff") == ""
    assert clean_model_text("Prefix <|channel>thought internal cutoff") == "Prefix"

    # E9: Byte-bounded LRU cache evicts entries when max_bytes is exceeded
    byte_cache: _BoundedLRUCache[str, Dict[str, str]] = _BoundedLRUCache(
        max_entries=50, ttl_seconds=600.0, max_bytes=300
    )
    byte_cache.put("k1", {"payload": "x" * 140})
    byte_cache.put("k2", {"payload": "y" * 140})
    assert len(byte_cache) == 1
    assert byte_cache.get("k1") is None
    assert byte_cache.get("k2") is not None

    async def _run_api_checks() -> None:
        captured_payloads: List[Dict[str, Any]] = []
        round_state = {"mode": "normal", "calls": 0}

        def llm_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"object": "list", "data": []})
            body = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(body)
            round_state["calls"] += 1
            if round_state["mode"] == "upstream_500":
                return httpx.Response(500, json={"error": "GPU overloaded"})
            if round_state["mode"] == "malformed_tool_args":
                has_tool = any(m.get("role") == "tool" for m in body.get("messages") or [])
                if not has_tool:
                    if body.get("stream"):
                        sse = "\n\n".join(
                            [
                                'data: {"id":"m1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Pre-tool draft line 1\\nLine 2 over lookahead."},"finish_reason":null}]}',
                                'data: {"id":"m1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc_null","type":"function","function":{"name":"search_chart_entries","arguments":"null"}},{"index":1,"id":"tc_bad_int","type":"function","function":{"name":"get_lab_history","arguments":"{\\"analyte\\":\\"potassium\\",\\"limit\\":\\"not-a-number\\"}"}}]},"finish_reason":"tool_calls"}]}',
                                "data: [DONE]",
                            ]
                        ) + "\n\n"
                        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())
                    return httpx.Response(
                        200,
                        json={
                            "id": "m-sync",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {"id": "tc1", "type": "function", "function": {"name": "search_chart_entries", "arguments": "[]"}},
                                            {"id": "tc2", "type": "function", "function": {"name": "read_clinical_note", "arguments": '{"note_id_or_query":"Note","offset":"bad","max_chars":"bad"}'}},
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                        },
                    )
            if body.get("stream"):
                sse_ok = (
                    'data: {"id":"ok","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"CLEAN_FINAL"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_ok.encode())
            return httpx.Response(
                200,
                json={
                    "id": "ok-sync",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "SYNC_OK"}, "finish_reason": "stop"}],
                },
            )

        settings = WahsEngineSettings(
            allow_local_bundle_mode=True,
            llm_base_url="http://mock-llm/v1",
            default_model="wahs-26b-kb",
            model_specs={
                "wahs-26b-kb": ModelSpec(
                    upstream_model="ollama/gemma4:26b",
                    context_window=32768,
                    chart_budget_ceiling=16384,
                    tokenizer_family="gemma4",
                ),
                "wahs-tiny": ModelSpec(
                    upstream_model="ollama/gemma4:e4b",
                    context_window=1400,
                    chart_budget_ceiling=512,
                    default_max_output_tokens=256,
                    tokenizer_family="gemma4",
                ),
            },
        )
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-rt", "birthDate": "1970-01-01"}},
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-story-cat",
                        "status": "final",
                        "meta": {"tag": [{"system": CATEGORY_SYSTEM, "code": "patient-story"}]},
                        "code": {"text": "Patient Story Entry"},
                        "valueString": "My personal running goal",
                    }
                },
                *[
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "id": f"obs-lab-{i}",
                            "status": "final",
                            "category": [{"coding": [{"code": "laboratory"}]}],
                            "code": {"text": f"Potassium Repeat Panel {i}"},
                            "effectiveDateTime": "2026-01-10",
                            "valueQuantity": {"value": 4.2, "unit": "mmol/L"},
                        }
                    }
                    for i in range(40)
                ],
            ],
        }

        async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
            app = create_app(settings, llm_http_client=llm_client)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://wahs-engine"
            ) as api:
                # E6a: stream=True with only assistant messages returns HTTP 400 (not 200 SSE!)
                r_400 = await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-26b-kb",
                        "stream": True,
                        "messages": [{"role": "assistant", "content": "No user turn"}],
                        "wahs": {"fhir_bundle": bundle},
                    },
                )
                assert r_400.status_code == 400

                # E6b: stream=True prompt overflow returns HTTP 422 before committing 200 SSE headers
                r_422 = await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-tiny",
                        "stream": True,
                        "max_tokens": 900,
                        "messages": [{"role": "user", "content": "word " * 400}],
                        "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512},
                    },
                )
                assert r_422.status_code == 422

                # E6c: stream=True when upstream returns 500 returns clean HTTP 502 before 200 SSE headers
                round_state["mode"] = "upstream_500"
                r_502 = await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-26b-kb",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Check potassium"}],
                        "wahs": {"fhir_bundle": bundle},
                    },
                )
                assert r_502.status_code == 502

                # E7 + E8 + E11: Malformed tool JSON ("null", "[]", non-int strings) + emit_stream_events=False
                # + OpenAI top_p/stop/user/stream_options forwarding + Mode-A category scope
                round_state["mode"] = "malformed_tool_args"
                r_vanilla_sse = await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-26b-kb",
                        "stream": True,
                        "top_p": 0.9,
                        "stop": ["END_STOP"],
                        "user": "caller-123",
                        "stream_options": {"include_usage": True},
                        "messages": [{"role": "user", "content": "Check potassium"}],
                        "wahs": {
                            "fhir_bundle": bundle,
                            "target_budget_tokens": 512,
                            "emit_stream_events": False,
                            "chart_scope": ChartScope(exclude_categories=["patient-story"]).model_dump(),
                        },
                    },
                )
                assert r_vanilla_sse.status_code == 200
                assert "x_wahs_stream" not in r_vanilla_sse.text
                assert "Pre-tool draft line 1" not in r_vanilla_sse.text
                assert "CLEAN_FINAL" in r_vanilla_sse.text
                last_sent = captured_payloads[-1]
                assert last_sent["top_p"] == 0.9
                assert last_sent["stop"] == ["END_STOP"]
                assert last_sent["user"] == "caller-123"
                assert last_sent["stream_options"] == {"include_usage": True}
                assert "My personal running goal" not in last_sent["messages"][0]["content"]

                r_sync_tools = await api.post(
                    "/v1/chat/completions",
                    json={
                        "model": "wahs-26b-kb",
                        "stream": False,
                        "messages": [{"role": "user", "content": "Check notes"}],
                        "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512},
                    },
                )
                assert r_sync_tools.status_code == 200
                assert r_sync_tools.json()["choices"][0]["message"]["content"] == "SYNC_OK"

    asyncio.run(_run_api_checks())
