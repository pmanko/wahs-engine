"""Multi-source WAHS HAPI FHIR R4 star-graph seeder (`MasterPatient` + linked per-source datasets).

Stamps every seeded `MasterPatient`, dataset `Patient`, and clinical resource with:
- CIRG `jwt-proxy` security labels (`http://keycloak.cirg.uw.edu/fhir/security-labels|<keycloak_sub>`)
- CIRG WAHS dataset tags (`wahealthsummary-category`, `wahealthsummary-method`, `wahealthsummary-source-name`)

so seeded personas (`alex` 4 sources, `leo` 3 sources) and imported Synthea/Connectathon
FHIR Bundles work identically across `shl-ips`, `wahs-mobile-frontend`, and `wahs-engine`.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import httpx

from .assembler import (
    CATEGORY_SYSTEM,
    IDENTIFIER_SYSTEM,
    METHOD_SYSTEM,
    SECURITY_SYSTEM,
    SOURCE_NAME_SYSTEM,
    WAHS_SYSTEM_BASE,
)

OBSERVATION_CATEGORY_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/observation-category"
SEED_DATASET_IDENTIFIER_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-dataset-key"
_NON_PATIENT_COMPARTMENT_TYPES = {
    "Medication",
    "Organization",
    "Practitioner",
    "PractitionerRole",
    "Location",
}


def _extract_dataset_tuple(pat_res: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    meta = pat_res.get("meta") if isinstance(pat_res.get("meta"), Mapping) else {}
    src_uri = str(meta.get("source") or "").strip()
    cat = ""
    meth = ""
    for tag in meta.get("tag") or []:
        if not isinstance(tag, Mapping):
            continue
        sys_u = str(tag.get("system") or "").strip()
        code = str(tag.get("code") or "").strip()
        if sys_u == CATEGORY_SYSTEM and not cat:
            cat = code
        elif sys_u == METHOD_SYSTEM and not meth:
            meth = code
    if cat or meth or src_uri:
        return (cat, meth, src_uri)
    return None


def _rewrite_references(node: Any, ref_map: Mapping[str, str]) -> Any:
    if isinstance(node, list):
        return [_rewrite_references(item, ref_map) for item in node]
    if isinstance(node, Mapping):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k == "reference" and isinstance(v, str):
                raw_ref = v.strip()
                mapped = ref_map.get(raw_ref)
                if mapped is None and "/" in raw_ref:
                    parts = raw_ref.rsplit("/", 2)
                    if len(parts) == 3:
                        mapped = ref_map.get(f"{parts[1]}/{parts[2]}")
                out[k] = mapped if mapped is not None else v
            else:
                out[k] = _rewrite_references(v, ref_map)
        return out
    return node


def _stamp_security_label(resource: Dict[str, Any], keycloak_sub: str) -> Dict[str, Any]:
    res = copy.deepcopy(resource)
    meta = dict(res.get("meta") or {})
    sec = list(meta.get("security") or [])
    if not any(
        isinstance(s, Mapping)
        and s.get("system") == SECURITY_SYSTEM
        and s.get("code") == keycloak_sub
        for s in sec
    ):
        sec.append({"system": SECURITY_SYSTEM, "code": keycloak_sub})
    meta["security"] = sec
    res["meta"] = meta
    return res


def build_persona_star_graph_datasets(
    persona: str = "alex",
) -> Dict[str, Any]:
    """Return `{demographics, datasets}` for `alex` (4 sources) or `leo` (3 sources)."""
    if persona == "leo":
        return {
            "demographics": {
                "given": ["Leo"],
                "family": "Johnson",
                "gender": "male",
                "birthDate": "2019-08-22",
                "email": "leo.proxy@example.org",
            },
            "datasets": [
                {
                    "category": "provider-health-record",
                    "method": "provider-health-record-sof",
                    "source_uri": "https://fhir.seattlechildrens.example.org/r4",
                    "source_name": "Seattle Children's Pulmonary",
                    "resources": [
                        {
                            "resourceType": "Condition",
                            "clinicalStatus": {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                        "code": "active",
                                    }
                                ]
                            },
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "195967001",
                                        "display": "Mild persistent asthma",
                                    }
                                ],
                                "text": "Mild persistent asthma",
                            },
                            "recordedDate": "2025-11-10",
                        },
                        {
                            "resourceType": "MedicationRequest",
                            "status": "active",
                            "intent": "order",
                            "medicationCodeableConcept": {
                                "coding": [
                                    {
                                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                        "code": "896188",
                                        "display": "Fluticasone propionate 44 MCG/ACTUAT Metered Dose Inhaler",
                                    }
                                ],
                                "text": "Fluticasone HFA 44 mcg inhaler (2 puffs twice daily)",
                            },
                            "authoredOn": "2026-08-18",
                        },
                        {
                            "resourceType": "AllergyIntolerance",
                            "criticality": "high",
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "91935009",
                                        "display": "Allergy to peanut",
                                    }
                                ],
                                "text": "Peanut (anaphylaxis — carries EpiPen Jr 0.15 mg)",
                            },
                        },
                    ],
                },
                {
                    "category": "provider-health-record",
                    "method": "provider-health-record-sof",
                    "source_uri": "https://fhir.eastsidepeds.example.org/r4",
                    "source_name": "Eastside Pediatrics",
                    "resources": [
                        {
                            "resourceType": "MedicationRequest",
                            "status": "active",
                            "intent": "order",
                            "medicationCodeableConcept": {
                                "text": "Albuterol HFA 90 mcg rescue inhaler (2 puffs every 4h PRN wheezing)"
                            },
                            "authoredOn": "2026-09-02",
                        },
                        {
                            "resourceType": "Immunization",
                            "status": "completed",
                            "vaccineCode": {
                                "coding": [
                                    {
                                        "system": "http://hl7.org/fhir/sid/cvx",
                                        "code": "03",
                                        "display": "MMR",
                                    }
                                ],
                                "text": "MMR Dose #2 (Kindergarten booster)",
                            },
                            "occurrenceDateTime": "2026-08-25",
                        },
                    ],
                },
                {
                    "category": "patient-story",
                    "method": "patient-story-form",
                    "source_uri": "http://localhost:4000",
                    "source_name": "My Story",
                    "resources": [
                        {
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
                                    {
                                        "system": "http://loinc.org",
                                        "code": "51855-5",
                                        "display": "Patient Note",
                                    }
                                ]
                            },
                            "effectiveDateTime": "2026-10-12",
                            "valueString": "Caregiver note: Leo coughs after outdoor recess when wildfire smoke AQI > 80. School nurse needs updated spacer instructions.",
                        },
                        {
                            "resourceType": "Goal",
                            "lifecycleStatus": "active",
                            "statusDate": "2026-10-12",
                            "description": {
                                "text": "Keep asthma controlled through soccer season without nighttime coughing"
                            },
                            "achievementStatus": {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/goal-achievement",
                                        "code": "in-progress",
                                        "display": "In Progress",
                                    }
                                ]
                            },
                        },
                    ],
                },
            ],
        }

    return {
        "demographics": {
            "given": ["Alex"],
            "family": "Johnson",
            "gender": "male",
            "birthDate": "1962-03-14",
            "email": "alex.johnson@example.org",
        },
        "datasets": [
            {
                "category": "provider-health-record",
                "method": "provider-health-record-sof",
                "source_uri": "https://fhir.uwmedicine.example.org/r4",
                "source_name": "UW Medicine Cardiology & Nephrology",
                "resources": [
                    {
                        "resourceType": "Condition",
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active",
                                }
                            ]
                        },
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "49436004",
                                    "display": "Atrial fibrillation",
                                }
                            ],
                            "text": "Paroxysmal atrial fibrillation",
                        },
                        "recordedDate": "2024-05-19",
                    },
                    {
                        "resourceType": "Condition",
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active",
                                }
                            ]
                        },
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "433144002",
                                    "display": "Chronic kidney disease stage 3A",
                                }
                            ],
                            "text": "Chronic kidney disease stage 3a (baseline eGFR 51 mL/min)",
                        },
                        "recordedDate": "2025-03-04",
                    },
                    {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                    "code": "855332",
                                    "display": "Warfarin Sodium 5 MG Oral Tablet",
                                }
                            ],
                            "text": "Warfarin sodium 5 mg tablet once daily (target INR 2.0-3.0)",
                        },
                        "authoredOn": "2026-09-12",
                    },
                    {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "text": "Lisinopril 20 mg tablet once daily"
                        },
                        "authoredOn": "2026-09-12",
                    },
                    {
                        "resourceType": "Observation",
                        "status": "final",
                        "category": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                        "code": "laboratory",
                                    }
                                ]
                            }
                        ],
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "62238-1",
                                    "display": "eGFR CKD-EPI",
                                }
                            ],
                            "text": "eGFR",
                        },
                        "effectiveDateTime": "2026-10-03",
                        "valueQuantity": {"value": 51, "unit": "mL/min/1.73m2"},
                    },
                ],
            },
            {
                "category": "provider-health-record",
                "method": "provider-health-record-sof",
                "source_uri": "https://fhir.cityclinic.example.org/r4",
                "source_name": "City Clinic Primary Care",
                "resources": [
                    {
                        "resourceType": "Condition",
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active",
                                }
                            ]
                        },
                        "code": {"text": "Type 2 diabetes mellitus"},
                        "recordedDate": "2021-07-22",
                    },
                    {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "text": "Metformin ER 500 mg twice daily with meals"
                        },
                        "authoredOn": "2026-09-28",
                    },
                    {
                        "resourceType": "Observation",
                        "status": "final",
                        "category": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                        "code": "laboratory",
                                    }
                                ]
                            }
                        ],
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "4548-4",
                                    "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
                                }
                            ],
                            "text": "Hemoglobin A1c",
                        },
                        "effectiveDateTime": "2026-10-05",
                        "valueQuantity": {"value": 7.4, "unit": "%"},
                    },
                    {
                        "resourceType": "AllergyIntolerance",
                        "criticality": "high",
                        "code": {"text": "Penicillin (hives and facial swelling)"},
                    },
                ],
            },
            {
                "category": "provider-health-record",
                "method": "provider-health-record-sof",
                "source_uri": "https://fhir.multicare.example.org/r4",
                "source_name": "MultiCare Urgent Care",
                "resources": [
                    {
                        "resourceType": "MedicationRequest",
                        "status": "active",
                        "intent": "order",
                        "medicationCodeableConcept": {
                            "text": "Ibuprofen 600 mg tablet every 8 hours PRN right knee pain (walk-in Rx)"
                        },
                        "authoredOn": "2026-10-22",
                    }
                ],
            },
            {
                "category": "patient-story",
                "method": "patient-story-form",
                "source_uri": "http://localhost:4000",
                "source_name": "My Story",
                "resources": [
                    {
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
                                {
                                    "system": "http://loinc.org",
                                    "code": "51855-5",
                                    "display": "Patient Note",
                                }
                            ]
                        },
                        "effectiveDateTime": "2026-10-26",
                        "valueString": "Lightheadedness when standing in the evening and mild forearm bruising since starting urgent care ibuprofen.",
                    },
                    {
                        "resourceType": "Goal",
                        "lifecycleStatus": "active",
                        "statusDate": "2026-10-26",
                        "description": {
                            "text": "Walk 30 minutes pain-free every morning and bring A1c under 7.0%"
                        },
                        "achievementStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/goal-achievement",
                                    "code": "in-progress",
                                    "display": "In Progress",
                                }
                            ]
                        },
                    },
                ],
            },
        ],
    }


async def seed_persona_to_hapi(
    *,
    fhir_base_url: str,
    keycloak_sub: str,
    persona: str = "alex",
    bearer_token: str = "",
    extra_bundle_files: Sequence[Path] = (),
    http_client: Optional[httpx.AsyncClient] = None,
    reseed: bool = False,
) -> Dict[str, Any]:
    """Idempotently seed a multi-source WAHS star-graph (`MasterPatient` + linked datasets) into HAPI R4."""
    base = fhir_base_url.rstrip("/")
    spec = build_persona_star_graph_datasets(persona)
    demo = spec["demographics"]

    datasets = list(spec["datasets"])
    for bpath in extra_bundle_files:
        raw_bundle = json.loads(Path(bpath).read_text(encoding="utf-8"))
        patient_refs: List[str] = []
        ext_entries: List[Dict[str, Any]] = []
        for e in raw_bundle.get("entry") or []:
            if not isinstance(e, Mapping) or not isinstance(e.get("resource"), Mapping):
                continue
            res_map = e["resource"]
            rt = str(res_map.get("resourceType") or "")
            full_url = str(e.get("fullUrl") or "").strip()
            rid = str(res_map.get("id") or "").strip()
            if rt == "Patient":
                if full_url:
                    patient_refs.append(full_url)
                if rid:
                    patient_refs.append(f"Patient/{rid}")
                    patient_refs.append(f"urn:uuid:{rid}")
                continue
            ext_entries.append({"fullUrl": full_url, "resource": dict(res_map)})
        if ext_entries:
            datasets.append(
                {
                    "category": "provider-health-record",
                    "method": "provider-health-record-file",
                    "source_uri": f"urn:wahealthsummary:bundle:{Path(bpath).stem}",
                    "source_name": Path(bpath).stem.replace("_", " ").title(),
                    "entries": ext_entries,
                    "patient_refs": patient_refs,
                    "resources": [item["resource"] for item in ext_entries],
                }
            )

    headers: Dict[str, str] = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=30.0)

    try:
        # Check if MasterPatient already exists for `keycloak_sub`
        search_resp = await client.get(
            f"{base}/Patient",
            params={"identifier": f"{IDENTIFIER_SYSTEM}|{keycloak_sub}"},
            headers=headers,
        )
        search_resp.raise_for_status()
        existing_entries = search_resp.json().get("entry") or []
        existing_master = existing_entries[0]["resource"] if existing_entries else None

        existing_links: List[Dict[str, Any]] = (
            list(existing_master.get("link") or []) if existing_master else []
        )
        existing_linked_by_tuple: Dict[Tuple[str, str, str], str] = {}
        link_tuple_by_ref: Dict[str, Tuple[str, str, str]] = {}
        for lnk in existing_links:
            other = lnk.get("other") if isinstance(lnk, Mapping) else None
            ref = str((other.get("reference") if isinstance(other, Mapping) else "") or "").strip()
            if not ref.startswith("Patient/"):
                continue
            pid = ref.split("/", 1)[1].strip()
            if not pid:
                continue
            get_pat = await client.get(f"{base}/Patient/{pid}", headers=headers)
            if get_pat.status_code == 404:
                continue
            get_pat.raise_for_status()
            linked_pat = get_pat.json()
            ds_tup = _extract_dataset_tuple(linked_pat)
            if ds_tup is not None:
                existing_linked_by_tuple.setdefault(ds_tup, ref)
                link_tuple_by_ref[ref] = ds_tup

        resolved_dataset_refs: List[str] = []
        total_resources = 0

        for ds in datasets:
            ds_tuple = (ds["category"], ds["method"], ds["source_uri"])
            ds_ident_val = f"{keycloak_sub}::{ds['category']}::{ds['method']}::{ds['source_uri']}"

            if not reseed:
                existing_ref = existing_linked_by_tuple.get(ds_tuple)
                if existing_ref:
                    resolved_dataset_refs.append(existing_ref)
                    continue
                stub_resp = await client.get(
                    f"{base}/Patient",
                    params={"identifier": f"{SEED_DATASET_IDENTIFIER_SYSTEM}|{ds_ident_val}"},
                    headers=headers,
                )
                stub_resp.raise_for_status()
                stub_entries = stub_resp.json().get("entry") or []
                if stub_entries:
                    stub_id = str((stub_entries[0].get("resource") or {}).get("id") or "").strip()
                    if stub_id:
                        resolved_dataset_refs.append(f"Patient/{stub_id}")
                        continue

            pat_urn = f"urn:uuid:{uuid.uuid4()}"
            ds_patient = _stamp_security_label(
                {
                    "resourceType": "Patient",
                    "identifier": [{"system": SEED_DATASET_IDENTIFIER_SYSTEM, "value": ds_ident_val}],
                    "meta": {
                        "source": ds["source_uri"],
                        "tag": [
                            {"system": CATEGORY_SYSTEM, "code": ds["category"]},
                            {"system": METHOD_SYSTEM, "code": ds["method"]},
                            {"system": SOURCE_NAME_SYSTEM, "code": ds["source_name"]},
                        ],
                    },
                    "name": [{"given": demo["given"], "family": demo["family"]}],
                    "gender": demo["gender"],
                    "birthDate": demo["birthDate"],
                },
                keycloak_sub,
            )

            tx_entries: List[Dict[str, Any]] = [
                {
                    "fullUrl": pat_urn,
                    "resource": ds_patient,
                    "request": {"method": "POST", "url": "Patient"},
                }
            ]

            raw_entries = ds.get("entries") or [
                {"fullUrl": "", "resource": r} for r in ds["resources"]
            ]
            ref_map: Dict[str, str] = {
                p_ref: pat_urn for p_ref in (ds.get("patient_refs") or []) if p_ref
            }
            staged_entries: List[Tuple[str, Dict[str, Any]]] = []
            for entry_item in raw_entries:
                raw_res = copy.deepcopy(dict(entry_item["resource"]))
                rt = str(raw_res["resourceType"])
                orig_id = str(raw_res.pop("id", "") or "").strip()
                orig_full_url = str(entry_item.get("fullUrl") or "").strip()
                entry_urn = f"urn:uuid:{uuid.uuid4()}"
                if orig_full_url:
                    ref_map[orig_full_url] = entry_urn
                if orig_id:
                    ref_map[f"{rt}/{orig_id}"] = entry_urn
                    ref_map[f"urn:uuid:{orig_id}"] = entry_urn
                staged_entries.append((entry_urn, raw_res))

            for entry_urn, raw_res in staged_entries:
                rewritten = _rewrite_references(raw_res, ref_map)
                res = _stamp_security_label(rewritten, keycloak_sub)
                rt = str(res["resourceType"])
                if rt in _NON_PATIENT_COMPARTMENT_TYPES:
                    pass
                elif rt in {"AllergyIntolerance", "Immunization", "Claim", "ExplanationOfBenefit"} or (
                    "patient" in res and "subject" not in res
                ):
                    res["patient"] = {"reference": pat_urn}
                elif rt == "Coverage" or ("beneficiary" in res and "subject" not in res):
                    res["beneficiary"] = {"reference": pat_urn}
                else:
                    res["subject"] = {"reference": pat_urn}
                tx_entries.append(
                    {
                        "fullUrl": entry_urn,
                        "resource": res,
                        "request": {"method": "POST", "url": rt},
                    }
                )

            tx_resp = await client.post(
                base,
                headers=headers,
                json={"resourceType": "Bundle", "type": "transaction", "entry": tx_entries},
            )
            tx_resp.raise_for_status()
            tx_json = tx_resp.json()
            pat_loc = next(
                (
                    e["response"]["location"].split("/_history")[0]
                    for e in (tx_json.get("entry") or [])
                    if str(e.get("response", {}).get("location", "")).startswith("Patient/")
                ),
                None,
            )
            if not pat_loc:
                raise RuntimeError(f"HAPI transaction did not return Patient location for {ds['source_name']}")
            resolved_dataset_refs.append(pat_loc)
            total_resources += len(tx_entries) - 1

        seeded_tuples = {(ds["category"], ds["method"], ds["source_uri"]) for ds in datasets}
        merged_links: List[Dict[str, Any]] = [
            {"other": {"reference": ref}, "type": "seealso"} for ref in resolved_dataset_refs
        ]
        seen_refs = set(resolved_dataset_refs)
        for lnk in existing_links:
            other = lnk.get("other") if isinstance(lnk, Mapping) else None
            ref = str((other.get("reference") if isinstance(other, Mapping) else "") or "").strip()
            if not ref or ref in seen_refs:
                continue
            if reseed and link_tuple_by_ref.get(ref) in seeded_tuples:
                continue
            merged_links.append(copy.deepcopy(dict(lnk)))
            seen_refs.add(ref)

        master_payload = _stamp_security_label(
            {
                "resourceType": "Patient",
                "identifier": [{"system": IDENTIFIER_SYSTEM, "value": keycloak_sub}],
                "name": [{"given": demo["given"], "family": demo["family"]}],
                "gender": demo["gender"],
                "birthDate": demo["birthDate"],
                "telecom": [{"system": "email", "value": demo["email"]}],
                "link": merged_links,
            },
            keycloak_sub,
        )

        if existing_master and existing_master.get("id"):
            if existing_links == merged_links:
                saved_master = existing_master
            else:
                mid = existing_master["id"]
                master_payload["id"] = mid
                put_resp = await client.put(f"{base}/Patient/{mid}", headers=headers, json=master_payload)
                put_resp.raise_for_status()
                saved_master = put_resp.json()
        else:
            post_resp = await client.post(f"{base}/Patient", headers=headers, json=master_payload)
            post_resp.raise_for_status()
            saved_master = post_resp.json()

        return {
            "persona": persona,
            "keycloak_sub": keycloak_sub,
            "master_patient_id": saved_master.get("id"),
            "dataset_references": resolved_dataset_refs,
            "clinical_resources_seeded": total_resources,
        }
    finally:
        if owns_client:
            await client.aclose()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed multi-source WAHS FHIR R4 star-graph personas into HAPI / fhir-auth."
    )
    parser.add_argument(
        "--fhir-url",
        default="http://localhost:8080/fhir",
        help="Target HAPI R4 or fhir-auth base URL (e.g. http://fhir-internal:8080/fhir for admin seed or http://localhost:8082/fhir with --bearer-token)",
    )
    parser.add_argument(
        "--persona",
        choices=["alex", "leo", "both"],
        default="both",
        help="Which multi-source WAHS persona(s) to seed",
    )
    parser.add_argument(
        "--alex-sub",
        default="demo-user-alex",
        help="Keycloak user `sub` identifier & security label for Alex Johnson",
    )
    parser.add_argument(
        "--leo-sub",
        default="demo-user-leo",
        help="Keycloak user `sub` identifier & security label for Leo Johnson",
    )
    parser.add_argument(
        "--bearer-token",
        default="",
        help="Optional Keycloak Bearer JWT when seeding through fhir-auth instead of admin HAPI",
    )
    parser.add_argument(
        "--bundle-file",
        action="append",
        default=[],
        help="Optional extra Synthea/Connectathon FHIR Bundle JSON file(s) to attach to Alex",
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="Re-create seeded datasets instead of reusing existing datasets on MasterPatient",
    )
    args = parser.parse_args(argv)

    async def _run() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        extra_paths = [Path(p) for p in args.bundle_file]
        if args.persona in {"alex", "both"}:
            results.append(
                await seed_persona_to_hapi(
                    fhir_base_url=args.fhir_url,
                    keycloak_sub=args.alex_sub,
                    persona="alex",
                    bearer_token=args.bearer_token,
                    extra_bundle_files=extra_paths,
                    reseed=args.reseed,
                )
            )
        if args.persona in {"leo", "both"}:
            results.append(
                await seed_persona_to_hapi(
                    fhir_base_url=args.fhir_url,
                    keycloak_sub=args.leo_sub,
                    persona="leo",
                    bearer_token=args.bearer_token,
                    reseed=args.reseed,
                )
            )
        return results

    summary = asyncio.run(_run())
    print(json.dumps({"status": "seeded", "results": summary}, indent=2))


if __name__ == "__main__":
    main()
