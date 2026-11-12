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
from typing import Any, Dict, List, Mapping, Optional, Sequence
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
) -> Dict[str, Any]:
    """Idempotently seed a multi-source WAHS star-graph (`MasterPatient` + linked datasets) into HAPI R4."""
    base = fhir_base_url.rstrip("/")
    spec = build_persona_star_graph_datasets(persona)
    demo = spec["demographics"]

    datasets = list(spec["datasets"])
    for bpath in extra_bundle_files:
        raw_bundle = json.loads(Path(bpath).read_text(encoding="utf-8"))
        ext_resources = [
            e["resource"]
            for e in (raw_bundle.get("entry") or [])
            if isinstance(e, Mapping)
            and isinstance(e.get("resource"), Mapping)
            and e["resource"].get("resourceType") != "Patient"
        ]
        if ext_resources:
            datasets.append(
                {
                    "category": "provider-health-record",
                    "method": "provider-health-record-file",
                    "source_uri": f"urn:wahealthsummary:bundle:{Path(bpath).stem}",
                    "source_name": Path(bpath).stem.replace("_", " ").title(),
                    "resources": ext_resources,
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

        created_dataset_refs: List[str] = []
        total_resources = 0

        for ds in datasets:
            pat_urn = f"urn:uuid:{uuid.uuid4()}"
            ds_patient = _stamp_security_label(
                {
                    "resourceType": "Patient",
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

            for raw_res in ds["resources"]:
                res = _stamp_security_label(dict(raw_res), keycloak_sub)
                rt = str(res["resourceType"])
                if rt in {"AllergyIntolerance", "Immunization"}:
                    res["patient"] = {"reference": pat_urn}
                else:
                    res["subject"] = {"reference": pat_urn}
                tx_entries.append(
                    {
                        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
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
            created_dataset_refs.append(pat_loc)
            total_resources += len(tx_entries) - 1

        master_payload = _stamp_security_label(
            {
                "resourceType": "Patient",
                "identifier": [{"system": IDENTIFIER_SYSTEM, "value": keycloak_sub}],
                "name": [{"given": demo["given"], "family": demo["family"]}],
                "gender": demo["gender"],
                "birthDate": demo["birthDate"],
                "telecom": [{"system": "email", "value": demo["email"]}],
                "link": [{"other": {"reference": ref}, "type": "seealso"} for ref in created_dataset_refs],
            },
            keycloak_sub,
        )

        if existing_master and existing_master.get("id"):
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
            "dataset_references": created_dataset_refs,
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
                )
            )
        if args.persona in {"leo", "both"}:
            results.append(
                await seed_persona_to_hapi(
                    fhir_base_url=args.fhir_url,
                    keycloak_sub=args.leo_sub,
                    persona="leo",
                    bearer_token=args.bearer_token,
                )
            )
        return results

    summary = asyncio.run(_run())
    print(json.dumps({"status": "seeded", "results": summary}, indent=2))


if __name__ == "__main__":
    main()
