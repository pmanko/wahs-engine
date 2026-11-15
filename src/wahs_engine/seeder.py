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
DEFAULT_DEMO_PASSWORD = "wahs-demo-2026"
_NON_PATIENT_COMPARTMENT_TYPES = {
    "Medication",
    "Organization",
    "Practitioner",
    "PractitionerRole",
    "Location",
    "Provenance",
}

DEMO_ACCOUNTS: Tuple[Dict[str, Any], ...] = (
    {
        "slug": "alex",
        "keycloak_sub": "wahs-demo-alex-johnson",
        "username": "alex.johnson@wahs.local",
        "email": "alex.johnson@wahs.local",
        "given": ["Alex"],
        "family": "Johnson",
        "birthDate": "1962-03-14",
        "gender": "male",
        "bundle_file": None,
        "summary": "Adult cardiology, CKD & diabetes (4 sources)",
    },
    {
        "slug": "leo",
        "keycloak_sub": "wahs-demo-leo-johnson",
        "username": "leo.johnson@wahs.local",
        "email": "leo.johnson@wahs.local",
        "given": ["Leo"],
        "family": "Johnson",
        "birthDate": "2019-08-22",
        "gender": "male",
        "bundle_file": None,
        "summary": "Pediatric asthma & peanut allergy (3 sources)",
    },
    {
        "slug": "arthur",
        "keycloak_sub": "wahs-demo-arthur-pendelton",
        "username": "arthur.pendelton@wahs.local",
        "email": "arthur.pendelton@wahs.local",
        "given": ["Arthur"],
        "family": "Pendelton",
        "birthDate": "1947-08-19",
        "gender": "male",
        "bundle_file": "Arthur_Pendelton_Anticoag.json",
        "summary": "Anticoagulation, GI & primary care (447 resources)",
    },
    {
        "slug": "byron",
        "keycloak_sub": "wahs-demo-byron-emard",
        "username": "byron.emard@wahs.local",
        "email": "byron.emard@wahs.local",
        "given": ["Byron"],
        "family": "Emard",
        "birthDate": "2009-12-14",
        "gender": "male",
        "bundle_file": "Byron202_Emard19.json",
        "summary": "Pediatric & adolescent longitudinal care (480 resources)",
    },
    {
        "slug": "elena",
        "keycloak_sub": "wahs-demo-elena-rostova",
        "username": "elena.rostova@wahs.local",
        "email": "elena.rostova@wahs.local",
        "given": ["Elena"],
        "family": "Rostova",
        "birthDate": "1954-04-12",
        "gender": "female",
        "bundle_file": "Elena_Rostova_Cardiorenal.json",
        "summary": "UW Heart Institute + NW Kidney cardiorenal (454 resources)",
    },
    {
        "slug": "lamar",
        "keycloak_sub": "wahs-demo-lamar-beer",
        "username": "lamar.beer@wahs.local",
        "email": "lamar.beer@wahs.local",
        "given": ["Lamar"],
        "family": "Beer",
        "birthDate": "1971-02-16",
        "gender": "male",
        "bundle_file": "Lamar193_Beer512.json",
        "summary": "Outpatient & urgent care longitudinal record (892 resources)",
    },
    {
        "slug": "julian",
        "keycloak_sub": "wahs-demo-julian-mercer",
        "username": "julian.mercer@wahs.local",
        "email": "julian.mercer@wahs.local",
        "given": ["Julian"],
        "family": "Mercer",
        "birthDate": "1954-10-08",
        "gender": "male",
        "bundle_file": "MIMIC_10014354_ICU_Heme_Cardiorenal.json",
        "summary": "MIMIC-IV ICU hematology & cardiorenal (4,048 resources)",
    },
    {
        "slug": "clara",
        "keycloak_sub": "wahs-demo-clara-hollis",
        "username": "clara.hollis@wahs.local",
        "email": "clara.hollis@wahs.local",
        "given": ["Clara"],
        "family": "Hollis",
        "birthDate": "1958-12-21",
        "gender": "female",
        "bundle_file": "MIMIC_10019003_ICU_Hepatopulmonary.json",
        "summary": "MIMIC-IV ICU hepatopulmonary (2,661 resources)",
    },
    {
        "slug": "marcus",
        "keycloak_sub": "wahs-demo-marcus-vance",
        "username": "marcus.vance@wahs.local",
        "email": "marcus.vance@wahs.local",
        "given": ["Marcus"],
        "family": "Vance",
        "birthDate": "1961-09-22",
        "gender": "male",
        "bundle_file": "Marcus_Vance_Oncology_Immunotherapy.json",
        "summary": "Fred Hutch + UW thoracic oncology immunotherapy (1,612 resources)",
    },
    {
        "slug": "margaret",
        "keycloak_sub": "wahs-demo-margaret-chen",
        "username": "margaret.chen@wahs.local",
        "email": "margaret.chen@wahs.local",
        "given": ["Margaret"],
        "family": "Chen",
        "birthDate": "1943-11-03",
        "gender": "female",
        "bundle_file": "Margaret_Chen_Polypharmacy.json",
        "summary": "Geriatrics & Overlake sleep polypharmacy (448 resources)",
    },
    {
        "slug": "maurice",
        "keycloak_sub": "wahs-demo-maurice-auer",
        "username": "maurice.auer@wahs.local",
        "email": "maurice.auer@wahs.local",
        "given": ["Maurice"],
        "family": "Auer",
        "birthDate": "1973-03-25",
        "gender": "male",
        "bundle_file": "Maurice742_Auer97.json",
        "summary": "Complex multi-site longitudinal record (7,099 resources)",
    },
    {
        "slug": "sofia",
        "keycloak_sub": "wahs-demo-sofia-alvarez",
        "username": "sofia.alvarez@wahs.local",
        "email": "sofia.alvarez@wahs.local",
        "given": ["Sofia"],
        "family": "Alvarez",
        "birthDate": "1991-06-14",
        "gender": "female",
        "bundle_file": "Sofia_Alvarez_Maternal_Endocrine.json",
        "summary": "UW Maternal-Fetal & Endocrine (436 resources)",
    },
)

DEMO_ACCOUNTS_BY_SLUG: Dict[str, Dict[str, Any]] = {
    item["slug"]: item for item in DEMO_ACCOUNTS
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


def _rewrite_references(
    node: Any,
    ref_map: Mapping[str, str],
    *,
    strip_conditional_refs: bool = False,
) -> Any:
    if isinstance(node, list):
        return [_rewrite_references(item, ref_map, strip_conditional_refs=strip_conditional_refs) for item in node]
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
                if mapped is not None:
                    out[k] = mapped
                elif strip_conditional_refs and "?" in raw_ref:
                    if not node.get("display"):
                        out["display"] = raw_ref.split("?", 1)[0] or "External Reference"
                else:
                    out[k] = v
            else:
                out[k] = _rewrite_references(v, ref_map, strip_conditional_refs=strip_conditional_refs)
        return out
    return node


def _set_patient_link_field(res: Dict[str, Any], pat_ref: str) -> None:
    """Attach the dataset Patient reference using the valid FHIR R4 element name."""
    rt = str(res.get("resourceType") or "")
    if rt in _NON_PATIENT_COMPARTMENT_TYPES:
        return
    if rt in {
        "AllergyIntolerance",
        "Immunization",
        "Claim",
        "ExplanationOfBenefit",
        "Device",
        "SupplyDelivery",
    } or ("patient" in res and "subject" not in res):
        res["patient"] = {"reference": pat_ref}
    elif rt == "Coverage" or ("beneficiary" in res and "subject" not in res):
        res["beneficiary"] = {"reference": pat_ref}
    else:
        res["subject"] = {"reference": pat_ref}


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
    client = http_client or httpx.AsyncClient(timeout=120.0)

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
                rewritten = _rewrite_references(raw_res, ref_map, strip_conditional_refs=True)
                res = _stamp_security_label(rewritten, keycloak_sub)
                rt = str(res["resourceType"])
                _set_patient_link_field(res, pat_urn)
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


def _infer_resource_source_name(
    resource: Mapping[str, Any],
    enc_source_map: Mapping[str, str],
    default_source: str,
) -> str:
    """Infer a clean clinical organization/facility display name for multi-source grouping."""
    rt = str(resource.get("resourceType") or "")
    if rt == "Encounter":
        sp = resource.get("serviceProvider")
        if isinstance(sp, Mapping) and str(sp.get("display") or "").strip():
            return str(sp["display"]).strip()

    for enc_key in ("encounter", "context"):
        enc_ref = resource.get(enc_key)
        if isinstance(enc_ref, Mapping):
            ref_str = str(enc_ref.get("reference") or "").strip()
            if ref_str in enc_source_map:
                return enc_source_map[ref_str]
            if "/" in ref_str:
                parts = ref_str.rsplit("/", 2)
                if len(parts) == 3 and f"{parts[1]}/{parts[2]}" in enc_source_map:
                    return enc_source_map[f"{parts[1]}/{parts[2]}"]

    performer = resource.get("performer")
    if isinstance(performer, list):
        for p in performer:
            if not isinstance(p, Mapping):
                continue
            actor = p.get("actor")
            if isinstance(actor, Mapping) and str(actor.get("display") or "").strip():
                return str(actor["display"]).strip()
            if str(p.get("display") or "").strip():
                return str(p["display"]).strip()
    elif isinstance(performer, Mapping) and str(performer.get("display") or "").strip():
        return str(performer["display"]).strip()

    for field in ("recorder", "asserter", "requester", "custodian"):
        val = resource.get(field)
        if isinstance(val, Mapping) and str(val.get("display") or "").strip():
            return str(val["display"]).strip()

    authors = resource.get("author")
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, Mapping) and str(a.get("display") or "").strip():
                return str(a["display"]).strip()

    agents = resource.get("agent")
    if isinstance(agents, list):
        for ag in agents:
            who = ag.get("who") if isinstance(ag, Mapping) else None
            if isinstance(who, Mapping) and str(who.get("display") or "").strip():
                return str(who["display"]).strip()

    return default_source


def _sort_key_for_tx(entry_res: Mapping[str, Any]) -> Tuple[int, str]:
    rt = str(entry_res.get("resourceType") or "")
    order = {
        "Patient": 0,
        "Organization": 1,
        "Practitioner": 2,
        "PractitionerRole": 3,
        "Location": 4,
        "Medication": 5,
        "Encounter": 10,
        "Condition": 20,
        "AllergyIntolerance": 21,
        "MedicationRequest": 22,
        "MedicationAdministration": 23,
        "Immunization": 24,
        "Procedure": 25,
        "Device": 26,
        "CarePlan": 27,
        "CareTeam": 28,
        "Observation": 30,
        "DiagnosticReport": 35,
        "DocumentReference": 36,
        "ImagingStudy": 37,
        "Claim": 40,
        "ExplanationOfBenefit": 41,
        "SupplyDelivery": 42,
        "Provenance": 50,
    }
    return (order.get(rt, 99), str(entry_res.get("id") or ""))


def build_bundle_star_graph_datasets(
    bundle_path: Path,
    account_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Partition an arbitrary clinical FHIR Bundle into multi-source WAHS datasets with deterministic IDs."""
    raw_bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    slug = str(account_meta["slug"]).lower()
    raw_entries = [
        e for e in (raw_bundle.get("entry") or [])
        if isinstance(e, Mapping) and isinstance(e.get("resource"), Mapping)
    ]

    orig_patient_refs: List[str] = []
    source_freq: Dict[str, int] = {}
    enc_source_map: Dict[str, str] = {}
    fallback_org = ""

    for e in raw_entries:
        r = e["resource"]
        rt = str(r.get("resourceType") or "")
        rid = str(r.get("id") or "").strip()
        full_url = str(e.get("fullUrl") or "").strip()
        if rt == "Patient":
            if full_url:
                orig_patient_refs.append(full_url)
            if rid:
                orig_patient_refs.append(f"Patient/{rid}")
                orig_patient_refs.append(f"urn:uuid:{rid}")
        elif rt == "Organization" and not fallback_org:
            fallback_org = str(r.get("name") or "").strip()
        elif rt == "Encounter":
            sp = r.get("serviceProvider")
            sp_disp = str(sp.get("display") or "").strip() if isinstance(sp, Mapping) else ""
            if sp_disp:
                source_freq[sp_disp] = source_freq.get(sp_disp, 0) + 1
                if full_url:
                    enc_source_map[full_url] = sp_disp
                if rid:
                    enc_source_map[f"Encounter/{rid}"] = sp_disp
                    enc_source_map[f"urn:uuid:{rid}"] = sp_disp

    if source_freq:
        primary_source = max(source_freq.items(), key=lambda kv: kv[1])[0]
    elif fallback_org:
        primary_source = fallback_org
    else:
        primary_source = Path(bundle_path).stem.replace("_", " ").title()

    global_ref_map: Dict[str, str] = {}
    shared_raw: List[Dict[str, Any]] = []
    grouped_raw: Dict[str, List[Dict[str, Any]]] = {}
    seq = 0

    for e in raw_entries:
        orig_res = copy.deepcopy(dict(e["resource"]))
        rt = str(orig_res.get("resourceType") or "")
        if rt == "Patient":
            continue
        seq += 1
        orig_id = str(orig_res.pop("id", "") or "").strip()
        orig_full_url = str(e.get("fullUrl") or "").strip()
        new_id = f"w-{slug[:10]}-{seq}"
        target_ref = f"{rt}/{new_id}"
        orig_res["id"] = new_id
        if orig_full_url:
            global_ref_map[orig_full_url] = target_ref
        if orig_id:
            global_ref_map[f"{rt}/{orig_id}"] = target_ref
            global_ref_map[f"urn:uuid:{orig_id}"] = target_ref

        if rt in _NON_PATIENT_COMPARTMENT_TYPES:
            shared_raw.append(orig_res)
        else:
            src_name = _infer_resource_source_name(orig_res, enc_source_map, primary_source)
            grouped_raw.setdefault(src_name, []).append(orig_res)

    # Keep primary_source first, then largest clinical sources
    ordered_sources = sorted(
        grouped_raw.items(),
        key=lambda kv: (0 if kv[0] == primary_source else 1, -len(kv[1]), kv[0]),
    )
    primary_ds_pat_ref = (
        f"Patient/w-{slug[:10]}-ds-1" if ordered_sources else f"Patient/wahs-master-{slug}"
    )
    shared_ref_map = dict(global_ref_map)
    for pref in orig_patient_refs:
        shared_ref_map[pref] = primary_ds_pat_ref

    shared_entries: List[Dict[str, Any]] = []
    for raw_res in shared_raw:
        rewritten = _rewrite_references(raw_res, shared_ref_map, strip_conditional_refs=True)
        shared_entries.append(rewritten)

    datasets: List[Dict[str, Any]] = []
    for src_idx, (src_name, res_list) in enumerate(ordered_sources, start=1):
        ds_pat_id = f"w-{slug[:10]}-ds-{src_idx}"
        ds_pat_ref = f"Patient/{ds_pat_id}"
        ds_ref_map = dict(global_ref_map)
        for pref in orig_patient_refs:
            ds_ref_map[pref] = ds_pat_ref

        slug_src = "".join(ch.lower() if ch.isalnum() else "-" for ch in src_name).strip("-")[:40]
        source_uri = f"urn:wahealthsummary:bundle:{slug}:{slug_src or src_idx}"
        converted_resources: List[Dict[str, Any]] = []
        for raw_res in sorted(res_list, key=_sort_key_for_tx):
            rewritten = _rewrite_references(raw_res, ds_ref_map, strip_conditional_refs=True)
            _set_patient_link_field(rewritten, ds_pat_ref)
            converted_resources.append(rewritten)

        datasets.append(
            {
                "dataset_patient_id": ds_pat_id,
                "category": "provider-health-record",
                "method": "provider-health-record-file",
                "source_uri": source_uri,
                "source_name": src_name,
                "resources": converted_resources,
            }
        )

    return {
        "demographics": {
            "given": list(account_meta["given"]),
            "family": str(account_meta["family"]),
            "gender": str(account_meta["gender"]),
            "birthDate": str(account_meta["birthDate"]),
            "email": str(account_meta["email"]),
        },
        "shared_resources": shared_entries,
        "datasets": datasets,
    }


async def _post_put_batches(
    client: httpx.AsyncClient,
    base: str,
    headers: Mapping[str, str],
    resources: Sequence[Dict[str, Any]],
    *,
    keycloak_sub: str,
    batch_size: int = 200,
) -> int:
    total = 0
    step = max(1, int(batch_size))
    for start in range(0, len(resources), step):
        chunk = resources[start : start + step]
        entries = []
        for raw_res in chunk:
            stamped = _stamp_security_label(raw_res, keycloak_sub)
            rt = str(stamped["resourceType"])
            rid = str(stamped["id"])
            entries.append(
                {
                    "fullUrl": f"{rt}/{rid}",
                    "resource": stamped,
                    "request": {"method": "PUT", "url": f"{rt}/{rid}"},
                }
            )
        resp = await client.post(
            base,
            headers=dict(headers),
            json={"resourceType": "Bundle", "type": "transaction", "entry": entries},
        )
        resp.raise_for_status()
        total += len(chunk)
    return total


async def seed_bundle_account_to_hapi(
    *,
    fhir_base_url: str,
    account_meta: Mapping[str, Any],
    bundle_path: Path,
    keycloak_sub: Optional[str] = None,
    bearer_token: str = "",
    http_client: Optional[httpx.AsyncClient] = None,
    reseed: bool = False,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Seed one bundle-backed clinical persona into HAPI using batched PUT transactions."""
    base = fhir_base_url.rstrip("/")
    slug = str(account_meta["slug"]).lower()
    sub = str(keycloak_sub or account_meta["keycloak_sub"]).strip()
    headers: Dict[str, str] = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=120.0)
    try:
        search_resp = await client.get(
            f"{base}/Patient",
            params={"identifier": f"{IDENTIFIER_SYSTEM}|{sub}"},
            headers=headers,
        )
        search_resp.raise_for_status()
        existing_entries = search_resp.json().get("entry") or []
        existing_master = existing_entries[0]["resource"] if existing_entries else None
        if not reseed and existing_master and (existing_master.get("link") or []):
            refs = [
                str(((lnk.get("other") or {}).get("reference")) or "")
                for lnk in existing_master.get("link") or []
                if isinstance(lnk, Mapping)
            ]
            return {
                "persona": slug,
                "keycloak_sub": sub,
                "master_patient_id": existing_master.get("id"),
                "dataset_references": [r for r in refs if r],
                "clinical_resources_seeded": 0,
                "skipped": True,
            }

        plan = build_bundle_star_graph_datasets(bundle_path, account_meta)
        demo = plan["demographics"]
        total_seeded = 0

        if plan["shared_resources"]:
            non_prov_shared = [
                r for r in plan["shared_resources"] if r.get("resourceType") != "Provenance"
            ]
            if non_prov_shared:
                total_seeded += await _post_put_batches(
                    client,
                    base,
                    headers,
                    non_prov_shared,
                    keycloak_sub=sub,
                    batch_size=batch_size,
                )

        dataset_refs: List[str] = []
        for ds in plan["datasets"]:
            ds_pat_id = ds["dataset_patient_id"]
            ds_ident_val = f"{sub}::{ds['category']}::{ds['method']}::{ds['source_uri']}"
            ds_patient = {
                "resourceType": "Patient",
                "id": ds_pat_id,
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
            }
            # Seed Patient first with its clinical resources in chunks
            ds_batch_items = [ds_patient, *ds["resources"]]
            seeded_here = await _post_put_batches(
                client,
                base,
                headers,
                ds_batch_items,
                keycloak_sub=sub,
                batch_size=batch_size,
            )
            total_seeded += max(0, seeded_here - 1)
            dataset_refs.append(f"Patient/{ds_pat_id}")

        prov_shared = [
            r for r in (plan["shared_resources"] or []) if r.get("resourceType") == "Provenance"
        ]
        if prov_shared:
            total_seeded += await _post_put_batches(
                client,
                base,
                headers,
                prov_shared,
                keycloak_sub=sub,
                batch_size=batch_size,
            )

        master_id = (
            str(existing_master.get("id") or "")
            if isinstance(existing_master, Mapping)
            else ""
        ) or f"wahs-master-{slug}"
        master_payload = _stamp_security_label(
            {
                "resourceType": "Patient",
                "id": master_id,
                "identifier": [{"system": IDENTIFIER_SYSTEM, "value": sub}],
                "name": [{"given": demo["given"], "family": demo["family"]}],
                "gender": demo["gender"],
                "birthDate": demo["birthDate"],
                "telecom": [{"system": "email", "value": demo["email"]}],
                "link": [{"other": {"reference": ref}, "type": "seealso"} for ref in dataset_refs],
            },
            sub,
        )
        put_master = await client.put(
            f"{base}/Patient/{master_id}", headers=headers, json=master_payload
        )
        put_master.raise_for_status()
        saved_master = put_master.json()

        return {
            "persona": slug,
            "keycloak_sub": sub,
            "master_patient_id": saved_master.get("id") or master_id,
            "dataset_references": dataset_refs,
            "clinical_resources_seeded": total_seeded,
            "skipped": False,
        }
    finally:
        if owns_client:
            await client.aclose()


async def ensure_keycloak_demo_users(
    *,
    keycloak_base_url: str,
    realm: str = "wahs",
    admin_username: str = "admin",
    admin_password: str = "admin",
    accounts: Sequence[Mapping[str, Any]] = DEMO_ACCOUNTS,
    demo_password: str = DEFAULT_DEMO_PASSWORD,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, str]:
    """Ensure all demo accounts exist in Keycloak and return `{slug: actual_keycloak_sub}`."""
    kc_base = keycloak_base_url.rstrip("/")
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=30.0)
    try:
        tok_resp = await client.post(
            f"{kc_base}/realms/master/protocol/openid-connect/token",
            data={
                "client_id": "admin-cli",
                "username": admin_username,
                "password": admin_password,
                "grant_type": "password",
            },
        )
        tok_resp.raise_for_status()
        admin_token = tok_resp.json()["access_token"]
        kc_headers = {
            "Authorization": f"Bearer {admin_token}",
            "Content-Type": "application/json",
        }

        resolved_subs: Dict[str, str] = {}
        for acct in accounts:
            slug = str(acct["slug"]).lower()
            username = str(acct["username"])
            email = str(acct["email"])
            search_resp = await client.get(
                f"{kc_base}/admin/realms/{realm}/users",
                params={"username": username, "exact": "true"},
                headers=kc_headers,
            )
            search_resp.raise_for_status()
            matches = search_resp.json() or []
            user_id = str(matches[0]["id"]) if matches else ""

            if not user_id:
                raw_gender = str(acct.get("gender") or "").strip().lower()
                kc_gender = {
                    "female": "Female",
                    "male": "Male",
                    "other": "Non-Binary",
                }.get(raw_gender, "Decline to State")
                create_payload = {
                    "username": username,
                    "email": email,
                    "firstName": " ".join(acct.get("given") or [slug.title()]),
                    "lastName": str(acct.get("family") or ""),
                    "enabled": True,
                    "emailVerified": True,
                    "requiredActions": [],
                    "attributes": {
                        "birthday": [str(acct.get("birthDate") or "1970-01-01")],
                        "gender": [kc_gender],
                        "terms_and_conditions": ["1759276801"],
                    },
                    "credentials": [
                        {
                            "type": "password",
                            "value": demo_password,
                            "temporary": False,
                        }
                    ],
                }
                cr = await client.post(
                    f"{kc_base}/admin/realms/{realm}/users",
                    headers=kc_headers,
                    json=create_payload,
                )
                if cr.status_code not in (200, 201, 204, 409):
                    cr.raise_for_status()
                re_resp = await client.get(
                    f"{kc_base}/admin/realms/{realm}/users",
                    params={"username": username, "exact": "true"},
                    headers=kc_headers,
                )
                re_resp.raise_for_status()
                re_matches = re_resp.json() or []
                if not re_matches:
                    raise RuntimeError(f"Keycloak failed to create or locate demo user {username}")
                user_id = str(re_matches[0]["id"])
            resolved_subs[slug] = user_id

        return resolved_subs
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
        choices=["alex", "leo", "both", "all", *[a["slug"] for a in DEMO_ACCOUNTS if a["slug"] not in {"alex", "leo"}]],
        default="both",
        help="Which multi-source WAHS persona(s) or 'all' 12 demo profiles to seed",
    )
    parser.add_argument(
        "--alex-sub",
        default="",
        help="Override Keycloak user `sub` identifier & security label for Alex Johnson",
    )
    parser.add_argument(
        "--leo-sub",
        default="",
        help="Override Keycloak user `sub` identifier & security label for Leo Johnson",
    )
    parser.add_argument(
        "--bundles-dir",
        default="",
        help="Directory containing the 10 clinical FHIR JSON bundles (e.g. /bundles)",
    )
    parser.add_argument(
        "--keycloak-url",
        default="",
        help="Optional Keycloak base URL (e.g. http://keycloak:8080) to ensure & resolve real demo user `sub` IDs",
    )
    parser.add_argument(
        "--keycloak-realm",
        default="wahs",
        help="Keycloak realm name (default: wahs)",
    )
    parser.add_argument(
        "--keycloak-admin",
        default="admin",
        help="Keycloak admin username (default: admin)",
    )
    parser.add_argument(
        "--keycloak-admin-password",
        default="admin",
        help="Keycloak admin password (default: admin)",
    )
    parser.add_argument(
        "--demo-password",
        default=DEFAULT_DEMO_PASSWORD,
        help="Password assigned when auto-creating missing Keycloak demo users",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Maximum entries per HAPI transaction Bundle batch (default: 200)",
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
        bundles_root = Path(args.bundles_dir) if args.bundles_dir else None

        if args.persona == "all":
            target_accounts = list(DEMO_ACCOUNTS)
        elif args.persona == "both":
            target_accounts = [DEMO_ACCOUNTS_BY_SLUG["alex"], DEMO_ACCOUNTS_BY_SLUG["leo"]]
        else:
            target_accounts = [DEMO_ACCOUNTS_BY_SLUG[args.persona]]

        kc_subs: Dict[str, str] = {}
        if args.keycloak_url:
            kc_subs = await ensure_keycloak_demo_users(
                keycloak_base_url=args.keycloak_url,
                realm=args.keycloak_realm,
                admin_username=args.keycloak_admin,
                admin_password=args.keycloak_admin_password,
                accounts=target_accounts,
                demo_password=args.demo_password,
            )

        async with httpx.AsyncClient(timeout=120.0) as client:
            for acct in target_accounts:
                slug = str(acct["slug"]).lower()
                if slug == "alex":
                    sub = args.alex_sub or kc_subs.get("alex") or (
                        "demo-user-alex" if args.persona in {"alex", "both"} and not args.keycloak_url else acct["keycloak_sub"]
                    )
                    results.append(
                        await seed_persona_to_hapi(
                            fhir_base_url=args.fhir_url,
                            keycloak_sub=sub,
                            persona="alex",
                            bearer_token=args.bearer_token,
                            extra_bundle_files=extra_paths,
                            http_client=client,
                            reseed=args.reseed,
                        )
                    )
                elif slug == "leo":
                    sub = args.leo_sub or kc_subs.get("leo") or (
                        "demo-user-leo" if args.persona in {"leo", "both"} and not args.keycloak_url else acct["keycloak_sub"]
                    )
                    results.append(
                        await seed_persona_to_hapi(
                            fhir_base_url=args.fhir_url,
                            keycloak_sub=sub,
                            persona="leo",
                            bearer_token=args.bearer_token,
                            http_client=client,
                            reseed=args.reseed,
                        )
                    )
                elif acct.get("bundle_file"):
                    if bundles_root is None:
                        continue
                    bpath = bundles_root / str(acct["bundle_file"])
                    if not bpath.exists():
                        results.append(
                            {
                                "persona": slug,
                                "keycloak_sub": kc_subs.get(slug) or acct["keycloak_sub"],
                                "skipped": True,
                                "reason": f"bundle_not_found: {bpath}",
                            }
                        )
                        continue
                    sub = kc_subs.get(slug) or str(acct["keycloak_sub"])
                    results.append(
                        await seed_bundle_account_to_hapi(
                            fhir_base_url=args.fhir_url,
                            account_meta=acct,
                            bundle_path=bpath,
                            keycloak_sub=sub,
                            bearer_token=args.bearer_token,
                            http_client=client,
                            reseed=args.reseed,
                            batch_size=args.batch_size,
                        )
                    )
        return results

    summary = asyncio.run(_run())
    print(json.dumps({"status": "seeded", "results": summary}, indent=2))


if __name__ == "__main__":
    main()
