"""Unit tests for pure-stdlib `wahs_core` extraction and 3-tier verbatim-first condensation."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from wahs_core import condense_bundle_to_budget, estimate_tokens_local, extract_bundle


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _sample_outpatient_bundle() -> Dict[str, Any]:
    note_txt = "Patient reports mild ankle swelling in evenings. Continue lisinopril 10 mg daily and follow low-sodium diet."
    note_b64 = _b64(note_txt)
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-1", "name": [{"given": ["Aria"], "family": "Ng"}], "birthDate": "1974-08-21", "gender": "female"}},
            {"resource": {"resourceType": "AllergyIntolerance", "id": "alg-1", "clinicalStatus": {"coding": [{"code": "active"}]}, "code": {"text": "Penicillin"}}},
            {"resource": {"resourceType": "Condition", "id": "cond-act", "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": "2021-04-02", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertension"}]}}},
            {"resource": {"resourceType": "Condition", "id": "cond-err", "clinicalStatus": {"coding": [{"code": "entered-in-error"}]}, "code": {"text": "Erroneous diagnosis"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "med-1", "status": "active", "authoredOn": "2026-09-01", "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "314076", "display": "Lisinopril 10 MG Oral Tablet"}]}}},
            {"resource": {"resourceType": "Observation", "id": "obs-cre", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2026-10-15", "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]}, "valueQuantity": {"value": 1.1, "unit": "mg/dL"}}},
            {"resource": {"resourceType": "DocumentReference", "id": "doc-1", "status": "current", "date": "2026-10-16", "type": {"text": "Progress Note"}, "content": [{"attachment": {"data": note_b64}}]}},
            {"resource": {"resourceType": "DiagnosticReport", "id": "dr-dup", "status": "final", "effectiveDateTime": "2026-10-16", "code": {"text": "Progress Note"}, "presentedForm": [{"data": note_b64}]}},
        ],
    }


def _large_longitudinal_bundle(num_labs: int = 160, num_notes: int = 10) -> Dict[str, Any]:
    entries = list(_sample_outpatient_bundle()["entry"])
    for i in range(num_labs):
        yr = 2016 + (i // 16)
        mo = 1 + (i % 12)
        day = 1 + (i % 25)
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-hist-{i:03d}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"{yr:04d}-{mo:02d}-{day:02d}",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]},
                    "valueQuantity": {"value": round(0.9 + (i % 8) * 0.05, 2), "unit": "mg/dL"},
                }
            }
        )
    for j in range(num_notes):
        txt = f"Longitudinal clinic visit summary #{j}. " + (
            "Discussed home blood pressure readings, diet, exercise, and kidney monitoring. " * 22
        )
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-long-{j:03d}",
                    "status": "current",
                    "date": f"2026-0{(j % 9) + 1}-12",
                    "type": {"text": f"Visit Note #{j}"},
                    "content": [{"attachment": {"data": _b64(txt)}}],
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_pure_stdlib_no_site_or_extensions() -> None:
    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.run(
        [sys.executable, "-S", "-c", "import sys, wahs_core; assert 'site' not in sys.modules and 'wahs_extensions' not in sys.modules"],
        env={"PYTHONPATH": src_dir},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_tier0_unbounded_default_and_verbatim_fidelity() -> None:
    bundle = _sample_outpatient_bundle()
    extracted = extract_bundle(bundle)
    assert extracted.deduplicated_note_count == 1
    assert len(extracted.notes) == 1
    assert "Erroneous diagnosis" not in " ".join(i.text for i in extracted.items)

    res = condense_bundle_to_budget(bundle)
    man = res.completeness_manifest
    assert res.tier == 0
    assert res.target_tokens is None
    assert res.compression_stats["budget_utilization"] == 1.0
    assert man["folded_items_count"] == 0
    assert man["omitted_clinical_items_count"] == 0
    assert man["excerpted_notes_count"] == 0
    assert "Continue lisinopril 10 mg daily and follow low-sodium diet." in res.chart_text


def test_tier2_continuous_note_waterfilling_fills_available_budget() -> None:
    # Single large note should dynamically expand to fill arbitrary budgets rather than chopping at 400 chars
    clauses = [f"Clinical hospital course detail #{idx:03d}: stable electrolytes and telemetry without arrhythmia." for idx in range(220)]
    long_note = " ".join(clauses)
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pat-wf", "name": [{"given": ["Nora"], "family": "Vance"}], "birthDate": "1968-02-11"}},
            {"resource": {"resourceType": "Condition", "id": "cond-wf", "clinicalStatus": {"coding": [{"code": "active"}]}, "code": {"text": "Atrial fibrillation"}}},
            {"resource": {"resourceType": "DocumentReference", "id": "doc-wf", "status": "current", "date": "2026-10-01", "type": {"text": "Discharge Summary"}, "content": [{"attachment": {"data": _b64(long_note)}}]}},
        ],
    }
    prev_len = 0
    for b in (900, 1800, 3600):
        res = condense_bundle_to_budget(bundle, target_tokens=b)
        assert res.tier == 2
        assert res.actual_tokens <= b
        # Water-filling should use nearly all of each budget (>= 80%) rather than a fixed ~400-char excerpt
        assert res.actual_tokens >= int(b * 0.80)
        assert len(res.chart_text) > prev_len
        prev_len = len(res.chart_text)


def test_monotonic_rle_and_budget_packing_across_budgets() -> None:
    bundle = _large_longitudinal_bundle()
    budgets = (1500, 3000, 6000, 12000, 24000)
    covered_sets: List[set[str]] = []

    for b in budgets:
        res = condense_bundle_to_budget(bundle, target_tokens=b)
        assert res.actual_tokens <= b
        uuids: set[str] = set()
        for m in res.mappings:
            assert isinstance(m["index"], int) and m["resourceUuid"]
            uuids.add(str(m["resourceUuid"]))
            uuids.update(str(x) for x in m.get("memberUuids") or [])
        covered_sets.append(uuids)
        # IPS Safety Core stays intact across every budget
        assert "Penicillin" in res.chart_text
        assert "Hypertension" in res.chart_text
        assert "Lisinopril" in res.chart_text

    for i in range(len(covered_sets) - 1):
        assert covered_sets[i].issubset(covered_sets[i + 1])

    res_tight = condense_bundle_to_budget(bundle, target_tokens=1500)
    assert "[... excerpted;" in res_tight.chart_text
    res_wide = condense_bundle_to_budget(bundle, target_tokens=12000)
    assert res_wide.tier in {0, 1}
    assert res_wide.completeness_manifest["omitted_clinical_items_count"] == 0
    assert estimate_tokens_local(res_wide.chart_text) > 0


def test_codings_openmrs_ciel_immunization_dose_status_and_note_horizons() -> None:
    long_note = "Cardiology follow-up visit summary. " + ("Stable blood pressure and electrolytes. " * 60)
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-core-hooks", "name": [{"given": ["Lin"], "family": "Tran"}]}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "79651AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "status": "active",
                    "authoredOn": "2026-09-10",
                    "medicationCodeableConcept": {
                        "coding": [
                            {"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "860975", "display": "Metformin 500 MG Oral Tablet"},
                            {"system": "http://www.whocc.no/atc", "code": "A10BA02", "display": "metformin"},
                        ]
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ciel-explicit",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"coding": [{"system": "https://openconceptlab.org/orgs/CIEL/sources/CIEL", "code": "142473", "display": "Type 2 diabetes mellitus"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Immunization",
                    "id": "imm-1",
                    "status": "completed",
                    "occurrenceDateTime": "2026-04-01",
                    "vaccineCode": {"coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": "187", "display": "Shingrix"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Immunization",
                    "id": "imm-2",
                    "status": "completed",
                    "occurrenceDateTime": "2026-08-01",
                    "vaccineCode": {"coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": "187", "display": "Shingrix"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "note-h1",
                    "status": "current",
                    "date": "2026-10-01",
                    "type": {"text": "Clinic Note"},
                    "content": [{"attachment": {"data": _b64(long_note)}}],
                }
            },
        ],
    }
    extracted = extract_bundle(bundle)
    med_item = next(x for x in extracted.items if x.resource_type == "MedicationRequest")
    med_systems = {(c["system"], c["code"]) for c in med_item.metadata["codings"]}
    assert ("rxnorm", "860975") in med_systems
    assert ("atc", "A10BA02") in med_systems
    assert ("ciel", "79651") in med_systems

    imm_items = [x for x in extracted.items if x.resource_type == "Immunization"]
    assert all(x.status == "completed" for x in imm_items)
    assert all("dose_status: administered" in x.text and "status: completed" not in x.text for x in imm_items)

    t0 = condense_bundle_to_budget(extracted)
    assert t0.tier == 0
    assert "dose_status: administered" in t0.chart_text
    assert "status: completed" not in t0.chart_text
    full_note_len = len(extracted.notes[0].text)
    assert t0.compression_stats["rendered_note_horizons"]["note-h1"] == full_note_len

    # Tier 2 budget: note is excerpted and RLE folds the 2 Shingrix doses with dose_status: administered
    t2 = condense_bundle_to_budget(extracted, target_tokens=320)
    assert t2.tier == 2
    assert "2 doses" in t2.chart_text
    assert "dose_status: administered" in t2.chart_text
    assert "status: completed" not in t2.chart_text
    h2 = t2.compression_stats["rendered_note_horizons"]["note-h1"]
    assert 0 <= h2 < full_note_len
