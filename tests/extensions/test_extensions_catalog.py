"""Contract and behavior tests for unplugged `wahs_extensions` catalog modules."""

from __future__ import annotations

import base64
from typing import Any, Dict

import pytest

from wahs_core import ContextEnricher, condense_bundle_to_budget, estimate_tokens_local, extract_bundle
from wahs_core.ports import EnrichmentInput
from wahs_extensions import list_available_enrichers, load_enricher
from wahs_extensions.overflow_lookup import get_lab_history, read_clinical_note, search_chart_entries


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _extension_test_bundle() -> Dict[str, Any]:
    entries = [
        {"resource": {"resourceType": "Patient", "id": "pat-ext", "name": [{"given": ["Jordan"], "family": "Lee"}], "birthDate": "1958-02-10"}},
        {"resource": {"resourceType": "Condition", "id": "c-ckd2", "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": "2022-01-01", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "431856006", "display": "Chronic kidney disease stage 2"}]}}},
        {"resource": {"resourceType": "Condition", "id": "c-ckd3", "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": "2024-06-01", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "433144002", "display": "Chronic kidney disease stage 3"}]}}},
        {"resource": {"resourceType": "MedicationRequest", "id": "m-warfarin", "status": "active", "authoredOn": "2025-05-01", "medicationCodeableConcept": {"text": "Warfarin 5 MG Oral Tablet"}}},
        {"resource": {"resourceType": "MedicationRequest", "id": "m-ibuprofen", "status": "active", "authoredOn": "2025-05-02", "medicationCodeableConcept": {"text": "Ibuprofen 400 MG Oral Tablet"}}},
        {"resource": {"resourceType": "Observation", "id": "obs-cr-1", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2024-01-10", "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]}, "valueQuantity": {"value": 1.2, "unit": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "obs-cr-2", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2025-01-10", "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]}, "valueQuantity": {"value": 1.8, "unit": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "obs-cmp", "status": "final", "category": [{"coding": [{"code": "laboratory"}]}], "effectiveDateTime": "2025-01-10", "code": {"text": "Microalbumin/Creatinine"}, "valueQuantity": {"value": 300, "comparator": ">", "unit": "mg/g"}}},
    ]
    for i in range(8):
        note_body = (
            f"Discharge Summary #{i}: cardiology clinic transition note. "
            + ("Patient advised regarding anticoagulation monitoring and dietary counseling. " * 25)
            + ("STOP ibuprofen immediately due to bleeding and renal hazard." if i == 0 else "Routine follow-up.")
        )
        entries.append(
            {
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": f"doc-ext-{i}",
                    "status": "current",
                    "date": f"2025-0{i+1}-15",
                    "type": {"text": f"Summary #{i}"},
                    "content": [{"attachment": {"data": _b64(note_body)}}],
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_all_enrichers_implement_protocol_and_emit_disclosures() -> None:
    bundle = _extension_test_bundle()
    extracted = extract_bundle(bundle)
    condensed = condense_bundle_to_budget(extracted, target_tokens=1024, token_counter=estimate_tokens_local)
    cit_map = {str(m["index"]): dict(m, id=m["resourceUuid"]) for m in condensed.mappings}

    for eid in list_available_enrichers():
        enricher = load_enricher(eid)
        assert isinstance(enricher, ContextEnricher)
        assert enricher.disclosure.enricher_id == eid
        out = enricher.enrich(
            EnrichmentInput(
                extracted=extracted,
                condensed=condensed,
                citation_map=cit_map,
                question="Check my creatinine trend, ibuprofen anticoagulation hazard, and older discharge summary",
                conversation=[],
                max_tokens=512,
                token_counter=estimate_tokens_local,
            )
        )
        if eid != "overflow_lookup" or condensed.completeness_manifest["omitted_clinical_items_count"] > 0:
            assert out is not None, eid
            assert out.token_count <= 512
            assert out.prompt_block

    exp = load_enricher("experimental.clinical_reconciliation")
    assert exp.disclosure.kind == "experimental_clinical_rules"
    assert "prototype" in exp.disclosure.limitations.lower()

    with pytest.raises(KeyError):
        load_enricher("nonexistent_enricher")


def test_overflow_read_helpers_work_cleanly_without_privacy_couplings() -> None:
    extracted = extract_bundle(_extension_test_bundle())
    s = search_chart_entries(extracted, "Creatinine")
    assert s["total_matches"] >= 2
    h = get_lab_history(extracted, "Creatinine")
    assert h["summary"]["latest"]["value"] == 1.8
    n = read_clinical_note(extracted, query="ibuprofen")
    assert n["total_matches"] == 1
