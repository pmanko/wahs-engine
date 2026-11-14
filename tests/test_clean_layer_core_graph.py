"""Unit tests for wahs_core clean-layer graph rules, WE7-1 cross-reactivity, and reference_date scoping."""

from __future__ import annotations

import json
import subprocess
import sys

from wahs_core import (
    ClinicalEntityGraph,
    build_wahs_hub_enrichment_block,
    condense_bundle_to_budget,
    extract_bundle,
    get_clinical_graph_rules,
    reset_kb_lookup_cache,
)


def _penicillin_allergy_bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "name": [{"use": "official", "given": ["Jane"], "family": "Doe"}],
                    "birthDate": "1972-04-12",
                    "gender": "female",
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-pen-1",
                    "clinicalStatus": {
                        "coding": [{"code": "active", "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"}]
                    },
                    "code": {
                        "coding": [{"code": "91936005", "display": "Allergy to penicillin (finding)"}],
                        "text": "Penicillin allergy",
                    },
                    "recordedDate": "2020-05-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ckd3",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [{"code": "433144002", "display": "Chronic kidney disease stage 3"}],
                        "text": "Chronic kidney disease stage 3",
                    },
                    "recordedDate": "2025-11-10",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-amox-1",
                    "status": "active",
                    "authoredOn": "2026-03-01",
                    "medicationCodeableConcept": {
                        "coding": [{"code": "723", "display": "Amoxicillin 500 MG Oral Capsule"}],
                        "text": "Amoxicillin 500 MG Oral Capsule",
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-cefaz-1",
                    "status": "active",
                    "authoredOn": "2026-03-02",
                    "medicationCodeableConcept": {
                        "coding": [{"code": "2180", "display": "Cefazolin 1000 MG Injection"}],
                        "text": "Cefazolin 1000 MG Injection",
                    },
                }
            },
        ],
    }


def test_cross_reacts_with_edges_from_rxnorm_lookup_rules() -> None:
    bundle = _penicillin_allergy_bundle()
    extracted = extract_bundle(bundle)
    graph = ClinicalEntityGraph.from_extracted_chart(extracted)
    cross_edges = [e for e in graph.edges if e.edge_type == "CROSS_REACTS_WITH"]
    assert len(cross_edges) >= 2
    targets = {graph.nodes[e.target_id].label.lower() for e in cross_edges}
    assert any("amoxicillin" in t for t in targets)
    assert any("cefazolin" in t for t in targets)
    assert all(e.kb_source and "RxNorm" not in e.kb_source or e.kb_source for e in cross_edges)
    assert any("AAAAI" in e.kb_source for e in cross_edges)


def test_clinical_graph_rules_pack_and_cache_reset() -> None:
    reset_kb_lookup_cache()
    rules = get_clinical_graph_rules()
    assert isinstance(rules.get("drug_indication_rules"), list) and len(rules["drug_indication_rules"]) >= 10
    assert isinstance(rules.get("safety_contraindication_rules"), list) and len(rules["safety_contraindication_rules"]) >= 6
    assert isinstance(rules.get("monitoring_rules"), list) and len(rules["monitoring_rules"]) >= 8
    assert isinstance(rules.get("note_hold_or_mod_cues"), list) and "held" in rules["note_hold_or_mod_cues"]
    assert isinstance(rules.get("note_resume_or_continue_cues"), list) and "resume " in rules["note_resume_or_continue_cues"]
    assert rules.get("kb_query_synonyms", {}).get("medicine") == ["medication"]
    reset_kb_lookup_cache()
    rules2 = get_clinical_graph_rules()
    assert json.dumps(rules, sort_keys=True) == json.dumps(rules2, sort_keys=True)


def test_hub_enrichment_result_uses_snake_case_telemetry_fields() -> None:
    bundle = _penicillin_allergy_bundle()
    extracted = extract_bundle(bundle)
    condensed = condense_bundle_to_budget(extracted, target_tokens=4000)
    res = build_wahs_hub_enrichment_block(
        "What is my chronic kidney disease CKD stage and are any active drugs unsafe with my penicillin allergy?",
        extracted=extracted,
        condensed=condensed,
    )
    assert res.kb_passages_matched >= 1
    assert isinstance(res.edge_breakdown, dict)
    assert res.edge_breakdown.get("CROSS_REACTS_WITH", 0) >= 2
    meta = res.to_metadata_dict()
    assert meta["kb_passages_matched"] == res.kb_passages_matched
    assert meta["edge_breakdown"] == res.edge_breakdown


def test_extract_bundle_reference_date_skips_future_cancelled_and_coverage() -> None:
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-2",
                    "name": [{"given": ["Sam"], "family": "Lee"}],
                    "birthDate": "1968-08-20",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-finished",
                    "status": "finished",
                    "period": {"start": "2026-01-15", "end": "2026-01-15"},
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-planned",
                    "status": "planned",
                    "period": {"start": "2027-06-01", "end": "2027-06-01"},
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-cancelled",
                    "status": "cancelled",
                    "period": {"start": "2028-01-01", "end": "2028-01-01"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-eie",
                    "status": "entered-in-error",
                    "effectiveDateTime": "2029-02-02",
                    "code": {"text": "Glucose"},
                    "valueQuantity": {"value": 99, "unit": "mg/dL"},
                }
            },
            {
                "resource": {
                    "resourceType": "Coverage",
                    "id": "cov-1",
                    "status": "active",
                    "period": {"start": "2026-12-31", "end": "2030-12-31"},
                }
            },
        ],
    }
    ext = extract_bundle(bundle)
    assert ext.reference_date == "2026-01-15"


def test_wahs_core_stdlib_purity_in_subprocess() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; sys.path.insert(0, 'src'); "
                "import wahs_core, wahs_core.chart_serializer, wahs_core.clinical_graph, "
                "wahs_core.condenser, wahs_core.extractors, wahs_core.kb, wahs_core.kb_lookup, "
                "wahs_core.normalizers, wahs_core.wahs_tools; "
                "assert 'site' not in sys.modules; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in proc.stdout
