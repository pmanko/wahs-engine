"""Unit tests for wahs_core clean-layer graph rules, WE7-1 cross-reactivity, and reference_date scoping."""

from __future__ import annotations

import base64
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


def test_active_med_hold_note_respects_order_date_and_newer_resume_note() -> None:
    import base64
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _note_entry(nid: str, ndate: str, text: str) -> Dict[str, Any]:
        return {
            "resource": {
                "resourceType": "DocumentReference",
                "id": nid,
                "status": "current",
                "date": ndate,
                "type": {"text": f"Note {nid}"},
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                        }
                    }
                ],
            }
        }

    # (a) 2020 hold note must be ignored for a 2026 active order
    bundle_old_hold = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-hold-1", "birthDate": "1965-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-1",
                    "status": "active",
                    "authoredOn": "2026-05-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note_entry("doc-2020", "2020-02-10", "Holding metformin perioperatively prior to contrast."),
        ],
    }
    ext_old = extract_bundle(bundle_old_hold)
    g_old = ClinicalEntityGraph.from_extracted_chart(ext_old)
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_old.edges)

    # (b) 2026-10-05 hold superseded by 2026-10-12 resume note must not emit hold contradiction
    bundle_resumed = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-hold-2", "birthDate": "1965-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-2",
                    "status": "active",
                    "authoredOn": "2026-10-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note_entry("doc-hold", "2026-10-05", "Hold metformin during inpatient procedure."),
            _note_entry("doc-resume", "2026-10-12", "Patient stable; resume metformin 500 mg daily."),
        ],
    }
    ext_res = extract_bundle(bundle_resumed)
    g_res = ClinicalEntityGraph.from_extracted_chart(ext_res)
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_res.edges)

    # (c) Un-superseded 2026-10-15 hold note on 2026-10-01 active order DOES emit contradiction
    bundle_active_hold = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-hold-3", "birthDate": "1965-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-3",
                    "status": "active",
                    "authoredOn": "2026-10-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note_entry("doc-new-hold", "2026-10-15", "Holding metformin due to worsening eGFR."),
        ],
    }
    ext_ah = extract_bundle(bundle_active_hold)
    g_ah = ClinicalEntityGraph.from_extracted_chart(ext_ah)
    assert any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_ah.edges)


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


def test_obs_safety_thresholds_convert_ucum_units_and_ignore_unknown_units() -> None:
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _bundle(cr_val: float, cr_unit: str, high_ref: float | None = None) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "resourceType": "Observation",
            "id": "obs-cr-1",
            "status": "final",
            "category": [{"coding": [{"code": "laboratory"}]}],
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma"}],
                "text": "Creatinine",
            },
            "effectiveDateTime": "2026-10-01",
            "valueQuantity": {"value": cr_val, "unit": cr_unit},
        }
        if high_ref is not None:
            obs["referenceRange"] = [{"low": {"value": 60.0, "unit": cr_unit}, "high": {"value": high_ref, "unit": cr_unit}}]
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-cr-1", "birthDate": "1960-01-01"}},
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-met-cr",
                        "status": "active",
                        "authoredOn": "2026-09-01",
                        "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                    }
                },
                {"resource": obs},
            ],
        }

    # Normal creatinine in SI units (100 umol/L == 1.13 mg/dL, ref 60-110) must NOT warn
    g_norm = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_bundle(100.0, "umol/L", high_ref=110.0)))
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_norm.edges)

    # Elevated creatinine in SI units (150 umol/L == 1.70 mg/dL >= 1.5 mg/dL) DOES warn
    g_high = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_bundle(150.0, "umol/L", high_ref=110.0)))
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_high.edges)

    # Unrecognized unit marked non-abnormal must NOT trigger hard-coded mg/dL numeric threshold
    g_unk = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_bundle(25.0, "{arbitrary_units}")))
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_unk.edges)


def test_refuted_and_inactive_allergies_excluded_from_cross_reactivity() -> None:
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _allergy_bundle(c_status: str, v_status: str) -> Dict[str, Any]:
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-al-1", "birthDate": "1960-01-01"}},
                {
                    "resource": {
                        "resourceType": "AllergyIntolerance",
                        "id": "al-pen-1",
                        "clinicalStatus": {"coding": [{"code": c_status}]},
                        "verificationStatus": {"coding": [{"code": v_status}]},
                        "code": {"text": "Penicillin G"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-cef-1",
                        "status": "active",
                        "authoredOn": "2026-09-01",
                        "medicationCodeableConcept": {"text": "Cefazolin 1000 MG"},
                    }
                },
            ],
        }

    g_refuted = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_allergy_bundle("active", "refuted")))
    assert not any(e.edge_type == "CROSS_REACTS_WITH" for e in g_refuted.edges)

    g_eie = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_allergy_bundle("active", "entered-in-error")))
    assert not any(e.edge_type == "CROSS_REACTS_WITH" for e in g_eie.edges)

    g_inactive = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_allergy_bundle("inactive", "confirmed")))
    assert not any(e.edge_type == "CROSS_REACTS_WITH" for e in g_inactive.edges)

    g_confirmed = ClinicalEntityGraph.from_extracted_chart(extract_bundle(_allergy_bundle("active", "confirmed")))
    assert any(e.edge_type == "CROSS_REACTS_WITH" for e in g_confirmed.edges)


def test_abnormal_labs_in_non_risky_direction_do_not_trigger_safety_warnings() -> None:
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _bundle(med_text: str, obs_code: str, obs_disp: str, val: float, unit: str, interp: str) -> Dict[str, Any]:
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-dir-1", "birthDate": "1960-01-01"}},
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-1",
                        "status": "active",
                        "authoredOn": "2026-09-01",
                        "medicationCodeableConcept": {"text": med_text},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "status": "final",
                        "category": [{"coding": [{"code": "laboratory"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": obs_code, "display": obs_disp}], "text": obs_disp},
                        "interpretation": [{"coding": [{"code": interp}]}],
                        "effectiveDateTime": "2026-10-01",
                        "valueQuantity": {"value": val, "unit": unit},
                    }
                },
            ],
        }

    # Apixaban + HIGH hemoglobin (18 g/dL, H) -> NOT an anemia/bleed risk
    g_hb_hi = ClinicalEntityGraph.from_extracted_chart(
        extract_bundle(_bundle("Apixaban 5 MG", "718-7", "Hemoglobin", 18.0, "g/dL", "H"))
    )
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_hb_hi.edges)

    # Apixaban + LOW hemoglobin (8.5 g/dL, L) -> DOES warn
    g_hb_lo = ClinicalEntityGraph.from_extracted_chart(
        extract_bundle(_bundle("Apixaban 5 MG", "718-7", "Hemoglobin", 8.5, "g/dL", "L"))
    )
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_hb_lo.edges)

    # Metformin + HIGH eGFR (130 mL/min/1.73m2, H) -> NOT renal impairment
    g_egfr_hi = ClinicalEntityGraph.from_extracted_chart(
        extract_bundle(_bundle("Metformin 500 MG", "33914-3", "Glomerular filtration rate (eGFR)", 130.0, "mL/min/1.73m2", "H"))
    )
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_egfr_hi.edges)

    # Metformin + LOW eGFR (28 mL/min/1.73m2, L) -> DOES warn
    g_egfr_lo = ClinicalEntityGraph.from_extracted_chart(
        extract_bundle(_bundle("Metformin 500 MG", "33914-3", "Glomerular filtration rate (eGFR)", 28.0, "mL/min/1.73m2", "L"))
    )
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_egfr_lo.edges)


def test_clinical_rule_keywords_match_term_boundaries_without_substring_false_positives() -> None:
    from wahs_core import ClinicalEntityGraph, extract_bundle

    # Active Ibuprofen + Adrenal insufficiency + Furosemide + Bunion + Sulfa allergy + Ferrous sulfate
    # must NOT emit false CONTRAINDICATED_OR_DOSE_ADJUST_IN or CROSS_REACTS_WITH edges from substrings ("renal", "bun", "sulfa").
    false_pos_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-bnd-1", "birthDate": "1960-01-01"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-adrenal",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": "Adrenal insufficiency"},
                    "onsetDateTime": "2024-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-bunion",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": "Bunion"},
                    "onsetDateTime": "2024-02-01",
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "al-sulfa",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "verificationStatus": {"coding": [{"code": "confirmed"}]},
                    "code": {"text": "Sulfa drug allergy"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-ibuprofen",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Ibuprofen 400 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-furosemide",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Furosemide 20 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-ferrous",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Ferrous sulfate 325 MG Oral Tablet"},
                }
            },
        ],
    }
    g_fp = ClinicalEntityGraph.from_extracted_chart(extract_bundle(false_pos_bundle))
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_fp.edges)
    assert not any(e.edge_type == "CROSS_REACTS_WITH" for e in g_fp.edges)

    # Genuine whole-term & inflected matches ("bleeding" <- "bleed", "Platelets" <- "platelet") still emit edges.
    true_pos_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-bnd-2", "birthDate": "1960-01-01"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-gibleed",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {"text": "Gastrointestinal bleeding"},
                    "onsetDateTime": "2026-08-01",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-ibu-2",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Ibuprofen 400 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-warf-2",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Warfarin 5 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-plt-2",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "777-3", "display": "Platelets"}], "text": "Platelets"},
                    "effectiveDateTime": "2026-10-01",
                    "valueQuantity": {"value": 210.0, "unit": "10*3/uL"},
                }
            },
        ],
    }
    g_tp = ClinicalEntityGraph.from_extracted_chart(extract_bundle(true_pos_bundle))
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in g_tp.edges)
    assert any(e.edge_type == "MONITORED_BY" and "Platelets" in g_tp.nodes[e.target_id].label for e in g_tp.edges)


def test_refuted_and_entered_in_error_conditions_excluded_from_safety_rules() -> None:
    """PRRT_kwDOVrPAic63kpAh: active Condition with refuted/entered-in-error verificationStatus must not fire safety rules."""
    for verif_code in ("refuted", "entered-in-error"):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-cond-verif", "birthDate": "1965-01-01"}},
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"cond-renal-{verif_code}",
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "verificationStatus": {"coding": [{"code": verif_code}]},
                        "code": {"text": "Chronic kidney disease stage 4"},
                        "onsetDateTime": "2024-01-01",
                    }
                },
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-ibu-verif",
                        "status": "active",
                        "authoredOn": "2026-09-01",
                        "medicationCodeableConcept": {"text": "Ibuprofen 400 MG Oral Tablet"},
                    }
                },
            ],
        }
        ext = extract_bundle(bundle)
        cond_item = next(it for it in ext.items if it.resource_type == "Condition")
        assert cond_item.section == "condition_resolved"
        assert cond_item.metadata.get("verification_status") == verif_code
        graph = ClinicalEntityGraph.from_extracted_chart(ext)
        assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in graph.edges)


def test_comparator_based_low_egfr_observation_preserves_risk_metadata() -> None:
    """PRRT_kwDOVrPAic63kpAl: Observation with valueQuantity.comparator preserves risk metadata and triggers safety edges."""
    bundle_with_interp = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-cmp-1", "birthDate": "1962-03-03"}},
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-egfr-lt30",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "33914-3", "display": "eGFR"}], "text": "eGFR"},
                    "effectiveDateTime": "2026-10-01",
                    "valueQuantity": {"value": 30, "comparator": "<", "unit": "mL/min/1.73m2"},
                    "interpretation": [{"coding": [{"code": "L", "display": "Low"}]}],
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-cmp",
                    "status": "active",
                    "authoredOn": "2026-09-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
        ],
    }
    ext = extract_bundle(bundle_with_interp)
    assert ext.numeric_points == []
    graph = ClinicalEntityGraph.from_extracted_chart(ext)
    obs_node = next(n for n in graph.nodes.values() if n.category == "Observation" and "egfr" in n.label.lower())
    assert obs_node.metadata.get("is_abnormal") is True
    assert obs_node.metadata.get("interpretation") == "L"
    assert obs_node.metadata.get("unit") == "mL/min/1.73m2"
    assert obs_node.metadata.get("value") == "<30 mL/min/1.73m2"
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in graph.edges)

    # Also verify comparator threshold <30 without explicit interpretation tag
    bundle_cmp_only = json.loads(json.dumps(bundle_with_interp))
    del bundle_cmp_only["entry"][1]["resource"]["interpretation"]
    graph_no_interp = ClinicalEntityGraph.from_extracted_chart(extract_bundle(bundle_cmp_only))
    assert any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in graph_no_interp.edges)

    # Normal comparator >60 mL/min/1.73m2 must NOT trigger contraindication
    bundle_gt60 = json.loads(json.dumps(bundle_cmp_only))
    bundle_gt60["entry"][1]["resource"]["valueQuantity"] = {"value": 60, "comparator": ">", "unit": "mL/min/1.73m2"}
    graph_gt60 = ClinicalEntityGraph.from_extracted_chart(extract_bundle(bundle_gt60))
    assert not any(e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN" for e in graph_gt60.edges)


def _note_doc(note_id: str, date_str: str, note_text: str) -> dict:
    return {
        "resource": {
            "resourceType": "DocumentReference",
            "id": note_id,
            "status": "current",
            "date": date_str,
            "type": {"text": "Progress Note"},
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": base64.b64encode(note_text.encode("utf-8")).decode("ascii"),
                    }
                }
            ],
        }
    }


def test_negated_hold_and_resume_directives_do_not_emit_false_note_conflicts() -> None:
    """PRRT_kwDOVrPAic63kpAo & PRRT_kwDOVrPAic63kpAs: negated hold/resume phrases must not invert medication reconciliation."""
    # 1. Active metformin + active lisinopril with negated hold on lisinopril/metformin vs affirmative hold on metformin
    active_neg_hold = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-neg-1", "birthDate": "1968-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-act",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-lis-act",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            _note_doc(
                "doc-neg-hold",
                "2026-10-05",
                "Plan: Do not stop metformin. Please do not hold lisinopril before follow-up.",
            ),
        ],
    }
    g_neg = ClinicalEntityGraph.from_extracted_chart(extract_bundle(active_neg_hold))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_neg.edges)

    # Contrast: mixed clause "Do not stop lisinopril, but hold metformin." -> flags ONLY metformin
    mixed_bundle = json.loads(json.dumps(active_neg_hold))
    mixed_bundle["entry"][3] = _note_doc(
        "doc-mixed-hold",
        "2026-10-05",
        "Plan: Do not stop lisinopril, but hold metformin until renal repeat.",
    )
    g_mixed = ClinicalEntityGraph.from_extracted_chart(extract_bundle(mixed_bundle))
    mixed_edges = [e for e in g_mixed.edges if e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE"]
    assert len(mixed_edges) == 1
    assert "metformin" in g_mixed.nodes[mixed_edges[0].source_id].label.lower()

    # Non-adherence phrase "not taking" still triggers hold/modification reconciliation
    nonadhere_bundle = json.loads(json.dumps(active_neg_hold))
    nonadhere_bundle["entry"][3] = _note_doc(
        "doc-nonadhere",
        "2026-10-05",
        "Patient reports not taking metformin due to GI upset.",
    )
    g_nonadhere = ClinicalEntityGraph.from_extracted_chart(extract_bundle(nonadhere_bundle))
    assert any(
        e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE"
        and "metformin" in g_nonadhere.nodes[e.source_id].label.lower()
        for e in g_nonadhere.edges
    )

    # 2. Historical/stopped metformin + negated resume directives -> NO resume conflict edge
    stopped_neg_resume = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-neg-2", "birthDate": "1968-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-stop",
                    "status": "stopped",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note_doc(
                "doc-neg-resume",
                "2026-10-06",
                "Plan: Do not resume metformin. Patient cannot restart metformin.",
            ),
        ],
    }
    g_stop_neg = ClinicalEntityGraph.from_extracted_chart(extract_bundle(stopped_neg_resume))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_stop_neg.edges)

    # Positive resume on stopped metformin STILL fires
    stopped_pos_resume = json.loads(json.dumps(stopped_neg_resume))
    stopped_pos_resume["entry"][2] = _note_doc(
        "doc-pos-resume",
        "2026-10-06",
        "Plan: Restart metformin 500 mg daily now that eGFR stabilized.",
    )
    g_stop_pos = ClinicalEntityGraph.from_extracted_chart(extract_bundle(stopped_pos_resume))
    assert any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_stop_pos.edges)


def test_refuted_ckd_condition_with_active_clinical_status_does_not_emit_stage_conflict() -> None:
    """PRRT_kwDOVrPAic63kws8: refuted/entered-in-error Condition keeping clinicalStatus=active must not trigger CONFLICTS_WITH."""
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-ckd-ref", "birthDate": "1960-01-01"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ckd-1",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "verificationStatus": {"coding": [{"code": "confirmed"}]},
                    "code": {
                        "coding": [{"system": "http://snomed.info/sct", "code": "431855005", "display": "Chronic kidney disease stage 1"}],
                        "text": "Chronic kidney disease stage 1",
                    },
                    "onsetDateTime": "2024-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-ckd-3-refuted",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "verificationStatus": {"coding": [{"code": "refuted"}]},
                    "code": {
                        "coding": [{"system": "http://snomed.info/sct", "code": "433144002", "display": "Chronic kidney disease stage 3"}],
                        "text": "Chronic kidney disease stage 3",
                    },
                    "onsetDateTime": "2024-02-01",
                }
            },
        ],
    }
    ext = extract_bundle(bundle)
    g = ClinicalEntityGraph.from_extracted_chart(ext)
    assert not any(e.edge_type == "CONFLICTS_WITH" for e in g.edges)
    assert g.conflict_alerts == []

    # Control: two confirmed active CKD stages DO emit CONFLICTS_WITH
    confirmed_bundle = json.loads(json.dumps(bundle))
    confirmed_bundle["entry"][2]["resource"]["verificationStatus"] = {"coding": [{"code": "confirmed"}]}
    g_conf = ClinicalEntityGraph.from_extracted_chart(extract_bundle(confirmed_bundle))
    assert any(e.edge_type == "CONFLICTS_WITH" for e in g_conf.edges)


def test_comparator_valued_observation_component_and_diagnostic_report_child_preserve_safety_edges() -> None:
    """PRRT_kwDOVrPAic63kws-: Observation components & DiagnosticReport children with comparators retain risk metadata & trigger edges."""
    comp_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-comp-cmp", "birthDate": "1955-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-comp",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 1000 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-renal-panel-comp",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "24362-6", "display": "Renal function panel"}]},
                    "effectiveDateTime": "2026-06-01",
                    "component": [
                        {
                            "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
                            "valueQuantity": {"value": 30, "comparator": "<", "unit": "mL/min/1.73m2"},
                            "referenceRange": [{"low": {"value": 60, "unit": "mL/min/1.73m2"}}],
                        }
                    ],
                }
            },
        ],
    }
    ext_comp = extract_bundle(comp_bundle)
    assert ext_comp.numeric_points == []
    obs_item = next(it for it in ext_comp.items if it.resource_uuid == "obs-renal-panel-comp")
    assert obs_item.metadata.get("is_abnormal") is True
    assert obs_item.metadata.get("interpretation") == "L"
    comp_entries = obs_item.metadata.get("comparator_observations") or []
    assert len(comp_entries) == 1
    assert comp_entries[0]["comparator"] == "<"
    assert comp_entries[0]["unit"] == "mL/min/1.73m2"
    assert comp_entries[0]["interpretation"] == "L"
    assert comp_entries[0]["is_abnormal"] is True

    g_comp = ClinicalEntityGraph.from_extracted_chart(ext_comp)
    assert any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_comp.edges
    )

    # Also test comparator component without explicit interpretation/referenceRange
    comp_no_rr = json.loads(json.dumps(comp_bundle))
    del comp_no_rr["entry"][2]["resource"]["component"][0]["referenceRange"]
    g_comp_no_rr = ClinicalEntityGraph.from_extracted_chart(extract_bundle(comp_no_rr))
    assert any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_comp_no_rr.edges
    )

    # Control: normal comparator >60 mL/min/1.73m2 must NOT trigger contraindication
    comp_normal = json.loads(json.dumps(comp_bundle))
    comp_normal["entry"][2]["resource"]["component"][0]["valueQuantity"] = {
        "value": 60,
        "comparator": ">",
        "unit": "mL/min/1.73m2",
    }
    g_normal = ClinicalEntityGraph.from_extracted_chart(extract_bundle(comp_normal))
    assert not any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_normal.edges
    )

    # DiagnosticReport child Observation with comparator <30 mL/min/1.73m2
    dr_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-dr-cmp", "birthDate": "1955-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-dr",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-egfr-child",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
                    "effectiveDateTime": "2026-06-01",
                    "valueQuantity": {"value": 30, "comparator": "<", "unit": "mL/min/1.73m2"},
                    "interpretation": [{"coding": [{"code": "L"}]}],
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr-renal-cmp",
                    "status": "final",
                    "code": {"coding": [{"code": "24362-6", "display": "Renal function panel"}]},
                    "effectiveDateTime": "2026-06-01",
                    "result": [{"reference": "Observation/obs-egfr-child"}],
                }
            },
        ],
    }
    ext_dr = extract_bundle(dr_bundle)
    assert ext_dr.numeric_points == []
    dr_item = next(it for it in ext_dr.items if it.resource_uuid == "dr-renal-cmp")
    assert dr_item.metadata.get("is_abnormal") is True
    assert dr_item.metadata.get("interpretation") == "L"
    g_dr = ClinicalEntityGraph.from_extracted_chart(ext_dr)
    assert any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_dr.edges
    )

    # Normal upper-bounded Creatinine <1.5 mg/dL component must NOT trigger contraindication
    cr_normal_comp = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-cr-cmp", "birthDate": "1955-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-cr",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-cr-cmp",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "24362-6", "display": "Renal function panel"}]},
                    "effectiveDateTime": "2026-06-01",
                    "component": [
                        {
                            "code": {"coding": [{"code": "2160-0", "display": "Creatinine"}]},
                            "valueQuantity": {"value": 1.5, "comparator": "<", "unit": "mg/dL"},
                        }
                    ],
                }
            },
        ],
    }
    g_cr = ClinicalEntityGraph.from_extracted_chart(extract_bundle(cr_normal_comp))
    assert not any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_cr.edges
    )

    # Multi-child DiagnosticReport: abnormal Glucose [L] must NOT cross-contaminate normal eGFR >60
    dr_multi_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-dr-multi", "birthDate": "1955-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-multi",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-glu-child",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "2345-7", "display": "Glucose"}]},
                    "effectiveDateTime": "2026-06-01",
                    "valueQuantity": {"value": 45, "unit": "mg/dL"},
                    "interpretation": [{"coding": [{"code": "L"}]}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-egfr-normal-child",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
                    "effectiveDateTime": "2026-06-01",
                    "valueQuantity": {"value": 60, "comparator": ">", "unit": "mL/min/1.73m2"},
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr-multi-panel",
                    "status": "final",
                    "code": {"coding": [{"code": "24323-8", "display": "Comprehensive metabolic panel"}]},
                    "effectiveDateTime": "2026-06-01",
                    "result": [
                        {"reference": "Observation/obs-glu-child"},
                        {"reference": "Observation/obs-egfr-normal-child"},
                    ],
                }
            },
        ],
    }
    g_dr_multi = ClinicalEntityGraph.from_extracted_chart(extract_bundle(dr_multi_bundle))
    assert not any(
        e.edge_type == "CONTRAINDICATED_OR_DOSE_ADJUST_IN"
        for e in g_dr_multi.edges
    )


def test_same_note_directive_order_reconciles_hold_vs_resume() -> None:
    """PRRT_kwDOVrPAic63k5oM: later affirmative resume in same note must supersede earlier hold."""
    import base64
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _note(nid: str, ndate: str, text: str) -> Dict[str, Any]:
        return {
            "resource": {
                "resourceType": "DocumentReference",
                "id": nid,
                "status": "current",
                "date": ndate,
                "type": {"text": "Discharge Summary"},
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                        }
                    }
                ],
            }
        }

    # 1. Hold then resume in same note (plus older hold note) -> no hold contradiction on active med
    same_note_resumed = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-same-note-1", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-1",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note("doc-old-hold", "2026-06-01", "Hold metformin pre-op."),
            _note(
                "doc-dc",
                "2026-06-05",
                "Metformin was held for surgery. Metformin was resumed at discharge.",
            ),
        ],
    }
    g_res = ClinicalEntityGraph.from_extracted_chart(extract_bundle(same_note_resumed))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_res.edges)

    # 2. Same-sentence hold then resume -> no hold contradiction
    same_sent_resumed = json.loads(json.dumps(same_note_resumed))
    same_sent_resumed["entry"][3] = _note(
        "doc-dc-same-sent",
        "2026-06-05",
        "Metformin was held for surgery, then metformin was resumed at discharge.",
    )
    g_same_sent = ClinicalEntityGraph.from_extracted_chart(extract_bundle(same_sent_resumed))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_same_sent.edges)

    # 3. Resume then later hold in same note -> final hold contradiction MUST still fire
    same_note_held_after = json.loads(json.dumps(same_note_resumed))
    same_note_held_after["entry"][2] = _note(
        "doc-dc-held-last",
        "2026-06-05",
        "Metformin was resumed at admission. Metformin was held at discharge due to AKI.",
    )
    same_note_held_after["entry"].pop(3)
    g_held_last = ClinicalEntityGraph.from_extracted_chart(extract_bundle(same_note_held_after))
    assert any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_held_last.edges)

    # 4. Stopped medication with resume followed by later hold in same note -> no resume contradiction
    stopped_held_last = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-same-note-4", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-stop",
                    "status": "stopped",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note(
                "doc-stop-held-last",
                "2026-06-05",
                "Metformin was resumed in clinic. Metformin was held at follow-up for rising creatinine.",
            ),
        ],
    }
    g_stop_held = ClinicalEntityGraph.from_extracted_chart(extract_bundle(stopped_held_last))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_stop_held.edges)

    # 5. Compound two-drug sentence: other drug's resume must NOT suppress Metformin's hold
    compound_two_drugs = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-same-note-5", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-5",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-lis-5",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            _note(
                "doc-compound-2drugs",
                "2026-06-05",
                "Metformin was held for surgery, and lisinopril was resumed at discharge.",
            ),
        ],
    }
    g_comp = ClinicalEntityGraph.from_extracted_chart(extract_bundle(compound_two_drugs))
    met_hold_edges = [
        e for e in g_comp.edges
        if e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE"
        and "metformin" in g_comp.nodes[e.source_id].label.lower()
    ]
    lis_edges = [
        e for e in g_comp.edges
        if e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE"
        and "lisinopril" in g_comp.nodes[e.source_id].label.lower()
    ]
    assert len(met_hold_edges) == 1
    assert lis_edges == []

    # 6. Same-day cross-offset notes: 09:00-05:00 (14:00Z) resume supersedes 11:00Z hold
    cross_offset_notes = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-same-note-6", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-6",
                    "status": "active",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note("doc-hold-utc", "2026-06-05T11:00:00Z", "Hold metformin during inpatient procedure."),
            _note("doc-resume-est", "2026-06-05T09:00:00-05:00", "Metformin was resumed at discharge."),
        ],
    }
    g_tz = ClinicalEntityGraph.from_extracted_chart(extract_bundle(cross_offset_notes))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_tz.edges)


def test_medication_timestamp_prevents_same_day_older_hold_note_false_positive() -> None:
    """PRRT_kwDOVrPAic63lAHO: same-day hold note older than MedicationRequest timestamp must not trigger contradiction."""
    import base64
    from wahs_core import ClinicalEntityGraph, extract_bundle

    def _note(nid: str, ndate: str, text: str) -> Dict[str, Any]:
        return {
            "resource": {
                "resourceType": "DocumentReference",
                "id": nid,
                "status": "current",
                "date": ndate,
                "type": {"text": "Progress Note"},
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                        }
                    }
                ],
            }
        }

    # (a) 09:00Z hold note precedes 15:00Z (and cross-offset 10:00-05:00) MedicationRequest -> NO contradiction
    for authored_on in ("2026-06-05T15:00:00Z", "2026-06-05T10:00:00-05:00"):
        bundle_older_hold = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p-med-ts-1", "birthDate": "1964-01-01"}},
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-met-ts",
                        "status": "active",
                        "authoredOn": authored_on,
                        "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                    }
                },
                _note("doc-morning-hold", "2026-06-05T09:00:00Z", "Hold metformin during inpatient procedure."),
            ],
        }
        g_older = ClinicalEntityGraph.from_extracted_chart(extract_bundle(bundle_older_hold))
        assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_older.edges)

    # (b) 17:00Z hold note after 15:00Z MedicationRequest -> MUST emit contradiction
    bundle_newer_hold = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-med-ts-2", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-ts-2",
                    "status": "active",
                    "authoredOn": "2026-06-05T15:00:00Z",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note("doc-evening-hold", "2026-06-05T17:00:00Z", "Hold metformin at discharge due to AKI."),
        ],
    }
    g_newer = ClinicalEntityGraph.from_extracted_chart(extract_bundle(bundle_newer_hold))
    assert any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_newer.edges)

    # (c) Stopped medication with two same-day hold notes exercises latest_hold_ts comparator cleanly
    bundle_two_holds = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-med-ts-3", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-met-ts-stop",
                    "status": "stopped",
                    "authoredOn": "2026-01-01",
                    "medicationCodeableConcept": {"text": "Metformin 500 MG Oral Tablet"},
                }
            },
            _note("doc-hold-a", "2026-06-05", "Hold metformin during inpatient procedure."),
            _note("doc-hold-b", "2026-06-05", "Continue to hold metformin while renal function recovers."),
        ],
    }
    g_two_holds = ClinicalEntityGraph.from_extracted_chart(extract_bundle(bundle_two_holds))
    assert not any(e.edge_type == "CONTRADICTED_OR_MODIFIED_BY_NOTE" for e in g_two_holds.edges)


def test_graph_preserves_linked_medication_order_uuids_against_duplicate_conflicts() -> None:
    """Linked MedicationDispense/MedicationAdministration fulfillments never emit false CONFLICTS_WITH."""
    linked_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-med-link", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "mr-lis-1",
                    "status": "active",
                    "authoredOn": "2026-06-01",
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationDispense",
                    "id": "md-lis-1",
                    "status": "in-progress",
                    "whenHandedOver": "2026-06-02",
                    "authorizingPrescription": [{"reference": "MedicationRequest/mr-lis-1"}],
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "id": "ma-lis-1",
                    "status": "in-progress",
                    "effectiveDateTime": "2026-06-03",
                    "request": {"reference": "MedicationRequest/mr-lis-1"},
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
        ],
    }
    ext_linked = extract_bundle(linked_bundle)
    disp_item = next(it for it in ext_linked.items if it.resource_uuid == "md-lis-1")
    admin_item = next(it for it in ext_linked.items if it.resource_uuid == "ma-lis-1")
    assert "mr-lis-1" in (disp_item.metadata or {}).get("linked_order_uuids", [])
    assert "mr-lis-1" in (admin_item.metadata or {}).get("linked_order_uuids", [])

    g_linked = ClinicalEntityGraph.from_extracted_chart(ext_linked)
    assert not any(e.edge_type == "CONFLICTS_WITH" for e in g_linked.edges)
    assert g_linked.conflict_alerts == []

    # Unlinked independent active MedicationRequest orders for the same ingredient still conflict
    unlinked_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p-med-unlinked", "birthDate": "1964-01-01"}},
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "mr-lis-a",
                    "status": "active",
                    "authoredOn": "2026-06-01",
                    "medicationCodeableConcept": {"text": "Lisinopril 10 MG Oral Tablet"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "mr-lis-b",
                    "status": "active",
                    "authoredOn": "2026-06-05",
                    "medicationCodeableConcept": {"text": "Lisinopril 20 MG Oral Tablet"},
                }
            },
        ],
    }
    g_unlinked = ClinicalEntityGraph.from_extracted_chart(extract_bundle(unlinked_bundle))
    assert any(e.edge_type == "CONFLICTS_WITH" for e in g_unlinked.edges)
