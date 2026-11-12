"""Unit tests for pure-stdlib `wahs_core` (AC-2.1)."""

from __future__ import annotations

import base64
import inspect
from typing import Any, Dict

import pytest

import wahs_core
from wahs_core import (
    condense_bundle_to_budget,
    extract_bundle,
    get_acip_rules,
    get_clinical_exclusivity_and_privacy,
    get_lab_history,
    get_rxnorm_lookup,
    read_clinical_note,
    search_chart_entries,
    search_kb,
)


def _sample_wahs_bundle() -> Dict[str, Any]:
    note_b64 = base64.b64encode(
        b"# Chief Complaint\nDizziness on standing.\n# Assessment and Plan\nMonitor potassium on spironolactone."
    ).decode("ascii")
    entries = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-1",
                "name": [{"given": ["Jordan"], "family": "Rivera"}],
                "birthDate": "1968-04-12",
                "gender": "female",
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-plavix",
                "status": "active",
                "authoredOn": "2026-02-10",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "309362",
                            "display": "Clopidogrel 75 MG Oral Tablet",
                        }
                    ]
                },
                "meta": {
                    "tag": [
                        {
                            "system": "http://fhir.wahealthsummary.cirg.uw.edu/CodeSystem/wahealthsummary-source-name",
                            "code": "UW Medicine",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-story-1",
                "status": "final",
                "effectiveDateTime": "2026-03-01",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "51855-5",
                            "display": "Patient Note",
                        }
                    ]
                },
                "valueString": "Walking 20 minutes daily; mild lightheadedness in the morning.",
                "meta": {
                    "tag": [
                        {
                            "system": "http://fhir.wahealthsummary.cirg.uw.edu/CodeSystem/wahealthsummary-source-name",
                            "code": "Patient Story",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "Goal",
                "id": "goal-walk-1",
                "lifecycleStatus": "active",
                "startDate": "2026-03-01",
                "description": {"text": "Walk 30 minutes five days per week without dizziness"},
                "target": [{"dueDate": "2026-09-01"}],
                "note": [{"text": "Discuss blood pressure target with cardiology"}],
                "meta": {
                    "tag": [
                        {
                            "system": "http://fhir.wahealthsummary.cirg.uw.edu/CodeSystem/wahealthsummary-source-name",
                            "code": "Patient Story",
                        }
                    ]
                },
            }
        },
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-cardio-1",
                "status": "current",
                "date": "2026-02-15T10:00:00Z",
                "type": {"text": "Cardiology Progress Note"},
                "content": [{"attachment": {"contentType": "text/plain", "data": note_b64}}],
            }
        },
    ]
    for i in range(6):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-egfr-{i}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "effectiveDateTime": f"2026-0{i + 1}-15",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "33914-3",
                                "display": "Glomerular filtration rate (eGFR)",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 52 - i, "unit": "mL/min/1.73_m2"},
                }
            }
        )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def test_wahs_core_is_pure_and_demo_free() -> None:
    src = inspect.getsource(wahs_core.clinical_graph)
    assert "wahs_demo" not in src
    assert "build_alex_johnson_bundle" not in src
    assert get_acip_rules()
    assert get_rxnorm_lookup()
    assert get_clinical_exclusivity_and_privacy()
    hits = search_kb("clopidogrel", k=2, corpus="wahs", codes=["309362"])
    assert len(hits) >= 1


@pytest.mark.parametrize("budget", [512, 1024, 2048, 8192])
def test_condenser_budgets_goal_patient_story_and_overflow_tools(budget: int) -> None:
    bundle = _sample_wahs_bundle()
    extracted = extract_bundle(bundle)
    goal_items = [it for it in extracted.items if it.resource_type == "Goal"]
    assert len(goal_items) == 1
    assert goal_items[0].source_org == "Patient Story"
    assert "Walk 30 minutes" in goal_items[0].text

    med_items = [it for it in extracted.items if it.resource_type == "MedicationRequest"]
    assert med_items[0].source_org == "UW Medicine"

    condensed = condense_bundle_to_budget(bundle, target_tokens=budget)
    assert condensed.actual_tokens <= budget
    assert "Completeness Manifest" in condensed.chart_text
    assert any(m["resourceUuid"] == "goal-walk-1" for m in condensed.mappings)

    lab_hist = get_lab_history(extracted, "eGFR")
    assert len(lab_hist["results"]) == 6
    note_res = read_clinical_note(extracted, resource_uuid="doc-cardio-1")
    assert note_res["total_matches"] == 1
    assert "spironolactone" in note_res["results"][0]["text"].lower()
    entry_hits = search_chart_entries(extracted, query="lightheadedness")
    assert len(entry_hits["results"]) >= 1


def test_no_bespoke_question_directives_or_persona_kb_entries() -> None:
    import json
    from pathlib import Path

    assert not hasattr(
        wahs_core.clinical_graph,
        "_build_question_conditioned_clinical_and_privacy_directives",
    )

    corpus_path = (
        Path(wahs_core.__file__).resolve().parent / "kb_data" / "wahs_corpus.jsonl"
    )
    entries = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) >= 8
    forbidden_ids = {
        "wahs-kb-coercion-bystander-safety",
        "wahs-kb-minimum-necessary-third-party",
        "wahs-kb-right-of-access-and-amendment",
        "wahs-kb-restraint-vs-criminal-record",
        "wahs-kb-single-patient-boundary",
        "wahs-kb-personal-devices-vs-facility-equipment",
        "wahs-cabg-icu-stay-education",
        "wahs-chronic-meds-insulin-hctz-metoprolol",
        "wahs-dental-sinus-anemia-sleep-education",
        "wahs-condition-status-conflict-sinusitis",
    }
    for entry in entries:
        assert entry["id"] not in forbidden_ids
        assert len(entry.get("codes") or []) >= 1

    bundle = _sample_wahs_bundle()
    condensed = condense_bundle_to_budget(bundle, target_tokens=1024)
    test_questions = [
        "My partner is reading this with me right now, what should I know?",
        "Write a letter for my employer and landlord accommodation.",
        "Do I have a criminal record?",
        "What is my HbA1c?",
    ]
    for q in test_questions:
        res = wahs_core.build_wahs_hub_enrichment_block(
            [
                {"role": "system", "content": condensed.chart_text},
                {"role": "user", "content": q},
            ]
        )
        assert "[KB wahs-" not in res.enrichment_text
        for forbidden_phrase in (
            "Screen-Watching Coercion Guardrail",
            "Third-Party",
            "Criminal Record Disambiguation",
            "HbA1c Plausibility Check",
            "Personal Device Filtering",
        ):
            assert forbidden_phrase not in res.enrichment_text
