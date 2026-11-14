"""Pure-Python stdlib clinical FHIR R4 condensation, 4-Layer US Core KB, and verification core."""

from __future__ import annotations

from .chart_serializer import render_chart
from .clinical_graph import (
    ClinicalEdge,
    ClinicalEntityGraph,
    ClinicalNode,
    HubEnrichmentResult,
    build_wahs_hub_enrichment_block,
    extract_clean_user_query,
)
from .condenser import (
    CondensedChartResult,
    condense_bundle_to_budget,
    estimate_tokens_local,
    items_to_querystore_records,
)
from .extractors import (
    ClinicalItem,
    ExtractedChart,
    NumericPoint,
    extract_bundle,
)
from .kb import WAHS_CORPUS_PATH, reset_cache as reset_kb_cache, search as search_kb
from .kb_lookup import (
    audit_cohort_code_coverage,
    extract_chart_kb_codes,
    get_acip_rules,
    get_clinical_exclusivity_and_privacy,
    get_rxnorm_lookup,
    reset_kb_lookup_cache,
    resolve_condition_code,
    resolve_immunization_code,
    resolve_lab_code,
    resolve_medication_code,
)
from .normalizers import (
    annotate_immunization_anomalies,
    classify_sensitivity,
    classify_sensitivity_with_source,
    classify_vaccine_group,
    detect_condition_conflicts,
    detect_medication_conflicts,
    is_clinic_room_equipment,
    is_non_medical_admin_code,
    normalize_drug_ingredient,
)
from .wahs_tools import (
    get_lab_history,
    read_clinical_note,
    retrieve_clinical_note_passages,
    search_chart_entries,
)

__version__ = "0.1.0"

__all__ = [
    "ClinicalEdge",
    "ClinicalEntityGraph",
    "ClinicalItem",
    "ClinicalNode",
    "CondensedChartResult",
    "ExtractedChart",
    "HubEnrichmentResult",
    "NumericPoint",
    "WAHS_CORPUS_PATH",
    "__version__",
    "annotate_immunization_anomalies",
    "audit_cohort_code_coverage",
    "build_wahs_hub_enrichment_block",
    "classify_sensitivity",
    "classify_sensitivity_with_source",
    "classify_vaccine_group",
    "condense_bundle_to_budget",
    "detect_condition_conflicts",
    "detect_medication_conflicts",
    "estimate_tokens_local",
    "extract_bundle",
    "extract_chart_kb_codes",
    "extract_clean_user_query",
    "get_acip_rules",
    "get_clinical_exclusivity_and_privacy",
    "get_lab_history",
    "get_rxnorm_lookup",
    "is_clinic_room_equipment",
    "is_non_medical_admin_code",
    "items_to_querystore_records",
    "normalize_drug_ingredient",
    "read_clinical_note",
    "render_chart",
    "reset_kb_cache",
    "reset_kb_lookup_cache",
    "resolve_condition_code",
    "resolve_immunization_code",
    "resolve_lab_code",
    "resolve_medication_code",
    "retrieve_clinical_note_passages",
    "search_chart_entries",
    "search_kb",
]
