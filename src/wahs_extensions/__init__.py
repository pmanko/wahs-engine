"""Toggleable context enrichment catalog for WAHS Engine (`profiles.yaml` / `WAHS_ENABLED_ENRICHERS`).

Core `wahs_engine` runs with zero enrichers enabled (`default_profile: baseline`).
Enrichers in this package implement `wahs_core.ports.ContextEnricher` and are
imported lazily ONLY when explicitly activated by a profile or request.
"""

from __future__ import annotations

from wahs_core.ports import ContextEnricher

_AVAILABLE_ENRICHER_IDS = (
    "temporal",
    "temporal_trends",
    "terminology_kb",
    "note_rag",
    "overflow_lookup",
    "guideline_kb",
    "experimental.clinical_reconciliation",
)


def list_available_enrichers() -> tuple[str, ...]:
    return _AVAILABLE_ENRICHER_IDS


def load_enricher(enricher_id: str) -> ContextEnricher:
    """Lazily instantiate a registered `ContextEnricher` by ID."""
    key = (enricher_id or "").strip()
    if key in {"temporal", "temporal_trends"}:
        from .temporal_trends import QuantitativeTrendEnricher

        return QuantitativeTrendEnricher(enricher_id=key)
    if key == "terminology_kb":
        from .terminology_kb import TerminologyConceptEnricher

        return TerminologyConceptEnricher()
    if key == "note_rag":
        from .note_rag import BM25NotePassageEnricher

        return BM25NotePassageEnricher()
    if key == "overflow_lookup":
        from .overflow_lookup import OmittedRecordRecoveryEnricher

        return OmittedRecordRecoveryEnricher()
    if key == "guideline_kb":
        from .guideline_kb import PublicGuidelineEnricher

        return PublicGuidelineEnricher()
    if key == "experimental.clinical_reconciliation":
        from .experimental.clinical_reconciliation import PrototypeClinicalGraphEnricher

        return PrototypeClinicalGraphEnricher()
    raise KeyError(
        f"Unknown WAHS enricher ID {enricher_id!r}; expected one of {_AVAILABLE_ENRICHER_IDS}"
    )


get_registered_enricher = load_enricher

__all__ = ["get_registered_enricher", "list_available_enrichers", "load_enricher"]
