"""Pure-Python stdlib FHIR R4 clinical projection, verbatim-first condensation, and extension port."""

from __future__ import annotations

from .chart_serializer import render_chart
from .condenser import (
    CondensedChartResult,
    condense_bundle_to_budget,
    estimate_tokens_local,
    items_to_querystore_records,
)
from .extractors import (
    ClinicalItem,
    ClinicalNote,
    ExtractedChart,
    NumericPoint,
    extract_bundle,
)
from .ports import (
    ContextEnricher,
    EnrichmentDisclosure,
    EnrichmentInput,
    EnrichmentOutput,
)

__version__ = "0.2.0"

__all__ = [
    "ClinicalItem",
    "ClinicalNote",
    "CondensedChartResult",
    "ContextEnricher",
    "EnrichmentDisclosure",
    "EnrichmentInput",
    "EnrichmentOutput",
    "ExtractedChart",
    "NumericPoint",
    "__version__",
    "condense_bundle_to_budget",
    "estimate_tokens_local",
    "extract_bundle",
    "items_to_querystore_records",
    "render_chart",
]
