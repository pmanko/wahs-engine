"""Pure-stdlib Hexagonal extension port for optional WAHS context enrichers.

Core ``wahs_engine`` performs strictly faithful, domain-neutral FHIR R4 structural
projection and verbatim-first condensation. Optional enrichment capabilities live
under ``wahs_extensions`` and plug into this ``Protocol`` only when explicitly
enabled by operator configuration (``WAHS_ENABLED_ENRICHERS``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .condenser import CondensedChartResult
    from .extractors import ExtractedChart


@dataclass(frozen=True)
class EnrichmentDisclosure:
    """Transparency disclosure surfaced in ``x_wahs.enrichers`` and ``AI-Provenance``."""

    enricher_id: str
    version: str
    kind: str  # "retrieval" | "quantitative_summary" | "experimental_clinical_rules"
    scope_summary: str
    limitations: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "enricher_id": self.enricher_id,
            "version": self.version,
            "kind": self.kind,
            "scope_summary": self.scope_summary,
            "limitations": self.limitations,
        }


@dataclass(frozen=True)
class EnrichmentInput:
    """Read-only per-request inputs passed to an enabled ``ContextEnricher``."""

    extracted: ExtractedChart
    condensed: CondensedChartResult
    citation_map: Mapping[str, Dict[str, Any]]
    question: str
    conversation: Sequence[Mapping[str, str]]
    max_tokens: int
    token_counter: Callable[[str], int]


@dataclass(frozen=True)
class EnrichmentOutput:
    """Bounded enrichment payload returned by an enabled ``ContextEnricher``."""

    enricher_id: str
    heading: str
    prompt_block: str
    token_count: int
    disclosure: EnrichmentDisclosure
    new_citations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextEnricher(Protocol):
    """Structural typing port (`PEP 544`) for optional context enrichers."""

    enricher_id: str
    disclosure: EnrichmentDisclosure

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None: ...


__all__ = [
    "ContextEnricher",
    "EnrichmentDisclosure",
    "EnrichmentInput",
    "EnrichmentOutput",
]
