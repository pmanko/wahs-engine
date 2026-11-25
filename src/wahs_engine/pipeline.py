"""Server-configured optional ContextEnricher pipeline (`WAHS_ENABLED_ENRICHERS`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from wahs_core.condenser import CondensedChartResult
from wahs_core.extractors import ExtractedChart
from wahs_core.ports import ContextEnricher, EnrichmentInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    enrichment_blocks: List[str] = field(default_factory=list)
    new_citations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    disclosures: List[Dict[str, str]] = field(default_factory=list)
    used_tokens: int = 0

    def combined_block(self) -> str:
        return "\n\n".join(b for b in self.enrichment_blocks if b)


class EnricherPipeline:
    """No-op pass-through when `enabled_ids` is empty; lazily loads `wahs_extensions` only when configured."""

    def __init__(self, enabled_ids: Sequence[str] = ()) -> None:
        self.enabled_ids: tuple[str, ...] = tuple(str(eid).strip() for eid in enabled_ids if str(eid).strip())
        self._enrichers: Optional[List[ContextEnricher]] = None

    def _get_enrichers(self) -> List[ContextEnricher]:
        if self._enrichers is None:
            if not self.enabled_ids:
                self._enrichers = []
            else:
                from wahs_extensions import load_enricher

                self._enrichers = [load_enricher(eid) for eid in self.enabled_ids]
        return self._enrichers

    def run(
        self,
        *,
        extracted: ExtractedChart,
        condensed: CondensedChartResult,
        citation_map: Dict[str, Dict[str, Any]],
        question: str,
        conversation: Sequence[Mapping[str, str]],
        headroom_tokens: int,
        token_counter: Callable[[str], int],
    ) -> PipelineResult:
        if not self.enabled_ids or headroom_tokens < 48:
            return PipelineResult()

        remaining = int(headroom_tokens)
        blocks: List[str] = []
        disclosures: List[Dict[str, str]] = []
        added_citations: Dict[str, Dict[str, Any]] = {}

        for enricher in self._get_enrichers():
            if remaining < 48:
                break
            try:
                out = enricher.enrich(
                    EnrichmentInput(
                        extracted=extracted,
                        condensed=condensed,
                        citation_map=citation_map,
                        question=question,
                        conversation=conversation,
                        max_tokens=remaining,
                        token_counter=token_counter,
                    )
                )
            except Exception as exc:
                logger.warning("WAHS enricher %s skipped due to error: %s", getattr(enricher, "enricher_id", enricher), exc)
                continue
            if out is None or not out.prompt_block:
                continue
            toks = int(out.token_count or token_counter(out.prompt_block))
            if toks > remaining:
                continue
            blocks.append(out.prompt_block)
            remaining -= toks
            disclosures.append(out.disclosure.to_dict())
            for ck, cv in (out.new_citations or {}).items():
                if ck not in citation_map:
                    citation_map[ck] = dict(cv)
                    added_citations[ck] = dict(cv)

        return PipelineResult(
            enrichment_blocks=blocks,
            new_citations=added_citations,
            disclosures=disclosures,
            used_tokens=int(headroom_tokens) - remaining,
        )
