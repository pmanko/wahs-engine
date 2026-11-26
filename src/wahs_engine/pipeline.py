"""Toggleable ContextEnricher pipeline with per-enricher token caps and zero-import baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from wahs_core.condenser import CondensedChartResult
from wahs_core.extractors import ExtractedChart
from wahs_core.ports import ContextEnricher, EnrichmentInput

logger = logging.getLogger(__name__)

ENRICHER_TOKEN_CAPS: Dict[str, int] = {
    "temporal": 450,
    "temporal_trends": 450,
    "terminology_kb": 550,
    "note_rag": 1100,
    "overflow_lookup": 550,
    "guideline_kb": 500,
    "experimental.clinical_reconciliation": 550,
}


@dataclass(frozen=True)
class PipelineResult:
    enrichment_blocks: List[str] = field(default_factory=list)
    new_citations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    used_tokens: int = 0

    def combined_block(self) -> str:
        return "\n\n".join(b for b in self.enrichment_blocks if b)


class EnricherPipeline:
    """No-op pass-through when active enrichers are empty; lazily imports `wahs_extensions` only when active."""

    def __init__(self, enabled_ids: Sequence[str] = (), *, max_enrichment_tokens: int = 2048) -> None:
        self.enabled_ids: tuple[str, ...] = tuple(str(eid).strip() for eid in enabled_ids if str(eid).strip())
        self.max_enrichment_tokens: int = max(64, int(max_enrichment_tokens))
        self._cache: Dict[str, ContextEnricher] = {}

    def _resolve_instances(self, active_ids: Sequence[str]) -> List[ContextEnricher]:
        ids = [str(eid).strip() for eid in active_ids if str(eid).strip()]
        if not ids:
            return []
        missing = [eid for eid in ids if eid not in self._cache]
        if missing:
            from wahs_extensions import load_enricher

            for eid in missing:
                self._cache[eid] = load_enricher(eid)
        return [self._cache[eid] for eid in ids]

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
        active_ids: Optional[Sequence[str]] = None,
        max_enrichment_tokens: Optional[int] = None,
    ) -> PipelineResult:
        eff_ids = tuple(self.enabled_ids if active_ids is None else active_ids)
        if not eff_ids:
            return PipelineResult()

        total_cap = min(
            max(0, int(headroom_tokens)),
            int(max_enrichment_tokens if max_enrichment_tokens is not None else self.max_enrichment_tokens),
        )
        if total_cap < 48:
            return PipelineResult()

        remaining = total_cap
        blocks: List[str] = []
        disclosures: List[Dict[str, Any]] = []
        added_citations: Dict[str, Dict[str, Any]] = {}

        for enricher in self._resolve_instances(eff_ids):
            eid = getattr(enricher, "enricher_id", "")
            per_cap = min(remaining, ENRICHER_TOKEN_CAPS.get(eid, 550))
            t0 = time.perf_counter()
            out = None
            if per_cap >= 32:
                try:
                    out = enricher.enrich(
                        EnrichmentInput(
                            extracted=extracted,
                            condensed=condensed,
                            citation_map=citation_map,
                            question=question,
                            conversation=conversation,
                            max_tokens=per_cap,
                            token_counter=token_counter,
                        )
                    )
                except Exception as exc:
                    logger.warning("WAHS enricher %s skipped due to error: %s", eid, exc)
                    out = None
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            base_disc = enricher.disclosure.to_dict()
            if out is None or not out.prompt_block:
                disclosures.append(
                    {
                        **base_disc,
                        "fired": False,
                        "tokens_used": 0,
                        "latency_ms": latency_ms,
                    }
                )
                continue

            toks = int(out.token_count or token_counter(out.prompt_block))
            if toks > per_cap or toks > remaining:
                disclosures.append(
                    {
                        **base_disc,
                        "fired": False,
                        "tokens_used": 0,
                        "latency_ms": latency_ms,
                    }
                )
                continue

            blocks.append(out.prompt_block)
            remaining -= toks
            disclosures.append(
                {
                    **out.disclosure.to_dict(),
                    "fired": True,
                    "tokens_used": toks,
                    "latency_ms": latency_ms,
                }
            )
            for ck, cv in (out.new_citations or {}).items():
                skey = str(ck)
                if skey not in citation_map:
                    citation_map[skey] = dict(cv)
                    added_citations[skey] = dict(cv)

        return PipelineResult(
            enrichment_blocks=blocks,
            new_citations=added_citations,
            disclosures=disclosures,
            used_tokens=total_cap - remaining,
        )
