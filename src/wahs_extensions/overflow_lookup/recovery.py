"""Read-only overflow lookup and omitted clinical item recovery (`ContextEnricher`)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from wahs_core.extractors import ClinicalItem, ExtractedChart, extract_bundle
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_INVALIDATED_STATUSES = {"entered-in-error", "cancelled", "noshow"}


def _ensure_extracted(bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart]) -> ExtractedChart:
    return (
        bundle_or_extracted
        if isinstance(bundle_or_extracted, ExtractedChart)
        else extract_bundle(bundle_or_extracted)
    )


def search_chart_entries(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    query: str = "",
    *,
    section: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 15,
) -> Dict[str, Any]:
    """Search uncondensed chart items by keyword, section, and date window."""
    extracted = _ensure_extracted(bundle_or_extracted)
    raw_tokens = [tok for tok in re.findall(r"[a-z0-9.\-]+", (query or "").lower()) if len(tok) >= 2]
    stopwords = {"record", "records", "status", "active", "condition", "medication", "observation", "the", "and", "for"}
    q_tokens = [t for t in raw_tokens if t not in stopwords] or raw_tokens

    matches: List[Dict[str, Any]] = []
    for item in extracted.items:
        if item.section == "patient":
            continue
        if section and section.lower() not in item.section.lower():
            continue
        item_date = item.clinical_date or item.date or ""
        if date_from and item_date and item_date < date_from:
            continue
        if date_to and item_date and item_date > date_to:
            continue
        haystack = f"{item.display} {item.text} {item.code} {item.section} {item.source_org or ''}".lower()
        score = sum(1 for tok in q_tokens if tok in haystack) if q_tokens else 1
        if q_tokens and score == 0:
            continue
        matches.append(
            {
                "score": score,
                "resource_type": item.resource_type,
                "resource_uuid": item.resource_uuid,
                "member_uuids": list(item.member_uuids),
                "section": item.section,
                "code": item.code,
                "display": item.display,
                "date": item_date,
                "source_org": item.source_org or "",
                "status": item.status,
                "text": item.text,
            }
        )
    matches.sort(key=lambda m: (m["score"], m["date"] or ""), reverse=True)
    top = matches[: max(1, int(limit))]
    return {
        "tool": "search_chart_entries",
        "query": query,
        "section": section,
        "total_matches": len(matches),
        "returned_count": len(top),
        "truncated": len(matches) > len(top),
        "results": top,
    }


def get_lab_history(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    analyte_or_loinc: str,
    *,
    limit: int = 25,
) -> Dict[str, Any]:
    """Retrieve longitudinal numeric measurements from an uncondensed chart."""
    extracted = _ensure_extracted(bundle_or_extracted)
    q_norm = (analyte_or_loinc or "").strip().lower()
    q_tokens = [t for t in re.findall(r"[a-z0-9.\-]+", q_norm) if len(t) >= 2]
    pts = [
        p
        for p in extracted.numeric_points
        if str(getattr(p, "status", "") or "").lower() not in _INVALIDATED_STATUSES
        and (
            q_norm in f"{p.concept} {p.code} {p.category}".lower()
            or (q_tokens and any(t in f"{p.concept} {p.code} {p.category}".lower() for t in q_tokens))
        )
    ]
    pts.sort(key=lambda p: p.date)
    selected = pts[-max(1, int(limit)) :]
    series = [
        {
            "resource_uuid": p.resource_uuid,
            "date": p.date,
            "code": p.code,
            "concept": p.concept,
            "value": p.value,
            "unit": p.unit,
            "category": p.category,
        }
        for p in selected
    ]
    summary: Dict[str, Any] = {}
    if pts:
        min_pt, max_pt = min(pts, key=lambda x: x.value), max(pts, key=lambda x: x.value)
        summary = {
            "total_measurements": len(pts),
            "initial": {"value": pts[0].value, "unit": pts[0].unit, "date": pts[0].date, "resource_uuid": pts[0].resource_uuid},
            "min": {"value": min_pt.value, "unit": min_pt.unit, "date": min_pt.date, "resource_uuid": min_pt.resource_uuid},
            "max": {"value": max_pt.value, "unit": max_pt.unit, "date": max_pt.date, "resource_uuid": max_pt.resource_uuid},
            "latest": {"value": pts[-1].value, "unit": pts[-1].unit, "date": pts[-1].date, "resource_uuid": pts[-1].resource_uuid},
        }
    return {
        "tool": "get_lab_history",
        "analyte_or_loinc": analyte_or_loinc,
        "total_matches": len(pts),
        "returned_count": len(series),
        "truncated": len(pts) > len(series),
        "summary": summary,
        "results": series,
    }


def read_clinical_note(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    resource_uuid: str = "",
    *,
    query: str = "",
) -> Dict[str, Any]:
    """Read deduplicated full-text clinical notes by UUID or keyword."""
    extracted = _ensure_extracted(bundle_or_extracted)
    target_uuid = (resource_uuid or "").strip()
    q_norm = (query or "").strip().lower()
    notes: List[Dict[str, Any]] = []
    for item in extracted.items:
        if item.section not in {"clinical_note", "note"}:
            continue
        if target_uuid and target_uuid not in ({item.resource_uuid} | set(item.member_uuids)):
            continue
        body = str(item.metadata.get("raw_decoded_text") or item.metadata.get("full_text") or item.text or "").strip()
        if q_norm and q_norm not in f"{item.display} {item.text} {body}".lower():
            continue
        notes.append(
            {
                "resource_type": item.resource_type,
                "resource_uuid": item.resource_uuid,
                "member_uuids": list(item.member_uuids),
                "date": item.clinical_date or item.date or "",
                "display": item.display,
                "source_org": item.source_org or "",
                "text": body or item.text,
            }
        )
    return {"tool": "read_clinical_note", "resource_uuid": target_uuid, "query": query, "total_matches": len(notes), "results": notes}


def _covered_uuids(citation_map: Mapping[str, Dict[str, Any]]) -> Tuple[Set[str], int]:
    covered: Set[str] = set()
    max_idx = 0
    for k_str, entry in citation_map.items():
        if str(k_str).isdigit():
            max_idx = max(max_idx, int(k_str))
        if isinstance(entry, Mapping):
            if entry.get("id"):
                covered.add(str(entry["id"]))
            for m in entry.get("memberIds") or ():
                if m:
                    covered.add(str(m))
    return covered, max_idx


class OmittedRecordRecoveryEnricher:
    """Recovers omitted structured records matching query terms when a chart overflows budget."""

    enricher_id = "overflow_lookup"
    disclosure = EnrichmentDisclosure(
        enricher_id="overflow_lookup",
        version="0.1.0",
        kind="retrieval",
        scope_summary="Recovers omitted FHIR items matching user query terms when Completeness Manifest shows omitted_clinical_items > 0.",
        limitations="Token-overlap ranking over omitted items only; inactive when Tier 0/1 already covers 100% of records.",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        omitted_count = int((inp.condensed.completeness_manifest or {}).get("omitted_clinical_items_count") or 0)
        if omitted_count <= 0 or inp.max_tokens < 48:
            return None
        q_terms = [t for t in re.findall(r"[a-z0-9]{3,}", (inp.question or "").lower())]
        if not q_terms:
            return None

        covered, max_idx = _covered_uuids(inp.citation_map)
        scored: List[Tuple[Tuple[int, str], ClinicalItem]] = []
        for it in inp.extracted.items:
            if it.section in {"patient", "manifest"}:
                continue
            item_uuids = {u for u in (it.resource_uuid, *it.member_uuids) if u}
            if not item_uuids or (item_uuids & covered):
                continue
            hay = f"{it.display} {it.text} {it.code} {it.source_org or ''}".lower()
            hits = sum(1 for t in set(q_terms) if t in hay)
            if hits > 0:
                scored.append(((-hits, it.date or "0000-00-00"), it))

        if not scored:
            return None
        scored.sort(key=lambda pair: (pair[0][0], -(int(pair[0][1][:10].replace("-", "")) if re.match(r"^\d{4}-\d{2}-\d{2}", pair[0][1]) else 0)))

        heading = f"=== RECOVERED OMITTED CHART ITEMS ({omitted_count} omitted overall; cite as [N]) ==="
        lines: List[str] = []
        new_cits: Dict[str, Dict[str, Any]] = {}
        next_idx = max_idx + 1
        for _, it in scored[:8]:
            ruuid = it.resource_uuid or (it.member_uuids[0] if it.member_uuids else "")
            if not ruuid:
                continue
            snippet = (it.text or it.display or it.resource_type).strip()
            if len(snippet) > 220:
                snippet = snippet[:217].rstrip() + "..."
            dt_p = f"({it.date}) " if it.date else ""
            cand_line = f"[{next_idx}] {dt_p}{snippet}"
            trial = "\n".join([heading, *lines, cand_line])
            if inp.token_counter(trial) > inp.max_tokens:
                break
            lines.append(cand_line)
            new_cits[str(next_idx)] = {
                "index": next_idx,
                "resourceType": it.resource_type,
                "id": ruuid,
                "source": it.source_org or "Overflow Recovery",
                "section": it.section,
                "date": it.date or "",
                "display": it.display or snippet,
                **({"memberIds": list(it.member_uuids)} if it.member_uuids else {}),
            }
            next_idx += 1

        if not lines:
            return None
        block = "\n".join([heading, *lines])
        return EnrichmentOutput(
            enricher_id=self.enricher_id,
            heading=heading,
            prompt_block=block,
            token_count=inp.token_counter(block),
            disclosure=self.disclosure,
            new_citations=new_cits,
            metadata={"omitted_clinical_items_count": omitted_count, "recovered_count": len(lines)},
        )
