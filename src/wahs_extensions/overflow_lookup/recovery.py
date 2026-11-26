"""Read-only overflow lookup and BM25 + terminology-bridged omitted item recovery (`ContextEnricher`)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from wahs_core.extractors import ClinicalItem, ExtractedChart, extract_bundle
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_INVALIDATED_STATUSES = {"entered-in-error", "cancelled", "noshow"}
_OVERFLOW_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "why", "how", "are", "was", "were", "have", "has", "had", "been",
    "record", "records", "status", "patient", "chart", "please", "check",
}


def _ensure_extracted(bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart]) -> ExtractedChart:
    return (
        bundle_or_extracted
        if isinstance(bundle_or_extracted, ExtractedChart)
        else extract_bundle(bundle_or_extracted)
    )


def _tokenize_overflow(text: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9.\-]*", (text or "").lower())
        if len(tok) >= 2 and tok not in _OVERFLOW_STOPWORDS
    ]


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
        codings_str = " ".join(
            f"{c.get('system', '')}:{c.get('code', '')} {c.get('display', '')}"
            for c in ((item.metadata or {}).get("codings") or ())
            if isinstance(c, Mapping)
        )
        haystack = f"{item.display} {item.text} {item.code} {item.section} {item.source_org or ''} {codings_str}".lower()
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
        "total_measurements": len(pts),
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
        if isinstance(entry, Mapping) and entry.get("resourceType") != "KnowledgeReference":
            if entry.get("id"):
                covered.add(str(entry["id"]))
            for m in entry.get("memberIds") or ():
                if m:
                    covered.add(str(m))
    return covered, max_idx


class OmittedRecordRecoveryEnricher:
    """Recovers omitted structured records via IDF-weighted Okapi BM25 + terminology synonym/code overlap."""

    enricher_id = "overflow_lookup"
    disclosure = EnrichmentDisclosure(
        enricher_id="overflow_lookup",
        version="0.2.0",
        kind="retrieval",
        scope_summary=(
            "Recovers omitted FHIR items matching user query terms and CIEL/RxNorm/LOINC synonyms "
            "using Okapi BM25 when Completeness Manifest shows omitted_clinical_items > 0."
        ),
        limitations="Self-gated to Tier 2 (inactive when Tier 0/1 already covers 100% of extracted clinical records).",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        omitted_count = int((inp.condensed.completeness_manifest or {}).get("omitted_clinical_items_count") or 0)
        if omitted_count <= 0 or inp.max_tokens < 48:
            return None

        literal_q = _tokenize_overflow(inp.question or "")
        q_weights: Dict[str, float] = {t: 1.0 for t in literal_q}
        if inp.question:
            from wahs_extensions.terminology_kb import expand_query_with_terminology

            for exp_t in expand_query_with_terminology(inp.question, inp.extracted):
                for sub_t in _tokenize_overflow(exp_t):
                    q_weights.setdefault(sub_t, 0.6)
        if not q_weights:
            return None

        covered, max_idx = _covered_uuids(inp.citation_map)
        omitted_items: List[ClinicalItem] = []
        doc_tfs: List[Dict[str, int]] = []
        total_dl = 0

        for it in inp.extracted.items:
            if it.section in {"patient", "manifest", "clinical_note"}:
                continue
            item_uuids = {u for u in (it.resource_uuid, *it.member_uuids) if u}
            if not item_uuids or (item_uuids & covered):
                continue
            codings_txt = " ".join(
                f"{c.get('code', '')} {c.get('display', '')}"
                for c in ((it.metadata or {}).get("codings") or ())
                if isinstance(c, Mapping)
            )
            toks = _tokenize_overflow(f"{it.display} {it.text} {it.code} {it.source_org or ''} {codings_txt}")
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            tf["__dl__"] = len(toks)
            total_dl += len(toks)
            omitted_items.append(it)
            doc_tfs.append(tf)

        if not omitted_items:
            return None

        n_docs = len(doc_tfs)
        avgdl = total_dl / max(1, n_docs)
        df: Dict[str, int] = {qt: 0 for qt in q_weights}
        for tf in doc_tfs:
            for qt in q_weights:
                if qt in tf:
                    df[qt] += 1

        k1, b = 1.5, 0.75
        scored: List[Tuple[float, int, ClinicalItem]] = []
        for idx, tf in enumerate(doc_tfs):
            dl = tf.get("__dl__", 1)
            score = 0.0
            for qt, w in q_weights.items():
                freq = tf.get(qt, 0)
                if freq <= 0:
                    continue
                doc_f = df.get(qt, 0)
                idf = math.log(1.0 + (n_docs - doc_f + 0.5) / (doc_f + 0.5))
                denom = freq + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                score += w * idf * ((freq * (k1 + 1.0)) / max(1e-9, denom))
            if score > 0.0:
                it = omitted_items[idx]
                d_int = int(it.date[:10].replace("-", "")) if (it.date and re.match(r"^\d{4}-\d{2}-\d{2}", it.date)) else 0
                scored.append((score, d_int, it))

        if not scored:
            return None
        scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)

        heading = f"=== RECOVERED OMITTED CHART ITEMS ({omitted_count} omitted overall; cite as [N]) ==="
        lines: List[str] = []
        new_cits: Dict[str, Dict[str, Any]] = {}
        next_idx = max_idx + 1
        for _, _dint, it in scored[:10]:
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
