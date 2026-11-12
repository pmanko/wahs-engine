"""Read-only, privacy-enforcing full-chart overflow lookup tools for WAHS+ (`W7`).

When a patient or clinician question asks for deep historical records (> 2 years old),
full-text clinical notes, or perioperative details that were condensed out of the
in-context IPS Safety Core (`Tier 3: 4k` or `Tier 4: 2k`), these tools query the
uncondensed `Tier 0/1` `ExtractedChart` in memory while strictly enforcing
`sharing_posture` (`external_share` redacts `SDOH/PRIVACY-SENSITIVE` records under
42 CFR Part 2 and the WA My Health My Data Act).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .extractors import ClinicalItem, ExtractedChart, extract_bundle
from .normalizers import classify_sensitivity


def _ensure_extracted(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
) -> ExtractedChart:
    if isinstance(bundle_or_extracted, ExtractedChart):
        return bundle_or_extracted
    return extract_bundle(bundle_or_extracted)


def _is_privacy_redacted(item: ClinicalItem, sharing_posture: str) -> bool:
    return (
        sharing_posture == "external_share"
        and item.sensitivity == "SDOH/PRIVACY-SENSITIVE"
    )


def search_chart_entries(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    query: str = "",
    *,
    section: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sharing_posture: str = "patient_self",
    limit: int = 15,
) -> Dict[str, Any]:
    """Search uncondensed `Tier 0/1` chart entries by keyword, section, and date window.

    Enforces `sharing_posture`: when `sharing_posture == 'external_share'`, all items
    tagged `SDOH/PRIVACY-SENSITIVE` are withheld and counted in `redacted_for_privacy_count`.
    """
    extracted = _ensure_extracted(bundle_or_extracted)
    raw_tokens = [
        tok
        for tok in re.findall(r"[a-z0-9.\-]+", (query or "").lower())
        if len(tok) >= 2
    ]
    stopwords = {"record", "records", "status", "active", "condition", "medication", "observation", "the", "and", "for"}
    specific_tokens = [t for t in raw_tokens if t not in stopwords]
    q_tokens = specific_tokens or raw_tokens

    matches: List[Dict[str, Any]] = []
    redacted_count = 0

    for item in extracted.items:
        if item.section == "patient":
            continue
        if _is_privacy_redacted(item, sharing_posture):
            redacted_count += 1
            continue
        if section and section.lower() not in item.section.lower():
            continue
        item_date = item.clinical_date or item.date or ""
        if date_from and item_date and item_date < date_from:
            continue
        if date_to and item_date and item_date > date_to:
            continue

        haystack = f"{item.display} {item.text} {item.code} {item.section} {item.source_org or ''}".lower()
        if q_tokens:
            score = sum(1 for tok in q_tokens if tok in haystack)
            if score == 0:
                continue
        else:
            score = 1

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
                "conflict_note": item.conflict_note or "",
                "text": item.text,
            }
        )

    matches.sort(key=lambda m: (m["score"], m["date"] or ""), reverse=True)
    top_matches = matches[: max(1, int(limit))]
    return {
        "tool": "search_chart_entries",
        "query": query,
        "section": section,
        "date_from": date_from,
        "date_to": date_to,
        "sharing_posture": sharing_posture,
        "total_matches": len(matches),
        "returned_count": len(top_matches),
        "truncated": len(matches) > len(top_matches),
        "redacted_for_privacy_count": redacted_count,
        "results": top_matches,
    }


def get_lab_history(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    analyte_or_loinc: str,
    *,
    limit: int = 25,
    sharing_posture: str = "patient_self",
) -> Dict[str, Any]:
    """Retrieve longitudinal quantitative lab/vital measurements from the uncondensed chart."""
    extracted = _ensure_extracted(bundle_or_extracted)
    q_norm = (analyte_or_loinc or "").strip().lower()
    q_tokens = [t for t in re.findall(r"[a-z0-9.\-]+", q_norm) if len(t) >= 2]

    redacted_uuids = {
        u
        for it in extracted.items
        if _is_privacy_redacted(it, sharing_posture)
        for u in ([it.resource_uuid] + list(it.member_uuids))
        if u
    }
    pts = []
    redacted_count = 0
    for p in extracted.numeric_points:
        if p.resource_uuid in redacted_uuids or (
            sharing_posture == "external_share"
            and classify_sensitivity(p.code, p.concept, [p.category]) == "SDOH/PRIVACY-SENSITIVE"
        ):
            redacted_count += 1
            continue
        hay = f"{p.concept} {p.code} {p.category}".lower()
        if q_norm in hay or (q_tokens and any(t in hay for t in q_tokens)):
            pts.append(p)

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
        min_pt = min(pts, key=lambda x: x.value)
        max_pt = max(pts, key=lambda x: x.value)
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
        "sharing_posture": sharing_posture,
        "total_matches": len(pts),
        "returned_count": len(series),
        "truncated": len(pts) > len(series),
        "redacted_for_privacy_count": redacted_count,
        "summary": summary,
        "results": series,
    }


def read_clinical_note(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    resource_uuid: str = "",
    *,
    query: str = "",
    sharing_posture: str = "patient_self",
) -> Dict[str, Any]:
    """Read full-text clinical notes (`DocumentReference` / `DiagnosticReport`) by UUID or keyword.

    Enforces `sharing_posture`: withholds any note tagged `SDOH/PRIVACY-SENSITIVE` when
    `sharing_posture == 'external_share'`.
    """
    extracted = _ensure_extracted(bundle_or_extracted)
    target_uuid = (resource_uuid or "").strip()
    q_norm = (query or "").strip().lower()

    notes: List[Dict[str, Any]] = []
    redacted_count = 0

    for item in extracted.items:
        if item.section not in {"clinical_note", "note"}:
            continue
        if _is_privacy_redacted(item, sharing_posture):
            redacted_count += 1
            continue
        all_uuids = {item.resource_uuid} | set(item.member_uuids)
        if target_uuid and target_uuid not in all_uuids:
            continue
        raw_decoded = str(
            item.metadata.get("raw_decoded_text")
            or item.metadata.get("full_text")
            or item.text
            or ""
        ).strip()
        if q_norm and q_norm not in f"{item.display} {item.text} {raw_decoded}".lower():
            continue
        full_note_text = raw_decoded or item.text
        notes.append(
            {
                "resource_type": item.resource_type,
                "resource_uuid": item.resource_uuid,
                "member_uuids": list(item.member_uuids),
                "date": item.clinical_date or item.date or "",
                "display": item.display,
                "source_org": item.source_org or "",
                "conflict_note": item.conflict_note or "",
                "text": full_note_text,
                "full_text": full_note_text,
                "summary_text": item.text,
            }
        )

    return {
        "tool": "read_clinical_note",
        "resource_uuid": target_uuid,
        "query": query,
        "sharing_posture": sharing_posture,
        "total_matches": len(notes),
        "redacted_for_privacy_count": redacted_count,
        "results": notes,
    }


_RAG_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "been",
    "please",
    "provide",
    "review",
    "summarize",
    "summary",
    "state",
    "identify",
    "report",
    "chart",
    "patient",
    "record",
    "records",
    "cited",
    "parts",
    "below",
    "covering",
    "three",
    "my",
    "me",
    "your",
}


def _tokenize_for_bm25(text: str) -> List[str]:
    raw = re.findall(r"#?\d+(?:\.\d+)?|[a-z0-9][a-z0-9.\-]*", (text or "").lower())
    return [tok for tok in raw if (len(tok) >= 2 or tok.isdigit()) and tok not in _RAG_STOPWORDS]


def _chunk_note_words(
    text: str,
    *,
    chunk_words: int = 160,
    overlap_words: int = 35,
) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    step = max(25, chunk_words - overlap_words)
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def retrieve_clinical_note_passages(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    query: str,
    *,
    sharing_posture: str = "patient_self",
    chunk_words: int = 160,
    overlap_words: int = 35,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve top-k BM25-ranked single-patient clinical note & report passages.

    Indexes deduplicated full-text `DocumentReference` and `DiagnosticReport`
    entries within the single authorized patient chart while enforcing
    `sharing_posture` (`external_share` redacts `SDOH/PRIVACY-SENSITIVE` items).
    """
    extracted = _ensure_extracted(bundle_or_extracted)
    q_tokens = _tokenize_for_bm25(query)
    q_unique = list(dict.fromkeys(q_tokens))

    candidate_chunks: List[Dict[str, Any]] = []
    doc_tokens_list: List[List[str]] = []
    redacted_count = 0

    for item in extracted.items:
        is_note_or_report = (
            item.section in {"clinical_note", "note"}
            or item.resource_type in {"DocumentReference", "DiagnosticReport"}
        )
        if not is_note_or_report:
            continue
        if _is_privacy_redacted(item, sharing_posture):
            redacted_count += 1
            continue

        raw_decoded = str(
            item.metadata.get("raw_decoded_text")
            or item.metadata.get("full_text")
            or item.text
            or ""
        ).strip()
        full_body = raw_decoded or item.text
        if not full_body:
            continue

        item_date = item.clinical_date or item.date or ""
        header = f"{item.display} ({item_date or 'undated'}, {item.source_org or 'Clinical Record'})"
        passages = _chunk_note_words(
            full_body,
            chunk_words=max(40, int(chunk_words)),
            overlap_words=max(0, int(overlap_words)),
        )
        for p_idx, passage in enumerate(passages):
            combined = f"{header}: {passage}"
            toks = _tokenize_for_bm25(combined)
            if not toks:
                continue
            candidate_chunks.append(
                {
                    "resource_type": item.resource_type,
                    "resource_uuid": item.resource_uuid,
                    "member_uuids": list(item.member_uuids),
                    "section": item.section,
                    "date": item_date,
                    "display": item.display,
                    "source_org": item.source_org or "",
                    "chunk_index": p_idx,
                    "passage_text": passage,
                }
            )
            doc_tokens_list.append(toks)

    if not candidate_chunks:
        return {
            "tool": "retrieve_clinical_note_passages",
            "query": query,
            "sharing_posture": sharing_posture,
            "total_chunks_indexed": 0,
            "redacted_for_privacy_count": redacted_count,
            "results": [],
        }

    n_docs = len(doc_tokens_list)
    avgdl = sum(len(t) for t in doc_tokens_list) / max(1, n_docs)
    df: Dict[str, int] = {qt: 0 for qt in q_unique}
    for toks in doc_tokens_list:
        tset = set(toks)
        for qt in q_unique:
            if qt in tset:
                df[qt] += 1

    k1 = 1.5
    b = 0.75
    scored: List[Tuple[float, str, int]] = []
    for idx, toks in enumerate(doc_tokens_list):
        if not q_unique:
            scored.append((1.0, candidate_chunks[idx]["date"], idx))
            continue
        tf_map: Dict[str, int] = {}
        for tok in toks:
            if tok in df:
                tf_map[tok] = tf_map.get(tok, 0) + 1
        if not tf_map:
            continue
        dl = len(toks)
        bm25_score = 0.0
        for qt, tf in tf_map.items():
            doc_freq = df.get(qt, 0)
            idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            bm25_score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        if bm25_score > 0.0:
            scored.append((bm25_score, candidate_chunks[idx]["date"], idx))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    selected: List[Dict[str, Any]] = []
    per_resource_cap: Dict[str, int] = {}
    for bm25_val, _dt, c_idx in scored:
        cand = dict(candidate_chunks[c_idx])
        ru = cand["resource_uuid"]
        if per_resource_cap.get(ru, 0) >= 2:
            continue
        per_resource_cap[ru] = per_resource_cap.get(ru, 0) + 1
        cand["bm25_score"] = round(bm25_val, 4)
        selected.append(cand)
        if len(selected) >= max(1, int(top_k)):
            break

    return {
        "tool": "retrieve_clinical_note_passages",
        "query": query,
        "sharing_posture": sharing_posture,
        "total_chunks_indexed": n_docs,
        "redacted_for_privacy_count": redacted_count,
        "results": selected,
    }
