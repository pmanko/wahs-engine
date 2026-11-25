"""Pure-stdlib single-patient Okapi BM25 clinical note passage retriever."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from wahs_core.extractors import ExtractedChart, extract_bundle
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_RAG_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "why", "how", "are", "was", "were", "have", "has", "had", "been",
    "please", "provide", "review", "summarize", "summary", "state", "identify",
    "report", "chart", "patient", "record", "records", "cited", "parts", "below",
    "covering", "three", "my", "me", "your",
}


def _tokenize_bm25(text: str) -> List[str]:
    raw = re.findall(r"#?\d+(?:\.\d+)?|[a-z0-9][a-z0-9.\-]*", (text or "").lower())
    return [tok for tok in raw if (len(tok) >= 2 or tok.isdigit()) and tok not in _RAG_STOPWORDS]


def _chunk_note_words(text: str, *, chunk_words: int = 160, overlap_words: int = 35) -> List[str]:
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
    chunk_words: int = 160,
    overlap_words: int = 35,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve top-k Okapi BM25 clinical note & report passages for a single patient chart."""
    extracted = (
        bundle_or_extracted
        if isinstance(bundle_or_extracted, ExtractedChart)
        else extract_bundle(bundle_or_extracted)
    )
    q_unique = list(dict.fromkeys(_tokenize_bm25(query)))
    candidates: List[Dict[str, Any]] = []
    doc_tokens: List[List[str]] = []

    for item in extracted.items:
        is_note_or_rep = item.section in {"clinical_note", "note"} or item.resource_type in {
            "DocumentReference", "DiagnosticReport"
        }
        if not is_note_or_rep:
            continue
        full_body = str(
            item.metadata.get("raw_decoded_text")
            or item.metadata.get("full_text")
            or item.text
            or ""
        ).strip()
        if not full_body:
            continue
        item_date = item.clinical_date or item.date or ""
        header = f"{item.display} ({item_date or 'undated'}, {item.source_org or 'Chart'})"
        for p_idx, passage in enumerate(
            _chunk_note_words(
                full_body,
                chunk_words=max(40, int(chunk_words)),
                overlap_words=max(0, int(overlap_words)),
            )
        ):
            toks = _tokenize_bm25(f"{header}: {passage}")
            if not toks:
                continue
            candidates.append(
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
            doc_tokens.append(toks)

    if not candidates:
        return {"tool": "retrieve_clinical_note_passages", "query": query, "total_chunks_indexed": 0, "results": []}

    n_docs = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens) / max(1, n_docs)
    df: Dict[str, int] = {qt: 0 for qt in q_unique}
    for toks in doc_tokens:
        tset = set(toks)
        for qt in q_unique:
            if qt in tset:
                df[qt] += 1

    k1, b = 1.5, 0.75
    scored: List[Tuple[float, str, int]] = []
    for idx, toks in enumerate(doc_tokens):
        if not q_unique:
            scored.append((1.0, candidates[idx]["date"], idx))
            continue
        tf_map: Dict[str, int] = {}
        for tok in toks:
            if tok in df:
                tf_map[tok] = tf_map.get(tok, 0) + 1
        if not tf_map:
            continue
        dl = len(toks)
        score = 0.0
        for qt, tf in tf_map.items():
            doc_f = df.get(qt, 0)
            idf = math.log(1.0 + (n_docs - doc_f + 0.5) / (doc_f + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        if score > 0.0:
            scored.append((score, candidates[idx]["date"], idx))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    selected: List[Dict[str, Any]] = []
    per_res_cap: Dict[str, int] = {}
    for bm25_val, _dt, c_idx in scored:
        cand = dict(candidates[c_idx])
        ru = cand["resource_uuid"]
        if per_res_cap.get(ru, 0) >= 2:
            continue
        per_res_cap[ru] = per_res_cap.get(ru, 0) + 1
        cand["bm25_score"] = round(bm25_val, 4)
        selected.append(cand)
        if len(selected) >= max(1, int(top_k)):
            break

    return {
        "tool": "retrieve_clinical_note_passages",
        "query": query,
        "total_chunks_indexed": n_docs,
        "results": selected,
    }


def _find_or_register_citation(
    citation_map: Mapping[str, Dict[str, Any]],
    new_citations: Dict[str, Dict[str, Any]],
    *,
    resource_uuid: str,
    resource_type: str,
    display: str,
    date: str,
    source: str,
    section: str,
    member_uuids: Sequence[str],
) -> int | None:
    ruuid = (resource_uuid or "").strip()
    if not ruuid:
        return None
    cand_ids = {ruuid, *[str(m).strip() for m in member_uuids if m]}
    for pool in (citation_map, new_citations):
        for k_str, entry in pool.items():
            if not str(k_str).isdigit():
                continue
            eid = str(entry.get("id") or "").strip()
            emems = {str(m).strip() for m in (entry.get("memberIds") or []) if m}
            if eid in cand_ids or (emems & cand_ids):
                return int(k_str)
    all_nums = [int(k) for k in citation_map if str(k).isdigit()] + [
        int(k) for k in new_citations if str(k).isdigit()
    ]
    next_idx = (max(all_nums) if all_nums else 0) + 1
    entry_obj: Dict[str, Any] = {
        "index": next_idx,
        "resourceType": resource_type or "DocumentReference",
        "id": ruuid,
        "source": source or "Clinical Note",
        "section": section or "clinical_note",
        "date": date or "",
        "display": display or ruuid,
    }
    if member_uuids:
        entry_obj["memberIds"] = list(member_uuids)
    new_citations[str(next_idx)] = entry_obj
    return next_idx


class BM25NotePassageEnricher:
    """Single-patient Okapi BM25 passage retriever (`ContextEnricher`)."""

    enricher_id = "note_rag"
    disclosure = EnrichmentDisclosure(
        enricher_id="note_rag",
        version="0.1.0",
        kind="retrieval",
        scope_summary="Single-patient Okapi BM25 passage retrieval over excerpted or omitted clinical notes and diagnostic reports.",
        limitations="Lexical TF-IDF term matching only; inactive when all notes already fit verbatim in the context window.",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        man = inp.condensed.completeness_manifest or {}
        if int(man.get("excerpted_notes_count") or 0) == 0 and int(man.get("omitted_notes_count") or 0) == 0:
            return None
        if inp.max_tokens < 48 or not (inp.question or "").strip():
            return None

        rag_res = retrieve_clinical_note_passages(inp.extracted, inp.question, top_k=5)
        passages = rag_res.get("results") or []
        if not passages:
            return None

        heading = "=== RETRIEVED CLINICAL NOTE PASSAGES (BM25 RAG; cite as [N]) ==="
        lines: List[str] = []
        new_cits: Dict[str, Dict[str, Any]] = {}
        for p in passages:
            c_idx = _find_or_register_citation(
                inp.citation_map,
                new_cits,
                resource_uuid=str(p.get("resource_uuid") or ""),
                resource_type=str(p.get("resource_type") or "DocumentReference"),
                display=str(p.get("display") or "Clinical Note"),
                date=str(p.get("date") or ""),
                source=str(p.get("source_org") or "Clinical Note"),
                section=str(p.get("section") or "clinical_note"),
                member_uuids=p.get("member_uuids") or (),
            )
            cite_pfx = f"[{c_idx}] " if c_idx is not None else ""
            cand = (
                f"{cite_pfx}({p.get('date') or 'undated'} | {p.get('display')} | "
                f"bm25={p.get('bm25_score')}): {p.get('passage_text')}"
            )
            trial = "\n".join([heading, *lines, cand])
            if inp.token_counter(trial) <= inp.max_tokens:
                lines.append(cand)
            elif not lines:
                break
            else:
                break

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
            metadata={
                "total_chunks_indexed": int(rag_res.get("total_chunks_indexed") or 0),
                "passages_retrieved": len(lines),
            },
        )
