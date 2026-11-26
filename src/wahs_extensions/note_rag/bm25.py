"""Pure-stdlib single-patient Okapi BM25 clinical note retriever with unseen-span slicing & synonym bridge."""

from __future__ import annotations

from collections import OrderedDict
import math
import re
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import weakref

from wahs_core.extractors import ExtractedChart, extract_bundle
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_RAG_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "why", "how", "are", "was", "were", "have", "has", "had", "been",
    "please", "provide", "review", "summarize", "summary", "state", "identify",
    "report", "chart", "patient", "record", "records", "cited", "parts", "below",
    "covering", "three", "my", "me", "your",
}

_CHUNK_CACHE_LOCK = threading.Lock()
_CHUNK_CACHE: "OrderedDict[Tuple[int, Tuple[Tuple[str, int], ...], int, int, bool], Tuple[weakref.ref[ExtractedChart], List[Dict[str, Any]], List[Dict[str, int]], float]]" = OrderedDict()


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


def _build_indexed_chunks(
    extracted: ExtractedChart,
    horizons: Optional[Mapping[str, int]],
    *,
    chunk_words: int,
    overlap_words: int,
    unseen_only: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]], float]:
    h_tuple = tuple(sorted((str(k), int(v)) for k, v in (horizons or {}).items())) if unseen_only else ()
    ckey = (id(extracted), h_tuple, chunk_words, overlap_words, unseen_only)
    with _CHUNK_CACHE_LOCK:
        cached = _CHUNK_CACHE.get(ckey)
        if cached is not None and cached[0]() is extracted:
            _CHUNK_CACHE.move_to_end(ckey)
            return cached[1], cached[2], cached[3]

    candidates: List[Dict[str, Any]] = []
    tf_maps: List[Dict[str, int]] = []
    total_len = 0

    for item in extracted.items:
        is_note_or_rep = item.section in {"clinical_note", "note"} or item.resource_type in {
            "DocumentReference", "DiagnosticReport"
        }
        if not is_note_or_rep:
            continue
        full_body = str(
            item.metadata.get("full_text")
            or item.metadata.get("raw_decoded_text")
            or item.text
            or ""
        ).strip()
        if not full_body:
            continue

        slice_body = full_body
        if unseen_only and horizons is not None and item.resource_uuid in horizons:
            h_val = int(horizons[item.resource_uuid])
            if h_val >= len(full_body):
                continue
            if 0 < h_val < len(full_body):
                start_char = max(0, h_val - 80)
                slice_body = full_body[start_char:].strip()
                if not slice_body:
                    continue

        item_date = item.clinical_date or item.date or ""
        header = f"{item.display} ({item_date or 'undated'}, {item.source_org or 'Chart'})"
        for p_idx, passage in enumerate(
            _chunk_note_words(
                slice_body,
                chunk_words=max(40, int(chunk_words)),
                overlap_words=max(0, int(overlap_words)),
            )
        ):
            toks = _tokenize_bm25(f"{header}: {passage}")
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            tf["__dl__"] = len(toks)
            total_len += len(toks)
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
                    "unseen_slice": bool(unseen_only and slice_body != full_body),
                }
            )
            tf_maps.append(tf)

    avgdl = total_len / max(1, len(tf_maps))
    try:
        ref = weakref.ref(extracted)
        with _CHUNK_CACHE_LOCK:
            _CHUNK_CACHE[ckey] = (ref, candidates, tf_maps, avgdl)
            while len(_CHUNK_CACHE) > 12:
                _CHUNK_CACHE.popitem(last=False)
    except TypeError:
        pass
    return candidates, tf_maps, avgdl


def _score_bm25(
    candidates: List[Dict[str, Any]],
    tf_maps: List[Dict[str, int]],
    avgdl: float,
    q_weights: Dict[str, float],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    n_docs = len(tf_maps)
    df: Dict[str, int] = {qt: 0 for qt in q_weights}
    for tf in tf_maps:
        for qt in q_weights:
            if qt in tf:
                df[qt] += 1

    k1, b = 1.5, 0.75
    scored: List[Tuple[float, str, int]] = []
    for idx, tf in enumerate(tf_maps):
        if not q_weights:
            scored.append((1.0, candidates[idx]["date"], idx))
            continue
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
    return selected


def retrieve_clinical_note_passages(
    bundle_or_extracted: Union[Mapping[str, Any], ExtractedChart],
    query: str,
    *,
    rendered_note_horizons: Optional[Mapping[str, int]] = None,
    expand_terminology: bool = True,
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
    literal_tokens = list(dict.fromkeys(_tokenize_bm25(query)))
    q_weights: Dict[str, float] = {t: 1.0 for t in literal_tokens}
    if expand_terminology and query:
        from wahs_extensions.terminology_kb import expand_query_with_terminology

        for exp_tok in expand_query_with_terminology(query, extracted):
            for sub_tok in _tokenize_bm25(exp_tok):
                q_weights.setdefault(sub_tok, 0.6)

    used_unseen_only = False
    if rendered_note_horizons:
        cands_u, tfs_u, avg_u = _build_indexed_chunks(
            extracted, rendered_note_horizons, chunk_words=chunk_words, overlap_words=overlap_words, unseen_only=True
        )
        sel_u = _score_bm25(cands_u, tfs_u, avg_u, q_weights, top_k=top_k)
        if sel_u:
            return {
                "tool": "retrieve_clinical_note_passages",
                "query": query,
                "unseen_only": True,
                "total_chunks_indexed": len(tfs_u),
                "results": sel_u,
            }

    cands_f, tfs_f, avg_f = _build_indexed_chunks(
        extracted, None, chunk_words=chunk_words, overlap_words=overlap_words, unseen_only=False
    )
    sel_f = _score_bm25(cands_f, tfs_f, avg_f, q_weights, top_k=top_k)
    return {
        "tool": "retrieve_clinical_note_passages",
        "query": query,
        "unseen_only": used_unseen_only,
        "total_chunks_indexed": len(tfs_f),
        "results": sel_f,
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
    """Tier-2 unseen-span Okapi BM25 clinical note passage retriever (`ContextEnricher`)."""

    enricher_id = "note_rag"
    disclosure = EnrichmentDisclosure(
        enricher_id="note_rag",
        version="0.2.0",
        kind="retrieval",
        scope_summary=(
            "Single-patient Okapi BM25 passage retrieval over unseen mid/tail spans of excerpted "
            "or omitted clinical notes using CIEL/RxNorm synonym query expansion."
        ),
        limitations="Self-gated to Tier 2 (inactive when all notes already fit verbatim in the context window).",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        man = inp.condensed.completeness_manifest or {}
        if int(man.get("excerpted_notes_count") or 0) == 0 and int(man.get("omitted_notes_count") or 0) == 0:
            return None
        if inp.max_tokens < 48 or not (inp.question or "").strip():
            return None

        horizons = (inp.condensed.compression_stats or {}).get("rendered_note_horizons") or {}
        rag_res = retrieve_clinical_note_passages(
            inp.extracted,
            inp.question,
            rendered_note_horizons=horizons,
            expand_terminology=True,
            top_k=6,
        )
        passages = rag_res.get("results") or []
        if not passages:
            return None

        heading = "=== RETRIEVED UNSEEN CLINICAL NOTE PASSAGES (BM25 RAG; cite as [N]) ==="
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
                "unseen_only": bool(rag_res.get("unseen_only")),
                "total_chunks_indexed": int(rag_res.get("total_chunks_indexed") or 0),
                "passages_retrieved": len(lines),
            },
        )
