"""In-memory SQLite FTS5 + chart-code intersection `PublicGuidelineEnricher` (`[K1]..[KM]`)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from wahs_core.extractors import ExtractedChart
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_GUIDELINES_PATH = Path(__file__).resolve().parent / "assets" / "public_guidelines.jsonl"
_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "why", "how", "are", "was", "were", "have", "has", "had", "been",
    "please", "review", "summarize", "summary", "chart", "patient", "record", "records",
    "should", "about", "there", "any", "my", "me", "your",
}

_RECORDS_LOCK = threading.Lock()
_RECORDS_CACHE: Optional[List[Dict[str, Any]]] = None
_TLS = threading.local()


def _load_records() -> List[Dict[str, Any]]:
    global _RECORDS_CACHE
    if _RECORDS_CACHE is not None:
        return _RECORDS_CACHE
    with _RECORDS_LOCK:
        if _RECORDS_CACHE is None:
            rows: List[Dict[str, Any]] = []
            if _GUIDELINES_PATH.is_file():
                for line in _GUIDELINES_PATH.read_text(encoding="utf-8").splitlines():
                    ln = line.strip()
                    if ln:
                        rows.append(json.loads(ln))
            _RECORDS_CACHE = rows
    return _RECORDS_CACHE


def _get_fts_conn() -> Tuple[sqlite3.Connection, bool]:
    cached = getattr(_TLS, "fts_pair", None)
    if cached is not None:
        return cached
    records = _load_records()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    has_fts5 = True
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE guidelines_fts USING fts5(gid UNINDEXED, title, keywords, body)"
        )
        for rec in records:
            conn.execute(
                "INSERT INTO guidelines_fts VALUES (?, ?, ?, ?)",
                (rec["id"], rec.get("title", ""), rec.get("keywords", ""), rec.get("text", "")),
            )
    except sqlite3.OperationalError:
        has_fts5 = False
    _TLS.fts_pair = (conn, has_fts5)
    return conn, has_fts5


def _extract_chart_codes(extracted: ExtractedChart) -> Dict[str, Set[str]]:
    by_sys: Dict[str, Set[str]] = {}
    for it in extracted.items:
        if it.section in {"patient", "manifest"}:
            continue
        for c in ((it.metadata or {}).get("codings") or ()):
            if isinstance(c, Mapping) and c.get("system") and c.get("code"):
                s = str(c["system"]).lower()
                code = str(c["code"])
                by_sys.setdefault(s, set()).add(code)
                if s == "atc" and len(code) >= 5:
                    by_sys.setdefault("atc", set()).add(code[:5])
    # Also derive RxNorm/ATC codes from active meds via terminology_kb when available
    try:
        from wahs_extensions.terminology_kb.enricher import _get_conn, _resolve_drug_row

        tconn = _get_conn()
        for it in extracted.items:
            if it.section == "med_active":
                drow = _resolve_drug_row(tconn, it)
                if drow:
                    if drow.get("rxcui_in"):
                        by_sys.setdefault("rxnorm", set()).add(str(drow["rxcui_in"]))
                    if drow.get("atc_code"):
                        by_sys.setdefault("atc", set()).add(str(drow["atc_code"]))
                    if drow.get("atc_class_code"):
                        by_sys.setdefault("atc", set()).add(str(drow["atc_class_code"]))
    except Exception:
        pass
    return by_sys


def _next_k_index(citation_map: Mapping[str, Any]) -> int:
    max_k = 0
    for k in citation_map:
        m = re.match(r"^K(\d+)$", str(k).strip())
        if m:
            max_k = max(max_k, int(m.group(1)))
    return max_k + 1


class PublicGuidelineEnricher:
    """Retrieves provenance-stamped public clinical reference snippets (`[K1]..[KM]`)."""

    enricher_id = "guideline_kb"
    disclosure = EnrichmentDisclosure(
        enricher_id="guideline_kb",
        version="0.2.0",
        kind="knowledge_reference",
        scope_summary=(
            "Retrieves concise public-domain clinical reference snippets (CDC ACIP, FDA/DailyMed, "
            "KDIGO, ADA, ACC/AHA, MedlinePlus, AHRQ) via SQLite FTS5 + chart coding intersection."
        ),
        limitations="Educational public guideline reference snippets ([K1]..[KM]) only; not patient-specific prescribing advice.",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        if inp.max_tokens < 64:
            return None
        records = _load_records()
        if not records:
            return None
        rec_by_id = {r["id"]: r for r in records}

        q_tokens = [
            t
            for t in re.findall(r"[a-z0-9]{3,}", (inp.question or "").lower())
            if t not in _STOPWORDS
        ]
        if inp.question:
            try:
                from wahs_extensions.terminology_kb import expand_query_with_terminology

                for exp in expand_query_with_terminology(inp.question, inp.extracted)[:10]:
                    for sub in re.findall(r"[a-z0-9]{3,}", exp.lower()):
                        if sub not in _STOPWORDS and sub not in q_tokens:
                            q_tokens.append(sub)
            except Exception:
                pass

        fts_scores: Dict[str, float] = {}
        conn, has_fts5 = _get_fts_conn()
        if has_fts5 and q_tokens:
            fts_query = " OR ".join(f'"{t}"' for t in q_tokens[:16] if t.isalnum())
            if fts_query:
                for row in conn.execute(
                    "SELECT gid, bm25(guidelines_fts, 0.0, 2.5, 2.0, 1.0) AS rank_score "
                    "FROM guidelines_fts WHERE guidelines_fts MATCH ?",
                    (fts_query,),
                ).fetchall():
                    fts_scores[str(row["gid"])] = max(0.1, -float(row["rank_score"]))
        elif q_tokens:
            for rec in records:
                hay = f"{rec.get('title', '')} {rec.get('keywords', '')} {rec.get('text', '')}".lower()
                hits = sum(1 for t in q_tokens if t in hay)
                if hits > 0:
                    fts_scores[rec["id"]] = float(hits)

        chart_codes = _extract_chart_codes(inp.extracted)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records:
            gid = rec["id"]
            f_score = fts_scores.get(gid, 0.0)
            code_hits = 0
            for sys_k, code_list in (rec.get("codes") or {}).items():
                present = chart_codes.get(sys_k.lower(), set())
                if present:
                    code_hits += sum(1 for c in code_list if str(c) in present)
            if f_score <= 0.0 and code_hits < 2:
                continue
            # Require either query match + chart relevance, or strong explicit query match
            total = f_score + (2.5 * min(code_hits, 4) if f_score > 0.0 else 0.8 * code_hits)
            if total >= 1.2:
                scored.append((total, rec_by_id[gid]))

        if not scored:
            return None
        scored.sort(key=lambda pair: (pair[0], pair[1]["id"]), reverse=True)

        heading = "=== PUBLIC CLINICAL REFERENCE SNIPPETS (cite as [K1]..[KM]) ==="
        lines: List[str] = []
        new_cits: Dict[str, Dict[str, Any]] = {}
        k_num = _next_k_index(inp.citation_map)

        for _sc, rec in scored[:3]:
            kid = f"K{k_num}"
            cand = f"[{kid}] ({rec['source']}, {rec['date']} - {rec['title']}): {rec['text']}"
            trial = "\n".join([heading, *lines, cand])
            if inp.token_counter(trial) > inp.max_tokens:
                break
            lines.append(cand)
            new_cits[kid] = {
                "index": kid,
                "resourceType": "KnowledgeReference",
                "id": rec["id"],
                "source": rec["source"],
                "date": rec["date"],
                "display": f"{rec['title']} ({rec['source']})",
            }
            k_num += 1

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
            metadata={"snippets_retrieved": len(lines)},
        )
