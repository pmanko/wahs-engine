"""In-memory read-only `sqlite3.Connection.deserialize` terminology & concept enricher."""

from __future__ import annotations

import gzip
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from wahs_core.extractors import ClinicalItem, ExtractedChart, NumericPoint
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SQLITE_GZ_PATH = _ASSETS_DIR / "terminology_kb.sqlite.gz"

_RAW_DB_BYTES: Optional[bytes] = None
_RAW_DB_LOCK = threading.Lock()
_TLS = threading.local()

_VACCINE_QUERY_CUES = (
    "vaccin", "immuniz", "shot", "booster", "preventive", "up to date", "up-to-date",
    "due", "shingrix", "shingles", "zoster", "zostavax", "covid", "tetanus", "tdap",
    "flu", "influenza", "pneumo", "prevnar", "rsv", "hepatitis", "hpv", "mmr",
)
_MED_QUERY_CUES = (
    "med", "medication", "drug", "pill", "prescription", "dose", "duplicat", "overlap",
    "interact", "statin", "cholesterol", "sedat", "sleep", "thinner", "anticoag",
    "bleed", "nsaid", "pain", "kidney", "renal", "insulin", "diabetes",
)
_LAB_QUERY_CUES = (
    "lab", "test", "result", "abnormal", "range", "a1c", "hba1c", "glucose", "sugar",
    "creatinine", "egfr", "kidney", "potassium", "sodium", "hemoglobin", "anemia",
    "tsh", "thyroid", "inr", "troponin", "bnp", "liver", "alt", "ast", "bilirubin",
)
_QUERY_SYNONYM_BRIDGES: Dict[str, Tuple[str, ...]] = {
    "a1c": ("hba1c", "hemoglobin", "4548-4", "glycated"),
    "hba1c": ("a1c", "hemoglobin", "4548-4", "glycated"),
    "sugar": ("glucose", "hypoglycemia", "hba1c", "insulin"),
    "hypoglycemia": ("glucose", "hba1c", "insulin", "glipizide", "sulfonylurea"),
    "kidney": ("renal", "creatinine", "egfr", "glomerular", "nephropathy"),
    "renal": ("kidney", "creatinine", "egfr", "glomerular"),
    "liver": ("hepatic", "aminotransferase", "alt", "ast", "bilirubin"),
    "thinner": ("anticoagulant", "warfarin", "apixaban", "rivaroxaban", "clopidogrel", "inr"),
    "anticoagulation": ("warfarin", "apixaban", "rivaroxaban", "inr", "bleeding"),
    "bleeding": ("anticoagulant", "warfarin", "apixaban", "clopidogrel", "ibuprofen", "naproxen", "inr"),
    "shingles": ("zoster", "shingrix", "zostavax", "187", "121"),
    "zoster": ("shingles", "shingrix", "zostavax", "187"),
    "statin": ("atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lipitor", "zocor"),
    "sedative": ("lorazepam", "ativan", "zolpidem", "ambien", "diazepam", "alprazolam", "benzodiazepine"),
    "sleep": ("zolpidem", "ambien", "trazodone", "lorazepam", "insomnia"),
    "thyroid": ("tsh", "thyrotropin", "levothyroxine", "synthroid"),
}
_GENERIC_DRUG_STOPWORDS: Set[str] = {
    "oral", "tablet", "tablets", "capsule", "capsules", "extended", "release", "delayed",
    "solution", "suspension", "injectable", "injection", "mucosal", "spray", "sodium",
    "hydrochloride", "potassium", "tartrate", "succinate", "sulfate", "calcium", "mg",
    "mcg", "ml", "unt", "meq", "actuat", "hr", "daily", "medication",
}


def _get_conn() -> sqlite3.Connection:
    global _RAW_DB_BYTES
    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        return conn
    if _RAW_DB_BYTES is None:
        with _RAW_DB_LOCK:
            if _RAW_DB_BYTES is None:
                if not _SQLITE_GZ_PATH.is_file():
                    raise FileNotFoundError(f"Missing compiled terminology KB at {_SQLITE_GZ_PATH}")
                _RAW_DB_BYTES = gzip.decompress(_SQLITE_GZ_PATH.read_bytes())
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    mem.deserialize(_RAW_DB_BYTES)
    _TLS.conn = mem
    return mem


def _uuid_to_cit(citation_map: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k_str, entry in citation_map.items():
        if not str(k_str).isdigit() or not isinstance(entry, Mapping):
            continue
        idx = int(k_str)
        if entry.get("id"):
            out.setdefault(str(entry["id"]), idx)
        for m in entry.get("memberIds") or ():
            if m:
                out.setdefault(str(m), idx)
    return out


def _item_codings(item: ClinicalItem) -> List[Dict[str, str]]:
    raw = (item.metadata or {}).get("codings")
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict) and c.get("system") and c.get("code")]
    return []


def _resolve_drug_row(conn: sqlite3.Connection, item: ClinicalItem) -> Optional[Dict[str, Any]]:
    codings = _item_codings(item)
    for c in codings:
        if c["system"] == "rxnorm":
            row = conn.execute(
                "SELECT * FROM rxnorm_drugs WHERE key_type = 'rxcui' AND key_val = ?",
                (c["code"],),
            ).fetchone()
            if row is not None:
                return dict(row)
    for c in codings:
        if c["system"] == "ciel":
            row = conn.execute(
                "SELECT * FROM rxnorm_drugs WHERE key_type = 'ciel' AND key_val = ?",
                (c["code"],),
            ).fetchone()
            if row is not None:
                return dict(row)
    if item.code and item.code.isdigit():
        row = conn.execute(
            "SELECT * FROM rxnorm_drugs WHERE key_type = 'rxcui' AND key_val = ?",
            (item.code,),
        ).fetchone()
        if row is not None:
            return dict(row)

    disp_low = (item.display or item.text or "").lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9/-]{2,}", disp_low) if w not in _GENERIC_DRUG_STOPWORDS]
    candidates: List[str] = []
    if len(words) >= 2:
        candidates.append(f"{words[0]} {words[1]}")
    candidates.extend(words[:4])
    for cand in candidates:
        row = conn.execute(
            "SELECT * FROM rxnorm_drugs WHERE key_type = 'name' AND key_val = ?",
            (cand,),
        ).fetchone()
        if row is not None:
            return dict(row)
    return None


def _resolve_vaccine_row(conn: sqlite3.Connection, item: ClinicalItem) -> Optional[Dict[str, Any]]:
    for c in _item_codings(item):
        if c["system"] == "cvx":
            row = conn.execute(
                "SELECT * FROM cvx_vaccines WHERE cvx_code = ?",
                (c["code"].lstrip("0") or c["code"],),
            ).fetchone() or conn.execute(
                "SELECT * FROM cvx_vaccines WHERE cvx_code = ?",
                (c["code"],),
            ).fetchone()
            if row is not None:
                return dict(row)
    if item.code and item.code.isdigit():
        row = conn.execute("SELECT * FROM cvx_vaccines WHERE cvx_code = ?", (item.code,)).fetchone()
        if row is not None:
            return dict(row)
    disp_low = (item.display or "").lower()
    if disp_low:
        for row in conn.execute("SELECT * FROM cvx_vaccines").fetchall():
            pats = [p for p in str(row["display_patterns"] or "").split("|") if p]
            if any(p in disp_low for p in pats):
                return dict(row)
    return None


def _units_compatible(obs_unit: str, ref_unit: str) -> bool:
    u1 = (obs_unit or "").strip().lower().replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("_", "")
    u2 = (ref_unit or "").strip().lower().replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("_", "")
    if not u1 or not u2 or u1 == u2:
        return True
    if "ml/min" in u1 and "ml/min" in u2:
        return True
    if u1 in {"10*3/ul", "k/ul", "10^3/ul"} and u2 in {"10*3/ul", "k/ul", "10^3/ul"}:
        return True
    if u1 == "inr" or u2 == "inr":
        return True
    return False


def _fmt_num(val: float) -> str:
    if float(val).is_integer():
        return str(int(val))
    return f"{val:.3g}" if abs(val) >= 100 else f"{val:.2f}".rstrip("0").rstrip(".")


def expand_query_with_terminology(
    question: str,
    extracted: Optional[ExtractedChart] = None,
) -> List[str]:
    """Return concise terminology/brand/synonym expansion tokens for `note_rag` & `overflow_lookup`."""
    q_low = (question or "").strip().lower()
    if not q_low:
        return []
    q_words = [w for w in re.findall(r"[a-z0-9][a-z0-9.\-]{2,}", q_low)]
    q_set = set(q_words)
    added: List[str] = []

    def _push(raw: str) -> None:
        for tok in re.findall(r"[a-z0-9][a-z0-9.\-]{2,}", (raw or "").lower()):
            if tok not in q_set and tok not in _GENERIC_DRUG_STOPWORDS and tok not in added:
                added.append(tok)

    for w in q_words:
        for syn in _QUERY_SYNONYM_BRIDGES.get(w, ()):
            _push(syn)

    try:
        conn = _get_conn()
    except FileNotFoundError:
        return added[:16]

    phrases = list(q_words[:12])
    for i in range(min(len(q_words) - 1, 10)):
        phrases.append(f"{q_words[i]} {q_words[i + 1]}")

    for ph in phrases:
        drow = conn.execute(
            "SELECT ingredient, brands, atc_class_name, rxcui_in FROM rxnorm_drugs WHERE key_type = 'name' AND key_val = ?",
            (ph,),
        ).fetchone()
        if drow is not None:
            _push(drow["ingredient"])
            _push(drow["rxcui_in"])
            for b in str(drow["brands"] or "").split("|"):
                _push(b)
            _push(drow["atc_class_name"])

        for trow in conn.execute(
            "SELECT c.preferred_name, c.synonyms FROM term_lookup t JOIN concepts c ON c.ciel_id = t.ciel_id WHERE t.term_norm = ? LIMIT 3",
            (ph,),
        ).fetchall():
            _push(trow["preferred_name"])
            for s in str(trow["synonyms"] or "").split("|")[:2]:
                _push(s)

    if extracted is not None:
        for it in extracted.items:
            if it.section == "med_active":
                drow = _resolve_drug_row(conn, it)
                if drow and (
                    drow["ingredient"] in q_low
                    or any(b.lower() in q_low for b in str(drow["brands"] or "").split("|") if b)
                    or any(c in q_low for c in ("med", "drug", "statin", "sedat", "thinner", "pain"))
                ):
                    _push(drow["ingredient"])
                    for b in str(drow["brands"] or "").split("|"):
                        _push(b)
            if len(added) >= 18:
                break

    return added[:18]


class TerminologyConceptEnricher:
    """Offline `CIEL v2026-04-28` + Public Standards (`RxNorm/ATC`, `CDC CVX/ACIP`, `LOINC`) enricher."""

    enricher_id = "terminology_kb"
    disclosure = EnrichmentDisclosure(
        enricher_id="terminology_kb",
        version="0.2.0",
        kind="terminology_grounding",
        scope_summary=(
            "Resolves chart codings and displays against CIEL v2026-04-28, RxNorm/WHO-ATC drug classes, "
            "CDC CVX/ACIP adult immunization series schedules, and missing-range LOINC/CIEL reference bounds."
        ),
        limitations=(
            "Descriptive concept normalization and public reference intervals only; never overrides explicit "
            "chart Observation.referenceRange values and never issues prescribing directives."
        ),
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        if inp.max_tokens < 48:
            return None
        try:
            conn = _get_conn()
        except FileNotFoundError:
            return None

        cit_lookup = _uuid_to_cit(inp.citation_map)
        q_low = (inp.question or "").lower()
        q_tokens = {t for t in re.findall(r"[a-z0-9]{3,}", q_low)}
        med_query = any(c in q_low for c in _MED_QUERY_CUES)
        vac_query = any(c in q_low for c in _VACCINE_QUERY_CUES)
        lab_query = any(c in q_low for c in _LAB_QUERY_CUES)

        med_lines: List[str] = []
        overlap_lines: List[str] = []
        vac_lines: List[str] = []
        lab_lines: List[str] = []
        concept_lines: List[str] = []

        # 1. Active Medication Resolution & Factual Class/Ingredient Overlaps
        active_meds: List[Tuple[ClinicalItem, Dict[str, Any], str]] = []
        for it in inp.extracted.items:
            if it.section != "med_active":
                continue
            drow = _resolve_drug_row(conn, it)
            if drow is None:
                continue
            cit = f"[{cit_lookup[it.resource_uuid]}]" if it.resource_uuid in cit_lookup else ""
            active_meds.append((it, drow, cit))

        by_ingredient: Dict[str, List[Tuple[ClinicalItem, Dict[str, Any], str]]] = {}
        by_atc_class: Dict[str, List[Tuple[ClinicalItem, Dict[str, Any], str]]] = {}
        by_pharm_group: Dict[str, List[Tuple[ClinicalItem, Dict[str, Any], str]]] = {}
        overlap_covered_uuids: Set[str] = set()

        for entry in active_meds:
            it, drow, _cit = entry
            by_ingredient.setdefault(drow["ingredient"], []).append(entry)
            if drow["atc_class_code"]:
                by_atc_class.setdefault(drow["atc_class_code"], []).append(entry)
            if drow["pharm_group"]:
                by_pharm_group.setdefault(drow["pharm_group"], []).append(entry)

        for ing, grp in by_ingredient.items():
            if len(grp) >= 2:
                d0 = grp[0][1]
                for x in grp:
                    overlap_covered_uuids.add(x[0].resource_uuid)
                refs = ", ".join(f"{x[0].display}{(' ' + x[2]) if x[2] else ''}" for x in grp)
                overlap_lines.append(
                    f"- Same Active Ingredient ({ing}, RxNorm:{d0['rxcui_in']}, ATC:{d0['atc_code']}): "
                    f"{len(grp)} active orders share identical ingredient ({refs})."
                )

        for atc_cls, grp in by_atc_class.items():
            distinct_ings = list(dict.fromkeys(x[1]["ingredient"] for x in grp))
            if len(distinct_ings) >= 2:
                for x in grp:
                    overlap_covered_uuids.add(x[0].resource_uuid)
                cls_name = next((x[1]["atc_class_name"] for x in grp if x[1]["atc_class_name"]), f"ATC {atc_cls}")
                desc_items = " and ".join(
                    f"{x[0].display}{(' ' + x[2]) if x[2] else ''} "
                    f"({x[1]['ingredient']}, RxNorm:{x[1]['rxcui_in']}, ATC:{x[1]['atc_code']})"
                    for x in grp
                )
                overlap_lines.append(
                    f"- Shared Class (ATC {atc_cls} - {cls_name}): "
                    f"{len(grp)} active distinct drugs: {desc_items}."
                )

        cns_grp = by_pharm_group.get("cns_sedative_hypnotic", [])
        if len(cns_grp) >= 2 and len({x[1]["atc_class_code"] for x in cns_grp}) >= 2:
            for x in cns_grp:
                overlap_covered_uuids.add(x[0].resource_uuid)
            desc_cns = " and ".join(
                f"{x[0].display}{(' ' + x[2]) if x[2] else ''} "
                f"({x[1]['ingredient']}, {x[1]['atc_class_name'] or x[1]['atc_code']}, ATC:{x[1]['atc_code']})"
                for x in cns_grp
            )
            overlap_lines.append(
                f"- Concurrent CNS Sedative-Hypnotics ({len(cns_grp)} active): {desc_cns}."
            )

        antithrombotics = by_pharm_group.get("anticoagulant", []) + by_pharm_group.get("antiplatelet", [])
        nsaids = by_pharm_group.get("nsaid", [])
        if antithrombotics and nsaids:
            at_s = ", ".join(f"{x[0].display}{(' ' + x[2]) if x[2] else ''} ({x[1]['ingredient']})" for x in antithrombotics)
            ns_s = ", ".join(f"{x[0].display}{(' ' + x[2]) if x[2] else ''} ({x[1]['ingredient']})" for x in nsaids)
            overlap_lines.append(
                f"- Concurrent Active Antithrombotic + NSAID Classes: antithrombotic [{at_s}] co-listed with systemic NSAID [{ns_s}]."
            )

        for it, drow, cit in active_meds:
            if it.resource_uuid in overlap_covered_uuids:
                continue
            ing = drow["ingredient"]
            disp_low = (it.display or "").lower()
            is_brand_or_ciel = (ing not in disp_low) or any(c["system"] == "ciel" for c in _item_codings(it))
            matches_q = any(t in disp_low or t in ing or t in (drow["atc_class_name"] or "").lower() for t in q_tokens)
            if is_brand_or_ciel or med_query or matches_q or len(active_meds) <= 8:
                codes_bits = [
                    *(f"RxNorm:{drow['rxcui_in']}" for _ in (1,) if drow["rxcui_in"]),
                    *(f"CIEL:{drow['ciel_id']}" for _ in (1,) if drow["ciel_id"]),
                    *(f"ATC:{drow['atc_code']}" for _ in (1,) if drow["atc_code"]),
                ]
                cls_lbl = f" - {drow['atc_class_name']}" if drow["atc_class_name"] else ""
                cite_sp = f" {cit}" if cit else ""
                med_lines.append(
                    f"- Active Medication: {it.display}{cite_sp} -> {ing} ({', '.join(codes_bits)}{cls_lbl})"
                )

        # 2. CDC CVX / ACIP Immunization Series & Dose Counts
        imm_groups: Dict[str, Dict[str, Any]] = {}
        for it in inp.extracted.items:
            if it.section != "immunization" or it.status != "completed":
                continue
            vrow = _resolve_vaccine_row(conn, it)
            if vrow is None:
                continue
            vgrp = vrow["vaccine_group"]
            bucket = imm_groups.setdefault(
                vgrp,
                {"vrow": vrow, "dates": [], "cits": [], "codes": []},
            )
            if it.date:
                bucket["dates"].append(it.date)
            if it.resource_uuid in cit_lookup:
                bucket["cits"].append(cit_lookup[it.resource_uuid])
            for m_id in it.member_uuids:
                if m_id in cit_lookup:
                    bucket["cits"].append(cit_lookup[m_id])
            if vrow["cvx_code"] not in bucket["codes"]:
                bucket["codes"].append(vrow["cvx_code"])

        for vgrp, info in imm_groups.items():
            vrow = info["vrow"]
            sdoses = int(vrow["series_doses"] or 1)
            # In Tier 0 each dose is 1 item; in Tier 1 member_uuids tracks folded doses
            dose_dates = sorted(info["dates"])
            admin_cnt = max(len(dose_dates), 1)
            for it in inp.extracted.items:
                if it.section == "immunization" and it.status == "completed":
                    r_check = _resolve_vaccine_row(conn, it)
                    if r_check and r_check["vaccine_group"] == vgrp and len(it.member_uuids) > admin_cnt:
                        admin_cnt = len(it.member_uuids)
            incomplete_multidose = sdoses >= 2 and admin_cnt < sdoses
            q_match = vac_query or any(t in vgrp.lower() for t in q_tokens)
            if q_match or incomplete_multidose:
                cites = "".join(f"[{i}]" for i in sorted(set(info["cits"]))[:3])
                dt_str = f" ({', '.join(dose_dates)})" if dose_dates else ""
                cvx_tag = f"CVX:{','.join(info['codes'])}"
                inc_note = f"; series incomplete ({admin_cnt}/{sdoses} doses)" if incomplete_multidose else ""
                vac_lines.append(
                    f"- Vaccine ({vgrp}, {cvx_tag}): {admin_cnt} administered{dt_str}"
                    f"{(' ' + cites) if cites else ''} vs {sdoses}-dose adult schedule ({vrow['interval_summary']}){inc_note}."
                )

        # 3. Missing-Range Numeric Labs (STRICTLY only when Observation.referenceRange is absent)
        obs_item_by_uuid = {it.resource_uuid: it for it in inp.extracted.items if it.resource_uuid}
        latest_missing_pts: Dict[Tuple[str, str], NumericPoint] = {}
        for pt in inp.extracted.numeric_points:
            if pt.reference_range or (pt.status or "").lower() in {"entered-in-error", "cancelled"}:
                continue
            key = ((pt.code or pt.concept).lower(), (pt.unit or "").lower())
            prev = latest_missing_pts.get(key)
            if prev is None or (pt.date or "") >= (prev.date or ""):
                latest_missing_pts[key] = pt

        for pt in latest_missing_pts.values():
            lrow = conn.execute(
                "SELECT * FROM lab_ranges WHERE system = 'loinc' AND code = ?",
                (pt.code,),
            ).fetchone()
            if lrow is None and pt.resource_uuid in obs_item_by_uuid:
                for c in _item_codings(obs_item_by_uuid[pt.resource_uuid]):
                    if c["system"] in {"loinc", "ciel"}:
                        lrow = conn.execute(
                            "SELECT * FROM lab_ranges WHERE system = ? AND code = ?",
                            (c["system"], c["code"]),
                        ).fetchone()
                        if lrow is not None:
                            break
            if lrow is None or not _units_compatible(pt.unit, lrow["unit"]):
                continue
            lo_n, hi_n = lrow["low_normal"], lrow["hi_normal"]
            lo_c, hi_c = lrow["low_critical"], lrow["hi_critical"]
            is_low = lo_n is not None and pt.value < lo_n
            is_high = hi_n is not None and pt.value > hi_n
            out_of_range = is_low or is_high
            matches_lab_q = lab_query or any(
                t in (lrow["analyte_name"] or "").lower() or t in (lrow["aliases"] or "")
                for t in q_tokens
            )
            if not out_of_range and not matches_lab_q:
                continue
            cit = f" [{cit_lookup[pt.resource_uuid]}]" if pt.resource_uuid in cit_lookup else ""
            u_lbl = pt.unit or lrow["unit"] or ""
            u_sp = f" {u_lbl}" if u_lbl and not u_lbl.startswith("{") else ""
            status_phrase = "within standard reference range"
            if is_low:
                crit_tag = f" and below critical threshold ({_fmt_num(lo_c)}{u_sp})" if (lo_c is not None and pt.value < lo_c) else ""
                status_phrase = f"below normal reference range{crit_tag}"
            elif is_high:
                crit_tag = f" and above critical threshold ({_fmt_num(hi_c)}{u_sp})" if (hi_c is not None and pt.value > hi_c) else ""
                status_phrase = f"above normal reference range{crit_tag}"
            short_ref = lrow["reference_text"].split("(")[0].strip() or lrow["reference_text"]
            short_note = lrow["clinical_note"].split(";")[0].strip() if (out_of_range and lrow["clinical_note"]) else ""
            note_sfx = f" ({short_note})" if short_note else ""
            lab_lines.append(
                f"- Missing-Range Lab ({lrow['analyte_name']}, {pt.date or 'undated'}): "
                f"{_fmt_num(pt.value)}{u_sp}{cit} | standard reference {short_ref} -> {status_phrase}{note_sfx}."
            )

        # 4. CIEL / SNOMED Concept Definitions & Cross-Walks
        seen_ciels: Set[int] = set()
        for it in inp.extracted.items:
            if it.section in {"patient", "manifest", "clinical_note"}:
                continue
            codings = _item_codings(it)
            cid_val: Optional[int] = None
            for c in codings:
                if c["system"] == "ciel" and c["code"].isdigit():
                    cid_val = int(c["code"])
                    break
            has_explicit_ciel = cid_val is not None
            if cid_val is None:
                item_rel = any(t in (it.display or "").lower() for t in q_tokens)
                if not item_rel:
                    continue
                for c in codings:
                    if c["system"] in {"snomed", "icd10", "loinc", "rxnorm"}:
                        mrow = conn.execute(
                            "SELECT ciel_id FROM code_mappings WHERE system = ? AND code = ? ORDER BY map_type DESC LIMIT 1",
                            (c["system"], c["code"]),
                        ).fetchone()
                        if mrow is not None:
                            cid_val = int(mrow["ciel_id"])
                            break
            if cid_val is None or cid_val in seen_ciels:
                continue
            crow = conn.execute("SELECT * FROM concepts WHERE ciel_id = ?", (cid_val,)).fetchone()
            if crow is None:
                continue
            seen_ciels.add(cid_val)
            if not has_explicit_ciel and not crow["synonyms"] and not crow["description"]:
                continue
            m_rows = conn.execute(
                "SELECT system, code FROM code_mappings WHERE ciel_id = ? AND map_type = 'S' LIMIT 2",
                (cid_val,),
            ).fetchall()
            xwalk = ", ".join(f"{r['system'].upper()}:{r['code']}" for r in m_rows)
            syns = ", ".join(s for s in str(crow["synonyms"] or "").split("|") if s)[:60]
            cit = f" [{cit_lookup[it.resource_uuid]}]" if it.resource_uuid in cit_lookup else ""
            parts = [f"CIEL:{cid_val} {crow['preferred_name']}{cit}"]
            if syns:
                parts.append(f"aka {syns}")
            if xwalk:
                parts.append(xwalk)
            concept_lines.append(f"- Concept ({crow['concept_class']}): {' | '.join(parts)}")
            if len(concept_lines) >= 4:
                break

        # Prioritize highest-signal lines within token cap (default <= 550 tokens)
        ordered_candidates = (
            overlap_lines
            + vac_lines
            + lab_lines
            + med_lines[:8]
            + concept_lines
        )
        if not ordered_candidates:
            return None

        heading = "=== TERMINOLOGY & CLINICAL CONCEPT GROUNDING (CIEL v2026-04-28 + RxNorm/ATC/CVX/LOINC) ==="
        chosen: List[str] = []
        for cand in ordered_candidates:
            trial = "\n".join([heading, *chosen, cand])
            if inp.token_counter(trial) <= inp.max_tokens:
                chosen.append(cand)
            elif not chosen:
                break

        if not chosen:
            return None
        block = "\n".join([heading, *chosen])
        return EnrichmentOutput(
            enricher_id=self.enricher_id,
            heading=heading,
            prompt_block=block,
            token_count=inp.token_counter(block),
            disclosure=self.disclosure,
            new_citations={},
            metadata={
                "active_meds_resolved": len(active_meds),
                "class_overlaps": len(overlap_lines),
                "vaccine_groups": len(vac_lines),
                "missing_range_labs": len(lab_lines),
                "ciel_concepts": len(concept_lines),
            },
        )
