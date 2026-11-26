#!/usr/bin/env python3
"""Compile dual-source `terminology_kb.sqlite.gz` from CIEL v2026-04-28 + public standards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

DEFAULT_CIEL_ZIP = Path(
    "/Users/pmanko/code/clinical-ai-validation-harness/datasets/sources/ocl/CIEL/v2026-04-28/CIEL_v2026-04-28.zip"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = REPO_ROOT / "data_sources" / "public_standards"
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "wahs_extensions" / "terminology_kb" / "assets"

KEPT_CIEL_CLASSES: Set[str] = {
    "Drug",
    "Diagnosis",
    "Test",
    "Procedure",
    "Finding",
    "Symptom/Finding",
    "Radiology/Imaging Procedure",
    "Pharmacologic Drug Class",
}

SOURCE_TAG_MAP: Dict[str, str] = {
    "snomed-ct": "snomed",
    "snomed ct": "snomed",
    "rxnorm": "rxnorm",
    "rxnorm-comb": "rxnorm",
    "whoatc": "atc",
    "loinc": "loinc",
    "icd-10-who": "icd10",
    "icd-10-who-2nd": "icd10",
    "ciel": "ciel",
}


def _clean_text(raw: Optional[str]) -> str:
    if not raw:
        return ""
    s = str(raw).replace("\u2014", " - ").replace("\u2013", "-").replace("&mdash;", " - ").replace("&#8212;", " - ")
    return re.sub(r"\s+", " ", s).strip()


def _one_sentence(raw: Optional[str], max_chars: int = 160) -> str:
    txt = _clean_text(raw)
    if not txt:
        return ""
    m = re.match(r"^(.+?[.!?])(?:\s|$)", txt)
    sent = m.group(1).strip() if m else txt
    if len(sent) > max_chars:
        sent = sent[: max_chars - 3].rstrip(" ,;:-") + "..."
    return sent


def _norm_term(raw: str) -> str:
    return re.sub(r"[^a-z0-9/+ -]+", " ", _clean_text(raw).lower()).strip()


def _float_or_none(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_kb(*, ciel_zip: Path, public_dir: Path, out_dir: Path) -> Dict[str, Any]:
    if not ciel_zip.is_file():
        raise FileNotFoundError(f"CIEL zip not found at {ciel_zip}")

    zip_bytes = ciel_zip.read_bytes()
    zip_sha = _sha256_bytes(zip_bytes)
    prov_file = ciel_zip.parent / "provenance.json"
    if prov_file.is_file():
        prov_meta = json.loads(prov_file.read_text(encoding="utf-8"))
        expected_sha = str(prov_meta.get("sha256") or "").strip()
        if expected_sha and expected_sha != zip_sha:
            raise ValueError(f"CIEL zip SHA-256 mismatch: expected {expected_sha}, got {zip_sha}")

    with zipfile.ZipFile(ciel_zip) as zf:
        export_obj = json.loads(zf.read(zf.namelist()[0]))

    rxnorm_pub = json.loads((public_dir / "rxnorm_atc_public.json").read_text(encoding="utf-8"))
    cvx_pub = json.loads((public_dir / "cdc_cvx_acip_public.json").read_text(encoding="utf-8"))
    loinc_pub = json.loads((public_dir / "loinc_ranges_public.json").read_text(encoding="utf-8"))

    kept_concepts: Dict[int, Tuple[int, str, str, str, str]] = {}
    ciel_name_to_id: Dict[str, int] = {}
    ciel_numeric_extras: Dict[int, Dict[str, Any]] = {}
    term_rows: Set[Tuple[str, int]] = set()

    for c in export_obj.get("concepts") or ():
        if c.get("retired"):
            continue
        cclass = _clean_text(c.get("concept_class"))
        if cclass not in KEPT_CIEL_CLASSES:
            continue
        raw_id = str(c.get("id") or "").strip()
        if not raw_id.isdigit():
            continue
        cid = int(raw_id)

        en_names: List[Tuple[int, str]] = []
        for n in c.get("names") or ():
            if str(n.get("locale") or "").lower() != "en":
                continue
            nm = _clean_text(n.get("name"))
            if not nm:
                continue
            rank = 0 if n.get("locale_preferred") else (1 if n.get("name_type") == "FULLY_SPECIFIED" else 2)
            en_names.append((rank, nm))
        en_names.sort(key=lambda pair: (pair[0], len(pair[1])))

        pref = en_names[0][1] if en_names else _clean_text(c.get("display_name"))
        if not pref:
            continue

        seen_low = {pref.lower()}
        syns: List[str] = []
        for _, nm in en_names[1:]:
            nl = nm.lower()
            if nl not in seen_low and len(nm) <= 72:
                seen_low.add(nl)
                syns.append(nm)
                if len(syns) >= 3:
                    break

        desc = ""
        if cclass in {"Drug", "Test", "Procedure", "Pharmacologic Drug Class"} or syns:
            for d in c.get("descriptions") or ():
                if str(d.get("locale") or "").lower() == "en" and d.get("description"):
                    cand_d = _one_sentence(d.get("description"), max_chars=140)
                    if cand_d and cand_d.lower().rstrip(".") != pref.lower().rstrip("."):
                        desc = cand_d
                        break

        kept_concepts[cid] = (cid, cclass, pref, "|".join(syns), desc)
        if cclass == "Drug":
            ciel_name_to_id.setdefault(pref.lower(), cid)
            for s in syns:
                ciel_name_to_id.setdefault(s.lower(), cid)

        if syns and cclass in {"Drug", "Diagnosis", "Test", "Procedure", "Pharmacologic Drug Class"}:
            pnorm = _norm_term(pref)
            if 3 <= len(pnorm) <= 48:
                term_rows.add((pnorm, cid))
            for s in syns:
                snorm = _norm_term(s)
                if 3 <= len(snorm) <= 48 and snorm != pnorm:
                    term_rows.add((snorm, cid))

        extras = c.get("extras") or {}
        if c.get("datatype") == "Numeric" and isinstance(extras, dict):
            lo_n = _float_or_none(extras.get("low_normal"))
            hi_n = _float_or_none(extras.get("hi_normal"))
            lo_c = _float_or_none(extras.get("low_critical"))
            hi_c = _float_or_none(extras.get("hi_critical"))
            if any(x is not None for x in (lo_n, hi_n, lo_c, hi_c)):
                ciel_numeric_extras[cid] = {
                    "analyte_name": pref,
                    "unit": _clean_text(extras.get("units")),
                    "low_normal": lo_n,
                    "hi_normal": hi_n,
                    "low_critical": lo_c,
                    "hi_critical": hi_c,
                }

    mapping_rows: Set[Tuple[str, str, int, str]] = set()
    ciel_to_rxnorm: Dict[int, List[str]] = {}
    ciel_to_atc: Dict[int, List[str]] = {}
    rxnorm_to_ciel: Dict[str, int] = {}
    ciel_to_loinc: Dict[int, List[str]] = {}

    for m in export_obj.get("mappings") or ():
        if m.get("retired"):
            continue
        mtype_raw = _clean_text(m.get("map_type")).upper()
        if mtype_raw not in {"SAME-AS", "NARROWER-THAN"}:
            continue
        from_c = str(m.get("from_concept_code") or "").strip()
        if not from_c.isdigit():
            continue
        cid = int(from_c)
        if cid not in kept_concepts:
            continue
        src_raw = _clean_text(m.get("to_source_name") or "").lower()
        sys_tag = SOURCE_TAG_MAP.get(src_raw, "")
        if not sys_tag:
            continue
        to_c = _clean_text(m.get("to_concept_code"))
        if not to_c:
            continue
        if sys_tag == "ciel" and to_c == from_c:
            continue
        mshort = "S" if mtype_raw == "SAME-AS" else "N"
        mapping_rows.add((sys_tag, to_c, cid, mshort))
        if sys_tag == "rxnorm":
            ciel_to_rxnorm.setdefault(cid, []).append(to_c)
            if mshort == "S" or to_c not in rxnorm_to_ciel:
                rxnorm_to_ciel[to_c] = cid
        elif sys_tag == "atc":
            ciel_to_atc.setdefault(cid, []).append(to_c)
        elif sys_tag == "loinc":
            ciel_to_loinc.setdefault(cid, []).append(to_c)

    # Build rxnorm_drugs lookup rows (curated public standards first, then CIEL drug backfill)
    drug_rows: Dict[Tuple[str, str], Tuple[str, str, str, str, Optional[int], str, str, str, str, str]] = {}
    for d in rxnorm_pub.get("drugs") or ():
        ing = _clean_text(d["ingredient"]).lower()
        rxcui_in = _clean_text(d.get("rxcui_in"))
        brands = [_clean_text(b) for b in (d.get("brands") or []) if _clean_text(b)]
        atc_code = _clean_text(d.get("atc_code"))
        atc_cls = _clean_text(d.get("atc_class_code")) or (atc_code[:5] if len(atc_code) >= 5 else "")
        atc_cls_name = _clean_text(d.get("atc_class_name"))
        pharm_group = _clean_text(d.get("pharm_group"))
        cid_match = rxnorm_to_ciel.get(rxcui_in) or ciel_name_to_id.get(ing)
        if cid_match is None:
            for b in brands:
                if b.lower() in ciel_name_to_id:
                    cid_match = ciel_name_to_id[b.lower()]
                    break
        brand_str = "|".join(brands)
        all_rxcuis = list(dict.fromkeys([rxcui_in, *[_clean_text(rc) for rc in (d.get("rxcuis") or []) if _clean_text(rc)]]))
        for rc in all_rxcuis:
            if rc:
                drug_rows[("rxcui", rc)] = (
                    "rxcui", rc, ing, rxcui_in, cid_match, atc_code, atc_cls, atc_cls_name, pharm_group, brand_str,
                )
        for nm in [ing, *[b.lower() for b in brands]]:
            if nm:
                drug_rows[("name", nm)] = (
                    "name", nm, ing, rxcui_in, cid_match, atc_code, atc_cls, atc_cls_name, pharm_group, brand_str,
                )
        if cid_match is not None:
            drug_rows[("ciel", str(cid_match))] = (
                "ciel", str(cid_match), ing, rxcui_in, cid_match, atc_code, atc_cls, atc_cls_name, pharm_group, brand_str,
            )

    # Backfill CIEL Drug concepts with RxNORM / WHOATC mappings
    for cid, (_, cclass, pref, syns_pipe, _) in kept_concepts.items():
        if cclass != "Drug":
            continue
        rxs = ciel_to_rxnorm.get(cid, [])
        atcs = ciel_to_atc.get(cid, [])
        if not rxs and not atcs:
            continue
        ing = pref.lower()
        rxcui_in = rxs[0] if rxs else ""
        atc_code = atcs[0] if atcs else ""
        atc_cls = atc_code[:5] if len(atc_code) >= 5 else ""
        syns_list = [s for s in syns_pipe.split("|") if s]
        bstr = "|".join(syns_list[:3])
        if ("ciel", str(cid)) not in drug_rows:
            drug_rows[("ciel", str(cid))] = (
                "ciel", str(cid), ing, rxcui_in, cid, atc_code, atc_cls, "", "", bstr,
            )
        for rc in rxs:
            drug_rows.setdefault(
                ("rxcui", rc),
                ("rxcui", rc, ing, rxcui_in, cid, atc_code, atc_cls, "", "", bstr),
            )
        if len(ing) <= 40:
            drug_rows.setdefault(
                ("name", ing),
                ("name", ing, ing, rxcui_in, cid, atc_code, atc_cls, "", "", bstr),
            )

    # CVX / ACIP vaccines
    cvx_rows: List[Tuple[str, str, int, str, str, str]] = []
    for v in cvx_pub.get("vaccines") or ():
        vgrp = _clean_text(v["vaccine_group"])
        sdoses = int(v.get("series_doses") or 1)
        isum = _clean_text(v.get("interval_summary"))
        rec_g = _clean_text(v.get("recency_guidance"))
        pats = "|".join(_clean_text(p).lower() for p in (v.get("display_patterns") or []) if _clean_text(p))
        for code in v.get("cvx_codes") or ():
            cs = _clean_text(code)
            if cs:
                cvx_rows.append((cs, vgrp, sdoses, isum, rec_g, pats))

    # Lab reference ranges (curated public LOINC + CIEL Numeric Test backfill)
    lab_rows: Dict[Tuple[str, str], Tuple[str, str, str, str, Optional[float], Optional[float], Optional[float], Optional[float], str, str, str]] = {}
    for r in loinc_pub.get("ranges") or ():
        aname = _clean_text(r["analyte_name"])
        unit = _clean_text(r.get("unit"))
        lo_n = _float_or_none(r.get("low_normal"))
        hi_n = _float_or_none(r.get("hi_normal"))
        lo_c = _float_or_none(r.get("low_critical"))
        hi_c = _float_or_none(r.get("hi_critical"))
        rtxt = _clean_text(r.get("reference_text"))
        cnote = _clean_text(r.get("clinical_note"))
        astr = "|".join(_clean_text(a).lower() for a in (r.get("aliases") or []) if _clean_text(a))
        for lc in r.get("loinc_codes") or ():
            lcs = _clean_text(lc)
            if lcs:
                lab_rows[("loinc", lcs)] = ("loinc", lcs, aname, unit, lo_n, hi_n, lo_c, hi_c, rtxt, cnote, astr)

    for cid, ex in ciel_numeric_extras.items():
        lo_n, hi_n = ex["low_normal"], ex["hi_normal"]
        lo_c, hi_c = ex["low_critical"], ex["hi_critical"]
        unit = ex["unit"]
        if lo_n is not None and hi_n is not None:
            rtxt = f"{lo_n:g}-{hi_n:g}" + (f" {unit}" if unit else "")
        elif hi_n is not None:
            rtxt = f"<={hi_n:g}" + (f" {unit}" if unit else "")
        elif lo_n is not None:
            rtxt = f">={lo_n:g}" + (f" {unit}" if unit else "")
        else:
            rtxt = ""
        if not rtxt:
            continue
        lab_rows.setdefault(
            ("ciel", str(cid)),
            ("ciel", str(cid), ex["analyte_name"], unit, lo_n, hi_n, lo_c, hi_c, rtxt, "", ""),
        )
        for lc in ciel_to_loinc.get(cid, ()):
            lab_rows.setdefault(
                ("loinc", lc),
                ("loinc", lc, ex["analyte_name"], unit, lo_n, hi_n, lo_c, hi_c, rtxt, "", ""),
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        conn = sqlite3.connect(str(tmp_path))
        conn.executescript(
            """
            PRAGMA page_size = 4096;
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE concepts (
                ciel_id INTEGER PRIMARY KEY,
                concept_class TEXT NOT NULL,
                preferred_name TEXT NOT NULL,
                synonyms TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE code_mappings (
                system TEXT NOT NULL,
                code TEXT NOT NULL,
                ciel_id INTEGER NOT NULL,
                map_type TEXT NOT NULL
            );
            CREATE TABLE rxnorm_drugs (
                key_type TEXT NOT NULL,
                key_val TEXT NOT NULL,
                ingredient TEXT NOT NULL,
                rxcui_in TEXT NOT NULL,
                ciel_id INTEGER,
                atc_code TEXT NOT NULL,
                atc_class_code TEXT NOT NULL,
                atc_class_name TEXT NOT NULL,
                pharm_group TEXT NOT NULL,
                brands TEXT NOT NULL,
                PRIMARY KEY (key_type, key_val)
            );
            CREATE TABLE cvx_vaccines (
                cvx_code TEXT PRIMARY KEY,
                vaccine_group TEXT NOT NULL,
                series_doses INTEGER NOT NULL,
                interval_summary TEXT NOT NULL,
                recency_guidance TEXT NOT NULL,
                display_patterns TEXT NOT NULL
            );
            CREATE TABLE lab_ranges (
                system TEXT NOT NULL,
                code TEXT NOT NULL,
                analyte_name TEXT NOT NULL,
                unit TEXT NOT NULL,
                low_normal REAL,
                hi_normal REAL,
                low_critical REAL,
                hi_critical REAL,
                reference_text TEXT NOT NULL,
                clinical_note TEXT NOT NULL,
                aliases TEXT NOT NULL,
                PRIMARY KEY (system, code)
            );
            CREATE TABLE term_lookup (
                term_norm TEXT NOT NULL,
                ciel_id INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO concepts VALUES (?, ?, ?, ?, ?)",
            sorted(kept_concepts.values(), key=lambda r: r[0]),
        )
        conn.executemany(
            "INSERT INTO code_mappings VALUES (?, ?, ?, ?)",
            sorted(mapping_rows),
        )
        conn.executemany(
            "INSERT INTO rxnorm_drugs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sorted(drug_rows.values(), key=lambda r: (r[0], r[1])),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO cvx_vaccines VALUES (?, ?, ?, ?, ?, ?)",
            sorted(cvx_rows, key=lambda r: r[0]),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO lab_ranges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sorted(lab_rows.values(), key=lambda r: (r[0], r[1])),
        )
        conn.executemany(
            "INSERT INTO term_lookup VALUES (?, ?)",
            sorted(term_rows),
        )
        conn.executescript(
            """
            CREATE INDEX idx_code_mappings_lookup ON code_mappings(system, code);
            CREATE INDEX idx_code_mappings_ciel ON code_mappings(ciel_id);
            CREATE INDEX idx_term_lookup_norm ON term_lookup(term_norm);
            ANALYZE;
            VACUUM;
            """
        )
        conn.commit()
        conn.close()

        raw_sqlite = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    gz_bytes = gzip.compress(raw_sqlite, compresslevel=9, mtime=0)
    if len(gz_bytes) > 6 * 1024 * 1024:
        raise RuntimeError(f"Compressed terminology_kb.sqlite.gz ({len(gz_bytes)} bytes) exceeds 6 MB budget!")

    gz_path = out_dir / "terminology_kb.sqlite.gz"
    gz_path.write_bytes(gz_bytes)

    provenance = {
        "artifact": "terminology_kb.sqlite.gz",
        "sqlite_uncompressed_bytes": len(raw_sqlite),
        "sqlite_gzip_bytes": len(gz_bytes),
        "sqlite_gzip_sha256": _sha256_bytes(gz_bytes),
        "sources": {
            "ciel": {
                "version": str(export_obj.get("version") or "v2026-04-28"),
                "zip_sha256": zip_sha,
                "kept_concepts": len(kept_concepts),
                "kept_mappings": len(mapping_rows),
            },
            "public_standards": {
                "rxnorm_atc_version": rxnorm_pub.get("version"),
                "cdc_cvx_acip_version": cvx_pub.get("version"),
                "loinc_ranges_version": loinc_pub.get("version"),
            },
        },
        "table_counts": {
            "concepts": len(kept_concepts),
            "code_mappings": len(mapping_rows),
            "rxnorm_drugs": len(drug_rows),
            "cvx_vaccines": len(cvx_rows),
            "lab_ranges": len(lab_rows),
            "term_lookup": len(term_rows),
        },
    }
    prov_out = out_dir / "terminology_kb.provenance.json"
    prov_out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline WAHS terminology_kb.sqlite.gz from CIEL + public standards.")
    parser.add_argument("--ciel-zip", type=Path, default=DEFAULT_CIEL_ZIP)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    prov = build_kb(ciel_zip=args.ciel_zip, public_dir=args.public_dir, out_dir=args.out_dir)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
