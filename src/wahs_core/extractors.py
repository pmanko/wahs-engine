"""Pure-stdlib FHIR R4 Bundle parser and domain-neutral clinical extractors."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

NON_CLINICAL_RESOURCE_TYPES: Set[str] = {
    "Claim", "ExplanationOfBenefit", "SupplyDelivery", "Coverage",
    "Account", "ChargeItem", "Invoice", "PaymentNotice", "PaymentReconciliation",
}
LOOKUP_ONLY_RESOURCE_TYPES: Set[str] = {
    "Provenance", "Organization", "Practitioner", "PractitionerRole", "Location", "Medication",
}
_INVALIDATED_STATUSES: Set[str] = {"entered-in-error", "cancelled"}
_INVALIDATED_VERIFICATION: Set[str] = {"entered-in-error", "refuted"}
_PREFERRED_CODING_SYSTEMS: Tuple[str, ...] = (
    "nlm.nih.gov/research/umls/rxnorm",
    "snomed.info/sct",
    "loinc.org",
    "hl7.org/fhir/sid/cvx",
)


@dataclass
class NumericPoint:
    resource_uuid: str
    date: str
    code: str
    concept: str
    value: float
    unit: str
    category: str
    interpretation: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: bool = False
    timestamp: str = ""
    status: str = ""


@dataclass(frozen=True)
class ClinicalNote:
    uuid: str
    date: Optional[str]
    title: str
    text: str
    source: Optional[str]
    char_count: int
    member_uuids: Tuple[str, ...] = ()


@dataclass
class ClinicalItem:
    resource_type: str
    resource_uuid: str
    date: Optional[str]
    clinical_date: Optional[str]
    date_kind: str  # "clinical_event" | "administrative" | "planned" | "unknown"
    section: str
    code: str
    display: str
    text: str
    mandatory: bool = False
    status: str = ""
    end_date: Optional[str] = None
    source_org: Optional[str] = None
    member_uuids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedChart:
    patient_id: Optional[str]
    patient_name: str
    birth_date: Optional[str]
    reference_date: Optional[str]
    items: List[ClinicalItem]
    notes: List[ClinicalNote]
    numeric_points: List[NumericPoint]
    excluded_resource_counts: Dict[str, int]
    deduplicated_note_count: int
    total_bundle_entries: int


def _strip_urn(ref: Optional[str]) -> str:
    s = str(ref or "").strip()
    if s.startswith("urn:uuid:"):
        return s[len("urn:uuid:") :]
    return s.rsplit("/", 1)[-1] if "/" in s else s


def _iso_date(val: Optional[str]) -> Optional[str]:
    s = str(val or "").strip()
    return s[:10] if len(s) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", s) else None


def _iso_timestamp(val: Optional[str]) -> Optional[str]:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})", str(val or "").strip())
    return f"{m.group(1)}T{m.group(2)}" if m else None


def _sort_timestamp_key(val: Optional[str]) -> str:
    s = str(val or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", s):
        return ""
    iso_norm = re.sub(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})($|[Zz]|[+-]\d{2}(?::?\d{2})?)$", r"\1:00\2", s)
    if iso_norm.endswith(("Z", "z")):
        iso_norm = f"{iso_norm[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", iso_norm))
    except ValueError:
        return ""
    if dt.tzinfo is not None:
        dt_utc = dt.astimezone(timezone.utc)
        return f"Z:{dt_utc.strftime('%Y-%m-%dT%H:%M:%S')}.{dt_utc.microsecond:06d}"
    return f"N:{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{dt.microsecond:06d}"


def _coding_pair(codeable: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not isinstance(codeable, Mapping):
        return "", ""
    codings = [c for c in (codeable.get("coding") or []) if isinstance(c, Mapping)]
    code, display = "", str(codeable.get("text") or "").strip()
    for pref in _PREFERRED_CODING_SYSTEMS:
        for c in codings:
            if pref in str(c.get("system") or "").lower() and c.get("code") is not None:
                code = str(c["code"]).strip()
                display = display or str(c.get("display") or "").strip()
                break
        if code:
            break
    for c in codings:
        if not code and c.get("code") is not None:
            code = str(c["code"]).strip()
        if not display and c.get("display"):
            display = str(c["display"]).strip()
    return code, display


def _codeable_status(val: Any) -> str:
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, Mapping):
        c, d = _coding_pair(val)
        return (c or d).lower()
    return ""


def _categories(res: Mapping[str, Any]) -> List[str]:
    raw = res.get("category")
    cats = [raw] if isinstance(raw, Mapping) else (raw or [])
    out: List[str] = []
    for cat in cats:
        if isinstance(cat, Mapping):
            c, d = _coding_pair(cat)
            v = (c or d).strip().lower()
            if v:
                out.append(v)
    return out


def _format_num(val: Any) -> str:
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        s = f"{val:.6f}".rstrip("0").rstrip(".")
        return s if s and s != "-0" else "0"
    return str(val)


def _format_obs_value(obs: Mapping[str, Any]) -> Tuple[str, Optional[float], str]:
    components = obs.get("component")
    if isinstance(components, list) and components:
        parts: List[str] = []
        for comp in components:
            if not isinstance(comp, Mapping):
                continue
            cc, cd = _coding_pair(comp.get("code"))
            vq = comp.get("valueQuantity")
            if isinstance(vq, Mapping) and vq.get("value") is not None:
                u = str(vq.get("unit") or vq.get("code") or "").strip()
                op = str(vq.get("comparator") or "").strip()
                parts.append(f"{cd or cc}: {op}{_format_num(vq['value'])} {u}".strip())
            else:
                _, vd = _coding_pair(comp.get("valueCodeableConcept"))
                if vd:
                    parts.append(f"{cd or cc}: {vd}")
        if parts:
            return "; ".join(parts), None, ""

    vq = obs.get("valueQuantity")
    if isinstance(vq, Mapping) and vq.get("value") is not None:
        raw_v, unit, op = vq["value"], str(vq.get("unit") or vq.get("code") or "").strip(), str(vq.get("comparator") or "").strip()
        try:
            nv: Optional[float] = float(raw_v)
        except (TypeError, ValueError):
            nv = None
        vs = f"{op}{_format_num(nv if nv is not None else raw_v)}"
        return (f"{vs} {unit}".strip() if unit else vs), (None if op else nv), unit
    if "valueCodeableConcept" in obs:
        return _coding_pair(obs.get("valueCodeableConcept"))[1], None, ""
    if "valueString" in obs:
        return str(obs.get("valueString") or "").strip(), None, ""
    if "valueBoolean" in obs:
        return str(obs.get("valueBoolean")), None, ""
    if "valueInteger" in obs:
        v = obs["valueInteger"]
        return str(v), float(v) if isinstance(v, int) else None, ""
    return "", None, ""


def _format_structured_dosage(d_inst: Mapping[str, Any]) -> str:
    d_txt = str(d_inst.get("text") or "").strip()
    parts: List[str] = []
    for dr in d_inst.get("doseAndRate") or []:
        if not isinstance(dr, Mapping):
            continue
        dq = dr.get("doseQuantity")
        if isinstance(dq, Mapping) and dq.get("value") is not None:
            parts.append(f"{_format_num(dq['value'])} {str(dq.get('unit') or dq.get('code') or '').strip()}".strip())
        drg = dr.get("doseRange")
        if isinstance(drg, Mapping):
            lo, hi = drg.get("low") or {}, drg.get("high") or {}
            if lo.get("value") is not None and hi.get("value") is not None:
                u = str(hi.get("unit") or lo.get("unit") or "").strip()
                parts.append(f"{_format_num(lo['value'])}-{_format_num(hi['value'])} {u}".strip())
    for key, pfx in (("dose", ""), ("rateQuantity", "rate ")):
        q = d_inst.get(key)
        if isinstance(q, Mapping) and q.get("value") is not None:
            parts.append(f"{pfx}{_format_num(q['value'])} {str(q.get('unit') or q.get('code') or '').strip()}".strip())
    _, route_d = _coding_pair(d_inst.get("route"))
    if route_d:
        parts.append(route_d)
    timing = d_inst.get("timing")
    if isinstance(timing, Mapping):
        rep = timing.get("repeat")
        if isinstance(rep, Mapping):
            f, p, pu = rep.get("frequency"), rep.get("period"), str(rep.get("periodUnit") or "").lower()
            un = {"d": "daily", "wk": "weekly", "mo": "monthly", "h": "h"}
            if f == 1 and p == 1 and pu == "d":
                parts.append("once daily")
            elif f and p and pu:
                parts.append(f"{f}x {un[pu]}" if (p == 1 and pu in un) else f"{f}x per {p}{pu}")
        _, td = _coding_pair(timing.get("code"))
        if td:
            parts.append(td)
    if d_inst.get("asNeededBoolean") is True:
        parts.append("PRN")
    for add in d_inst.get("additionalInstruction") or []:
        _, ad = _coding_pair(add)
        if ad:
            parts.append(ad)
    s = " ".join(parts).strip()
    return f"{d_txt} ({s})" if (d_txt and s and s.lower() not in d_txt.lower()) else (d_txt or s)


def _extract_obs_interp_range(obs: Mapping[str, Any], num_val: Optional[float]) -> Tuple[Optional[str], Optional[str], bool]:
    raw_i = obs.get("interpretation")
    ic, idisp = "", ""
    for e in (raw_i if isinstance(raw_i, list) else ([raw_i] if isinstance(raw_i, Mapping) else [])):
        if isinstance(e, Mapping):
            c, d = _coding_pair(e)
            if c or d:
                ic, idisp = c, d
                break
    rstr, lo, hi = None, None, None
    rr = obs.get("referenceRange")
    if isinstance(rr, list) and rr and isinstance(rr[0], Mapping):
        r0 = rr[0]
        lobj, hobj = (r0.get("low") or {}), (r0.get("high") or {})
        try:
            lo = float(lobj["value"]) if isinstance(lobj, Mapping) and lobj.get("value") is not None else None
        except (TypeError, ValueError):
            pass
        try:
            hi = float(hobj["value"]) if isinstance(hobj, Mapping) and hobj.get("value") is not None else None
        except (TypeError, ValueError):
            pass
        u = str((hobj.get("unit") if isinstance(hobj, Mapping) else "") or (lobj.get("unit") if isinstance(lobj, Mapping) else "") or "").strip()
        if r0.get("text"):
            rstr = str(r0["text"]).strip()
        elif lo is not None and hi is not None:
            rstr = f"{_format_num(lo)}-{_format_num(hi)}" + (f" {u}" if u else "")
        elif hi is not None:
            rstr = f"<= {_format_num(hi)}" + (f" {u}" if u else "")
        elif lo is not None:
            rstr = f">= {_format_num(lo)}" + (f" {u}" if u else "")

    nc = ic.upper()
    if nc in {"H", "HH", "HU", "L", "LL", "LU", "A", "AA", "CRIT", "POS", "ABNORMAL", "HIGH", "LOW"}:
        abn, lbl = True, ic or idisp
    elif nc in {"N", "NORMAL"}:
        abn, lbl = False, ic
    elif idisp:
        abn, lbl = idisp.lower() in {"high", "low", "critical", "abnormal", "positive"}, idisp
    else:
        abn, lbl = False, None

    if lbl is None and num_val is not None:
        if lo is not None and num_val < lo:
            lbl, abn = "L", True
        elif hi is not None and num_val > hi:
            lbl, abn = "H", True
    elif lbl is None and isinstance(obs.get("valueQuantity"), Mapping):
        vq = obs["valueQuantity"]
        op = str(vq.get("comparator") or "").strip()
        try:
            cv: Optional[float] = float(vq["value"]) if vq.get("value") is not None else None
        except (TypeError, ValueError):
            cv = None
        if cv is not None:
            if lo is not None and ((op == "<" and cv <= lo) or (op == "<=" and cv < lo)):
                lbl, abn = "L", True
            elif hi is not None and ((op == ">" and cv >= hi) or (op == ">=" and cv > hi)):
                lbl, abn = "H", True
    return lbl, rstr, abn


def _normalize_note_text(decoded_text: str) -> str:
    lines = [ln.strip() for ln in (decoded_text or "").splitlines()]
    if not lines:
        return ""
    sections: Dict[str, List[str]] = {}
    cur = "preamble"
    for ln in lines:
        if ln.startswith("#"):
            cur = ln.lstrip("#").strip().lower()
            sections.setdefault(cur, [])
        elif ln and not re.match(r"^[-=_*]{3,}$", ln):
            sections.setdefault(cur, []).append(ln)
    if any(k != "preamble" for k in sections):
        parts = ([" ".join(sections["preamble"]).strip()] if sections.get("preamble") else []) + [
            f"{hdr.title()}: {' '.join(blines).strip()}"
            for hdr, blines in sections.items()
            if hdr != "preamble" and " ".join(blines).strip()
        ]
        if parts:
            return " | ".join(p for p in parts if p)
    return " ".join(ln for ln in lines if ln and not re.match(r"^[-=_*]{3,}$", ln)).strip()


def _compute_reference_date(resources: Sequence[Mapping[str, Any]]) -> Optional[str]:
    skip = {"planned", "booked", "scheduled", "proposed", "pending", "waitlist", "cancelled", "entered-in-error", "noshow"}
    ctypes = {
        "AllergyIntolerance", "Condition", "MedicationRequest", "MedicationAdministration",
        "MedicationStatement", "MedicationDispense", "Immunization", "DocumentReference",
        "DiagnosticReport", "Observation", "Procedure", "CarePlan", "Goal", "ImagingStudy",
        "Encounter", "Appointment",
    }
    dates: List[str] = []
    for res in resources:
        rt = str(res.get("resourceType") or "")
        if rt not in ctypes:
            continue
        st = _codeable_status(res.get("status")) or _codeable_status(res.get("clinicalStatus"))
        if st in skip or _codeable_status(res.get("verificationStatus")) in _INVALIDATED_VERIFICATION:
            continue
        for fn in ("effectiveDateTime", "occurrenceDateTime", "onsetDateTime", "authoredOn", "performedDateTime", "recordedDate", "issued", "date"):
            d = _iso_date(res.get(fn))
            if d:
                dates.append(d)
                break
        else:
            for pf in ("period", "effectivePeriod", "performedPeriod"):
                p = res.get(pf)
                if isinstance(p, Mapping):
                    d = _iso_date(p.get("start") if (rt == "Encounter" and st not in {"finished", "completed", "fulfilled", ""}) else (p.get("end") or p.get("start")))
                    if d:
                        dates.append(d)
                        break
            else:
                if rt == "Appointment" and st == "fulfilled":
                    d = _iso_date(res.get("end") or res.get("start"))
                    if d:
                        dates.append(d)
    return max(dates) if dates else None


def extract_bundle(bundle: Mapping[str, Any], *, preserve_intraday_timestamps: bool = False) -> ExtractedChart:
    """Parse a FHIR R4 Bundle into domain-neutral `ClinicalItem`, `ClinicalNote`, and `NumericPoint` objects."""
    raw_entries = bundle.get("entry") or []
    resources: List[Mapping[str, Any]] = []
    for idx, entry in enumerate(raw_entries, start=1):
        res = entry.get("resource") if isinstance(entry, Mapping) else None
        if isinstance(res, Mapping) and res.get("resourceType"):
            resources.append(res if str(res.get("id") or "").strip() else {**res, "id": f"anon-{str(res['resourceType']).lower()}-{idx}"})
    if not resources and bundle.get("resourceType") and bundle.get("resourceType") != "Bundle":
        resources = [bundle if str(bundle.get("id") or "").strip() else {**bundle, "id": f"anon-{str(bundle['resourceType']).lower()}-1"}]

    excluded: Dict[str, int] = {}
    orgs, pracs, meds, encs, obs_by_id, conds, prov_orgs = {}, {}, {}, {}, {}, {}, {}
    patient_res: Optional[Mapping[str, Any]] = None

    for res in resources:
        rt, rid = str(res.get("resourceType") or ""), str(res.get("id") or "")
        if rt in NON_CLINICAL_RESOURCE_TYPES:
            excluded[rt] = excluded.get(rt, 0) + 1
        elif rt == "Patient" and patient_res is None:
            patient_res = res
        elif rt == "Organization" and rid:
            orgs[rid] = str(res.get("name") or rid).strip()
            excluded[rt] = excluded.get(rt, 0) + 1
        elif rt == "Practitioner" and rid:
            nm = (res.get("name") or [{}])[0]
            if isinstance(nm, Mapping):
                pracs[rid] = f"{' '.join(nm.get('given') or [])} {nm.get('family') or ''}".strip()
            excluded[rt] = excluded.get(rt, 0) + 1
        elif rt == "Medication" and rid:
            meds[rid] = _coding_pair(res.get("code"))
            excluded[rt] = excluded.get(rt, 0) + 1
        elif rt == "Encounter" and rid:
            encs[rid] = res
        elif rt == "Observation" and rid:
            obs_by_id[rid] = res
        elif rt == "Condition" and rid:
            _, cd = _coding_pair(res.get("code"))
            if cd:
                conds[rid] = cd
        elif rt == "Provenance":
            excluded[rt] = excluded.get(rt, 0) + 1
            aname = ""
            for ag in res.get("agent") or []:
                if isinstance(ag, Mapping):
                    who, ob = ag.get("who") or {}, ag.get("onBehalfOf") or {}
                    oid = _strip_urn(ob.get("reference") or who.get("reference"))
                    aname = str(ob.get("display") or who.get("display") or orgs.get(oid) or "").strip()
                    if aname:
                        break
            for tgt in (res.get("target") or []) if aname else ():
                if isinstance(tgt, Mapping):
                    tid = _strip_urn(tgt.get("reference"))
                    if tid and tid not in prov_orgs:
                        prov_orgs[tid] = aname
        elif rt in LOOKUP_ONLY_RESOURCE_TYPES:
            excluded[rt] = excluded.get(rt, 0) + 1

    def src_of(res: Mapping[str, Any]) -> Optional[str]:
        rid = str(res.get("id") or "")
        for rf in ("serviceProvider", "recorder", "asserter", "requester"):
            ro = res.get(rf)
            if isinstance(ro, Mapping):
                rref = _strip_urn(ro.get("reference"))
                d = str(ro.get("display") or orgs.get(rref) or pracs.get(rref) or "").strip()
                if d:
                    return d
        for pf in res.get("performer") or []:
            if isinstance(pf, Mapping):
                act = pf.get("actor") or pf
                pref = _strip_urn(act.get("reference"))
                d = str(act.get("display") or orgs.get(pref) or pracs.get(pref) or "").strip()
                if d:
                    return d
        enc_ref = _strip_urn((res.get("encounter") or {}).get("reference"))
        if enc_ref in encs and isinstance(encs[enc_ref].get("serviceProvider"), Mapping):
            sp = encs[enc_ref]["serviceProvider"]
            d = str(sp.get("display") or orgs.get(_strip_urn(sp.get("reference"))) or "").strip()
            if d:
                return d
        if rid in prov_orgs:
            return prov_orgs[rid]
        for tag in ((res.get("meta") or {}).get("tag") or []) if isinstance(res.get("meta"), Mapping) else ():
            if isinstance(tag, Mapping) and str(tag.get("system") or "").endswith("/wahealthsummary-source-name"):
                tc = str(tag.get("code") or tag.get("display") or "").strip()
                if tc:
                    return tc
        return None

    def reasons_of(res: Mapping[str, Any]) -> List[str]:
        out = [_coding_pair(rc)[1] for rc in (res.get("reasonCode") or []) if _coding_pair(rc)[1]]
        for rr in res.get("reasonReference") or []:
            if isinstance(rr, Mapping):
                d = str(rr.get("display") or conds.get(_strip_urn(rr.get("reference"))) or "").strip()
                if d:
                    out.append(d)
        return out

    items: List[ClinicalItem] = []
    notes_list: List[ClinicalNote] = []
    numeric_points: List[NumericPoint] = []
    ref_date = _compute_reference_date(resources)
    pid, pname, dob = None, "Unknown Patient", None

    if patient_res is not None:
        pid = str(patient_res.get("id") or "patient-1")
        dob = _iso_date(patient_res.get("birthDate"))
        names = patient_res.get("name") or []
        if names and isinstance(names[0], Mapping):
            ch = next((n for n in names if n.get("use") == "official"), names[0])
            pname = f"{' '.join(ch.get('given') or [])} {ch.get('family') or ''}".strip() or pname
        age_str = ""
        if dob and ref_date:
            try:
                age_str = f", Age: {int((date.fromisoformat(ref_date) - date.fromisoformat(dob)).days // 365.2425)}y (as of {ref_date})"
            except ValueError:
                pass
        langs = [_coding_pair(cm.get("language"))[1] for cm in (patient_res.get("communication") or []) if isinstance(cm, Mapping) and _coding_pair(cm.get("language"))[1]]
        items.append(ClinicalItem("Patient", pid, ref_date or dob, None, "administrative", "patient", "patient-demographics", pname, f"Patient: {pname}, DOB: {dob or 'unknown'}, Gender: {patient_res.get('gender') or 'unknown'}{age_str}{(', Language: ' + ', '.join(langs)) if langs else ''}", True, "active"))

    notes_by_hash: Dict[str, Dict[str, Any]] = {}
    dedup_note_count = 0
    claimed_obs: Set[str] = set()

    def _add_note(ptype: str, rid: str, raw_ts: str, title: str, b64: str, org: Optional[str]) -> None:
        nonlocal dedup_note_count
        iso_d = _iso_date(raw_ts)
        dig = hashlib.sha256(b64.strip().encode("ascii", errors="ignore")).hexdigest()
        if dig in notes_by_hash:
            notes_by_hash[dig]["members"].append(rid)
            if iso_d:
                notes_by_hash[dig]["dates"].append(iso_d)
            dedup_note_count += 1
            return
        try:
            decoded = base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            decoded = ""
        notes_by_hash[dig] = {"ptype": ptype, "uuid": rid, "date": iso_d, "dates": [iso_d] if iso_d else [], "title": title or "Clinical Note", "decoded": decoded, "org": org, "members": [rid]}

    def _add_components(obs_res: Mapping[str, Any], ouuid: str, odate: str, ocat: str, ots: str, ost: str) -> None:
        for comp in obs_res.get("component") or []:
            if not isinstance(comp, Mapping):
                continue
            cc, cd = _coding_pair(comp.get("code"))
            vq = comp.get("valueQuantity")
            if isinstance(vq, Mapping) and vq.get("value") is not None and not vq.get("comparator"):
                try:
                    cnum = float(vq["value"])
                except (TypeError, ValueError):
                    continue
                ci, cr, ca = _extract_obs_interp_range(comp, cnum)
                numeric_points.append(NumericPoint(ouuid, odate, cc or cd, cd or cc, cnum, str(vq.get("unit") or vq.get("code") or "").strip(), ocat, ci, cr, ca, ots or _sort_timestamp_key(odate), ost))

    for res in resources:
        rt, rid = str(res.get("resourceType") or ""), str(res.get("id") or "")
        if rt == "AllergyIntolerance":
            cs, vs = _codeable_status(res.get("clinicalStatus")) or "active", _codeable_status(res.get("verificationStatus"))
            if cs == "entered-in-error" or vs == "entered-in-error":
                excluded["AllergyIntolerance(entered-in-error)"] = excluded.get("AllergyIntolerance(entered-in-error)", 0) + 1
                continue
            c, d = _coding_pair(res.get("code"))
            rxns = [f"{md} ({rx.get('severity')})" if rx.get("severity") else md for rx in (res.get("reaction") or []) if isinstance(rx, Mapping) for mn in (rx.get("manifestation") or []) for _, md in [_coding_pair(mn)] if md]
            rd = _iso_date(res.get("recordedDate") or res.get("onsetDateTime"))
            quals = [f"status: {cs}"] + ([f"verification: {vs}"] if vs else []) + ([f"criticality: {res['criticality']}"] if res.get("criticality") else []) + ([f"reaction: {', '.join(rxns)}"] if rxns else [])
            items.append(ClinicalItem(rt, rid, rd, rd if vs != "refuted" else None, "clinical_event" if (rd and vs != "refuted") else "unknown", "allergy", c or d, d or c or "Unspecified allergy", f"Allergy: {d or c} ({', '.join(quals)})", cs not in {"inactive", "resolved"} and vs != "refuted", cs, None, src_of(res), metadata={"verification_status": vs} if vs else {}))
        elif rt == "Condition":
            cs, vs = _codeable_status(res.get("clinicalStatus")) or "active", _codeable_status(res.get("verificationStatus"))
            if cs == "entered-in-error" or vs == "entered-in-error":
                excluded["Condition(entered-in-error)"] = excluded.get("Condition(entered-in-error)", 0) + 1
                continue
            c, d = _coding_pair(res.get("code"))
            on, ab, so = _iso_date(res.get("onsetDateTime") or res.get("recordedDate")), _iso_date(res.get("abatementDateTime")), src_of(res)
            act = cs in {"active", "recurrence", "relapse"} and vs not in _INVALIDATED_VERIFICATION
            ev = on if act else (ab or on)
            det = [f"status: {cs}"] + ([f"verification: {vs}"] if vs else []) + ([f"onset: {on}"] if on else []) + ([f"resolved: {ab}"] if ab else []) + ([f"source: {so}"] if so else []) + ([f"SNOMED: {c}"] if c else [])
            items.append(ClinicalItem(rt, rid, ev, ev if vs != "refuted" else None, "clinical_event" if (ev and vs != "refuted") else "unknown", "condition_active" if act else "condition_resolved", c or d, d or c or "Condition", f"Condition: {d or c} ({', '.join(det)})", act, cs, ab, so, metadata={"onset": on, "abatement": ab, **({"verification_status": vs} if vs else {})}))
        elif rt in {"MedicationRequest", "MedicationAdministration", "MedicationStatement", "MedicationDispense"}:
            st = str(res.get("status") or "unknown").strip().lower()
            if st == "entered-in-error":
                excluded[f"{rt}(entered-in-error)"] = excluded.get(f"{rt}(entered-in-error)", 0) + 1
                continue
            c, d = _coding_pair(res.get("medicationCodeableConcept"))
            if not d and isinstance(res.get("medicationReference"), Mapping):
                mr = _strip_urn(res["medicationReference"].get("reference"))
                c, d = meds.get(mr, (c, str(res["medicationReference"].get("display") or "")))
            mts = str(res.get("authoredOn") or res.get("effectiveDateTime") or (res.get("effectivePeriod") or {}).get("start") or res.get("whenHandedOver") or res.get("whenPrepared") or res.get("dateAsserted") or "").strip()
            md_d, ed_d = _iso_date(mts), _iso_date((res.get("effectivePeriod") or {}).get("end"))
            raw_dos = res.get("dosageInstruction") or (res.get("dosage") if isinstance(res.get("dosage"), list) else ([res.get("dosage")] if isinstance(res.get("dosage"), Mapping) else []))
            doses = [dt for di in raw_dos if isinstance(di, Mapping) for dt in [_format_structured_dosage(di)] if dt]
            rsn, setting, so = reasons_of(res), ("inpatient/administered" if rt == "MedicationAdministration" else "prescribed"), src_of(res)
            act, mdisp = st in {"active", "in-progress", "on-hold"}, d or c or "Medication"
            qls = [f"status: {st}", f"type: {setting}"] + ([f"dosage: {'; '.join(doses[:2])}"] if doses else []) + ([f"reason: {', '.join(dict.fromkeys(rsn))}"] if rsn else []) + ([f"source: {so}"] if so else []) + ([f"RxNorm: {c}"] if c else [])
            items.append(ClinicalItem(rt, rid, md_d, md_d, "clinical_event" if md_d else "unknown", "med_active" if act else "med_historical", c or mdisp.lower(), mdisp, f"Medication: {mdisp} ({', '.join(qls)})", act, st, ed_d, so, metadata={"reasons": rsn, "dosage": doses, "setting": setting}))
        elif rt == "Immunization":
            st = str(res.get("status") or "completed").strip().lower()
            if st == "entered-in-error":
                excluded["Immunization(entered-in-error)"] = excluded.get("Immunization(entered-in-error)", 0) + 1
                continue
            c, d = _coding_pair(res.get("vaccineCode"))
            od, so, comp = _iso_date(res.get("occurrenceDateTime")), src_of(res), (st == "completed")
            qls = [f"status: {st}"] + ([f"CVX: {c}"] if c else []) + ([f"source: {so}"] if so else [])
            items.append(ClinicalItem(rt, rid, od, od if comp else None, "clinical_event" if (od and comp) else "unknown", "immunization", c or (d or "vaccine"), d or c or "Vaccine", f"Immunization: {d or c or 'Vaccine'} ({', '.join(qls)})", comp, st, None, so))
        elif rt == "DocumentReference":
            if str(res.get("status") or "").lower() in _INVALIDATED_STATUSES or str(res.get("docStatus") or "").lower() == "entered-in-error":
                excluded["DocumentReference(invalid)"] = excluded.get("DocumentReference(invalid)", 0) + 1
                continue
            nts = str(res.get("date") or ((res.get("context") or {}).get("period") or {}).get("start") or "").strip()
            _, nt = _coding_pair(res.get("type"))
            for ci in res.get("content") or []:
                b64 = ((ci or {}).get("attachment") or {}).get("data")
                if isinstance(b64, str) and b64:
                    _add_note("DocumentReference", rid, nts, nt, b64, src_of(res))
        elif rt == "DiagnosticReport":
            dr_st = str(res.get("status") or "final").strip().lower()
            rrefs, dr_conc = (res.get("result") or []), str(res.get("conclusion") or "").strip()
            ccs = [_coding_pair(cc)[1] for cc in (res.get("conclusionCode") or []) if isinstance(cc, Mapping) and _coding_pair(cc)[1]]
            if dr_st in _INVALIDATED_STATUSES:
                for ro in rrefs:
                    if isinstance(ro, Mapping) and _strip_urn(ro.get("reference")):
                        claimed_obs.add(_strip_urn(ro.get("reference")))
                excluded[f"DiagnosticReport({dr_st})"] = excluded.get(f"DiagnosticReport({dr_st})", 0) + 1
                continue
            for pf in res.get("presentedForm") or []:
                b64 = (pf or {}).get("data")
                if isinstance(b64, str) and b64:
                    _add_note("DiagnosticReport", rid, str(res.get("effectiveDateTime") or res.get("issued") or ""), _coding_pair(res.get("code"))[1], b64, src_of(res))
            if not rrefs and not dr_conc and not ccs:
                continue
            dr_cats, (code, disp) = _categories(res), _coding_pair(res.get("code"))
            p_raw = res.get("effectiveDateTime") or res.get("issued")
            p_dt = _iso_date(p_raw)
            muuids, cparts, pabn = [rid], [], False
            for ro in rrefs:
                oid = _strip_urn(ro.get("reference")) if isinstance(ro, Mapping) else ""
                if not oid:
                    continue
                claimed_obs.add(oid)
                muuids.append(oid)
                co = obs_by_id.get(oid)
                if not co:
                    if ro.get("display"):
                        cparts.append(str(ro["display"]).strip())
                    continue
                cst = str(co.get("status") or "final").strip().lower()
                if cst in _INVALIDATED_STATUSES:
                    continue
                oc, od = _coding_pair(co.get("code"))
                vs_o, nv_o, un_o = _format_obs_value(co)
                oi, orr, oa = _extract_obs_interp_range(co, nv_o)
                ocats = _categories(co)
                od_raw = co.get("effectiveDateTime") or co.get("issued")
                od_dt = (_iso_timestamp(od_raw) if preserve_intraday_timestamps else _iso_date(od_raw)) or p_dt
                ots = _sort_timestamp_key(od_raw or p_raw)
                ocat = ocats[0] if ocats else ("survey" if "survey" in dr_cats else "laboratory")
                sl = od or oc or "item"
                pabn = pabn or oa
                if co.get("component"):
                    _add_components(co, oid, od_dt or "", ocat, ots, cst)
                elif nv_o is not None:
                    numeric_points.append(NumericPoint(oid, od_dt or "", oc or sl, sl, nv_o, un_o, ocat, oi, orr, oa, ots, cst))
                if vs_o:
                    ann = [x for x in (oi, f"ref: {orr}" if orr else "") if x]
                    cparts.append(f"{sl}: {vs_o}" + (f" ({'; '.join(ann)})" if ann else ""))
            segs = (["; ".join(cparts)] if cparts else []) + ([f"Conclusion: {dr_conc}"] if dr_conc else ([f"Conclusion: {', '.join(dict.fromkeys(ccs))}"] if ccs else []))
            sec = "survey_panel" if "survey" in dr_cats else ("imaging" if "imaging" in dr_cats else "lab_panel")
            pfx = "Survey Panel" if sec == "survey_panel" else ("Imaging Report" if sec == "imaging" else "Diagnostic Panel")
            items.append(ClinicalItem(rt, rid, p_dt, p_dt, "clinical_event" if p_dt else "unknown", sec, code or disp, disp or code or "Diagnostic Report", f"{pfx} ({disp or code}): {' | '.join(segs) if segs else 'completed'}", False, dr_st, None, src_of(res), tuple(dict.fromkeys(muuids)), {"is_abnormal": pabn, **({"conclusion": dr_conc} if dr_conc else {})}))
        elif rt == "Procedure":
            st = str(res.get("status") or "completed").strip().lower()
            if st == "entered-in-error":
                excluded["Procedure(entered-in-error)"] = excluded.get("Procedure(entered-in-error)", 0) + 1
                continue
            c, d = _coding_pair(res.get("code"))
            pd = _iso_date(res.get("performedDateTime") or (res.get("performedPeriod") or {}).get("start") or (res.get("performedPeriod") or {}).get("end"))
            rsn, perf = reasons_of(res), st not in {"not-done", "stopped", "preparation"}
            qls = ([f"status: {st}"] if st != "completed" else []) + ([f"reason: {', '.join(dict.fromkeys(rsn))}"] if rsn else [])
            items.append(ClinicalItem(rt, rid, pd, pd if perf else None, "clinical_event" if (pd and perf) else "unknown", "procedure", c or d, d or c or "Procedure", f"Procedure: {d or c}" + (f" ({', '.join(qls)})" if qls else ""), False, st, None, src_of(res)))
        elif rt == "CarePlan":
            st = str(res.get("status") or "active").strip().lower()
            if st in _INVALIDATED_STATUSES:
                excluded[f"CarePlan({st})"] = excluded.get(f"CarePlan({st})", 0) + 1
                continue
            cd, ed = _iso_date((res.get("period") or {}).get("start") or res.get("created")), _iso_date((res.get("period") or {}).get("end"))
            cats = [_coding_pair(ci)[1] for ci in (res.get("category") or []) if _coding_pair(ci)[1] and _coding_pair(ci)[1].lower() != "assess-plan"]
            acts = [_coding_pair(((a or {}).get("detail") or {}).get("code"))[1] for a in (res.get("activity") or []) if _coding_pair(((a or {}).get("detail") or {}).get("code"))[1]]
            lbl = ", ".join(cats) or str(res.get("title") or "").strip() or "Care Plan"
            items.append(ClinicalItem(rt, rid, cd, cd, "clinical_event" if cd else "unknown", "careplan_active" if st == "active" else "careplan_historical", lbl, lbl, f"CarePlan: {lbl} (status: {st}{(', activities: ' + ', '.join(acts[:4])) if acts else ''})", False, st, ed, src_of(res)))
        elif rt == "Goal":
            st = str(res.get("lifecycleStatus") or "active").strip().lower()
            if st in _INVALIDATED_STATUSES:
                excluded[f"Goal({st})"] = excluded.get(f"Goal({st})", 0) + 1
                continue
            gd = _iso_date(res.get("startDate") or res.get("statusDate"))
            dues = [_iso_date(t.get("dueDate")) for t in (res.get("target") or []) if isinstance(t, Mapping) and _iso_date(t.get("dueDate"))]
            desc = res.get("description")
            gc, gdisp = _coding_pair(desc) if isinstance(desc, Mapping) else ("", str(desc or "").strip())
            glbl = gdisp or "Patient Health Goal"
            ntxt = "; ".join(str(n.get("text") or "").strip() for n in (res.get("note") or []) if isinstance(n, Mapping) and str(n.get("text") or "").strip())
            so = src_of(res)
            items.append(ClinicalItem(rt, rid, gd, gd, "clinical_event" if gd else "unknown", "careplan_active" if st in {"active", "accepted", "planned", "proposed", "on-hold"} else "careplan_historical", gc or glbl, glbl, f"Goal: {glbl} (status: {st}{(', target due: ' + dues[0]) if dues else ''}){(' - ' + ntxt) if ntxt else ''}{(' [source: ' + so + ']') if so else ''}", False, st, dues[0] if dues else None, so))
        elif rt == "ImagingStudy":
            st = str(res.get("status") or "available").strip().lower()
            if st in _INVALIDATED_STATUSES:
                continue
            imd = _iso_date(res.get("started"))
            _, pdisp = _coding_pair((res.get("procedureCode") or [{}])[0] if res.get("procedureCode") else {})
            sites = [_coding_pair(s.get("bodySite"))[1] for s in (res.get("series") or []) if isinstance(s, Mapping) and _coding_pair(s.get("bodySite"))[1]]
            mods = [str((s.get("modality") or {}).get("display") or "") for s in (res.get("series") or []) if isinstance(s, Mapping) and (s.get("modality") or {}).get("display")]
            desc = pdisp or ", ".join(dict.fromkeys(mods)) or "Imaging Study"
            items.append(ClinicalItem(rt, rid, imd, imd, "clinical_event" if imd else "unknown", "imaging", desc, desc, f"ImagingStudy: {desc}{(' (' + ', '.join(dict.fromkeys(sites)) + ')') if sites else ''}", False, st, None, src_of(res)))
        elif rt == "Device":
            st = str(res.get("status") or "active").strip().lower()
            if st == "entered-in-error":
                continue
            c, d = _coding_pair(res.get("type"))
            if not d and isinstance((res.get("deviceName") or [None])[0], Mapping):
                d = str(res["deviceName"][0].get("name") or "")
            dd = _iso_date(res.get("manufactureDate"))
            items.append(ClinicalItem(rt, rid, dd, None, "administrative" if dd else "unknown", "device", c or d, d or c or "Medical Device", f"Device: {d or c} (status: {st})", st == "active", st, None, src_of(res)))
        elif rt == "Encounter":
            st = str(res.get("status") or "finished").strip().lower()
            if st == "entered-in-error":
                excluded["Encounter(entered-in-error)"] = excluded.get("Encounter(entered-in-error)", 0) + 1
                continue
            ep = res.get("period") or {}
            es, ee = _iso_date(ep.get("start")), _iso_date(ep.get("end"))
            term, plan, exc = {"finished", "completed", "fulfilled", ""}, {"planned", "booked", "scheduled", "proposed", "pending", "waitlist"}, {"cancelled", "entered-in-error", "noshow"}
            if st in term:
                ed, ecd, ek = ee or es, ee or es, "clinical_event" if (ee or es) else "unknown"
            elif st in plan or st in exc:
                ed, ecd, ek = es or ee, None, "planned" if ((es or ee) and st in plan) else "unknown"
            else:
                ed, ecd, ek = es or ee, es, "clinical_event" if es else ("planned" if ee else "unknown")
            ec = str((res.get("class") or {}).get("code") or "AMB").upper()
            _, td = _coding_pair((res.get("type") or [{}])[0] if res.get("type") else {})
            rsn, so = reasons_of(res), src_of(res)
            qls = [f"class: {ec}"] + ([f"discharged: {ee}" if st in term else f"est. end: {ee}"] if ee and ee != es else []) + ([f"reason: {', '.join(dict.fromkeys(rsn))}"] if rsn else []) + ([f"facility: {so}"] if so else [])
            items.append(ClinicalItem(rt, rid, ed, ecd, ek, "encounter_acuity" if ec in {"IMP", "INPATIENT", "EMER", "EMERGENCY", "ACUTE"} else "encounter_routine", ec, td or ec, f"Encounter: {td or ec} ({', '.join(qls)})", False, st, ee, so, metadata={"start": es, "end": ee, "class": ec}))
        elif rt == "Appointment":
            st = str(res.get("status") or "booked").strip().lower()
            if st == "entered-in-error":
                continue
            as_d, ae_d = _iso_date(res.get("start")), _iso_date(res.get("end"))
            ad = as_d or ae_d
            sc, sd = _coding_pair((res.get("serviceType") or [{}])[0] if res.get("serviceType") else {})
            tc, td = _coding_pair(res.get("appointmentType") or {})
            disp = str(res.get("description") or "").strip() or sd or td or "Appointment"
            rsn, so = reasons_of(res), src_of(res)
            qls = [f"status: {st}"] + ([f"reason: {', '.join(dict.fromkeys(rsn))}"] if rsn else []) + ([f"facility: {so}"] if so else [])
            dk = "clinical_event" if (st == "fulfilled" and ad) else ("administrative" if (st in {"cancelled", "noshow"} and ad) else ("planned" if ad else "unknown"))
            items.append(ClinicalItem(rt, rid, ad, ad if st == "fulfilled" else None, dk, "encounter_routine", sc or tc, disp, f"Appointment: {disp} ({', '.join(qls)})", False, st, ae_d, so, metadata={"start": as_d, "end": ae_d}))

    for res in resources:
        if res.get("resourceType") != "Observation":
            continue
        rid = str(res.get("id") or "")
        if rid in claimed_obs:
            continue
        ost = str(res.get("status") or "final").strip().lower()
        if ost in _INVALIDATED_STATUSES:
            excluded[f"Observation({ost})"] = excluded.get(f"Observation({ost})", 0) + 1
            continue
        code, disp = _coding_pair(res.get("code"))
        raw_dt = res.get("effectiveDateTime") or res.get("issued")
        odate = _iso_date(raw_dt)
        ots = (_iso_timestamp(raw_dt) if preserve_intraday_timestamps else None) or odate
        sort_ts = _sort_timestamp_key(raw_dt)
        cats = _categories(res)
        vstr, nval, unit = _format_obs_value(res)
        oi, orr, oa = _extract_obs_interp_range(res, nval)
        pcat = cats[0] if cats else "observation"
        if res.get("component"):
            _add_components(res, rid, ots or odate or "", pcat, sort_ts, ost)
        elif nval is not None:
            numeric_points.append(NumericPoint(rid, ots or odate or "", code or disp, disp or code or "Observation", nval, unit, pcat, oi, orr, oa, sort_ts, ost))
        sec_map = {"vital-signs": ("vital_sign", "Vital Sign"), "laboratory": ("lab_observation", "Lab"), "social-history": ("social_history", "Social History"), "survey": ("survey_observation", "Survey"), "imaging": ("imaging", "Imaging Finding")}
        sec, clbl = next((sec_map[c] for c in cats if c in sec_map), ("observation", "Observation"))
        ann = [b for b in (f"interp: {oi}" if oi else "", f"ref: {orr}" if orr else "") if b]
        ts_sfx = f" (timestamp: {ots})" if (preserve_intraday_timestamps and ots and "T" in ots) else ""
        bval = f": {vstr}" + (f" [{'; '.join(ann)}]" if ann else "") if vstr else ""
        top_vq = res.get("valueQuantity")
        tcmp = str(top_vq.get("comparator") or "").strip() if isinstance(top_vq, Mapping) else ""
        ometa = {"value": vstr, "numeric": nval, "unit": unit, "category": pcat, "is_abnormal": oa, **({"comparator": tcmp} if tcmp else {}), **({"interpretation": oi} if oi else {}), **({"reference_range": orr} if orr else {})}
        items.append(ClinicalItem("Observation", rid, odate, odate, "clinical_event" if odate else "unknown", sec, code or disp, disp or code or "Observation", f"{clbl} ({disp or code}){bval}{ts_sfx}", False, ost, None, src_of(res), metadata=ometa))

    for ni in notes_by_hash.values():
        snip = _normalize_note_text(ni["decoded"])
        if not snip:
            continue
        mems = tuple(dict.fromkeys(ni["members"]))
        ddates = sorted({d for d in ni["dates"] if d})
        pdt = ddates[-1] if ddates else ni["date"]
        extra = f" (recorded across dates: {', '.join(ddates)})" if len(ddates) > 1 else ""
        notes_list.append(ClinicalNote(ni["uuid"], pdt, ni["title"], snip, ni["org"], len(snip), mems))
        items.append(ClinicalItem(ni["ptype"], ni["uuid"], pdt, pdt, "clinical_event" if pdt else "unknown", "clinical_note", "clinical-note", ni["title"], f"Clinical Note ({ni['title']}): {snip}{extra}", False, "current", None, ni["org"], mems, {"full_text": snip, "raw_decoded_text": ni["decoded"], "title": ni["title"], "duplicate_wrapper_count": len(mems)}))

    return ExtractedChart(pid, pname, dob, ref_date, items, notes_list, numeric_points, excluded, dedup_note_count, len(resources))


__all__ = [
    "ClinicalItem", "ClinicalNote", "ExtractedChart", "LOOKUP_ONLY_RESOURCE_TYPES",
    "NON_CLINICAL_RESOURCE_TYPES", "NumericPoint", "extract_bundle",
]
