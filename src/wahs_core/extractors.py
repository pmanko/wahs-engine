"""FHIR R4 Bundle parser and clinical resource extractors for progressive chart condensation.

Handles arbitrary US Core / IPS / Synthea FHIR R4 Bundles:
- Resolves internal references (`Patient`, `Organization`, `Practitioner`, `Encounter`, `Medication`)
- Inverts `Provenance.target[]` into per-resource `source_org` metadata
- Deduplicates Base64 clinical notes across `DocumentReference` and `DiagnosticReport.presentedForm`
- Groups `DiagnosticReport.result[]` panels and `Observation.component[]` multi-part measurements
- Unifies `MedicationRequest`, `MedicationAdministration`, `MedicationStatement`, and `MedicationDispense`
- Classifies SDOH privacy sensitivity (`SDOH/PRIVACY-SENSITIVE` vs `SDOH-SOCIAL`)
- Pre-annotates active condition stage conflicts and immunization duplicate/ACIP anomalies
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .kb_lookup import (
    resolve_condition_code,
    resolve_immunization_code,
    resolve_lab_code,
    resolve_medication_code,
)
from .normalizers import (
    annotate_immunization_anomalies,
    classify_sensitivity,
    classify_sensitivity_with_source,
    classify_vaccine_group,
    detect_condition_conflicts,
    detect_medication_conflicts,
    is_clinic_room_equipment,
    normalize_drug_ingredient,
)

# Resources that carry billing/supply/structural lookup data rather than direct clinical events.
NON_CLINICAL_RESOURCE_TYPES: Set[str] = {
    "Claim",
    "ExplanationOfBenefit",
    "SupplyDelivery",
    "Coverage",
    "Account",
    "ChargeItem",
    "Invoice",
    "PaymentNotice",
    "PaymentReconciliation",
}

LOOKUP_ONLY_RESOURCE_TYPES: Set[str] = {
    "Provenance",
    "Organization",
    "Practitioner",
    "PractitionerRole",
    "Location",
    "Medication",
}

_SURVEY_PANEL_PATTERNS = re.compile(
    r"\b(?:"
    r"phq-?[29]|patient\s+health\s+questionnaire|"
    r"gad-?7|generalized\s+anxiety|"
    r"audit-?c|alcohol\s+use\s+disorder|"
    r"dast-?10|drug\s+abuse\s+screening|"
    r"hark|humiliation|intimate\s+partner|"
    r"prapare|fall\s+scale|morse\s+fall|"
    r"web-?based\s+danger\s+assessment"
    r")\b",
    re.IGNORECASE,
)

_ROUTINE_SCREENING_PROCEDURE_PATTERNS = re.compile(
    r"\b(?:"
    r"depression\s+screening|"
    r"assessment\s+of\s+(?:health\s+and\s+social\s+care\s+needs|anxiety|substance\s+use|alcohol)|"
    r"screening\s+for\s+(?:domestic\s+abuse|drug\s+abuse|occult\s+blood)|"
    r"medication\s+reconciliation|"
    r"assessment\s+using\s+(?:morse\s+fall\s+scale|hark|audit|dast|alcohol\s+use\s+disorders\s+identification)|"
    r"sleep\s+apnea\s+assessment"
    r")\b",
    re.IGNORECASE,
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


@dataclass
class ClinicalItem:
    resource_type: str
    resource_uuid: str
    date: Optional[str]
    clinical_date: Optional[str]
    date_kind: str  # "clinical_event" | "administrative" | "unknown"
    section: str    # "patient" | "allergy" | "condition_active" | "condition_resolved" | "med_active" | ...
    code: str
    display: str
    text: str
    mandatory: bool = False
    status: str = ""
    end_date: Optional[str] = None
    source_org: Optional[str] = None
    sensitivity: Optional[str] = None
    conflict_note: Optional[str] = None
    ingredient_key: str = ""
    is_routine: bool = False
    member_uuids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedChart:
    patient_id: Optional[str]
    patient_name: str
    birth_date: Optional[str]
    reference_date: Optional[str]
    items: List[ClinicalItem]
    numeric_points: List[NumericPoint]
    excluded_resource_counts: Dict[str, int]
    deduplicated_note_count: int
    total_bundle_entries: int


def _strip_urn(ref: Optional[str]) -> str:
    if not ref:
        return ""
    s = str(ref).strip()
    if s.startswith("urn:uuid:"):
        return s[len("urn:uuid:") :]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _iso_date(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    if len(s) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    return None


def _iso_timestamp(val: Optional[str]) -> Optional[str]:
    """Return `YYYY-MM-DDTHH:MM` when an intraday time component is present on an ISO datetime string."""
    if not val:
        return None
    s = str(val).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})", s)
    if m:
        return f"{m.group(1)}T{m.group(2)}"
    return None


def _coding_pair(codeable: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not isinstance(codeable, Mapping):
        return "", ""
    codings = codeable.get("coding") or []
    code = ""
    display = str(codeable.get("text") or "").strip()
    for c in codings:
        if isinstance(c, Mapping):
            if not code and c.get("code") is not None:
                code = str(c.get("code")).strip()
            if not display and c.get("display"):
                display = str(c.get("display")).strip()
    return code, display


def _codeable_status(val: Any) -> str:
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, Mapping):
        code, display = _coding_pair(val)
        return (code or display).lower()
    return ""


def _categories(res: Mapping[str, Any]) -> List[str]:
    cats: List[str] = []
    raw_cats = res.get("category")
    if isinstance(raw_cats, Mapping):
        raw_cats = [raw_cats]
    for cat in raw_cats or []:
        if isinstance(cat, Mapping):
            code, disp = _coding_pair(cat)
            if code:
                cats.append(code.lower())
            elif disp:
                cats.append(disp.lower())
    return cats


def _format_num(val: Any) -> str:
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        formatted = f"{val:.6f}".rstrip("0").rstrip(".")
        return formatted if formatted and formatted != "-0" else "0"
    return str(val)


def _format_obs_value(obs: Mapping[str, Any]) -> Tuple[str, Optional[float], str]:
    """Return (formatted_value_with_unit, numeric_float_or_None, unit_str)."""
    components = obs.get("component")
    if isinstance(components, list) and components:
        comp_parts: List[str] = []
        sbp: Optional[float] = None
        dbp: Optional[float] = None
        bp_unit = "mm[Hg]"
        for comp in components:
            if not isinstance(comp, Mapping):
                continue
            c_code, c_disp = _coding_pair(comp.get("code"))
            vq = comp.get("valueQuantity")
            if isinstance(vq, Mapping) and vq.get("value") is not None:
                v = vq.get("value")
                u = str(vq.get("unit") or vq.get("code") or "").strip()
                comp_cmp = str(vq.get("comparator") or "").strip()
                if not comp_cmp:
                    if c_code == "8480-6" or "systolic" in c_disp.lower():
                        sbp = float(v)
                        bp_unit = u or bp_unit
                    elif c_code == "8462-4" or "diastolic" in c_disp.lower():
                        dbp = float(v)
                        bp_unit = u or bp_unit
                comp_parts.append(f"{c_disp or c_code}: {comp_cmp}{_format_num(v)} {u}".strip())
            else:
                _, vc_disp = _coding_pair(comp.get("valueCodeableConcept"))
                if vc_disp:
                    comp_parts.append(f"{c_disp or c_code}: {vc_disp}")
        if comp_parts:
            return "; ".join(comp_parts), None, ""

    vq = obs.get("valueQuantity")
    if isinstance(vq, Mapping) and vq.get("value") is not None:
        raw_val = vq.get("value")
        unit = str(vq.get("unit") or vq.get("code") or "").strip()
        comparator = str(vq.get("comparator") or "").strip()
        try:
            num_val = float(raw_val)
        except (TypeError, ValueError):
            num_val = None
        val_str = f"{comparator}{_format_num(num_val if num_val is not None else raw_val)}"
        exact_num = None if comparator else num_val
        return (f"{val_str} {unit}".strip() if unit else val_str), exact_num, unit

    if "valueCodeableConcept" in obs:
        _, disp = _coding_pair(obs.get("valueCodeableConcept"))
        return disp, None, ""
    if "valueString" in obs:
        return str(obs.get("valueString") or "").strip(), None, ""
    if "valueBoolean" in obs:
        return str(obs.get("valueBoolean")), None, ""
    if "valueInteger" in obs:
        v = obs.get("valueInteger")
        return str(v), float(v) if isinstance(v, int) else None, ""
    return "", None, ""


def _format_structured_dosage(d_inst: Mapping[str, Any]) -> str:
    """Format narrative and/or structured FHIR Dosage fields (doseAndRate, timing, route, asNeeded)."""
    d_txt = str(d_inst.get("text") or "").strip()
    struct_parts: List[str] = []
    for dr in d_inst.get("doseAndRate") or []:
        if not isinstance(dr, Mapping):
            continue
        dq = dr.get("doseQuantity")
        if isinstance(dq, Mapping) and dq.get("value") is not None:
            dv = _format_num(float(dq["value"]) if isinstance(dq["value"], (int, float)) else dq["value"])
            du = str(dq.get("unit") or dq.get("code") or "").strip()
            struct_parts.append(f"{dv} {du}".strip())
        drange = dr.get("doseRange")
        if isinstance(drange, Mapping):
            low = drange.get("low") or {}
            high = drange.get("high") or {}
            if low.get("value") is not None and high.get("value") is not None:
                u = str(high.get("unit") or low.get("unit") or "").strip()
                struct_parts.append(f"{_format_num(low['value'])}-{_format_num(high['value'])} {u}".strip())
    _, route_disp = _coding_pair(d_inst.get("route"))
    if route_disp:
        struct_parts.append(route_disp)
    timing = d_inst.get("timing")
    if isinstance(timing, Mapping):
        rep = timing.get("repeat")
        if isinstance(rep, Mapping):
            freq = rep.get("frequency")
            period = rep.get("period")
            p_unit = str(rep.get("periodUnit") or "").strip().lower()
            unit_names = {"d": "daily", "wk": "weekly", "mo": "monthly", "h": "h"}
            if freq == 1 and period == 1 and p_unit == "d":
                struct_parts.append("once daily")
            elif freq and period and p_unit:
                if period == 1 and p_unit in {"d", "wk", "mo"}:
                    struct_parts.append(f"{freq}x {unit_names[p_unit]}")
                else:
                    struct_parts.append(f"{freq}x per {period}{p_unit}")
        _, t_code_disp = _coding_pair(timing.get("code"))
        if t_code_disp:
            struct_parts.append(t_code_disp)
    if d_inst.get("asNeededBoolean") is True:
        struct_parts.append("PRN")
    for add_inst in d_inst.get("additionalInstruction") or []:
        _, a_disp = _coding_pair(add_inst)
        if a_disp:
            struct_parts.append(a_disp)
    structured_str = " ".join(struct_parts).strip()
    if d_txt and structured_str and structured_str.lower() not in d_txt.lower():
        return f"{d_txt} ({structured_str})"
    return d_txt or structured_str


def _extract_meta_security_codes(res: Mapping[str, Any]) -> Tuple[str, ...]:
    """Extract uppercased HL7 ``Resource.meta.security[].code`` labels."""
    meta_obj = res.get("meta")
    if not isinstance(meta_obj, Mapping):
        return ()
    codes: List[str] = []
    for sec in meta_obj.get("security") or []:
        if isinstance(sec, Mapping):
            code = str(sec.get("code") or "").strip().upper()
            if code:
                codes.append(code)
    return tuple(dict.fromkeys(codes))


def _extract_obs_interpretation_and_range(
    obs: Mapping[str, Any],
    num_val: Optional[float],
) -> Tuple[Optional[str], Optional[str], bool]:
    """Extract native FHIR ``Observation.interpretation`` and ``referenceRange`` cleanly."""
    interp_code = ""
    interp_disp = ""
    raw_interp = obs.get("interpretation")
    if isinstance(raw_interp, Mapping):
        raw_interp = [raw_interp]
    if isinstance(raw_interp, list):
        for entry in raw_interp:
            if isinstance(entry, Mapping):
                c, d = _coding_pair(entry)
                if c or d:
                    interp_code, interp_disp = c, d
                    break

    range_str: Optional[str] = None
    low_bound: Optional[float] = None
    high_bound: Optional[float] = None
    raw_ranges = obs.get("referenceRange")
    if isinstance(raw_ranges, list) and raw_ranges and isinstance(raw_ranges[0], Mapping):
        rr0 = raw_ranges[0]
        low_obj = rr0.get("low") if isinstance(rr0.get("low"), Mapping) else {}
        high_obj = rr0.get("high") if isinstance(rr0.get("high"), Mapping) else {}
        try:
            low_bound = float(low_obj["value"]) if low_obj.get("value") is not None else None
        except (TypeError, ValueError):
            low_bound = None
        try:
            high_bound = float(high_obj["value"]) if high_obj.get("value") is not None else None
        except (TypeError, ValueError):
            high_bound = None
        if rr0.get("text"):
            range_str = str(rr0["text"]).strip()
        elif low_bound is not None and high_bound is not None:
            u = str(high_obj.get("unit") or low_obj.get("unit") or "").strip()
            u_sp = f" {u}" if u else ""
            range_str = f"{_format_num(low_bound)}-{_format_num(high_bound)}{u_sp}"
        elif high_bound is not None:
            u = str(high_obj.get("unit") or "").strip()
            range_str = f"<= {_format_num(high_bound)}" + (f" {u}" if u else "")
        elif low_bound is not None:
            u = str(low_obj.get("unit") or "").strip()
            range_str = f">= {_format_num(low_bound)}" + (f" {u}" if u else "")

    norm_code = interp_code.upper()
    if norm_code in {"H", "HH", "HU", "L", "LL", "LU", "A", "AA", "CRIT", "POS", "ABNORMAL", "HIGH", "LOW"}:
        is_abnormal = True
        interp_label: Optional[str] = interp_code or interp_disp
    elif norm_code in {"N", "NORMAL"}:
        is_abnormal = False
        interp_label = interp_code
    elif interp_disp:
        is_abnormal = interp_disp.lower() in {"high", "low", "critical", "abnormal", "positive"}
        interp_label = interp_disp
    else:
        is_abnormal = False
        interp_label = None

    if interp_label is None and num_val is not None:
        if low_bound is not None and num_val < low_bound:
            interp_label = "L"
            is_abnormal = True
        elif high_bound is not None and num_val > high_bound:
            interp_label = "H"
            is_abnormal = True
    elif interp_label is None:
        vq = obs.get("valueQuantity")
        if isinstance(vq, Mapping) and vq.get("value") is not None:
            cmp_op = str(vq.get("comparator") or "").strip()
            try:
                cmp_val = float(vq["value"])
            except (TypeError, ValueError):
                cmp_val = None
            if cmp_val is not None:
                if low_bound is not None and (
                    (cmp_op == "<" and cmp_val <= low_bound)
                    or (cmp_op == "<=" and cmp_val < low_bound)
                ):
                    interp_label = "L"
                    is_abnormal = True
                elif high_bound is not None and (
                    (cmp_op == ">" and cmp_val >= high_bound)
                    or (cmp_op == ">=" and cmp_val > high_bound)
                ):
                    interp_label = "H"
                    is_abnormal = True

    return interp_label, range_str, is_abnormal


def _extract_unique_note_narrative(decoded_text: str, *, preserve_all_sections: bool = True) -> str:
    """Extract normalized clinical note narrative, preserving all non-empty Markdown sections faithfully."""
    del preserve_all_sections
    lines = [line.strip() for line in (decoded_text or "").splitlines()]
    if not lines:
        return ""

    sections: Dict[str, List[str]] = {}
    current_header = "preamble"
    for line in lines:
        if line.startswith("#"):
            current_header = line.lstrip("#").strip().lower()
            sections.setdefault(current_header, [])
            continue
        if not line or re.match(r"^[-=_*]{3,}$", line):
            continue
        sections.setdefault(current_header, []).append(line)

    kept_parts: List[str] = []
    has_markdown_headers = any(k != "preamble" for k in sections)
    if has_markdown_headers:
        if sections.get("preamble"):
            pre_body = " ".join(sections["preamble"]).strip()
            if pre_body:
                kept_parts.append(pre_body)
        for header, body_lines in sections.items():
            if header == "preamble":
                continue
            compact_body = " ".join(body_lines).strip()
            if not compact_body:
                continue
            kept_parts.append(f"{header.title()}: {compact_body}")
        if kept_parts:
            return " | ".join(kept_parts)

    non_empty = [l for l in lines if l and not re.match(r"^[-=_*]{3,}$", l)]
    return " ".join(non_empty).strip()


def extract_bundle(
    bundle: Mapping[str, Any],
    *,
    include_conflict_annotations: bool = True,
    enable_privacy_classification: bool = True,
    preserve_intraday_timestamps: bool = False,
    preserve_all_note_sections: bool = False,
    filter_room_equipment_devices: bool = True,
) -> ExtractedChart:
    """Parse a FHIR R4 Bundle dict into normalized `ClinicalItem` objects and numeric series points."""
    entries = bundle.get("entry") or []
    resources: List[Mapping[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, Mapping):
            res = entry.get("resource")
            if isinstance(res, Mapping) and res.get("resourceType"):
                if not str(res.get("id") or "").strip():
                    rtype_slug = str(res.get("resourceType") or "resource").lower()
                    res = {**res, "id": f"anon-{rtype_slug}-{idx}"}
                resources.append(res)
    if not resources and bundle.get("resourceType") and bundle.get("resourceType") != "Bundle":
        res = bundle
        if not str(res.get("id") or "").strip():
            rtype_slug = str(res.get("resourceType") or "resource").lower()
            res = {**res, "id": f"anon-{rtype_slug}-1"}
        resources = [res]

    excluded_counts: Dict[str, int] = {}
    orgs_by_id: Dict[str, str] = {}
    pracs_by_id: Dict[str, str] = {}
    meds_by_id: Dict[str, Tuple[str, str]] = {}
    encounters_by_id: Dict[str, Mapping[str, Any]] = {}
    obs_by_id: Dict[str, Mapping[str, Any]] = {}
    conds_by_id: Dict[str, str] = {}
    provenance_org_by_target: Dict[str, str] = {}

    patient_res: Optional[Mapping[str, Any]] = None

    # Pass 1: Index lookup tables (Organization, Practitioner, Medication, Encounter, Observation, Condition)
    for res in resources:
        rtype = str(res.get("resourceType") or "")
        rid = str(res.get("id") or "")
        if rtype in NON_CLINICAL_RESOURCE_TYPES:
            excluded_counts[rtype] = excluded_counts.get(rtype, 0) + 1
            continue
        if rtype == "Patient" and patient_res is None:
            patient_res = res
        elif rtype == "Organization" and rid:
            orgs_by_id[rid] = str(res.get("name") or rid).strip()
            excluded_counts[rtype] = excluded_counts.get(rtype, 0) + 1
        elif rtype == "Practitioner" and rid:
            names = res.get("name") or []
            if names and isinstance(names[0], Mapping):
                given = " ".join(names[0].get("given") or [])
                family = str(names[0].get("family") or "")
                pracs_by_id[rid] = f"{given} {family}".strip()
            excluded_counts[rtype] = excluded_counts.get(rtype, 0) + 1
        elif rtype == "Medication" and rid:
            meds_by_id[rid] = _coding_pair(res.get("code"))
            excluded_counts[rtype] = excluded_counts.get(rtype, 0) + 1
        elif rtype == "Encounter" and rid:
            encounters_by_id[rid] = res
        elif rtype == "Observation" and rid:
            obs_by_id[rid] = res
        elif rtype == "Condition" and rid:
            _, c_disp = _coding_pair(res.get("code"))
            if c_disp:
                conds_by_id[rid] = c_disp
        elif rtype in LOOKUP_ONLY_RESOURCE_TYPES and rtype != "Provenance":
            excluded_counts[rtype] = excluded_counts.get(rtype, 0) + 1

    # Pass 1b: Invert Provenance.target[] -> source organization/author
    for res in resources:
        if res.get("resourceType") != "Provenance":
            continue
        excluded_counts["Provenance"] = excluded_counts.get("Provenance", 0) + 1
        agent_name = ""
        for agent in res.get("agent") or []:
            if not isinstance(agent, Mapping):
                continue
            who = agent.get("who") or {}
            on_behalf = agent.get("onBehalfOf") or {}
            org_id = _strip_urn(on_behalf.get("reference") or who.get("reference"))
            disp = str(on_behalf.get("display") or who.get("display") or orgs_by_id.get(org_id) or "").strip()
            if disp:
                agent_name = disp
                break
        if agent_name:
            for tgt in res.get("target") or []:
                if isinstance(tgt, Mapping):
                    tid = _strip_urn(tgt.get("reference"))
                    if tid and tid not in provenance_org_by_target:
                        provenance_org_by_target[tid] = agent_name

    def resolve_source_org(res: Mapping[str, Any]) -> Optional[str]:
        rid = str(res.get("id") or "")
        # Check explicit serviceProvider / recorder / asserter / requester / performer first
        for role_field in ("serviceProvider", "recorder", "asserter", "requester"):
            role_obj = res.get(role_field)
            if isinstance(role_obj, Mapping):
                r_ref = _strip_urn(role_obj.get("reference"))
                r_disp = str(
                    role_obj.get("display") or orgs_by_id.get(r_ref) or pracs_by_id.get(r_ref) or ""
                ).strip()
                if r_disp:
                    return r_disp
        for perf in res.get("performer") or []:
            if isinstance(perf, Mapping):
                actor = perf.get("actor") or perf
                p_ref = _strip_urn(actor.get("reference"))
                p_disp = str(actor.get("display") or orgs_by_id.get(p_ref) or pracs_by_id.get(p_ref) or "").strip()
                if p_disp:
                    return p_disp
        enc_ref = _strip_urn((res.get("encounter") or {}).get("reference"))
        if enc_ref and enc_ref in encounters_by_id:
            enc_sp = encounters_by_id[enc_ref].get("serviceProvider")
            if isinstance(enc_sp, Mapping):
                sp_disp = str(
                    enc_sp.get("display") or orgs_by_id.get(_strip_urn(enc_sp.get("reference"))) or ""
                ).strip()
                if sp_disp:
                    return sp_disp
        prov_org = provenance_org_by_target.get(rid)
        if prov_org:
            return prov_org
        meta_obj = res.get("meta")
        if isinstance(meta_obj, Mapping):
            for tag in meta_obj.get("tag") or []:
                if isinstance(tag, Mapping) and str(tag.get("system") or "").endswith(
                    "/wahealthsummary-source-name"
                ):
                    t_code = str(tag.get("code") or tag.get("display") or "").strip()
                    if t_code:
                        return t_code
        return None

    def _classify_res_sensitivity(
        res: Mapping[str, Any],
        code: Optional[str],
        display: str,
        categories: Sequence[str] = (),
        *,
        fallback_kb_sensitivity: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not enable_privacy_classification:
            return None, None
        sec_codes = _extract_meta_security_codes(res)
        sens, src = classify_sensitivity_with_source(
            code, display, categories, meta_security_codes=sec_codes
        )
        if not sens and fallback_kb_sensitivity:
            return str(fallback_kb_sensitivity), "kb_coded_privacy"
        return sens, src

    items: List[ClinicalItem] = []
    numeric_points: List[NumericPoint] = []
    all_clinical_dates: List[str] = []

    _skip_ref_date_statuses = {
        "planned",
        "booked",
        "scheduled",
        "proposed",
        "cancelled",
        "entered-in-error",
        "noshow",
    }

    # Pre-scan clinical dates to establish reference_date (excluding Patient, Device, Coverage & future/cancelled items)
    for res in resources:
        rtype = str(res.get("resourceType") or "")
        if (
            rtype in NON_CLINICAL_RESOURCE_TYPES
            or rtype in LOOKUP_ONLY_RESOURCE_TYPES
            or rtype in {"Patient", "Device", "Coverage"}
        ):
            continue
        raw_st = (
            _codeable_status(res.get("status"))
            or _codeable_status(res.get("clinicalStatus"))
        )
        ver_st = _codeable_status(res.get("verificationStatus"))
        if raw_st in _skip_ref_date_statuses or ver_st in _skip_ref_date_statuses:
            continue
        for field_name in (
            "effectiveDateTime",
            "occurrenceDateTime",
            "onsetDateTime",
            "authoredOn",
            "performedDateTime",
            "recordedDate",
            "issued",
            "date",
        ):
            d = _iso_date(res.get(field_name))
            if d:
                all_clinical_dates.append(d)
                break
        else:
            for period_field in ("period", "effectivePeriod", "performedPeriod"):
                p = res.get(period_field)
                if isinstance(p, Mapping):
                    d = _iso_date(p.get("end") or p.get("start"))
                    if d:
                        all_clinical_dates.append(d)
                        break

    reference_date = max(all_clinical_dates) if all_clinical_dates else None

    # 1. Extract Patient Demographics Header
    patient_id: Optional[str] = None
    patient_name = "Unknown Patient"
    birth_date: Optional[str] = None
    if patient_res is not None:
        patient_id = str(patient_res.get("id") or "patient-1")
        birth_date = _iso_date(patient_res.get("birthDate"))
        names = patient_res.get("name") or []
        if names and isinstance(names[0], Mapping):
            chosen = next((n for n in names if n.get("use") == "official"), names[0])
            given = " ".join(chosen.get("given") or [])
            family = str(chosen.get("family") or "")
            patient_name = f"{given} {family}".strip() or patient_name
        gender = str(patient_res.get("gender") or "unknown")
        age_str = ""
        if birth_date and reference_date:
            try:
                b_dt = date.fromisoformat(birth_date)
                r_dt = date.fromisoformat(reference_date)
                age_yrs = int((r_dt - b_dt).days // 365.2425)
                age_str = f", Age: {age_yrs}y (as of {reference_date})"
            except ValueError:
                pass
        langs: List[str] = []
        for comm in patient_res.get("communication") or []:
            if isinstance(comm, Mapping):
                _, l_disp = _coding_pair(comm.get("language"))
                if l_disp:
                    langs.append(l_disp)
        lang_str = f", Language: {', '.join(langs)}" if langs else ""
        p_date = reference_date or birth_date
        items.append(
            ClinicalItem(
                resource_type="Patient",
                resource_uuid=patient_id,
                date=p_date,
                clinical_date=None,
                date_kind="administrative",
                section="patient",
                code="patient-demographics",
                display=patient_name,
                text=f"Patient: {patient_name}, DOB: {birth_date or 'unknown'}, Gender: {gender}{age_str}{lang_str}",
                mandatory=True,
                status="active",
            )
        )

    # 2. Extract AllergyIntolerance (100% Mandatory IPS Safety Core)
    for res in resources:
        if res.get("resourceType") != "AllergyIntolerance":
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("code"))
        c_status = _codeable_status(res.get("clinicalStatus")) or "active"
        v_status = _codeable_status(res.get("verificationStatus"))
        crit = str(res.get("criticality") or "").strip()
        cats = [str(c) for c in (res.get("category") or []) if c]
        reactions: List[str] = []
        for rxn in res.get("reaction") or []:
            if isinstance(rxn, Mapping):
                sev = str(rxn.get("severity") or "").strip()
                for manif in rxn.get("manifestation") or []:
                    _, m_disp = _coding_pair(manif)
                    if m_disp:
                        reactions.append(f"{m_disp} ({sev})" if sev else m_disp)
        rec_date = _iso_date(res.get("recordedDate") or res.get("onsetDateTime"))
        sens, sens_source = _classify_res_sensitivity(res, code, disp, cats)
        qualifiers = [f"status: {c_status}"]
        if v_status:
            qualifiers.append(f"verification: {v_status}")
        if crit:
            qualifiers.append(f"criticality: {crit}")
        if cats:
            qualifiers.append(f"category: {'/'.join(cats)}")
        if reactions:
            qualifiers.append(f"reaction: {', '.join(reactions)}")
        allergy_meta: Dict[str, Any] = {}
        if sens_source:
            allergy_meta["sensitivity_source"] = sens_source
        if v_status:
            allergy_meta["verification_status"] = v_status
        items.append(
            ClinicalItem(
                resource_type="AllergyIntolerance",
                resource_uuid=rid,
                date=rec_date,
                clinical_date=rec_date,
                date_kind="clinical_event" if rec_date else "unknown",
                section="allergy",
                code=code or disp,
                display=disp or code or "Unspecified allergy",
                text=f"Allergy: {disp or code} ({', '.join(qualifiers)})",
                mandatory=True,
                status=c_status,
                source_org=resolve_source_org(res),
                sensitivity=sens,
                metadata=allergy_meta,
            )
        )

    # 3. Extract Condition (Active vs Resolved/Historical)
    raw_conditions: List[Dict[str, Any]] = []
    for res in resources:
        if res.get("resourceType") != "Condition":
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("code"))
        c_status = _codeable_status(res.get("clinicalStatus")) or "active"
        v_status = _codeable_status(res.get("verificationStatus"))
        onset = _iso_date(res.get("onsetDateTime") or res.get("recordedDate"))
        abatement = _iso_date(res.get("abatementDateTime"))
        cats = _categories(res)
        cond_kb = resolve_condition_code(code, disp)
        kb_sens = cond_kb.get("sensitivity") if cond_kb.get("resolved_by") != "heuristic_fallback" else None
        sens, sens_source = _classify_res_sensitivity(
            res, code, disp, cats, fallback_kb_sensitivity=kb_sens
        )
        raw_conditions.append(
            {
                "resourceUuid": rid,
                "code": code,
                "display": disp or str(cond_kb.get("display") or code or "Condition"),
                "status": c_status,
                "verification": v_status,
                "onset": onset,
                "abatement": abatement,
                "categories": cats,
                "sensitivity": sens,
                "sensitivity_source": sens_source,
                "source_org": resolve_source_org(res),
                "kb_resolution": cond_kb,
            }
        )

    _non_actionable_cond_verif = {"refuted", "entered-in-error"}
    active_cond_dicts = [
        c
        for c in raw_conditions
        if c["status"] in {"active", "recurrence", "relapse"}
        and c["verification"] not in _non_actionable_cond_verif
    ]
    resolved_cond_dicts = [
        c
        for c in raw_conditions
        if not (
            c["status"] in {"active", "recurrence", "relapse"}
            and c["verification"] not in _non_actionable_cond_verif
        )
    ]
    condition_conflicts = (
        detect_condition_conflicts(active_cond_dicts, resolved_cond_dicts)
        if include_conflict_annotations
        else {}
    )

    for c in raw_conditions:
        rid = c["resourceUuid"]
        is_active = (
            c["status"] in {"active", "recurrence", "relapse"}
            and c["verification"] not in _non_actionable_cond_verif
        )
        conflict = condition_conflicts.get(rid)
        sens = c["sensitivity"]
        event_date = c["onset"] if is_active else (c["abatement"] or c["onset"])
        details = [f"status: {c['status']}"]
        if c["verification"]:
            details.append(f"verification: {c['verification']}")
        if c["onset"]:
            details.append(f"onset: {c['onset']}")
        if c["abatement"]:
            details.append(f"resolved: {c['abatement']}")
        if c["source_org"]:
            details.append(f"source: {c['source_org']}")
        if c["code"]:
            details.append(f"SNOMED: {c['code']}")
        cond_kb = c["kb_resolution"]
        cond_meta: Dict[str, Any] = {
            "onset": c["onset"],
            "abatement": c["abatement"],
            "kb_resolution": cond_kb,
        }
        if c["verification"]:
            cond_meta["verification_status"] = c["verification"]
        if c["sensitivity_source"]:
            cond_meta["sensitivity_source"] = c["sensitivity_source"]
        items.append(
            ClinicalItem(
                resource_type="Condition",
                resource_uuid=rid,
                date=event_date,
                clinical_date=event_date,
                date_kind="clinical_event" if event_date else "unknown",
                section="condition_active" if is_active else "condition_resolved",
                code=c["code"] or c["display"],
                display=c["display"],
                text=f"Condition: {c['display']} ({', '.join(details)})",
                mandatory=is_active or bool(conflict),
                status=c["status"],
                end_date=c["abatement"],
                source_org=c["source_org"],
                sensitivity=sens,
                conflict_note=conflict,
                is_routine=(sens == "SDOH-SOCIAL"),
                metadata=cond_meta,
            )
        )

    # 4. Extract Unified Medications (MedicationRequest, MedicationAdministration, MedicationStatement, MedicationDispense)
    med_types = {
        "MedicationRequest",
        "MedicationAdministration",
        "MedicationStatement",
        "MedicationDispense",
    }
    raw_meds: List[Dict[str, Any]] = []
    for res in resources:
        rtype = str(res.get("resourceType") or "")
        if rtype not in med_types:
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("medicationCodeableConcept"))
        if not disp and isinstance(res.get("medicationReference"), Mapping):
            m_ref = _strip_urn(res["medicationReference"].get("reference"))
            if m_ref in meds_by_id:
                code, disp = meds_by_id[m_ref]
            else:
                disp = str(res["medicationReference"].get("display") or "")
        status = str(res.get("status") or "unknown").strip().lower()
        m_date = _iso_date(
            res.get("authoredOn")
            or res.get("effectiveDateTime")
            or (res.get("effectivePeriod") or {}).get("start")
            or res.get("whenHandedOver")
        )
        end_date = _iso_date((res.get("effectivePeriod") or {}).get("end"))

        # Extract dosage text and/or structured dosage instructions if available
        dose_strs: List[str] = []
        for d_inst in res.get("dosageInstruction") or []:
            if isinstance(d_inst, Mapping):
                d_txt = _format_structured_dosage(d_inst)
                if d_txt:
                    dose_strs.append(d_txt)
        if not dose_strs and isinstance(res.get("dosage"), Mapping):
            d_txt = _format_structured_dosage(res["dosage"])
            if d_txt:
                dose_strs.append(d_txt)

        # Extract reason (reasonCode or reasonReference -> Condition)
        reasons: List[str] = []
        for rc in res.get("reasonCode") or []:
            _, r_disp = _coding_pair(rc)
            if r_disp:
                reasons.append(r_disp)
        for rr in res.get("reasonReference") or []:
            if isinstance(rr, Mapping):
                c_id = _strip_urn(rr.get("reference"))
                r_disp = str(rr.get("display") or conds_by_id.get(c_id) or "").strip()
                if r_disp:
                    reasons.append(r_disp)

        setting = "inpatient/administered" if rtype == "MedicationAdministration" else "prescribed"
        is_active = status in {"active", "in-progress", "on-hold"}
        med_kb = resolve_medication_code(code, disp)
        resolved_disp = disp or (
            str(med_kb.get("display") or "")
            if med_kb.get("resolved_by") != "heuristic_fallback"
            else ""
        )
        ingredient = str(
            med_kb.get("ingredient")
            if med_kb.get("resolved_by") != "heuristic_fallback" and med_kb.get("ingredient")
            else normalize_drug_ingredient(resolved_disp or code)
        )
        linked_order_uuids: List[str] = []
        for link_field in ("basedOn", "partOf", "request", "prescription", "authorizingPrescription"):
            raw_link = res.get(link_field)
            if isinstance(raw_link, Mapping) and raw_link.get("reference"):
                linked_order_uuids.append(_strip_urn(str(raw_link.get("reference"))))
            elif isinstance(raw_link, list):
                for entry_link in raw_link:
                    if isinstance(entry_link, Mapping) and entry_link.get("reference"):
                        linked_order_uuids.append(_strip_urn(str(entry_link.get("reference"))))
        med_sens, med_sens_src = _classify_res_sensitivity(res, code, resolved_disp or disp)
        raw_meds.append(
            {
                "resourceType": rtype,
                "resourceUuid": rid,
                "code": code,
                "display": resolved_disp or code or "Medication",
                "status": status,
                "is_active": is_active,
                "date": m_date,
                "end_date": end_date,
                "dose_strs": dose_strs,
                "reasons": reasons,
                "setting": setting,
                "ingredient_key": ingredient,
                "source_org": resolve_source_org(res),
                "sensitivity": med_sens,
                "sensitivity_source": med_sens_src,
                "linked_order_uuids": tuple(dict.fromkeys(u for u in linked_order_uuids if u)),
                "kb_resolution": med_kb,
            }
        )

    med_conflicts = (
        detect_medication_conflicts([m for m in raw_meds if m["is_active"]])
        if include_conflict_annotations
        else {}
    )
    for m in raw_meds:
        rid = m["resourceUuid"]
        m_conflict = med_conflicts.get(rid)
        med_kb = m["kb_resolution"]
        ing_grp = str(med_kb.get("ingredient") or m["ingredient_key"])
        t_class = str(med_kb.get("therapeutic_class") or "medication")
        quals = [f"status: {m['status']}", f"type: {m['setting']}"]
        if m["dose_strs"]:
            quals.append(f"dosage: {'; '.join(m['dose_strs'][:2])}")
        if m["reasons"]:
            quals.append(f"reason: {', '.join(dict.fromkeys(m['reasons']))}")
        if m["source_org"]:
            quals.append(f"source: {m['source_org']}")
        if m["code"]:
            quals.append(f"RxNorm: {m['code']}")
        med_meta: Dict[str, Any] = {
            "reasons": m["reasons"],
            "dosage": m["dose_strs"],
            "setting": m["setting"],
            "ingredient_group": ing_grp,
            "therapeutic_class": t_class,
            "kb_resolution": med_kb,
        }
        if m["sensitivity_source"]:
            med_meta["sensitivity_source"] = m["sensitivity_source"]
        items.append(
            ClinicalItem(
                resource_type=m["resourceType"],
                resource_uuid=rid,
                date=m["date"],
                clinical_date=m["date"],
                date_kind="clinical_event" if m["date"] else "unknown",
                section="med_active" if m["is_active"] else "med_historical",
                code=m["code"] or m["ingredient_key"],
                display=m["display"],
                text=f"Medication: {m['display']} ({', '.join(quals)})",
                mandatory=m["is_active"],
                status=m["status"],
                end_date=m["end_date"],
                source_org=m["source_org"],
                sensitivity=m["sensitivity"],
                conflict_note=m_conflict,
                ingredient_key=m["ingredient_key"],
                metadata=med_meta,
            )
        )

    # 5. Extract Immunization
    raw_imms: List[Dict[str, Any]] = []
    for res in resources:
        if res.get("resourceType") != "Immunization":
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("vaccineCode"))
        occ_date = _iso_date(res.get("occurrenceDateTime"))
        status = str(res.get("status") or "completed").strip().lower()
        imm_sens, imm_sens_src = _classify_res_sensitivity(res, code, disp)
        raw_imms.append(
            {
                "resourceUuid": rid,
                "code": code,
                "display": disp or code or "Vaccine",
                "date": occ_date,
                "status": status,
                "source_org": resolve_source_org(res),
                "sensitivity": imm_sens,
                "sensitivity_source": imm_sens_src,
                "group": classify_vaccine_group(code, disp),
            }
        )

    imm_anomalies = (
        annotate_immunization_anomalies(raw_imms, birth_date)
        if include_conflict_annotations
        else {}
    )
    imms_by_group: Dict[str, List[str]] = {}
    for imm in raw_imms:
        imms_by_group.setdefault(imm["group"], []).append(imm["resourceUuid"])

    for imm in raw_imms:
        rid = imm["resourceUuid"]
        anom = imm_anomalies.get(rid)
        quals = [f"status: {imm['status']}"]
        if imm["code"]:
            quals.append(f"CVX: {imm['code']}")
        if imm["source_org"]:
            quals.append(f"source: {imm['source_org']}")
        series_members: Tuple[str, ...] = (
            tuple(imms_by_group.get(imm["group"], [rid]))
            if anom and anom.startswith("acip_schedule_note:")
            else ()
        )
        imm_kb = resolve_immunization_code(imm["code"], imm["display"])
        imm_meta: Dict[str, Any] = {
            "vaccine_group": imm_kb.get("vaccine_group", imm["group"]),
            "kb_resolution": imm_kb,
        }
        if imm["sensitivity_source"]:
            imm_meta["sensitivity_source"] = imm["sensitivity_source"]
        items.append(
            ClinicalItem(
                resource_type="Immunization",
                resource_uuid=rid,
                date=imm["date"],
                clinical_date=imm["date"],
                date_kind="clinical_event" if imm["date"] else "unknown",
                section="immunization",
                code=imm["code"] or imm["group"],
                display=imm["display"],
                text=f"Immunization: {imm['display']} ({', '.join(quals)})",
                mandatory=bool(anom),
                status=imm["status"],
                source_org=imm["source_org"],
                sensitivity=imm["sensitivity"],
                conflict_note=anom,
                ingredient_key=imm["group"],
                member_uuids=series_members,
                metadata=imm_meta,
            )
        )

    # 6. Deduplicate Base64 Clinical Notes across DocumentReference & DiagnosticReport.presentedForm
    notes_by_hash: Dict[str, Dict[str, Any]] = {}
    dedup_note_count = 0

    for res in resources:
        rtype = str(res.get("resourceType") or "")
        rid = str(res.get("id") or "")
        if rtype == "DocumentReference":
            n_date = _iso_date(res.get("date") or (res.get("context") or {}).get("period", {}).get("start"))
            _, n_type = _coding_pair(res.get("type"))
            sec_codes = _extract_meta_security_codes(res)
            for content_item in res.get("content") or []:
                att = (content_item or {}).get("attachment") or {}
                b64_data = att.get("data")
                if isinstance(b64_data, str) and b64_data:
                    digest = hashlib.sha256(b64_data.strip().encode("ascii", errors="ignore")).hexdigest()
                    if digest in notes_by_hash:
                        notes_by_hash[digest]["member_uuids"].append(rid)
                        if n_date:
                            notes_by_hash[digest]["dates"].append(n_date)
                        notes_by_hash[digest]["meta_security"].extend(sec_codes)
                        dedup_note_count += 1
                    else:
                        try:
                            decoded = base64.b64decode(b64_data).decode("utf-8", errors="replace")
                        except Exception:
                            decoded = ""
                        notes_by_hash[digest] = {
                            "primary_type": "DocumentReference",
                            "primary_uuid": rid,
                            "date": n_date,
                            "dates": [n_date] if n_date else [],
                            "title": n_type or "Clinical Note",
                            "decoded": decoded,
                            "source_org": resolve_source_org(res),
                            "member_uuids": [rid],
                            "meta_security": list(sec_codes),
                        }
        elif rtype == "DiagnosticReport" and (res.get("presentedForm") or []):
            n_date = _iso_date(res.get("effectiveDateTime") or res.get("issued"))
            _, n_type = _coding_pair(res.get("code"))
            sec_codes = _extract_meta_security_codes(res)
            for pf in res.get("presentedForm") or []:
                b64_data = (pf or {}).get("data")
                if isinstance(b64_data, str) and b64_data:
                    digest = hashlib.sha256(b64_data.strip().encode("ascii", errors="ignore")).hexdigest()
                    if digest in notes_by_hash:
                        notes_by_hash[digest]["member_uuids"].append(rid)
                        if n_date:
                            notes_by_hash[digest]["dates"].append(n_date)
                        notes_by_hash[digest]["meta_security"].extend(sec_codes)
                        dedup_note_count += 1
                    else:
                        try:
                            decoded = base64.b64decode(b64_data).decode("utf-8", errors="replace")
                        except Exception:
                            decoded = ""
                        notes_by_hash[digest] = {
                            "primary_type": "DiagnosticReport",
                            "primary_uuid": rid,
                            "date": n_date,
                            "dates": [n_date] if n_date else [],
                            "title": n_type or "Clinical Note",
                            "decoded": decoded,
                            "source_org": resolve_source_org(res),
                            "member_uuids": [rid],
                            "meta_security": list(sec_codes),
                        }

    for note_info in notes_by_hash.values():
        snippet = _extract_unique_note_narrative(
            note_info["decoded"],
            preserve_all_sections=preserve_all_note_sections,
        )
        if not snippet:
            continue
        members = tuple(dict.fromkeys(note_info["member_uuids"]))
        distinct_dates = sorted({d for d in (note_info.get("dates") or []) if d})
        primary_date = distinct_dates[-1] if distinct_dates else note_info["date"]
        extra_dates_str = (
            f" (recorded across dates: {', '.join(distinct_dates)})"
            if len(distinct_dates) > 1
            else ""
        )
        if enable_privacy_classification:
            note_sens, note_sens_src = classify_sensitivity_with_source(
                "",
                f"{note_info['title']} {note_info['decoded']} {snippet}",
                meta_security_codes=note_info.get("meta_security") or (),
            )
        else:
            note_sens, note_sens_src = None, None
        note_meta: Dict[str, Any] = {
            "full_text": snippet,
            "raw_decoded_text": note_info["decoded"],
            "title": note_info["title"],
            "duplicate_wrapper_count": len(members),
            **({"dates": distinct_dates} if distinct_dates else {}),
            **({"sensitivity_source": note_sens_src} if note_sens_src else {}),
        }
        items.append(
            ClinicalItem(
                resource_type=note_info["primary_type"],
                resource_uuid=note_info["primary_uuid"],
                date=primary_date,
                clinical_date=primary_date,
                date_kind="clinical_event" if primary_date else "unknown",
                section="clinical_note",
                code="clinical-note",
                display=note_info["title"],
                text=f"Clinical Note — {note_info['title']}: {snippet}{extra_dates_str}",
                mandatory=False,
                status="current",
                source_org=note_info["source_org"],
                sensitivity=note_sens,
                member_uuids=members,
                metadata=note_meta,
            )
        )

    # 7. Extract DiagnosticReport Lab & Survey Panels (and track claimed child Observations)
    def _extract_obs_components(
        obs_res: Mapping[str, Any],
        obs_uuid: str,
        obs_date: str,
        obs_cat: str,
        obs_cats: Sequence[str],
    ) -> Tuple[List[NumericPoint], List[str], Optional[str]]:
        pts: List[NumericPoint] = []
        comp_sens: List[str] = []
        comp_src: Optional[str] = None
        parent_sec_codes = _extract_meta_security_codes(obs_res)
        for comp in obs_res.get("component") or []:
            if not isinstance(comp, Mapping):
                continue
            c_code, c_disp = _coding_pair(comp.get("code"))
            vq = comp.get("valueQuantity")
            c_val_str = ""
            c_num_for_rr: Optional[float] = None
            if isinstance(vq, Mapping) and vq.get("value") is not None:
                c_cmp = str(vq.get("comparator") or "").strip()
                c_unit = str(vq.get("unit") or vq.get("code") or "").strip()
                c_val_str = f"{c_cmp}{_format_num(vq.get('value'))} {c_unit}".strip()
                if not c_cmp:
                    try:
                        c_num = float(vq["value"])
                        c_num_for_rr = c_num
                        c_interp, c_rr, c_abn = _extract_obs_interpretation_and_range(
                            comp, c_num
                        )
                        pts.append(
                            NumericPoint(
                                resource_uuid=obs_uuid,
                                date=obs_date,
                                code=c_code or c_disp,
                                concept=c_disp or c_code,
                                value=c_num,
                                unit=c_unit,
                                category=obs_cat,
                                interpretation=c_interp,
                                reference_range=c_rr,
                                is_abnormal=c_abn,
                            )
                        )
                    except (TypeError, ValueError):
                        pass
            else:
                _, vc_disp = _coding_pair(comp.get("valueCodeableConcept"))
                c_val_str = vc_disp
            if enable_privacy_classification:
                cs, csrc = classify_sensitivity_with_source(
                    c_code, f"{c_disp} {c_val_str}", obs_cats, meta_security_codes=parent_sec_codes
                )
                if cs:
                    comp_sens.append(cs)
                    if cs == "SDOH/PRIVACY-SENSITIVE" or not comp_src:
                        comp_src = csrc
        return pts, comp_sens, comp_src

    claimed_obs_ids: Set[str] = set()
    for res in resources:
        if res.get("resourceType") != "DiagnosticReport":
            continue
        result_refs = res.get("result") or []
        if not result_refs:
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("code"))
        p_date = _iso_date(res.get("effectiveDateTime") or res.get("issued"))
        is_survey = bool(_SURVEY_PANEL_PATTERNS.search(disp or ""))

        member_uuids: List[str] = [rid]
        child_parts: List[str] = []
        child_sensitivities: List[str] = []
        child_sens_sources: List[str] = []

        for ref_obj in result_refs:
            if not isinstance(ref_obj, Mapping):
                continue
            oid = _strip_urn(ref_obj.get("reference"))
            if not oid:
                continue
            claimed_obs_ids.add(oid)
            member_uuids.append(oid)
            child_obs = obs_by_id.get(oid)
            if not child_obs:
                r_disp = str(ref_obj.get("display") or "").strip()
                if r_disp:
                    child_parts.append(r_disp)
                continue
            o_code, o_disp = _coding_pair(child_obs.get("code"))
            val_str, num_val, unit = _format_obs_value(child_obs)
            o_interp, o_range, o_abn = _extract_obs_interpretation_and_range(child_obs, num_val)
            o_cats = _categories(child_obs)
            o_date_raw = child_obs.get("effectiveDateTime") or child_obs.get("issued")
            o_date = (
                (_iso_timestamp(o_date_raw) if preserve_intraday_timestamps else _iso_date(o_date_raw))
                or p_date
            )
            o_primary_cat = o_cats[0] if o_cats else ("survey" if is_survey else "laboratory")
            o_sens, o_sens_src = _classify_res_sensitivity(
                child_obs, o_code, f"{o_disp} {val_str}", o_cats
            )
            if o_sens:
                child_sensitivities.append(o_sens)
            if o_sens_src:
                child_sens_sources.append(o_sens_src)
            short_label = o_disp or o_code or "item"
            if val_str:
                annot_bits: List[str] = []
                if o_interp:
                    annot_bits.append(o_interp)
                if o_range:
                    annot_bits.append(f"ref: {o_range}")
                annot_str = f" ({'; '.join(annot_bits)})" if annot_bits else ""
                child_parts.append(f"{short_label}: {val_str}{annot_str}")
            if child_obs.get("component"):
                comp_pts, comp_sens, comp_src = _extract_obs_components(
                    child_obs, oid, o_date or "", o_primary_cat, o_cats
                )
                numeric_points.extend(comp_pts)
                child_sensitivities.extend(comp_sens)
                if comp_src:
                    child_sens_sources.append(comp_src)
            elif num_val is not None:
                numeric_points.append(
                    NumericPoint(
                        resource_uuid=oid,
                        date=o_date or "",
                        code=o_code or short_label,
                        concept=short_label,
                        value=num_val,
                        unit=unit,
                        category=o_primary_cat,
                        interpretation=o_interp,
                        reference_range=o_range,
                        is_abnormal=o_abn,
                    )
                )

        dr_sens, dr_sens_src = _classify_res_sensitivity(res, code, disp)
        if enable_privacy_classification:
            if "SDOH/PRIVACY-SENSITIVE" in child_sensitivities or dr_sens == "SDOH/PRIVACY-SENSITIVE":
                panel_sens: Optional[str] = "SDOH/PRIVACY-SENSITIVE"
                panel_sens_src: Optional[str] = (
                    dr_sens_src
                    if dr_sens == "SDOH/PRIVACY-SENSITIVE" and dr_sens_src
                    else (child_sens_sources[0] if child_sens_sources else "kb_coded_privacy")
                )
            else:
                panel_sens = dr_sens
                panel_sens_src = dr_sens_src
        else:
            panel_sens = None
            panel_sens_src = None
        joined_children = "; ".join(child_parts) if child_parts else "completed"
        panel_prefix = "Survey Panel" if is_survey else "Diagnostic Panel"
        panel_kb = resolve_lab_code(code, disp) if not is_survey else {}
        panel_meta: Dict[str, Any] = {}
        if panel_kb:
            panel_meta["kb_resolution"] = panel_kb
        if panel_sens_src:
            panel_meta["sensitivity_source"] = panel_sens_src
        items.append(
            ClinicalItem(
                resource_type="DiagnosticReport",
                resource_uuid=rid,
                date=p_date,
                clinical_date=p_date,
                date_kind="clinical_event" if p_date else "unknown",
                section="survey_panel" if is_survey else "lab_panel",
                code=code or disp,
                display=disp or code or "Diagnostic Report",
                text=f"{panel_prefix} — {disp or code}: {joined_children}",
                mandatory=False,
                status=str(res.get("status") or "final").lower(),
                source_org=resolve_source_org(res),
                sensitivity=panel_sens,
                is_routine=is_survey,
                member_uuids=tuple(dict.fromkeys(member_uuids)),
                metadata=panel_meta,
            )
        )

    # 8. Extract Standalone Observations (Vitals, Unbundled Labs, Social History, Survey answers)
    for res in resources:
        if res.get("resourceType") != "Observation":
            continue
        rid = str(res.get("id") or "")
        if rid in claimed_obs_ids:
            continue
        code, disp = _coding_pair(res.get("code"))
        raw_obs_dt = res.get("effectiveDateTime") or res.get("issued")
        o_date = _iso_date(raw_obs_dt)
        o_ts = (_iso_timestamp(raw_obs_dt) if preserve_intraday_timestamps else None) or o_date
        cats = _categories(res)
        val_str, num_val, unit = _format_obs_value(res)
        obs_interp, obs_range, obs_abn = _extract_obs_interpretation_and_range(res, num_val)
        sens, sens_src = _classify_res_sensitivity(res, code, f"{disp} {val_str}", cats)

        primary_cat = cats[0] if cats else "observation"
        has_components = bool(res.get("component"))
        if has_components:
            comp_pts, comp_sens, comp_src = _extract_obs_components(
                res, rid, o_ts or o_date or "", primary_cat, cats
            )
            numeric_points.extend(comp_pts)
            if any(pt.is_abnormal for pt in comp_pts):
                obs_abn = True
            if "SDOH/PRIVACY-SENSITIVE" in comp_sens:
                sens = "SDOH/PRIVACY-SENSITIVE"
                sens_src = comp_src or sens_src
            elif not sens and comp_sens:
                sens = comp_sens[0]
                sens_src = comp_src or sens_src
        elif num_val is not None:
            numeric_points.append(
                NumericPoint(
                    resource_uuid=rid,
                    date=o_ts or o_date or "",
                    code=code or disp,
                    concept=disp or code or "Observation",
                    value=num_val,
                    unit=unit,
                    category=primary_cat,
                    interpretation=obs_interp,
                    reference_range=obs_range,
                    is_abnormal=obs_abn,
                )
            )

        if "vital-signs" in cats:
            sec = "vital_sign"
            cls_label = "Vital Sign"
        elif "laboratory" in cats:
            sec = "lab_observation"
            cls_label = "Lab"
        elif "social-history" in cats or sens == "SDOH/PRIVACY-SENSITIVE":
            sec = "social_history"
            cls_label = "Social History"
        elif "survey" in cats or _SURVEY_PANEL_PATTERNS.search(disp or ""):
            sec = "survey_observation"
            cls_label = "Survey"
        else:
            sec = "observation"
            cls_label = "Finding"

        ts_suffix = f" (timestamp: {o_ts})" if (preserve_intraday_timestamps and o_ts and "T" in o_ts) else ""
        annot_bits_o: List[str] = []
        if obs_interp:
            annot_bits_o.append(f"interp: {obs_interp}")
        if obs_range:
            annot_bits_o.append(f"ref: {obs_range}")
        annot_str_o = f" [{'; '.join(annot_bits_o)}]" if annot_bits_o else ""
        body_val = f": {val_str}{annot_str_o}" if val_str else ""
        obs_kb = resolve_lab_code(code, disp) if sec in {"vital_sign", "lab_observation"} else {}
        obs_meta: Dict[str, Any] = {
            "value": val_str,
            "numeric": num_val,
            "unit": unit,
            "category": primary_cat,
            "timestamp": o_ts,
            "is_abnormal": obs_abn,
            **({"interpretation": obs_interp} if obs_interp else {}),
            **({"reference_range": obs_range} if obs_range else {}),
            **({"kb_resolution": obs_kb} if obs_kb else {}),
            **({"sensitivity_source": sens_src} if sens_src else {}),
        }
        items.append(
            ClinicalItem(
                resource_type="Observation",
                resource_uuid=rid,
                date=o_date,
                clinical_date=o_date,
                date_kind="clinical_event" if o_date else "unknown",
                section=sec,
                code=code or disp,
                display=disp or code or "Observation",
                text=f"{cls_label} — {disp or code}{body_val}{ts_suffix}",
                mandatory=False,
                status=str(res.get("status") or "final").lower(),
                source_org=resolve_source_org(res),
                sensitivity=sens,
                is_routine=(sec in {"survey_observation", "vital_sign"}),
                metadata=obs_meta,
            )
        )

    # 9. Extract Procedures (Surgical/Therapeutic vs Routine Screening)
    for res in resources:
        if res.get("resourceType") != "Procedure":
            continue
        rid = str(res.get("id") or "")
        code, disp = _coding_pair(res.get("code"))
        p_date = _iso_date(
            res.get("performedDateTime")
            or (res.get("performedPeriod") or {}).get("start")
            or (res.get("performedPeriod") or {}).get("end")
        )
        status = str(res.get("status") or "completed").lower()
        is_routine = bool(_ROUTINE_SCREENING_PROCEDURE_PATTERNS.search(disp or ""))
        proc_sens, proc_sens_src = _classify_res_sensitivity(res, code, disp)
        reasons: List[str] = []
        for rc in res.get("reasonCode") or []:
            _, r_disp = _coding_pair(rc)
            if r_disp:
                reasons.append(r_disp)
        for rr in res.get("reasonReference") or []:
            if isinstance(rr, Mapping):
                c_id = _strip_urn(rr.get("reference"))
                r_disp = str(rr.get("display") or conds_by_id.get(c_id) or "").strip()
                if r_disp:
                    reasons.append(r_disp)
        reason_str = f" (reason: {', '.join(dict.fromkeys(reasons))})" if reasons else ""
        items.append(
            ClinicalItem(
                resource_type="Procedure",
                resource_uuid=rid,
                date=p_date,
                clinical_date=p_date,
                date_kind="clinical_event" if p_date else "unknown",
                section="procedure_routine" if is_routine else "procedure_major",
                code=code or disp,
                display=disp or code or "Procedure",
                text=f"Procedure: {disp or code}{reason_str}",
                mandatory=False,
                status=status,
                source_org=resolve_source_org(res),
                sensitivity=proc_sens,
                is_routine=is_routine,
                metadata={"sensitivity_source": proc_sens_src} if proc_sens_src else {},
            )
        )

    # 10. Extract CarePlan, Goal, ImagingStudy, Patient-Carried Device, and Encounter
    for res in resources:
        rtype = str(res.get("resourceType") or "")
        rid = str(res.get("id") or "")
        if rtype == "CarePlan":
            status = str(res.get("status") or "active").lower()
            cp_date = _iso_date(
                (res.get("period") or {}).get("start") or res.get("created")
            )
            end_date = _iso_date((res.get("period") or {}).get("end"))
            cats: List[str] = []
            for c_item in res.get("category") or []:
                _, c_disp = _coding_pair(c_item)
                if c_disp and c_disp.lower() != "assess-plan":
                    cats.append(c_disp)
            acts: List[str] = []
            for act in res.get("activity") or []:
                detail = (act or {}).get("detail") or {}
                _, a_disp = _coding_pair(detail.get("code"))
                if a_disp:
                    acts.append(a_disp)
            label = ", ".join(cats) or "Care Plan"
            act_str = f" — activities: {', '.join(acts[:4])}" if acts else ""
            cp_sens, cp_sens_src = _classify_res_sensitivity(res, "", label)
            items.append(
                ClinicalItem(
                    resource_type="CarePlan",
                    resource_uuid=rid,
                    date=cp_date,
                    clinical_date=cp_date,
                    date_kind="clinical_event" if cp_date else "unknown",
                    section="careplan_active" if status == "active" else "careplan_historical",
                    code=label,
                    display=label,
                    text=f"CarePlan: {label} (status: {status}{act_str})",
                    mandatory=False,
                    status=status,
                    end_date=end_date,
                    source_org=resolve_source_org(res),
                    sensitivity=cp_sens,
                    metadata={"sensitivity_source": cp_sens_src} if cp_sens_src else {},
                )
            )
        elif rtype == "Goal":
            lifecycle = str(res.get("lifecycleStatus") or "active").lower()
            g_date = _iso_date(res.get("startDate") or res.get("statusDate"))
            due_dates: List[str] = []
            for tgt in res.get("target") or []:
                if isinstance(tgt, Mapping):
                    dd = _iso_date(tgt.get("dueDate"))
                    if dd:
                        due_dates.append(dd)
            g_code, g_disp = _coding_pair(res.get("description"))
            if not g_disp:
                desc_obj = res.get("description")
                if isinstance(desc_obj, str):
                    g_disp = desc_obj.strip()
            g_label = g_disp or "Patient Health Goal"
            notes_txt = "; ".join(
                str(n.get("text") or "").strip()
                for n in (res.get("note") or [])
                if isinstance(n, Mapping) and str(n.get("text") or "").strip()
            )
            due_str = f", target due: {due_dates[0]}" if due_dates else ""
            note_str = f" — {notes_txt}" if notes_txt else ""
            is_active_goal = lifecycle in {"active", "accepted", "planned", "proposed", "on-hold"}
            src_org = resolve_source_org(res)
            src_suffix = f" [source: {src_org}]" if src_org else ""
            g_sens, g_sens_src = _classify_res_sensitivity(res, g_code, g_label)
            items.append(
                ClinicalItem(
                    resource_type="Goal",
                    resource_uuid=rid,
                    date=g_date,
                    clinical_date=g_date,
                    date_kind="clinical_event" if g_date else "unknown",
                    section="careplan_active" if is_active_goal else "careplan_historical",
                    code=g_code or g_label,
                    display=g_label,
                    text=f"Goal: {g_label} (status: {lifecycle}{due_str}){note_str}{src_suffix}",
                    mandatory=is_active_goal,
                    status=lifecycle,
                    end_date=due_dates[0] if due_dates else None,
                    source_org=src_org,
                    sensitivity=g_sens,
                    metadata={"sensitivity_source": g_sens_src} if g_sens_src else {},
                )
            )
        elif rtype == "ImagingStudy":
            im_date = _iso_date(res.get("started"))
            _, proc_disp = _coding_pair((res.get("procedureCode") or [{}])[0] if res.get("procedureCode") else {})
            series_list = res.get("series") or []
            body_sites: List[str] = []
            modalities: List[str] = []
            for s in series_list:
                if isinstance(s, Mapping):
                    _, bs_disp = _coding_pair(s.get("bodySite"))
                    if bs_disp:
                        body_sites.append(bs_disp)
                    mod = s.get("modality") or {}
                    if isinstance(mod, Mapping) and mod.get("display"):
                        modalities.append(str(mod.get("display")))
            desc = proc_disp or ", ".join(dict.fromkeys(modalities)) or "Imaging Study"
            site_str = f" ({', '.join(dict.fromkeys(body_sites))})" if body_sites else ""
            im_sens, im_sens_src = _classify_res_sensitivity(res, "", desc)
            items.append(
                ClinicalItem(
                    resource_type="ImagingStudy",
                    resource_uuid=rid,
                    date=im_date,
                    clinical_date=im_date,
                    date_kind="clinical_event" if im_date else "unknown",
                    section="imaging",
                    code=desc,
                    display=desc,
                    text=f"ImagingStudy: {desc}{site_str}",
                    mandatory=False,
                    status=str(res.get("status") or "available").lower(),
                    source_org=resolve_source_org(res),
                    sensitivity=im_sens,
                    metadata={"sensitivity_source": im_sens_src} if im_sens_src else {},
                )
            )
        elif rtype == "Device":
            code, disp = _coding_pair(res.get("type"))
            if not disp:
                names = res.get("deviceName") or []
                if names and isinstance(names[0], Mapping):
                    disp = str(names[0].get("name") or "")
            is_room_equip = is_clinic_room_equipment(disp)
            if is_room_equip and filter_room_equipment_devices:
                excluded_counts["Device(clinic_equipment)"] = (
                    excluded_counts.get("Device(clinic_equipment)", 0) + 1
                )
                continue
            elif is_room_equip:
                excluded_counts["Device(clinic_equipment_retained)"] = (
                    excluded_counts.get("Device(clinic_equipment_retained)", 0) + 1
                )
            d_date = _iso_date(res.get("manufactureDate"))
            status = str(res.get("status") or "active").lower()
            dev_sens, dev_sens_src = _classify_res_sensitivity(res, code, disp)
            dev_meta: Dict[str, Any] = {"is_room_equipment": is_room_equip}
            if dev_sens_src:
                dev_meta["sensitivity_source"] = dev_sens_src
            items.append(
                ClinicalItem(
                    resource_type="Device",
                    resource_uuid=rid,
                    date=d_date,
                    clinical_date=None,
                    date_kind="administrative" if d_date else "unknown",
                    section="device",
                    code=code or disp,
                    display=disp or code or "Medical Device",
                    text=f"Device: {disp or code} (status: {status})",
                    mandatory=(status == "active" and not is_room_equip),
                    status=status,
                    source_org=resolve_source_org(res),
                    sensitivity=dev_sens,
                    is_routine=is_room_equip,
                    metadata=dev_meta,
                )
            )
        elif rtype == "Encounter":
            enc_period = res.get("period") or {}
            e_start = _iso_date(enc_period.get("start"))
            e_end = _iso_date(enc_period.get("end"))
            enc_date = e_end or e_start
            enc_class = str((res.get("class") or {}).get("code") or "AMB").upper()
            types = res.get("type") or []
            _, t_disp = _coding_pair(types[0] if types else {})
            reasons: List[str] = []
            for rc in res.get("reasonCode") or []:
                _, r_disp = _coding_pair(rc)
                if r_disp:
                    reasons.append(r_disp)
            for rr in res.get("reasonReference") or []:
                if isinstance(rr, Mapping):
                    c_id = _strip_urn(rr.get("reference"))
                    r_disp = str(rr.get("display") or conds_by_id.get(c_id) or "").strip()
                    if r_disp:
                        reasons.append(r_disp)
            org = resolve_source_org(res)
            enc_sens, enc_sens_src = _classify_res_sensitivity(res, enc_class, t_disp or "")
            is_high_acuity = enc_class in {"IMP", "INPATIENT", "EMER", "EMERGENCY"} or any(
                w in (t_disp or "").lower() for w in ("inpatient", "emergency", "admission", "intensive")
            )
            quals = [f"class: {enc_class}"]
            if e_end and e_end != e_start:
                quals.append(f"discharged: {e_end}")
            if reasons:
                quals.append(f"reason: {', '.join(dict.fromkeys(reasons))}")
            if org:
                quals.append(f"facility: {org}")
            enc_meta: Dict[str, Any] = {"start": e_start, "end": e_end, "class": enc_class}
            if enc_sens_src:
                enc_meta["sensitivity_source"] = enc_sens_src
            items.append(
                ClinicalItem(
                    resource_type="Encounter",
                    resource_uuid=rid,
                    date=enc_date,
                    clinical_date=enc_date,
                    date_kind="clinical_event" if enc_date else "unknown",
                    section="encounter_acuity" if is_high_acuity else "encounter_routine",
                    code=enc_class,
                    display=t_disp or enc_class,
                    text=f"Encounter: {t_disp or enc_class} ({', '.join(quals)})",
                    mandatory=False,
                    status=str(res.get("status") or "finished").lower(),
                    end_date=e_end,
                    source_org=org,
                    sensitivity=enc_sens,
                    is_routine=not is_high_acuity,
                    metadata=enc_meta,
                )
            )

    return ExtractedChart(
        patient_id=patient_id,
        patient_name=patient_name,
        birth_date=birth_date,
        reference_date=reference_date,
        items=items,
        numeric_points=numeric_points,
        excluded_resource_counts=excluded_counts,
        deduplicated_note_count=dedup_note_count,
        total_bundle_entries=len(resources),
    )
