"""5-Tier Progressive FHIR Chart Condenser and Token-Budget Controller.

Condenses any arbitrary-size FHIR R4 Bundle (`ExtractedChart`) into a target token budget
(`target_tokens`) while preserving:
- 100% of safety-critical IPS core state (Allergies, Active Conditions, Active Medications,
  Anomalous/Duplicate Immunizations, Active Patient Devices)
- 100% `[N] -> (resourceType, resourceUuid, memberUuids)` provenance for citation grounding
- Temporal fidelity: folded spans always carry the latest clinical date in `(YYYY-MM-DD)` and
  state `first recorded YYYY-MM-DD through YYYY-MM-DD` (plus `(initial <val> on <date> -> ...)`
  for trajectories) so `temporal.py` validates both endpoints and trend directions cleanly
- Explicit `[Completeness Manifest]` so models never mistake budget omission for a clinical negative
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .chart_serializer import render_chart
from .extractors import (
    ClinicalItem,
    ExtractedChart,
    NumericPoint,
    extract_bundle,
)
from .normalizers import classify_sensitivity

# Priority analytes surfaced first in Tier 3 / Tier 4 LabTrajectory rollups.
_PRIORITY_ANALYTE_PATTERNS = (
    r"glomerular\s+filtration|egfr",
    r"creatinine\s*\[mass/volume\]|^creatinine$",
    r"hemoglobin\s+a1c|hba1c",
    r"glucose\s*\(blood/serum\)|^glucose$",
    r"potassium",
    r"sodium",
    r"urea\s+nitrogen|bun",
    r"microalbumin|albumin/creatinine",
    r"systolic\s+blood\s+pressure|blood\s+pressure",
    r"diastolic\s+blood\s+pressure",
    r"low\s+density\s+lipoprotein|ldl",
    r"high\s+density\s+lipoprotein|hdl",
    r"cholesterol",
    r"triglycerides",
    r"hemoglobin\s*\[mass/volume\]|^hemoglobin$",
    r"leukocytes|white\s+blood",
    r"platelets",
    r"body\s+mass\s+index|bmi",
    r"body\s+weight",
    r"fev1/fvc",
    r"phq-?[29]|depression",
    r"gad-?7|anxiety",
    r"audit-?c|alcohol",
)


@dataclass(frozen=True)
class CondensedChartResult:
    tier: int
    tier_name: str
    target_tokens: int
    actual_tokens: int
    chart_text: str
    records: List[Dict[str, Any]]
    mappings: List[Dict[str, Any]]
    completeness_manifest: Dict[str, Any]
    redacted_for_privacy: List[Dict[str, Any]]
    compression_stats: Dict[str, Any]


def estimate_tokens_local(text: str) -> int:
    """Deterministic pure-stdlib fallback token estimator calibrated to Gemma 4 / GPT-4o BPE.

    Gemma 4's SentencePiece tokenizer splits every digit individually and keeps common short words
    and 6-character clinical stems whole. ``wahs-engine`` uses the exact offline tokenizer; this
    fallback models per-digit splitting and 6-char subwords in pure stdlib (~0.96x-1.02x Gemma 4).
    """
    if not text:
        return 0
    pieces = re.findall(r"[A-Za-z]{1,6}|\d|\S", text)
    by_pieces = len(pieces)
    by_chars = int(math.ceil(len(text.encode("utf-8")) / 3.7))
    return max(by_pieces, by_chars)


def _date_sort_key(d: Optional[str]) -> str:
    return d if d and re.match(r"^\d{4}-\d{2}-\d{2}$", d) else "0000-00-00"


def _int_date(d: Optional[str]) -> int:
    if not d or not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return 0
    return int(d.replace("-", ""))


def _merge_uuids(items: Sequence[ClinicalItem]) -> Tuple[str, ...]:
    out: List[str] = []
    for it in items:
        if it.resource_uuid:
            out.append(it.resource_uuid)
        for m in it.member_uuids:
            if m:
                out.append(m)
    return tuple(dict.fromkeys(out))


def _span_phrase(dates: Sequence[Optional[str]]) -> Tuple[Optional[str], str]:
    valid = sorted({d for d in dates if d and re.match(r"^\d{4}-\d{2}-\d{2}$", d)})
    if not valid:
        return None, ""
    if len(valid) == 1:
        return valid[-1], f"recorded {valid[0]}"
    return valid[-1], f"first recorded {valid[0]} through {valid[-1]}"


def _disp_with_date(display: str, date_val: Optional[str], extra: str = "") -> str:
    """Format ``display (qualifiers)`` safely without ever emitting ``'(None)'``."""
    parts: List[str] = []
    if date_val:
        parts.append(date_val)
    if extra:
        parts.append(extra)
    return f"{display} ({', '.join(parts)})" if parts else display


def _fold_devices(devices: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Fold repeated identical patient devices (e.g. annual CPAP appliance replacements) by (code/display, status)."""
    by_code: Dict[Tuple[str, str], List[ClinicalItem]] = {}
    for d in devices:
        by_code.setdefault((d.code or d.display, d.status), []).append(d)

    folded: List[ClinicalItem] = []
    for (_code, _status), group in by_code.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) == 1:
            folded.append(latest)
            continue
        latest_date, span_str = _span_phrase([x.date for x in ordered])
        span_part = f", {span_str}" if span_str else ""
        eff_date = latest_date or latest.date
        folded.append(
            ClinicalItem(
                resource_type="Device",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=None,
                date_kind="administrative" if eff_date else latest.date_kind,
                section="device",
                code=latest.code,
                display=latest.display,
                text=f"Device: {latest.display} (status: {latest.status}, {len(ordered)} records{span_part})",
                mandatory=any(x.mandatory for x in ordered),
                status=latest.status,
                source_org=latest.source_org,
                sensitivity=latest.sensitivity,
                member_uuids=_merge_uuids(ordered),
            )
        )
    return sorted(folded, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_social_history(social_items: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Fold repeated social-history observations (e.g. 62 identical Tobacco smoking status assessments)."""
    by_key: Dict[Tuple[str, str], List[ClinicalItem]] = {}
    for s in social_items:
        val = str(s.metadata.get("value") or s.text)
        by_key.setdefault((s.code or s.display, val), []).append(s)

    folded: List[ClinicalItem] = []
    for (_code, _val), group in by_key.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) == 1:
            folded.append(latest)
            continue
        latest_date, span_str = _span_phrase([x.date for x in ordered])
        span_part = f", {span_str}" if span_str else ""
        eff_date = latest_date or latest.date
        folded.append(
            ClinicalItem(
                resource_type="Observation",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section="social_history",
                code=latest.code,
                display=latest.display,
                text=f"{latest.text} ({len(ordered)} assessments{span_part})",
                mandatory=latest.mandatory,
                status=latest.status,
                sensitivity=latest.sensitivity,
                member_uuids=_merge_uuids(ordered),
                metadata=latest.metadata,
            )
        )
    return sorted(folded, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_encounters(encounters: Sequence[ClinicalItem], *, keep_latest_individual: int = 3) -> List[ClinicalItem]:
    """Keep the `keep_latest_individual` most recent encounters unpacked and fold older repeated encounters.

    Unfulfilled future/pre-visit Appointments remain unpacked so upcoming visit details stay intact.
    """
    unfulfilled_appts = [
        e for e in encounters if e.resource_type == "Appointment" and (e.status or "").lower() != "fulfilled"
    ]
    completed_encounters = [
        e for e in encounters if not (e.resource_type == "Appointment" and (e.status or "").lower() != "fulfilled")
    ]
    ordered_desc = sorted(completed_encounters, key=lambda x: _date_sort_key(x.date), reverse=True)
    kept = sorted(unfulfilled_appts + list(ordered_desc[:keep_latest_individual]), key=lambda x: _date_sort_key(x.date), reverse=True)
    older = ordered_desc[keep_latest_individual:]
    if not older:
        return kept

    by_type: Dict[str, List[ClinicalItem]] = {}
    for e in older:
        by_type.setdefault(e.display or e.code, []).append(e)

    folded_older: List[ClinicalItem] = []
    for _disp, group in by_type.items():
        chron = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = chron[-1]
        if len(chron) == 1:
            folded_older.append(latest)
            continue
        latest_date, span_str = _span_phrase([x.date for x in chron])
        orgs = sorted({x.source_org for x in chron if x.source_org})
        org_str = f", facility: {orgs[0]}" if orgs else ""
        span_prefix = f"{span_str}, " if span_str else ""
        eff_date = latest_date or latest.date
        folded_older.append(
            ClinicalItem(
                resource_type="Encounter",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section=latest.section,
                code=latest.code,
                display=latest.display,
                text=(
                    f"Encounter History: {latest.display} — {len(chron)} visits "
                    f"({span_prefix}class: {latest.code}{org_str})"
                ),
                mandatory=False,
                status=latest.status,
                source_org=latest.source_org,
                sensitivity=latest.sensitivity,
                is_routine=latest.is_routine,
                member_uuids=_merge_uuids(chron),
            )
        )
    return kept + sorted(folded_older, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_lab_panels_by_type(
    panels: Sequence[ClinicalItem], *, keep_latest_per_type: int = 2
) -> List[ClinicalItem]:
    """Keep the `keep_latest_per_type` most recent panels of each panel code in full, and fold older
    panels of the same code into a single historical summary line carrying all constituent UUIDs."""
    by_code: Dict[str, List[ClinicalItem]] = {}
    for p in panels:
        by_code.setdefault(p.code or p.display, []).append(p)

    out: List[ClinicalItem] = []
    for _code, group in by_code.items():
        ordered_desc = sorted(group, key=lambda x: _date_sort_key(x.date), reverse=True)
        out.extend(ordered_desc[:keep_latest_per_type])
        older = ordered_desc[keep_latest_per_type:]
        if not older:
            continue
        chron = list(reversed(older))
        latest_older = chron[-1]
        l_date, span_str = _span_phrase([x.date for x in chron])
        span_part = f" ({span_str})" if span_str else ""
        eff_date = l_date or latest_older.date
        out.append(
            ClinicalItem(
                resource_type=latest_older.resource_type,
                resource_uuid=latest_older.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest_older.date_kind,
                section=latest_older.section,
                code=latest_older.code,
                display=latest_older.display,
                text=(
                    f"Diagnostic Panel History — {latest_older.display}: "
                    f"{len(chron)} prior panels{span_part}"
                ),
                mandatory=False,
                status=latest_older.status,
                source_org=latest_older.source_org,
                sensitivity=latest_older.sensitivity,
                is_routine=latest_older.is_routine,
                member_uuids=_merge_uuids(chron),
            )
        )
    return sorted(out, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_stopped_medications(meds: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Fold historical/stopped prescriptions by normalized ingredient while keeping setting distinct."""
    by_key: Dict[Tuple[str, str], List[ClinicalItem]] = {}
    for m in meds:
        setting = str(m.metadata.get("setting") or "")
        key = (m.ingredient_key or m.code or m.display.lower(), setting)
        by_key.setdefault(key, []).append(m)

    folded: List[ClinicalItem] = []
    for (_ing, setting), group in by_key.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) == 1:
            folded.append(latest)
            continue
        latest_date, span_str = _span_phrase([x.date for x in ordered])
        reasons: List[str] = []
        for x in ordered:
            for r in x.metadata.get("reasons") or []:
                if r:
                    reasons.append(r)
        reason_str = f", reason: {', '.join(dict.fromkeys(reasons))}" if reasons else ""
        span_prefix = f"{span_str}, " if span_str else ""
        eff_date = latest_date or latest.date
        folded.append(
            ClinicalItem(
                resource_type=latest.resource_type,
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section="med_historical",
                code=latest.code,
                display=latest.display,
                text=(
                    f"Medication History: {latest.display} — {len(ordered)} records "
                    f"({span_prefix}status: {latest.status}, type: {setting}{reason_str})"
                ),
                mandatory=False,
                status=latest.status,
                source_org=latest.source_org,
                sensitivity=latest.sensitivity,
                ingredient_key=latest.ingredient_key,
                member_uuids=_merge_uuids(ordered),
                metadata={"count": len(ordered), "reasons": list(dict.fromkeys(reasons)), "setting": setting},
            )
        )
    return sorted(folded, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_resolved_conditions(conds: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Run-length fold resolved conditions by code/display cleanly without leaking sensitivity tags."""
    by_code: Dict[str, List[ClinicalItem]] = {}
    for c in conds:
        by_code.setdefault(c.code or c.display, []).append(c)

    folded: List[ClinicalItem] = []
    for _code, group in by_code.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) == 1:
            folded.append(latest)
            continue
        all_dates: List[Optional[str]] = []
        for x in ordered:
            all_dates.append(x.metadata.get("onset") or x.date)
            if x.metadata.get("abatement"):
                all_dates.append(x.metadata.get("abatement"))
        latest_date, span_str = _span_phrase(all_dates)
        span_prefix = f"{span_str}, " if span_str else ""
        eff_date = latest_date or latest.date
        folded.append(
            ClinicalItem(
                resource_type="Condition",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section="condition_resolved",
                code=latest.code,
                display=latest.display,
                text=(
                    f"Condition History: {latest.display} — {len(ordered)} episodes "
                    f"({span_prefix}status: {latest.status}, SNOMED: {latest.code})"
                ),
                mandatory=False,
                status=latest.status,
                source_org=latest.source_org,
                sensitivity=latest.sensitivity,
                is_routine=latest.is_routine,
                member_uuids=_merge_uuids(ordered),
                metadata={"count": len(ordered)},
            )
        )
    return sorted(folded, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_immunizations(imms: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Keep anomalous or <=3-dose vaccine series unpacked individually; fold longer (>3-dose) routine series."""
    anomalous = [x for x in imms if x.conflict_note]
    routine = [x for x in imms if not x.conflict_note]

    by_group: Dict[str, List[ClinicalItem]] = {}
    for x in routine:
        by_group.setdefault(x.ingredient_key or x.display, []).append(x)

    folded_routine: List[ClinicalItem] = []
    for _grp, group in by_group.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) <= 3:
            folded_routine.extend(ordered)
            continue
        latest_date, span_str = _span_phrase([x.date for x in ordered])
        orgs = sorted({x.source_org for x in ordered if x.source_org})
        org_str = f", sources: {'; '.join(orgs[:2])}" if orgs else ""
        span_prefix = f"{span_str}, " if span_str else ""
        eff_date = latest_date or latest.date
        folded_routine.append(
            ClinicalItem(
                resource_type="Immunization",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section="immunization",
                code=latest.code,
                display=latest.display,
                text=(
                    f"Immunization Series: {latest.display} — {len(ordered)} doses "
                    f"({span_prefix}status: completed, CVX: {latest.code}{org_str})"
                ),
                mandatory=False,
                status=latest.status,
                source_org=latest.source_org,
                sensitivity=latest.sensitivity,
                ingredient_key=latest.ingredient_key,
                member_uuids=_merge_uuids(ordered),
            )
        )
    combined = anomalous + folded_routine
    return sorted(combined, key=lambda x: _date_sort_key(x.date), reverse=True)


def _fold_procedures(procs: Sequence[ClinicalItem], *, section: str) -> List[ClinicalItem]:
    """Fold repeated procedures by code/display."""
    by_code: Dict[str, List[ClinicalItem]] = {}
    for p in procs:
        by_code.setdefault(p.code or p.display, []).append(p)

    folded: List[ClinicalItem] = []
    for _code, group in by_code.items():
        ordered = sorted(group, key=lambda x: _date_sort_key(x.date))
        latest = ordered[-1]
        if len(ordered) == 1:
            folded.append(latest)
            continue
        latest_date, span_str = _span_phrase([x.date for x in ordered])
        span_part = f" ({span_str})" if span_str else ""
        eff_date = latest_date or latest.date
        folded.append(
            ClinicalItem(
                resource_type="Procedure",
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date,
                date_kind="clinical_event" if eff_date else latest.date_kind,
                section=section,
                code=latest.code,
                display=latest.display,
                text=f"Procedure History: {latest.display} — {len(ordered)} times{span_part}",
                mandatory=False,
                status=latest.status,
                sensitivity=latest.sensitivity,
                is_routine=(section == "procedure_routine"),
                member_uuids=_merge_uuids(ordered),
            )
        )
    return sorted(folded, key=lambda x: _date_sort_key(x.date), reverse=True)


def _group_vitals_by_date(vitals: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Group standalone Vital Sign observations on the same date into a single Vital Signs Panel line.

    Preserves distinct intraday readings for the same vital sign when values or timestamps differ.
    """
    by_date: Dict[str, List[ClinicalItem]] = {}
    for v in vitals:
        by_date.setdefault(v.date or "unknown", []).append(v)

    grouped: List[ClinicalItem] = []
    for _d, items_on_date in sorted(by_date.items(), reverse=True):
        if len(items_on_date) == 1:
            grouped.append(items_on_date[0])
            continue
        values_by_label: Dict[str, List[str]] = {}
        for it in items_on_date:
            val = str(it.metadata.get("value") or "").strip()
            ts = str(it.metadata.get("timestamp") or "").strip()
            label = it.display
            if not val:
                continue
            formatted_val = f"{val} @{ts[11:16]}" if (ts and "T" in ts) else val
            bucket = values_by_label.setdefault(label, [])
            if formatted_val not in bucket:
                bucket.append(formatted_val)
        parts = [
            f"{label}: {' -> '.join(vals)}"
            for label, vals in values_by_label.items()
        ]
        lead = items_on_date[0]
        grouped.append(
            ClinicalItem(
                resource_type="Observation",
                resource_uuid=lead.resource_uuid,
                date=lead.date,
                clinical_date=lead.clinical_date,
                date_kind="clinical_event" if lead.date else lead.date_kind,
                section="vital_sign",
                code="vital-signs-panel",
                display="Vital Signs Panel",
                text=f"Vital Signs Panel: {'; '.join(parts)}",
                mandatory=False,
                status="final",
                sensitivity=lead.sensitivity,
                is_routine=True,
                member_uuids=_merge_uuids(items_on_date),
            )
        )
    return grouped


def _canonical_analyte_name(raw: str) -> str:
    """Normalize verbose LOINC display names into clean, clinical analyte names."""
    s = re.sub(r"\s*\([^)]*\)\s*$", "", (raw or "")).strip() or raw
    low = s.lower()
    if "glomerular filtration rate" in low:
        return "Glomerular filtration rate (eGFR)"
    if "hemoglobin a1c" in low:
        return "Hemoglobin A1c (HbA1c)"
    if "cholesterol in ldl" in low:
        return "LDL Cholesterol"
    if "cholesterol in hdl" in low:
        return "HDL Cholesterol"
    if low == "cholesterol [mass/volume] in serum or plasma":
        return "Total Cholesterol"
    if "triglyceride" in low and ("serum" in low or "blood" in low):
        return "Triglycerides"
    if "microalbumin/creatinine" in low or "albumin/creatinine" in low:
        return "Urine Microalbumin/Creatinine Ratio"
    # Merge identical blood vs serum/plasma electrolyte/glucose/BUN series from BMP vs BMP-2000
    for chem in ("glucose", "potassium", "sodium", "urea nitrogen", "calcium", "chloride", "carbon dioxide"):
        if low.startswith(chem) and ("in blood" in low or "in serum or plasma" in low) and "urine" not in low:
            return f"{chem.title()} (Blood/Serum)"
    return s


def _build_lab_trajectories(
    numeric_points: Sequence[NumericPoint],
    *,
    max_analytes: Optional[int] = None,
) -> List[ClinicalItem]:
    """Collapse numeric Observation series into 1 longitudinal LabTrajectory record per (analyte, unit).

    Includes `(initial <val> on <date> -> latest <val> on <date>; ...)` so `temporal.py`
    extracts both endpoints (`first` and `latest`), intermediate min/max extrema, computes
    `trend_supported = True`, and validates trend claims deterministically.
    """
    by_concept_unit: Dict[Tuple[str, str], List[NumericPoint]] = {}
    units_per_concept: Dict[str, Set[str]] = {}
    for pt in numeric_points:
        clean_name = _canonical_analyte_name(pt.concept)
        norm_unit = (pt.unit or "").strip()
        by_concept_unit.setdefault((clean_name, norm_unit), []).append(pt)
        units_per_concept.setdefault(clean_name, set()).add(norm_unit)

    def analyte_priority(key: Tuple[str, str]) -> Tuple[int, int, str, str]:
        name, unit_key = key
        lower = name.lower()
        for idx, pat in enumerate(_PRIORITY_ANALYTE_PATTERNS):
            if re.search(pat, lower):
                return (0, idx, lower, unit_key.lower())
        return (1, 999, lower, unit_key.lower())

    trajectories: List[ClinicalItem] = []
    sorted_keys = sorted(by_concept_unit.keys(), key=analyte_priority)
    if max_analytes is not None:
        sorted_keys = sorted_keys[:max_analytes]

    def fmt(v: float) -> str:
        if v.is_integer():
            return str(int(v))
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s not in {"", "-0"} else "0"

    for clean_name, norm_unit in sorted_keys:
        pts = sorted(by_concept_unit[(clean_name, norm_unit)], key=lambda p: p.date)
        dedup_pts: List[NumericPoint] = []
        seen_date_vals: Set[Tuple[str, float]] = set()
        for p in pts:
            dv_key = (p.date, p.value)
            if dv_key not in seen_date_vals:
                seen_date_vals.add(dv_key)
                dedup_pts.append(p)
        if not dedup_pts:
            continue
        latest = dedup_pts[-1]
        first = dedup_pts[0]
        latest_day = (latest.date or "")[:10]
        first_day = (first.date or "")[:10]
        p_min = min(dedup_pts, key=lambda p: p.value)
        p_max = max(dedup_pts, key=lambda p: p.value)
        min_day = (p_min.date or "")[:10]
        max_day = (p_max.date or "")[:10]
        unit = norm_unit or latest.unit

        concept = (
            f"{clean_name} [{unit or 'unitless'}]"
            if len(units_per_concept.get(clean_name, set())) > 1
            else clean_name
        )

        unit_sp = f" {unit}" if unit else ""
        interp_tag = f" [interp: {latest.interpretation}]" if getattr(latest, "interpretation", None) else ""
        latest_label = f"recorded {latest_day}" if latest_day else "date unknown"
        if len(dedup_pts) == 1:
            text = f"Finding — {concept}: {fmt(latest.value)}{unit_sp}{interp_tag} ({latest_label}, n=1)"
        else:
            if latest.value > first.value:
                direction = "up"
            elif latest.value < first.value:
                direction = "down"
            else:
                direction = "flat"
            f_on = f" on {first_day}" if first_day else " (date unknown)"
            l_on = f" on {latest_day}" if latest_day else " (date unknown)"
            min_on = f" on {min_day}" if min_day else ""
            max_on = f" on {max_day}" if max_day else ""
            span_desc = (
                f"first recorded {first_day} through {latest_day}, "
                if (first_day and latest_day)
                else ""
            )
            text = (
                f"Finding — {concept}: {fmt(latest.value)}{unit_sp}{interp_tag} "
                f"(initial {fmt(first.value)}{f_on} -> latest {fmt(latest.value)}{l_on}; "
                f"range {fmt(p_min.value)}{min_on} to {fmt(p_max.value)}{max_on}{unit_sp} "
                f"over n={len(dedup_pts)} measurements, {span_desc}trend: {direction})"
            )
        member_uuids = tuple(dict.fromkeys(p.resource_uuid for p in pts if p.resource_uuid))
        p_group, p_idx, _, _ = analyte_priority((clean_name, norm_unit))
        trajectories.append(
            ClinicalItem(
                resource_type="Observation",
                resource_uuid=latest.resource_uuid,
                date=latest_day or None,
                clinical_date=latest_day or None,
                date_kind="clinical_event" if latest_day else "unknown",
                section="lab_trajectory",
                code=latest.code,
                display=concept,
                text=text,
                mandatory=False,
                status="final",
                member_uuids=member_uuids,
                metadata={"n_points": len(dedup_pts), "analyte_rank": p_group * 100 + p_idx},
            )
        )
    return trajectories


def _build_completeness_manifest(
    extracted: ExtractedChart,
    tier: int,
    tier_name: str,
    included_items: Sequence[ClinicalItem],
    redacted_count: int,
    *,
    excerpted_notes_count: int = 0,
) -> Tuple[ClinicalItem, Dict[str, Any]]:
    """Build the mandatory `[Completeness Manifest]` item and structured manifest dictionary."""
    ex_str = ", ".join(
        f"{k}={v}" for k, v in sorted(extracted.excluded_resource_counts.items())
    )
    redacted_str = f"; privacy_redacted={redacted_count}" if redacted_count else ""
    dedup_str = (
        f"; base64_duplicate_notes_merged={extracted.deduplicated_note_count}"
        if extracted.deduplicated_note_count
        else ""
    )

    covered_uuids: Set[str] = set()
    for it in included_items:
        if it.resource_uuid:
            covered_uuids.add(it.resource_uuid)
        if it.section != "lab_trajectory":
            for m in it.member_uuids:
                if m:
                    covered_uuids.add(m)

    omitted_by_section: Dict[str, int] = {}
    for src_it in extracted.items:
        if src_it.section == "patient":
            continue
        item_uuids = {src_it.resource_uuid} | set(src_it.member_uuids)
        if not (item_uuids & covered_uuids):
            omitted_by_section[src_it.section] = omitted_by_section.get(src_it.section, 0) + 1
    omitted_clinical_count = max(0, sum(omitted_by_section.values()) - redacted_count)
    omitted_str = (
        f"; omitted_clinical_items={omitted_clinical_count}"
        if omitted_clinical_count > 0
        else ""
    )
    excerpted_str = (
        f"; excerpted_notes={excerpted_notes_count}"
        if excerpted_notes_count > 0
        else ""
    )

    if tier == 0 and omitted_clinical_count == 0:
        coverage = "ALL clinical sections COMPLETE (lossless 1-line projection)"
    elif tier == 0:
        coverage = (
            "IPS Safety Core and structured clinical ledger COMPLETE; "
            f"{omitted_clinical_count} clinical item(s) omitted by mode/budget filter"
        )
    elif tier == 1:
        coverage = (
            "IPS Safety Core (Allergies, Active Conditions, Active Medications, Anomalous Immunizations, "
            "Devices) COMPLETE; historical renewals, resolved conditions, repeated panels, and routine "
            "screenings RUN-LENGTH FOLDED with full date spans and UUIDs"
        )
    elif tier == 2:
        coverage = (
            "IPS Safety Core COMPLETE; Diagnostics/Vitals/Notes WINDOWED to recent horizon + "
            "longitudinal analyte trajectories + folded historical spans"
        )
    elif tier == 3:
        coverage = (
            "IPS Safety Core COMPLETE; Labs & Vitals condensed to longitudinal per-analyte trajectories "
            "(initial/latest/min/max/trend); routine notes & outpatient visits summarized"
        )
    else:
        coverage = (
            "IPS Micro-Capsule: Safety Core (Allergies, Active Conditions, Active Medications, Key Alerts) "
            "COMPLETE; historical labs/visits condensed to key trajectories and surgical/inpatient summary"
        )

    manifest_text = (
        f"Completeness Manifest: Tier {tier} ({tier_name}) | {coverage} | "
        f"bundle_entries={extracted.total_bundle_entries}, rendered_lines={len(included_items) + 1}"
        f"{dedup_str}{redacted_str}{omitted_str}{excerpted_str}"
        + (f" | non_clinical_excluded: {ex_str}" if ex_str else "")
    )
    if tier == 4 and "Ultra-Compact" in tier_name:
        manifest_text = (
            f"Completeness Manifest: Tier 4 ({tier_name}) | IPS Safety Core COMPLETE | "
            f"entries={extracted.total_bundle_entries}, lines={len(included_items) + 1}{redacted_str}{omitted_str}{excerpted_str}"
        )

    manifest_item = ClinicalItem(
        resource_type="Bundle",
        resource_uuid=f"manifest-{extracted.patient_id or 'chart'}-t{tier}",
        date=extracted.reference_date,
        clinical_date=None,
        date_kind="administrative",
        section="manifest",
        code="completeness-manifest",
        display="Completeness Manifest",
        text=manifest_text,
        mandatory=True,
        status="current",
    )
    manifest_dict = {
        "tier": tier,
        "tier_name": tier_name,
        "coverage": coverage,
        "total_bundle_entries": extracted.total_bundle_entries,
        "rendered_records": len(included_items) + 1,
        "deduplicated_note_count": extracted.deduplicated_note_count,
        "excluded_non_clinical": dict(extracted.excluded_resource_counts),
        "redacted_for_privacy_count": redacted_count,
        "omitted_clinical_items_count": omitted_clinical_count,
        "excerpted_notes_count": excerpted_notes_count,
        "omitted_by_section": omitted_by_section,
        "omitted_section_counts": dict(omitted_by_section),
    }
    return manifest_item, manifest_dict


_HOLD_CLAUSE_RE = re.compile(
    r"\b(?:hold|stop|discontinue|held|stopped|discontinued|do\s+not\s+take)\s+[a-z][a-z0-9-]{2,25}\b",
    re.IGNORECASE,
)


_VITAL_HEADER_RE = re.compile(
    r"\b(?:BP|Blood\s+Pressure|HR|Pulse|SpO2|Temp(?:erature)?|Wt|Weight)\s*[:=]?\s*\d{2,3}(?:/\d{2,3})?(?:\.\d+)?[^.;|\n]{0,30}",
    re.IGNORECASE,
)


def _extract_note_conflict_clauses(
    text: str,
    max_chars: int = 160,
    *,
    include_vitals: bool = False,
) -> str:
    clauses = [
        c.strip()
        for c in re.split(r"[.;|\n]+", text or "")
        if c.strip() and _HOLD_CLAUSE_RE.search(c)
    ]
    if not clauses:
        return (text or "").strip()[:max_chars]
    joined = "; ".join(clauses)
    if include_vitals:
        m_vit = _VITAL_HEADER_RE.search(text or "")
        if m_vit and m_vit.group(0).strip().lower() not in joined.lower():
            joined = f"{m_vit.group(0).strip()}; {joined}"
    return joined[:max_chars] + ("..." if len(joined) > max_chars else "")


def _excerpt_notes(notes: Sequence[ClinicalItem], max_chars: int = 400) -> List[ClinicalItem]:
    """Excerpt clinical notes to `max_chars` for condensed Stages while keeping raw text structural."""
    out: List[ClinicalItem] = []
    for n in notes:
        full_norm = str(n.metadata.get("full_text") or n.text or "")
        title = str(n.metadata.get("title") or n.display or "Clinical Note")
        if full_norm and len(full_norm) > max_chars:
            has_hold_clause = bool(_HOLD_CLAUSE_RE.search(full_norm))
            if n.conflict_note or has_hold_clause:
                c_cap = min(260, max(120, max_chars - 90))
                c_clause = _extract_note_conflict_clauses(full_norm, max_chars=c_cap, include_vitals=True)
                head_budget = max(60, max_chars - len(c_clause) - 6)
                head = full_norm[:head_budget].strip()
                if c_clause and c_clause.lower() not in head.lower():
                    excerpt = f"{head} ... {c_clause}"
                else:
                    excerpt = full_norm[:max_chars] + "..."
            else:
                excerpt = full_norm[:max_chars] + "..."
            txt = f"Note ({title}): {excerpt}"
        else:
            txt = n.text
        out.append(
            ClinicalItem(
                resource_type=n.resource_type,
                resource_uuid=n.resource_uuid,
                date=n.date,
                clinical_date=n.clinical_date,
                date_kind=n.date_kind,
                section=n.section,
                code=n.code,
                display=n.display,
                text=txt,
                mandatory=n.mandatory,
                status=n.status,
                end_date=n.end_date,
                source_org=n.source_org,
                sensitivity=n.sensitivity,
                conflict_note=n.conflict_note,
                ingredient_key=n.ingredient_key,
                is_routine=n.is_routine,
                member_uuids=n.member_uuids,
                metadata=n.metadata,
            )
        )
    return out


def items_to_querystore_records(items: Sequence[ClinicalItem]) -> List[Dict[str, Any]]:
    """Convert `ClinicalItem` objects into the exact dict schema expected by `chart_serializer.render_chart`."""
    records: List[Dict[str, Any]] = []
    for it in items:
        meta = dict(it.metadata)
        if it.mandatory:
            meta["mandatory_context"] = True
        if it.member_uuids:
            meta["memberUuids"] = list(it.member_uuids)
        if it.sensitivity:
            meta["sensitivity"] = it.sensitivity
        if it.source_org:
            meta["source_org"] = it.source_org
        if it.conflict_note:
            meta["conflict_note"] = it.conflict_note
        records.append(
            {
                "resourceType": it.resource_type,
                "resourceUuid": it.resource_uuid,
                "date": it.date,
                "clinicalDate": it.clinical_date,
                "dateKind": it.date_kind,
                "text": it.text,
                "metadata": meta,
            }
        )
    return records


def _filter_source_items_for_posture(
    extracted: ExtractedChart,
    *,
    mode: str,
    sharing_posture: str,
) -> Tuple[List[ClinicalItem], List[Dict[str, Any]], List[NumericPoint]]:
    """Filter extracted items and numeric points for sharing posture and mode."""
    redacted: List[Dict[str, Any]] = []
    redacted_uuids: Set[str] = set()
    source_items: List[ClinicalItem] = []
    for it in extracted.items:
        eff_sens = it.sensitivity or (
            classify_sensitivity(it.code, f"{it.display} {it.text}")
            if sharing_posture == "external_share"
            else None
        )
        if sharing_posture == "external_share" and eff_sens == "SDOH/PRIVACY-SENSITIVE":
            redacted.append(
                {
                    "resourceType": it.resource_type,
                    "resourceUuid": it.resource_uuid,
                    "date": it.date,
                    "display": "Redacted privacy-sensitive SDOH record",
                    "sensitivity": eff_sens,
                }
            )
            if it.resource_uuid:
                redacted_uuids.add(it.resource_uuid)
            for m_uuid in it.member_uuids:
                if m_uuid:
                    redacted_uuids.add(m_uuid)
            continue
        if mode == "reconcile" and it.section == "clinical_note":
            continue
        source_items.append(it)

    allowed_numeric_points = [
        pt
        for pt in extracted.numeric_points
        if pt.resource_uuid not in redacted_uuids
        and not (
            sharing_posture == "external_share"
            and classify_sensitivity(pt.code, pt.concept, [pt.category]) == "SDOH/PRIVACY-SENSITIVE"
        )
    ]
    return source_items, redacted, allowed_numeric_points


def _build_stage_a_parts(
    extracted: ExtractedChart,
    *,
    mode: str = "full",
    sharing_posture: str = "patient_self",
) -> Tuple[List[ClinicalItem], List[ClinicalItem], Dict[str, ClinicalItem], List[Dict[str, Any]]]:
    """Stage A: Lossless clinical normalization + redundant-repeat RLE folding with zero slice caps or note chops."""
    source_items, redacted, allowed_numeric_points = _filter_source_items_for_posture(
        extracted, mode=mode, sharing_posture=sharing_posture
    )

    patient_items = [x for x in source_items if x.section == "patient"]
    allergies = [x for x in source_items if x.section == "allergy"]
    cond_active = [x for x in source_items if x.section == "condition_active"]
    cond_resolved = [x for x in source_items if x.section == "condition_resolved"]
    med_active = [x for x in source_items if x.section == "med_active"]
    med_hist = [x for x in source_items if x.section == "med_historical"]
    imms = [x for x in source_items if x.section == "immunization"]
    devices = [x for x in source_items if x.section == "device"]
    careplans_active = [x for x in source_items if x.section == "careplan_active"]
    careplans_hist = [x for x in source_items if x.section == "careplan_historical"]
    procs_major = [x for x in source_items if x.section == "procedure_major"]
    procs_routine = [x for x in source_items if x.section == "procedure_routine"]
    enc_acuity = [x for x in source_items if x.section == "encounter_acuity"]
    enc_routine = [x for x in source_items if x.section == "encounter_routine"]
    lab_panels = [x for x in source_items if x.section == "lab_panel"]
    survey_panels = [x for x in source_items if x.section == "survey_panel"]
    vitals = [x for x in source_items if x.section == "vital_sign"]
    labs_standalone = sorted(
        [x for x in source_items if x.section == "lab_observation"],
        key=lambda x: _date_sort_key(x.date),
        reverse=True,
    )
    social_obs = [x for x in source_items if x.section == "social_history"]
    survey_obs = sorted(
        [x for x in source_items if x.section == "survey_observation"],
        key=lambda x: _date_sort_key(x.date),
        reverse=True,
    )
    other_obs = [x for x in source_items if x.section == "observation"]
    imaging = sorted(
        [x for x in source_items if x.section == "imaging"],
        key=lambda x: _date_sort_key(x.date),
        reverse=True,
    )
    notes_full = sorted(
        [x for x in source_items if x.section == "clinical_note"],
        key=lambda x: _date_sort_key(x.date),
        reverse=True,
    )

    folded_imms = _fold_immunizations(imms)
    folded_procs_major = _fold_procedures(procs_major, section="procedure_major")
    folded_procs_routine = _fold_procedures(procs_routine, section="procedure_routine")
    # Preserve every inpatient/ICU/ED/urgent encounter verbatim in Stage A so acute
    # admissions are never collapsed into cumulative count lines.
    folded_enc_acuity = sorted(
        enc_acuity,
        key=lambda x: _date_sort_key(x.date),
        reverse=True,
    )
    folded_enc_routine = _fold_encounters(enc_routine, keep_latest_individual=5)
    grouped_vitals = _group_vitals_by_date(vitals)
    folded_social = _fold_social_history(social_obs)
    folded_labs = _fold_lab_panels_by_type(lab_panels, keep_latest_per_type=3)
    folded_surveys = _fold_lab_panels_by_type(survey_panels, keep_latest_per_type=2)
    trajectories = _build_lab_trajectories(allowed_numeric_points)
    folded_cond_res = _fold_resolved_conditions(cond_resolved)
    folded_med_hist = _fold_stopped_medications(med_hist)
    folded_devices = _fold_devices(devices)

    all_candidates = (
        patient_items
        + allergies
        + cond_active
        + med_active
        + folded_devices
        + folded_imms
        + folded_procs_major
        + folded_enc_acuity
        + careplans_active
        + folded_cond_res
        + folded_med_hist
        + trajectories
        + folded_labs
        + labs_standalone
        + grouped_vitals
        + folded_social
        + other_obs
        + notes_full
        + imaging
        + folded_surveys
        + survey_obs
        + folded_procs_routine
        + careplans_hist
        + folded_enc_routine
    )

    mandatory_items = [it for it in all_candidates if it.mandatory and it.section != "manifest"]
    optional_items = [it for it in all_candidates if not it.mandatory and it.section != "manifest"]
    full_note_by_uuid = {
        n.resource_uuid: n
        for n in notes_full
        if not n.mandatory and n.resource_uuid
    }
    return mandatory_items, optional_items, full_note_by_uuid, redacted


def _build_compressed_and_ultra_mandatory(
    extracted: ExtractedChart,
    mandatory_items: Sequence[ClinicalItem],
    target_tokens: int,
) -> Tuple[List[ClinicalItem], Callable[[bool], List[ClinicalItem]]]:
    """Build compacted mandatory representations preserving 100% of mandatory UUIDs & facts."""
    patient_only = [it for it in mandatory_items if it.section == "patient"]
    allergy_only = [it for it in mandatory_items if it.section == "allergy"]
    device_only = [it for it in mandatory_items if it.section == "device"]
    conflict_conds = [
        it
        for it in mandatory_items
        if it.section in {"condition_active", "condition_resolved"}
        and (it.conflict_note or it.sensitivity == "SDOH/PRIVACY-SENSITIVE")
    ]
    conflict_imms = [
        it
        for it in mandatory_items
        if it.section == "immunization" and it.conflict_note
    ]
    conflict_notes = [
        it
        for it in mandatory_items
        if it.section in {"clinical_note", "note"}
    ]
    other_mandatory = [
        it
        for it in mandatory_items
        if it.section
        not in {
            "patient",
            "allergy",
            "device",
            "condition_active",
            "condition_resolved",
            "clinical_note",
            "note",
            "med_active",
        }
        and it not in conflict_imms
    ]
    other_conds = [
        it
        for it in mandatory_items
        if it.section == "condition_active" and it not in conflict_conds
    ]
    other_meds = [it for it in mandatory_items if it.section == "med_active"]

    compact_conflict_notes: List[ClinicalItem] = []
    for cn in conflict_notes:
        raw_nt = str(cn.metadata.get("full_text") or cn.text or "")
        clause_str = _extract_note_conflict_clauses(raw_nt, max_chars=260, include_vitals=True)
        compact_conflict_notes.append(
            ClinicalItem(
                resource_type=cn.resource_type,
                resource_uuid=cn.resource_uuid,
                date=cn.date,
                clinical_date=cn.clinical_date,
                date_kind=cn.date_kind,
                section=cn.section,
                code=cn.code,
                display=cn.display,
                text=f"Note ({cn.display or 'Clinical Note'}): {clause_str}",
                mandatory=True,
                status=cn.status,
                sensitivity=cn.sensitivity,
                conflict_note=cn.conflict_note,
                member_uuids=cn.member_uuids,
                metadata=cn.metadata,
            )
        )

    compressed_mandatory: List[ClinicalItem] = list(
        patient_only + allergy_only + compact_conflict_notes + other_mandatory
    )
    by_conflict_key: Dict[str, List[ClinicalItem]] = {}
    for cc in conflict_conds:
        key = cc.conflict_note or f"sens:{cc.resource_uuid}"
        by_conflict_key.setdefault(key, []).append(cc)
    for _ckey, cgroup in by_conflict_key.items():
        lead = cgroup[-1]
        l_date, _ = _span_phrase([x.date for x in cgroup])
        has_mixed_status = any((x.status or "active") != "active" for x in cgroup)
        if has_mixed_status:
            names = "; ".join(
                _disp_with_date(x.display, x.date, f"status: {x.status or 'active'}")
                for x in cgroup
            )
            cond_prefix = "Condition:"
        else:
            names = "; ".join(_disp_with_date(x.display, x.date) for x in cgroup)
            cond_prefix = "Condition (status: active):"
        compressed_mandatory.append(
            ClinicalItem(
                resource_type="Condition",
                resource_uuid=lead.resource_uuid,
                date=l_date or lead.date,
                clinical_date=l_date or lead.clinical_date,
                date_kind="clinical_event" if (l_date or lead.date) else lead.date_kind,
                section="condition_active",
                code=lead.code,
                display=lead.display,
                text=f"{cond_prefix} {names}",
                mandatory=True,
                status=lead.status,
                sensitivity=lead.sensitivity,
                conflict_note=lead.conflict_note,
                member_uuids=_merge_uuids(cgroup),
            )
        )
    for ci in conflict_imms:
        compressed_mandatory.append(
            ClinicalItem(
                resource_type="Immunization",
                resource_uuid=ci.resource_uuid,
                date=ci.date,
                clinical_date=ci.clinical_date,
                date_kind="clinical_event" if ci.date else ci.date_kind,
                section="immunization",
                code=ci.code,
                display=ci.display,
                text=f"Immunization: {_disp_with_date(ci.display, ci.date)}",
                mandatory=True,
                conflict_note=ci.conflict_note,
                member_uuids=ci.member_uuids,
            )
        )
    if other_conds:
        l_date, _ = _span_phrase([c.date for c in other_conds])
        c_names = "; ".join(_disp_with_date(c.display, c.date) for c in other_conds)
        compressed_mandatory.append(
            ClinicalItem(
                resource_type="Condition",
                resource_uuid=other_conds[0].resource_uuid,
                date=l_date,
                clinical_date=l_date,
                date_kind="clinical_event" if l_date else "unknown",
                section="condition_active",
                code="active-conditions-folded",
                display="Active Conditions",
                text=f"Condition (status: active): {c_names}",
                mandatory=True,
                member_uuids=_merge_uuids(other_conds),
            )
        )
    if other_meds:
        l_date, _ = _span_phrase([m.date for m in other_meds])
        m_parts: List[str] = []
        for m in other_meds:
            dm = re.search(r"dosage:\s*([^,\)\]]+)", m.text)
            dosage_str = dm.group(1).strip() if dm else ""
            quals: List[str] = []
            if m.date:
                quals.append(m.date)
            if m.status and m.status != "active":
                quals.append(f"status: {m.status}")
            if dosage_str:
                quals.append(f"dosage: {dosage_str}")
            q_str = f" ({', '.join(quals)})" if quals else ""
            m_parts.append(f"{m.display}{q_str}")
        compressed_mandatory.append(
            ClinicalItem(
                resource_type=other_meds[0].resource_type,
                resource_uuid=other_meds[0].resource_uuid,
                date=l_date,
                clinical_date=l_date,
                date_kind="clinical_event" if l_date else "unknown",
                section="med_active",
                code="active-meds-folded",
                display="Active Medications",
                text=f"Medication (status: active): {'; '.join(m_parts)}",
                mandatory=True,
                member_uuids=_merge_uuids(other_meds),
            )
        )
    if device_only:
        l_date, _ = _span_phrase([d.date for d in device_only])
        d_names = "; ".join(d.display for d in device_only)
        compressed_mandatory.append(
            ClinicalItem(
                resource_type="Device",
                resource_uuid=device_only[0].resource_uuid,
                date=l_date,
                clinical_date=None,
                date_kind="administrative" if l_date else device_only[0].date_kind,
                section="device",
                code="active-devices-folded",
                display="Active Devices",
                text=f"Device (status: active): {d_names}",
                mandatory=True,
                member_uuids=_merge_uuids(device_only),
            )
        )

    def _build_ultra(*, compact_details: bool = False) -> List[ClinicalItem]:
        raw_all_conds = conflict_conds + other_conds
        all_imms = [it for it in compressed_mandatory if it.section == "immunization"]
        non_cond_imm_med = [
            it
            for it in compressed_mandatory
            if it.section not in {"condition_active", "immunization", "med_active", "clinical_note", "note"}
        ]
        ultra: List[ClinicalItem] = list(non_cond_imm_med)
        if conflict_notes:
            l_date, _ = _span_phrase([cn.date for cn in conflict_notes])
            per_note_chars = max(24, min(90, 220 // max(1, len(conflict_notes))))
            n_parts = [
                f"{cn.date or 'note'}: {_extract_note_conflict_clauses(str(cn.metadata.get('full_text') or cn.text or ''), max_chars=per_note_chars)}"
                for cn in conflict_notes
            ]
            lead_cn = conflict_notes[0]
            ultra.append(
                ClinicalItem(
                    resource_type=lead_cn.resource_type,
                    resource_uuid=lead_cn.resource_uuid,
                    date=l_date or lead_cn.date,
                    clinical_date=l_date or lead_cn.clinical_date,
                    date_kind="clinical_event" if (l_date or lead_cn.date) else lead_cn.date_kind,
                    section="clinical_note",
                    code="conflict-notes-ultra",
                    display="Clinical Notes",
                    text=f"Clinical Notes: {'; '.join(n_parts)}",
                    mandatory=True,
                    conflict_note=lead_cn.conflict_note,
                    member_uuids=_merge_uuids(conflict_notes),
                )
            )
        if raw_all_conds:
            l_date, _ = _span_phrase([c.date for c in raw_all_conds])
            has_non_active_cond = any((c.status or "active") != "active" for c in raw_all_conds)
            c_parts = []
            for c in raw_all_conds:
                extra_st = f"status: {c.status or 'active'}" if has_non_active_cond else ""
                if compact_details:
                    c_text_part = f"{c.display} ({extra_st})" if extra_st else f"{c.display}"
                else:
                    c_text_part = _disp_with_date(c.display, c.date, extra_st)
                c_parts.append(c_text_part)
            ultra_cond_prefix = "Condition:" if has_non_active_cond else "Condition (status: active):"
            ultra.append(
                ClinicalItem(
                    resource_type="Condition",
                    resource_uuid=raw_all_conds[0].resource_uuid,
                    date=l_date,
                    clinical_date=l_date,
                    date_kind="clinical_event" if l_date else "unknown",
                    section="condition_active",
                    code="active-conditions-ultra",
                    display="Active Conditions",
                    text=f"{ultra_cond_prefix} {'; '.join(c_parts)}",
                    mandatory=True,
                    member_uuids=_merge_uuids(raw_all_conds),
                )
            )
        if other_meds:
            l_date, _ = _span_phrase([m.date for m in other_meds])
            by_med_qual: Dict[Tuple[str, str], List[ClinicalItem]] = {}
            for m in other_meds:
                dm = re.search(r"dosage:\s*([^,\)\]]+)", m.text)
                dosage_str = dm.group(1).strip() if dm else ""
                by_med_qual.setdefault((m.status or "active", dosage_str), []).append(m)
            group_strs: List[str] = []
            for (st, dos), mgroup in by_med_qual.items():
                _, g_span = _span_phrase([x.date for x in mgroup])
                short_names = [
                    re.sub(r"\s+Oral\s+(?:Tablet|Capsule|Solution)s?\b", "", x.display, flags=re.IGNORECASE).strip()
                    for x in mgroup
                ]
                quals: List[str] = []
                if g_span and not compact_details:
                    quals.append(g_span)
                if st and st != "active":
                    quals.append(f"status: {st}")
                if not compact_details and dos:
                    quals.append(f"dosage: {dos}")
                q_str = f" ({', '.join(quals)})" if quals else ""
                group_strs.append(f"{', '.join(short_names)}{q_str}")
            ultra.append(
                ClinicalItem(
                    resource_type=other_meds[0].resource_type,
                    resource_uuid=other_meds[0].resource_uuid,
                    date=l_date,
                    clinical_date=l_date,
                    date_kind="clinical_event" if l_date else "unknown",
                    section="med_active",
                    code="active-meds-ultra",
                    display="Active Medications",
                    text=f"Medication (status: active): {'; '.join(group_strs)}",
                    mandatory=True,
                    member_uuids=_merge_uuids(other_meds),
                )
            )
        if all_imms:
            l_date, _ = _span_phrase([i.date for i in all_imms])
            i_parts = [_disp_with_date(i.display, i.date) for i in all_imms]
            ultra.append(
                ClinicalItem(
                    resource_type="Immunization",
                    resource_uuid=all_imms[0].resource_uuid,
                    date=l_date,
                    clinical_date=l_date,
                    date_kind="clinical_event" if l_date else "unknown",
                    section="immunization",
                    code="anomalous-imms-ultra",
                    display="Immunizations",
                    text=f"Immunizations: {'; '.join(i_parts)}",
                    mandatory=True,
                    member_uuids=_merge_uuids(all_imms),
                )
            )
        return ultra

    return compressed_mandatory, _build_ultra


_PRIMARY_SECTION_QUOTAS: Dict[str, int] = {
    "lab_trajectory": 16,
    "clinical_note": 10,
    "vital_sign": 8,
    "careplan_active": 12,
    "imaging": 8,
    "lab_panel": 8,
    "lab_observation": 16,
    "med_historical": 12,
    "condition_resolved": 15,
    "procedure_major": 10,
    "encounter_acuity": 10,
    "immunization": 10,
    "social_history": 8,
    "survey_panel": 6,
    "procedure_routine": 4,
    "encounter_routine": 4,
}

_MICROBIOLOGY_PATHOLOGY_KW: Tuple[str, ...] = (
    "culture",
    "microbiology",
    "gram stain",
    "susceptib",
    "isolate",
    "growth",
    "mrsa",
    "klebsiella",
    "coli",
    "biopsy",
    "pathology",
)


def _rank_optional_items_deterministic(
    optional_items: Sequence[ClinicalItem],
    *,
    question: str,
) -> List[ClinicalItem]:
    """Budget-independent 2-pass priority ordering for optional items in Stage B.

    Pass 0 selects a fixed primary horizon per clinical section (and prioritizes
    microbiology/cultures over routine numeric repeats) so large historical
    lab/condition tails cannot starve recent notes, vitals, imaging, or careplans
    at 8k–32k budgets. Pass 1 emits the remaining historical tail in balanced
    section priority order. Because quotas and rankings do not depend on
    ``target_tokens``, prefix monotonicity (``covered_uuids(B1) <= covered_uuids(B2)``)
    is strictly guaranteed.
    """
    q_terms = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", question or "")
        if len(tok) >= 3
    }
    section_rank = {
        "lab_trajectory": 1,
        "clinical_note": 2,
        "vital_sign": 3,
        "careplan_active": 4,
        "imaging": 5,
        "lab_panel": 6,
        "med_historical": 7,
        "procedure_major": 8,
        "encounter_acuity": 9,
        "immunization": 10,
        "social_history": 11,
        "lab_observation": 12,
        "condition_resolved": 13,
        "survey_panel": 14,
        "procedure_routine": 15,
        "encounter_routine": 16,
    }

    def _within_section_key(it: ClinicalItem) -> Tuple[Any, ...]:
        hay = f"{it.display} {it.text}".lower()
        q_hits = sum(1 for t in q_terms if t in hay)
        is_micro = 0 if (it.section == "lab_observation" and any(kw in hay for kw in _MICROBIOLOGY_PATHOLOGY_KW)) else 1
        return (
            -q_hits,
            is_micro,
            0 if int(it.metadata.get("n_points", 0)) >= 2 else 1,
            int(it.metadata.get("analyte_rank", 999)),
            -_int_date(it.date),
            str(it.resource_uuid or ""),
        )

    by_section: Dict[str, List[ClinicalItem]] = {}
    for it in optional_items:
        by_section.setdefault(it.section, []).append(it)

    tagged: List[Tuple[Tuple[Any, ...], ClinicalItem]] = []
    for sec, sec_items in by_section.items():
        ordered_sec = sorted(sec_items, key=_within_section_key)
        quota = _PRIMARY_SECTION_QUOTAS.get(sec, 4)
        s_rank = section_rank.get(sec, 20)
        for idx, it in enumerate(ordered_sec):
            hay = f"{it.display} {it.text}".lower()
            q_hits = sum(1 for t in q_terms if t in hay)
            pass_tier = 0 if idx < quota else 1
            tagged.append(((pass_tier, -q_hits, s_rank, idx), it))

    tagged.sort(key=lambda pair: pair[0])
    return [pair[1] for pair in tagged]


def _condense_two_stage_monotonic(
    extracted: ExtractedChart,
    *,
    target_tokens: int,
    mode: str,
    question: str,
    sharing_posture: str,
    counter: Callable[[str], int],
) -> CondensedChartResult:
    """2-Stage Monotonic Chart Condensation Pipeline:

    Stage A: Lossless normalization + redundant-repeat RLE folding with zero magic slice caps
             and zero 400-char note chops.
    Stage B: If Stage A exceeds ``target_tokens``, binary-search a budget-independent prefix of
             prioritized optional items (notes entering as 400-char excerpts), then upgrade
             mandatory lines & expand clinical notes newest-first while under budget.
             Guarantees ``covered_uuids(B1) <= covered_uuids(B2)`` for all ``B1 < B2``.
    """
    mandatory_items, optional_full, full_note_by_uuid, redacted = _build_stage_a_parts(
        extracted, mode=mode, sharing_posture=sharing_posture
    )

    def render_body(
        cand_body: Sequence[ClinicalItem],
        *,
        tier: int,
        tier_label: str,
        excerpted_notes_count: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, List[Dict[str, Any]], int]:
        manifest_item, manifest_dict = _build_completeness_manifest(
            extracted,
            tier,
            tier_label,
            cand_body,
            len(redacted),
            excerpted_notes_count=excerpted_notes_count,
        )
        if cand_body and cand_body[0].section == "patient":
            ordered = [cand_body[0], manifest_item] + list(cand_body[1:])
        else:
            ordered = [manifest_item] + list(cand_body)
        recs = items_to_querystore_records(ordered)
        c_text, c_maps = render_chart(recs)
        toks = counter(c_text)
        return recs, manifest_dict, c_text, c_maps, toks

    # Stage A fast-path: complete normalized + RLE-folded ledger with full untruncated notes
    stage_a_sorted_opt = sorted(optional_full, key=lambda x: _date_sort_key(x.date), reverse=True)
    recs_a, man_a, text_a, maps_a, toks_a = render_body(
        mandatory_items + stage_a_sorted_opt,
        tier=1,
        tier_label="Normalized & Folded Complete Ledger",
        excerpted_notes_count=0,
    )
    if toks_a <= target_tokens:
        return CondensedChartResult(
            tier=1,
            tier_name=man_a["tier_name"],
            target_tokens=target_tokens,
            actual_tokens=toks_a,
            chart_text=text_a,
            records=recs_a,
            mappings=maps_a,
            completeness_manifest=man_a,
            redacted_for_privacy=redacted,
            compression_stats={
                "bundle_entries": extracted.total_bundle_entries,
                "extracted_clinical_items": len(extracted.items),
                "rendered_records": len(recs_a),
                "target_tokens": target_tokens,
                "actual_tokens": toks_a,
                "budget_utilization": round(toks_a / target_tokens, 3),
            },
        )

    # Stage B: Prepare optional items with 400-char excerpted notes for compact packing
    optional_excerpted: List[ClinicalItem] = []
    excerpted_uuids: Set[str] = set()
    for it in optional_full:
        if it.section == "clinical_note":
            ex = _excerpt_notes([it], max_chars=400)[0]
            optional_excerpted.append(ex)
            if ex.text != it.text and it.resource_uuid:
                excerpted_uuids.add(it.resource_uuid)
        else:
            optional_excerpted.append(it)

    ranked_optional = _rank_optional_items_deterministic(optional_excerpted, question=question)
    compressed_mandatory, build_ultra = _build_compressed_and_ultra_mandatory(
        extracted, mandatory_items, target_tokens
    )

    _, _, _, _, comp_toks = render_body(
        compressed_mandatory,
        tier=3,
        tier_label="IPS Core + Budget-Packed Ledger",
        excerpted_notes_count=len(excerpted_uuids),
    )
    if comp_toks > target_tokens:
        # Tight budget: fit ultra-compact mandatory core with 0 optional items so UUID monotonicity holds
        for compact_flag in (False, True):
            ultra_body = build_ultra(compact_details=compact_flag)
            recs_u, man_u, text_u, maps_u, toks_u = render_body(
                ultra_body,
                tier=4,
                tier_label="IPS Ultra-Compact Micro-Capsule",
                excerpted_notes_count=0,
            )
            if toks_u <= target_tokens:
                return CondensedChartResult(
                    tier=4,
                    tier_name=man_u["tier_name"],
                    target_tokens=target_tokens,
                    actual_tokens=toks_u,
                    chart_text=text_u,
                    records=recs_u,
                    mappings=maps_u,
                    completeness_manifest=man_u,
                    redacted_for_privacy=redacted,
                    compression_stats={
                        "bundle_entries": extracted.total_bundle_entries,
                        "extracted_clinical_items": len(extracted.items),
                        "rendered_records": len(recs_u),
                        "target_tokens": target_tokens,
                        "actual_tokens": toks_u,
                        "budget_utilization": round(toks_u / target_tokens, 3),
                    },
                )
        raise ValueError(
            f"target_tokens ({target_tokens}) is too small to fit mandatory clinical safety core ({toks_u} tokens required)"
        )

    def _fit_prefix(k: int) -> Tuple[List[ClinicalItem], int, int]:
        chosen = ranked_optional[:k]
        chosen_sorted = sorted(chosen, key=lambda x: _date_sort_key(x.date), reverse=True)
        ex_cnt = sum(1 for x in chosen if x.resource_uuid in excerpted_uuids)
        omitted_est = len(ranked_optional) - k
        t_num = 2 if (omitted_est == 0 and ex_cnt > 0) else (1 if omitted_est == 0 else 3)
        t_lbl = (
            "Normalized & Folded Ledger (Excerpted Notes)"
            if t_num == 2
            else ("Normalized & Folded Complete Ledger" if t_num == 1 else "IPS Core + Budget-Packed Ledger")
        )
        _, _, _, _, toks = render_body(
            compressed_mandatory + chosen_sorted,
            tier=t_num,
            tier_label=t_lbl,
            excerpted_notes_count=ex_cnt,
        )
        return compressed_mandatory + chosen_sorted, toks, ex_cnt

    # Binary search largest prefix k in 0..len(ranked_optional)
    lo, hi = 0, len(ranked_optional)
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _, mid_toks, _ = _fit_prefix(mid)
        if mid_toks <= target_tokens:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Advance greedily in case of 1-token BPE boundary jitter, or back off if needed
    while best_k < len(ranked_optional) and _fit_prefix(best_k + 1)[1] <= target_tokens:
        best_k += 1
    while best_k > 0 and _fit_prefix(best_k)[1] > target_tokens:
        best_k -= 1

    accepted_opt = list(ranked_optional[:best_k])
    active_mandatory = compressed_mandatory
    remaining_excerpted = {it.resource_uuid for it in accepted_opt if it.resource_uuid in excerpted_uuids}

    # Once ALL optional items fit, progressively restore uncompressed mandatory lines and full notes
    if best_k == len(ranked_optional):
        trial_uncomp = sorted(accepted_opt, key=lambda x: _date_sort_key(x.date), reverse=True)
        t_num_u = 2 if remaining_excerpted else 1
        t_lbl_u = "Normalized & Folded Ledger (Excerpted Notes)" if remaining_excerpted else "Normalized & Folded Complete Ledger"
        _, _, _, _, toks_uncomp = render_body(
            mandatory_items + trial_uncomp,
            tier=t_num_u,
            tier_label=t_lbl_u,
            excerpted_notes_count=len(remaining_excerpted),
        )
        if toks_uncomp <= target_tokens:
            active_mandatory = mandatory_items

        # Greedily expand excerpted clinical notes newest-first while under budget
        note_indices_newest = [
            idx
            for idx, it in sorted(
                ((i, x) for i, x in enumerate(accepted_opt) if x.resource_uuid in remaining_excerpted),
                key=lambda pair: (_date_sort_key(pair[1].date), str(pair[1].resource_uuid)),
                reverse=True,
            )
        ]
        for idx_n in note_indices_newest:
            uid = accepted_opt[idx_n].resource_uuid
            full_item = full_note_by_uuid.get(uid)
            if full_item is None:
                continue
            cand_opt = list(accepted_opt)
            cand_opt[idx_n] = full_item
            cand_sorted = sorted(cand_opt, key=lambda x: _date_sort_key(x.date), reverse=True)
            rem_cnt = len(remaining_excerpted) - 1
            t_n = 2 if rem_cnt > 0 else 1
            t_l = "Normalized & Folded Ledger (Excerpted Notes)" if rem_cnt > 0 else "Normalized & Folded Complete Ledger"
            _, _, _, _, toks_cand = render_body(
                active_mandatory + cand_sorted,
                tier=t_n,
                tier_label=t_l,
                excerpted_notes_count=rem_cnt,
            )
            if toks_cand <= target_tokens:
                accepted_opt = cand_opt
                remaining_excerpted.discard(uid)

    final_sorted_opt = sorted(accepted_opt, key=lambda x: _date_sort_key(x.date), reverse=True)
    final_body = active_mandatory + final_sorted_opt
    final_ex_cnt = len(remaining_excerpted)
    if best_k < len(ranked_optional):
        final_tier = 3
        final_label = "IPS Core + Budget-Packed Ledger"
    elif final_ex_cnt > 0:
        final_tier = 2
        final_label = "Normalized & Folded Ledger (Excerpted Notes)"
    else:
        final_tier = 1
        final_label = "Normalized & Folded Complete Ledger"

    final_recs, final_man, final_text, final_maps, final_toks = render_body(
        final_body,
        tier=final_tier,
        tier_label=final_label,
        excerpted_notes_count=final_ex_cnt,
    )
    return CondensedChartResult(
        tier=final_tier,
        tier_name=final_man["tier_name"],
        target_tokens=target_tokens,
        actual_tokens=final_toks,
        chart_text=final_text,
        records=final_recs,
        mappings=final_maps,
        completeness_manifest=final_man,
        redacted_for_privacy=redacted,
        compression_stats={
            "bundle_entries": extracted.total_bundle_entries,
            "extracted_clinical_items": len(extracted.items),
            "rendered_records": len(final_recs),
            "target_tokens": target_tokens,
            "actual_tokens": final_toks,
            "budget_utilization": round(final_toks / target_tokens, 3),
        },
    )


def condense_bundle_to_budget(
    bundle: Mapping[str, Any] | ExtractedChart,
    *,
    target_tokens: int = 8192,
    mode: str = "full",
    condense_mode: str = "tiered",
    question: str = "",
    sharing_posture: str = "patient_self",
    filter_room_equipment_devices: Optional[bool] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> CondensedChartResult:
    """Condense a FHIR R4 Bundle (or `ExtractedChart`) into `target_tokens`.

    Supports `condense_mode in {"tiered", "hierarchical"}` (2-Stage Monotonic
    Structural Condensation Pipeline: Stage A lossless clinical normalization and
    redundant-repeat RLE with unpacked inpatient acuity encounters; Stage B
    strict prefix priority packing with progressive full-note expansion).
    """
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if mode not in {"full", "reconcile"}:
        raise ValueError(f"Unsupported condensation mode: {mode!r}")
    if condense_mode not in {"tiered", "hierarchical"}:
        raise ValueError(
            f"Unsupported condense_mode: {condense_mode!r}; expected 'tiered' or 'hierarchical'"
        )
    if sharing_posture not in {"patient_self", "external_share"}:
        raise ValueError(f"Unsupported sharing_posture: {sharing_posture!r}")

    counter = token_counter or estimate_tokens_local
    filter_equip_tiered = True if filter_room_equipment_devices is None else filter_room_equipment_devices
    extracted = (
        bundle
        if isinstance(bundle, ExtractedChart)
        else extract_bundle(bundle, filter_room_equipment_devices=filter_equip_tiered)
    )
    return _condense_two_stage_monotonic(
        extracted,
        target_tokens=target_tokens,
        mode=mode,
        question=question,
        sharing_posture=sharing_posture,
        counter=counter,
    )
