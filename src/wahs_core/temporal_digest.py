"""Deterministic Chart-Derived Temporal Digest (`wahs_core` Layer 3).

Pure Python stdlib module that builds a compact, deterministic temporal grounding block
directly from Layer-1 ``ExtractedChart`` and Layer-2 ``CondensedChartResult`` dataclasses:
- Separates clinical event dates from administrative/recorded dates so insurance coverage
  or device manufacturing dates never masquerade as clinical encounters.
- Resolves an explicit reference date anchor ("chart now") plus 6-month and 12-month lookback
  boundaries and surfaces upcoming/scheduled encounters (or explicit absence thereof).
- Summarizes longitudinal numeric series sorted chronologically oldest -> newest with range,
  window, delta, and trend direction — enforcing a strict **no-trend-from-single-point guard**
  whenever ``n == 1``.
- Summarizes active vs. stopped/historical medication timelines.
- Respects ``sharing_posture == "external_share"`` privacy segmentation and never emits
  ``"(None)"`` placeholders.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
import datetime as _dt
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .clinical_graph import _build_citation_lookup, _is_sensitive_item, _query_terms, extract_clean_user_query
from .condenser import CondensedChartResult, estimate_tokens_local
from .extractors import ClinicalItem, ExtractedChart, NumericPoint
from .normalizers import classify_sensitivity, is_non_medical_admin_code

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_UPCOMING_ENCOUNTER_STATUSES: Set[str] = {"planned", "booked", "scheduled", "proposed"}
_FUTURE_APPT_STATUSES: Set[str] = {"planned", "booked", "scheduled", "proposed"}
_EXCLUDED_EVENT_STATUSES: Set[str] = {"cancelled", "entered-in-error", "noshow"}
_COMPLETED_ENCOUNTER_STATUSES: Set[str] = {"finished", "completed", ""}
_STOPPED_MED_STATUSES: Set[str] = {
    "stopped",
    "cancelled",
    "entered-in-error",
    "completed",
    "on-hold",
    "ended",
    "inactive",
}
_ADMIN_SECTIONS: Set[str] = {"patient", "manifest", "coverage"}
_PRIORITY_SERIES_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"glomerular\s+filtration|egfr", re.I),
    re.compile(r"creatinine", re.I),
    re.compile(r"a1c|hba1c", re.I),
    re.compile(r"glucose", re.I),
    re.compile(r"potassium", re.I),
    re.compile(r"sodium", re.I),
    re.compile(r"blood\s+pressure|systolic|diastolic", re.I),
    re.compile(r"hemoglobin|hematocrit|platelets|leukocytes", re.I),
    re.compile(r"cholesterol|ldl|hdl|triglycerides", re.I),
    re.compile(r"weight|bmi|body\s+mass", re.I),
    re.compile(r"bnp|troponin|d-dimer|inr", re.I),
)


@dataclass(frozen=True)
class TemporalDigestResult:
    """Deterministic temporal grounding block and structured metadata."""

    reference_date: Optional[str]
    digest_text: str
    digest_tokens: int
    summary: Dict[str, Any]


def _parse_iso_date(val: Optional[str]) -> Optional[_dt.date]:
    if not val:
        return None
    s = str(val).strip()
    if not _ISO_DATE_RE.match(s):
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _iso10(val: Optional[str]) -> Optional[str]:
    parsed = _parse_iso_date(val)
    return parsed.isoformat() if parsed else None


def _shift_months(anchor: _dt.date, months: int) -> _dt.date:
    month_idx = anchor.year * 12 + (anchor.month - 1) + months
    year, month0 = divmod(month_idx, 12)
    month = month0 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return _dt.date(year, month, day)


def _cite_for_uuid(
    uuid_to_cit: Mapping[str, int],
    resource_uuid: Optional[str],
    member_uuids: Sequence[str] = (),
) -> str:
    if resource_uuid and resource_uuid in uuid_to_cit:
        return f" [{uuid_to_cit[resource_uuid]}]"
    for m_uuid in member_uuids:
        if m_uuid and m_uuid in uuid_to_cit:
            return f" [{uuid_to_cit[m_uuid]}]"
    return ""


def _clinical_item_date(item: ClinicalItem) -> Optional[str]:
    """Return canonical YYYY-MM-DD clinical event date if item represents a dated clinical event."""
    if item.section in _ADMIN_SECTIONS:
        return None
    if item.date_kind == "administrative":
        return None
    return _iso10(item.clinical_date or (item.date if item.date_kind == "clinical_event" else None))


def _any_non_patient_date(item: ClinicalItem) -> Optional[str]:
    if item.section in {"patient", "manifest"}:
        return None
    return _iso10(item.clinical_date or item.date or item.end_date)


def _format_num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}" if abs(value) >= 100 else f"{value:.2f}".rstrip("0").rstrip(".")


def _series_direction(first_val: float, last_val: float) -> str:
    delta = last_val - first_val
    scale = max(abs(first_val), abs(last_val), 1.0)
    if abs(delta) <= max(1e-6, 0.01 * scale):
        return "stable"
    return "increasing" if delta > 0 else "decreasing"


def _series_priority(
    concept: str,
    latest: NumericPoint,
    n_points: int,
    q_terms: Set[str],
    *,
    past_12m_start: Optional[str] = None,
) -> Tuple[int, int, int, int, int, int, str]:
    c_low = (concept or "").lower()
    overlap = sum(1 for t in q_terms if t in c_low)
    last_dt = _iso10(latest.date) or "0000-00-00"
    is_recent = 0 if (not past_12m_start or last_dt >= past_12m_start) else 1
    abn = 0 if latest.is_abnormal else 1
    multi = 0 if n_points >= 2 else 1
    prio = next(
        (idx for idx, pat in enumerate(_PRIORITY_SERIES_PATTERNS) if pat.search(c_low)),
        len(_PRIORITY_SERIES_PATTERNS) + 1,
    )
    neg_dt = -int(last_dt.replace("-", "")) if re.match(r"^\d{4}-\d{2}-\d{2}$", last_dt) else 0
    return (-overlap, is_recent, abn, multi, prio, neg_dt, c_low)


def _day_boundary_points(
    day_pts: Sequence[NumericPoint],
    *,
    pick_max: bool,
) -> Tuple[bool, List[NumericPoint]]:
    """Return whether a boundary day yields a deterministic single value (no UUID tie-breaking)."""
    if len(day_pts) == 1:
        return True, list(day_pts)
    all_utc = all((p.timestamp or "").startswith("Z:") for p in day_pts)
    all_naive = all((p.timestamp or "").startswith("N:") for p in day_pts)
    if all_utc or all_naive:
        extreme_ts = (max if pick_max else min)(p.timestamp for p in day_pts)
        extreme_pts = [p for p in day_pts if p.timestamp == extreme_ts]
        return len({p.value for p in extreme_pts}) == 1, extreme_pts
    if len({p.value for p in day_pts}) == 1:
        return True, list(day_pts)
    return False, list(day_pts)


def _resolve_series_bounds(
    raw_pts: Sequence[NumericPoint],
) -> Tuple[bool, bool, List[NumericPoint], List[NumericPoint], str, str]:
    """Resolve chronological first/last groups without falling back to UUID ordering."""
    day_of = lambda p: _iso10(p.date) or ""
    all_utc = all((p.timestamp or "").startswith("Z:") for p in raw_pts)
    all_naive = all((p.timestamp or "").startswith("N:") for p in raw_pts)
    if all_utc or all_naive:
        pts_sorted = sorted(raw_pts, key=lambda p: (p.timestamp, day_of(p)))
        min_ts = pts_sorted[0].timestamp
        max_ts = pts_sorted[-1].timestamp
        first_group = [p for p in pts_sorted if p.timestamp == min_ts]
        last_group = [p for p in pts_sorted if p.timestamp == max_ts]
        first_ok = len({p.value for p in first_group}) == 1
        last_ok = len({p.value for p in last_group}) == 1
        trend_supported = (min_ts < max_ts) and first_ok and last_ok
        return (
            trend_supported,
            last_ok,
            first_group,
            last_group,
            day_of(first_group[0]),
            day_of(last_group[-1]),
        )

    min_day = min(day_of(p) for p in raw_pts)
    max_day = max(day_of(p) for p in raw_pts)
    first_day_pts = [p for p in raw_pts if day_of(p) == min_day]
    last_day_pts = [p for p in raw_pts if day_of(p) == max_day]
    first_ok, first_group = _day_boundary_points(first_day_pts, pick_max=False)
    last_ok, last_group = _day_boundary_points(last_day_pts, pick_max=True)
    trend_supported = (min_day < max_day) and first_ok and last_ok
    return trend_supported, last_ok, first_group, last_group, min_day, max_day


def _cites_for_group(uuid_to_cit: Mapping[str, int], pts: Sequence[NumericPoint]) -> str:
    indices = sorted(
        {uuid_to_cit[p.resource_uuid] for p in pts if p.resource_uuid and p.resource_uuid in uuid_to_cit}
    )
    return "".join(f"[{idx}]" for idx in indices)


def _abn_note_for_group(pts: Sequence[NumericPoint]) -> Tuple[str, bool]:
    if not any(p.is_abnormal for p in pts):
        return "", False
    interps = sorted({str(p.interpretation).strip() for p in pts if p.is_abnormal and p.interpretation})
    if len(interps) == 1:
        return f" [{interps[0]}]", True
    return " [ABNORMAL]", True


def _collect_numeric_series(
    points: Sequence[NumericPoint],
    *,
    redacted_uuids: Set[str],
    is_external: bool,
    uuid_to_cit: Mapping[str, int],
    q_terms: Set[str],
    past_12m_start: Optional[str] = None,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], int, int]:
    """Group typed ``NumericPoint`` objects into chronological series with n=1 & tie trend guards."""
    grouped: Dict[Tuple[str, str], List[NumericPoint]] = {}
    seen_point_keys: Set[Tuple[str, str, str, float]] = set()

    for pt in points:
        if pt.resource_uuid and pt.resource_uuid in redacted_uuids:
            continue
        if is_external and classify_sensitivity(pt.code, pt.concept, [pt.category]) == "SDOH/PRIVACY-SENSITIVE":
            continue
        iso_dt = _iso10(pt.date)
        if iso_dt is None:
            continue
        unit_norm = (pt.unit or "").strip()
        code_norm = (pt.code or "").strip().lower()
        concept_norm = re.sub(r"\s+", " ", (pt.concept or "").strip()).lower()
        group_key = (code_norm or concept_norm or "observation", unit_norm.lower())
        dedup_key = (pt.resource_uuid or "", code_norm or concept_norm, pt.timestamp or iso_dt, float(pt.value))
        if dedup_key in seen_point_keys:
            continue
        seen_point_keys.add(dedup_key)
        grouped.setdefault(group_key, []).append(pt)

    rendered_ranked: List[Tuple[Tuple[int, int, int, int, int, int, str], str, Dict[str, Any]]] = []
    multi_point_count = 0

    for _, raw_pts in grouped.items():
        n_pts = len(raw_pts)
        concept_label = sorted(
            {
                (p.concept or p.code or "Observation").strip()
                for p in raw_pts
            }
        )[-1]
        unit_label_raw = sorted({(p.unit or "").strip() for p in raw_pts})[-1]
        unit_label = f" {unit_label_raw}" if unit_label_raw else ""

        if n_pts == 1:
            last = raw_pts[0]
            last_dt = _iso10(last.date) or ""
            abn_note = (
                f" [{last.interpretation}]"
                if last.interpretation and last.is_abnormal
                else (" [ABNORMAL]" if last.is_abnormal else "")
            )
            last_cite = _cite_for_uuid(uuid_to_cit, last.resource_uuid)
            line = (
                f"- {concept_label}: {_format_num(last.value)}{unit_label}{abn_note} on {last_dt}{last_cite} "
                f"(n=1, single point — trend not supported)"
            )
            meta: Dict[str, Any] = {
                "concept": concept_label,
                "n_points": 1,
                "trend_supported": False,
                "direction": "none",
                "delta": None,
                "start_date": last_dt,
                "end_date": last_dt,
                "latest_value": last.value,
                "unit": unit_label_raw,
            }
            rank_pt = last
        else:
            trend_supported, latest_known, first_group, last_group, first_dt, last_dt = _resolve_series_bounds(raw_pts)
            vals = [p.value for p in raw_pts]
            vmin, vmax = min(vals), max(vals)
            abn_note, any_abn = _abn_note_for_group(last_group)
            last_cite = _cites_for_group(uuid_to_cit, last_group)

            if trend_supported:
                multi_point_count += 1
                first_val = first_group[0].value
                last_val = last_group[0].value
                delta = last_val - first_val
                direction = _series_direction(first_val, last_val)
                sign = "+" if delta > 0 else ""
                first_cite = _cites_for_group(uuid_to_cit, first_group)
                first_cite_str = first_cite if first_cite != last_cite else ""
                line = (
                    f"- {concept_label}: latest {_format_num(last_val)}{unit_label}{abn_note} on {last_dt}{last_cite} "
                    f"| window {first_dt}{first_cite_str}..{last_dt}{last_cite} "
                    f"(n={n_pts}, {direction} {sign}{_format_num(delta)}; range {_format_num(vmin)}..{_format_num(vmax)})"
                )
                meta = {
                    "concept": concept_label,
                    "n_points": n_pts,
                    "trend_supported": True,
                    "direction": direction,
                    "delta": round(delta, 4),
                    "start_date": first_dt,
                    "end_date": last_dt,
                    "latest_value": last_val,
                    "unit": unit_label_raw,
                }
                rank_pt = NumericPoint(
                    resource_uuid="",
                    date=last_dt,
                    code=last_group[0].code,
                    concept=concept_label,
                    value=last_val,
                    unit=unit_label_raw,
                    category=last_group[0].category,
                    is_abnormal=any_abn,
                )
            else:
                latest_val_opt: Optional[float] = last_group[0].value if latest_known else None
                if latest_val_opt is not None:
                    head_val = f"latest {_format_num(latest_val_opt)}{unit_label}{abn_note}"
                else:
                    head_val = f"range {_format_num(vmin)}..{_format_num(vmax)}{unit_label}{abn_note}"
                line = (
                    f"- {concept_label}: {head_val} on {last_dt}{last_cite} "
                    f"| window {first_dt}..{last_dt} "
                    f"(n={n_pts}, intraday order unknown — trend not supported; range {_format_num(vmin)}..{_format_num(vmax)})"
                )
                meta = {
                    "concept": concept_label,
                    "n_points": n_pts,
                    "trend_supported": False,
                    "direction": "none",
                    "delta": None,
                    "start_date": first_dt,
                    "end_date": last_dt,
                    "latest_value": latest_val_opt,
                    "unit": unit_label_raw,
                }
                rank_pt = NumericPoint(
                    resource_uuid="",
                    date=last_dt,
                    code=last_group[0].code,
                    concept=concept_label,
                    value=latest_val_opt if latest_val_opt is not None else vmax,
                    unit=unit_label_raw,
                    category=last_group[0].category,
                    is_abnormal=any_abn,
                )

        rank_key = _series_priority(
            concept_label, rank_pt, n_pts, q_terms, past_12m_start=past_12m_start
        )
        rendered_ranked.append((rank_key, line, meta))

    rendered_ranked.sort(key=lambda pair: pair[0])
    out_entries = [(line, meta) for _, line, meta in rendered_ranked]
    return out_entries, len(rendered_ranked), multi_point_count


def build_temporal_digest(
    extracted: ExtractedChart,
    condensed: CondensedChartResult,
    *,
    question: str = "",
    sharing_posture: str = "patient_self",
    extra_uuid_to_citation: Optional[Mapping[str, int]] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> TemporalDigestResult:
    """Build deterministic chart-derived temporal anchor, encounter horizon, and trend digest."""
    is_external = sharing_posture == "external_share"
    clean_q = extract_clean_user_query(question) or question
    q_terms = _query_terms(clean_q)

    uuid_to_cit, _, _ = _build_citation_lookup(condensed, None)
    if extra_uuid_to_citation:
        for k, v in extra_uuid_to_citation.items():
            if isinstance(k, str) and k and isinstance(v, int) and v > 0:
                uuid_to_cit.setdefault(k, v)

    redacted_uuids: Set[str] = set()
    kept_items: List[ClinicalItem] = []
    for it in extracted.items:
        if is_external and _is_sensitive_item(it):
            if it.resource_uuid:
                redacted_uuids.add(it.resource_uuid)
            redacted_uuids.update(u for u in it.member_uuids if u)
            continue
        kept_items.append(it)

    # 1. Resolve reference date anchor strictly from completed clinical events (never admin/future).
    completed_clinical_dates: List[Tuple[str, ClinicalItem]] = []
    fallback_clinical_dates: List[str] = []
    upcoming_items: List[Tuple[str, ClinicalItem]] = []
    encounters_completed: List[Tuple[str, ClinicalItem]] = []

    for it in kept_items:
        if it.section in {"patient", "manifest"}:
            continue
        st = (it.status or "").strip().lower()
        if st in _EXCLUDED_EVENT_STATUSES:
            continue
        c_dt = _clinical_item_date(it)
        any_dt = _any_non_patient_date(it)

        if st in _FUTURE_APPT_STATUSES:
            upcoming_items.append((any_dt or "date TBD", it))
            continue

        if c_dt is not None:
            completed_clinical_dates.append((c_dt, it))
            is_encounter = it.section in {"encounter_acuity", "encounter_routine"} or it.resource_type == "Encounter"
            if is_encounter and st in _COMPLETED_ENCOUNTER_STATUSES:
                encounters_completed.append((c_dt, it))
        elif any_dt is not None and it.section not in _ADMIN_SECTIONS and it.date_kind != "administrative":
            fallback_clinical_dates.append(any_dt)

    if completed_clinical_dates:
        reference_date = max(dt for dt, _ in completed_clinical_dates)
    elif fallback_clinical_dates:
        reference_date = max(fallback_clinical_dates)
    else:
        reference_date = _iso10(extracted.reference_date)

    # Reclassify any Encounter or active CarePlan dated strictly after reference_date as upcoming.
    filtered_completed_encounters: List[Tuple[str, ClinicalItem]] = []
    seen_upcoming_uuids: Set[str] = {it.resource_uuid for _, it in upcoming_items if it.resource_uuid}
    for dt, it in encounters_completed:
        if reference_date and dt > reference_date:
            if not it.resource_uuid or it.resource_uuid not in seen_upcoming_uuids:
                upcoming_items.append((dt, it))
                if it.resource_uuid:
                    seen_upcoming_uuids.add(it.resource_uuid)
        else:
            filtered_completed_encounters.append((dt, it))

    ref_obj = _parse_iso_date(reference_date)
    past_6m_start = _shift_months(ref_obj, -6).isoformat() if ref_obj else None
    past_12m_start = _shift_months(ref_obj, -12).isoformat() if ref_obj else None

    last_enc_pair = (
        max(filtered_completed_encounters, key=lambda pair: (pair[0], pair[1].resource_uuid or ""))
        if filtered_completed_encounters
        else None
    )
    last_enc_date = last_enc_pair[0] if last_enc_pair else None

    latest_activity_pair = (
        max(completed_clinical_dates, key=lambda pair: (pair[0], pair[1].resource_uuid or ""))
        if completed_clinical_dates
        else None
    )
    latest_activity_date = latest_activity_pair[0] if latest_activity_pair else reference_date

    # Administrative/device dates that post-date the last clinical encounter must not masquerade as visits.
    post_visit_admin_lines: List[str] = []
    compare_floor = last_enc_date or reference_date
    if compare_floor:
        for it in kept_items:
            if it.section in {"patient", "manifest"}:
                continue
            if it.date_kind == "administrative" or it.section in {"device", "coverage"}:
                a_dt = _iso10(it.date or it.end_date)
                if a_dt and a_dt > compare_floor:
                    cite = _cite_for_uuid(uuid_to_cit, it.resource_uuid, it.member_uuids)
                    lbl = (it.display or it.resource_type or "Record").strip()
                    post_visit_admin_lines.append(
                        f"- Administrative/Non-Visit Record Post-Dating Last Clinical Encounter: {lbl} ({a_dt}; NOT a clinical encounter){cite}"
                    )

    # 2. Longitudinal Numeric Series (with no-trend-from-one-point guard & stem dedup)
    ranked_series, total_series_count, multi_point_count = _collect_numeric_series(
        extracted.numeric_points,
        redacted_uuids=redacted_uuids,
        is_external=is_external,
        uuid_to_cit=uuid_to_cit,
        q_terms=q_terms,
        past_12m_start=past_12m_start,
    )
    deduped_series: List[Tuple[str, Dict[str, Any]]] = []
    seen_stems: Set[str] = set()
    for entry in ranked_series:
        c_raw = str(entry[1].get("concept") or "").lower()
        stem = re.sub(r"\s*\[[^\]]+\].*$", "", c_raw).strip() or c_raw
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        deduped_series.append(entry)
    selected_series = deduped_series[:4]

    # 3. Medication Order Horizon (active vs stopped/historical count & latest date guard)
    med_items = [
        it
        for it in kept_items
        if (
            it.section in {"med_active", "med_historical"}
            or it.resource_type
            in {"MedicationRequest", "MedicationStatement", "MedicationAdministration", "MedicationDispense"}
        )
        and not is_non_medical_admin_code(it.code)
        and str((it.metadata or {}).get("setting") or "").lower() != "inpatient/administered"
        and not (it.display or it.text or "").strip().lower().startswith(("inpatient ", "naclflush"))
    ]
    active_meds: List[ClinicalItem] = []
    stopped_meds: List[ClinicalItem] = []
    for m in med_items:
        st = (m.status or "").strip().lower()
        if m.section == "med_historical" or st in _STOPPED_MED_STATUSES:
            stopped_meds.append(m)
        else:
            active_meds.append(m)

    latest_active_dt = max(
        (_iso10(m.clinical_date or m.date) or "" for m in active_meds),
        default="",
    )
    latest_stopped_dt = max(
        (_iso10(m.clinical_date or m.date) or "" for m in stopped_meds),
        default="",
    )
    med_horizon_line = ""
    if active_meds or stopped_meds:
        act_note = f"{len(active_meds)} active (latest {latest_active_dt or 'undated'})"
        stop_note = (
            f"{len(stopped_meds)} stopped/historical (latest {latest_stopped_dt or 'undated'}; "
            "do NOT report stopped/completed orders as currently taken unless explicitly restarted later)"
            if stopped_meds
            else "0 stopped/historical"
        )
        med_horizon_line = f"- Medication Order Horizon: {act_note}, {stop_note}."

    # Build static header & encounter lines
    header_lines: List[str] = [
        "=== DETERMINISTIC TEMPORAL DIGEST (Chart-Derived Anchor, Encounter Horizon, & Longitudinal Trends) ==="
    ]
    if reference_date:
        header_lines.append(
            f"- Reference Date Anchor (Chart 'Now'): {reference_date} | "
            f"Past-6m Start: {past_6m_start} | Past-12m Start: {past_12m_start} "
            "(Copy chart YYYY-MM-DD ISO dates verbatim; treat relative time windows against this anchor.)"
        )
    else:
        header_lines.append(
            "- Reference Date Anchor: undetermined (no dated clinical events present in chart)."
        )

    if last_enc_pair is not None:
        enc_dt, enc_it = last_enc_pair
        enc_cite = _cite_for_uuid(uuid_to_cit, enc_it.resource_uuid, enc_it.member_uuids)
        enc_org = f" @ {enc_it.source_org}" if enc_it.source_org else ""
        header_lines.append(
            f"- Last Completed Clinical Encounter: {enc_dt} — {(enc_it.display or enc_it.code or 'Encounter').strip()}{enc_org}{enc_cite}"
        )
    elif latest_activity_pair is not None:
        act_dt, act_it = latest_activity_pair
        act_cite = _cite_for_uuid(uuid_to_cit, act_it.resource_uuid, act_it.member_uuids)
        header_lines.append(
            f"- Latest Clinical Activity: {act_dt} — {(act_it.display or act_it.resource_type).strip()} ({act_it.section}){act_cite}"
        )

    if latest_activity_pair is not None and last_enc_pair is not None and latest_activity_date != last_enc_date:
        _, act_it = latest_activity_pair
        act_cite = _cite_for_uuid(uuid_to_cit, act_it.resource_uuid, act_it.member_uuids)
        header_lines.append(
            f"- Latest Clinical Record Activity: {latest_activity_date} — {(act_it.display or act_it.resource_type).strip()}{act_cite}"
        )

    header_lines.extend(post_visit_admin_lines[:3])

    if upcoming_items:
        up_sorted = sorted(upcoming_items, key=lambda pair: pair[0])
        up_parts = []
        for up_dt, up_it in up_sorted[:4]:
            up_cite = _cite_for_uuid(uuid_to_cit, up_it.resource_uuid, up_it.member_uuids)
            up_parts.append(f"{(up_it.display or up_it.resource_type).strip()} ({up_dt}, status: {up_it.status}){up_cite}")
        header_lines.append(f"- Upcoming/Scheduled Visits ({len(upcoming_items)}): " + "; ".join(up_parts))
    else:
        header_lines.append("- Upcoming/Scheduled Visits: none documented in chart.")

    def _render_digest(
        series_slice: Sequence[Tuple[str, Dict[str, Any]]],
        *,
        include_med_horizon: bool = True,
    ) -> str:
        sections: List[str] = list(header_lines)
        if series_slice:
            sections.append(
                f"Longitudinal Numeric Series ({len(series_slice)} of {total_series_count} shown; n>=2 shows direction/delta, n=1 forbids trend claims):"
            )
            sections.extend(line for line, _ in series_slice)
        else:
            sections.append("Longitudinal Numeric Series: none documented with ISO clinical dates.")

        if med_horizon_line and include_med_horizon:
            sections.append(med_horizon_line)
        return "\n".join(sections)

    count_tokens = token_counter or estimate_tokens_local
    cur_series = list(selected_series)
    include_med_horizon = bool(med_horizon_line)

    digest_text = _render_digest(cur_series, include_med_horizon=include_med_horizon)
    if max_tokens is not None and max_tokens > 0:
        while count_tokens(digest_text) > max_tokens:
            if len(cur_series) > 2:
                cur_series.pop()
            elif include_med_horizon:
                include_med_horizon = False
            elif cur_series:
                cur_series.pop()
            else:
                lo, hi = 0, len(digest_text)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    cand = digest_text[:mid].rsplit("\n", 1)[0] if mid < len(digest_text) else digest_text
                    if count_tokens(cand) <= max_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                digest_text = digest_text[:lo].rsplit("\n", 1)[0] if lo > 0 else ""
                if count_tokens(digest_text) > max_tokens and lo > 0:
                    digest_text = digest_text[:lo]
                break
            digest_text = _render_digest(cur_series, include_med_horizon=include_med_horizon)

    final_tokens = count_tokens(digest_text)
    summary: Dict[str, Any] = {
        "enabled": True,
        "reference_date": reference_date,
        "past_6m_start": past_6m_start,
        "past_12m_start": past_12m_start,
        "last_clinical_encounter_date": last_enc_date,
        "latest_clinical_activity_date": latest_activity_date,
        "upcoming_appointments_count": len(upcoming_items),
        "numeric_series_count": total_series_count,
        "numeric_series_rendered": len(cur_series),
        "multi_point_series_count": multi_point_count,
        "active_medications_count": len(active_meds),
        "stopped_or_modified_medications_count": len(stopped_meds),
        "digest_tokens": final_tokens,
    }
    return TemporalDigestResult(
        reference_date=reference_date,
        digest_text=digest_text,
        digest_tokens=final_tokens,
        summary=summary,
    )


__all__ = [
    "TemporalDigestResult",
    "build_temporal_digest",
]
