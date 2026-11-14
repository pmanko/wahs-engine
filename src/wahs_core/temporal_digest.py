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

from .clinical_graph import _build_citation_lookup, _is_sensitive_item, _query_terms
from .condenser import CondensedChartResult, estimate_tokens_local
from .extractors import ClinicalItem, ExtractedChart, NumericPoint
from .normalizers import classify_sensitivity

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_UPCOMING_ENCOUNTER_STATUSES: Set[str] = {"planned", "booked", "scheduled", "proposed", "arrived"}
_FUTURE_APPT_STATUSES: Set[str] = {"planned", "booked", "scheduled", "proposed"}
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


def _series_priority(concept: str, latest: NumericPoint, n_points: int, q_terms: Set[str]) -> Tuple[int, int, int, int, str]:
    c_low = (concept or "").lower()
    overlap = sum(1 for t in q_terms if t in c_low)
    abn = 0 if latest.is_abnormal else 1
    multi = 0 if n_points >= 2 else 1
    prio = next(
        (idx for idx, pat in enumerate(_PRIORITY_SERIES_PATTERNS) if pat.search(c_low)),
        len(_PRIORITY_SERIES_PATTERNS) + 1,
    )
    return (-overlap, abn, multi, prio, c_low)


def _collect_numeric_series(
    points: Sequence[NumericPoint],
    *,
    redacted_uuids: Set[str],
    is_external: bool,
    uuid_to_cit: Mapping[str, int],
    q_terms: Set[str],
) -> Tuple[List[Tuple[str, Dict[str, Any]]], int, int]:
    """Group typed ``NumericPoint`` objects into chronological series with n=1 trend guard."""
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
        dedup_key = (pt.resource_uuid or "", code_norm or concept_norm, iso_dt, float(pt.value))
        if dedup_key in seen_point_keys:
            continue
        seen_point_keys.add(dedup_key)
        grouped.setdefault(group_key, []).append(pt)

    rendered_ranked: List[Tuple[Tuple[int, int, int, int, str], str, Dict[str, Any]]] = []
    multi_point_count = 0

    for _, raw_pts in grouped.items():
        pts_sorted = sorted(raw_pts, key=lambda p: (p.date or "", p.resource_uuid or ""))
        n_pts = len(pts_sorted)
        first = pts_sorted[0]
        last = pts_sorted[-1]
        first_dt = _iso10(first.date) or ""
        last_dt = _iso10(last.date) or ""
        concept_label = (last.concept or first.concept or last.code or "Observation").strip()
        unit_label = f" {last.unit.strip()}" if (last.unit or "").strip() else ""
        abn_note = f" [{last.interpretation}]" if last.interpretation and last.is_abnormal else (" [ABNORMAL]" if last.is_abnormal else "")
        last_cite = _cite_for_uuid(uuid_to_cit, last.resource_uuid)

        if n_pts >= 2:
            multi_point_count += 1
            vals = [p.value for p in pts_sorted]
            vmin, vmax = min(vals), max(vals)
            delta = last.value - first.value
            direction = _series_direction(first.value, last.value)
            sign = "+" if delta > 0 else ""
            first_cite = _cite_for_uuid(uuid_to_cit, first.resource_uuid)
            first_cite_str = first_cite if first_cite != last_cite else ""
            line = (
                f"- {concept_label}: latest {_format_num(last.value)}{unit_label}{abn_note} on {last_dt}{last_cite} "
                f"| window {first_dt}{first_cite_str}..{last_dt}{last_cite} "
                f"(n={n_pts}, {direction} {sign}{_format_num(delta)}; range {_format_num(vmin)}..{_format_num(vmax)})"
            )
            meta: Dict[str, Any] = {
                "concept": concept_label,
                "n_points": n_pts,
                "trend_supported": True,
                "direction": direction,
                "delta": round(delta, 4),
                "start_date": first_dt,
                "end_date": last_dt,
                "latest_value": last.value,
                "unit": (last.unit or "").strip(),
            }
        else:
            line = (
                f"- {concept_label}: {_format_num(last.value)}{unit_label}{abn_note} on {last_dt}{last_cite} "
                f"(n=1, single point — trend not supported)"
            )
            meta = {
                "concept": concept_label,
                "n_points": 1,
                "trend_supported": False,
                "direction": "none",
                "delta": None,
                "start_date": last_dt,
                "end_date": last_dt,
                "latest_value": last.value,
                "unit": (last.unit or "").strip(),
            }

        rank_key = _series_priority(concept_label, last, n_pts, q_terms)
        rendered_ranked.append((rank_key, line, meta))

    rendered_ranked.sort(key=lambda pair: pair[0])
    out_entries = [(line, meta) for _, line, meta in rendered_ranked]
    return out_entries, len(rendered_ranked), multi_point_count


def _format_med_line(item: ClinicalItem, uuid_to_cit: Mapping[str, int]) -> str:
    label = (item.display or item.text or "Medication").strip()
    dt = _iso10(item.clinical_date or item.date) or "date unknown"
    status = (item.status or "unknown").strip().lower()
    cite = _cite_for_uuid(uuid_to_cit, item.resource_uuid, item.member_uuids)
    org_part = f" | {item.source_org}" if item.source_org else ""
    return f"  - {label} ({dt}, status: {status}{org_part}){cite}"


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
    q_terms = _query_terms(question)

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
        c_dt = _clinical_item_date(it)
        any_dt = _any_non_patient_date(it)

        if st in _FUTURE_APPT_STATUSES:
            upcoming_items.append((any_dt or "date TBD", it))
            continue

        if c_dt is not None:
            completed_clinical_dates.append((c_dt, it))
            if it.section in {"encounter_acuity", "encounter_routine"} or it.resource_type == "Encounter":
                encounters_completed.append((c_dt, it))
        elif any_dt is not None and it.section not in _ADMIN_SECTIONS and it.date_kind != "administrative":
            fallback_clinical_dates.append(any_dt)

    if completed_clinical_dates:
        reference_date = max(dt for dt, _ in completed_clinical_dates)
    elif fallback_clinical_dates:
        reference_date = max(fallback_clinical_dates)
    else:
        all_valid = [d for d in (_any_non_patient_date(it) for it in kept_items) if d is not None]
        reference_date = max(all_valid) if all_valid else _iso10(extracted.reference_date)

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

    # 2. Longitudinal Numeric Series (with no-trend-from-one-point guard)
    ranked_series, total_series_count, multi_point_count = _collect_numeric_series(
        extracted.numeric_points,
        redacted_uuids=redacted_uuids,
        is_external=is_external,
        uuid_to_cit=uuid_to_cit,
        q_terms=q_terms,
    )
    max_rendered_series = 14
    selected_series = ranked_series[:max_rendered_series]

    # 3. Medication Timeline Summary (active vs stopped/historical)
    med_items = [
        it
        for it in kept_items
        if it.section in {"med_active", "med_historical"}
        or it.resource_type
        in {"MedicationRequest", "MedicationStatement", "MedicationAdministration", "MedicationDispense"}
    ]
    active_meds: List[ClinicalItem] = []
    stopped_meds: List[ClinicalItem] = []
    for m in med_items:
        st = (m.status or "").strip().lower()
        if m.section == "med_historical" or st in _STOPPED_MED_STATUSES:
            stopped_meds.append(m)
        else:
            active_meds.append(m)

    active_meds.sort(key=lambda m: (_iso10(m.clinical_date or m.date) or "0000-00-00", m.display or ""), reverse=True)
    stopped_meds.sort(key=lambda m: (_iso10(m.clinical_date or m.date) or "0000-00-00", m.display or ""), reverse=True)

    active_med_lines = [_format_med_line(m, uuid_to_cit) for m in active_meds[:8]]
    stopped_med_lines = [_format_med_line(m, uuid_to_cit) for m in stopped_meds[:8]]

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
        act_meds_slice: Sequence[str],
        stop_meds_slice: Sequence[str],
    ) -> str:
        sections: List[str] = list(header_lines)
        if series_slice:
            sections.append(
                f"Longitudinal Numeric Series ({len(series_slice)} of {total_series_count} shown; n>=2 shows direction/delta, n=1 forbids trend claims):"
            )
            sections.extend(line for line, _ in series_slice)
        else:
            sections.append("Longitudinal Numeric Series: none documented with ISO clinical dates.")

        if act_meds_slice or stop_meds_slice:
            sections.append(
                f"Medication Timeline ({len(active_meds)} active, {len(stopped_meds)} stopped/historical):"
            )
            if act_meds_slice:
                sections.append("  Active Medications:")
                sections.extend(act_meds_slice)
            if stop_meds_slice:
                sections.append("  Stopped/Historical Medications:")
                sections.extend(stop_meds_slice)
        return "\n".join(sections)

    count_tokens = token_counter or estimate_tokens_local
    cur_series = list(selected_series)
    cur_act_meds = list(active_med_lines)
    cur_stop_meds = list(stopped_med_lines)

    digest_text = _render_digest(cur_series, cur_act_meds, cur_stop_meds)
    if max_tokens is not None and max_tokens > 0:
        while count_tokens(digest_text) > max_tokens:
            if len(cur_stop_meds) > 2:
                cur_stop_meds.pop()
            elif len(cur_series) > 4:
                cur_series.pop()
            elif len(cur_act_meds) > 2:
                cur_act_meds.pop()
            elif cur_stop_meds:
                cur_stop_meds.pop()
            elif cur_series:
                cur_series.pop()
            elif cur_act_meds:
                cur_act_meds.pop()
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
            digest_text = _render_digest(cur_series, cur_act_meds, cur_stop_meds)

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
