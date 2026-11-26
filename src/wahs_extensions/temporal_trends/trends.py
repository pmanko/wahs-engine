"""Chart Temporal Anchor (`reference_date`, `6m`/`12m` cutoffs, visits, vaccine elapsed time) + Tie-Safe Lab Trends."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, replace
import datetime as _dt
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from wahs_core.condenser import CondensedChartResult, estimate_tokens_local
from wahs_core.extractors import ClinicalItem, ExtractedChart, NumericPoint
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_EXCLUDED_STATUSES: Set[str] = {"cancelled", "entered-in-error", "noshow"}
_VACCINE_TEMPORAL_CUES = (
    "vaccin", "immuniz", "shot", "booster", "preventive", "up to date", "up-to-date",
    "due", "shingrix", "shingles", "zoster", "covid", "tetanus", "tdap", "flu",
    "influenza", "pneumo", "rsv", "hepatitis", "hpv",
)


@dataclass(frozen=True)
class TemporalTrendResult:
    reference_date: Optional[str]
    digest_text: str
    digest_tokens: int
    summary: Dict[str, Any]


def _iso10(val: Optional[str]) -> Optional[str]:
    s = str(val or "").strip()
    if not _ISO_DATE_RE.match(s):
        return None
    try:
        return _dt.date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


def _subtract_months(d: _dt.date, months: int) -> _dt.date:
    total_m = (d.year * 12 + (d.month - 1)) - months
    ny, nm = divmod(total_m, 12)
    nm += 1
    nd = min(d.day, calendar.monthrange(ny, nm)[1])
    return _dt.date(ny, nm, nd)


def _fmt_num(val: float) -> str:
    if float(val).is_integer():
        return str(int(val))
    return f"{val:.3g}" if abs(val) >= 100 else f"{val:.2f}".rstrip("0").rstrip(".")


def _normalize_ucum_point(pt: NumericPoint) -> Tuple[float, str]:
    raw_u = (pt.unit or "").strip()
    low_u = raw_u.lower()
    if low_u in {"g/l"}:
        return round(pt.value / 10.0, 4), "g/dL"
    if low_u in {"ug/dl", "mcg/dl"}:
        return pt.value, "ug/dL"
    if low_u in {"mg/dl"}:
        return pt.value, "mg/dL"
    if low_u in {"mmol/l"}:
        return pt.value, "mmol/L"
    if low_u in {"10*3/ul", "k/ul", "10^3/ul"}:
        return pt.value, "10*3/uL"
    if low_u in {"ml/min/1.73m2", "ml/min/{1.73_m2}", "ml/min/1.73_m2"}:
        return pt.value, "mL/min/1.73m2"
    return pt.value, raw_u


def _direction(first_v: float, last_v: float) -> str:
    delta = last_v - first_v
    scale = max(abs(first_v), abs(last_v), 1.0)
    if abs(delta) <= max(1e-6, 0.01 * scale):
        return "stable"
    return "increasing" if delta > 0 else "decreasing"


def _day_boundary(pts: Sequence[Tuple[NumericPoint, float]], *, pick_max: bool) -> Tuple[bool, List[Tuple[NumericPoint, float]]]:
    if len(pts) == 1:
        return True, list(pts)
    all_utc = all((p.timestamp or "").startswith("Z:") for p, _ in pts)
    all_naive = all((p.timestamp or "").startswith("N:") for p, _ in pts)
    if all_utc or all_naive:
        target_ts = (max if pick_max else min)(p.timestamp for p, _ in pts)
        sub = [(p, v) for p, v in pts if p.timestamp == target_ts]
        return len({v for _, v in sub}) == 1, sub
    if len({v for _, v in pts}) == 1:
        return True, list(pts)
    return False, list(pts)


def _resolve_bounds(raw_pts: Sequence[Tuple[NumericPoint, float]]) -> Tuple[bool, bool, List[Tuple[NumericPoint, float]], List[Tuple[NumericPoint, float]], str, str]:
    day_of = lambda p: _iso10(p.date) or ""
    has_utc = any((p.timestamp or "").startswith("Z:") for p, _ in raw_pts)
    has_non_utc = any(not (p.timestamp or "").startswith("Z:") for p, _ in raw_pts)
    if has_utc and has_non_utc:
        min_d, max_d = min(day_of(p) for p, _ in raw_pts), max(day_of(p) for p, _ in raw_pts)
        return False, len({v for _, v in raw_pts}) == 1, list(raw_pts), list(raw_pts), min_d, max_d
    if all((p.timestamp or "").startswith("Z:") for p, _ in raw_pts) or all((p.timestamp or "").startswith("N:") for p, _ in raw_pts):
        ordered = sorted(raw_pts, key=lambda pv: (pv[0].timestamp, day_of(pv[0])))
        min_ts, max_ts = ordered[0][0].timestamp, ordered[-1][0].timestamp
        fg = [(p, v) for p, v in ordered if p.timestamp == min_ts]
        lg = [(p, v) for p, v in ordered if p.timestamp == max_ts]
        fok, lok = len({v for _, v in fg}) == 1, len({v for _, v in lg}) == 1
        return (min_ts < max_ts) and fok and lok, lok, fg, lg, day_of(fg[0][0]), day_of(lg[-1][0])
    min_d, max_d = min(day_of(p) for p, _ in raw_pts), max(day_of(p) for p, _ in raw_pts)
    fok, fg = _day_boundary([(p, v) for p, v in raw_pts if day_of(p) == min_d], pick_max=False)
    lok, lg = _day_boundary([(p, v) for p, v in raw_pts if day_of(p) == max_d], pick_max=True)
    return (min_d < max_d) and fok and lok, lok, fg, lg, min_d, max_d


def _uuid_to_cit(citation_map: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k_str, entry in citation_map.items():
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("index") if isinstance(entry.get("index"), int) else (int(k_str) if str(k_str).isdigit() else 0)
        if idx <= 0:
            continue
        if entry.get("id"):
            out.setdefault(str(entry["id"]), idx)
        for m in entry.get("memberIds") or ():
            if m:
                out.setdefault(str(m), idx)
    return out


def _build_anchor_lines(
    extracted: ExtractedChart,
    cit_lookup: Mapping[str, int],
    question: str,
) -> List[str]:
    ref_iso = _iso10(extracted.reference_date)
    if not ref_iso:
        return []
    ref_dt = _dt.date.fromisoformat(ref_iso)
    cut_6m = _subtract_months(ref_dt, 6).isoformat()
    cut_12m = _subtract_months(ref_dt, 12).isoformat()

    lines = [
        f"Chart Anchor Date (latest realized clinical event): {ref_iso}",
        f"Lookback Windows: past 6 months >= {cut_6m} | past 12 months >= {cut_12m}",
    ]

    completed_visits: List[Tuple[str, ClinicalItem]] = []
    upcoming_visits: List[Tuple[str, ClinicalItem]] = []
    admin_records: List[Tuple[str, ClinicalItem]] = []

    for it in extracted.items:
        if it.section in {"patient", "manifest"}:
            continue
        d_iso = _iso10(it.clinical_date or it.date)
        if it.resource_type in {"Encounter", "Appointment"}:
            if it.date_kind == "clinical_event" and d_iso:
                completed_visits.append((d_iso, it))
            elif it.date_kind == "planned" and d_iso:
                upcoming_visits.append((d_iso, it))
        if it.date_kind == "administrative" and d_iso:
            admin_records.append((d_iso, it))

    last_visit_dt = ""
    if completed_visits:
        completed_visits.sort(key=lambda pair: (pair[0], pair[1].resource_uuid))
        last_visit_dt, last_it = completed_visits[-1]
        cit = f" [{cit_lookup[last_it.resource_uuid]}]" if last_it.resource_uuid in cit_lookup else ""
        src = f", {last_it.source_org}" if last_it.source_org else ""
        lines.append(f"Last Completed Clinical Visit: {last_visit_dt} ({last_it.display}{src}){cit}")
    else:
        lines.append("Last Completed Clinical Visit: none documented")

    if upcoming_visits:
        upcoming_visits.sort(key=lambda pair: (pair[0], pair[1].resource_uuid))
        up_parts = []
        for u_dt, u_it in upcoming_visits[:3]:
            cit = f" [{cit_lookup[u_it.resource_uuid]}]" if u_it.resource_uuid in cit_lookup else ""
            up_parts.append(f"{u_dt} ({u_it.display}){cit}")
        lines.append(f"Upcoming Scheduled Visits/Appointments: {'; '.join(up_parts)}")
    else:
        lines.append("Upcoming Scheduled Visits/Appointments: none documented")

    post_visit_admins = [
        (a_dt, a_it)
        for a_dt, a_it in admin_records
        if (last_visit_dt and a_dt > last_visit_dt) or a_it.resource_type == "Device"
    ]
    if post_visit_admins:
        post_visit_admins.sort(key=lambda pair: pair[0], reverse=True)
        warn_parts = []
        for a_dt, a_it in post_visit_admins[:2]:
            cit = f" [{cit_lookup[a_it.resource_uuid]}]" if a_it.resource_uuid in cit_lookup else ""
            warn_parts.append(f"{a_it.resource_type} ({a_it.display}) dated {a_dt}{cit}")
        lines.append(
            f"Administrative/Device Date Warning: {', '.join(warn_parts)} - administrative/device record date, NOT a clinical visit."
        )

    q_low = (question or "").lower()
    if any(cue in q_low for cue in _VACCINE_TEMPORAL_CUES):
        latest_imm: Dict[str, Tuple[str, ClinicalItem]] = {}
        for it in extracted.items:
            if it.section == "immunization" and it.status == "completed":
                i_dt = _iso10(it.clinical_date or it.date)
                if not i_dt:
                    continue
                k = (it.code or it.display).lower()
                prev = latest_imm.get(k)
                if prev is None or i_dt >= prev[0]:
                    latest_imm[k] = (i_dt, it)
        if latest_imm:
            imm_parts: List[str] = []
            for i_dt, i_it in sorted(latest_imm.values(), key=lambda x: x[0], reverse=True)[:5]:
                days = max(0, (ref_dt - _dt.date.fromisoformat(i_dt)).days)
                elapsed = f"{round(days / 365.25, 1)} years" if days >= 335 else f"{round(days / 30.44, 1)} months"
                cit = f" [{cit_lookup[i_it.resource_uuid]}]" if i_it.resource_uuid in cit_lookup else ""
                imm_parts.append(f"{i_it.display} last administered {i_dt}{cit} ({elapsed} before Chart Anchor {ref_iso})")
            lines.append(f"Vaccine Elapsed Time vs Chart Anchor: {'; '.join(imm_parts)}")

    return lines


def build_temporal_trends(
    extracted: ExtractedChart,
    condensed: CondensedChartResult,
    *,
    citation_map: Mapping[str, Any] = (),
    question: str = "",
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> TemporalTrendResult:
    """Build deterministic Chart Temporal Anchor + longitudinal lab/comparator summary."""
    cit_lookup = _uuid_to_cit(citation_map or {str(m["index"]): m for m in (condensed.mappings or []) if isinstance(m, Mapping)})
    q_terms = {t for t in re.findall(r"[a-z0-9]{3,}", (question or "").lower())}

    anchor_lines = _build_anchor_lines(extracted, cit_lookup, question)

    grouped: Dict[Tuple[str, str], List[Tuple[NumericPoint, float]]] = {}
    seen_pts: Set[Tuple[str, str, str, float]] = set()
    for pt in extracted.numeric_points:
        if (pt.status or "").lower() in _EXCLUDED_STATUSES:
            continue
        iso_dt = _iso10(pt.date)
        if not iso_dt:
            continue
        norm_v, norm_u = _normalize_ucum_point(pt)
        code_k = (pt.code or pt.concept or "observation").strip().lower()
        dedup = (pt.resource_uuid or "", code_k, pt.timestamp or iso_dt, norm_v)
        if dedup in seen_pts:
            continue
        seen_pts.add(dedup)
        grouped.setdefault((code_k, norm_u.lower()), []).append((pt, norm_v))

    series_rows: List[Tuple[Tuple[int, int, str], str]] = []
    multi_count = 0
    for (_ck, _uk), pts in grouped.items():
        concept = sorted({(p.concept or p.code or "Observation").strip() for p, _ in pts})[-1]
        unit_lbl = sorted({_normalize_ucum_point(p)[1] for p, _ in pts})[-1]
        u_sp = f" {unit_lbl}" if unit_lbl else ""
        overlap = sum(1 for t in q_terms if t in concept.lower())

        if len(pts) == 1:
            p0, v0 = pts[0]
            dt0 = _iso10(p0.date) or ""
            cit = f" [{cit_lookup[p0.resource_uuid]}]" if p0.resource_uuid in cit_lookup else ""
            line = f"- {concept}: {_fmt_num(v0)}{u_sp} on {dt0}{cit} (n=1, single point - trend not supported)"
            series_rows.append(((-overlap, 1, concept.lower()), line))
        else:
            ok, latest_known, fg, lg, fdt, ldt = _resolve_bounds(pts)
            vals = [v for _, v in pts]
            vmin, vmax = min(vals), max(vals)
            cites = "".join(f"[{i}]" for i in sorted({cit_lookup[p.resource_uuid] for p, _ in lg if p.resource_uuid in cit_lookup}))
            if ok:
                multi_count += 1
                fv, lv = fg[0][1], lg[0][1]
                delta = lv - fv
                sign = "+" if delta > 0 else ""
                line = f"- {concept}: latest {_fmt_num(lv)}{u_sp} on {ldt}{cites} | window {fdt}..{ldt} (n={len(pts)}, {_direction(fv, lv)} {sign}{_fmt_num(delta)}; range {_fmt_num(vmin)}..{_fmt_num(vmax)})"
            else:
                hval = f"latest {_fmt_num(lg[0][1])}{u_sp}" if latest_known else f"range {_fmt_num(vmin)}..{_fmt_num(vmax)}{u_sp}"
                line = f"- {concept}: {hval} on {ldt}{cites} | window {fdt}..{ldt} (n={len(pts)}, tie-safe boundary - trend indeterminate)"
            series_rows.append(((-overlap, 0, concept.lower()), line))

    for it in extracted.items:
        cmp_op = str((it.metadata or {}).get("comparator") or "").strip()
        if cmp_op and it.section in {"lab_observation", "vital_sign", "observation"}:
            cit = f" [{cit_lookup[it.resource_uuid]}]" if it.resource_uuid in cit_lookup else ""
            val_s = str((it.metadata or {}).get("value") or "")
            dt_s = it.date or "undated"
            ov = sum(1 for t in q_terms if t in (it.display or "").lower())
            series_rows.append(((-ov, 0, (it.display or "").lower()), f"- {it.display}: {val_s} on {dt_s}{cit} (comparator threshold)"))

    series_rows.sort(key=lambda pair: pair[0])
    heading = f"=== CHART TEMPORAL ANCHOR & LONGITUDINAL QUANTITATIVE TRENDS (anchor: {extracted.reference_date or 'undated'}) ==="
    chosen = [*anchor_lines, *[line for _, line in series_rows[:8]]]
    counter = token_counter or estimate_tokens_local
    text = "\n".join([heading, *chosen]) if chosen else ""
    if max_tokens and max_tokens > 0:
        while chosen and counter("\n".join([heading, *chosen])) > max_tokens:
            chosen.pop()
        text = "\n".join([heading, *chosen]) if chosen else ""

    return TemporalTrendResult(
        reference_date=extracted.reference_date,
        digest_text=text,
        digest_tokens=counter(text) if text else 0,
        summary={
            "enabled": bool(text),
            "has_anchor": bool(anchor_lines),
            "numeric_series_count": len(grouped),
            "multi_point_series_count": multi_count,
            "lines_rendered": len(chosen),
        },
    )


_BASE_TEMPORAL_DISCLOSURE = EnrichmentDisclosure(
    enricher_id="temporal",
    version="0.2.0",
    kind="quantitative_summary",
    scope_summary=(
        "Emits Chart Anchor Date, 6m/12m lookback cutoffs, visit vs administrative date guards, "
        "vaccine elapsed time, and tie-safe UCUM longitudinal lab deltas."
    ),
    limitations="Deterministic calendar and extracted NumericPoint math only; skips free-text-only values and cross-analyte conversions.",
)


class QuantitativeTrendEnricher:
    """Chart Temporal Anchor & tie-safe quantitative trend calculator (`ContextEnricher`)."""

    def __init__(self, enricher_id: str = "temporal") -> None:
        self.enricher_id = enricher_id
        self.disclosure = replace(_BASE_TEMPORAL_DISCLOSURE, enricher_id=enricher_id)

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        if inp.max_tokens < 48:
            return None
        if not inp.extracted.reference_date and not inp.extracted.numeric_points:
            return None

        res = build_temporal_trends(
            inp.extracted,
            inp.condensed,
            citation_map=inp.citation_map,
            question=inp.question,
            max_tokens=inp.max_tokens,
            token_counter=inp.token_counter,
        )
        if not res.digest_text:
            return None
        return EnrichmentOutput(
            enricher_id=self.enricher_id,
            heading=res.digest_text.splitlines()[0],
            prompt_block=res.digest_text,
            token_count=res.digest_tokens,
            disclosure=self.disclosure,
            new_citations={},
            metadata=res.summary,
        )
