"""Tie-safe quantitative trend digest enrichment (`ContextEnricher`)."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from wahs_core.condenser import CondensedChartResult, estimate_tokens_local
from wahs_core.extractors import ExtractedChart, NumericPoint
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_EXCLUDED_STATUSES: Set[str] = {"cancelled", "entered-in-error", "noshow"}
_LONGITUDINAL_CUES = (
    "trend", "over time", "history", "change", "changed", "increase", "decrease",
    "worse", "better", "improve", "progression", "serial", "trajectory", "compare",
    "last year", "past year", "6 months", "12 months",
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


def _fmt_num(val: float) -> str:
    if float(val).is_integer():
        return str(int(val))
    return f"{val:.3g}" if abs(val) >= 100 else f"{val:.2f}".rstrip("0").rstrip(".")


def _normalize_ucum_point(pt: NumericPoint) -> Tuple[float, str]:
    """Normalize closely related UCUM synonyms without cross-guessing molecular weight."""
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
    if low_u in {"ml/min/1.73m2", "ml/min/{1.73_m2}"}:
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


def build_temporal_trends(
    extracted: ExtractedChart,
    condensed: CondensedChartResult,
    *,
    citation_map: Mapping[str, Any] = (),
    question: str = "",
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> TemporalTrendResult:
    """Build deterministic longitudinal lab/comparator summary with strict `n == 1` & tie guards."""
    cit_lookup = _uuid_to_cit(citation_map or {str(m["index"]): m for m in (condensed.mappings or []) if isinstance(m, Mapping)})
    q_terms = {t for t in re.findall(r"[a-z0-9]{3,}", (question or "").lower())}

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

    # Surface `<`/`>` comparator observations explicitly so threshold tests are never lost
    for it in extracted.items:
        cmp_op = str((it.metadata or {}).get("comparator") or "").strip()
        if cmp_op and it.section in {"lab_observation", "vital_sign", "observation"}:
            cit = f" [{cit_lookup[it.resource_uuid]}]" if it.resource_uuid in cit_lookup else ""
            val_s = str((it.metadata or {}).get("value") or "")
            dt_s = it.date or "undated"
            ov = sum(1 for t in q_terms if t in (it.display or "").lower())
            series_rows.append(((-ov, 0, (it.display or "").lower()), f"- {it.display}: {val_s} on {dt_s}{cit} (comparator threshold)"))

    series_rows.sort(key=lambda pair: pair[0])
    heading = f"=== LONGITUDINAL QUANTITATIVE TRENDS (anchor: {extracted.reference_date or 'undated'}) ==="
    chosen = [line for _, line in series_rows[:8]]
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
        summary={"enabled": bool(text), "numeric_series_count": len(grouped), "multi_point_series_count": multi_count, "lines_rendered": len(chosen)},
    )


class QuantitativeTrendEnricher:
    """Tie-safe quantitative trend calculator (`ContextEnricher`)."""

    enricher_id = "temporal_trends"
    disclosure = EnrichmentDisclosure(
        enricher_id="temporal_trends",
        version="0.1.0",
        kind="quantitative_summary",
        scope_summary="Computes longitudinal deltas, ranges, and comparator summaries from extracted structured NumericPoints when longitudinal queries are asked.",
        limitations="Triggers only when longitudinal cues appear or >=2-point numeric series exist; skips free-text-only values and cross-analyte conversions.",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        if inp.max_tokens < 48 or not inp.extracted.numeric_points:
            return None
        q_low = (inp.question or "").lower()
        has_long_cue = any(cue in q_low for cue in _LONGITUDINAL_CUES)
        by_code: Dict[str, int] = {}
        for pt in inp.extracted.numeric_points:
            by_code[pt.code or pt.concept] = by_code.get(pt.code or pt.concept, 0) + 1
        if not has_long_cue and not any(c >= 2 for c in by_code.values()):
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
