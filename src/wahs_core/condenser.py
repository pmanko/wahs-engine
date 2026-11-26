"""3-Tier Verbatim-First Monotonic Structural FHIR Chart Condenser (`wahs_core`).

Zero hardcoded budget ceilings, zero static per-section item counts, and zero static
note character cutoffs:
- Tier 0 (Verbatim Complete Ledger): Whenever `target_tokens is None` or all extracted
  clinical items and full decoded clinical notes fit within `target_tokens`, emits 100%
  verbatim lines and full notes (`folded_items_count=0`, `omitted_clinical_items_count=0`).
- Tier 1 (Uniform Ontology-Free RLE Folded Ledger): When Tier 0 exceeds `target_tokens`,
  uniformly run-length folds repeated historical items sharing the same FHIR
  `(section, code, unit/status)` key into the latest record + count + date span + numeric
  range + merged `member_uuids`, keeping the HL7 IPS Safety Core, acute encounters, and
  full clinical notes verbatim.
- Tier 2 (Budget-Packed Ledger with Continuous Note Water-Filling): Preserves the HL7 IPS
  Safety Core intact (stripping verbose qualifiers or collapsing by section only when the
  IPS core itself exceeds budget), binary-searches a budget-independent section-interleaved
  recency prefix using minimal note headers so UUID coverage is strictly monotonic in
  `target_tokens`, and binary-searches a continuous character horizon across included notes
  so short notes remain full text and longer notes expand dynamically to fill remaining
  capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .chart_serializer import render_chart
from .extractors import ClinicalItem, ExtractedChart, extract_bundle


@dataclass(frozen=True)
class CondensedChartResult:
    tier: int
    tier_name: str
    target_tokens: Optional[int]
    actual_tokens: int
    chart_text: str
    records: List[Dict[str, Any]]
    mappings: List[Dict[str, Any]]
    completeness_manifest: Dict[str, Any]
    compression_stats: Dict[str, Any]


def estimate_tokens_local(text: str) -> int:
    """Deterministic pure-stdlib token estimator calibrated to Gemma 4 / BPE tokenizers."""
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


def _fold_group_by_key(
    items: Sequence[ClinicalItem],
    key_fn: Callable[[ClinicalItem], Tuple[Any, ...]],
    text_fn: Callable[[ClinicalItem, List[ClinicalItem], str], str],
    *,
    section: Optional[str] = None,
) -> Tuple[List[ClinicalItem], int]:
    """Uniformly run-length fold repeated items sharing `key_fn`, returning `(items, folded_count)`."""
    if not items:
        return [], 0

    groups: Dict[Tuple[Any, ...], List[ClinicalItem]] = {}
    for it in items:
        groups.setdefault(key_fn(it), []).append(it)

    out: List[ClinicalItem] = []
    folded_count = 0
    for group in groups.values():
        chron = sorted(group, key=lambda x: (_date_sort_key(x.date), x.resource_uuid))
        latest = chron[-1]
        if len(chron) == 1:
            out.append(latest)
            continue
        folded_count += len(chron) - 1
        latest_date, span_str = _span_phrase([x.date for x in chron])
        eff_date = latest_date or latest.date
        out.append(
            ClinicalItem(
                resource_type=latest.resource_type,
                resource_uuid=latest.resource_uuid,
                date=eff_date,
                clinical_date=eff_date if latest.date_kind == "clinical_event" else None,
                date_kind="clinical_event" if (eff_date and latest.date_kind == "clinical_event") else latest.date_kind,
                section=section or latest.section,
                code=latest.code,
                display=latest.display,
                text=text_fn(latest, chron, span_str),
                mandatory=any(x.mandatory for x in chron),
                status=latest.status,
                source_org=latest.source_org,
                member_uuids=_merge_uuids(chron),
                metadata=dict(latest.metadata),
            )
        )
    out.sort(key=lambda x: (_date_sort_key(x.date), x.resource_uuid), reverse=True)
    return out, folded_count


def _fold_observations_rle(items: Sequence[ClinicalItem]) -> Tuple[List[ClinicalItem], int]:
    """Fold repeated identical `(section, code, unit)` observations ontology-free."""

    def _obs_text(latest: ClinicalItem, chron: List[ClinicalItem], span: str) -> str:
        nums = [
            it.metadata["numeric"]
            for it in chron
            if isinstance(it.metadata.get("numeric"), (int, float))
        ]
        unit = str(latest.metadata.get("unit") or "").strip()
        u_sp = f" {unit}" if unit else ""
        range_part = ""
        if len(nums) >= 2:
            lo_v = min(nums)
            hi_v = max(nums)
            fmt_lo = str(int(lo_v)) if float(lo_v).is_integer() else f"{lo_v:.4g}"
            fmt_hi = str(int(hi_v)) if float(hi_v).is_integer() else f"{hi_v:.4g}"
            range_part = f", range {fmt_lo}..{fmt_hi}{u_sp}"
        span_part = f", {span}" if span else ""
        return f"{latest.text} ({len(chron)} measurements{span_part}{range_part})"

    return _fold_group_by_key(
        items,
        key_fn=lambda x: (x.section, (x.code or x.display).lower(), str(x.metadata.get("unit") or "").lower()),
        text_fn=_obs_text,
    )


def _fold_panels_rle(items: Sequence[ClinicalItem]) -> Tuple[List[ClinicalItem], int]:
    """Uniformly fold repeated diagnostic/survey panels sharing `(code or display)`."""

    return _fold_group_by_key(
        items,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status),
        text_fn=lambda latest, chron, span: f"{latest.text} ({len(chron)} panels{', ' + span if span else ''})",
    )


def _build_tier1_items(extracted: ExtractedChart) -> Tuple[List[ClinicalItem], List[ClinicalItem], int]:
    """Return `(mandatory_ips, optional_rle_items, folded_items_count)` uniformly without heuristics."""
    src = extracted.items
    patient = [x for x in src if x.section == "patient"]
    allergies = [x for x in src if x.section == "allergy" and x.mandatory]
    allergies_inactive = [x for x in src if x.section == "allergy" and not x.mandatory]
    cond_active = [x for x in src if x.section == "condition_active"]
    cond_res = [x for x in src if x.section == "condition_resolved"]
    med_active = [x for x in src if x.section == "med_active"]
    med_hist = [x for x in src if x.section == "med_historical"]
    imms = [x for x in src if x.section == "immunization"]
    devices_active = [x for x in src if x.section == "device" and x.mandatory]
    devices_inactive = [x for x in src if x.section == "device" and not x.mandatory]
    enc_acuity = sorted([x for x in src if x.section == "encounter_acuity"], key=lambda x: _date_sort_key(x.date), reverse=True)
    enc_routine = [x for x in src if x.section == "encounter_routine"]
    careplans_active = sorted([x for x in src if x.section == "careplan_active"], key=lambda x: _date_sort_key(x.date), reverse=True)
    careplans_hist = [x for x in src if x.section == "careplan_historical"]
    procs = [x for x in src if x.section == "procedure"]
    lab_panels = [x for x in src if x.section == "lab_panel"]
    survey_panels = [x for x in src if x.section == "survey_panel"]
    vitals = [x for x in src if x.section == "vital_sign"]
    labs = [x for x in src if x.section == "lab_observation"]
    social = [x for x in src if x.section == "social_history"]
    surveys = [x for x in src if x.section == "survey_observation"]
    imaging = sorted([x for x in src if x.section == "imaging"], key=lambda x: _date_sort_key(x.date), reverse=True)
    other_obs = [x for x in src if x.section == "observation"]
    notes = sorted([x for x in src if x.section == "clinical_note"], key=lambda x: _date_sort_key(x.date), reverse=True)

    folded_total = 0
    f_imms, fc = _fold_group_by_key(
        imms,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status),
        text_fn=lambda lat, grp, sp: f"Immunization: {lat.display} ({len(grp)} doses{', ' + sp if sp else ''}, status: {lat.status})",
    )
    folded_total += fc
    f_med_hist, fc = _fold_group_by_key(
        med_hist,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status, str(x.metadata.get("setting") or "")),
        text_fn=lambda lat, grp, sp: f"Medication History: {lat.display} ({len(grp)} records{', ' + sp if sp else ''}, status: {lat.status})",
    )
    folded_total += fc
    f_cond_res, fc = _fold_group_by_key(
        cond_res,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status),
        text_fn=lambda lat, grp, sp: f"Condition History: {lat.display} ({len(grp)} records{', ' + sp if sp else ''}, status: {lat.status})",
    )
    folded_total += fc
    f_enc_routine, fc = _fold_group_by_key(
        enc_routine,
        key_fn=lambda x: ((x.display or x.code).lower(), x.code),
        text_fn=lambda lat, grp, sp: f"Encounter History: {lat.display} ({len(grp)} visits{', ' + sp if sp else ''}, class: {lat.code})",
    )
    folded_total += fc
    f_procs, fc = _fold_group_by_key(
        procs,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status),
        text_fn=lambda lat, grp, sp: f"Procedure History: {lat.display} ({len(grp)} records{', ' + sp if sp else ''})",
    )
    folded_total += fc
    f_lab_panels, fc = _fold_panels_rle(lab_panels)
    folded_total += fc
    f_survey_panels, fc = _fold_panels_rle(survey_panels)
    folded_total += fc
    f_labs, fc = _fold_observations_rle(labs)
    folded_total += fc
    f_vitals, fc = _fold_observations_rle(vitals)
    folded_total += fc
    f_social, fc = _fold_observations_rle(social)
    folded_total += fc
    f_surveys, fc = _fold_observations_rle(surveys)
    folded_total += fc
    f_other, fc = _fold_observations_rle(other_obs)
    folded_total += fc
    f_dev_inact, fc = _fold_group_by_key(
        devices_inactive,
        key_fn=lambda x: ((x.code or x.display).lower(), x.status),
        text_fn=lambda lat, grp, sp: f"Device History: {lat.display} ({len(grp)} records{', ' + sp if sp else ''}, status: {lat.status})",
    )
    folded_total += fc

    mandatory_ips = patient + allergies + cond_active + med_active + devices_active + f_imms
    optional_rle = (
        notes
        + enc_acuity
        + careplans_active
        + imaging
        + f_lab_panels
        + f_vitals
        + f_labs
        + f_med_hist
        + f_cond_res
        + f_procs
        + allergies_inactive
        + f_social
        + f_survey_panels
        + f_surveys
        + f_other
        + careplans_hist
        + f_dev_inact
        + f_enc_routine
    )
    return mandatory_ips, optional_rle, folded_total


def _strip_ips_code_qualifiers(mandatory_items: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Pass A IPS compaction: strip verbose coding system suffixes while retaining 1 line per item."""
    out: List[ClinicalItem] = []
    for it in mandatory_items:
        if it.section == "patient":
            out.append(it)
            continue
        short = re.sub(r",\s*(?:RxNorm|SNOMED|CVX):\s*[^,)]+", "", it.text)
        out.append(
            ClinicalItem(
                resource_type=it.resource_type,
                resource_uuid=it.resource_uuid,
                date=it.date,
                clinical_date=it.clinical_date,
                date_kind=it.date_kind,
                section=it.section,
                code=it.code,
                display=it.display,
                text=short,
                mandatory=True,
                status=it.status,
                source_org=it.source_org,
                member_uuids=it.member_uuids,
            )
        )
    return out


def _collapse_ips_by_section(mandatory_items: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Pass B IPS compaction: collapse each IPS section into a single summary preserving all UUIDs."""
    out: List[ClinicalItem] = []
    grouped_by_sec: Dict[str, List[ClinicalItem]] = {}
    for it in mandatory_items:
        if it.section == "patient":
            out.append(it)
        else:
            grouped_by_sec.setdefault(it.section, []).append(it)

    for sec in ("allergy", "condition_active", "med_active", "device", "immunization"):
        sec_items = grouped_by_sec.get(sec, [])
        if not sec_items:
            continue
        if len(sec_items) == 1:
            out.append(sec_items[0])
            continue
        latest_d, _ = _span_phrase([x.date for x in sec_items])
        labels = "; ".join(
            f"{x.display}" + (f" ({x.date})" if x.date else "")
            for x in sec_items
        )
        lead = sec_items[0]
        out.append(
            ClinicalItem(
                resource_type=lead.resource_type,
                resource_uuid=lead.resource_uuid,
                date=latest_d or lead.date,
                clinical_date=latest_d or lead.clinical_date,
                date_kind="clinical_event" if (latest_d or lead.date) else lead.date_kind,
                section=sec,
                code=f"{sec}-ips-core",
                display=sec,
                text=f"IPS {sec}: {labels}",
                mandatory=True,
                status=lead.status,
                member_uuids=_merge_uuids(sec_items),
            )
        )
    return out


def _render_note_at_horizon(note: ClinicalItem, char_cap: int) -> Tuple[ClinicalItem, bool]:
    """Render a clinical note up to `char_cap` characters without any fixed static cutoff."""
    full = str(note.metadata.get("full_text") or note.text or "").strip()
    title = str(note.metadata.get("title") or note.display or "Clinical Note")
    if char_cap >= len(full):
        return note, False
    stub_suffix = f"[... excerpted; {len(full)} chars total]"
    head = full[: max(0, char_cap)].rstrip()
    body = f"{head} {stub_suffix}" if head else stub_suffix
    excerpt_txt = f"Note ({title}): {body}"
    if len(excerpt_txt) >= len(note.text):
        return note, False
    return (
        ClinicalItem(
            resource_type=note.resource_type,
            resource_uuid=note.resource_uuid,
            date=note.date,
            clinical_date=note.clinical_date,
            date_kind=note.date_kind,
            section=note.section,
            code=note.code,
            display=note.display,
            text=excerpt_txt,
            mandatory=note.mandatory,
            status=note.status,
            source_org=note.source_org,
            member_uuids=note.member_uuids,
            metadata=note.metadata,
        ),
        True,
    )


def _rank_optional_balanced(optional_items: Sequence[ClinicalItem]) -> List[ClinicalItem]:
    """Budget-independent round-robin section-interleaved recency ordering (guarantees monotonicity)."""
    section_order = (
        "clinical_note", "encounter_acuity", "careplan_active", "imaging",
        "lab_panel", "vital_sign", "lab_observation", "med_historical",
        "condition_resolved", "procedure", "allergy", "social_history",
        "survey_panel", "survey_observation", "observation", "careplan_historical",
        "device", "encounter_routine",
    )
    by_sec: Dict[str, List[ClinicalItem]] = {}
    for it in optional_items:
        by_sec.setdefault(it.section, []).append(it)
    for sec_list in by_sec.values():
        sec_list.sort(key=lambda x: (_int_date(x.date), x.resource_uuid), reverse=True)

    max_len = max((len(v) for v in by_sec.values()), default=0)
    ordered: List[ClinicalItem] = []
    remaining_secs = [s for s in section_order if s in by_sec] + [
        s for s in sorted(by_sec) if s not in section_order
    ]
    for round_idx in range(max_len):
        for sec in remaining_secs:
            sec_list = by_sec[sec]
            if round_idx < len(sec_list):
                ordered.append(sec_list[round_idx])
    return ordered


def items_to_querystore_records(items: Sequence[ClinicalItem]) -> List[Dict[str, Any]]:
    """Convert `ClinicalItem` objects into `chart_serializer.render_chart` input dicts."""
    records: List[Dict[str, Any]] = []
    for it in items:
        meta = dict(it.metadata)
        if it.mandatory:
            meta["mandatory_context"] = True
        if it.member_uuids:
            meta["memberUuids"] = list(it.member_uuids)
        if it.source_org:
            meta["source_org"] = it.source_org
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


def _build_completeness_manifest(
    extracted: ExtractedChart,
    *,
    tier: int,
    tier_name: str,
    included_items: Sequence[ClinicalItem],
    folded_items_count: int,
    excerpted_notes_count: int,
) -> Tuple[ClinicalItem, Dict[str, Any]]:
    covered_uuids: Set[str] = set()
    for it in included_items:
        if it.resource_uuid:
            covered_uuids.add(it.resource_uuid)
        covered_uuids.update(u for u in it.member_uuids if u)

    omitted_by_section: Dict[str, int] = {}
    included_clinical_count = 0
    total_clinical_count = 0
    total_notes = 0
    included_notes = 0
    dates: List[str] = []
    sources: Set[str] = set()

    for src_it in extracted.items:
        if src_it.section == "patient":
            continue
        total_clinical_count += 1
        if src_it.date and re.match(r"^\d{4}-\d{2}-\d{2}$", src_it.date):
            dates.append(src_it.date)
        if src_it.source_org:
            sources.add(src_it.source_org)
        if src_it.section == "clinical_note":
            total_notes += 1
        item_uuids = {src_it.resource_uuid} | set(src_it.member_uuids)
        if item_uuids & covered_uuids:
            included_clinical_count += 1
            if src_it.section == "clinical_note":
                included_notes += 1
        else:
            omitted_by_section[src_it.section] = omitted_by_section.get(src_it.section, 0) + 1

    omitted_clinical_count = sum(omitted_by_section.values())
    omitted_notes_count = max(0, total_notes - included_notes)
    full_notes_count = max(0, included_notes - excerpted_notes_count)
    date_span = f"{min(dates)}..{max(dates)}" if dates else ""

    if tier == 0:
        coverage = "ALL extracted clinical records and notes emitted verbatim"
    elif tier == 1:
        coverage = (
            "IPS Safety Core and full notes COMPLETE; repeated historical items run-length folded with full date spans and UUIDs"
        )
    else:
        coverage = (
            f"IPS Safety Core COMPLETE; budget-packed ({omitted_clinical_count} historical item(s) omitted, "
            f"{excerpted_notes_count} note(s) excerpted)"
        )

    parts = [
        f"Completeness Manifest: Tier {tier} ({tier_name})",
        coverage,
        f"clinical_items={included_clinical_count}/{total_clinical_count}",
    ]
    if folded_items_count > 0:
        parts.append(f"folded_items={folded_items_count}")
    if omitted_clinical_count > 0:
        parts.append(f"omitted_clinical_items={omitted_clinical_count}")
    if excerpted_notes_count > 0:
        parts.append(f"excerpted_notes={excerpted_notes_count}")
    if omitted_notes_count > 0:
        parts.append(f"omitted_notes={omitted_notes_count}")
    if date_span:
        parts.append(f"date_span={date_span}")

    manifest_item = ClinicalItem(
        resource_type="Bundle",
        resource_uuid=f"manifest-{extracted.patient_id or 'chart'}",
        date=extracted.reference_date,
        clinical_date=None,
        date_kind="administrative",
        section="manifest",
        code="completeness-manifest",
        display="Completeness Manifest",
        text=" | ".join(parts),
        mandatory=True,
        status="current",
    )
    manifest_dict: Dict[str, Any] = {
        "tier": tier,
        "tier_name": tier_name,
        "coverage": coverage,
        "total_bundle_entries": extracted.total_bundle_entries,
        "total_clinical_items": total_clinical_count,
        "included_clinical_items": included_clinical_count,
        "folded_items_count": folded_items_count,
        "omitted_clinical_items_count": omitted_clinical_count,
        "total_notes": total_notes,
        "full_notes_count": full_notes_count,
        "excerpted_notes_count": excerpted_notes_count,
        "omitted_notes_count": omitted_notes_count,
        "date_span": date_span,
        "sources": sorted(sources),
        "deduplicated_note_count": extracted.deduplicated_note_count,
        "excluded_non_clinical": dict(extracted.excluded_resource_counts),
        "omitted_by_section": omitted_by_section,
        "rendered_records": len(included_items) + 1,
    }
    return manifest_item, manifest_dict


def _assemble_render(
    extracted: ExtractedChart,
    body_items: Sequence[ClinicalItem],
    *,
    tier: int,
    tier_name: str,
    folded_items_count: int,
    excerpted_notes_count: int,
    counter: Callable[[str], int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, List[Dict[str, Any]], int]:
    manifest_item, manifest_dict = _build_completeness_manifest(
        extracted,
        tier=tier,
        tier_name=tier_name,
        included_items=body_items,
        folded_items_count=folded_items_count,
        excerpted_notes_count=excerpted_notes_count,
    )
    if body_items and body_items[0].section == "patient":
        ordered = [body_items[0], manifest_item] + list(body_items[1:])
    else:
        ordered = [manifest_item] + list(body_items)
    recs = items_to_querystore_records(ordered)
    chart_text, mappings = render_chart(recs)
    return recs, manifest_dict, chart_text, mappings, counter(chart_text)


def _make_result(
    *,
    extracted: ExtractedChart,
    tier: int,
    target_tokens: Optional[int],
    actual_tokens: int,
    chart_text: str,
    recs: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> CondensedChartResult:
    util = round(actual_tokens / target_tokens, 3) if target_tokens else 1.0
    return CondensedChartResult(
        tier=tier,
        tier_name=manifest["tier_name"],
        target_tokens=target_tokens,
        actual_tokens=actual_tokens,
        chart_text=chart_text,
        records=recs,
        mappings=mappings,
        completeness_manifest=manifest,
        compression_stats={
            "bundle_entries": extracted.total_bundle_entries,
            "extracted_clinical_items": len(extracted.items),
            "rendered_records": len(recs),
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "budget_utilization": util,
        },
    )


def condense_bundle_to_budget(
    bundle: Mapping[str, Any] | ExtractedChart,
    *,
    target_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> CondensedChartResult:
    """Condense a FHIR R4 Bundle (or `ExtractedChart`) via 3-tier verbatim-first condensation."""
    if target_tokens is not None and (
        not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens <= 0
    ):
        raise ValueError("target_tokens must be a positive integer or None")

    counter = token_counter or estimate_tokens_local
    extracted = bundle if isinstance(bundle, ExtractedChart) else extract_bundle(bundle)

    # Tier 0: 100% Verbatim Complete Ledger (patient first, then newest -> oldest items & full notes)
    t0_patient = [x for x in extracted.items if x.section == "patient"]
    t0_ips = [x for x in extracted.items if x.section != "patient" and x.mandatory]
    t0_other = sorted(
        [x for x in extracted.items if x.section != "patient" and not x.mandatory],
        key=lambda x: (_date_sort_key(x.date), x.resource_uuid),
        reverse=True,
    )
    recs0, man0, text0, maps0, toks0 = _assemble_render(
        extracted,
        t0_patient + t0_ips + t0_other,
        tier=0,
        tier_name="Verbatim Complete Ledger",
        folded_items_count=0,
        excerpted_notes_count=0,
        counter=counter,
    )
    if target_tokens is None or toks0 <= target_tokens:
        return _make_result(
            extracted=extracted, tier=0, target_tokens=target_tokens, actual_tokens=toks0,
            chart_text=text0, recs=recs0, mappings=maps0, manifest=man0,
        )

    # Tier 1: Uniform Ontology-Free RLE Folded Ledger (full notes + IPS core preserved verbatim)
    mandatory_ips, optional_rle, folded_count = _build_tier1_items(extracted)
    opt_sorted_t1 = sorted(optional_rle, key=lambda x: (_date_sort_key(x.date), x.resource_uuid), reverse=True)
    recs1, man1, text1, maps1, toks1 = _assemble_render(
        extracted,
        mandatory_ips + opt_sorted_t1,
        tier=1,
        tier_name="Ontology-Free RLE Folded Ledger",
        folded_items_count=folded_count,
        excerpted_notes_count=0,
        counter=counter,
    )
    if toks1 <= target_tokens:
        return _make_result(
            extracted=extracted, tier=1, target_tokens=target_tokens, actual_tokens=toks1,
            chart_text=text1, recs=recs1, mappings=maps1, manifest=man1,
        )

    # Tier 2: Budget-Packed Ledger with monotonic UUID prefix & continuous note water-filling
    active_ips = list(mandatory_ips)
    _, _, _, _, ips_toks = _assemble_render(
        extracted, active_ips, tier=2, tier_name="Budget-Packed Ledger",
        folded_items_count=folded_count, excerpted_notes_count=0, counter=counter,
    )
    if ips_toks > target_tokens:
        cand_ips = _strip_ips_code_qualifiers(mandatory_ips)
        _, _, _, _, toks_a = _assemble_render(
            extracted, cand_ips, tier=2, tier_name="Budget-Packed Ledger",
            folded_items_count=folded_count, excerpted_notes_count=0, counter=counter,
        )
        if toks_a <= target_tokens:
            active_ips = cand_ips
        else:
            collapsed_ips = _collapse_ips_by_section(cand_ips)
            recs_b, man_b, text_b, maps_b, toks_b = _assemble_render(
                extracted, collapsed_ips, tier=2, tier_name="Budget-Packed Ledger",
                folded_items_count=folded_count, excerpted_notes_count=0, counter=counter,
            )
            if toks_b > target_tokens:
                raise ValueError(
                    f"target_tokens ({target_tokens}) cannot fit mandatory IPS safety core ({toks_b} tokens required)"
                )
            return _make_result(
                extracted=extracted, tier=2, target_tokens=target_tokens, actual_tokens=toks_b,
                chart_text=text_b, recs=recs_b, mappings=maps_b, manifest=man_b,
            )

    full_note_by_uuid = {
        it.resource_uuid: it
        for it in optional_rle
        if it.section == "clinical_note" and it.resource_uuid
    }
    optional_stubs: List[ClinicalItem] = []
    stub_excerpted_uuids: Set[str] = set()
    for it in optional_rle:
        if it.section == "clinical_note":
            stub_it, did_ex = _render_note_at_horizon(it, 0)
            optional_stubs.append(stub_it)
            if did_ex and it.resource_uuid:
                stub_excerpted_uuids.add(it.resource_uuid)
        else:
            optional_stubs.append(it)

    ranked_optional = _rank_optional_balanced(optional_stubs)

    def _fit_prefix(k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, List[Dict[str, Any]], int]:
        chosen = sorted(
            ranked_optional[:k],
            key=lambda x: (_date_sort_key(x.date), x.resource_uuid),
            reverse=True,
        )
        ex_cnt = sum(1 for x in chosen if x.resource_uuid in stub_excerpted_uuids)
        return _assemble_render(
            extracted,
            active_ips + chosen,
            tier=2,
            tier_name="Budget-Packed Ledger",
            folded_items_count=folded_count,
            excerpted_notes_count=ex_cnt,
            counter=counter,
        )

    lo, hi = 0, len(ranked_optional)
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fit_prefix(mid)[4] <= target_tokens:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1
    while best_k < len(ranked_optional) and _fit_prefix(best_k + 1)[4] <= target_tokens:
        best_k += 1
    while best_k > 0 and _fit_prefix(best_k)[4] > target_tokens:
        best_k -= 1

    accepted_opt = list(ranked_optional[:best_k])
    included_note_positions = [
        idx for idx, x in enumerate(accepted_opt) if x.resource_uuid and x.resource_uuid in full_note_by_uuid
    ]

    if included_note_positions:
        max_note_chars = max(
            len(
                str(
                    full_note_by_uuid[accepted_opt[idx].resource_uuid].metadata.get("full_text")
                    or full_note_by_uuid[accepted_opt[idx].resource_uuid].text
                    or ""
                ).strip()
            )
            for idx in included_note_positions
        )

        def _render_notes_at_cap(cap_chars: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, List[Dict[str, Any]], int]:
            cand = list(accepted_opt)
            ex_cnt = 0
            for idx in included_note_positions:
                uid = accepted_opt[idx].resource_uuid
                rendered_note, did_ex = _render_note_at_horizon(full_note_by_uuid[uid], cap_chars)
                cand[idx] = rendered_note
                if did_ex:
                    ex_cnt += 1
            cand_sorted = sorted(cand, key=lambda x: (_date_sort_key(x.date), x.resource_uuid), reverse=True)
            return _assemble_render(
                extracted,
                active_ips + cand_sorted,
                tier=2,
                tier_name="Budget-Packed Ledger",
                folded_items_count=folded_count,
                excerpted_notes_count=ex_cnt,
                counter=counter,
            )

        recs_full, man_full, text_full, maps_full, toks_full = _render_notes_at_cap(max_note_chars)
        if toks_full <= target_tokens:
            return _make_result(
                extracted=extracted, tier=2, target_tokens=target_tokens, actual_tokens=toks_full,
                chart_text=text_full, recs=recs_full, mappings=maps_full, manifest=man_full,
            )

        lo_c, hi_c = 0, max(0, max_note_chars - 1)
        best_c = 0
        while lo_c <= hi_c:
            mid_c = (lo_c + hi_c) // 2
            if _render_notes_at_cap(mid_c)[4] <= target_tokens:
                best_c = mid_c
                lo_c = mid_c + 1
            else:
                hi_c = mid_c - 1
        while best_c > 0 and _render_notes_at_cap(best_c)[4] > target_tokens:
            best_c -= 1
        recs2, man2, text2, maps2, toks2 = _render_notes_at_cap(best_c)
        return _make_result(
            extracted=extracted, tier=2, target_tokens=target_tokens, actual_tokens=toks2,
            chart_text=text2, recs=recs2, mappings=maps2, manifest=man2,
        )

    final_opt = sorted(accepted_opt, key=lambda x: (_date_sort_key(x.date), x.resource_uuid), reverse=True)
    recs2, man2, text2, maps2, toks2 = _assemble_render(
        extracted,
        active_ips + final_opt,
        tier=2,
        tier_name="Budget-Packed Ledger",
        folded_items_count=folded_count,
        excerpted_notes_count=0,
        counter=counter,
    )
    return _make_result(
        extracted=extracted, tier=2, target_tokens=target_tokens, actual_tokens=toks2,
        chart_text=text2, recs=recs2, mappings=maps2, manifest=man2,
    )


__all__ = [
    "CondensedChartResult",
    "condense_bundle_to_budget",
    "estimate_tokens_local",
    "items_to_querystore_records",
]
