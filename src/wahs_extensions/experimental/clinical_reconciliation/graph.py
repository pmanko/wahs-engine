"""Experimental clinical reconciliation prototype (`ContextEnricher`).

Houses cleaned sample rule packs extracted from earlier benchmark prototypes:
- Active condition exclusivity (CKD stages 1-5, prediabetes vs. diabetes, HF classes)
- Brand-generic & therapeutic-class medication duplication + interaction checks
- Sample CDC ACIP immunization duplicate & series checks + guideline snippets

Marked strictly as `kind="experimental_clinical_rules"` so operators and downstream
AI-Provenance consumers know this prototype covers sample ontologies only.
"""

from __future__ import annotations

from datetime import date
import functools
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from wahs_core.extractors import ClinicalItem, ExtractedChart
from wahs_core.ports import EnrichmentDisclosure, EnrichmentInput, EnrichmentOutput

_PACKS_DIR = Path(__file__).resolve().parent / "sample_packs"


@functools.lru_cache(maxsize=8)
def _load_json_pack(name: str) -> Dict[str, Any]:
    path = _PACKS_DIR / name
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@functools.lru_cache(maxsize=4)
def _load_snippets() -> List[Dict[str, Any]]:
    path = _PACKS_DIR / "guideline_snippets.jsonl"
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
        except json.JSONDecodeError:
            continue
    return out


_DOSE_NOISE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|iu|unit|units|meq|%|hr)(?:/(?:ml|mg|hr|dose))?|"
    r"oral|injectable|injection|tablet|tablets|capsule|capsules|solution|suspension|"
    r"extended\s+release|delayed\s+release|hydrochloride|hcl|sodium|potassium|calcium)\b",
    re.IGNORECASE,
)


def _normalize_ingredient(display: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", (display or "").lower())
    brand_map = {
        str(k).lower(): str(v).lower()
        for k, v in (_load_json_pack("rxnorm_sample.json").get("brand_to_generic") or {}).items()
        if k and v
    }
    for brand, generic in sorted(brand_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.search(rf"\b{re.escape(brand)}\b", text):
            return generic
    text = re.sub(r"[^a-z0-9/+\s-]", " ", _DOSE_NOISE.sub(" ", text))
    tokens = [t for t in text.split() if len(t) > 1 and not t.isdigit()]
    return brand_map.get(tokens[0], " ".join(tokens[:2])) if tokens else (display or "").lower()


def _detect_condition_alerts(items: Sequence[ClinicalItem]) -> List[str]:
    pack = _load_json_pack("cardiorenal_graph_rules.json")
    snomed_meta = pack.get("snomed_conditions") or {}
    ckd_stages: Set[int] = set()
    prediab: List[str] = []
    diab: List[str] = []
    active_by_name: Dict[str, List[ClinicalItem]] = {}
    resolved_by_name: Dict[str, List[ClinicalItem]] = {}

    for it in items:
        if it.section not in {"condition_active", "condition_resolved"}:
            continue
        norm_disp = re.sub(r"\s*\((?:disorder|finding|situation)\)\s*$", "", it.display or "", flags=re.I).lower().strip()
        if it.section == "condition_active":
            active_by_name.setdefault(norm_disp, []).append(it)
            entry = snomed_meta.get(it.code) if isinstance(snomed_meta.get(it.code), Mapping) else {}
            grp, rank = entry.get("exclusivity_group"), entry.get("stage_rank")
            if grp == "ckd_stage" and isinstance(rank, int):
                ckd_stages.add(rank)
            else:
                m = re.search(r"\b(?:chronic\s+kidney\s+disease|ckd)\s*(?:stage\s*)?([1-5])\b", it.display or "", re.I)
                if m:
                    ckd_stages.add(int(m.group(1)))
            if (grp == "glycemic_progression" and rank == 1) or "prediabetes" in norm_disp:
                prediab.append(it.display)
            elif (grp == "glycemic_progression" and isinstance(rank, int) and rank >= 2) or "type 2 diabetes" in norm_disp or "diabetes mellitus type 2" in norm_disp:
                diab.append(it.display)
        elif it.status == "resolved" and it.end_date:
            resolved_by_name.setdefault(norm_disp, []).append(it)

    alerts: List[str] = []
    if len(ckd_stages) > 1:
        alerts.append(
            f"- Condition Stage Alert: Multiple active CKD stages documented ({', '.join(f'stage {s}' for s in sorted(ckd_stages))})."
        )
    if prediab and diab:
        alerts.append(
            f"- Condition Progression Alert: Both Prediabetes and established Diabetes ({diab[0]}) appear active."
        )
    for name, act_list in active_by_name.items():
        if name in resolved_by_name and act_list:
            res_dt = resolved_by_name[name][-1].end_date
            alerts.append(
                f"- Condition Status Contradiction: {act_list[0].display} is marked active and also resolved ({res_dt})."
            )
    return alerts


def _detect_medication_alerts(items: Sequence[ClinicalItem]) -> List[str]:
    rx_pack = _load_json_pack("rxnorm_sample.json")
    class_rules = rx_pack.get("therapeutic_class_rules") or {}
    hazard_rules = rx_pack.get("interaction_hazard_rules") or ()

    active_meds = [it for it in items if it.section == "med_active"]
    by_ing: Dict[str, List[ClinicalItem]] = {}
    by_class: Dict[str, List[ClinicalItem]] = {}
    norm_list: List[Tuple[str, ClinicalItem]] = []

    for m in active_meds:
        ing = _normalize_ingredient(m.display)
        if not ing:
            continue
        by_ing.setdefault(ing, []).append(m)
        norm_list.append((ing, m))
        for ckey, rule in class_rules.items():
            cands = [str(c).lower() for c in (rule.get("ingredients") or ())]
            if any(c == ing or c in ing for c in cands):
                by_class.setdefault(ckey, []).append(m)

    alerts: List[str] = []
    for ing, grp in by_ing.items():
        if len(grp) > 1:
            names = ", ".join(m.display for m in grp[:3])
            alerts.append(f"- Duplicate Ingredient: Multiple active orders for '{ing}' ({names}).")
    for ckey, grp in by_class.items():
        if len({m.display.lower() for m in grp}) > 1:
            lbl = str((class_rules.get(ckey) or {}).get("class_label") or ckey)
            names = ", ".join(m.display for m in grp[:3])
            alerts.append(f"- Same-Class Duplication ({lbl}): Concurrent active {names}.")
    for ir in hazard_rules:
        ga = [m.display for ing, m in norm_list if any(c in ing for c in (ir.get("group_a_ingredients") or ()))]
        gb = [m.display for ing, m in norm_list if any(c in ing for c in (ir.get("group_b_ingredients") or ()))]
        if ga and gb:
            alerts.append(f"- Interaction Hazard ({ir.get('domain') or 'Rx Hazard'}): {ga[0]} + {gb[0]}.")
    return alerts


def _detect_immunization_alerts(extracted: ExtractedChart) -> List[str]:
    cvx_groups = (_load_json_pack("acip_sample.json").get("cvx_groups") or {})
    by_group: Dict[str, List[ClinicalItem]] = {}
    for it in extracted.items:
        if it.section != "immunization" or it.status != "completed":
            continue
        grp = str((cvx_groups.get(it.code) or {}).get("vaccine_group") or it.display or "Vaccine")
        by_group.setdefault(grp, []).append(it)
    alerts: List[str] = []
    for grp, grp_items in by_group.items():
        ordered = sorted(grp_items, key=lambda x: x.date or "")
        for i in range(1, len(ordered)):
            d1, d2 = ordered[i - 1].date or "", ordered[i].date or ""
            try:
                days = abs((date.fromisoformat(d2[:10]) - date.fromisoformat(d1[:10])).days)
            except ValueError:
                days = 999
            if days <= 14:
                alerts.append(f"- Possible Duplicate Immunization ({grp}): Recorded {days}d apart ({d1} vs {d2}).")
                break
    return alerts


class PrototypeClinicalGraphEnricher:
    """Experimental cardiorenal/medication/ACIP reconciliation helper (`ContextEnricher`)."""

    enricher_id = "experimental.clinical_reconciliation"
    disclosure = EnrichmentDisclosure(
        enricher_id="experimental.clinical_reconciliation",
        version="0.1.0",
        kind="experimental_clinical_rules",
        scope_summary="Experimental sample rule packs covering selected cardiorenal, glycemic staging, duplicate medication, and ACIP vaccine checks.",
        limitations="Prototype sample ontology covering a narrow set of chronic-care rules; NOT comprehensive across all conditions/drugs and NOT validated as standalone CDS.",
    )

    def enrich(self, inp: EnrichmentInput) -> EnrichmentOutput | None:
        if inp.max_tokens < 64:
            return None
        alerts = (
            _detect_condition_alerts(inp.extracted.items)
            + _detect_medication_alerts(inp.extracted.items)
            + _detect_immunization_alerts(inp.extracted)
        )
        chart_codes = {it.code for it in inp.extracted.items if it.code}
        q_terms = {t for t in re.findall(r"[a-z0-9]{4,}", (inp.question or "").lower())}
        snippet_lines: List[str] = []
        for row in _load_snippets():
            row_codes = {str(c) for c in (row.get("codes") or ())}
            hay = f"{row.get('title', '')} {row.get('text', '')}".lower()
            if (row_codes & chart_codes) or any(t in hay for t in q_terms):
                snippet_lines.append(f"- Reference ({row.get('title')}): {row.get('text')}")
            if len(snippet_lines) >= 2:
                break

        body_lines = alerts + snippet_lines
        if not body_lines:
            return None
        heading = "=== EXPERIMENTAL CLINICAL RECONCILIATION NOTES (Sample Prototype Rules Only) ==="
        chosen = list(body_lines)
        while chosen and inp.token_counter("\n".join([heading, *chosen])) > inp.max_tokens:
            chosen.pop()
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
            metadata={"alerts_count": len(alerts), "snippets_count": len(snippet_lines)},
        )
