"""Deterministic US Core / FHIR R4 clinical normalizers, privacy classifiers, and conflict detectors.

Domain tables (privacy/security codes, SDOH codes, SNOMED exclusivity groups,
RxNorm brand->generic mappings, therapeutic class rules, and ACIP vaccine groups)
are loaded dynamically from ``wahs_core/kb_data/*.json`` rather than duplicated in code.

Privacy / Security Governance Note:
- Primary gate inspects HL7 FHIR ``Resource.meta.security`` confidentiality/sensitivity
  labels (e.g., ``R``, ``V``, ``ETH``, ``PSY``, ``SDV``, ``HIV``, ``SUD``, ``42CFRPART2``)
  plus structured LOINC/SNOMED codes in ``clinical_exclusivity_and_privacy.json``.
- Secondary lexical regex patterns serve strictly as a defense-in-depth fallback for
  uncoded notes/text and do NOT claim exhaustive 42 CFR Part 2 or WA MHMDA compliance.
"""

from __future__ import annotations

from datetime import date
import functools
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Sequence, Set, Tuple

_KB_DATA_DIR = os.path.join(os.path.dirname(__file__), "kb_data")

VALID_SHARING_POSTURES: Set[str] = {"patient_self", "external_share"}


def validate_sharing_posture(sharing_posture: str) -> str:
    """Validate `sharing_posture` fail-closed so typos or unknown strings never bypass privacy redaction."""
    if not isinstance(sharing_posture, str) or sharing_posture not in VALID_SHARING_POSTURES:
        raise ValueError(
            f"Unsupported sharing_posture: {sharing_posture!r}; expected 'patient_self' or 'external_share'"
        )
    return sharing_posture


_ELECTROLYTE_SUPPLEMENT_RE = re.compile(
    r"\b("
    r"potassium\s+(?:chloride|citrate|bicarbonate|gluconate|phosphate|acetate|iodide)|"
    r"sodium\s+(?:bicarbonate|chloride|citrate|phosphate|polystyrene\s+sulfonate)|"
    r"calcium\s+(?:carbonate|citrate|gluconate|acetate|chloride|lactate)|"
    r"magnesium\s+(?:oxide|sulfate|citrate|hydroxide|gluconate|chloride)|"
    r"ferrous\s+(?:sulfate|gluconate|fumarate)|"
    r"lithium\s+(?:carbonate|citrate)|"
    r"zinc\s+(?:sulfate|gluconate)"
    r")\b",
    re.IGNORECASE,
)

_DOSE_NOISE = re.compile(
    r"\b(?:"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|iu|unit|units|meq|%|hr|actuat|unt)(?:/(?:ml|mg|hr|actuat|dose))?|"
    r"oral|injectable|injection|tablet|tablets|capsule|capsules|solution|suspension|spray|gel|cream|"
    r"extended\s+release|delayed\s+release|sustained\s+release|24\s*hr|12\s*hr|"
    r"sublingual|transdermal|inhalation|prefilled\s+syringe|auto-injector|pen\s+injector|"
    r"tartrate|succinate|besylate|hydrochloride|hcl|sodium|potassium|calcium|maleate|fumarate|mononitrate"
    r")\b",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=8)
def _load_kb_pack(filename: str) -> Dict[str, Any]:
    path = os.path.join(_KB_DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def _privacy_tables() -> Tuple[
    Set[str],
    Set[str],
    Set[str],
    Set[str],
    Set[str],
    Optional[Pattern[str]],
    Optional[Pattern[str]],
]:
    pack = _load_kb_pack("clinical_exclusivity_and_privacy.json")
    hl7_codes = {str(c).strip().upper() for c in (pack.get("hl7_restricted_security_codes") or []) if c}
    admin_codes = {str(c).strip() for c in (pack.get("non_medical_administrative_codes") or []) if c}
    loinc_sens = {str(c).strip() for c in (pack.get("sensitive_screening_loinc_codes") or {}).keys() if c}

    snomed_sens: Set[str] = set()
    snomed_social: Set[str] = set()
    for code, entry in (pack.get("snomed_conditions") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        sens = str(entry.get("sensitivity") or "")
        if sens == "SDOH/PRIVACY-SENSITIVE":
            snomed_sens.add(str(code).strip())
        elif sens == "SDOH-SOCIAL":
            snomed_social.add(str(code).strip())

    raw_lex = [str(p) for p in (pack.get("lexical_sensitivity_patterns") or []) if p]
    lex_re = re.compile("|".join(f"(?:{p})" for p in raw_lex), re.IGNORECASE) if raw_lex else None

    raw_dev = [str(p) for p in (pack.get("clinic_room_device_patterns") or []) if p]
    dev_re = re.compile("|".join(f"(?:{p})" for p in raw_dev), re.IGNORECASE) if raw_dev else None
    return hl7_codes, admin_codes, loinc_sens, snomed_sens, snomed_social, lex_re, dev_re


def is_non_medical_admin_code(code: Optional[str]) -> bool:
    """True if SNOMED/code denotes a routine social/employment/administrative status item."""
    clean = (code or "").strip()
    if not clean:
        return False
    _, admin_codes, _, _, snomed_social, _, _ = _privacy_tables()
    return clean in admin_codes or clean in snomed_social


def is_clinic_room_equipment(display: str) -> bool:
    """True when a FHIR Device resource represents clinic room hardware rather than a patient device."""
    _, _, _, _, _, _, dev_re = _privacy_tables()
    return bool(dev_re and dev_re.search(display or ""))


def classify_sensitivity_with_source(
    code: Optional[str],
    display: str = "",
    categories: Sequence[str] = (),
    *,
    meta_security_codes: Sequence[str] = (),
) -> Tuple[Optional[str], Optional[str]]:
    """Classify privacy/SDOH sensitivity, returning ``(sensitivity_label, detection_source)``.

    Evaluates ``meta.security`` codes first (``"fhir_meta_security"``), then KB-coded
    privacy/SDOH identifiers (``"kb_coded_privacy"`` / ``"kb_coded_sdoh_social"``),
    and finally KB-backed lexical fallback patterns (``"lexical_fallback"``).
    """
    hl7_codes, admin_codes, loinc_sens, snomed_sens, snomed_social, lex_re, _ = _privacy_tables()

    for sec_code in meta_security_codes:
        norm = str(sec_code or "").strip().upper()
        if norm and norm in hl7_codes:
            return "SDOH/PRIVACY-SENSITIVE", "fhir_meta_security"

    clean_code = (code or "").strip()
    if clean_code and (clean_code in loinc_sens or clean_code in snomed_sens):
        return "SDOH/PRIVACY-SENSITIVE", "kb_coded_privacy"

    if lex_re is not None and lex_re.search(display or ""):
        return "SDOH/PRIVACY-SENSITIVE", "lexical_fallback"

    if clean_code and (clean_code in admin_codes or clean_code in snomed_social):
        return "SDOH-SOCIAL", "kb_coded_sdoh_social"

    lowered_cats = {str(c).lower() for c in categories if c}
    if "social-history" in lowered_cats and lex_re is not None and lex_re.search(display or ""):
        return "SDOH/PRIVACY-SENSITIVE", "lexical_fallback"

    return None, None


def classify_sensitivity(
    code: Optional[str],
    display: str = "",
    categories: Sequence[str] = (),
    *,
    meta_security_codes: Sequence[str] = (),
) -> Optional[str]:
    """Return ``'SDOH/PRIVACY-SENSITIVE'``, ``'SDOH-SOCIAL'``, or ``None``."""
    label, _ = classify_sensitivity_with_source(
        code, display, categories, meta_security_codes=meta_security_codes
    )
    return label


def _get_brand_to_generic_map() -> Dict[str, str]:
    kb_rx = _load_kb_pack("rxnorm_lookup.json")
    return {
        str(k).lower(): str(v).lower()
        for k, v in (kb_rx.get("brand_to_generic") or {}).items()
        if k and v
    }


def normalize_drug_ingredient(display: str) -> str:
    """Normalize a medication display string to its canonical active-ingredient key."""
    text = (display or "").lower()
    brand_map = _get_brand_to_generic_map()
    if "penicillin v potassium" in text or "penicillin v" in text:
        return "penicillin v potassium"
    if "protamine sulfate" in text or "protamine" in text:
        return "protamine sulfate"
    if "sodium fluoride" in text:
        return "sodium fluoride"
    if "amoxicillin" in text and "clavulanate" in text:
        return "amoxicillin/clavulanate"
    if "insulin isophane" in text and "regular" in text:
        return "insulin isophane/regular"
    for brand, generic in brand_map.items():
        if re.search(rf"\b{re.escape(brand)}\b", text):
            return generic
    text = re.sub(r"\[[^\]]*\]", " ", text)
    m_elec = _ELECTROLYTE_SUPPLEMENT_RE.search(text)
    if m_elec:
        return re.sub(r"\s+", " ", m_elec.group(1)).strip().lower()
    text = _DOSE_NOISE.sub(" ", text)
    text = re.sub(r"[^a-z0-9/+\s-]", " ", text)
    tokens = [tok for tok in text.split() if len(tok) > 1 and not tok.isdigit()]
    if not tokens:
        return (display or "unknown-medication").strip().lower()
    if tokens[0] in {"heparin", "rocuronium", "metoprolol", "naproxen"}:
        return tokens[0]
    head = " ".join(tokens[:3]).strip()
    return brand_map.get(tokens[0], head)


def detect_condition_conflicts(
    active_conditions: Sequence[Mapping[str, Any]],
    resolved_conditions: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, str]:
    """Return ``resourceUuid -> conflict_note`` for mutually exclusive active conditions
    (KB-driven CKD staging, glycemic progression, HF staging) or active-vs-resolved contradictions."""
    pack = _load_kb_pack("clinical_exclusivity_and_privacy.json")
    snomed_meta = pack.get("snomed_conditions") or {}
    group_rules = pack.get("exclusivity_group_rules") or {}
    conflicts: Dict[str, str] = {}
    ckd_by_stage: Dict[int, List[str]] = {}
    hf_by_stage: Dict[int, List[Tuple[str, str]]] = {}
    other_groups_by_stage: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}
    prediabetes_uuids: List[str] = []
    diabetes_uuids: List[Tuple[str, str]] = []

    for cond in active_conditions:
        uuid = str(cond.get("resourceUuid") or "")
        code = str(cond.get("code") or "")
        display = str(cond.get("display") or "")
        if not uuid:
            continue

        entry = snomed_meta.get(code) if isinstance(snomed_meta.get(code), Mapping) else {}
        excl_group = str(entry.get("exclusivity_group") or "")
        stage_rank = entry.get("stage_rank")

        if excl_group == "ckd_stage" and isinstance(stage_rank, int):
            ckd_by_stage.setdefault(stage_rank, []).append(uuid)
        else:
            m = re.search(
                r"\b(?:chronic\s+kidney\s+disease|ckd)\s*(?:stage\s*)?([1-5])(?:[ab])?\b",
                display,
                re.IGNORECASE,
            )
            if m:
                ckd_by_stage.setdefault(int(m.group(1)), []).append(uuid)

        if excl_group == "heart_failure_stage" and isinstance(stage_rank, int):
            hf_by_stage.setdefault(stage_rank, []).append((uuid, display))
        else:
            m_nyha = re.search(
                r"\bnyha\s*(?:class\s*)?(iv|iii|ii|i|[1-4])\b",
                display,
                re.IGNORECASE,
            )
            if m_nyha:
                nyha_map = {"i": 1, "1": 1, "ii": 2, "2": 2, "iii": 3, "3": 3, "iv": 4, "4": 4}
                r_val = nyha_map.get(m_nyha.group(1).lower())
                if r_val is not None:
                    hf_by_stage.setdefault(r_val, []).append((uuid, display))

        if (
            excl_group
            and excl_group not in {"ckd_stage", "glycemic_progression", "heart_failure_stage"}
            and isinstance(stage_rank, int)
        ):
            other_groups_by_stage.setdefault(excl_group, {}).setdefault(stage_rank, []).append((uuid, display))

        if (excl_group == "glycemic_progression" and stage_rank == 1) or re.search(
            r"\bprediabetes\b", display, re.IGNORECASE
        ):
            prediabetes_uuids.append(uuid)
        elif (excl_group == "glycemic_progression" and isinstance(stage_rank, int) and stage_rank >= 2) or re.search(
            r"\b(?:type\s+[12]\s+diabetes|due\s+to\s+(?:type\s+[12]\s+)?diabetes\s+mellitus)\b",
            display,
            re.IGNORECASE,
        ):
            diabetes_uuids.append((uuid, display))

    if len(ckd_by_stage) > 1:
        stages_str = ", ".join(f"stage {s}" for s in sorted(ckd_by_stage))
        max_stage = max(ckd_by_stage)
        for uuids in ckd_by_stage.values():
            for uuid in uuids:
                conflicts[uuid] = (
                    f"stage_conflict: CKD {stages_str} simultaneously marked ACTIVE "
                    f"(highest documented: stage {max_stage})"
                )

    if len(hf_by_stage) > 1:
        hf_rule = group_rules.get("heart_failure_stage") if isinstance(group_rules, Mapping) else {}
        hf_rule_id = str((hf_rule or {}).get("rule_id") or "wahs-kb-hf-staging-01")
        stages_str = ", ".join(f"stage/class {s}" for s in sorted(hf_by_stage))
        max_stage = max(hf_by_stage)
        for entries_hf in hf_by_stage.values():
            for uuid, _ in entries_hf:
                conflicts[uuid] = (
                    f"stage_conflict [{hf_rule_id}]: Heart failure {stages_str} simultaneously marked ACTIVE "
                    f"(highest documented: {max_stage})"
                )

    for grp_name, by_rank in other_groups_by_stage.items():
        if len(by_rank) > 1:
            g_rule = group_rules.get(grp_name) if isinstance(group_rules, Mapping) else {}
            g_title = str((g_rule or {}).get("title") or grp_name)
            for entries_grp in by_rank.values():
                for uuid, _ in entries_grp:
                    conflicts.setdefault(uuid, f"stage_conflict: {g_title}")

    if prediabetes_uuids and diabetes_uuids:
        dm_names = ", ".join(sorted({disp for _, disp in diabetes_uuids})[:2])
        for uuid in prediabetes_uuids:
            conflicts[uuid] = (
                f"stage_conflict: Prediabetes marked ACTIVE alongside established diabetes ({dm_names})"
            )

    def _norm_cond_key(display_val: str) -> str:
        d = re.sub(
            r"\s*\((?:disorder|finding|situation)\)\s*$",
            "",
            (display_val or "").strip(),
            flags=re.IGNORECASE,
        ).lower()
        return f"disp:{d}" if d else ""

    resolved_by_key: Dict[str, List[Mapping[str, Any]]] = {}
    for rc in resolved_conditions:
        rcode = str(rc.get("code") or "")
        rdisp = str(rc.get("display") or "")
        rstatus = str(rc.get("status") or "").strip().lower()
        rabatement = str(rc.get("abatement") or "").strip()
        if is_non_medical_admin_code(rcode) or rstatus != "resolved" or not rabatement:
            continue
        if rcode:
            resolved_by_key.setdefault(f"code:{rcode}", []).append(rc)
        dkey = _norm_cond_key(rdisp)
        if dkey:
            resolved_by_key.setdefault(dkey, []).append(rc)

    for ac in active_conditions:
        acode = str(ac.get("code") or "")
        adisp = str(ac.get("display") or "")
        auuid = str(ac.get("resourceUuid") or "")
        a_onset = str(ac.get("onset") or "")
        if not auuid:
            continue
        candidate_keys = [k for k in (f"code:{acode}" if acode else "", _norm_cond_key(adisp)) if k]
        matched_rcs: List[Mapping[str, Any]] = []
        for ck in candidate_keys:
            for rc in resolved_by_key.get(ck, []):
                if rc not in matched_rcs:
                    matched_rcs.append(rc)
        for rc in matched_rcs:
            r_abatement = str(rc.get("abatement") or "").strip()
            ruuid = str(rc.get("resourceUuid") or "")
            if r_abatement and (not a_onset or r_abatement >= a_onset):
                msg = (
                    f"status_conflict: {ac.get('display')} marked ACTIVE (onset {a_onset or 'unknown'}) "
                    f"but also recorded as RESOLVED on {r_abatement}"
                )
                conflicts.setdefault(auuid, msg)
                if ruuid:
                    conflicts.setdefault(ruuid, msg)

    return conflicts


def _ingredient_matches_group(ing: str, candidates: Iterable[str]) -> bool:
    ing_low = (ing or "").lower()
    ing_tokens = set(re.split(r"[/+\s-]+", ing_low))
    for cand in candidates:
        c_low = str(cand).lower()
        if not c_low:
            continue
        if ing_low == c_low or c_low in ing_low or c_low in ing_tokens:
            return True
    return False


def detect_medication_conflicts(
    active_medications: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Return ``resourceUuid -> conflict_tag`` for duplicate ingredients, KB therapeutic classes, and interaction hazards."""
    conflicts: Dict[str, str] = {}
    by_ingredient: Dict[str, List[Mapping[str, Any]]] = {}

    kb_rx = _load_kb_pack("rxnorm_lookup.json")
    class_rules: Dict[str, Mapping[str, Any]] = kb_rx.get("therapeutic_class_rules") or {}
    interaction_rules: Sequence[Mapping[str, Any]] = kb_rx.get("interaction_hazard_rules") or ()

    by_class: Dict[str, List[Mapping[str, Any]]] = {}
    normalized_meds: List[Tuple[str, Mapping[str, Any]]] = []
    known_uuids = {str(m.get("resourceUuid") or "") for m in active_medications if m.get("resourceUuid")}

    def _order_root(med: Mapping[str, Any]) -> str:
        uuid = str(med.get("resourceUuid") or "")
        for ref_id in med.get("linked_order_uuids") or ():
            clean_ref = str(ref_id or "").strip()
            if clean_ref and clean_ref in known_uuids:
                return clean_ref
        return uuid

    for med in active_medications:
        uuid = str(med.get("resourceUuid") or "")
        disp = str(med.get("display") or "")
        ing = str(med.get("ingredient_key") or normalize_drug_ingredient(disp))
        if not uuid or not ing:
            continue
        by_ingredient.setdefault(ing, []).append(med)
        normalized_meds.append((ing, med))
        for class_key, rule in class_rules.items():
            if _ingredient_matches_group(ing, rule.get("ingredients") or []):
                by_class.setdefault(class_key, []).append(med)

    for ing, group in by_ingredient.items():
        if len({_order_root(m) for m in group}) > 1:
            names = ", ".join(str(m.get("display") or "") for m in group)
            for m in group:
                u = str(m.get("resourceUuid") or "")
                if u:
                    conflicts[u] = f"duplicate_therapy: multiple ACTIVE prescriptions for ingredient '{ing}' ({names})"

    for class_key, group in by_class.items():
        if len({_order_root(m) for m in group}) > 1:
            rule = class_rules.get(class_key) or {}
            rule_id = str(rule.get("rule_id") or f"wahs-kb-rx-class-{class_key}")
            label = str(rule.get("class_label") or rule.get("label") or class_key.replace("_", " ").title())
            names = ", ".join(str(m.get("display") or "") for m in group)
            for m in group:
                u = str(m.get("resourceUuid") or "")
                if u and u not in conflicts:
                    conflicts[u] = f"duplicate_therapy [{rule_id}]: concurrent ACTIVE {label} prescriptions ({names})"

    for irule in interaction_rules:
        group_a = [m for ing, m in normalized_meds if _ingredient_matches_group(ing, irule.get("group_a_ingredients") or [])]
        group_b = [m for ing, m in normalized_meds if _ingredient_matches_group(ing, irule.get("group_b_ingredients") or [])]
        if group_a and group_b:
            rule_id = str(irule.get("rule_id") or "wahs-kb-rx-interaction")
            label = str(irule.get("label") or "High-Risk Drug Interaction")
            hazard = str(irule.get("hazard") or "Concurrent high-risk therapy requires reconciliation.")
            combined = group_a + [m for m in group_b if m not in group_a]
            names = ", ".join(str(m.get("display") or "") for m in combined)
            for m in combined:
                u = str(m.get("resourceUuid") or "")
                if u and u not in conflicts:
                    conflicts[u] = f"interaction_hazard [{rule_id}]: {label} ({names}) — {hazard}"

    return conflicts


def classify_vaccine_group(code: Optional[str], display: str) -> str:
    """Map CVX code or vaccine display string to a canonical vaccine group using ``acip_rules.json``."""
    c = (code or "").strip()
    d = (display or "").lower()
    acip = _load_kb_pack("acip_rules.json")
    cvx_groups = acip.get("cvx_groups") or {}
    if c and c in cvx_groups and isinstance(cvx_groups[c], Mapping):
        grp = str(cvx_groups[c].get("vaccine_group") or "").strip()
        if grp:
            return grp

    if c in {"62", "118", "137", "165"} or "hpv" in d or "papillomavirus" in d:
        return "HPV"
    if c in {"121"} or ("zoster" in d and "live" in d):
        return "Zoster-Live"
    if c in {"187", "188"} or "zoster" in d or "shingrix" in d:
        return "Zoster-Recombinant"
    if "influenza" in d or "flu" in d or c in {"88", "140", "141", "150", "158", "168", "197"}:
        return "Influenza"
    if "covid" in d or "sars-cov-2" in d or c in {"207", "208", "212", "213", "300", "301", "309"}:
        return "COVID-19"
    if "meningococcal" in d or "mcv4" in d or c in {"114", "136", "147"}:
        return "Meningococcal"
    if "tdap" in d or "tetanus" in d or "diphtheria" in d or c in {"115", "09", "113", "20", "106"}:
        return "Tdap/DTaP/Td"
    if "pneumococcal" in d or c in {"33", "133", "152", "215", "216"}:
        return "Pneumococcal"
    if "hepatitis b" in d or "hep b" in d or c in {"08", "43", "44", "45"}:
        return "HepB"
    if "hepatitis a" in d or "hep a" in d or c in {"83", "85"}:
        return "HepA"
    if "mmr" in d or "measles" in d or c in {"03", "94"}:
        return "MMR"
    if "varicella" in d or c in {"21"}:
        return "Varicella"
    return (display or "Vaccine").strip()


def _age_in_years(birth_date_iso: Optional[str], event_date_iso: Optional[str]) -> Optional[float]:
    if not birth_date_iso or not event_date_iso:
        return None
    try:
        b = date.fromisoformat(birth_date_iso[:10])
        e = date.fromisoformat(event_date_iso[:10])
        return round((e - b).days / 365.2425, 2)
    except ValueError:
        return None


def annotate_immunization_anomalies(
    immunizations: Sequence[Mapping[str, Any]],
    birth_date_iso: Optional[str],
) -> Dict[str, str]:
    """Return ``resourceUuid -> acip_or_duplicate_note`` for immunization series deviations or near-date duplicates."""
    notes: Dict[str, str] = {}
    by_group: Dict[str, List[Mapping[str, Any]]] = {}
    for imm in immunizations:
        group = classify_vaccine_group(imm.get("code"), str(imm.get("display") or ""))
        by_group.setdefault(group, []).append(imm)

    for group, items in by_group.items():
        ordered = sorted(items, key=lambda x: str(x.get("date") or ""))
        for i in range(1, len(ordered)):
            prev, curr = ordered[i - 1], ordered[i]
            d1, d2 = str(prev.get("date") or ""), str(curr.get("date") or "")
            try:
                days_apart = abs((date.fromisoformat(d2[:10]) - date.fromisoformat(d1[:10])).days)
            except ValueError:
                days_apart = 999
            if days_apart <= 14:
                u1, u2 = str(prev.get("resourceUuid") or ""), str(curr.get("resourceUuid") or "")
                msg = f"duplicate_candidate: {group} recorded {days_apart}d apart ({d1} vs {d2})"
                if u1:
                    notes[u1] = msg
                if u2:
                    notes[u2] = msg

        if group == "HPV" and len(ordered) >= 3:
            first_date = str(ordered[0].get("date") or "")
            second_date = str(ordered[1].get("date") or "")
            age_first = _age_in_years(birth_date_iso, first_date)
            try:
                d1_d2_days: Optional[int] = (
                    date.fromisoformat(second_date[:10]) - date.fromisoformat(first_date[:10])
                ).days
            except ValueError:
                d1_d2_days = None
            if age_first is not None and age_first < 15.0 and d1_d2_days is not None and d1_d2_days >= 150:
                last_uuid = str(ordered[-1].get("resourceUuid") or "")
                dates_list = ", ".join(str(x.get("date") or "") for x in ordered)
                if last_uuid and last_uuid not in notes:
                    notes[last_uuid] = (
                        f"acip_schedule_note: 3rd HPV dose recorded ({dates_list}) after series initiated at age "
                        f"{int(age_first)} ({first_date}); CDC ACIP specifies 2-dose series (minimum interval 5 months) when started before age 15"
                    )

        if group == "Zoster-Live" and len(ordered) >= 2:
            last_uuid = str(ordered[-1].get("resourceUuid") or "")
            dates_list = ", ".join(str(x.get("date") or "") for x in ordered)
            if last_uuid and last_uuid not in notes:
                notes[last_uuid] = (
                    f"acip_schedule_note: {len(ordered)} doses of live zoster vaccine recorded ({dates_list}); "
                    "live zoster was a single-dose vaccine (superseded by 2-dose recombinant Shingrix in US)"
                )

    return notes
