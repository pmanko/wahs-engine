"""Layer 3: Typed Patient Clinical Entity Graph & Hub Context Enrichment (`wahs_core`).

Pure Python stdlib module that builds a patient-specific typed clinical entity graph
directly from Layer-1 ``ExtractedChart`` and Layer-2 ``CondensedChartResult`` objects,
assigns sequential ``[N]`` Tier-0 overflow recovery citations without regex re-parsing,
and attaches 4-layer Clinical Knowledge Base (KB) entries bounded cleanly by an
optional token budget.

Strict Layer Boundaries:
- Consumes typed dataclasses (``ExtractedChart``, ``CondensedChartResult``,
  ``citation_map``) rather than regex-parsing condensed chart text.
- Respects ``sharing_posture == "external_share"`` by excluding any node, numeric
  observation, edge, overflow recovery item, or KB code derived from
  ``SDOH/PRIVACY-SENSITIVE`` items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .condenser import CondensedChartResult, estimate_tokens_local
from .extractors import ClinicalItem, ExtractedChart, NumericPoint, extract_bundle
from .kb import search as search_kb
from .kb_lookup import extract_chart_kb_codes, get_rxnorm_lookup
from .normalizers import (
    annotate_immunization_anomalies,
    classify_sensitivity,
    classify_vaccine_group,
    detect_condition_conflicts,
    detect_medication_conflicts,
    normalize_drug_ingredient,
)

NODE_CATEGORIES = (
    "Condition",
    "Medication",
    "Observation",
    "Procedure",
    "Encounter",
    "DocumentReference",
    "CarePlan",
    "AllergyIntolerance",
    "Immunization",
    "Device",
    "Organization",
)

EDGE_TYPES = (
    "CROSS_REACTS_WITH",
    "CONTRAINDICATED_OR_DOSE_ADJUST_IN",
    "CONFLICTS_WITH",
    "CONTRADICTED_OR_MODIFIED_BY_NOTE",
    "MONITORED_BY",
    "TREATS",
    "REPORTED_BY_ORG",
)

_SECTION_TO_CATEGORY: Dict[str, str] = {
    "condition_active": "Condition",
    "condition_resolved": "Condition",
    "med_active": "Medication",
    "med_historical": "Medication",
    "allergy": "AllergyIntolerance",
    "immunization": "Immunization",
    "lab_panel": "Observation",
    "lab_observation": "Observation",
    "vital_sign": "Observation",
    "social_history": "Observation",
    "survey_panel": "Observation",
    "survey_observation": "Observation",
    "observation": "Observation",
    "imaging": "Observation",
    "clinical_note": "DocumentReference",
    "note": "DocumentReference",
    "procedure_major": "Procedure",
    "procedure_routine": "Procedure",
    "encounter_acuity": "Encounter",
    "encounter_routine": "Encounter",
    "careplan_active": "CarePlan",
    "careplan_historical": "CarePlan",
    "device": "Device",
}

_EDGE_PRIORITY: Dict[str, int] = {
    "CROSS_REACTS_WITH": 0,
    "CONTRAINDICATED_OR_DOSE_ADJUST_IN": 1,
    "CONFLICTS_WITH": 2,
    "CONTRADICTED_OR_MODIFIED_BY_NOTE": 3,
    "MONITORED_BY": 4,
    "TREATS": 5,
    "REPORTED_BY_ORG": 6,
}

_SECTION_OVERFLOW_PRIORITY: Dict[str, int] = {
    "allergy": 0,
    "condition_active": 1,
    "med_active": 2,
    "clinical_note": 3,
    "note": 3,
    "lab_observation": 4,
    "lab_panel": 4,
    "vital_sign": 5,
    "encounter_acuity": 5,
    "procedure_major": 6,
    "immunization": 6,
    "imaging": 7,
    "careplan_active": 7,
    "med_historical": 8,
    "condition_resolved": 9,
    "social_history": 10,
    "procedure_routine": 11,
    "encounter_routine": 12,
    "careplan_historical": 13,
    "device": 14,
}

_QUERY_STOPWORDS: Set[str] = {
    "about",
    "after",
    "again",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "chart",
    "clinical",
    "could",
    "current",
    "currently",
    "date",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "during",
    "each",
    "ever",
    "explain",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "health",
    "here",
    "how",
    "if",
    "in",
    "include",
    "including",
    "info",
    "information",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "last",
    "latest",
    "list",
    "look",
    "me",
    "medical",
    "medication",
    "medications",
    "meds",
    "more",
    "most",
    "my",
    "need",
    "not",
    "note",
    "notes",
    "now",
    "of",
    "on",
    "only",
    "or",
    "other",
    "out",
    "over",
    "patient",
    "please",
    "recent",
    "recently",
    "record",
    "records",
    "report",
    "results",
    "review",
    "should",
    "show",
    "since",
    "so",
    "some",
    "still",
    "summarize",
    "summary",
    "taking",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "time",
    "to",
    "today",
    "up",
    "use",
    "using",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

_CITATION_TAG_RE = re.compile(r"\[(\d+)\]")


@dataclass
class ClinicalNode:
    """A node in the patient-specific clinical entity graph."""

    node_id: str
    category: str
    label: str
    code: str = ""
    citation_idx: Optional[int] = None
    resource_uuid: str = ""
    date_str: str = ""
    status: str = ""
    value_summary: str = ""
    raw_line: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation_tag(self) -> str:
        if self.citation_idx is not None:
            return f"[{self.citation_idx}]"
        if self.resource_uuid:
            return f"[UUID: {self.resource_uuid}]"
        return ""

    def short_display(self) -> str:
        tag = f"[{self.citation_idx}] " if self.citation_idx is not None else ""
        if self.category == "DocumentReference":
            date_suffix = f" ({self.date_str})" if self.date_str else ""
            return f"{tag}{self.label}{date_suffix}".strip()
        val_clean = (self.value_summary or "").strip()
        val = f" ({val_clean})" if val_clean else ""
        return f"{tag}{self.label}{val}".strip()


@dataclass
class ClinicalEdge:
    """A directed, provenance-linked clinical relationship edge."""

    source_id: str
    target_id: str
    edge_type: str
    rationale: str
    kb_source: str = ""
    source_citation: Optional[int] = None
    target_citation: Optional[int] = None


@dataclass
class HubEnrichmentResult:
    """Structured result from Layer-3 pre-generation context enrichment."""

    enrichment_text: str
    enrichment_tokens: int
    omitted_clinical_items: int
    overflow_triggered: bool
    overflow_recovered_count: int
    subgraph_nodes: int
    subgraph_edges: int
    kb_Passages_matched: int
    max_citation_idx_before: int
    max_citation_idx_after: int
    active_question: str = ""
    patient_name: str = ""
    omitted_by_category: Dict[str, int] = field(default_factory=dict)
    edge_Breakdown: Dict[str, int] = field(default_factory=dict)
    recovered_mappings: List[Dict[str, Any]] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "omitted_clinical_items": self.omitted_clinical_items,
            "overflow_triggered": self.overflow_triggered,
            "overflow_recovered_count": self.overflow_recovered_count,
            "overflow_items_recovered": self.overflow_recovered_count,
            "subgraph_nodes": self.subgraph_nodes,
            "subgraph_edges": self.subgraph_edges,
            "kb_passages_matched": self.kb_Passages_matched,
            "enrichment_tokens": self.enrichment_tokens,
            "max_citation_idx_before": self.max_citation_idx_before,
            "max_citation_idx_after": self.max_citation_idx_after,
            "patient_name": self.patient_name,
            "omitted_by_category": dict(self.omitted_by_category),
            "edge_breakdown": dict(self.edge_Breakdown),
            "recovered_mappings": list(self.recovered_mappings),
        }


# ---------------------------------------------------------------------------
# Clinical Domain Rules for Graph Edge Construction
# ---------------------------------------------------------------------------

_DRUG_INDICATION_KEYWORDS: List[Tuple[Tuple[str, ...], Tuple[str, ...], str, str]] = [
    (
        ("apixaban", "eliquis", "rivaroxaban", "xarelto", "warfarin", "coumadin", "dabigatran", "edoxaban"),
        ("atrial fibrillation", "afib", "flutter", "thrombosis", "embolism", "dvt", "pe", "stroke", "mechanical", "aortic valve", "mitral valve"),
        "Anticoagulation for thromboembolic protection / atrial fibrillation / prosthetic valve",
        "AHA/ACC/HRS AFib & Anticoagulation Guideline",
    ),
    (
        ("sacubitril", "valsartan", "entresto", "lisinopril", "losartan", "ramipril", "enalapril", "carvedilol", "metoprolol", "bisoprolol", "spironolactone", "eplerenone", "furosemide", "lasix", "torsemide", "bumetanide"),
        ("heart failure", "hfref", "hfpef", "cardiomyopathy", "ventricular dysfunction", "edema", "overload"),
        "Guideline-directed medical therapy (GDMT) / decongestion for heart failure",
        "AHA/ACC/HFSA Heart Failure Guideline",
    ),
    (
        ("empagliflozin", "jardiance", "dapagliflozin", "farxiga", "canagliflozin"),
        ("heart failure", "hfref", "hfpef", "diabetes", "kidney disease", "ckd", "renal", "nephropathy"),
        "SGLT2 inhibitor organ-protective therapy for HF / T2DM / CKD",
        "KDIGO CKD & ADA/AHA Cardiorenal Guideline",
    ),
    (
        ("lisinopril", "losartan", "valsartan", "amlodipine", "hydrochlorothiazide", "chlorthalidone", "metoprolol", "carvedilol", "diltiazem", "spironolactone", "hydralazine"),
        ("hypertension", "high blood pressure", "elevated blood pressure"),
        "Antihypertensive therapy for blood pressure control",
        "ACC/AHA Hypertension Guideline",
    ),
    (
        ("metformin", "insulin", "glargine", "lispro", "aspart", "glipizide", "glimepiride", "semaglutide", "ozempic", "liraglutide", "dulaglutide", "sitagliptin"),
        ("diabetes", "hyperglycemia", "prediabetes", "a1c"),
        "Glycemic control therapy for diabetes mellitus",
        "ADA Standards of Care in Diabetes",
    ),
    (
        ("atorvastatin", "lipitor", "rosuvastatin", "crestor", "simvastatin", "pravastatin", "ezetimibe", "evolocumab"),
        ("hyperlipidemia", "dyslipidemia", "hypercholesterolemia", "coronary", "cad", "atherosclerosis", "myocardial infarction", "stroke"),
        "Lipid-lowering and atherosclerotic cardiovascular risk reduction",
        "ACC/AHA Cholesterol & ASCVD Guideline",
    ),
    (
        ("levothyroxine", "synthroid"),
        ("hypothyroidism", "thyroid"),
        "Thyroid hormone replacement for hypothyroidism",
        "ATA Hypothyroidism Guideline",
    ),
    (
        ("allopurinol", "febuxostat", "colchicine"),
        ("gout", "hyperuricemia"),
        "Urate-lowering / flare prophylaxis therapy for gout",
        "ACR Gout Management Guideline",
    ),
    (
        ("albuterol", "salbutamol", "tiotropium", "ipratropium", "fluticasone", "budesonide", "formoterol", "salmeterol", "montelukast"),
        ("asthma", "copd", "chronic obstructive", "bronchitis", "wheezing"),
        "Bronchodilator / anti-inflammatory airway therapy",
        "GOLD COPD / GINA Asthma Strategy",
    ),
    (
        ("sertraline", "escitalopram", "fluoxetine", "citalopram", "venlafaxine", "duloxetine", "bupropion"),
        ("depression", "depressive", "anxiety", "ptsd", "mood"),
        "Pharmacotherapy for depressive or anxiety disorder",
        "APA Mood & Anxiety Clinical Practice Guideline",
    ),
    (
        ("buprenorphine", "suboxone", "naloxone", "methadone", "naltrexone"),
        ("opioid", "dependence", "substance", "addiction"),
        "Medication for opioid use disorder (MOUD)",
        "SAMHSA / ASAM OUD Guideline",
    ),
    (
        ("aspirin", "clopidogrel", "plavix", "ticagrelor", "prasugrel", "isosorbide", "nitroglycerin"),
        ("coronary", "cad", "angina", "ischemic", "myocardial infarction", "stent", "pci", "bypass", "cabg", "heart failure", "cardiomyopathy", "vascular"),
        "Antiplatelet / antianginal therapy for ischemic or cardiovascular disease",
        "ACC/AHA Chronic Coronary Disease Guideline",
    ),
]

_SAFETY_CONTRAINDICATION_RULES: List[Tuple[Tuple[str, ...], Tuple[str, ...], str, str]] = [
    (
        ("metformin",),
        ("stage 4", "stage 5", "ckd 4", "ckd 5", "kidney disease, stage 4", "kidney disease stage 4", "egfr", "glomerular filtration", "creatinine", "acute kidney", "aki"),
        "Metformin renal safety: contraindicated when eGFR < 30 mL/min/1.73m2; avoid initiation and reduce dose when eGFR 30-44 due to lactic acidosis risk",
        "FDA Metformin Boxed Warning & KDIGO Diabetes in CKD",
    ),
    (
        ("spironolactone", "eplerenone", "sacubitril", "valsartan", "entresto", "lisinopril", "losartan", "ramipril", "enalapril"),
        ("hyperkalemia", "potassium", "stage 4", "stage 5", "ckd 4", "kidney disease, stage 4", "kidney disease stage 4", "acute kidney", "aki", "creatinine", "egfr", "glomerular filtration"),
        "RAAS / MRA cardiorenal safety: monitor potassium (hyperkalemia risk) and renal function (eGFR/creatinine) in advanced CKD or volume depletion",
        "AHA/ACC/HFSA & KDIGO Cardiorenal Monitoring Rule",
    ),
    (
        ("apixaban", "eliquis", "rivaroxaban", "xarelto", "dabigatran", "edoxaban"),
        ("stage 4", "stage 5", "ckd 4", "kidney disease, stage 4", "kidney disease stage 4", "creatinine", "egfr", "glomerular filtration", "bleed", "hemorrhage", "anemia", "hemoglobin"),
        "DOAC renal & bleeding safety: check renal dose-reduction criteria and monitor renal clearance/hemoglobin",
        "FDA Apixaban/DOAC Renal Dosing & Anticoagulation Safety",
    ),
    (
        ("warfarin", "coumadin"),
        ("inr", "prothrombin", "bleed", "hemorrhage", "bruising", "hematuria", "melena"),
        "Warfarin narrow therapeutic index: supratherapeutic INR or active bleeding requires prompt review and interaction check",
        "CHEST Antithrombotic Therapy Guideline",
    ),
    (
        ("ibuprofen", "naproxen", "meloxicam", "ketorolac", "diclofenac", "celecoxib", "indomethacin", "nsaid"),
        ("kidney disease", "ckd", "renal", "heart failure", "hfref", "creatinine", "egfr", "apixaban", "warfarin", "rivaroxaban", "ulcer", "bleed"),
        "NSAID high-risk interaction: associated with AKI, fluid retention/HF decompensation, and GI/anticoagulant bleeding",
        "KDIGO & AGS Beers Criteria High-Risk Medication Rule",
    ),
    (
        ("allopurinol", "colchicine", "gabapentin", "pregabalin", "digoxin", "fenofibrate"),
        ("kidney disease", "ckd", "stage 3", "stage 4", "stage 5", "creatinine", "egfr", "glomerular filtration"),
        "Renally cleared medication: check eGFR-guided dosing to prevent drug accumulation",
        "Renal Pharmacotherapy Dosing Guideline",
    ),
    (
        ("furosemide", "lasix", "torsemide", "bumetanide", "hydrochlorothiazide", "chlorthalidone", "metolazone"),
        ("potassium", "sodium", "creatinine", "egfr", "urea nitrogen", "bun", "hypotension", "dehydration", "acute kidney"),
        "Diuretic electrolyte & renal safety: monitor potassium, sodium, volume status, and creatinine",
        "ACC/AHA Heart Failure Diuretic Monitoring Rule",
    ),
    (
        ("insulin", "glargine", "lispro", "aspart", "glipizide", "glimepiride", "glyburide"),
        ("hypoglycemia", "glucose", "hemoglobin a1c", "a1c", "kidney disease", "ckd"),
        "Hypoglycemia risk: reduced renal clearance and tight glycemic control in CKD or older adults increase hypoglycemia risk",
        "ADA Glycemic Targets & Older Adults Safety Guideline",
    ),
]

_MONITORING_RULES: List[Tuple[Tuple[str, ...], Tuple[str, ...], str]] = [
    (
        ("kidney", "ckd", "renal", "nephropathy", "proteinuria", "spironolactone", "eplerenone", "sacubitril", "valsartan", "lisinopril", "losartan", "ramipril", "furosemide", "torsemide", "empagliflozin", "dapagliflozin", "metformin", "allopurinol"),
        ("creatinine", "glomerular filtration", "egfr", "potassium", "urea nitrogen", "bun", "albumin/creatinine", "uacr", "cystatin"),
        "Renal function & electrolyte surveillance (Cr / eGFR / K+ / BUN / UACR)",
    ),
    (
        ("heart failure", "hfref", "hfpef", "cardiomyopathy", "edema", "dyspnea", "sacubitril", "valsartan", "furosemide", "torsemide", "spironolactone", "carvedilol", "metoprolol"),
        ("natriuretic", "nt-probnp", "bnp", "body weight", "weight", "potassium", "creatinine", "sodium", "ejection fraction", "lvef"),
        "Heart failure congestion & cardiorenal trajectory monitoring (NT-proBNP / Weight / K+ / Cr)",
    ),
    (
        ("diabetes", "hyperglycemia", "metformin", "insulin", "glargine", "glipizide", "semaglutide", "liraglutide", "empagliflozin", "dapagliflozin"),
        ("hemoglobin a1c", "hba1c", "a1c", "glucose", "albumin/creatinine", "uacr"),
        "Glycemic control & diabetic nephropathy surveillance (HbA1c / Glucose / UACR)",
    ),
    (
        ("atrial fibrillation", "afib", "thrombosis", "embolism", "warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin"),
        ("inr", "prothrombin", "hemoglobin", "hematocrit", "platelet", "creatinine", "heart rate"),
        "Anticoagulation safety, bleeding & rate surveillance (INR / Hgb / Platelets / Cr)",
    ),
    (
        ("hyperlipidemia", "dyslipidemia", "coronary", "cad", "atherosclerosis", "atorvastatin", "rosuvastatin", "simvastatin", "ezetimibe"),
        ("cholesterol", "ldl", "hdl", "triglyceride", "aminotransferase", "ast", "alt"),
        "Lipid panel & statin safety surveillance (LDL-C / Total Cholesterol / Triglycerides / Hepatic enzymes)",
    ),
    (
        ("hypertension", "blood pressure", "amlodipine", "lisinopril", "losartan", "valsartan", "hydrochlorothiazide", "chlorthalidone", "metoprolol", "carvedilol"),
        ("blood pressure", "systolic", "diastolic", "mean arterial"),
        "Hemodynamic blood pressure trajectory monitoring",
    ),
    (
        ("hypothyroidism", "thyroid", "levothyroxine"),
        ("thyrotropin", "tsh", "thyroxine", "free t4"),
        "Thyroid function surveillance (TSH / Free T4)",
    ),
    (
        ("gout", "hyperuricemia", "allopurinol", "febuxostat"),
        ("urate", "uric acid"),
        "Serum urate target surveillance",
    ),
    (
        ("anemia", "iron deficiency", "bleed", "hemorrhage"),
        ("hemoglobin", "hematocrit", "ferritin", "iron", "mcv", "reticulocyte"),
        "Hematologic & iron status trajectory monitoring",
    ),
]

_NOTE_HOLD_OR_MOD_CUES: Tuple[str, ...] = (
    "stopped",
    "discontinued",
    "held",
    "holding",
    "hold ",
    "not taking",
    "non-adherent",
    "nonadherent",
    "self-discontinued",
    "ran out",
    "reduced",
    "half dose",
    "declined",
    "intolerant",
    "switched",
    "outside",
)


def _slugify(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return s[:48] or "item"


def _query_terms(question: str) -> Set[str]:
    tokens = re.findall(r"[a-z0-9]{2,}", (question or "").lower())
    return {t for t in tokens if t not in _QUERY_STOPWORDS and not t.isdigit()}


def _is_sensitive_item(item: ClinicalItem) -> bool:
    sens = item.sensitivity or classify_sensitivity(item.code, f"{item.display} {item.text}")
    return sens == "SDOH/PRIVACY-SENSITIVE"


def _extract_last_user_question(messages_or_prompt: Union[str, Sequence[Mapping[str, Any]]]) -> str:
    if isinstance(messages_or_prompt, str):
        txt = messages_or_prompt.strip()
        m = re.search(r"USER_QUESTION:\s*(.+)$", txt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return txt
    for msg in reversed(messages_or_prompt):
        if (msg.get("role") or "").lower() != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            parts = [str(p.get("text") or "") for p in content if isinstance(p, Mapping)]
            joined = "\n".join(p for p in parts if p).strip()
        else:
            joined = str(content or "").strip()
        if joined:
            m = re.search(r"USER_QUESTION:\s*(.+)$", joined, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else joined
    return ""


def _build_citation_lookup(
    condensed: Optional[CondensedChartResult],
    citation_map: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, int], Set[str], int]:
    """Build exact UUID->[N] lookup and represented-UUID set from structured Layer-2 mappings."""
    uuid_to_cit: Dict[str, int] = {}
    represented_uuids: Set[str] = set()
    max_cit = 0

    def _register(idx_val: Any, ruuid: Any, members: Any) -> None:
        nonlocal max_cit
        if not isinstance(idx_val, int) or idx_val <= 0:
            return
        max_cit = max(max_cit, idx_val)
        r_s = str(ruuid or "").strip()
        if r_s:
            uuid_to_cit.setdefault(r_s, idx_val)
            represented_uuids.add(r_s)
        if isinstance(members, (list, tuple, set)):
            for m in members:
                ms = str(m or "").strip()
                if ms:
                    uuid_to_cit.setdefault(ms, idx_val)
                    represented_uuids.add(ms)

    if condensed is not None:
        for m in condensed.mappings or ():
            if isinstance(m, Mapping):
                _register(m.get("index"), m.get("resourceUuid"), m.get("memberUuids"))

    if citation_map is not None:
        for k_str, entry in citation_map.items():
            if not isinstance(entry, Mapping):
                continue
            idx_raw = entry.get("index")
            idx_val = int(idx_raw) if isinstance(idx_raw, int) else (int(k_str) if str(k_str).isdigit() else None)
            _register(
                idx_val,
                entry.get("id") or entry.get("resourceUuid"),
                entry.get("memberIds") or entry.get("memberUuids"),
            )

    return uuid_to_cit, represented_uuids, max_cit


class ClinicalEntityGraph:
    """Typed patient-specific clinical entity graph constructed from ExtractedChart."""

    def __init__(self, patient_id: str = "", patient_name: str = "") -> None:
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.nodes: Dict[str, ClinicalNode] = {}
        self.edges: List[ClinicalEdge] = []
        self._edge_keys: Set[Tuple[str, str, str, str]] = set()
        self.conflict_alerts: List[str] = []

    def add_node(self, node: ClinicalNode) -> ClinicalNode:
        existing = self.nodes.get(node.node_id)
        if existing is not None:
            if existing.citation_idx is None and node.citation_idx is not None:
                existing.citation_idx = node.citation_idx
            if not existing.resource_uuid and node.resource_uuid:
                existing.resource_uuid = node.resource_uuid
            return existing
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: ClinicalEdge) -> None:
        key = (edge.source_id, edge.target_id, edge.edge_type, edge.rationale)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(edge)

    def nodes_by_category(self, category: str) -> List[ClinicalNode]:
        return [n for n in self.nodes.values() if n.category == category]

    def edge_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.edges:
            counts[e.edge_type] = counts.get(e.edge_type, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "edge_breakdown": self.edge_breakdown(),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "category": n.category,
                    "label": n.label,
                    "code": n.code,
                    "citation_idx": n.citation_idx,
                    "resource_uuid": n.resource_uuid,
                    "date": n.date_str,
                    "status": n.status,
                    "value_summary": n.value_summary,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type,
                    "rationale": e.rationale,
                    "kb_source": e.kb_source,
                    "source_citation": e.source_citation,
                    "target_citation": e.target_citation,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def build(
        cls,
        source: Any,
        *,
        extracted: Optional[ExtractedChart] = None,
        citation_by_uuid: Optional[Mapping[str, int]] = None,
        sharing_posture: str = "patient_self",
    ) -> "ClinicalEntityGraph":
        """Build a ``ClinicalEntityGraph`` from an ``ExtractedChart``, FHIR Bundle, or empty fallback."""
        if isinstance(source, ExtractedChart):
            return cls.from_extracted_chart(
                source, citation_by_uuid=citation_by_uuid, sharing_posture=sharing_posture
            )
        if extracted is not None:
            return cls.from_extracted_chart(
                extracted, citation_by_uuid=citation_by_uuid, sharing_posture=sharing_posture
            )
        if isinstance(source, Mapping) and source.get("resourceType") == "Bundle":
            return cls.from_bundle(source, sharing_posture=sharing_posture)
        return cls()

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        sharing_posture: str = "patient_self",
    ) -> "ClinicalEntityGraph":
        return cls.from_extracted_chart(extract_bundle(bundle), sharing_posture=sharing_posture)

    @classmethod
    def from_extracted_chart(
        cls,
        extracted: ExtractedChart,
        *,
        citation_by_uuid: Optional[Mapping[str, int]] = None,
        sharing_posture: str = "patient_self",
    ) -> "ClinicalEntityGraph":
        graph = cls(patient_id=extracted.patient_id, patient_name=extracted.patient_name)
        cit_map = dict(citation_by_uuid or {})
        is_external = sharing_posture == "external_share"

        def _cit_for(item: ClinicalItem) -> Optional[int]:
            if item.resource_uuid and item.resource_uuid in cit_map:
                return cit_map[item.resource_uuid]
            for m_uuid in item.member_uuids:
                if m_uuid in cit_map:
                    return cit_map[m_uuid]
            return None

        redacted_uuids: Set[str] = set()
        kept_items: List[ClinicalItem] = []
        org_nodes: Dict[str, ClinicalNode] = {}

        def _register_org(org_name: Optional[str], src_node: ClinicalNode) -> None:
            org = (org_name or "").strip()
            if not org:
                return
            org_id = f"org:{_slugify(org)}"
            org_node = org_nodes.get(org_id)
            if org_node is None:
                org_node = graph.add_node(
                    ClinicalNode(node_id=org_id, category="Organization", label=org)
                )
                org_nodes[org_id] = org_node
            graph.add_edge(
                ClinicalEdge(
                    source_id=src_node.node_id,
                    target_id=org_node.node_id,
                    edge_type="REPORTED_BY_ORG",
                    rationale=f"Documented by {org}",
                    source_citation=src_node.citation_idx,
                )
            )

        for idx, item in enumerate(extracted.items):
            if item.section in {"patient", "manifest"}:
                continue
            if is_external and _is_sensitive_item(item):
                if item.resource_uuid:
                    redacted_uuids.add(item.resource_uuid)
                redacted_uuids.update(u for u in item.member_uuids if u)
                continue

            kept_items.append(item)
            cat = _SECTION_TO_CATEGORY.get(item.section, "Observation")
            r_uuid = item.resource_uuid or f"{cat.lower()}-{idx}"
            node_id = f"{cat.lower()}:{r_uuid}"
            raw_note = ""
            summary = item.status or ""
            if cat == "DocumentReference":
                raw_note = str(
                    item.metadata.get("raw_decoded_text")
                    or item.metadata.get("full_text")
                    or item.text
                    or ""
                ).strip()
                summary = item.source_org or ""
            elif cat in {"Medication", "Condition", "AllergyIntolerance", "CarePlan"}:
                summary = item.status or ""
            elif cat == "Immunization":
                summary = item.date or item.status or ""

            node = graph.add_node(
                ClinicalNode(
                    node_id=node_id,
                    category=cat,
                    label=item.display or item.code or cat,
                    code=item.code or "",
                    citation_idx=_cit_for(item),
                    resource_uuid=item.resource_uuid or "",
                    date_str=item.date or "",
                    status=item.status or "",
                    value_summary=summary,
                    raw_line=raw_note or item.text,
                    metadata={
                        "section": item.section,
                        "ingredient_key": item.ingredient_key or "",
                        "dosage": list(item.metadata.get("dosage") or []),
                        "therapeutic_class": str(item.metadata.get("therapeutic_class") or ""),
                    },
                )
            )
            _register_org(item.source_org, node)

        # Group numeric_points by code/concept into trajectory Observation nodes
        series_by_key: Dict[str, List[NumericPoint]] = {}
        for pt in extracted.numeric_points:
            if pt.resource_uuid and pt.resource_uuid in redacted_uuids:
                continue
            if is_external and classify_sensitivity(pt.code, pt.concept, [pt.category]) == "SDOH/PRIVACY-SENSITIVE":
                continue
            key = f"{pt.code or ''}:{pt.concept}".strip(":")
            series_by_key.setdefault(key, []).append(pt)

        for code_key, points in series_by_key.items():
            points_sorted = sorted(points, key=lambda p: p.date or "")
            latest = points_sorted[-1]
            abn_tag = f" [{latest.interpretation}]" if latest.interpretation and latest.is_abnormal else ""
            date_tag = f" on {latest.date[:10]}" if latest.date else " (date unknown)"
            prev_vals = ", ".join(
                f"{p.value:g} ({p.date[:10] if p.date else 'undated'})"
                for p in points_sorted[-4:-1]
            )
            val_summary = f"latest {latest.value:g} {latest.unit}{abn_tag}{date_tag}".strip()
            if prev_vals:
                val_summary += f"; prev: {prev_vals}"
            obs_id = f"obs:{_slugify(code_key)}"
            graph.add_node(
                ClinicalNode(
                    node_id=obs_id,
                    category="Observation",
                    label=latest.concept or code_key,
                    code=latest.code,
                    citation_idx=cit_map.get(latest.resource_uuid),
                    resource_uuid=latest.resource_uuid,
                    date_str=latest.date,
                    status="final",
                    value_summary=val_summary,
                    metadata={
                        "latest_value": latest.value,
                        "unit": latest.unit,
                        "count": len(points_sorted),
                        "is_abnormal": latest.is_abnormal,
                        "interpretation": latest.interpretation,
                        "min_value": min(p.value for p in points_sorted),
                        "max_value": max(p.value for p in points_sorted),
                    },
                )
            )

        graph._infer_clinical_edges(kept_items, extracted.birth_date)
        return graph

    def _infer_clinical_edges(self, kept_items: Sequence[ClinicalItem], birth_date: str) -> None:
        conds = self.nodes_by_category("Condition")
        meds = self.nodes_by_category("Medication")
        allergies = self.nodes_by_category("AllergyIntolerance")
        obs = self.nodes_by_category("Observation")
        notes = self.nodes_by_category("DocumentReference")

        node_by_uuid: Dict[str, ClinicalNode] = {
            n.resource_uuid: n for n in self.nodes.values() if n.resource_uuid
        }

        # 1. Layer-0 normalizer reconciliation edges (Conditions, Meds, Immunizations)
        cond_dicts = [
            {
                "resourceUuid": it.resource_uuid,
                "code": it.code,
                "display": it.display,
                "status": it.status or "active",
                "onset": (it.metadata.get("onset") or it.date or "") if isinstance(it.metadata, Mapping) else (it.date or ""),
                "abatement": (it.metadata.get("abatement") or it.end_date or "") if isinstance(it.metadata, Mapping) else (it.end_date or ""),
                "source_org": it.source_org or "",
            }
            for it in kept_items
            if it.section in {"condition_active", "condition_resolved"} and it.resource_uuid
        ]
        active_c_dicts = [c for c in cond_dicts if c["status"] in {"active", "recurrence", "relapse"}]
        resolved_c_dicts = [c for c in cond_dicts if c not in active_c_dicts]
        cond_conflicts = detect_condition_conflicts(active_c_dicts, resolved_c_dicts)
        for rid, note_txt in cond_conflicts.items():
            src_node = node_by_uuid.get(rid)
            if src_node is not None:
                self.conflict_alerts.append(f"{src_node.short_display()}: {note_txt}")
                peers = [
                    n for n in conds
                    if n.node_id != src_node.node_id
                    and n.resource_uuid in cond_conflicts
                    and cond_conflicts[n.resource_uuid] == note_txt
                ]
                tgt = peers[0] if peers else src_node
                self.add_edge(
                    ClinicalEdge(
                        source_id=src_node.node_id,
                        target_id=tgt.node_id,
                        edge_type="CONFLICTS_WITH",
                        rationale=note_txt,
                        kb_source="Condition Reconciliation Normalizer",
                        source_citation=src_node.citation_idx,
                        target_citation=tgt.citation_idx if tgt is not src_node else None,
                    )
                )

        active_med_dicts = [
            {
                "resourceUuid": it.resource_uuid,
                "ingredient_key": it.ingredient_key or normalize_drug_ingredient(it.code, it.display),
                "display": it.display,
                "setting": str((it.metadata or {}).get("setting") or "prescribed"),
                "date": it.date or "",
            }
            for it in kept_items
            if it.section == "med_active" and it.resource_uuid
        ]
        med_conflicts = detect_medication_conflicts(active_med_dicts)
        for rid, note_txt in med_conflicts.items():
            src_node = node_by_uuid.get(rid)
            if src_node is not None:
                self.conflict_alerts.append(f"{src_node.short_display()}: {note_txt}")
                other_m = [
                    node_by_uuid[o_id]
                    for o_id, o_txt in med_conflicts.items()
                    if o_id != rid and o_id in node_by_uuid and o_txt == note_txt
                ]
                tgt = other_m[0] if other_m else src_node
                self.add_edge(
                    ClinicalEdge(
                        source_id=src_node.node_id,
                        target_id=tgt.node_id,
                        edge_type="CONFLICTS_WITH",
                        rationale=note_txt,
                        kb_source="Medication Safety & Reconciliation Normalizer",
                        source_citation=src_node.citation_idx,
                        target_citation=tgt.citation_idx if tgt is not src_node else None,
                    )
                )

        imm_dicts = [
            {
                "resourceUuid": it.resource_uuid,
                "code": it.code,
                "display": it.display,
                "date": it.date or "",
                "status": it.status or "completed",
                "group": classify_vaccine_group(it.code, it.display),
            }
            for it in kept_items
            if it.section == "immunization" and it.resource_uuid
        ]
        imm_anomalies = annotate_immunization_anomalies(imm_dicts, birth_date)
        for rid, anom_txt in imm_anomalies.items():
            imm_node = node_by_uuid.get(rid)
            if imm_node is not None:
                self.conflict_alerts.append(f"{imm_node.short_display()}: {anom_txt}")
                self.add_edge(
                    ClinicalEdge(
                        source_id=imm_node.node_id,
                        target_id=imm_node.node_id,
                        edge_type="CONFLICTS_WITH",
                        rationale=anom_txt,
                        kb_source="CDC ACIP Schedule Validation",
                        source_citation=imm_node.citation_idx,
                    )
                )

        # 2. Allergy cross-reactivity from kb_data/rxnorm_lookup.json
        rx_rules = get_rxnorm_lookup().get("allergy_cross_reactivity_rules") or []
        for al in allergies:
            al_low = f"{al.label} {al.raw_line}".lower()
            for rule in rx_rules:
                if not isinstance(rule, Mapping):
                    continue
                al_terms = [str(t).lower() for t in (rule.get("allergen_terms") or ())]
                med_terms = [str(t).lower() for t in (rule.get("medication_terms") or ())]
                if not any(t in al_low for t in al_terms):
                    continue
                for m in meds:
                    m_low = f"{m.label} {m.metadata.get('ingredient_key', '')}".lower()
                    if any(mt in m_low for mt in med_terms):
                        self.add_edge(
                            ClinicalEdge(
                                source_id=al.node_id,
                                target_id=m.node_id,
                                edge_type="CROSS_REACTS_WITH",
                                rationale=str(rule.get("rationale") or "Potential allergy cross-reactivity"),
                                kb_source="RxNorm Allergy Cross-Reactivity Rules",
                                source_citation=al.citation_idx,
                                target_citation=m.citation_idx,
                            )
                        )

        # 3. Drug -> Condition TREATS & Safety/Contraindication edges
        for m in meds:
            m_low = f"{m.label} {m.metadata.get('ingredient_key', '')} {m.raw_line}".lower()
            for drug_kws, cond_kws, rationale, kb_src in _DRUG_INDICATION_KEYWORDS:
                if any(dk in m_low for dk in drug_kws):
                    for c in conds:
                        c_low = f"{c.label} {c.raw_line}".lower()
                        if any(ck in c_low for ck in cond_kws):
                            self.add_edge(
                                ClinicalEdge(
                                    source_id=m.node_id,
                                    target_id=c.node_id,
                                    edge_type="TREATS",
                                    rationale=rationale,
                                    kb_source=kb_src,
                                    source_citation=m.citation_idx,
                                    target_citation=c.citation_idx,
                                )
                            )

            for drug_kws, risk_kws, rationale, kb_src in _SAFETY_CONTRAINDICATION_RULES:
                if not any(dk in m_low for dk in drug_kws):
                    continue
                for tgt in conds + obs + meds:
                    if tgt.node_id == m.node_id:
                        continue
                    t_low = f"{tgt.label} {tgt.value_summary} {tgt.raw_line}".lower()
                    if any(rk in t_low for rk in risk_kws):
                        self.add_edge(
                            ClinicalEdge(
                                source_id=m.node_id,
                                target_id=tgt.node_id,
                                edge_type="CONTRAINDICATED_OR_DOSE_ADJUST_IN",
                                rationale=rationale,
                                kb_source=kb_src,
                                source_citation=m.citation_idx,
                                target_citation=tgt.citation_idx,
                            )
                        )

        # 4. Condition/Medication -> Observation MONITORING edges
        for entity in conds + meds:
            e_low = f"{entity.label} {entity.metadata.get('ingredient_key', '')} {entity.raw_line}".lower()
            for ent_kws, lab_kws, rationale in _MONITORING_RULES:
                if not any(ek in e_low for ek in ent_kws):
                    continue
                for ob in obs:
                    ob_low = f"{ob.label} {ob.value_summary}".lower()
                    if any(lk in ob_low for lk in lab_kws):
                        self.add_edge(
                            ClinicalEdge(
                                source_id=entity.node_id,
                                target_id=ob.node_id,
                                edge_type="MONITORED_BY",
                                rationale=rationale,
                                kb_source="Clinical Monitoring & Cardiorenal Panel Link",
                                source_citation=entity.citation_idx,
                                target_citation=ob.citation_idx,
                            )
                        )

        # 5. ClinicalNote -> Medication hold/discontinue/modification reconciliation edges
        for n in notes:
            n_low = (n.raw_line or n.value_summary or "").lower()
            if not n_low or not any(cue in n_low for cue in _NOTE_HOLD_OR_MOD_CUES):
                continue
            for m in meds:
                ing = str(m.metadata.get("ingredient_key") or "").lower()
                med_base = (m.label.split()[0] if m.label else "").lower()
                probe = ing if len(ing) >= 4 else med_base
                if not probe or len(probe) < 4 or probe not in n_low:
                    continue
                for sent in re.split(r"(?<=[.!?;])\s+|\n+", n.raw_line or ""):
                    s_low = sent.lower()
                    if probe in s_low and any(cue in s_low for cue in _NOTE_HOLD_OR_MOD_CUES):
                        snippet = sent.strip()
                        if len(snippet) > 160:
                            snippet = snippet[:157].rstrip() + "..."
                        self.add_edge(
                            ClinicalEdge(
                                source_id=m.node_id,
                                target_id=n.node_id,
                                edge_type="CONTRADICTED_OR_MODIFIED_BY_NOTE",
                                rationale=f"Clinical note statement ({n.date_str or 'undated'}): {snippet}",
                                kb_source="Longitudinal Note-to-Medication Reconciliation",
                                source_citation=m.citation_idx,
                                target_citation=n.citation_idx,
                            )
                        )
                        break

    def format_subgraph_lines(self, question: str = "", *, max_lines: Optional[int] = None) -> List[str]:
        q_terms = _query_terms(question)

        def _score_edge(edge: ClinicalEdge) -> Tuple[int, int, str]:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            hay = " ".join(
                [
                    src.label if src else "",
                    tgt.label if tgt else "",
                    edge.rationale,
                ]
            ).lower()
            overlap = sum(1 for t in q_terms if t in hay)
            prio = _EDGE_PRIORITY.get(edge.edge_type, 9)
            return (-overlap, prio, edge.edge_type)

        ranked = sorted(
            [e for e in self.edges if e.edge_type != "REPORTED_BY_ORG"],
            key=_score_edge,
        )
        seen: Set[str] = set()
        lines: List[str] = []
        for edge in ranked:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src is None or tgt is None:
                continue
            if tgt.node_id == src.node_id:
                line = f"- {edge.edge_type}: {src.short_display()} — {edge.rationale}"
            else:
                line = (
                    f"- {edge.edge_type}: {src.short_display()} -> {tgt.short_display()} "
                    f"({edge.rationale})"
                )
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
        return lines


def _recover_omitted_tier0_items(
    *,
    extracted: Optional[ExtractedChart],
    represented_uuids: Set[str],
    question: str,
    sharing_posture: str,
    max_cit_before: int,
    max_recover: int = 8,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Recover highest-relevance omitted items using typed ``ExtractedChart.items``."""
    if extracted is None:
        return []
    is_external = sharing_posture == "external_share"
    q_terms = _query_terms(question)

    candidates: List[Tuple[Tuple[int, int, str], ClinicalItem]] = []
    for it in extracted.items:
        if it.section in {"patient", "manifest"}:
            continue
        if is_external and _is_sensitive_item(it):
            continue
        uuids_here = {u for u in (it.resource_uuid, *it.member_uuids) if u}
        if not uuids_here or uuids_here.issubset(represented_uuids):
            continue
        hay = f"{it.display} {it.text} {it.source_org or ''} {it.date or ''}".lower()
        overlap = sum(1 for t in q_terms if t in hay)
        if q_terms and overlap == 0:
            continue
        sec_prio = _SECTION_OVERFLOW_PRIORITY.get(it.section, 15)
        is_abn = 0 if (isinstance(it.metadata, Mapping) and it.metadata.get("is_abnormal")) else 1
        candidates.append(((-overlap, sec_prio, is_abn, it.date or ""), it))  # type: ignore[arg-type]

    candidates.sort(key=lambda pair: pair[0])
    recovered: List[Tuple[str, Dict[str, Any]]] = []
    next_idx = max_cit_before + 1
    seen_uuids: Set[str] = set()

    for _, it in candidates:
        ruuid = it.resource_uuid or (it.member_uuids[0] if it.member_uuids else "")
        if not ruuid or ruuid in seen_uuids:
            continue
        seen_uuids.add(ruuid)
        dt_part = f"({it.date}) " if it.date else ""
        snippet = it.text or it.display or it.resource_type
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        line = f"  [{next_idx}] {dt_part}{it.resource_type}: {snippet}"
        mapping_dict: Dict[str, Any] = {
            "index": next_idx,
            "resourceType": it.resource_type,
            "resourceUuid": ruuid,
            "date": it.date or "",
            "display": it.display or it.resource_type,
            "memberUuids": list(it.member_uuids) if it.member_uuids else [ruuid],
        }
        recovered.append((line, mapping_dict))
        next_idx += 1
        if len(recovered) >= max_recover:
            break

    return recovered


def build_wahs_hub_enrichment_block(
    messages_or_prompt: Union[str, Sequence[Mapping[str, Any]]],
    *,
    extracted: Optional[ExtractedChart] = None,
    condensed: Optional[CondensedChartResult] = None,
    citation_map: Optional[Mapping[str, Any]] = None,
    max_tokens: Optional[int] = None,
    sharing_posture: str = "patient_self",
    token_counter: Optional[Callable[[str], int]] = None,
) -> HubEnrichmentResult:
    """Construct Layer-3 clinical entity graph, Tier-0 overflow recovery, and KB enrichment.

    When ``max_tokens`` is ``None``, emits the full coherent enrichment block without
    artificial truncation. When an integer ``max_tokens`` ceiling is supplied,
    progressively trims lower-priority items to stay strictly within ``max_tokens``.
    """
    active_question = _extract_last_user_question(messages_or_prompt)
    uuid_to_cit, represented_uuids, max_cit_before = _build_citation_lookup(
        condensed, citation_map
    )
    if max_cit_before == 0:
        combined = (
            messages_or_prompt
            if isinstance(messages_or_prompt, str)
            else "\n".join(str(m.get("content") or "") for m in messages_or_prompt)
        )
        found_cits = [int(x) for x in _CITATION_TAG_RE.findall(combined) if x.isdigit()]
        max_cit_before = max(found_cits, default=0)

    # Read omission stats directly from typed Layer-2 manifest when available.
    omitted_count = 0
    omitted_by_cat: Dict[str, int] = {}
    if condensed is not None and isinstance(condensed.completeness_manifest, Mapping):
        omitted_count = int(condensed.completeness_manifest.get("omitted_clinical_items_count") or 0)
        raw_by_sec = condensed.completeness_manifest.get("omitted_by_section")
        if isinstance(raw_by_sec, Mapping):
            omitted_by_cat = {str(k): int(v) for k, v in raw_by_sec.items() if int(v or 0) > 0}
    elif extracted is not None and represented_uuids:
        is_ext = sharing_posture == "external_share"
        for it in extracted.items:
            if it.section in {"patient", "manifest"}:
                continue
            if is_ext and _is_sensitive_item(it):
                continue
            u_set = {u for u in (it.resource_uuid, *it.member_uuids) if u}
            if u_set and not u_set.issubset(represented_uuids):
                omitted_count += 1
                omitted_by_cat[it.section] = omitted_by_cat.get(it.section, 0) + 1

    graph = ClinicalEntityGraph.build(
        extracted if extracted is not None else messages_or_prompt,
        extracted=extracted,
        citation_by_uuid=uuid_to_cit,
        sharing_posture=sharing_posture,
    )

    overflow_triggered = omitted_count > 0
    recovered_entries = (
        _recover_omitted_tier0_items(
            extracted=extracted,
            represented_uuids=represented_uuids,
            question=active_question,
            sharing_posture=sharing_posture,
            max_cit_before=max_cit_before,
            max_recover=8,
        )
        if overflow_triggered
        else []
    )

    chart_codes = (
        extract_chart_kb_codes(extracted, sharing_posture=sharing_posture)
        if extracted is not None
        else []
    )
    kb_hits = (
        search_kb(active_question, k=4, corpus="wahs", codes=chart_codes)
        if (active_question or chart_codes)
        else []
    )
    kb_lines: List[str] = []
    for hit in kb_hits:
        title = str(hit.get("title") or "Clinical Guideline").strip()
        summary = str(hit.get("summary") or "").strip()
        source = str(hit.get("source") or "").strip()
        if not summary:
            continue
        src_note = f" ({source})" if source else ""
        kb_lines.append(f"- {title}{src_note}: {summary}")

    subgraph_lines = graph.format_subgraph_lines(active_question)

    header_line = (
        "=== CONTEXT ENRICHMENT (Patient Entity Subgraph, Overflow Recovery, & Clinical Guidelines) ==="
    )
    instruction_line = (
        "Cite any recovered items below inline as [N]. Use ONLY subgraph links directly relevant "
        "to the user's question, and treat guidelines as general clinical reference rather than patient diagnoses."
    )
    cat_desc = (
        ", ".join(f"{k}={v}" for k, v in omitted_by_cat.items())
        if omitted_by_cat
        else f"total={omitted_count}"
    )

    def _render_block(
        rec_slice: List[Tuple[str, Dict[str, Any]]],
        sub_slice: List[str],
        kb_slice: List[str],
    ) -> str:
        if overflow_triggered:
            overflow_header = (
                f"1. Tier-0 Overflow Recovery (ACTIVE: omitted_clinical_items={omitted_count} [{cat_desc}]; "
                f"recovered={len(rec_slice)} records):"
            )
        else:
            overflow_header = (
                "1. Tier-0 Overflow Recovery: SKIPPED (omitted_clinical_items=0; "
                "all non-redacted clinical items are represented in the condensed chart)."
            )
        sections: List[str] = [header_line, instruction_line, overflow_header]
        if rec_slice:
            sections.extend(line for line, _ in rec_slice)
        sections.append(
            f"2. Patient Clinical Entity Subgraph ({len(graph.nodes)} nodes, {len(graph.edges)} edges):"
        )
        if sub_slice:
            sections.extend(sub_slice)
        else:
            sections.append("- No cross-entity conflicts or safety links triggered for this view.")
        sections.append(
            f"3. Question-Conditioned Clinical Knowledge Base ({len(kb_slice)} matched guidelines):"
        )
        if kb_slice:
            sections.extend(kb_slice)
        else:
            sections.append("- Standard clinical reconciliation rules apply.")
        return "\n".join(sections)

    count_tokens = token_counter or estimate_tokens_local
    cur_rec = list(recovered_entries)
    cur_sub = list(subgraph_lines)
    cur_kb = list(kb_lines)

    block_text = _render_block(cur_rec, cur_sub, cur_kb)
    if max_tokens is not None and max_tokens > 0:
        while count_tokens(block_text) > max_tokens:
            if len(cur_sub) > 3:
                cur_sub.pop()
            elif len(cur_kb) > 1:
                cur_kb.pop()
            elif len(cur_sub) > 1:
                cur_sub.pop()
            elif cur_kb:
                cur_kb.pop()
            elif len(cur_rec) > 1:
                cur_rec.pop()
            else:
                lo, hi = 0, len(block_text)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    cand = (
                        block_text[:mid].rsplit("\n", 1)[0]
                        if mid < len(block_text)
                        else block_text
                    )
                    if count_tokens(cand) <= max_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                block_text = block_text[:lo].rsplit("\n", 1)[0] if lo > 0 else ""
                if count_tokens(block_text) > max_tokens and lo > 0:
                    block_text = block_text[:lo]
                break
            block_text = _render_block(cur_rec, cur_sub, cur_kb)

    final_tokens = count_tokens(block_text)
    rec_mappings = [m for _, m in cur_rec]
    max_cit_after = max([max_cit_before] + [int(m["index"]) for m in rec_mappings])

    return HubEnrichmentResult(
        enrichment_text=block_text,
        enrichment_tokens=final_tokens,
        omitted_clinical_items=omitted_count,
        overflow_triggered=overflow_triggered,
        overflow_recovered_count=len(cur_rec),
        subgraph_nodes=len(graph.nodes),
        subgraph_edges=len(graph.edges),
        kb_Passages_matched=len(cur_kb),
        max_citation_idx_before=max_cit_before,
        max_citation_idx_after=max_cit_after,
        active_question=active_question,
        patient_name=graph.patient_name,
        omitted_by_category=omitted_by_cat,
        edge_Breakdown=graph.edge_breakdown(),
        recovered_mappings=rec_mappings,
    )
