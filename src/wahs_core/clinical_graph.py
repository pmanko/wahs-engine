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
from .kb_lookup import extract_chart_kb_codes, get_clinical_graph_rules, get_rxnorm_lookup
from .normalizers import (
    annotate_immunization_anomalies,
    classify_sensitivity,
    classify_vaccine_group,
    detect_condition_conflicts,
    detect_medication_conflicts,
    is_non_medical_admin_code,
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

_HIGH_PRIORITY_EDGE_TYPES: Set[str] = {
    "CROSS_REACTS_WITH",
    "CONTRAINDICATED_OR_DOSE_ADJUST_IN",
    "CONFLICTS_WITH",
    "CONTRADICTED_OR_MODIFIED_BY_NOTE",
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
    "write",
    "you",
    "your",
    "ask",
    "condition",
    "conditions",
    "disease",
    "draft",
    "function",
    "give",
    "letter",
    "level",
    "levels",
    "medicine",
    "medicines",
    "new",
    "problem",
    "problems",
    "result",
    "right",
    "share",
    "stage",
    "status",
    "value",
    "values",
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
            lbl = self.label[:61].rstrip() + "..." if len(self.label) > 64 else self.label
            date_suffix = f" ({self.date_str})" if self.date_str else ""
            return f"{tag}{lbl}{date_suffix}".strip()
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
    kb_passages_matched: int
    max_citation_idx_before: int
    max_citation_idx_after: int
    active_question: str = ""
    patient_name: str = ""
    omitted_by_category: Dict[str, int] = field(default_factory=dict)
    edge_breakdown: Dict[str, int] = field(default_factory=dict)
    recovered_mappings: List[Dict[str, Any]] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "omitted_clinical_items": self.omitted_clinical_items,
            "overflow_triggered": self.overflow_triggered,
            "overflow_recovered_count": self.overflow_recovered_count,
            "overflow_items_recovered": self.overflow_recovered_count,
            "subgraph_nodes": self.subgraph_nodes,
            "subgraph_edges": self.subgraph_edges,
            "kb_passages_matched": self.kb_passages_matched,
            "enrichment_tokens": self.enrichment_tokens,
            "max_citation_idx_before": self.max_citation_idx_before,
            "max_citation_idx_after": self.max_citation_idx_after,
            "patient_name": self.patient_name,
            "omitted_by_category": dict(self.omitted_by_category),
            "edge_breakdown": dict(self.edge_breakdown),
            "recovered_mappings": list(self.recovered_mappings),
        }


# ---------------------------------------------------------------------------
# Clinical Domain Rules for Graph Edge Construction (loaded from kb_data/clinical_graph_rules.json)
# ---------------------------------------------------------------------------

_NON_DRUG_PROBE_WORDS: Set[str] = {
    "admin",
    "administered",
    "capsule",
    "capsules",
    "delayed",
    "device",
    "dose",
    "drops",
    "extended",
    "flush",
    "injection",
    "inpatient",
    "liquid",
    "medication",
    "medications",
    "naclflush",
    "order",
    "oral",
    "packet",
    "patch",
    "release",
    "sodium",
    "solution",
    "spray",
    "succinate",
    "suspension",
    "syringe",
    "tablet",
    "tablets",
    "topical",
    "unspecified",
    "vial",
}

_TRAILING_WRAPPER_SPLIT_RE = re.compile(
    r"\n+\s*(?:GROUNDING\s*&\s*FORMATTING\s*RULES\s*:|===\s*[A-Z0-9])",
    flags=re.IGNORECASE,
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


def _med_name_probes(node: ClinicalNode) -> Set[str]:
    """Extract normalized ingredient and brand/generic name probes for note matching."""
    probes: Set[str] = set()
    sources = [
        str(node.metadata.get("ingredient_key") or ""),
        normalize_drug_ingredient(node.label),
    ]
    for src in sources:
        for tok in re.findall(r"[a-z][a-z0-9-]{3,}", src.lower()):
            if tok not in _NON_DRUG_PROBE_WORDS:
                probes.add(tok)
    lbl = (node.label or "").strip().lower()
    for br in re.findall(r"\[([^\]]+)\]", lbl):
        for tok in re.findall(r"[a-z][a-z0-9-]{3,}", br):
            if tok not in _NON_DRUG_PROBE_WORDS:
                probes.add(tok)
    first_word = re.match(r"^([a-z][a-z0-9-]{3,})", lbl)
    if first_word:
        w = first_word.group(1)
        if w not in _NON_DRUG_PROBE_WORDS:
            probes.add(w)
    return probes


def _dedup_nodes_by_identity(nodes: Sequence[ClinicalNode]) -> List[ClinicalNode]:
    """Deduplicate clinical nodes by normalized ingredient/code/label, preferring newest and cited."""
    ordered = sorted(
        nodes,
        key=lambda n: (n.date_str or "", 1 if n.citation_idx is not None else 0),
        reverse=True,
    )
    out: List[ClinicalNode] = []
    seen: Set[str] = set()
    for n in ordered:
        ing = str(n.metadata.get("ingredient_key") or "").strip().lower()
        key = ing if ing else (n.code or n.label or n.node_id).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def extract_clean_user_query(text: str) -> str:
    """Strip wrapper headers (`USER_QUESTION:`) and trailing harness/enrichment blocks."""
    txt = str(text or "").strip()
    if not txt:
        return ""
    m = re.search(r"USER_QUESTION:\s*(.+)$", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        txt = m.group(1).strip()
    return _TRAILING_WRAPPER_SPLIT_RE.split(txt, maxsplit=1)[0].strip()


def _extract_last_user_question(messages_or_prompt: Union[str, Sequence[Mapping[str, Any]]]) -> str:
    if isinstance(messages_or_prompt, str):
        return extract_clean_user_query(messages_or_prompt)
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
            return extract_clean_user_query(joined)
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
        if edge.edge_type == "CONFLICTS_WITH":
            src_k, tgt_k = sorted((edge.source_id, edge.target_id))
            key = (src_k, tgt_k, edge.edge_type, edge.rationale)
        else:
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

        numeric_point_uuids: Set[str] = {
            pt.resource_uuid for pt in extracted.numeric_points if pt.resource_uuid
        }

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
            if (
                cat == "Observation"
                and item.section in {"lab_observation", "vital_sign"}
                and item.resource_uuid
                and item.resource_uuid in numeric_point_uuids
            ):
                continue
            item_setting = str(item.metadata.get("setting") or "")
            if item.section == "med_historical" and (
                item_setting.lower() == "inpatient/administered"
                or (item.display or "").strip().lower().startswith(("inpatient ", "naclflush"))
                or bool(re.match(r"^\d{8,}\b", (item.display or "").strip()))
            ):
                continue
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
                        "setting": item_setting,
                        "ingredient_key": item.ingredient_key or "",
                        "dosage": list(item.metadata.get("dosage") or []),
                        "therapeutic_class": str(item.metadata.get("therapeutic_class") or ""),
                        "verification_status": str(item.metadata.get("verification_status") or "").lower(),
                    },
                )
            )
            if item.section not in {"med_historical", "condition_resolved"}:
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
            val_summary = f"latest {latest.value:g} {latest.unit}{abn_tag}{date_tag}".strip()
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
                "ingredient_key": it.ingredient_key or normalize_drug_ingredient(it.display),
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

        active_med_nodes = [
            m
            for m in meds
            if m.metadata.get("section") == "med_active"
            and not is_non_medical_admin_code(m.code)
        ]
        scoped_meds = _dedup_nodes_by_identity(active_med_nodes)
        active_cond_nodes = [
            c
            for c in conds
            if c.metadata.get("section") == "condition_active"
            and not is_non_medical_admin_code(c.code)
        ]
        scoped_conds = _dedup_nodes_by_identity(active_cond_nodes)

        # 2. Allergy cross-reactivity from kb_data/rxnorm_lookup.json (actionable allergies + active meds only)
        def _is_actionable_allergy(al_node: ClinicalNode) -> bool:
            c_st = (al_node.status or "").strip().lower()
            if c_st in {"inactive", "resolved", "entered-in-error"}:
                return False
            v_st = str(al_node.metadata.get("verification_status") or "").strip().lower()
            if v_st in {"refuted", "entered-in-error"}:
                return False
            al_raw = f"{al_node.label} {al_node.raw_line}".lower()
            if "verification: refuted" in al_raw or "verification: entered-in-error" in al_raw:
                return False
            return True

        rx_rules = get_rxnorm_lookup().get("allergy_cross_reactivity_rules") or []
        for al in allergies:
            if not _is_actionable_allergy(al):
                continue
            al_low = f"{al.label} {al.raw_line}".lower()
            for rule in rx_rules:
                if not isinstance(rule, Mapping):
                    continue
                al_terms = [
                    str(t).lower()
                    for t in (rule.get("allergen_tokens") or rule.get("allergen_terms") or ())
                ]
                med_terms = [
                    str(t).lower()
                    for t in (rule.get("med_tokens") or rule.get("medication_terms") or ())
                ]
                if not any(t in al_low for t in al_terms):
                    continue
                for m in scoped_meds:
                    m_low = f"{m.label} {m.metadata.get('ingredient_key', '')}".lower()
                    if any(mt in m_low for mt in med_terms):
                        self.add_edge(
                            ClinicalEdge(
                                source_id=al.node_id,
                                target_id=m.node_id,
                                edge_type="CROSS_REACTS_WITH",
                                rationale=str(rule.get("rationale") or "Potential allergy cross-reactivity"),
                                kb_source=str(rule.get("kb_source") or "RxNorm Allergy Cross-Reactivity Rules"),
                                source_citation=al.citation_idx,
                                target_citation=m.citation_idx,
                            )
                        )

        graph_rules = get_clinical_graph_rules()
        indication_rules = graph_rules.get("drug_indication_rules") or ()
        safety_rules = graph_rules.get("safety_contraindication_rules") or ()
        monitoring_rules = graph_rules.get("monitoring_rules") or ()
        hold_cues = tuple(str(c).lower() for c in (graph_rules.get("note_hold_or_mod_cues") or ()))
        resume_cues = tuple(str(c).lower() for c in (graph_rules.get("note_resume_or_continue_cues") or ()))

        # 3. Drug -> Condition TREATS & Safety/Contraindication edges (scoped to active/trajectory entities)
        conflict_pairs: Set[frozenset[str]] = {
            frozenset((e.source_id, e.target_id))
            for e in self.edges
            if e.edge_type == "CONFLICTS_WITH" and e.source_id != e.target_id
        }

        def _is_clinically_significant_obs(tgt: ClinicalNode, t_low: str) -> bool:
            if tgt.category != "Observation":
                return True
            is_abn = bool(tgt.metadata.get("is_abnormal"))
            interp = str(tgt.metadata.get("interpretation") or "").strip().lower()
            is_high = interp in {"h", "hh", "high", "above high normal", "critical high", ">"}
            is_low = interp in {"l", "ll", "low", "below low normal", "critical low", "<"}
            val = tgt.metadata.get("latest_value")
            has_num = isinstance(val, (int, float))
            unit_compact = re.sub(r"\s+", "", str(tgt.metadata.get("unit") or "").strip().lower())

            if "egfr" in t_low or "glomerular filtration" in t_low:
                if has_num and (not unit_compact or "ml/min" in unit_compact):
                    return float(val) < 60
                return is_abn and is_low
            if "creatinine" in t_low:
                if has_num:
                    if unit_compact in {"", "mg/dl", "mg/100ml", "mg%"}:
                        return float(val) >= 1.5
                    if unit_compact == "mg/l":
                        return (float(val) / 10.0) >= 1.5
                    if unit_compact in {"umol/l", "µmol/l", "μmol/l", "mcmol/l", "micromol/l", "mmol/m3"}:
                        return (float(val) / 88.42) >= 1.5
                    if unit_compact == "mmol/l":
                        return ((float(val) * 1000.0) / 88.42) >= 1.5
                return is_abn and is_high
            if "potassium" in t_low:
                if has_num and unit_compact in {"", "meq/l", "mmol/l", "meq/liter", "mmol/liter"}:
                    return float(val) >= 5.0 or float(val) <= 3.4
                return is_abn and (is_high or is_low or not interp)
            if "inr" in t_low:
                if has_num and unit_compact in {"", "{inr}", "inr", "ratio", "1", "none"}:
                    return float(val) >= 3.5
                return is_abn and is_high
            if "hemoglobin" in t_low and "a1c" not in t_low:
                if has_num:
                    if unit_compact in {"", "g/dl", "gm/dl", "g/100ml", "g%"}:
                        return float(val) < 11.5
                    if unit_compact in {"g/l", "gm/l", "mg/ml"}:
                        return (float(val) / 10.0) < 11.5
                    if unit_compact == "mmol/l":
                        return (float(val) * 1.6114) < 11.5
                return is_abn and is_low
            if "glucose" in t_low:
                if has_num:
                    if unit_compact in {"", "mg/dl", "mg/100ml", "mg%"}:
                        return float(val) < 70
                    if unit_compact == "mmol/l":
                        return float(val) < 3.9
                return is_abn and is_low
            return is_abn

        for m in scoped_meds:
            m_low = f"{m.label} {m.metadata.get('ingredient_key', '')} {m.raw_line}".lower()
            for rule in indication_rules:
                if not isinstance(rule, Mapping):
                    continue
                drug_kws = tuple(str(t).lower() for t in (rule.get("med_tokens") or ()))
                cond_kws = tuple(str(t).lower() for t in (rule.get("target_tokens") or ()))
                rationale = str(rule.get("rationale") or "")
                kb_src = str(rule.get("kb_source") or "")
                if any(dk in m_low for dk in drug_kws):
                    for c in scoped_conds:
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

            for rule in safety_rules:
                if not isinstance(rule, Mapping):
                    continue
                drug_kws = tuple(str(t).lower() for t in (rule.get("med_tokens") or ()))
                risk_kws = tuple(str(t).lower() for t in (rule.get("target_tokens") or ()))
                rationale = str(rule.get("rationale") or "")
                kb_src = str(rule.get("kb_source") or "")
                if not any(dk in m_low for dk in drug_kws):
                    continue
                matched_tgts: List[ClinicalNode] = []
                for tgt in scoped_conds + obs + scoped_meds:
                    if tgt.node_id == m.node_id:
                        continue
                    if frozenset((m.node_id, tgt.node_id)) in conflict_pairs:
                        continue
                    t_low = f"{tgt.label} {tgt.value_summary} {tgt.raw_line}".lower()
                    if not any(
                        rk in t_low and not (rk == "hemoglobin" and "a1c" in t_low)
                        for rk in risk_kws
                    ):
                        continue
                    if not _is_clinically_significant_obs(tgt, t_low):
                        continue
                    matched_tgts.append(tgt)
                matched_tgts.sort(
                    key=lambda n: (1 if n.metadata.get("is_abnormal") else 0, n.date_str or ""),
                    reverse=True,
                )
                seen_tgt_stem: Set[Tuple[str, str]] = set()
                emitted_for_rule = 0
                for tgt in matched_tgts:
                    stem = re.split(r"[^a-z0-9]+", (tgt.label or "").strip().lower())[0]
                    stem_key = (tgt.category, stem)
                    if stem_key in seen_tgt_stem:
                        continue
                    seen_tgt_stem.add(stem_key)
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
                    emitted_for_rule += 1
                    if emitted_for_rule >= 2:
                        break

        # 4. Condition/Medication -> Observation MONITORING edges (active entities + trajectory obs)
        for entity in scoped_conds + scoped_meds:
            e_low = f"{entity.label} {entity.metadata.get('ingredient_key', '')} {entity.raw_line}".lower()
            seen_rationale_counts: Dict[str, int] = {}
            seen_rationale_stems: Dict[str, Set[str]] = {}
            for rule in monitoring_rules:
                if not isinstance(rule, Mapping):
                    continue
                ent_kws = tuple(str(t).lower() for t in (rule.get("trigger_tokens") or ()))
                lab_kws = tuple(str(t).lower() for t in (rule.get("obs_tokens") or ()))
                rationale = str(rule.get("rationale") or "")
                if not any(ek in e_low for ek in ent_kws):
                    continue
                matched_obs = [
                    ob
                    for ob in obs
                    if any(lk in f"{ob.label} {ob.value_summary}".lower() for lk in lab_kws)
                ]
                matched_obs.sort(
                    key=lambda o: (1 if o.metadata.get("is_abnormal") else 0, o.date_str or ""),
                    reverse=True,
                )
                stems = seen_rationale_stems.setdefault(rationale, set())
                for ob in matched_obs:
                    if seen_rationale_counts.get(rationale, 0) >= 3:
                        break
                    stem = re.split(r"[^a-z0-9]+", (ob.label or "").strip().lower())[0]
                    if stem and stem in stems:
                        continue
                    if stem:
                        stems.add(stem)
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
                    seen_rationale_counts[rationale] = seen_rationale_counts.get(rationale, 0) + 1

        # 5. ClinicalNote -> Medication hold/discontinue/reconcile & resume reconciliation edges
        notes_sorted = sorted(notes, key=lambda n: n.date_str or "", reverse=True)
        for m in scoped_meds:
            probes = _med_name_probes(m)
            if not probes:
                continue
            m_date = (m.date_str or "")[:10]
            latest_resume_date = ""
            matched_m = False
            for n in notes_sorted:
                raw_note = n.raw_line or ""
                note_body = raw_note or n.value_summary or ""
                n_low = note_body.lower()
                n_date = (n.date_str or "")[:10]
                if m_date and n_date and n_date < m_date:
                    continue
                if n_low and any(rc in n_low for rc in resume_cues):
                    for r_sent in re.split(r"(?<=[.!?;])\s+|\n+", note_body):
                        rs_low = r_sent.lower()
                        if (
                            any(rc in rs_low for rc in resume_cues)
                            and not re.search(
                                r"\b(?:do\s+not|don't|never|avoid|cannot)\s+(?:continue|resume|restart)\b",
                                rs_low,
                            )
                            and (probes & set(re.findall(r"[a-z][a-z0-9-]{3,}", rs_low)))
                        ):
                            if n_date > latest_resume_date:
                                latest_resume_date = n_date
                            break
                if latest_resume_date and n_date and n_date < latest_resume_date:
                    continue
                if not n_low or not any(cue in n_low for cue in hold_cues):
                    continue
                n_toks = set(re.findall(r"[a-z][a-z0-9-]{3,}", n_low))
                if not (probes & n_toks):
                    continue
                for sent in re.split(r"(?<=[.!?;])\s+|\n+", raw_note):
                    s_low = sent.lower()
                    if not any(cue in s_low for cue in hold_cues):
                        continue
                    directive_clause = re.split(
                        r"\b(?:due to|secondary to|because of|despite)\b",
                        s_low,
                        maxsplit=1,
                    )[0]
                    directive_toks = set(re.findall(r"[a-z][a-z0-9-]{3,}", directive_clause))
                    if probes & directive_toks:
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
                        matched_m = True
                        break
                if matched_m:
                    break

        active_med_probes = {p for m in scoped_meds for p in _med_name_probes(m)}
        stopped_candidates = [
            m
            for m in meds
            if m.metadata.get("section") == "med_historical"
            and not is_non_medical_admin_code(m.code)
            and str(m.metadata.get("setting") or "").lower() != "inpatient/administered"
            and not (m.label or "").strip().lower().startswith(("inpatient ", "naclflush"))
        ]
        for m in _dedup_nodes_by_identity(stopped_candidates):
            probes = _med_name_probes(m)
            if not probes or (probes & active_med_probes):
                continue
            m_date = (m.date_str or "")[:10]
            for n in notes_sorted:
                n_date = (n.date_str or "")[:10]
                if m_date and n_date and n_date < m_date:
                    continue
                raw_note = n.raw_line or ""
                n_low = (raw_note or n.value_summary or "").lower()
                if not n_low or not any(cue in n_low for cue in resume_cues):
                    continue
                n_toks = set(re.findall(r"[a-z][a-z0-9-]{3,}", n_low))
                if not (probes & n_toks):
                    continue
                matched_note = False
                for sent in re.split(r"(?<=[.!?;])\s+|\n+", raw_note):
                    s_low = sent.lower()
                    s_toks = set(re.findall(r"[a-z][a-z0-9-]{3,}", s_low))
                    if (probes & s_toks) and any(cue in s_low for cue in resume_cues):
                        snippet = sent.strip()
                        if len(snippet) > 160:
                            snippet = snippet[:157].rstrip() + "..."
                        self.add_edge(
                            ClinicalEdge(
                                source_id=m.node_id,
                                target_id=n.node_id,
                                edge_type="CONTRADICTED_OR_MODIFIED_BY_NOTE",
                                rationale=(
                                    f"Stopped/historical order but newer note ({n.date_str or 'undated'}) "
                                    f"instructs continue/resume: {snippet}"
                                ),
                                kb_source="Longitudinal Note-to-Medication Reconciliation",
                                source_citation=m.citation_idx,
                                target_citation=n.citation_idx,
                            )
                        )
                        matched_note = True
                        break
                if matched_note:
                    break

    def format_subgraph_lines(self, question: str = "", *, max_lines: Optional[int] = None) -> List[str]:
        q_terms = _query_terms(question)

        def _edge_overlap(edge: ClinicalEdge) -> int:
            if not q_terms:
                return 0
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            hay = " ".join(
                [
                    src.label if src else "",
                    tgt.label if tgt else "",
                    edge.rationale,
                ]
            ).lower()
            return sum(1 for t in q_terms if t in hay)

        def _score_edge(edge: ClinicalEdge) -> Tuple[int, int, str]:
            overlap = _edge_overlap(edge)
            prio = _EDGE_PRIORITY.get(edge.edge_type, 9)
            return (-overlap, prio, edge.edge_type)

        ranked = sorted(
            [e for e in self.edges if e.edge_type != "REPORTED_BY_ORG"],
            key=_score_edge,
        )
        high_prio_targets: Set[str] = set()
        for edge in ranked:
            if edge.edge_type not in _HIGH_PRIORITY_EDGE_TYPES:
                continue
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src is not None and tgt is not None and tgt.node_id != src.node_id:
                tgt_disp = tgt.short_display()
                if tgt_disp:
                    high_prio_targets.add(tgt_disp)

        seen_raw: Set[Tuple[str, str, str, str]] = set()
        stage_a: Dict[Tuple[str, str, str], List[str]] = {}
        for edge in ranked:
            is_high_prio = edge.edge_type in _HIGH_PRIORITY_EDGE_TYPES
            if not is_high_prio:
                if not q_terms or _edge_overlap(edge) == 0:
                    continue
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src is None or tgt is None:
                continue
            src_disp = src.short_display()
            tgt_disp = "" if tgt.node_id == src.node_id else tgt.short_display()
            if (
                not is_high_prio
                and edge.edge_type == "MONITORED_BY"
                and tgt_disp
                and tgt_disp in high_prio_targets
            ):
                continue
            raw_key = (edge.edge_type, src_disp, tgt_disp, edge.rationale)
            if raw_key in seen_raw:
                continue
            seen_raw.add(raw_key)
            key_a = (edge.edge_type, src_disp, edge.rationale)
            targets_list = stage_a.setdefault(key_a, [])
            if tgt_disp and tgt_disp not in targets_list and len(targets_list) < 4:
                targets_list.append(tgt_disp)

        stage_b: Dict[Tuple[str, str, str], List[str]] = {}
        for (edge_type, src_disp, rationale), targets_list in stage_a.items():
            if edge_type not in _HIGH_PRIORITY_EDGE_TYPES and not targets_list:
                continue
            joined_targets = "; ".join(targets_list)
            key_b = (edge_type, joined_targets, rationale)
            sources_list = stage_b.setdefault(key_b, [])
            if src_disp not in sources_list:
                sources_list.append(src_disp)

        high_prio_lines = sum(
            1 for (et, _, _) in stage_b if et in _HIGH_PRIORITY_EDGE_TYPES
        )
        max_bg = 0 if high_prio_lines >= 8 else (1 if high_prio_lines >= 6 else 3)
        lines: List[str] = []
        bg_emitted = 0
        for (edge_type, joined_targets, rationale), sources_list in stage_b.items():
            if edge_type not in _HIGH_PRIORITY_EDGE_TYPES:
                if bg_emitted >= max_bg:
                    continue
                bg_emitted += 1
            joined_sources = "; ".join(sources_list)
            if not joined_targets:
                line = f"- {edge_type}: {joined_sources} — {rationale}"
            else:
                line = f"- {edge_type}: {joined_sources} -> {joined_targets} ({rationale})"
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
        return lines


_LONGITUDINAL_QUERY_RE = re.compile(
    r"\b(?:trajector\w*|trend\w*|histor\w*|chang\w*|over time|prior|previous|baseline|20\d{2})\b",
    flags=re.IGNORECASE,
)


def _recover_omitted_tier0_items(
    *,
    extracted: Optional[ExtractedChart],
    represented_uuids: Set[str],
    question: str,
    sharing_posture: str,
    max_cit_before: int,
    max_recover: int = 5,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Recover highest-relevance omitted items using typed ``ExtractedChart.items``."""
    if extracted is None or max_recover <= 0:
        return []
    is_external = sharing_posture == "external_share"
    q_terms = _query_terms(question)
    if not q_terms:
        return []

    is_longitudinal_q = bool(_LONGITUDINAL_QUERY_RE.search(question or ""))
    series_latest: Dict[str, Tuple[str, str]] = {}
    uuid_to_series: Dict[str, str] = {}
    for pt in extracted.numeric_points:
        if not pt.resource_uuid:
            continue
        s_key = f"{pt.code or ''}:{pt.concept}".strip(":")
        uuid_to_series[pt.resource_uuid] = s_key
        pt_dt = pt.date or ""
        prev = series_latest.get(s_key)
        if prev is None or pt_dt >= prev[0]:
            series_latest[s_key] = (pt_dt, pt.resource_uuid)
    series_with_latest_in_chart: Set[str] = {
        s_key
        for s_key, (_, latest_uuid) in series_latest.items()
        if latest_uuid in represented_uuids
    }

    candidates: List[Tuple[Tuple[int, int, str], ClinicalItem]] = []
    for it in extracted.items:
        if it.section in {"patient", "manifest"}:
            continue
        if is_external and _is_sensitive_item(it):
            continue
        if (
            not is_longitudinal_q
            and it.section in {"condition_resolved", "med_historical"}
        ):
            continue
        if (
            not is_longitudinal_q
            and it.section in {"lab_observation", "vital_sign"}
            and it.resource_uuid
            and uuid_to_series.get(it.resource_uuid) in series_with_latest_in_chart
        ):
            continue
        uuids_here = {u for u in (it.resource_uuid, *it.member_uuids) if u}
        if not uuids_here or uuids_here.issubset(represented_uuids):
            continue
        hay = f"{it.display} {it.text} {it.source_org or ''} {it.date or ''}".lower()
        hay_tokens = set(re.findall(r"[a-z0-9]{2,}", hay))
        overlap = sum(
            1
            for t in q_terms
            if any(ht == t or (len(t) >= 4 and ht.startswith(t)) for ht in hay_tokens)
        )
        if q_terms and overlap == 0:
            continue
        sec_prio = _SECTION_OVERFLOW_PRIORITY.get(it.section, 15)
        is_abn = 0 if (isinstance(it.metadata, Mapping) and it.metadata.get("is_abnormal")) else 1
        dt10 = (it.date or "")[:10]
        neg_dt = -int(dt10.replace("-", "")) if re.match(r"^\d{4}-\d{2}-\d{2}$", dt10) else 0
        candidates.append(((-overlap, sec_prio, is_abn, neg_dt), it))  # type: ignore[arg-type]

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
        snippet = (it.text or it.display or it.resource_type).strip()
        rt_prefix = f"{it.resource_type}:"
        if snippet.lower().startswith(rt_prefix.lower()):
            snippet = snippet[len(rt_prefix):].strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
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
    max_overflow_recover: int = 5,
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
            max_recover=max_overflow_recover,
        )
        if overflow_triggered and max_overflow_recover > 0
        else []
    )

    chart_codes = (
        extract_chart_kb_codes(extracted, sharing_posture=sharing_posture)
        if extracted is not None
        else []
    )
    chart_code_set = {str(c) for c in chart_codes if c}
    clean_q = extract_clean_user_query(active_question) or active_question
    kb_synonyms = get_clinical_graph_rules().get("kb_query_synonyms") or {}
    kb_terms_list: List[str] = list(_query_terms(clean_q))
    for raw_tok in re.findall(r"[a-z0-9]+", (clean_q or "").lower()):
        if raw_tok in kb_synonyms:
            kb_terms_list.extend(str(sym).lower() for sym in (kb_synonyms[raw_tok] or ()))
    kb_terms = list(dict.fromkeys(kb_terms_list))
    raw_kb_hits = (
        search_kb(" ".join(kb_terms), k=4, corpus="wahs", codes=chart_codes)
        if kb_terms
        else []
    )
    kb_lines: List[str] = []
    for hit in raw_kb_hits:
        is_privacy_kb = "privacy" in str(hit.get("id") or "").lower() or "sdoh" in {
            str(t).lower() for t in (hit.get("tags") or [])
        }
        if is_privacy_kb and (sharing_posture == "external_share" or "privacy" not in kb_terms):
            continue
        hit_hay = f"{hit.get('title', '')} {hit.get('text', '')} {' '.join(hit.get('tags') or [])}".lower()
        if not any(t in hit_hay or (len(t) >= 5 and t[:5] in hit_hay) for t in kb_terms):
            continue
        hit_codes = {str(c) for c in (hit.get("codes") or []) if c}
        if (
            hit_codes
            and chart_code_set
            and not is_privacy_kb
            and not (hit_codes & chart_code_set)
        ):
            continue
        title = str(hit.get("title") or "Clinical Guideline").strip()
        summary = str(hit.get("text") or hit.get("summary") or "").strip()
        source = str(hit.get("source") or "").strip()
        if not summary:
            continue
        src_short = source.split(" / ")[0].split(" — ")[0].strip()
        src_note = f" ({src_short})" if src_short and len(src_short) <= 48 else ""
        kb_lines.append(f"- {title}{src_note}: {summary}")

    subgraph_lines = graph.format_subgraph_lines(active_question)
    kb_lines = kb_lines[: (1 if len(subgraph_lines) >= 8 else 2)]

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
        kb_passages_matched=len(cur_kb),
        max_citation_idx_before=max_cit_before,
        max_citation_idx_after=max_cit_after,
        active_question=active_question,
        patient_name=graph.patient_name,
        omitted_by_category=omitted_by_cat,
        edge_breakdown=graph.edge_breakdown(),
        recovered_mappings=rec_mappings,
    )
