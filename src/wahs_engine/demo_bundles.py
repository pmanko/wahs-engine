"""Opt-in read-only FHIR R4 demo bundle registry for sandbox deployments.

Only enabled when both ``WAHS_ALLOW_LOCAL_BUNDLE_MODE=true`` and
``WAHS_DEMO_BUNDLES_DIR`` points to a directory of ``*.json`` FHIR R4 Bundles.
Production deployments leave ``WAHS_DEMO_BUNDLES_DIR`` unset so demo routes
are disabled and return HTTP 404.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import HTTPException

_SAFE_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_INFRASTRUCTURE_TYPES = frozenset({"Patient", "Organization", "Practitioner", "PractitionerRole", "Location", "Device", "Provenance"})
_PRIORITY_ORDER = (
    "margaret-chen-polypharmacy",
    "elena-rostova-cardiorenal",
    "arthur-pendelton-anticoag",
    "sofia-alvarez-maternal-endocrine",
    "marcus-vance-oncology-immunotherapy",
    "mimic-10014354-icu-heme-cardiorenal",
    "mimic-10019003-icu-hepatopulmonary",
    "byron202-emard19",
    "lamar193-beer512",
    "maurice742-auer97",
)
_PRIORITY_RANK = {slug: idx for idx, slug in enumerate(_PRIORITY_ORDER)}


def _slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "bundle"


def _extract_patient_name(patient: Mapping[str, Any]) -> str:
    names = patient.get("name")
    if isinstance(names, list):
        for entry in names:
            if not isinstance(entry, Mapping):
                continue
            text = str(entry.get("text") or "").strip()
            if text:
                return text
            given = " ".join(str(g).strip() for g in (entry.get("given") or []) if str(g).strip())
            family = str(entry.get("family") or "").strip()
            combined = f"{given} {family}".strip()
            if combined:
                return combined
    return str(patient.get("id") or "Unknown Patient")


def _first_code_text(obj: Any) -> str:
    if not isinstance(obj, Mapping):
        return ""
    if obj.get("text"):
        return str(obj["text"]).strip()
    codings = obj.get("coding")
    if isinstance(codings, list):
        for coding in codings:
            if isinstance(coding, Mapping):
                disp = str(coding.get("display") or coding.get("code") or "").strip()
                if disp:
                    return disp
    return ""


def _extract_date(res: Mapping[str, Any]) -> str:
    for key in (
        "effectiveDateTime",
        "authoredOn",
        "onsetDateTime",
        "recordedDate",
        "occurrenceDateTime",
        "date",
        "issued",
        "performedDateTime",
        "created",
    ):
        val = res.get(key)
        if isinstance(val, str) and len(val) >= 10:
            return val[:10]
    for period_key in ("effectivePeriod", "performedPeriod", "period", "onsetPeriod"):
        period = res.get(period_key)
        if isinstance(period, Mapping):
            start = period.get("start")
            if isinstance(start, str) and len(start) >= 10:
                return start[:10]
    return ""


def _extract_source(res: Mapping[str, Any]) -> str:
    meta = res.get("meta")
    if isinstance(meta, Mapping):
        for tag in meta.get("tag") or []:
            if isinstance(tag, Mapping) and str(tag.get("system") or "").endswith("/wahealthsummary-source-name"):
                code = str(tag.get("code") or tag.get("display") or "").strip()
                if code:
                    return code
        src = str(meta.get("source") or "").strip()
        if src:
            return src
    sp = res.get("serviceProvider")
    if isinstance(sp, Mapping) and sp.get("display"):
        return str(sp["display"]).strip()
    return "Local FHIR Bundle"


def _extract_status(res: Mapping[str, Any]) -> str:
    cs = _first_code_text(res.get("clinicalStatus"))
    if cs:
        return cs
    for key in ("status", "verificationStatus"):
        val = res.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, Mapping):
            txt = _first_code_text(val)
            if txt:
                return txt
    return ""


def _extract_display_and_detail(res: Mapping[str, Any]) -> Tuple[str, str]:
    rt = str(res.get("resourceType") or "")
    display = (
        _first_code_text(res.get("code"))
        or _first_code_text(res.get("medicationCodeableConcept"))
        or _first_code_text(res.get("vaccineCode"))
        or _first_code_text(res.get("type"))
        or str(res.get("description") or "").strip()
        or str(res.get("title") or "").strip()
    )
    if not display and isinstance(res.get("type"), list) and res["type"]:
        display = _first_code_text(res["type"][0])
    if not display and rt == "Patient":
        display = _extract_patient_name(res)
    if not display:
        display = f"{rt} {res.get('id') or ''}".strip()

    detail_parts: List[str] = []
    if rt == "Observation":
        vq = res.get("valueQuantity")
        if isinstance(vq, Mapping) and vq.get("value") is not None:
            unit = str(vq.get("unit") or vq.get("code") or "").strip()
            detail_parts.append(f"{vq['value']} {unit}".strip())
        elif res.get("valueString"):
            detail_parts.append(str(res["valueString"]).strip())
        elif res.get("valueCodeableConcept"):
            vcc = _first_code_text(res["valueCodeableConcept"])
            if vcc:
                detail_parts.append(vcc)
        elif isinstance(res.get("component"), list) and res["component"]:
            comp_vals = []
            for comp in res["component"][:3]:
                if not isinstance(comp, Mapping):
                    continue
                cname = _first_code_text(comp.get("code"))
                cvq = comp.get("valueQuantity")
                if isinstance(cvq, Mapping) and cvq.get("value") is not None:
                    cunit = str(cvq.get("unit") or cvq.get("code") or "").strip()
                    comp_vals.append(f"{cname}: {cvq['value']} {cunit}".strip())
            if comp_vals:
                detail_parts.append("; ".join(comp_vals))
        interps = res.get("interpretation")
        if isinstance(interps, list) and interps:
            interp_txt = _first_code_text(interps[0])
            if interp_txt:
                detail_parts.append(f"[{interp_txt}]")
    elif rt in {"MedicationRequest", "MedicationStatement"}:
        dosage = res.get("dosageInstruction") or res.get("dosage")
        if isinstance(dosage, list) and dosage and isinstance(dosage[0], Mapping):
            dtxt = str(dosage[0].get("text") or "").strip()
            if dtxt:
                detail_parts.append(dtxt)
    elif rt == "AllergyIntolerance":
        crit = str(res.get("criticality") or "").strip()
        if crit:
            detail_parts.append(f"criticality: {crit}")
        reactions = res.get("reaction")
        if isinstance(reactions, list) and reactions and isinstance(reactions[0], Mapping):
            manifs = [_first_code_text(m) for m in (reactions[0].get("manifestation") or []) if _first_code_text(m)]
            if manifs:
                detail_parts.append(", ".join(manifs[:3]))
    elif rt == "DiagnosticReport":
        conc = str(res.get("conclusion") or "").strip()
        if conc:
            detail_parts.append(conc[:180])
    elif rt == "DocumentReference":
        desc = str(res.get("description") or "").strip()
        if desc and desc != display:
            detail_parts.append(desc[:180])

    return display[:180], " ".join(detail_parts)[:220]


@dataclass(frozen=True)
class LoadedDemoBundle:
    bundle_id: str
    filename: str
    path: Path
    size_bytes: int
    sha256: str
    bundle: Dict[str, Any]
    summary: Dict[str, Any]
    items: Tuple[Dict[str, Any], ...]
    raw_by_key: Dict[str, Dict[str, Any]]


class DemoBundleRegistry:
    """Thread-safe, lazily cached reader for mounted sandbox FHIR R4 bundles."""

    def __init__(self, directory: str) -> None:
        self.root = Path(directory).expanduser().resolve()
        self._index: Dict[str, Path] = {}
        self._stem_aliases: Dict[str, str] = {}
        self._ordered_ids: List[str] = []
        self._loaded: Dict[str, LoadedDemoBundle] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.root.is_dir():
            return
        candidates: List[Tuple[int, str, Path]] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            slug = _slugify(path.stem)
            rank = _PRIORITY_RANK.get(slug, 1000)
            candidates.append((rank, slug, path))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _rank, slug, path in candidates:
            if slug in self._index:
                continue
            self._index[slug] = path
            self._stem_aliases[path.stem.lower()] = slug
            self._stem_aliases[path.name.lower()] = slug
            self._ordered_ids.append(slug)

    def resolve_id(self, raw_id: str) -> str:
        cleaned = (raw_id or "").strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned or not _SAFE_BUNDLE_ID_RE.match(cleaned):
            raise HTTPException(status_code=400, detail="Invalid bundle_id format.")
        low = cleaned.lower()
        if low in self._index:
            return low
        if low in self._stem_aliases:
            return self._stem_aliases[low]
        slug = _slugify(cleaned)
        if slug in self._index:
            return slug
        raise HTTPException(status_code=404, detail=f"Demo bundle '{cleaned}' not found.")

    def get(self, raw_id: str) -> LoadedDemoBundle:
        bundle_id = self.resolve_id(raw_id)
        cached = self._loaded.get(bundle_id)
        if cached is not None:
            return cached

        path = self._index[bundle_id]
        raw_bytes = path.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(data, dict) or data.get("resourceType") != "Bundle":
            raise HTTPException(status_code=500, detail=f"Demo bundle '{bundle_id}' is not a valid FHIR R4 Bundle.")

        entries = data.get("entry") or []
        patient_res: Mapping[str, Any] = {}
        counts_by_type: Dict[str, int] = {}
        sources_set: List[str] = []
        seen_sources = set()
        items: List[Dict[str, Any]] = []
        raw_by_key: Dict[str, Dict[str, Any]] = {}

        for idx, entry in enumerate(entries):
            res = entry.get("resource") if isinstance(entry, Mapping) else None
            if not isinstance(res, dict):
                continue
            rt = str(res.get("resourceType") or "Unknown")
            rid = str(res.get("id") or f"entry-{idx}")
            counts_by_type[rt] = counts_by_type.get(rt, 0) + 1
            if rt == "Patient" and not patient_res:
                patient_res = res

            src = _extract_source(res)
            if src not in seen_sources and rt not in _INFRASTRUCTURE_TYPES:
                seen_sources.add(src)
                sources_set.append(src)

            full_key = f"{rt}/{rid}"
            raw_by_key[full_key] = res
            if rid not in raw_by_key:
                raw_by_key[rid] = res

            display, detail = _extract_display_and_detail(res)
            items.append(
                {
                    "key": full_key,
                    "resourceType": rt,
                    "id": rid,
                    "date": _extract_date(res),
                    "display": display,
                    "detail": detail,
                    "status": _extract_status(res),
                    "source": src,
                }
            )

        items.sort(key=lambda it: (it["date"] or "0000-00-00", it["resourceType"], it["display"]), reverse=True)
        patient_name = _extract_patient_name(patient_res) if patient_res else path.stem.replace("_", " ")
        clinical_count = sum(v for k, v in counts_by_type.items() if k not in _INFRASTRUCTURE_TYPES)
        summary = {
            "id": bundle_id,
            "filename": path.name,
            "label": path.stem.replace("_", " "),
            "patient_name": patient_name,
            "birth_date": str(patient_res.get("birthDate") or ""),
            "gender": str(patient_res.get("gender") or ""),
            "size_bytes": len(raw_bytes),
            "size_kb": round(len(raw_bytes) / 1024.0, 1),
            "total_resources": len(items),
            "clinical_resources": clinical_count,
            "counts_by_type": dict(sorted(counts_by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
            "sources": sources_set or ["Local FHIR Bundle"],
        }
        loaded = LoadedDemoBundle(
            bundle_id=bundle_id,
            filename=path.name,
            path=path,
            size_bytes=len(raw_bytes),
            sha256=sha,
            bundle=data,
            summary=summary,
            items=tuple(items),
            raw_by_key=raw_by_key,
        )
        self._loaded[bundle_id] = loaded
        return loaded

    def list_summaries(self) -> List[Dict[str, Any]]:
        return [self.get(bid).summary for bid in self._ordered_ids]

    def query_resources(
        self,
        raw_id: str,
        *,
        resource_type: Optional[str] = None,
        q: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        resource_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        loaded = self.get(raw_id)
        if resource_id:
            target = loaded.raw_by_key.get(resource_id.strip())
            if target is None:
                raise HTTPException(status_code=404, detail=f"Resource '{resource_id}' not found in demo bundle '{loaded.bundle_id}'.")
            return {
                "bundle_id": loaded.bundle_id,
                "patient_name": loaded.summary["patient_name"],
                "resource": target,
            }

        rt_filter = (resource_type or "").strip()
        query = (q or "").strip().lower()
        filtered = []
        for item in loaded.items:
            if rt_filter and rt_filter.lower() != "all" and item["resourceType"] != rt_filter:
                continue
            if query:
                hay = f"{item['resourceType']} {item['display']} {item['detail']} {item['date']} {item['source']} {item['status']}".lower()
                if query not in hay:
                    continue
            filtered.append(item)

        safe_offset = max(0, int(offset))
        safe_limit = min(500, max(1, int(limit)))
        return {
            "bundle_id": loaded.bundle_id,
            "patient_name": loaded.summary["patient_name"],
            "total_matching": len(filtered),
            "offset": safe_offset,
            "limit": safe_limit,
            "counts_by_type": loaded.summary["counts_by_type"],
            "items": filtered[safe_offset : safe_offset + safe_limit],
        }
