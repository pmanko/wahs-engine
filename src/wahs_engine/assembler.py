"""Zero-privilege `fhir-auth` star-graph `WAHSBundleAssembler`, `chart_scope`, and scoped caches."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar
from urllib.parse import urlsplit
import uuid

from fastapi import HTTPException
import httpx

from wahs_core import (
    CondensedChartResult,
    ExtractedChart,
    build_wahs_hub_enrichment_block,
    condense_bundle_to_budget,
    extract_bundle,
)

from .config import ModelSpec, WahsEngineSettings
from .jwt_claims import VerifiedCallerIdentity
from .schemas import ChartScope, WahsRequestOptions
from .tokenizers_registry import get_exact_token_counter

# Official CIRG WAHS CodeSystem URIs (matching `uwcirg/shl-ips` & `uwcirg/jwt-proxy`)
WAHS_SYSTEM_BASE = "http://fhir.wahealthsummary.cirg.uw.edu"
IDENTIFIER_SYSTEM = "http://keycloak.cirg.uw.edu"
SECURITY_SYSTEM = "http://keycloak.cirg.uw.edu/fhir/security-labels"
CATEGORY_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-category"
METHOD_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-method"
PLACEHOLDER_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-placeholder"
SOURCE_NAME_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-source-name"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DATASET_REF_RE = re.compile(r"^Patient/([A-Za-z0-9._-]+)$")

_INFRASTRUCTURE_RESOURCE_TYPES: Set[str] = {
    "Organization",
    "Practitioner",
    "PractitionerRole",
    "Location",
    "Provenance",
    "Medication",
}

_DATE_FIELDS_BY_RESOURCE: Tuple[str, ...] = (
    "effectiveDateTime",
    "authoredOn",
    "occurrenceDateTime",
    "recordedDate",
    "onsetDateTime",
    "issued",
    "date",
    "started",
    "startDate",
    "statusDate",
    "created",
)

WAHS_SYSTEM_PROMPT = """You are the Washington Health Summary (WAHS) AI Assistant, a plain-language clinical record companion.

CRITICAL SAFETY & PROVENANCE RULES:
1. GROUNDED ONLY IN PROVIDED EVIDENCE: Answer strictly from the Condensed Clinical Record (`[1]..[N]`) and any Patient-Authored Drafts/Flags (`[P1]..[PM]`). Never fabricate lab values, medications, dates, or diagnoses. If a fact is not present in the scoped record, state clearly that it is not recorded in the current WA Health Summary view (and check the `[Completeness Manifest]` before claiming a historical negative).
2. INLINE CITATIONS REQUIRED: Cite clinical record items inline as `[1]`, `[2]`, etc., and draft overlay items as `[P1]`, `[P2]`, etc., so the UI can render interactive source pills.
3. DISTINGUISH CLINIC RECORDS FROM PATIENT-AUTHORED ITEMS: Highlight which health system or patient-authored WAHS section (`Patient Story`, `Care Needs`, `Goals`) reported each item when reconciling discrepancies or answering questions.
4. CLEAR, EMPATHETIC COMMUNICATION: Use plain, accessible language and suggest 1-2 concrete questions for the patient's care team when clinically relevant."""

K = TypeVar("K")
V = TypeVar("V")


class _BoundedLRUCache(Generic[K, V]):
    def __init__(self, max_entries: int, ttl_seconds: float = 1800.0) -> None:
        self._max = max(1, max_entries)
        self._ttl = max(0.0, float(ttl_seconds))
        self._data: OrderedDict[K, Tuple[float, V]] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        expires_at, value = self._data[key]
        if self._ttl > 0.0 and time.monotonic() >= expires_at:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        expires_at = (time.monotonic() + self._ttl) if self._ttl > 0.0 else float("inf")
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def _parse_fhir_instant(ts: str) -> Tuple[int, float, str]:
    """Parse an ISO-8601 / FHIR instant into a comparable UTC epoch tuple."""
    raw = str(ts or "").strip()
    if not raw:
        return (0, 0.0, "")
    norm = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return (1, dt.timestamp(), raw)
    except ValueError:
        return (0, 0.0, raw)


def derive_effective_chart_budget(
    spec: ModelSpec,
    options: WahsRequestOptions,
    *,
    system_prompt_header: str = WAHS_SYSTEM_PROMPT,
) -> Tuple[int, Optional[str]]:
    """Derive exact model-window-aware chart token budget (`at-most` semantics for request overrides)."""
    counter = get_exact_token_counter(spec.tokenizer_family)
    base_framing = counter(system_prompt_header) + 96
    if (options.enable_kb and options.retrieval_strategy != "chart_only") or options.retrieval_strategy == "note_rag":
        base_framing += 800
    if options.enable_overflow_tools or options.retrieval_strategy == "dynamic_fhir_tools":
        base_framing += 420

    output_reserve = max(spec.default_max_output_tokens, min(4096, int(spec.context_window * 0.125)))
    conversation_reserve = max(256, min(16384, int(spec.context_window * 0.15)))
    window_derived = spec.context_window - base_framing - output_reserve - conversation_reserve
    if window_derived < 512:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Configured model context_window={spec.context_window} cannot fit required framing and "
                f"reserves ({base_framing + output_reserve + conversation_reserve} tokens)."
            ),
        )

    ceiling_cap = window_derived
    clamp_reason = "context_window_derived"
    if spec.chart_budget_ceiling is not None and spec.chart_budget_ceiling < ceiling_cap:
        ceiling_cap = spec.chart_budget_ceiling
        clamp_reason = "chart_budget_ceiling"

    budget = ceiling_cap
    if options.retrieval_strategy == "dynamic_fhir_tools" and budget > 2048:
        budget = 2048
        clamp_reason = "dynamic_fhir_tools_header"

    if options.target_budget_tokens is not None:
        req_b = int(options.target_budget_tokens)
        if req_b <= budget:
            budget = req_b
            clamp_reason = "caller_target_budget_tokens"
        else:
            if options.retrieval_strategy == "dynamic_fhir_tools" and budget == 2048:
                clamp_reason = "dynamic_fhir_tools_header"
            elif spec.chart_budget_ceiling is not None and ceiling_cap == spec.chart_budget_ceiling and ceiling_cap < window_derived:
                clamp_reason = "clamped_to_chart_budget_ceiling"
            else:
                clamp_reason = "clamped_to_context_window"

    return budget, clamp_reason


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str
    last_updated: str
    category: str
    method: str
    source: str
    source_name: str
    is_placeholder_patient: bool
    has_version_stamp: bool = True


@dataclass
class AssembledWahsContext:
    verified_sub: str
    patient_id: str
    patient_name: str
    birth_date: str
    system_prompt: str
    prefix_cache_key_hash: str
    datasets_fingerprint: str
    scope_hash: str
    datasets_included: List[Dict[str, str]]
    scope_applied: Dict[str, Any]
    security_audit: Dict[str, Any]
    extracted: ExtractedChart
    condensed: CondensedChartResult
    citation_map: Dict[str, Dict[str, Any]]
    privacy_notices: List[str]
    cache_hit: bool = False
    dataset_cache_hits: int = 0
    dataset_fetches: int = 0
    budget_requested: Optional[int] = None
    budget_effective: int = 0
    context_window: int = 32768
    chart_budget_ceiling: Optional[int] = 16384
    clamp_reason: Optional[str] = None
    tokenizer_family: str = "gemma4"
    partial_fetch: bool = False
    partial_datasets: List[Dict[str, Any]] = field(default_factory=list)


def _canonical_sha256(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_patient_tags(patient_res: Mapping[str, Any]) -> Tuple[str, str, str, str, bool]:
    meta = patient_res.get("meta") if isinstance(patient_res.get("meta"), Mapping) else {}
    tags = meta.get("tag") if isinstance(meta.get("tag"), list) else []
    category = ""
    method = ""
    source_name = ""
    placeholder = False
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        sys_uri = str(tag.get("system") or "")
        code = str(tag.get("code") or "").strip()
        if sys_uri == CATEGORY_SYSTEM and not category:
            category = code
        elif sys_uri == METHOD_SYSTEM and not method:
            method = code
        elif sys_uri == SOURCE_NAME_SYSTEM and not source_name:
            source_name = code
        elif sys_uri == PLACEHOLDER_SYSTEM and code:
            placeholder = True

    raw_source = str(meta.get("source") or "").split("#", 1)[0].strip()
    effective_source_name = source_name or raw_source or category or "WAHS Dataset"
    return (
        category or "provider-health-record",
        method or "unknown",
        raw_source or effective_source_name,
        effective_source_name,
        placeholder,
    )


def _stamp_resource_source_tag(res: Dict[str, Any], desc: DatasetDescriptor) -> Dict[str, Any]:
    stamped = copy.deepcopy(res)
    meta = stamped.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        stamped["meta"] = meta
    tags = meta.setdefault("tag", [])
    if not isinstance(tags, list):
        tags = []
        meta["tag"] = tags
    has_source_tag = any(
        isinstance(t, Mapping) and t.get("system") == SOURCE_NAME_SYSTEM for t in tags
    )
    if not has_source_tag and desc.source_name:
        tags.append({"system": SOURCE_NAME_SYSTEM, "code": desc.source_name})
    has_cat_tag = any(
        isinstance(t, Mapping) and t.get("system") == CATEGORY_SYSTEM for t in tags
    )
    if not has_cat_tag and desc.category:
        tags.append({"system": CATEGORY_SYSTEM, "code": desc.category})
    return stamped


def _dataset_matches_scope(desc: DatasetDescriptor, scope: ChartScope) -> bool:
    norm_candidates = {
        desc.source_name.lower(),
        desc.source.lower(),
        desc.dataset_id.lower(),
    }
    if scope.include_sources:
        inc = {s.strip().lower() for s in scope.include_sources if s.strip()}
        if inc and not any(
            c in inc or any(i in c for i in inc) for c in norm_candidates if c
        ):
            return False
    if scope.exclude_sources:
        exc = {s.strip().lower() for s in scope.exclude_sources if s.strip()}
        if exc and any(
            c in exc or any(e in c for e in exc) for c in norm_candidates if c
        ):
            return False

    cat_lower = desc.category.lower()
    if scope.include_categories:
        inc_c = {c.strip().lower() for c in scope.include_categories if c.strip()}
        if inc_c and cat_lower not in inc_c:
            return False
    if scope.exclude_categories:
        exc_c = {c.strip().lower() for c in scope.exclude_categories if c.strip()}
        if exc_c and cat_lower in exc_c:
            return False
    return True


def _extract_resource_iso_date(res: Mapping[str, Any]) -> Optional[str]:
    for field_name in _DATE_FIELDS_BY_RESOURCE:
        val = res.get(field_name)
        if isinstance(val, str) and len(val) >= 10 and val[4] == "-" and val[7] == "-":
            return val[:10]
    for period_field in ("effectivePeriod", "period"):
        per = res.get(period_field)
        if isinstance(per, Mapping):
            for bound in ("end", "start"):
                val = per.get(bound)
                if isinstance(val, str) and len(val) >= 10 and val[4] == "-" and val[7] == "-":
                    return val[:10]
    return None


def _extract_subject_ref(resource: Mapping[str, Any]) -> Optional[str]:
    for field_name in ("subject", "patient"):
        val = resource.get(field_name)
        if isinstance(val, Mapping):
            ref = str(val.get("reference") or "").strip()
            if ref:
                return ref
    return None


def enforce_single_patient_scope(
    bundle: Dict[str, Any],
    *,
    authorized_patient_id: str,
    allowed_patient_refs: Set[str],
    strict_reject: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strip or reject any clinical resource referencing an unauthorized Patient subject."""
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        raise HTTPException(status_code=400, detail="Payload must be a FHIR R4 Bundle.")

    allowed: Set[str] = set()
    for raw_ref in {authorized_patient_id, *allowed_patient_refs}:
        clean = str(raw_ref or "").strip()
        if clean:
            allowed.add(clean)
            if clean.startswith("Patient/"):
                allowed.add(clean.split("/", 1)[1])

    entries = bundle.get("entry") or []
    for entry in entries:
        res = entry.get("resource") if isinstance(entry, Mapping) else None
        if isinstance(res, Mapping) and res.get("resourceType") == "Patient":
            pid = str(res.get("id") or "").strip()
            if pid and not allowed:
                allowed.add(pid)
            break

    filtered_entries: List[Dict[str, Any]] = []
    stripped: List[Dict[str, str]] = []
    seen_master_patient = False

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        res = entry.get("resource")
        if not isinstance(res, Mapping):
            continue
        rtype = str(res.get("resourceType") or "")
        rid = str(res.get("id") or "")

        if rtype == "Patient":
            if rid and rid not in allowed:
                if strict_reject:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Foreign Patient resource '{rid}' rejected by single-patient guardrail.",
                    )
                stripped.append({"resourceType": rtype, "id": rid, "reason": "foreign_patient"})
                continue
            if not seen_master_patient:
                seen_master_patient = True
                filtered_entries.append({"resource": dict(res)})
            continue

        if rtype in _INFRASTRUCTURE_RESOURCE_TYPES:
            filtered_entries.append({"resource": dict(res)})
            continue

        subj_ref = _extract_subject_ref(res)
        if subj_ref:
            subj_id = subj_ref.split("/", 1)[1] if "/" in subj_ref else subj_ref
            if subj_id.startswith("urn:uuid:"):
                subj_id = subj_id[len("urn:uuid:") :]
            if subj_id not in allowed and subj_ref not in allowed:
                if strict_reject:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Resource {rtype}/{rid} references unauthorized subject '{subj_ref}'.",
                    )
                stripped.append(
                    {"resourceType": rtype, "id": rid, "subject": subj_ref, "reason": "unauthorized_subject"}
                )
                continue

        filtered_entries.append({"resource": dict(res)})

    out_bundle = dict(bundle)
    out_bundle["entry"] = filtered_entries
    audit = {
        "authorized_patient_id": authorized_patient_id,
        "allowed_patient_ids": sorted(allowed),
        "input_entries": len(entries),
        "scoped_entries": len(filtered_entries),
        "stripped_foreign_count": len(stripped),
        "stripped_foreign_resources": stripped,
    }
    return out_bundle, audit


def _rewrite_next_link_to_fhir_auth(fhir_auth_base_url: str, raw_next_url: str) -> str:
    """Safely rewrite HAPI pagination URLs onto `fhir_auth_base_url`.

    Prevents HAPI's `use_apache_address_strategy` links (e.g. `https://fhir-internal:8080/fhir?...`)
    from bypassing `fhir-auth` (`jwt-proxy`) or leaking internal hostnames, while keeping the
    `/fhir/` prefix required by `jwt-proxy`'s `10_allow_fhir.py` policy.
    """
    base = fhir_auth_base_url.rstrip("/")
    parsed = urlsplit(raw_next_url)
    path = parsed.path or ""
    idx = path.find("/fhir")
    rel = path[idx + len("/fhir") :] if idx >= 0 else path
    if not rel or rel == "/":
        # Ensure `/fhir/?_getpages=...` has trailing slash so jwt-proxy `path.startswith("/fhir/")` allows it
        target = f"{base}/"
    else:
        target = f"{base}/{rel.lstrip('/')}"
    return f"{target}?{parsed.query}" if parsed.query else target


def _format_overlay_items(
    overlay_resources: Sequence[Mapping[str, Any]],
    authorized_patient_id: str,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Dict[str, Any]]]:
    """Split `wahs.overlay_resources` into FHIR resource entries and `[P1]..[PM]` draft lines."""
    fhir_entries: List[Dict[str, Any]] = []
    draft_lines: List[str] = []
    draft_citations: Dict[str, Dict[str, Any]] = {}
    p_idx = 1

    for idx, item in enumerate(overlay_resources):
        if not isinstance(item, Mapping):
            continue
        if item.get("resourceType"):
            res_copy = copy.deepcopy(dict(item))
            res_copy.setdefault("id", f"overlay-{idx + 1}")
            if res_copy["resourceType"] != "Patient" and not _extract_subject_ref(res_copy):
                res_copy["subject"] = {"reference": f"Patient/{authorized_patient_id}"}
            meta = res_copy.setdefault("meta", {})
            tags = meta.setdefault("tag", [])
            tags.append({"system": SOURCE_NAME_SYSTEM, "code": "WAHS Unsaved Draft"})
            fhir_entries.append({"resource": res_copy})
        else:
            kind = str(item.get("type") or item.get("category") or "Draft Note").strip()
            label = str(item.get("item") or item.get("title") or "").strip()
            text = str(item.get("text") or item.get("note") or "").strip()
            body = f"{label}: {text}" if (label and text) else (label or text)
            if not body:
                continue
            pid = f"P{p_idx}"
            p_idx += 1
            draft_lines.append(f"[{pid}] ({kind}) {body} [source: WAHS Unsaved Draft]")
            draft_citations[pid] = {
                "index": pid,
                "resourceType": "PatientDraftOverlay",
                "id": str(item.get("id") or f"overlay-draft-{p_idx - 1}"),
                "source": "WAHS Unsaved Draft",
                "category": "patient-story",
                "date": str(item.get("date") or "draft"),
                "display": body,
            }

    block = ""
    if draft_lines:
        block = "=== UNSAVED PATIENT DRAFT NOTES & FLAGS ([P1]..[PM]) ===\n" + "\n".join(draft_lines)
    return fhir_entries, block, draft_citations


def _extract_ips_referenced_keys(ips_bundle: Mapping[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for entry in ips_bundle.get("entry") or []:
        res = entry.get("resource") if isinstance(entry, Mapping) else None
        if not isinstance(res, Mapping):
            continue
        rt = str(res.get("resourceType") or "")
        rid = str(res.get("id") or "")
        if rt and rid:
            keys.add(f"{rt}/{rid}")
            keys.add(rid)
        if rt == "Composition":
            for sec in res.get("section") or []:
                if isinstance(sec, Mapping):
                    for ref_obj in sec.get("entry") or []:
                        if isinstance(ref_obj, Mapping):
                            ref = str(ref_obj.get("reference") or "").strip()
                            if ref:
                                keys.add(ref)
                                if "/" in ref:
                                    keys.add(ref.split("/", 1)[1])
    return keys


_SENTINEL = object()


def _clone_context_for_request(
    ctx: AssembledWahsContext,
    *,
    cache_hit: bool,
    budget_requested: Any = _SENTINEL,
    clamp_reason: Any = _SENTINEL,
) -> AssembledWahsContext:
    """Return a per-request shallow clone with deep-copied mutable citation & notice state."""
    cloned = copy.copy(ctx)
    cloned.citation_map = copy.deepcopy(ctx.citation_map)
    cloned.privacy_notices = list(ctx.privacy_notices)
    cloned.partial_datasets = [dict(d) for d in ctx.partial_datasets]
    cloned.cache_hit = cache_hit
    if budget_requested is not _SENTINEL:
        cloned.budget_requested = budget_requested
    if clamp_reason is not _SENTINEL:
        cloned.clamp_reason = clamp_reason
    return cloned


class WAHSBundleAssembler:
    """Assembles, scopes, caches, and condenses multi-dataset WAHS patient charts via `fhir-auth`."""

    def __init__(
        self,
        settings: WahsEngineSettings,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings
        self._external_client = http_client
        self._dataset_cache: _BoundedLRUCache[Tuple[str, str, str], List[Dict[str, Any]]] = (
            _BoundedLRUCache(settings.dataset_cache_max_entries, ttl_seconds=settings.cache_ttl_seconds)
        )
        self._prefix_cache: _BoundedLRUCache[Tuple[str, str, str], AssembledWahsContext] = (
            _BoundedLRUCache(settings.prefix_cache_max_entries, ttl_seconds=settings.cache_ttl_seconds)
        )

    def _validate_overlay_budget(self, overlay_resources: Sequence[Mapping[str, Any]]) -> None:
        if not overlay_resources:
            return
        raw_bytes = len(
            json.dumps(list(overlay_resources), separators=(",", ":"), default=str).encode("utf-8")
        )
        if raw_bytes > self.settings.max_overlay_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"wahs.overlay_resources exceeds maximum size of {self.settings.max_overlay_bytes} bytes "
                    f"(got {raw_bytes} bytes). Persist larger updates to HAPI via fhir-auth first."
                ),
            )

    async def _fhir_get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        access_token: str,
        params: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json, application/json",
        }
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code in (401, 403):
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"fhir-auth denied request ({resp.status_code}).",
            )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"FHIR resource not found at {url}.")
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("resourceType") == "OperationOutcome":
            diag = ((payload.get("issue") or [{}])[0]).get("diagnostics") or "FHIR OperationOutcome error"
            raise HTTPException(status_code=502, detail=f"FHIR OperationOutcome: {diag}")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Unexpected non-JSON-object response from fhir-auth.")
        return payload

    async def _fetch_paginated_everything(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        dataset_id: str,
        access_token: str,
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        url = f"{base_url}/Patient/{dataset_id}/$everything"
        params: Optional[Dict[str, str]] = {"_count": str(self.settings.everything_page_size)}
        collected: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        max_res = max(1, int(self.settings.max_everything_resources))
        is_partial = False
        partial_reason: Optional[str] = None

        for _page in range(max(1, int(self.settings.max_everything_pages))):
            bundle = await self._fhir_get_json(client, url, access_token, params=params)
            entries = bundle.get("entry") or []
            reached_res_cap_mid_page = False
            for idx_e, entry in enumerate(entries):
                res = entry.get("resource") if isinstance(entry, Mapping) else None
                if not isinstance(res, dict):
                    continue
                rt = str(res.get("resourceType") or "")
                rid = str(res.get("id") or "")
                dedup_key = f"{rt}/{rid}" if rid else uuid.uuid4().hex
                if dedup_key not in seen:
                    if len(collected) >= max_res:
                        reached_res_cap_mid_page = True
                        is_partial = True
                        partial_reason = "max_everything_resources"
                        break
                    seen.add(dedup_key)
                    collected.append(res)

            next_url = ""
            for link in bundle.get("link") or []:
                if isinstance(link, Mapping) and link.get("relation") == "next":
                    next_url = str(link.get("url") or "").strip()
                    break

            if reached_res_cap_mid_page:
                break
            if len(collected) >= max_res and next_url:
                is_partial = True
                partial_reason = "max_everything_resources"
                break
            if not next_url:
                break
            url = _rewrite_next_link_to_fhir_auth(base_url, next_url)
            params = None
        else:
            if next_url:
                is_partial = True
                partial_reason = "max_everything_pages"

        return collected, is_partial, partial_reason

    async def assemble(
        self,
        *,
        options: WahsRequestOptions,
        identity: Optional[VerifiedCallerIdentity],
        requested_model: Optional[str] = None,
        force_refresh: bool = False,
    ) -> AssembledWahsContext:
        self._validate_overlay_budget(options.overlay_resources)
        scope = options.chart_scope
        spec = self.settings.resolve_model_spec(requested_model)
        budget_effective, clamp_reason = derive_effective_chart_budget(spec, options)
        effective_refresh = bool(options.refresh or force_refresh)

        # Mode A: Explicit local/harness inline FHIR Bundle (strictly gated by WAHS_ALLOW_LOCAL_BUNDLE_MODE)
        if options.fhir_bundle is not None:
            if not self.settings.allow_local_bundle_mode:
                raise HTTPException(
                    status_code=403,
                    detail="Inline wahs.fhir_bundle is disabled when WAHS_ALLOW_LOCAL_BUNDLE_MODE=false.",
                )
            verified_sub = identity.sub if identity is not None else (options.user_sub or "local-harness")
            raw_bundle = options.fhir_bundle
            if not isinstance(raw_bundle, dict) or raw_bundle.get("resourceType") != "Bundle":
                raise HTTPException(status_code=400, detail="wahs.fhir_bundle must be a FHIR R4 Bundle.")
            pat_id = str(options.master_patient_id or "").strip()
            if not pat_id:
                for entry in raw_bundle.get("entry") or []:
                    res = entry.get("resource") if isinstance(entry, Mapping) else None
                    if isinstance(res, Mapping) and res.get("resourceType") == "Patient":
                        pat_id = str(res.get("id") or "local-patient")
                        break
            pat_id = pat_id or "local-patient"
            datasets_fp = _canonical_sha256(raw_bundle)
            return self._scope_and_condense_bundle(
                raw_bundle=raw_bundle,
                verified_sub=verified_sub,
                authorized_patient_id=pat_id,
                allowed_patient_refs={pat_id},
                datasets_included=[{"dataset_id": pat_id, "source": "Inline Bundle", "category": "local"}],
                datasets_fingerprint=datasets_fp,
                options=options,
                spec=spec,
                budget_effective=budget_effective,
                clamp_reason=clamp_reason,
                dataset_cache_hits=0,
                dataset_fetches=0,
                cacheable=True,
                force_refresh=effective_refresh,
                partial_fetch=False,
                partial_datasets=[],
            )

        # Mode B: Production zero-privilege `fhir-auth` star-graph assembly
        if identity is None:
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization: Bearer <keycloak_jwt> header.",
            )

        base = self.settings.fhir_auth_base_url.rstrip("/")
        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self.settings.fhir_timeout_seconds)

        async def _run_live_assembly() -> AssembledWahsContext:
            # 1. Live MasterPatient fetch through `fhir-auth` (cryptographically verifies JWT + security label!)
            if options.master_patient_id:
                if not _SAFE_ID_RE.match(options.master_patient_id):
                    raise HTTPException(status_code=400, detail="Invalid master_patient_id format.")
                master_pat = await self._fhir_get_json(
                    client, f"{base}/Patient/{options.master_patient_id}", identity.raw_token
                )
            else:
                search_bundle = await self._fhir_get_json(
                    client,
                    f"{base}/Patient",
                    identity.raw_token,
                    params={"identifier": f"{IDENTIFIER_SYSTEM}|{identity.sub}"},
                )
                entries = search_bundle.get("entry") or []
                if not entries:
                    raise HTTPException(
                        status_code=404,
                        detail="No WAHS Master Patient found for authenticated user.",
                    )
                master_pat = (entries[0] or {}).get("resource") or {}

            if master_pat.get("resourceType") != "Patient":
                raise HTTPException(status_code=502, detail="Expected FHIR Patient resource for Master Patient.")

            master_id = str(master_pat.get("id") or "").strip()
            if not master_id:
                raise HTTPException(status_code=502, detail="Master Patient is missing resource id.")

            # 2. Extract & validate linked dataset Patient references (`masterPatient.link[]`)
            linked_ids: List[str] = []
            for link in master_pat.get("link") or []:
                ref = str(((link or {}).get("other") or {}).get("reference") or "").strip()
                if not ref:
                    continue
                m = _DATASET_REF_RE.match(ref)
                if not m:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Security guardrail: invalid Patient.link reference '{ref}'.",
                    )
                linked_ids.append(m.group(1))

            # 3. Fetch dataset Patient stubs in parallel via `fhir-auth` to read live `meta.lastUpdated` & WAHS tags
            raw_descriptors: List[DatasetDescriptor] = []
            if linked_ids:
                stub_sem = asyncio.Semaphore(max(1, int(self.settings.everything_concurrency)))

                async def _get_stub(ds_id: str) -> Dict[str, Any]:
                    async with stub_sem:
                        return await self._fhir_get_json(
                            client, f"{base}/Patient/{ds_id}", identity.raw_token
                        )

                stub_results = await asyncio.gather(*(_get_stub(ds_id) for ds_id in linked_ids))
                for ds_id, ds_pat in zip(linked_ids, stub_results):
                    meta = ds_pat.get("meta") if isinstance(ds_pat.get("meta"), Mapping) else {}
                    raw_stamp = str(meta.get("lastUpdated") or meta.get("versionId") or "").strip()
                    has_stamp = bool(raw_stamp)
                    last_upd = raw_stamp if has_stamp else f"ephemeral-{uuid.uuid4().hex}"
                    cat, meth, src, src_name, is_ph = _extract_patient_tags(ds_pat)
                    raw_descriptors.append(
                        DatasetDescriptor(
                            dataset_id=ds_id,
                            last_updated=last_upd,
                            category=cat,
                            method=meth,
                            source=src,
                            source_name=src_name,
                            is_placeholder_patient=is_ph,
                            has_version_stamp=has_stamp,
                        )
                    )

            # 4. Deduplicate collections sharing `(category, method, source)` keeping newest parsed UTC instant
            grouped: Dict[Tuple[str, str, str], List[DatasetDescriptor]] = {}
            for desc in raw_descriptors:
                grouped.setdefault((desc.category, desc.method, desc.source), []).append(desc)
            deduped_descriptors: List[DatasetDescriptor] = [
                sorted(
                    group,
                    key=lambda d: (_parse_fhir_instant(d.last_updated), d.dataset_id),
                    reverse=True,
                )[0]
                for group in grouped.values()
            ]

            # 5. Pre-filter datasets against `chart_scope` source/category filters BEFORE `$everything`
            included_descriptors = [
                d for d in deduped_descriptors if _dataset_matches_scope(d, scope)
            ]

            ips_allowed_keys: Optional[Set[str]] = None
            ips_version_marker = ""
            ips_has_stamp = True
            if scope.ips_bundle_id:
                if not _SAFE_ID_RE.match(scope.ips_bundle_id):
                    raise HTTPException(status_code=400, detail="Invalid ips_bundle_id format.")
                ips_bundle = await self._fhir_get_json(
                    client, f"{base}/Bundle/{scope.ips_bundle_id}", identity.raw_token
                )
                ips_allowed_keys = _extract_ips_referenced_keys(ips_bundle)
                ips_meta = ips_bundle.get("meta") if isinstance(ips_bundle.get("meta"), Mapping) else {}
                raw_ips_ver = str(ips_meta.get("lastUpdated") or ips_meta.get("versionId") or "").strip()
                ips_has_stamp = bool(raw_ips_ver)
                ips_version_marker = raw_ips_ver if ips_has_stamp else f"ephemeral-{uuid.uuid4().hex}"

            master_meta = master_pat.get("meta") if isinstance(master_pat.get("meta"), Mapping) else {}
            raw_master_ver = str(master_meta.get("lastUpdated") or master_meta.get("versionId") or "").strip()
            master_has_stamp = bool(raw_master_ver)
            master_last_updated = raw_master_ver if master_has_stamp else f"ephemeral-{uuid.uuid4().hex}"

            cacheable = (
                master_has_stamp
                and ips_has_stamp
                and all(d.has_version_stamp for d in included_descriptors)
            )

            datasets_fp = _canonical_sha256(
                {
                    "master": (master_id, master_last_updated),
                    "datasets": [(d.dataset_id, d.last_updated) for d in included_descriptors],
                    "ips": ips_version_marker,
                }
            )
            scope_hash = _canonical_sha256(
                {
                    "scope": scope.model_dump(),
                    "overlays": options.overlay_resources,
                    "budget": budget_effective,
                    "posture": options.sharing_posture,
                    "tokenizer_family": spec.tokenizer_family,
                }
            )

            # Check prefix cache BEFORE fetching `$everything` when all timestamps are present & refresh not forced
            prefix_key = (identity.sub, datasets_fp, scope_hash)
            if cacheable and not effective_refresh:
                cached_ctx = self._prefix_cache.get(prefix_key)
                if cached_ctx is not None:
                    return _clone_context_for_request(
                        cached_ctx,
                        cache_hit=True,
                        budget_requested=options.target_budget_tokens,
                        clamp_reason=clamp_reason,
                    )

            # 6. Fetch paginated `$everything` concurrently with semaphore bound for uncached / stale datasets
            pooled_entries: List[Dict[str, Any]] = [{"resource": master_pat}]
            seen_resource_keys: Set[str] = {f"Patient/{master_id}"}
            allowed_refs: Set[str] = {master_id, *(d.dataset_id for d in deduped_descriptors)}
            fetch_sem = asyncio.Semaphore(max(1, int(self.settings.everything_concurrency)))

            async def _fetch_dataset(
                desc: DatasetDescriptor,
            ) -> Tuple[DatasetDescriptor, List[Dict[str, Any]], bool, Optional[str], int, int]:
                ds_cache_key = (identity.sub, desc.dataset_id, desc.last_updated)
                if cacheable and desc.has_version_stamp and not effective_refresh:
                    cached_resources = self._dataset_cache.get(ds_cache_key)
                    if cached_resources is not None:
                        return desc, cached_resources, False, None, 1, 0

                async with fetch_sem:
                    raw_ds_resources, is_partial, partial_reason = await self._fetch_paginated_everything(
                        client, base, desc.dataset_id, identity.raw_token
                    )
                ds_resources = [
                    _stamp_resource_source_tag(r, desc)
                    for r in raw_ds_resources
                    if not (r.get("resourceType") == "Patient" and r.get("id") in allowed_refs)
                ]
                if not is_partial and desc.has_version_stamp:
                    self._dataset_cache.put(ds_cache_key, ds_resources)
                return desc, ds_resources, is_partial, partial_reason, 0, 1

            ds_results = await asyncio.gather(*(_fetch_dataset(desc) for desc in included_descriptors))
            ds_hits = sum(item[4] for item in ds_results)
            ds_fetches = sum(item[5] for item in ds_results)
            partial_datasets: List[Dict[str, Any]] = []

            for desc, ds_resources, is_partial, partial_reason, _, _ in ds_results:
                if is_partial:
                    partial_datasets.append(
                        {
                            "dataset_id": desc.dataset_id,
                            "source": desc.source_name,
                            "reason": partial_reason or "truncated",
                            "collected_resources": len(ds_resources),
                        }
                    )
                for res in ds_resources:
                    rt = str(res.get("resourceType") or "")
                    rid = str(res.get("id") or "")
                    if ips_allowed_keys is not None and rt not in _INFRASTRUCTURE_RESOURCE_TYPES:
                        if f"{rt}/{rid}" not in ips_allowed_keys and rid not in ips_allowed_keys:
                            continue
                    rkey = f"{rt}/{rid}" if rid else uuid.uuid4().hex
                    if rkey not in seen_resource_keys:
                        seen_resource_keys.add(rkey)
                        pooled_entries.append({"resource": res})

            partial_fetch = bool(partial_datasets)
            if (
                partial_fetch
                and options.sharing_posture == "external_share"
                and not options.allow_partial_external_share
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "External share rejected because patient chart fetch was truncated "
                        "(allow_partial_external_share=false)."
                    ),
                )

            pooled_bundle = {
                "resourceType": "Bundle",
                "id": f"wahs-pooled-{master_id}",
                "type": "collection",
                "entry": pooled_entries,
            }
            return self._scope_and_condense_bundle(
                raw_bundle=pooled_bundle,
                verified_sub=identity.sub,
                authorized_patient_id=master_id,
                allowed_patient_refs=allowed_refs,
                datasets_included=[
                    {
                        "dataset_id": d.dataset_id,
                        "source": d.source_name,
                        "category": d.category,
                        "method": d.method,
                        "last_updated": d.last_updated,
                    }
                    for d in included_descriptors
                ],
                datasets_fingerprint=datasets_fp,
                options=options,
                spec=spec,
                budget_effective=budget_effective,
                clamp_reason=clamp_reason,
                dataset_cache_hits=ds_hits,
                dataset_fetches=ds_fetches,
                cacheable=(cacheable and not partial_fetch),
                force_refresh=effective_refresh,
                partial_fetch=partial_fetch,
                partial_datasets=partial_datasets,
            )

        try:
            return await asyncio.wait_for(
                _run_live_assembly(),
                timeout=max(1.0, float(self.settings.assembly_timeout_seconds)),
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"FHIR chart assembly timed out after "
                    f"{self.settings.assembly_timeout_seconds:.1f}s."
                ),
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    def _scope_and_condense_bundle(
        self,
        *,
        raw_bundle: Dict[str, Any],
        verified_sub: str,
        authorized_patient_id: str,
        allowed_patient_refs: Set[str],
        datasets_included: List[Dict[str, str]],
        datasets_fingerprint: str,
        options: WahsRequestOptions,
        spec: ModelSpec,
        budget_effective: int,
        clamp_reason: Optional[str],
        dataset_cache_hits: int,
        dataset_fetches: int,
        cacheable: bool = True,
        force_refresh: bool = False,
        partial_fetch: bool = False,
        partial_datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> AssembledWahsContext:
        scope = options.chart_scope
        scope_hash = _canonical_sha256(
            {
                "scope": scope.model_dump(),
                "overlays": options.overlay_resources,
                "budget": budget_effective,
                "posture": options.sharing_posture,
                "tokenizer_family": spec.tokenizer_family,
            }
        )
        prefix_key = (verified_sub, datasets_fingerprint, scope_hash)
        if cacheable and not force_refresh and not partial_fetch:
            cached = self._prefix_cache.get(prefix_key)
            if cached is not None:
                return _clone_context_for_request(
                    cached,
                    cache_hit=True,
                    budget_requested=options.target_budget_tokens,
                    clamp_reason=clamp_reason,
                )

        overlay_fhir_entries, draft_block, draft_citations = _format_overlay_items(
            options.overlay_resources, authorized_patient_id
        )

        merged_entries = list(raw_bundle.get("entry") or []) + overlay_fhir_entries
        scoped_entries: List[Dict[str, Any]] = []
        inc_types = {t.strip() for t in scope.include_resource_types if t.strip()}
        exc_types = {t.strip() for t in scope.exclude_resource_types if t.strip()}
        inc_ids = {i.strip() for i in scope.include_resource_ids if i.strip()}
        exc_ids = {i.strip() for i in scope.exclude_resource_ids if i.strip()}
        inc_sources = {s.strip().lower() for s in scope.include_sources if s.strip()}
        exc_sources = {s.strip().lower() for s in scope.exclude_sources if s.strip()}

        for entry in merged_entries:
            res = entry.get("resource") if isinstance(entry, Mapping) else None
            if not isinstance(res, dict):
                continue
            rt = str(res.get("resourceType") or "")
            rid = str(res.get("id") or "")
            full_id = f"{rt}/{rid}"

            if rt == "Patient":
                scoped_entries.append({"resource": res})
                continue

            if rid in exc_ids or full_id in exc_ids:
                continue
            if inc_ids and rid not in inc_ids and full_id not in inc_ids and rt not in _INFRASTRUCTURE_RESOURCE_TYPES:
                continue
            if rt in exc_types:
                continue
            if inc_types and rt not in inc_types and rt not in _INFRASTRUCTURE_RESOURCE_TYPES:
                continue

            # Resource-level source filter (also supports inline bundles with performer/meta tags)
            if (inc_sources or exc_sources) and rt not in _INFRASTRUCTURE_RESOURCE_TYPES:
                res_src = ""
                meta = res.get("meta") if isinstance(res.get("meta"), Mapping) else {}
                for tag in meta.get("tag") or []:
                    if isinstance(tag, Mapping) and str(tag.get("system") or "").endswith("/wahealthsummary-source-name"):
                        res_src = str(tag.get("code") or "").strip().lower()
                        break
                if not res_src:
                    res_src = str(meta.get("source") or "").strip().lower()
                if not res_src and isinstance(res.get("serviceProvider"), Mapping):
                    res_src = str(res["serviceProvider"].get("display") or "").strip().lower()
                if not res_src:
                    for perf in res.get("performer") or []:
                        if isinstance(perf, Mapping):
                            actor = perf.get("actor") or perf
                            if isinstance(actor, Mapping) and actor.get("display"):
                                res_src = str(actor["display"]).strip().lower()
                                break
                if exc_sources and res_src and any(e in res_src for e in exc_sources):
                    continue
                if inc_sources and (not res_src or not any(i in res_src for i in inc_sources)):
                    continue

            if (scope.date_from or scope.date_to) and rt not in _INFRASTRUCTURE_RESOURCE_TYPES:
                r_date = _extract_resource_iso_date(res)
                if r_date:
                    if scope.date_from and r_date < scope.date_from:
                        continue
                    if scope.date_to and r_date > scope.date_to:
                        continue

            scoped_entries.append({"resource": res})

        candidate_bundle = {
            "resourceType": "Bundle",
            "id": raw_bundle.get("id") or f"scoped-{authorized_patient_id}",
            "type": "collection",
            "entry": scoped_entries,
        }
        guarded_bundle, security_audit = enforce_single_patient_scope(
            candidate_bundle,
            authorized_patient_id=authorized_patient_id,
            allowed_patient_refs=allowed_patient_refs,
        )

        exact_counter = get_exact_token_counter(spec.tokenizer_family)
        partial_list = list(partial_datasets or [])
        partial_warning = (
            "[PARTIAL RECORD WARNING: Dataset pagination/resource ceiling reached before full retrieval; record may be incomplete.]"
            if partial_fetch
            else ""
        )
        warning_reserve = (exact_counter(partial_warning) + 4) if partial_fetch else 0
        condense_target = max(512, budget_effective - warning_reserve)

        extracted = extract_bundle(guarded_bundle)
        try:
            condensed = condense_bundle_to_budget(
                extracted,
                target_tokens=condense_target,
                sharing_posture=options.sharing_posture,
                token_counter=exact_counter,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if partial_fetch:
            updated_manifest = dict(condensed.completeness_manifest)
            updated_manifest["partial_fetch"] = True
            updated_manifest["partial_datasets"] = partial_list
            updated_chart_text = f"{condensed.chart_text.rstrip()}\n{partial_warning}"
            updated_tokens = exact_counter(updated_chart_text)
            updated_stats = dict(condensed.compression_stats)
            updated_stats["actual_tokens"] = updated_tokens
            updated_stats["target_tokens"] = budget_effective
            updated_stats["budget_utilization"] = round(updated_tokens / budget_effective, 3)
            condensed = replace(
                condensed,
                target_tokens=budget_effective,
                actual_tokens=updated_tokens,
                chart_text=updated_chart_text,
                completeness_manifest=updated_manifest,
                compression_stats=updated_stats,
            )
        else:
            updated_manifest = dict(condensed.completeness_manifest)
            updated_manifest.setdefault("partial_fetch", False)
            condensed = replace(condensed, completeness_manifest=updated_manifest)

        # Build rich `[N] -> source pill` citation map
        source_by_uuid: Dict[str, Tuple[str, str]] = {}
        sensitive_items: List[str] = []
        for item in extracted.items:
            source_by_uuid[item.resource_uuid] = (
                item.source_org or "Patient Record",
                item.section or "",
            )
            if item.sensitivity == "SDOH/PRIVACY-SENSITIVE":
                sensitive_items.append(item.display)

        citation_map: Dict[str, Dict[str, Any]] = {}
        for m in condensed.mappings or []:
            if not isinstance(m, Mapping):
                continue
            idx_str = str(m.get("index") or "")
            if not idx_str:
                continue
            ruuid = str(m.get("resourceUuid") or "")
            src_label, section_name = source_by_uuid.get(ruuid, ("WA Health Summary", ""))
            entry_info: Dict[str, Any] = {
                "index": m.get("index"),
                "resourceType": m.get("resourceType"),
                "id": ruuid,
                "source": src_label,
                "section": section_name,
                "date": m.get("date") or "",
                "display": m.get("text") or "",
            }
            if m.get("memberUuids"):
                entry_info["memberIds"] = list(m["memberUuids"])
            if m.get("sensitivity"):
                entry_info["sensitivity"] = m["sensitivity"]
            citation_map[idx_str] = entry_info

        for pk, pv in draft_citations.items():
            citation_map[pk] = pv

        privacy_notices: List[str] = []
        if options.sharing_posture == "external_share" and condensed.redacted_for_privacy:
            privacy_notices.append(
                f"{len(condensed.redacted_for_privacy)} privacy-sensitive item(s) redacted under external_share posture."
            )
        elif options.sharing_posture == "patient_self" and sensitive_items:
            privacy_notices.append(
                f"{len(sensitive_items)} SDOH/privacy-sensitive item(s) present in record; review before external sharing."
            )

        if partial_fetch:
            ds_names = ", ".join(str(p.get("source") or p.get("dataset_id") or "") for p in partial_list)
            privacy_notices.append(
                f"Partial record warning: FHIR fetch reached configured limit on {ds_names or 'one or more datasets'}; answer completeness cannot be guaranteed."
            )

        system_prompt = (
            f"{WAHS_SYSTEM_PROMPT}\n\n"
            f"=== ACTIVE PATIENT: {extracted.patient_name} (DOB: {extracted.birth_date or 'unknown'}) ===\n"
            f"=== CONDENSED CLINICAL RECORD ([1]..[N]) ===\n"
            f"{condensed.chart_text}"
        )
        if draft_block:
            system_prompt = f"{system_prompt}\n\n{draft_block}"

        prefix_hash = hashlib.sha256(
            f"{verified_sub}:{datasets_fingerprint}:{scope_hash}".encode("utf-8")
        ).hexdigest()[:20]

        ctx = AssembledWahsContext(
            verified_sub=verified_sub,
            patient_id=authorized_patient_id,
            patient_name=extracted.patient_name,
            birth_date=extracted.birth_date or "",
            system_prompt=system_prompt,
            prefix_cache_key_hash=prefix_hash,
            datasets_fingerprint=datasets_fingerprint,
            scope_hash=scope_hash,
            datasets_included=datasets_included,
            scope_applied=scope.model_dump(exclude_defaults=False),
            security_audit=security_audit,
            extracted=extracted,
            condensed=condensed,
            citation_map=citation_map,
            privacy_notices=privacy_notices,
            cache_hit=False,
            dataset_cache_hits=dataset_cache_hits,
            dataset_fetches=dataset_fetches,
            budget_requested=options.target_budget_tokens,
            budget_effective=budget_effective,
            context_window=spec.context_window,
            chart_budget_ceiling=spec.chart_budget_ceiling,
            clamp_reason=clamp_reason,
            tokenizer_family=spec.tokenizer_family,
            partial_fetch=partial_fetch,
            partial_datasets=partial_list,
        )
        if cacheable and not partial_fetch:
            self._prefix_cache.put(prefix_key, _clone_context_for_request(ctx, cache_hit=False))
        return _clone_context_for_request(ctx, cache_hit=False)
