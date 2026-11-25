"""Zero-privilege `fhir-auth` star-graph `WAHSBundleAssembler`, `chart_scope`, and bounded caches."""

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

from wahs_core import CondensedChartResult, ExtractedChart, condense_bundle_to_budget, extract_bundle

from .config import ModelSpec, WahsEngineSettings
from .jwt_claims import VerifiedCallerIdentity
from .schemas import ChartScope, WahsRequestOptions
from .tokenizers_registry import count_chat_messages_tokens, get_exact_token_counter

WAHS_SYSTEM_PROMPT = """You are the Washington Health Summary (WAHS) AI Assistant, a plain-language clinical record companion.

GROUNDING & PROVENANCE RULES:
1. RECORD GROUNDING & INLINE [N] CITATIONS: Answer strictly from the Condensed Clinical Record (`[1]..[N]`) and any Patient-Authored Drafts/Flags (`[P1]..[PM]`). Cite supporting facts inline as `[1]`, `[2]`, or `[P1]` so the UI renders interactive source links, and note which care organization or patient draft reported each item when multiple sources appear.
2. COMPLETENESS & SCOPE HONESTY: Never fabricate diagnoses, medications, lab values, or dates. Check the `[Completeness Manifest]` before concluding a historical absence: if items were omitted or notes excerpted due to token budget, or if the chart is explicitly scoped, state what is visible in the current view rather than asserting absence across omitted history.
3. EXACT QUANTITATIVE & TEMPORAL FIDELITY: Copy numeric values, units, comparators (`<`, `>`), and ISO `YYYY-MM-DD` clinical event dates verbatim without rounding or inferring trends from a single measurement. Do not treat administrative, coverage, or device manufacture dates as clinical visits.
4. SUPPORTIVE, NON-PRESCRIPTIVE CLINICAL FRAMING: Use clear, accessible language tailored to the user's question and suggest 1-2 concrete follow-up questions for the patient's care team when helpful."""

WAHS_SYSTEM_BASE = "http://fhir.wahealthsummary.cirg.uw.edu"
IDENTIFIER_SYSTEM = "http://keycloak.cirg.uw.edu"
SECURITY_SYSTEM = "http://keycloak.cirg.uw.edu/fhir/security-labels"
CATEGORY_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-category"
METHOD_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-method"
PLACEHOLDER_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-placeholder"
SOURCE_NAME_SYSTEM = f"{WAHS_SYSTEM_BASE}/CodeSystem/wahealthsummary-source-name"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DATASET_REF_RE = re.compile(r"^Patient/([A-Za-z0-9._-]+)$")
_INFRASTRUCTURE_TYPES: Set[str] = {"Organization", "Practitioner", "PractitionerRole", "Location", "Provenance", "Medication"}
_DATE_FIELDS: Tuple[str, ...] = ("effectiveDateTime", "authoredOn", "occurrenceDateTime", "recordedDate", "onsetDateTime", "issued", "date", "started", "startDate", "statusDate", "created")
_SENTINEL = object()
K, V = TypeVar("K"), TypeVar("V")


def _estimate_cache_bytes(value: Any) -> int:
    if isinstance(value, AssembledWahsContext):
        return max(256, len(value.system_prompt.encode("utf-8")) + len(json.dumps(value.citation_map, default=str).encode("utf-8")) + sum(len(it.text.encode("utf-8")) + 192 for it in value.extracted.items))
    if isinstance(value, tuple) and value and isinstance(value[0], ExtractedChart):
        ext = value[0]
        return max(256, sum(len(it.text.encode("utf-8")) + 192 for it in ext.items))
    try:
        return max(64, len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")))
    except Exception:
        return 1024


class _BoundedLRUCache(Generic[K, V]):
    def __init__(self, max_entries: int, ttl_seconds: float = 1800.0, max_bytes: Optional[int] = None) -> None:
        self._max, self._ttl, self._total_bytes = max(1, max_entries), max(0.0, float(ttl_seconds)), 0
        self._max_bytes = int(max_bytes) if (max_bytes is not None and int(max_bytes) > 0) else None
        self._data: OrderedDict[K, Tuple[float, int, V]] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            return None
        exp, sb, val = self._data[key]
        if self._ttl > 0.0 and time.monotonic() >= exp:
            self._total_bytes = max(0, self._total_bytes - sb)
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return val

    def put(self, key: K, value: V) -> None:
        sb = _estimate_cache_bytes(value)
        if key in self._data:
            self._total_bytes = max(0, self._total_bytes - self._data.pop(key)[1])
        if self._max_bytes is not None and sb > self._max_bytes:
            return
        self._data[key] = ((time.monotonic() + self._ttl) if self._ttl > 0.0 else float("inf"), sb, value)
        self._total_bytes += sb
        self._data.move_to_end(key)
        while len(self._data) > self._max or (self._max_bytes is not None and self._total_bytes > self._max_bytes and len(self._data) > 0):
            self._total_bytes = max(0, self._total_bytes - self._data.popitem(last=False)[1][1])

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._data)


def _parse_fhir_instant(ts: str) -> Tuple[int, float, str]:
    raw = str(ts or "").strip()
    if not raw:
        return (0, 0.0, "")
    try:
        dt = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
        return (1, (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)).timestamp(), raw)
    except ValueError:
        return (0, 0.0, raw)


def normalize_conversation_turns(conversation_messages: Sequence[Any]) -> List[Dict[str, str]]:
    """Extract normalized non-empty user/assistant chat turns from OpenAI message objects."""
    turns: List[Dict[str, str]] = []
    for m in conversation_messages or ():
        role_raw = m.get("role") if isinstance(m, Mapping) else getattr(m, "role", "")
        content_raw = m.get("content") if isinstance(m, Mapping) else getattr(m, "content", "")
        role = str(role_raw or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if isinstance(content_raw, list):
            parts = [str(p.get("text") or "") for p in content_raw if isinstance(p, Mapping) and p.get("type") in {"text", "input_text", None}]
            text_val = "\n".join(p for p in parts if p)
        else:
            text_val = str(content_raw or "")
        if text_val.strip():
            turns.append({"role": role, "content": text_val})
    return turns


def resolve_chart_budget(
    spec: ModelSpec,
    options: WahsRequestOptions,
    *,
    system_framing_text: str = WAHS_SYSTEM_PROMPT,
    conversation_messages: Sequence[Any] = (),
    requested_max_tokens: Optional[int] = None,
) -> Tuple[int, str]:
    """Compute physical remaining chart budget from exact turn overhead and resolve policy."""
    effective_max_tokens = int(requested_max_tokens) if requested_max_tokens is not None else int(spec.default_max_output_tokens)
    if effective_max_tokens >= spec.context_window:
        raise HTTPException(
            status_code=422,
            detail=f"Requested max_tokens ({effective_max_tokens}) equals or exceeds model context_window ({spec.context_window}).",
        )
    turns = normalize_conversation_turns(conversation_messages)
    probe_messages: List[Dict[str, str]] = [{"role": "system", "content": system_framing_text}, *turns]
    while len(probe_messages) > 2:
        toks = count_chat_messages_tokens(probe_messages, tokenizer_family=spec.tokenizer_family)
        if toks + effective_max_tokens < spec.context_window:
            break
        probe_messages.pop(1)
        while len(probe_messages) > 2 and probe_messages[1].get("role") != "user":
            probe_messages.pop(1)
    # 4-token BPE boundary allowance for joining chart_text into system framing
    non_chart_tokens = count_chat_messages_tokens(probe_messages, tokenizer_family=spec.tokenizer_family) + effective_max_tokens + 4
    physical_chart_capacity = spec.context_window - non_chart_tokens
    if physical_chart_capacity <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"Configured model context_window={spec.context_window} cannot fit required system framing, latest turn, and max_tokens={effective_max_tokens}.",
        )
    if options.target_budget_tokens is not None:
        req_b = int(options.target_budget_tokens)
        if req_b <= physical_chart_capacity:
            return req_b, "caller_target_budget_tokens"
        return physical_chart_capacity, "clamped_to_context_window"
    if spec.preferred_chart_tokens is not None and spec.preferred_chart_tokens < physical_chart_capacity:
        return int(spec.preferred_chart_tokens), "model_preferred_chart_tokens"
    return physical_chart_capacity, "context_window_available"


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
    notices: List[str] = field(default_factory=list)
    cache_hit: bool = False
    dataset_cache_hits: int = 0
    dataset_fetches: int = 0
    budget_requested: Optional[int] = None
    budget_effective: int = 0
    context_window: int = 32768
    preferred_chart_tokens: Optional[int] = None
    clamp_reason: Optional[str] = None
    tokenizer_family: str = "gemma4"
    partial_fetch: bool = False
    partial_datasets: List[Dict[str, Any]] = field(default_factory=list)


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _extract_patient_tags(pat: Mapping[str, Any]) -> Tuple[str, str, str, str, bool]:
    meta = pat.get("meta") if isinstance(pat.get("meta"), Mapping) else {}
    cat, meth, sn, ph = "", "", "", False
    for t in (meta.get("tag") or []) if isinstance(meta.get("tag"), list) else ():
        if isinstance(t, Mapping):
            su, code = str(t.get("system") or ""), str(t.get("code") or "").strip()
            cat = code if (su == CATEGORY_SYSTEM and not cat) else cat
            meth = code if (su == METHOD_SYSTEM and not meth) else meth
            sn = code if (su == SOURCE_NAME_SYSTEM and not sn) else sn
            ph = True if (su == PLACEHOLDER_SYSTEM and code) else ph
    raw_src = str(meta.get("source") or "").split("#", 1)[0].strip()
    eff = sn or raw_src or cat or "WAHS Dataset"
    return cat or "provider-health-record", meth or "unknown", raw_src or eff, eff, ph


def _stamp_resource_source_tag(res: Dict[str, Any], desc: DatasetDescriptor) -> Dict[str, Any]:
    stamped = copy.deepcopy(res)
    meta = stamped.setdefault("meta", {}) if isinstance(stamped.get("meta"), dict) else stamped.setdefault("meta", {})
    tags = meta.setdefault("tag", []) if isinstance(meta.get("tag"), list) else meta.setdefault("tag", [])
    if desc.source_name and not any(isinstance(t, Mapping) and t.get("system") == SOURCE_NAME_SYSTEM for t in tags):
        tags.append({"system": SOURCE_NAME_SYSTEM, "code": desc.source_name})
    if desc.category and not any(isinstance(t, Mapping) and t.get("system") == CATEGORY_SYSTEM for t in tags):
        tags.append({"system": CATEGORY_SYSTEM, "code": desc.category})
    return stamped


def _dataset_matches_scope(desc: DatasetDescriptor, scope: ChartScope) -> bool:
    cands = {desc.source_name.lower(), desc.source.lower(), desc.dataset_id.lower()}
    if scope.include_sources:
        inc = {s.strip().lower() for s in scope.include_sources if s.strip()}
        if inc and not any(c in inc or any(i in c for i in inc) for c in cands if c):
            return False
    if scope.exclude_sources:
        exc = {s.strip().lower() for s in scope.exclude_sources if s.strip()}
        if exc and any(c in exc or any(e in c for e in exc) for c in cands if c):
            return False
    cat_l = desc.category.lower()
    return not ((scope.include_categories and cat_l not in {c.strip().lower() for c in scope.include_categories if c.strip()}) or (scope.exclude_categories and cat_l in {c.strip().lower() for c in scope.exclude_categories if c.strip()}))


def _extract_resource_iso_date(res: Mapping[str, Any]) -> Optional[str]:
    for fn in _DATE_FIELDS:
        v = res.get(fn)
        if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-":
            return v[:10]
    for pf in ("effectivePeriod", "period"):
        per = res.get(pf)
        if isinstance(per, Mapping):
            for b in ("end", "start"):
                v = per.get(b)
                if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-":
                    return v[:10]
    return None


def _extract_subject_ref(res: Mapping[str, Any]) -> Optional[str]:
    for fn in ("subject", "patient"):
        v = res.get(fn)
        if isinstance(v, Mapping) and str(v.get("reference") or "").strip():
            return str(v["reference"]).strip()
    return None


def enforce_single_patient_scope(bundle: Dict[str, Any], *, authorized_patient_id: str, allowed_patient_refs: Set[str], strict_reject: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        raise HTTPException(status_code=400, detail="Payload must be a FHIR R4 Bundle.")
    allowed: Set[str] = {c for raw in {authorized_patient_id, *allowed_patient_refs} for r in [str(raw or "").strip()] if r for c in ((r, r.split("/", 1)[1]) if r.startswith("Patient/") else (r,))}
    entries = bundle.get("entry") or []
    for e in entries:
        res = e.get("resource") if isinstance(e, Mapping) else None
        if isinstance(res, Mapping) and res.get("resourceType") == "Patient" and res.get("id") and not allowed:
            allowed.add(str(res["id"]).strip())
            break
    filtered, stripped, seen_pat = [], [], False
    for entry in entries:
        res = entry.get("resource") if isinstance(entry, Mapping) else None
        if not isinstance(res, Mapping):
            continue
        rt, rid = str(res.get("resourceType") or ""), str(res.get("id") or "")
        if rt == "Patient":
            if rid and rid not in allowed:
                if strict_reject:
                    raise HTTPException(status_code=403, detail=f"Foreign Patient resource '{rid}' rejected.")
                stripped.append({"resourceType": rt, "id": rid, "reason": "foreign_patient"})
                continue
            if not seen_pat:
                seen_pat = True
                filtered.append({"resource": dict(res)})
        elif rt in _INFRASTRUCTURE_TYPES:
            filtered.append({"resource": dict(res)})
        else:
            sref = _extract_subject_ref(res)
            if sref:
                sid = sref.split("/", 1)[1] if "/" in sref else sref
                sid = sid[len("urn:uuid:") :] if sid.startswith("urn:uuid:") else sid
                if sid not in allowed and sref not in allowed:
                    if strict_reject:
                        raise HTTPException(status_code=403, detail=f"Resource {rt}/{rid} references unauthorized subject '{sref}'.")
                    stripped.append({"resourceType": rt, "id": rid, "subject": sref, "reason": "unauthorized_subject"})
                    continue
            filtered.append({"resource": dict(res)})
    return {**bundle, "entry": filtered}, {"authorized_patient_id": authorized_patient_id, "allowed_patient_ids": sorted(allowed), "input_entries": len(entries), "scoped_entries": len(filtered), "stripped_foreign_count": len(stripped), "stripped_foreign_resources": stripped}


def _rewrite_next_link_to_fhir_auth(fhir_auth_base_url: str, raw_next_url: str) -> str:
    base, parsed = fhir_auth_base_url.rstrip("/"), urlsplit(raw_next_url)
    path = parsed.path or ""
    idx = path.find("/fhir")
    rel = path[idx + len("/fhir") :] if idx >= 0 else path
    target = f"{base}/" if (not rel or rel == "/") else f"{base}/{rel.lstrip('/')}"
    return f"{target}?{parsed.query}" if parsed.query else target


def _format_overlay_items(overlay_resources: Sequence[Mapping[str, Any]], authorized_patient_id: str) -> Tuple[List[Dict[str, Any]], str, Dict[str, Dict[str, Any]]]:
    fhir_entries, draft_lines, draft_cits, p_idx = [], [], {}, 1
    for idx, item in enumerate(overlay_resources):
        if not isinstance(item, Mapping):
            continue
        if item.get("resourceType"):
            rc = copy.deepcopy(dict(item))
            rc.setdefault("id", f"overlay-{idx + 1}")
            if rc["resourceType"] != "Patient" and not _extract_subject_ref(rc):
                rc["subject"] = {"reference": f"Patient/{authorized_patient_id}"}
            rc.setdefault("meta", {}).setdefault("tag", []).append({"system": SOURCE_NAME_SYSTEM, "code": "WAHS Unsaved Draft"})
            fhir_entries.append({"resource": rc})
        else:
            kind, lbl, txt = str(item.get("type") or item.get("category") or "Draft Note").strip(), str(item.get("item") or item.get("title") or "").strip(), str(item.get("text") or item.get("note") or "").strip()
            body = f"{lbl}: {txt}" if (lbl and txt) else (lbl or txt)
            if body:
                pid = f"P{p_idx}"
                p_idx += 1
                draft_lines.append(f"[{pid}] ({kind}) {body} [source: WAHS Unsaved Draft]")
                draft_cits[pid] = {"index": pid, "resourceType": "PatientDraftOverlay", "id": str(item.get("id") or f"overlay-draft-{p_idx - 1}"), "source": "WAHS Unsaved Draft", "category": "patient-story", "date": str(item.get("date") or "draft"), "display": body}
    return fhir_entries, ("=== UNSAVED PATIENT DRAFT NOTES & FLAGS ([P1]..[PM]) ===\n" + "\n".join(draft_lines)) if draft_lines else "", draft_cits


def _extract_ips_referenced_keys(ips_bundle: Mapping[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for e in ips_bundle.get("entry") or []:
        res = e.get("resource") if isinstance(e, Mapping) else None
        if not isinstance(res, Mapping):
            continue
        rt, rid = str(res.get("resourceType") or ""), str(res.get("id") or "")
        if rt and rid:
            keys.update((f"{rt}/{rid}", rid))
        if rt == "Composition":
            for sec in res.get("section") or []:
                for ro in ((sec or {}).get("entry") or []) if isinstance(sec, Mapping) else ():
                    ref = str((ro or {}).get("reference") or "").strip()
                    if ref:
                        keys.add(ref)
                        if "/" in ref:
                            keys.add(ref.split("/", 1)[1])
    return keys


def _clone_context(ctx: AssembledWahsContext, *, cache_hit: bool, budget_requested: Any = _SENTINEL, budget_effective: Any = _SENTINEL, clamp_reason: Any = _SENTINEL, dataset_cache_hits: Any = _SENTINEL, dataset_fetches: Any = _SENTINEL) -> AssembledWahsContext:
    cloned = copy.copy(ctx)
    cloned.citation_map = copy.deepcopy(ctx.citation_map)
    cloned.notices, cloned.partial_datasets, cloned.cache_hit = list(ctx.notices), [dict(d) for d in ctx.partial_datasets], cache_hit
    if budget_requested is not _SENTINEL:
        cloned.budget_requested = budget_requested
    if budget_effective is not _SENTINEL:
        cloned.budget_effective = budget_effective
    if clamp_reason is not _SENTINEL:
        cloned.clamp_reason = clamp_reason
    if dataset_cache_hits is not _SENTINEL:
        cloned.dataset_cache_hits = dataset_cache_hits
    if dataset_fetches is not _SENTINEL:
        cloned.dataset_fetches = dataset_fetches
    return cloned


def _trimmed_turns_for_window(
    turns: Sequence[Dict[str, str]],
    *,
    system_text: str,
    spec: ModelSpec,
    effective_max_tokens: int,
) -> List[Dict[str, str]]:
    probe: List[Dict[str, str]] = [{"role": "system", "content": system_text}, *turns]
    while len(probe) > 2:
        if count_chat_messages_tokens(probe, tokenizer_family=spec.tokenizer_family) + effective_max_tokens < spec.context_window:
            break
        probe.pop(1)
        while len(probe) > 2 and probe[1].get("role") != "user":
            probe.pop(1)
    return probe[1:]


class WAHSBundleAssembler:
    """Assembles, scopes, caches, and condenses multi-dataset WAHS charts via `fhir-auth`."""

    def __init__(self, settings: WahsEngineSettings, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.settings, self._external_client, self._shared_client = settings, http_client, None
        self._dataset_cache: _BoundedLRUCache[Tuple[str, str, str], List[Dict[str, Any]]] = _BoundedLRUCache(settings.dataset_cache_max_entries, ttl_seconds=settings.cache_ttl_seconds, max_bytes=settings.dataset_cache_max_bytes)
        self._extracted_cache: _BoundedLRUCache[Tuple[str, str, str], Tuple[ExtractedChart, Dict[str, Any], str, Dict[str, Dict[str, Any]], str]] = _BoundedLRUCache(settings.prefix_cache_max_entries, ttl_seconds=settings.cache_ttl_seconds, max_bytes=settings.prefix_cache_max_bytes)
        self._prefix_cache: _BoundedLRUCache[Tuple[str, str, str, str, str], AssembledWahsContext] = _BoundedLRUCache(settings.prefix_cache_max_entries, ttl_seconds=settings.cache_ttl_seconds, max_bytes=settings.prefix_cache_max_bytes)
        self._inflight_datasets: Dict[Tuple[str, str, str], asyncio.Task[Tuple[List[Dict[str, Any]], bool, Optional[str]]]] = {}

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._shared_client is None or self._shared_client.is_closed:
            self._shared_client = httpx.AsyncClient(timeout=self.settings.fhir_timeout_seconds)
        return self._shared_client

    async def aclose(self) -> None:
        if self._shared_client is not None and not self._shared_client.is_closed:
            await self._shared_client.aclose()
        self._shared_client = None

    async def _fhir_get_json(self, client: httpx.AsyncClient, url: str, token: str, params: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json, application/json"})
        if resp.status_code in (401, 403):
            raise HTTPException(status_code=resp.status_code, detail=f"fhir-auth denied request ({resp.status_code}).")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"FHIR resource not found at {url}.")
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("resourceType") == "OperationOutcome":
            raise HTTPException(status_code=502, detail=f"FHIR OperationOutcome: {((payload.get('issue') or [{}])[0]).get('diagnostics') or 'error'}")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Unexpected non-JSON-object response from fhir-auth.")
        return payload

    async def _fetch_paginated_everything(self, client: httpx.AsyncClient, base_url: str, dataset_id: str, token: str) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        url, params, collected, seen = f"{base_url}/Patient/{dataset_id}/$everything", {"_count": str(self.settings.everything_page_size)}, [], set()
        max_res, is_partial, reason = max(1, int(self.settings.max_everything_resources)), False, None
        for _ in range(max(1, int(self.settings.max_everything_pages))):
            bundle = await self._fhir_get_json(client, url, token, params=params)
            reached = False
            for entry in bundle.get("entry") or []:
                res = entry.get("resource") if isinstance(entry, Mapping) else None
                if not isinstance(res, dict):
                    continue
                dk = f"{res.get('resourceType')}/{res.get('id')}" if res.get("id") else uuid.uuid4().hex
                if dk not in seen:
                    if len(collected) >= max_res:
                        reached, is_partial, reason = True, True, "max_everything_resources"
                        break
                    seen.add(dk)
                    collected.append(res)
            next_url = next((str(lk.get("url") or "").strip() for lk in (bundle.get("link") or []) if isinstance(lk, Mapping) and lk.get("relation") == "next"), "")
            if reached or (len(collected) >= max_res and next_url):
                return collected, True, reason or "max_everything_resources"
            if not next_url:
                return collected, False, None
            url, params = _rewrite_next_link_to_fhir_auth(base_url, next_url), None
        return collected, bool(next_url), "max_everything_pages" if next_url else reason

    async def assemble(
        self,
        *,
        options: WahsRequestOptions,
        identity: Optional[VerifiedCallerIdentity],
        requested_model: Optional[str] = None,
        conversation_messages: Sequence[Any] = (),
        requested_max_tokens: Optional[int] = None,
        force_refresh: bool = False,
    ) -> AssembledWahsContext:
        if options.overlay_resources:
            rb = len(json.dumps(list(options.overlay_resources), separators=(",", ":"), default=str).encode("utf-8"))
            if rb > self.settings.max_overlay_bytes:
                raise HTTPException(status_code=413, detail=f"wahs.overlay_resources exceeds {self.settings.max_overlay_bytes} bytes.")
        scope, spec = options.chart_scope, self.settings.resolve_model_spec(requested_model)
        # Fail fast if framing + max_tokens alone cannot fit configured context_window
        resolve_chart_budget(
            spec,
            options,
            system_framing_text=WAHS_SYSTEM_PROMPT,
            conversation_messages=conversation_messages,
            requested_max_tokens=requested_max_tokens,
        )
        refresh = bool(options.refresh or force_refresh)
        scope_only_hash = _canonical_sha256({"scope": scope.model_dump(), "overlays": options.overlay_resources})

        if options.fhir_bundle is not None:
            if not self.settings.allow_local_bundle_mode:
                raise HTTPException(status_code=403, detail="Inline wahs.fhir_bundle is disabled when WAHS_ALLOW_LOCAL_BUNDLE_MODE=false.")
            if not isinstance(options.fhir_bundle, dict) or options.fhir_bundle.get("resourceType") != "Bundle":
                raise HTTPException(status_code=400, detail="wahs.fhir_bundle must be a FHIR R4 Bundle.")
            pat_id = str(options.master_patient_id or "").strip() or next((str((e.get("resource") or {}).get("id") or "local-patient") for e in (options.fhir_bundle.get("entry") or []) if isinstance(e, Mapping) and (e.get("resource") or {}).get("resourceType") == "Patient"), "local-patient")
            return await asyncio.to_thread(
                self._scope_and_condense_bundle, raw_bundle=options.fhir_bundle, verified_sub=identity.sub if identity else "local-harness",
                authorized_patient_id=pat_id, allowed_patient_refs={pat_id}, datasets_included=[{"dataset_id": pat_id, "source": "Inline Bundle", "category": "local"}],
                datasets_fingerprint=_canonical_sha256(options.fhir_bundle), scope_only_hash=scope_only_hash, options=options, spec=spec,
                conversation_messages=conversation_messages, requested_max_tokens=requested_max_tokens,
                dataset_cache_hits=0, dataset_fetches=0, cacheable=True, force_refresh=refresh, partial_fetch=False, partial_datasets=[],
            )
        if identity is None:
            raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <keycloak_jwt> header.")

        base, client = self.settings.fhir_auth_base_url.rstrip("/"), self._get_http_client()

        async def _run_live() -> AssembledWahsContext:
            if options.master_patient_id:
                if not _SAFE_ID_RE.match(options.master_patient_id):
                    raise HTTPException(status_code=400, detail="Invalid master_patient_id format.")
                master_pat = await self._fhir_get_json(client, f"{base}/Patient/{options.master_patient_id}", identity.raw_token)
                bound = {str(o.get("value") or "").strip() for o in (master_pat.get("identifier") or []) if isinstance(o, Mapping) and o.get("system") == IDENTIFIER_SYSTEM and o.get("value")} | {str(o.get("code") or "").strip() for o in ((master_pat.get("meta") or {}).get("security") or []) if isinstance(o, Mapping) and o.get("system") == SECURITY_SYSTEM and o.get("code")}
                if bound and identity.sub not in bound:
                    raise HTTPException(status_code=403, detail="MasterPatient ownership mismatch.")
            else:
                sb = await self._fhir_get_json(client, f"{base}/Patient", identity.raw_token, params={"identifier": f"{IDENTIFIER_SYSTEM}|{identity.sub}"})
                if not (sb.get("entry") or []):
                    raise HTTPException(status_code=404, detail="No WAHS Master Patient found for authenticated user.")
                master_pat = ((sb["entry"])[0] or {}).get("resource") or {}
            master_id = str(master_pat.get("id") or "").strip()
            if master_pat.get("resourceType") != "Patient" or not master_id:
                raise HTTPException(status_code=502, detail="Invalid Master Patient resource.")

            linked_ids: List[str] = []
            for link in master_pat.get("link") or []:
                ref = str(((link or {}).get("other") or {}).get("reference") or "").strip()
                if ref:
                    m = _DATASET_REF_RE.match(ref)
                    if not m:
                        raise HTTPException(status_code=403, detail=f"Invalid Patient.link reference '{ref}'.")
                    linked_ids.append(m.group(1))

            raw_descs: List[DatasetDescriptor] = []
            if linked_ids:
                sem = asyncio.Semaphore(max(1, int(self.settings.everything_concurrency)))
                stubs = await asyncio.gather(*(self._stub_with_sem(sem, client, base, did, identity.raw_token) for did in linked_ids))
                for did, dp in zip(linked_ids, stubs):
                    stamp = str((dp.get("meta") or {}).get("lastUpdated") or (dp.get("meta") or {}).get("versionId") or "").strip()
                    cat, meth, src, sn, is_ph = _extract_patient_tags(dp)
                    raw_descs.append(DatasetDescriptor(did, stamp or f"ephemeral-{uuid.uuid4().hex}", cat, meth, src, sn, is_ph, bool(stamp)))

            grouped: Dict[Tuple[str, str, str], List[DatasetDescriptor]] = {}
            for d in raw_descs:
                grouped.setdefault((d.category, d.method, d.source), []).append(d)
            deduped = [sorted(g, key=lambda d: (_parse_fhir_instant(d.last_updated), d.dataset_id), reverse=True)[0] for g in grouped.values()]
            included = [d for d in deduped if _dataset_matches_scope(d, scope)]

            ips_keys, ips_ver, ips_stamp = None, "", True
            if scope.ips_bundle_id:
                if not _SAFE_ID_RE.match(scope.ips_bundle_id):
                    raise HTTPException(status_code=400, detail="Invalid ips_bundle_id format.")
                ib = await self._fhir_get_json(client, f"{base}/Bundle/{scope.ips_bundle_id}", identity.raw_token)
                ips_keys = _extract_ips_referenced_keys(ib)
                rv = str((ib.get("meta") or {}).get("lastUpdated") or (ib.get("meta") or {}).get("versionId") or "").strip()
                ips_stamp, ips_ver = bool(rv), rv or f"ephemeral-{uuid.uuid4().hex}"

            m_ver = str((master_pat.get("meta") or {}).get("lastUpdated") or (master_pat.get("meta") or {}).get("versionId") or "").strip()
            cacheable = bool(m_ver) and ips_stamp and all(d.has_version_stamp for d in included)
            ds_fp = _canonical_sha256({"master": (master_id, m_ver or "ephemeral"), "datasets": [(d.dataset_id, d.last_updated) for d in included], "ips": ips_ver})
            datasets_inc_meta = [{"dataset_id": d.dataset_id, "source": d.source_name, "category": d.category, "method": d.method, "last_updated": d.last_updated} for d in included]

            if cacheable and not refresh:
                ext_cached = self._extracted_cache.get((identity.sub, ds_fp, scope_only_hash))
                if ext_cached is not None:
                    extracted_obj, sec_audit, draft_blk, draft_cits, auth_pat = ext_cached
                    return await asyncio.to_thread(
                        self._condense_from_extracted,
                        extracted=extracted_obj, security_audit=sec_audit, draft_block=draft_blk, draft_cits=draft_cits,
                        verified_sub=identity.sub, authorized_patient_id=auth_pat,
                        datasets_included=datasets_inc_meta, datasets_fingerprint=ds_fp, scope_only_hash=scope_only_hash,
                        options=options, spec=spec, conversation_messages=conversation_messages, requested_max_tokens=requested_max_tokens,
                        dataset_cache_hits=len(included), dataset_fetches=0, cacheable=True, force_refresh=False, partial_fetch=False, partial_datasets=[],
                    )

            pooled, seen_keys, allowed_refs = [{"resource": master_pat}], {f"Patient/{master_id}"}, {master_id, *(d.dataset_id for d in deduped)}
            fsem = asyncio.Semaphore(max(1, int(self.settings.everything_concurrency)))
            ds_results = await asyncio.gather(*(self._fetch_one_ds(fsem, client, base, identity.raw_token, identity.sub, d, allowed_refs, cacheable, refresh) for d in included))
            partial_ds = []
            for desc, ds_res, isp, pr, _, _ in ds_results:
                if isp:
                    partial_ds.append({"dataset_id": desc.dataset_id, "source": desc.source_name, "reason": pr or "truncated", "collected_resources": len(ds_res)})
                for r in ds_res:
                    rt, rid = str(r.get("resourceType") or ""), str(r.get("id") or "")
                    if ips_keys is not None and rt not in _INFRASTRUCTURE_TYPES and f"{rt}/{rid}" not in ips_keys and rid not in ips_keys:
                        continue
                    rk = f"{rt}/{rid}" if rid else uuid.uuid4().hex
                    if rk not in seen_keys:
                        seen_keys.add(rk)
                        pooled.append({"resource": r})

            return await asyncio.to_thread(
                self._scope_and_condense_bundle,
                raw_bundle={"resourceType": "Bundle", "id": f"wahs-pooled-{master_id}", "type": "collection", "entry": pooled},
                verified_sub=identity.sub, authorized_patient_id=master_id, allowed_patient_refs=allowed_refs,
                datasets_included=datasets_inc_meta, datasets_fingerprint=ds_fp, scope_only_hash=scope_only_hash,
                options=options, spec=spec, conversation_messages=conversation_messages, requested_max_tokens=requested_max_tokens,
                dataset_cache_hits=sum(x[4] for x in ds_results), dataset_fetches=sum(x[5] for x in ds_results),
                cacheable=(cacheable and not partial_ds), force_refresh=refresh, partial_fetch=bool(partial_ds), partial_datasets=partial_ds,
            )

        try:
            return await asyncio.wait_for(_run_live(), timeout=max(1.0, float(self.settings.assembly_timeout_seconds)))
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"FHIR chart assembly timed out after {self.settings.assembly_timeout_seconds:.1f}s.") from exc

    async def _stub_with_sem(self, sem: asyncio.Semaphore, client: httpx.AsyncClient, base: str, did: str, token: str) -> Dict[str, Any]:
        async with sem:
            return await self._fhir_get_json(client, f"{base}/Patient/{did}", token)

    async def _fetch_one_ds(self, sem: asyncio.Semaphore, client: httpx.AsyncClient, base: str, token: str, sub: str, desc: DatasetDescriptor, allowed_refs: Set[str], cacheable: bool, refresh: bool) -> Tuple[DatasetDescriptor, List[Dict[str, Any]], bool, Optional[str], int, int]:
        ckey = (sub, desc.dataset_id, desc.last_updated)
        if cacheable and desc.has_version_stamp and not refresh:
            cr = self._dataset_cache.get(ckey)
            if cr is not None:
                return desc, cr, False, None, 1, 0
            if ckey in self._inflight_datasets:
                r, isp, pr = await asyncio.shield(self._inflight_datasets[ckey])
                return desc, r, isp, pr, 1, 0
            async def _bg() -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
                try:
                    async with sem:
                        raw_r, isp, pr = await self._fetch_paginated_everything(client, base, desc.dataset_id, token)
                    st = [_stamp_resource_source_tag(r, desc) for r in raw_r if not (r.get("resourceType") == "Patient" and r.get("id") in allowed_refs)]
                    if not isp and desc.has_version_stamp:
                        self._dataset_cache.put(ckey, st)
                    return st, isp, pr
                finally:
                    self._inflight_datasets.pop(ckey, None)
            t = asyncio.create_task(_bg())
            self._inflight_datasets[ckey] = t
            r, isp, pr = await asyncio.shield(t)
            return desc, r, isp, pr, 0, 1
        async with sem:
            raw_r, isp, pr = await self._fetch_paginated_everything(client, base, desc.dataset_id, token)
        st = [_stamp_resource_source_tag(r, desc) for r in raw_r if not (r.get("resourceType") == "Patient" and r.get("id") in allowed_refs)]
        if not isp and desc.has_version_stamp:
            self._dataset_cache.put(ckey, st)
        return desc, st, isp, pr, 0, 1

    def _scope_and_condense_bundle(
        self,
        *,
        raw_bundle: Dict[str, Any],
        verified_sub: str,
        authorized_patient_id: str,
        allowed_patient_refs: Set[str],
        datasets_included: List[Dict[str, str]],
        datasets_fingerprint: str,
        scope_only_hash: str,
        options: WahsRequestOptions,
        spec: ModelSpec,
        conversation_messages: Sequence[Any],
        requested_max_tokens: Optional[int],
        dataset_cache_hits: int,
        dataset_fetches: int,
        cacheable: bool = True,
        force_refresh: bool = False,
        partial_fetch: bool = False,
        partial_datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> AssembledWahsContext:
        scope = options.chart_scope
        ext_key = (verified_sub, datasets_fingerprint, scope_only_hash)
        ext_cached = self._extracted_cache.get(ext_key) if (cacheable and not force_refresh and not partial_fetch) else None
        if ext_cached is not None:
            extracted, security_audit, draft_block, draft_cits, authorized_patient_id = ext_cached
        else:
            ov_entries, draft_block, draft_cits = _format_overlay_items(options.overlay_resources, authorized_patient_id)
            inc_t, exc_t = {t.strip() for t in scope.include_resource_types if t.strip()}, {t.strip() for t in scope.exclude_resource_types if t.strip()}
            inc_i, exc_i = {i.strip() for i in scope.include_resource_ids if i.strip()}, {i.strip() for i in scope.exclude_resource_ids if i.strip()}
            inc_s, exc_s = {s.strip().lower() for s in scope.include_sources if s.strip()}, {s.strip().lower() for s in scope.exclude_sources if s.strip()}
            inc_c, exc_c = {c.strip().lower() for c in scope.include_categories if c.strip()}, {c.strip().lower() for c in scope.exclude_categories if c.strip()}

            scoped: List[Dict[str, Any]] = []
            for entry in list(raw_bundle.get("entry") or []) + ov_entries:
                res = entry.get("resource") if isinstance(entry, Mapping) else None
                if not isinstance(res, dict):
                    continue
                rt, rid = str(res.get("resourceType") or ""), str(res.get("id") or "")
                fid = f"{rt}/{rid}"
                if rt == "Patient":
                    scoped.append({"resource": res})
                    continue
                if rid in exc_i or fid in exc_i or rt in exc_t or (inc_i and rid not in inc_i and fid not in inc_i and rt not in _INFRASTRUCTURE_TYPES) or (inc_t and rt not in inc_t and rt not in _INFRASTRUCTURE_TYPES):
                    continue
                if (inc_s or exc_s) and rt not in _INFRASTRUCTURE_TYPES:
                    meta = res.get("meta") if isinstance(res.get("meta"), Mapping) else {}
                    rsrc = next((str(t.get("code") or "").strip().lower() for t in (meta.get("tag") or []) if isinstance(t, Mapping) and str(t.get("system") or "").endswith("/wahealthsummary-source-name")), "") or str(meta.get("source") or "").strip().lower() or str((res.get("serviceProvider") or {}).get("display") or "").strip().lower()
                    if (exc_s and rsrc and any(e in rsrc for e in exc_s)) or (inc_s and (not rsrc or not any(i in rsrc for i in inc_s))):
                        continue
                if (inc_c or exc_c) and rt not in _INFRASTRUCTURE_TYPES:
                    rcats = {str(c.get("code")).strip().lower() for co in (res.get("category") if isinstance(res.get("category"), list) else ([res.get("category")] if isinstance(res.get("category"), Mapping) else [])) if isinstance(co, Mapping) for c in (co.get("coding") or []) if isinstance(c, Mapping) and c.get("code")} or {"provider-health-record"}
                    if (exc_c and (rcats & exc_c)) or (inc_c and not (rcats & inc_c)):
                        continue
                if (scope.date_from or scope.date_to) and rt not in _INFRASTRUCTURE_TYPES:
                    rd = _extract_resource_iso_date(res)
                    if rd and ((scope.date_from and rd < scope.date_from) or (scope.date_to and rd > scope.date_to)):
                        continue
                scoped.append({"resource": res})

            guarded, security_audit = enforce_single_patient_scope({"resourceType": "Bundle", "id": raw_bundle.get("id") or f"scoped-{authorized_patient_id}", "type": "collection", "entry": scoped}, authorized_patient_id=authorized_patient_id, allowed_patient_refs=allowed_patient_refs)
            extracted = extract_bundle(guarded)
            if cacheable and not partial_fetch:
                self._extracted_cache.put(ext_key, (extracted, security_audit, draft_block, draft_cits, authorized_patient_id))

        return self._condense_from_extracted(
            extracted=extracted, security_audit=security_audit, draft_block=draft_block, draft_cits=draft_cits,
            verified_sub=verified_sub, authorized_patient_id=authorized_patient_id,
            datasets_included=datasets_included, datasets_fingerprint=datasets_fingerprint, scope_only_hash=scope_only_hash,
            options=options, spec=spec, conversation_messages=conversation_messages, requested_max_tokens=requested_max_tokens,
            dataset_cache_hits=dataset_cache_hits, dataset_fetches=dataset_fetches,
            cacheable=cacheable, force_refresh=force_refresh, partial_fetch=partial_fetch, partial_datasets=partial_datasets,
        )

    def _condense_from_extracted(
        self,
        *,
        extracted: ExtractedChart,
        security_audit: Dict[str, Any],
        draft_block: str,
        draft_cits: Dict[str, Dict[str, Any]],
        verified_sub: str,
        authorized_patient_id: str,
        datasets_included: List[Dict[str, str]],
        datasets_fingerprint: str,
        scope_only_hash: str,
        options: WahsRequestOptions,
        spec: ModelSpec,
        conversation_messages: Sequence[Any],
        requested_max_tokens: Optional[int],
        dataset_cache_hits: int,
        dataset_fetches: int,
        cacheable: bool,
        force_refresh: bool,
        partial_fetch: bool,
        partial_datasets: Optional[List[Dict[str, Any]]],
    ) -> AssembledWahsContext:
        counter = get_exact_token_counter(spec.tokenizer_family)
        partial_list = list(partial_datasets or [])
        partial_warn = "[PARTIAL RECORD WARNING: Dataset pagination/resource ceiling reached before full retrieval; record may be incomplete.]" if partial_fetch else ""
        framing_without_chart = (
            f"{WAHS_SYSTEM_PROMPT}\n\n=== ACTIVE PATIENT: {extracted.patient_name} (DOB: {extracted.birth_date or 'unknown'}) ===\n"
            "=== CONDENSED CLINICAL RECORD ([1]..[N]) ===\n"
            + (f"\n\n{draft_block}" if draft_block else "")
            + (f"\n{partial_warn}" if partial_fetch else "")
        )
        budget_effective, clamp_reason = resolve_chart_budget(
            spec,
            options,
            system_framing_text=framing_without_chart,
            conversation_messages=conversation_messages,
            requested_max_tokens=requested_max_tokens,
        )
        t0_key = (verified_sub, datasets_fingerprint, scope_only_hash, spec.tokenizer_family, "tier0")
        b_key = (verified_sub, datasets_fingerprint, scope_only_hash, spec.tokenizer_family, str(budget_effective))
        if cacheable and not force_refresh and not partial_fetch:
            cached_t0 = self._prefix_cache.get(t0_key)
            if cached_t0 is not None and cached_t0.condensed.actual_tokens <= budget_effective:
                return _clone_context(cached_t0, cache_hit=True, budget_requested=options.target_budget_tokens, budget_effective=budget_effective, clamp_reason=clamp_reason, dataset_cache_hits=len(datasets_included), dataset_fetches=0)
            cached_exact = self._prefix_cache.get(b_key)
            if cached_exact is not None:
                return _clone_context(cached_exact, cache_hit=True, budget_requested=options.target_budget_tokens, budget_effective=budget_effective, clamp_reason=clamp_reason, dataset_cache_hits=len(datasets_included), dataset_fetches=0)

        chart_target = max(1, budget_effective - ((counter(partial_warn) + 4) if partial_fetch else 0))
        try:
            condensed = condense_bundle_to_budget(extracted, target_tokens=chart_target, token_counter=counter)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def _render_sys(ct_text: str) -> str:
            return f"{WAHS_SYSTEM_PROMPT}\n\n=== ACTIVE PATIENT: {extracted.patient_name} (DOB: {extracted.birth_date or 'unknown'}) ===\n=== CONDENSED CLINICAL RECORD ([1]..[N]) ===\n{ct_text}" + (f"\n\n{draft_block}" if draft_block else "")

        upd_text = f"{condensed.chart_text.rstrip()}\n{partial_warn}" if partial_fetch else condensed.chart_text
        sys_prompt = _render_sys(upd_text)

        eff_max_toks = int(requested_max_tokens) if requested_max_tokens is not None else int(spec.default_max_output_tokens)
        trimmed_turns = _trimmed_turns_for_window(
            normalize_conversation_turns(conversation_messages),
            system_text=framing_without_chart,
            spec=spec,
            effective_max_tokens=eff_max_toks,
        )
        total_with_turns = count_chat_messages_tokens([{"role": "system", "content": sys_prompt}, *trimmed_turns], tokenizer_family=spec.tokenizer_family)
        limit_with_turns = spec.context_window - eff_max_toks
        if total_with_turns > limit_with_turns and chart_target > 1:
            over = total_with_turns - limit_with_turns
            tighter_target = max(1, chart_target - over - 4)
            try:
                condensed = condense_bundle_to_budget(extracted, target_tokens=tighter_target, token_counter=counter)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            upd_text = f"{condensed.chart_text.rstrip()}\n{partial_warn}" if partial_fetch else condensed.chart_text
            sys_prompt = _render_sys(upd_text)

        manifest = {**condensed.completeness_manifest, "partial_fetch": partial_fetch, **({"partial_datasets": partial_list} if partial_fetch else {})}
        if partial_fetch:
            upd_toks = counter(upd_text)
            condensed = replace(condensed, target_tokens=budget_effective, actual_tokens=upd_toks, chart_text=upd_text, completeness_manifest=manifest, compression_stats={**condensed.compression_stats, "actual_tokens": upd_toks, "target_tokens": budget_effective, "budget_utilization": round(upd_toks / budget_effective, 3)})
        else:
            condensed = replace(condensed, completeness_manifest=manifest)

        src_by_uuid = {it.resource_uuid: (it.source_org or "Patient Record", it.section or "") for it in extracted.items}
        citation_map: Dict[str, Dict[str, Any]] = {
            str(m["index"]): {"index": m.get("index"), "resourceType": m.get("resourceType"), "id": str(m.get("resourceUuid") or ""), "source": src_by_uuid.get(str(m.get("resourceUuid") or ""), ("WA Health Summary", ""))[0], "section": src_by_uuid.get(str(m.get("resourceUuid") or ""), ("WA Health Summary", ""))[1], "date": m.get("date") or "", "display": m.get("text") or "", **({"memberIds": list(m["memberUuids"])} if m.get("memberUuids") else {})}
            for m in (condensed.mappings or []) if isinstance(m, Mapping) and str(m.get("index") or "")
        }
        citation_map.update(draft_cits)
        notices = [f"Partial record warning: FHIR fetch reached configured limit on {', '.join(str(p.get('source') or p.get('dataset_id') or '') for p in partial_list) or 'one or more datasets'}; completeness cannot be guaranteed."] if partial_fetch else []

        s_hash = _canonical_sha256({"scope": scope_only_hash, "budget": budget_effective, "tokenizer_family": spec.tokenizer_family})
        prefix_hash = hashlib.sha256(f"{verified_sub}:{sys_prompt}".encode("utf-8")).hexdigest()[:20]
        ctx = AssembledWahsContext(
            verified_sub, authorized_patient_id, extracted.patient_name, extracted.birth_date or "",
            sys_prompt, prefix_hash, datasets_fingerprint, s_hash, datasets_included,
            options.chart_scope.model_dump(exclude_defaults=False), security_audit, extracted, condensed, citation_map,
            notices, False, dataset_cache_hits, dataset_fetches, options.target_budget_tokens, budget_effective,
            spec.context_window, spec.preferred_chart_tokens, clamp_reason, spec.tokenizer_family, partial_fetch, partial_list,
        )
        if cacheable and not partial_fetch:
            self._prefix_cache.put(b_key, _clone_context(ctx, cache_hit=False))
            if condensed.tier == 0:
                self._prefix_cache.put(t0_key, _clone_context(ctx, cache_hit=False))
        return _clone_context(ctx, cache_hit=False)
