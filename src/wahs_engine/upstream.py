"""Southbound OpenAI `/v1` inference client, Tier-0 overflow recovery, and SSE streamer."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from fastapi import HTTPException
import httpx

from wahs_core import (
    build_temporal_digest,
    build_wahs_hub_enrichment_block,
    extract_clean_user_query,
    get_lab_history,
    read_clinical_note,
    retrieve_clinical_note_passages,
    search_chart_entries,
)

from .assembler import AssembledWahsContext
from .config import ModelSpec, WahsEngineSettings
from .schemas import OpenAIChatCompletionRequest, WahsRequestOptions
from .tokenizers_registry import count_chat_messages_tokens, get_exact_token_counter

logger = logging.getLogger(__name__)

_OVERFLOW_TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_chart_entries",
            "description": "Search full uncondensed Tier-0 patient chart entries by clinical keyword or code when the Completeness Manifest shows omitted_clinical_items > 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "section": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_history",
            "description": "Retrieve full longitudinal numeric lab/vital measurements for an analyte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analyte": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["analyte"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clinical_note",
            "description": "Read the deduplicated text of a clinical note (DocumentReference) by UUID or keyword, with optional offset/max_chars pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id_or_query": {"type": "string"},
                    "offset": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["note_id_or_query"],
            },
        },
    },
]


PRE_TOOL_LOOKAHEAD_CHARS: int = 96


class _StreamingThinkStripper:
    """Stateful filter stripping `<think>...</think>` and `<|channel>thought...<channel|>` across SSE chunks."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_tag: Optional[str] = None

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        self._buffer += piece
        out: List[str] = []
        while self._buffer:
            if self._in_tag is not None:
                end_idx = self._buffer.find(self._in_tag)
                if end_idx < 0:
                    # Keep tail in case closing tag is split across chunk boundary
                    keep = len(self._in_tag) - 1
                    if len(self._buffer) > keep:
                        self._buffer = self._buffer[-keep:]
                    break
                self._buffer = self._buffer[end_idx + len(self._in_tag) :].lstrip("\n\r ")
                self._in_tag = None
                continue

            idx_think = self._buffer.find("<think>")
            idx_chan = self._buffer.find("<|channel>thought")
            starts = [(i, close) for i, close in ((idx_think, "</think>"), (idx_chan, "<channel|>")) if i >= 0]
            if not starts:
                # Hold back partial opening prefix (`<` at end of chunk)
                partial_hold = 0
                for opener in ("<think>", "<|channel>thought"):
                    for k in range(min(len(opener) - 1, len(self._buffer)), 0, -1):
                        if self._buffer.endswith(opener[:k]):
                            partial_hold = max(partial_hold, k)
                            break
                if partial_hold:
                    emit = self._buffer[:-partial_hold]
                    self._buffer = self._buffer[-partial_hold:]
                    if emit:
                        out.append(emit)
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break

            start_pos, close_marker = min(starts, key=lambda x: x[0])
            if start_pos > 0:
                out.append(self._buffer[:start_pos])
            opener_len = len("<think>") if close_marker == "</think>" else len("<|channel>thought")
            self._buffer = self._buffer[start_pos + opener_len :]
            self._in_tag = close_marker

        return "".join(out)

    def flush(self) -> str:
        if self._in_tag is not None:
            self._buffer = ""
            self._in_tag = None
            return ""
        rem = self._buffer
        self._buffer = ""
        return rem


def clean_model_text(content: str) -> str:
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        if "<think>" in content:
            content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    if "<|channel>thought" in content:
        content = re.sub(r"<\|channel>thought.*?<channel\|>\s*", "", content, flags=re.DOTALL)
        if "<|channel>thought" in content:
            content = re.sub(r"<\|channel>thought.*$", "", content, flags=re.DOTALL)
    return content.strip()


def _register_or_get_citation_index(
    ctx: AssembledWahsContext,
    *,
    resource_uuid: str,
    resource_type: str = "Resource",
    display: str = "",
    date: str = "",
    source: str = "",
    section: str = "",
    member_uuids: Optional[Sequence[str]] = None,
) -> Optional[int]:
    ruuid = str(resource_uuid or "").replace("urn:uuid:", "").strip()
    members = [
        str(m).replace("urn:uuid:", "").strip()
        for m in (member_uuids or [])
        if str(m or "").strip()
    ]
    if not ruuid and members:
        ruuid = members[0]
    if not ruuid:
        return None

    candidate_ids = {ruuid, *members}
    for k_str, entry in ctx.citation_map.items():
        if not str(k_str).isdigit() or not isinstance(entry, Mapping):
            continue
        existing_id = str(entry.get("id") or "").replace("urn:uuid:", "").strip()
        existing_members = {
            str(m).replace("urn:uuid:", "").strip()
            for m in (entry.get("memberIds") or [])
            if str(m or "").strip()
        }
        if existing_id in candidate_ids or (existing_members & candidate_ids):
            return int(k_str)

    numeric_keys = [int(k) for k in ctx.citation_map if str(k).isdigit()]
    next_idx = (max(numeric_keys) if numeric_keys else 0) + 1
    entry_obj: Dict[str, Any] = {
        "index": next_idx,
        "resourceType": resource_type or "Resource",
        "id": ruuid,
        "source": source or "WA Health Summary",
        "section": section or "",
        "date": date or "",
        "display": display or ruuid,
    }
    if members:
        entry_obj["memberIds"] = members
    ctx.citation_map[str(next_idx)] = entry_obj
    return next_idx


def _omitted_manifest_stats(ctx: AssembledWahsContext) -> Tuple[int, Dict[str, int]]:
    manifest = ctx.condensed.completeness_manifest or {}
    omitted_total = int(
        manifest.get("omitted_clinical_items_count")
        or manifest.get("omitted_clinical_items")
        or 0
    )
    raw_by_sec = manifest.get("omitted_by_section")
    omitted_by_cat = (
        {str(k): int(v) for k, v in raw_by_sec.items() if int(v or 0) > 0}
        if isinstance(raw_by_sec, Mapping)
        else {}
    )
    return omitted_total, omitted_by_cat


def _should_allow_overflow_tools(
    ctx: AssembledWahsContext,
    options: WahsRequestOptions,
) -> Tuple[bool, int, Dict[str, int]]:
    omitted_total, omitted_by_cat = _omitted_manifest_stats(ctx)
    allow = options.retrieval_strategy == "dynamic_fhir_tools" or (
        options.enable_overflow_tools
        and omitted_total > 0
        and options.retrieval_strategy != "chart_only"
    )
    return allow, omitted_total, omitted_by_cat


def _uuid_to_citation_from_map(citation_map: Mapping[str, Any]) -> Dict[str, int]:
    """Extract UUID -> [N] mapping (including memberIds) from assembled `citation_map`."""
    out: Dict[str, int] = {}
    for k_str, entry in citation_map.items():
        if not isinstance(entry, Mapping):
            continue
        idx_raw = entry.get("index")
        c_idx = (
            int(idx_raw)
            if isinstance(idx_raw, int)
            else (int(k_str) if str(k_str).isdigit() else 0)
        )
        if c_idx <= 0:
            continue
        eid = str(entry.get("id") or "").strip()
        if eid:
            out.setdefault(eid, c_idx)
        for m_id in entry.get("memberIds") or ():
            ms = str(m_id or "").strip()
            if ms:
                out.setdefault(ms, c_idx)
    return out


def build_x_wahs_envelope(
    *,
    ctx: AssembledWahsContext,
    requested_model: str,
    upstream_model: str,
    options: WahsRequestOptions,
    engine_version: str,
    overflow_recovered: int = 0,
    kb_meta: Optional[Dict[str, Any]] = None,
    note_rag_meta: Optional[Dict[str, Any]] = None,
    temporal_digest_meta: Optional[Dict[str, Any]] = None,
    tool_calls_log: Optional[List[Dict[str, Any]]] = None,
    timings: Optional[Dict[str, Any]] = None,
    exact_prompt_tokens: int = 0,
    upstream_prompt_tokens: int = 0,
    suspected_upstream_truncation: bool = False,
    history_trimmed_turns: int = 0,
) -> Dict[str, Any]:
    omitted_total, omitted_by_cat = _omitted_manifest_stats(ctx)
    effective_kb = bool(options.enable_kb) and options.retrieval_strategy != "chart_only"
    budget_effective = ctx.budget_effective or ctx.condensed.target_tokens
    env: Dict[str, Any] = {
        "retrieval_strategy": options.retrieval_strategy,
        "citation_map": ctx.citation_map,
        "scope_applied": {
            **ctx.scope_applied,
            "datasets_included": ctx.datasets_included,
            "sharing_posture": options.sharing_posture,
        },
        "condensation": {
            "tier": ctx.condensed.tier,
            "tier_name": ctx.condensed.tier_name,
            "budget_tokens": ctx.condensed.target_tokens,
            "budget_effective": budget_effective,
            "budget_requested": ctx.budget_requested,
            "requested_budget_tokens": options.target_budget_tokens,
            "context_window": ctx.context_window,
            "chart_budget_ceiling": ctx.chart_budget_ceiling,
            "clamp_reason": ctx.clamp_reason,
            "tokenizer_family": ctx.tokenizer_family,
            "actual_tokens": ctx.condensed.actual_tokens,
            "omitted_clinical_items": omitted_total,
            "omitted_by_category": omitted_by_cat,
            "overflow_recovered": overflow_recovered,
            "datasets_included": len(ctx.datasets_included),
            "prefix_cache_key_hash": ctx.prefix_cache_key_hash,
            "cache_hit": ctx.cache_hit,
            "dataset_cache_hits": ctx.dataset_cache_hits,
            "dataset_fetches": ctx.dataset_fetches,
            "partial_fetch": ctx.partial_fetch,
            "partial_datasets": ctx.partial_datasets,
            "exact_prompt_tokens": exact_prompt_tokens,
            "upstream_prompt_tokens": upstream_prompt_tokens,
            "suspected_upstream_truncation": suspected_upstream_truncation,
            "history_trimmed_turns": history_trimmed_turns,
        },
        "completeness_manifest": ctx.condensed.completeness_manifest,
        "privacy_notices": ctx.privacy_notices,
        "ai_provenance": {
            "resourceType": "Provenance",
            "profile": "http://hl7.org/fhir/uv/aitransparency/StructureDefinition/AI-Provenance",
            "recorded_model": upstream_model,
            "requested_model": requested_model,
            "engine": f"wahs-engine/{engine_version}",
            "retrieval_strategy": options.retrieval_strategy,
            "kb_enabled": effective_kb,
            "kb_enrichment": kb_meta,
            "human_review_required": True,
        },
        "temporal_digest": (
            dict(temporal_digest_meta)
            if temporal_digest_meta is not None
            else {"enabled": False}
        ),
    }
    if note_rag_meta is not None:
        env["note_rag"] = note_rag_meta
    if tool_calls_log is not None:
        env["tool_calls"] = tool_calls_log
    if timings is not None:
        env["timings"] = timings
    return env


def _annotate_tool_results_with_citations(
    ctx: AssembledWahsContext,
    res: Dict[str, Any],
    *,
    default_resource_type: str = "Observation",
) -> None:
    results = res.get("results")
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, dict):
            continue
        c_idx = _register_or_get_citation_index(
            ctx,
            resource_uuid=str(item.get("resource_uuid") or ""),
            resource_type=str(item.get("resource_type") or default_resource_type),
            display=str(item.get("display") or item.get("concept") or item.get("text") or ""),
            date=str(item.get("date") or ""),
            source=str(item.get("source_org") or "WA Health Summary"),
            section=str(item.get("section") or item.get("category") or ""),
            member_uuids=item.get("member_uuids"),
        )
        if c_idx is not None:
            item["citation_index"] = c_idx
            item["citation_tag"] = f"[{c_idx}]"
    summary = res.get("summary")
    if isinstance(summary, dict):
        for endpoint_key in ("initial", "min", "max", "latest"):
            pt = summary.get(endpoint_key)
            if isinstance(pt, dict) and pt.get("resource_uuid"):
                c_idx = _register_or_get_citation_index(
                    ctx,
                    resource_uuid=str(pt["resource_uuid"]),
                    resource_type="Observation",
                    display=f"{res.get('analyte_or_loinc') or 'Lab'} {pt.get('value')} {pt.get('unit') or ''}".strip(),
                    date=str(pt.get("date") or ""),
                )
                if c_idx is not None:
                    pt["citation_tag"] = f"[{c_idx}]"


def _safe_int(
    raw: Any,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    val = default
    if raw is not None and not isinstance(raw, bool):
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = default
    if minimum is not None:
        val = max(minimum, val)
    if maximum is not None:
        val = min(maximum, val)
    return val


def _execute_local_overflow_tool(
    ctx: AssembledWahsContext,
    fn_name: str,
    fn_args: Mapping[str, Any],
    sharing_posture: str = "patient_self",
) -> Tuple[str, int]:
    safe_args: Mapping[str, Any] = fn_args if isinstance(fn_args, Mapping) else {}
    if fn_name == "search_chart_entries":
        res = search_chart_entries(
            ctx.extracted,
            query=str(safe_args.get("query") or ""),
            section=str(safe_args["section"]) if safe_args.get("section") is not None else None,
            sharing_posture=sharing_posture,
            limit=_safe_int(safe_args.get("limit"), 10, minimum=1, maximum=100),
        )
        _annotate_tool_results_with_citations(ctx, res, default_resource_type="Resource")
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "get_lab_history":
        res = get_lab_history(
            ctx.extracted,
            str(safe_args.get("analyte") or ""),
            limit=_safe_int(safe_args.get("limit"), 20, minimum=1, maximum=200),
            sharing_posture=sharing_posture,
        )
        _annotate_tool_results_with_citations(ctx, res, default_resource_type="Observation")
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "read_clinical_note":
        key = str(safe_args.get("note_id_or_query") or "")
        offset = _safe_int(safe_args.get("offset"), 0, minimum=0)
        res = read_clinical_note(
            ctx.extracted,
            resource_uuid=key,
            sharing_posture=sharing_posture,
        )
        matched_by_uuid = bool(res.get("results"))
        if not matched_by_uuid and key:
            res = read_clinical_note(
                ctx.extracted,
                query=key,
                sharing_posture=sharing_posture,
            )
        results_list = [r for r in (res.get("results") or []) if isinstance(r, dict)]
        raw_max_chars = safe_args.get("max_chars")
        if raw_max_chars is not None:
            default_mc = 12000 if (matched_by_uuid and len(results_list) == 1) else 3200
            max_chars = _safe_int(raw_max_chars, default_mc, minimum=256, maximum=24000)
        else:
            max_chars = 12000 if (matched_by_uuid and len(results_list) == 1) else 3200
        # Drop duplicate full_text field and paginate via offset/max_chars when needed
        for note_item in results_list:
            note_item.pop("full_text", None)
            txt = str(note_item.get("text") or "")
            total_len = len(txt)
            note_item["offset"] = min(offset, total_len)
            note_item["total_chars"] = total_len
            if offset >= total_len and total_len > 0:
                note_item["text"] = ""
                note_item["has_more"] = False
            elif total_len - offset > max_chars:
                end_pos = offset + max_chars
                note_item["text"] = (
                    txt[offset:end_pos]
                    + f" ...[truncated at {end_pos}/{total_len} chars; call read_clinical_note with offset={end_pos} to continue]"
                )
                note_item["has_more"] = True
                note_item["next_offset"] = end_pos
            else:
                note_item["text"] = txt[offset:]
                note_item["has_more"] = False
        _annotate_tool_results_with_citations(ctx, res, default_resource_type="DocumentReference")
        return json.dumps(res, default=str), len(results_list)
    return json.dumps({"error": f"Unknown tool '{fn_name}'"}), 0


def _apply_provider_southbound_params(payload: Dict[str, Any], spec: ModelSpec) -> None:
    """Explicitly configure southbound context window & thinking controls by provider flavor."""
    flavor = (spec.provider_flavor or "litellm").strip().lower()
    if flavor in {"litellm", "ollama"}:
        payload["num_ctx"] = int(spec.context_window)
        if spec.disable_thinking:
            payload["reasoning_effort"] = "none"
    elif flavor == "llamacpp":
        if spec.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    # OpenAI ("openai") and unknown flavors: do not inject Ollama/llama.cpp extensions


def _trim_history_to_fit(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[Sequence[Mapping[str, Any]]],
    tokenizer_family: str,
    context_window: int,
    max_output_tokens: int,
) -> Tuple[int, int]:
    """Trim oldest conversation turns (preserving `messages[0]` and the latest user turn + tail).

    Returns ``(exact_prompt_tokens, history_trimmed_turns)``. Raises HTTP 422 if the prompt still
    exceeds ``context_window - max_output_tokens`` after trimming all prior turns.
    """
    prompt_limit = int(context_window) - int(max_output_tokens)
    trimmed = 0
    while True:
        exact_toks = count_chat_messages_tokens(
            messages, tools=tools, tokenizer_family=tokenizer_family
        )
        if exact_toks <= prompt_limit:
            return exact_toks, trimmed
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=1,
        )
        if last_user_idx <= 1:
            break
        messages.pop(1)
        trimmed += 1
        last_user_idx -= 1
        while last_user_idx > 1 and messages[1].get("role") != "user":
            messages.pop(1)
            trimmed += 1
            last_user_idx -= 1

    exact_toks = count_chat_messages_tokens(
        messages, tools=tools, tokenizer_family=tokenizer_family
    )
    if exact_toks > prompt_limit:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Assembled prompt ({exact_toks} exact {tokenizer_family} tokens) + max_tokens "
                f"({max_output_tokens}) exceeds model context_window ({context_window})."
            ),
        )
    return exact_toks, trimmed


def _compute_suspected_truncation(
    *,
    llm_rounds: int,
    upstream_prompt_tokens: int,
    exact_prompt_tokens: int,
) -> bool:
    return bool(
        llm_rounds == 1
        and upstream_prompt_tokens > 0
        and exact_prompt_tokens >= 1024
        and upstream_prompt_tokens < int(exact_prompt_tokens * 0.85)
    )


def prepare_upstream_messages(
    *,
    ctx: AssembledWahsContext,
    req: OpenAIChatCompletionRequest,
    options: WahsRequestOptions,
    spec: Optional[ModelSpec] = None,
    effective_max_tokens: int = 1024,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    int,
    Optional[Dict[str, Any]],
    Dict[str, Any],
    int,
    int,
]:
    tok_family = spec.tokenizer_family if spec is not None else ctx.tokenizer_family
    ctx_window = spec.context_window if spec is not None else ctx.context_window
    exact_counter = get_exact_token_counter(tok_family)

    turns: List[Dict[str, Any]] = []
    for m in req.messages:
        role = (m.role or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = m.content
        if isinstance(content, list):
            parts = [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, Mapping) and p.get("type") in {"text", "input_text", None}
            ]
            text_val = "\n".join(p for p in parts if p).strip()
        else:
            text_val = str(content or "").strip()
        if text_val:
            turns.append({"role": role, "content": text_val})

    if not turns or not any(t["role"] == "user" for t in turns):
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one non-empty user message.",
        )

    raw_last_user_query = next(
        (t["content"] for t in reversed(turns) if t["role"] == "user"), ""
    )
    clean_last_user_query = (
        extract_clean_user_query(raw_last_user_query) or raw_last_user_query
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ctx.system_prompt},
        *turns,
    ]

    kb_meta: Optional[Dict[str, Any]] = None
    note_rag_meta: Optional[Dict[str, Any]] = None
    temporal_meta: Dict[str, Any] = {"enabled": False}
    overflow_recovered = 0
    effective_kb = bool(options.enable_kb) and options.retrieval_strategy != "chart_only"
    effective_temporal_digest = (
        bool(options.enable_temporal_digest) and options.retrieval_strategy != "chart_only"
    )

    minimal_turn_toks = count_chat_messages_tokens(
        [messages[0], messages[-1]], tools=tools, tokenizer_family=tok_family
    )
    true_window_slack = max(
        0, ctx_window - effective_max_tokens - minimal_turn_toks - 64
    )

    tool_hint = ""
    if options.retrieval_strategy == "dynamic_fhir_tools":
        tool_hint = (
            "=== DYNAMIC FHIR CHART TOOLS ACTIVE ===\n"
            "The Condensed Clinical Record above is a compact IPS safety header. "
            "Whenever the question asks about prior hospital admissions, clinical notes, procedures, "
            "microbiology/pathology reports, or specific longitudinal lab dates not shown in full above, "
            "call `search_chart_entries`, `read_clinical_note`, or `get_lab_history` first, then synthesize "
            "your final patient-friendly answer citing each item's `[N]` (`citation_tag`) inline."
        )
    tool_hint_toks = exact_counter(tool_hint) if tool_hint else 0

    rag_block = ""
    if options.retrieval_strategy == "note_rag":
        try:
            rag_header = (
                "=== SINGLE-PATIENT CLINICAL NOTE & REPORT PASSAGES (BM25 RAG — cite inline as [N]) ===\n"
                "Note Precedence: Reconcile retrieved passages by date—newer Discharge Summaries, Addenda, or hold/stop instructions supersede older admission or baseline notes. Answer strictly what the user asked without listing unrelated conditions.\n"
            )
            rag_avail = max(0, true_window_slack - tool_hint_toks)
            rag_res = retrieve_clinical_note_passages(
                ctx.extracted,
                clean_last_user_query,
                sharing_posture=options.sharing_posture,
                top_k=5,
            )
            passages = rag_res.get("results") or []
            rag_lines: List[str] = []
            rag_indices: List[int] = []
            for p in passages:
                if not isinstance(p, Mapping) or rag_avail < 48:
                    continue
                c_idx = _register_or_get_citation_index(
                    ctx,
                    resource_uuid=str(p.get("resource_uuid") or ""),
                    resource_type=str(p.get("resource_type") or "DocumentReference"),
                    display=str(p.get("display") or "Clinical Note"),
                    date=str(p.get("date") or ""),
                    source=str(p.get("source_org") or "Clinical Note"),
                    section=str(p.get("section") or "clinical_note"),
                    member_uuids=p.get("member_uuids"),
                )
                cite_prefix = f"[{c_idx}] " if c_idx is not None else ""
                line_prefix = (
                    f"{cite_prefix}({p.get('date') or 'undated'} | {p.get('display')} | "
                    f"source={p.get('source_org') or 'Chart'} | bm25={p.get('bm25_score')}): "
                )
                passage_txt = str(p.get("passage_text") or "")
                cand_line = f"{line_prefix}{passage_txt}"
                cand_block = rag_header + "\n".join([*rag_lines, cand_line])
                if exact_counter(cand_block) <= rag_avail:
                    rag_lines.append(cand_line)
                    if c_idx is not None and c_idx not in rag_indices:
                        rag_indices.append(c_idx)
                elif not rag_lines and passage_txt:
                    lo, hi = 0, len(passage_txt)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        trunc_txt = passage_txt[:mid].rstrip() + ("..." if mid < len(passage_txt) else "")
                        if exact_counter(rag_header + line_prefix + trunc_txt) <= rag_avail:
                            lo = mid
                        else:
                            hi = mid - 1
                    if lo >= 24:
                        trunc_txt = passage_txt[:lo].rstrip() + ("..." if lo < len(passage_txt) else "")
                        rag_lines.append(f"{line_prefix}{trunc_txt}")
                        if c_idx is not None and c_idx not in rag_indices:
                            rag_indices.append(c_idx)
                    break
                else:
                    break
            note_rag_meta = {
                "total_chunks_indexed": int(rag_res.get("total_chunks_indexed") or 0),
                "passages_retrieved": len(rag_lines),
                "cited_indices": rag_indices,
                "redacted_for_privacy_count": int(rag_res.get("redacted_for_privacy_count") or 0),
            }
            if rag_lines:
                rag_block = rag_header + "\n".join(rag_lines)
        except Exception as exc:
            logger.warning("WAHS Note RAG skipped: %s", exc)

    rag_toks_used = exact_counter(rag_block) if rag_block else 0
    remaining_after_rag = max(0, true_window_slack - tool_hint_toks - rag_toks_used)
    total_headroom = min(
        remaining_after_rag, max(1280, int(ctx.condensed.target_tokens * 0.25))
    )

    kb_enrichment_text = ""
    if effective_kb and total_headroom >= 64:
        try:
            kb_reserve_for_digest = (
                min(max(128, total_headroom // 3), 512)
                if (effective_temporal_digest and total_headroom >= 256)
                else 0
            )
            kb_budget = max(64, total_headroom - kb_reserve_for_digest)
            enrichment = build_wahs_hub_enrichment_block(
                messages,
                extracted=ctx.extracted,
                condensed=ctx.condensed,
                citation_map=ctx.citation_map,
                max_tokens=kb_budget,
                sharing_posture=options.sharing_posture,
                token_counter=exact_counter,
                max_overflow_recover=(
                    0 if options.retrieval_strategy == "dynamic_fhir_tools" else 5
                ),
            )
            kb_meta = enrichment.to_metadata_dict()
            overflow_recovered = int(
                kb_meta.get("overflow_recovered_count")
                or kb_meta.get("overflow_items_recovered")
                or 0
            )
            for rm in kb_meta.get("recovered_mappings") or []:
                if isinstance(rm, Mapping) and rm.get("resourceUuid"):
                    idx_rm = rm.get("index")
                    if isinstance(idx_rm, int) and str(idx_rm) not in ctx.citation_map:
                        ctx.citation_map[str(idx_rm)] = {
                            "index": idx_rm,
                            "resourceType": rm.get("resourceType") or "Resource",
                            "id": str(rm.get("resourceUuid")),
                            "source": "Tier-0 Overflow Recovery",
                            "section": "",
                            "date": str(rm.get("date") or ""),
                            "display": str(rm.get("display") or ""),
                            **(
                                {"memberIds": list(rm["memberUuids"])}
                                if rm.get("memberUuids")
                                else {}
                            ),
                        }
            if enrichment.enrichment_text:
                kb_enrichment_text = enrichment.enrichment_text
        except Exception as exc:
            logger.warning("WAHS KB enrichment skipped: %s", exc)

    temporal_digest_block = ""
    if effective_temporal_digest:
        try:
            kb_toks_used = exact_counter(kb_enrichment_text) if kb_enrichment_text else 0
            digest_budget = max(0, total_headroom - kb_toks_used)
            if digest_budget < 64:
                temporal_meta = {
                    "enabled": False,
                    "skipped_reason": "insufficient_context_headroom",
                }
            else:
                extra_cits = _uuid_to_citation_from_map(ctx.citation_map)
                temporal_res = build_temporal_digest(
                    ctx.extracted,
                    ctx.condensed,
                    question=clean_last_user_query,
                    sharing_posture=options.sharing_posture,
                    extra_uuid_to_citation=extra_cits,
                    max_tokens=digest_budget,
                    token_counter=exact_counter,
                )
                temporal_meta = dict(temporal_res.summary)
                temporal_digest_block = temporal_res.digest_text
        except Exception as exc:
            logger.warning("WAHS Temporal Digest skipped: %s", exc)
            temporal_meta = {"enabled": False, "error": str(exc)}

    max_prompt_budget = max(0, ctx_window - effective_max_tokens)
    orig_last_content = str(messages[-1]["content"] or "")
    candidate_blocks = [
        ("temporal_digest", temporal_digest_block),
        ("kb", kb_enrichment_text),
        ("rag", rag_block),
        ("tool_hint", tool_hint),
    ]
    while any(text for _, text in candidate_blocks):
        active_texts = [text for _, text in candidate_blocks if text]
        cand_last = f"{orig_last_content.rstrip()}\n\n" + "\n\n".join(active_texts)
        cand_toks = count_chat_messages_tokens(
            [messages[0], {"role": messages[-1]["role"], "content": cand_last}],
            tools=tools,
            tokenizer_family=tok_family,
        )
        if cand_toks <= max_prompt_budget:
            messages[-1] = {"role": messages[-1]["role"], "content": cand_last}
            break
        # Drop lowest-priority optional enrichment block first so the base prompt stays within context_window.
        dropped = False
        for idx_drop, (kind, text) in enumerate(candidate_blocks):
            if text:
                candidate_blocks[idx_drop] = (kind, "")
                if kind == "temporal_digest" and temporal_meta.get("enabled"):
                    temporal_meta = {
                        "enabled": False,
                        "skipped_reason": "insufficient_context_headroom",
                    }
                dropped = True
                break
        if not dropped:
            break

    exact_prompt_tokens, history_trimmed_turns = _trim_history_to_fit(
        messages,
        tools=tools,
        tokenizer_family=tok_family,
        context_window=ctx_window,
        max_output_tokens=effective_max_tokens,
    )
    return (
        messages,
        kb_meta,
        overflow_recovered,
        note_rag_meta,
        temporal_meta,
        exact_prompt_tokens,
        history_trimmed_turns,
    )


def _build_southbound_payload(
    *,
    upstream_model: str,
    messages: Sequence[Mapping[str, Any]],
    req: OpenAIChatCompletionRequest,
    spec: ModelSpec,
    effective_max_tokens: int,
    stream: bool,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build southbound OpenAI `/v1/chat/completions` request payload."""
    payload: Dict[str, Any] = {
        "model": upstream_model,
        "messages": list(messages),
        "temperature": req.temperature,
        "max_tokens": int(effective_max_tokens),
        "stream": stream,
    }
    _apply_provider_southbound_params(payload, spec)
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop is not None:
        payload["stop"] = req.stop
    if req.presence_penalty is not None:
        payload["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        payload["frequency_penalty"] = req.frequency_penalty
    if req.seed is not None:
        payload["seed"] = req.seed
    if req.user is not None:
        payload["user"] = req.user
    if stream and isinstance(req.stream_options, Mapping) and req.stream_options:
        payload["stream_options"] = dict(req.stream_options)
    if tools is not None:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    return payload


def _accumulate_stream_tool_calls(
    by_index: Dict[int, Dict[str, Any]],
    tc_deltas: Sequence[Any],
) -> None:
    """Assemble streamed OpenAI `delta.tool_calls` fragments indexed by call index."""
    for pos, raw_tc in enumerate(tc_deltas):
        if not isinstance(raw_tc, Mapping):
            continue
        idx = int(raw_tc.get("index") if raw_tc.get("index") is not None else pos)
        entry = by_index.setdefault(
            idx,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if raw_tc.get("id"):
            entry["id"] = str(raw_tc["id"])
        if raw_tc.get("type"):
            entry["type"] = str(raw_tc["type"])
        fn = raw_tc.get("function")
        if isinstance(fn, Mapping):
            if fn.get("name"):
                name_piece = str(fn["name"])
                if name_piece != entry["function"]["name"]:
                    entry["function"]["name"] = str(entry["function"]["name"]) + name_piece
            if fn.get("arguments") is not None:
                entry["function"]["arguments"] = str(entry["function"]["arguments"]) + str(
                    fn["arguments"]
                )


def _normalize_tool_calls_with_ids(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    round_idx: int,
) -> List[Dict[str, Any]]:
    """Copy tool call dicts and ensure each has a stable, non-empty ``id`` for pairing."""
    out: List[Dict[str, Any]] = []
    for idx, tc in enumerate(tool_calls):
        item = dict(tc) if isinstance(tc, Mapping) else {}
        item["id"] = str(item.get("id") or f"call_{round_idx + 1}_{idx + 1}")
        out.append(item)
    return out


def _execute_tool_call_batch(
    ctx: AssembledWahsContext,
    options: WahsRequestOptions,
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    round_idx: int,
    tool_calls_log: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Execute up to 4 overflow tool calls and pair every call ID with a ``tool`` message."""
    tool_msgs: List[Dict[str, Any]] = []
    recovered_total = 0
    for tc_idx, tc in enumerate(tool_calls):
        fn = (tc.get("function") or {}) if isinstance(tc, Mapping) else {}
        fn_name = str(fn.get("name") or "")
        tc_id = str(
            (tc.get("id") if isinstance(tc, Mapping) else None)
            or f"call_{round_idx + 1}_{tc_idx + 1}"
        )
        if tc_idx >= 4:
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": json.dumps(
                        {
                            "error": "tool_call_limit_exceeded",
                            "message": "Maximum 4 tool calls executed per round; prioritize key lookups.",
                        }
                    ),
                }
            )
            continue
        try:
            parsed_args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            parsed_args = {}
        fn_args: Dict[str, Any] = dict(parsed_args) if isinstance(parsed_args, Mapping) else {}
        tool_out, rec_count = _execute_local_overflow_tool(
            ctx, fn_name, fn_args, sharing_posture=options.sharing_posture
        )
        recovered_total += rec_count
        tool_calls_log.append(
            {
                "round": round_idx + 1,
                "name": fn_name,
                "arguments": fn_args,
                "results_returned": rec_count,
            }
        )
        tool_msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": tool_out,
            }
        )
    return tool_msgs, recovered_total


class SouthboundOpenAIClient:
    """Plug-and-play OpenAI `/v1` client for any `LLM_BASE_URL` backend."""

    def __init__(
        self,
        settings: WahsEngineSettings,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings
        self._external_client = http_client
        self._shared_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._shared_client is None or self._shared_client.is_closed:
            self._shared_client = httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds)
        return self._shared_client

    async def aclose(self) -> None:
        if self._shared_client is not None and not self._shared_client.is_closed:
            await self._shared_client.aclose()
        self._shared_client = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return headers

    async def list_models(self) -> Dict[str, Any]:
        created = int(time.time())
        models_out: List[Dict[str, Any]] = [
            {"id": m, "object": "model", "created": created, "owned_by": "wahs-engine"}
            for m in self.settings.available_models
        ]
        seen = {m["id"] for m in models_out}

        client = self._get_http_client()
        try:
            resp = await client.get(
                self.settings.upstream_models_url,
                headers=self._headers(),
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in (data.get("data") or []) if isinstance(data, Mapping) else []:
                    if isinstance(item, Mapping) and item.get("id") and item["id"] not in seen:
                        seen.add(str(item["id"]))
                        models_out.append(dict(item))
        except Exception:
            pass

        return {"object": "list", "data": models_out}

    async def complete(
        self,
        *,
        ctx: AssembledWahsContext,
        req: OpenAIChatCompletionRequest,
        options: WahsRequestOptions,
        assemble_ms: float = 0.0,
    ) -> Dict[str, Any]:
        prep_t0 = time.perf_counter()
        ctx = copy.copy(ctx)
        ctx.citation_map = copy.deepcopy(ctx.citation_map)
        requested_model = (req.model or self.settings.default_model).strip()
        spec = self.settings.resolve_model_spec(requested_model)
        upstream_model = spec.upstream_model
        effective_max_tokens = (
            int(req.max_tokens)
            if req.max_tokens is not None
            else int(spec.default_max_output_tokens)
        )
        allow_tools, _, _ = _should_allow_overflow_tools(ctx, options)
        tools_spec = _OVERFLOW_TOOLS_SPEC if allow_tools else None

        (
            messages,
            kb_meta,
            overflow_recovered,
            note_rag_meta,
            temporal_digest_meta,
            exact_prompt_tokens,
            history_trimmed_turns,
        ) = await asyncio.to_thread(
            prepare_upstream_messages,
            ctx=ctx,
            req=req,
            options=options,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            tools=tools_spec,
        )
        engine_prep_ms = round(assemble_ms + (time.perf_counter() - prep_t0) * 1000.0, 2)

        payload = _build_southbound_payload(
            upstream_model=upstream_model,
            messages=messages,
            req=req,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            stream=False,
            tools=tools_spec,
        )

        tool_calls_log: List[Dict[str, Any]] = []
        prompt_ms_total = 0.0
        predicted_ms_total = 0.0
        prompt_n_total = 0
        predicted_n_total = 0
        cache_n_total = 0
        llm_rounds = 0

        def _accumulate_timings(resp_body: Mapping[str, Any]) -> None:
            nonlocal prompt_ms_total, predicted_ms_total, prompt_n_total, predicted_n_total, cache_n_total, llm_rounds
            llm_rounds += 1
            t_obj = resp_body.get("timings")
            if isinstance(t_obj, Mapping):
                prompt_ms_total += float(t_obj.get("prompt_ms") or 0.0)
                predicted_ms_total += float(t_obj.get("predicted_ms") or 0.0)
                prompt_n_total += int(t_obj.get("prompt_n") or 0)
                predicted_n_total += int(t_obj.get("predicted_n") or 0)
                cache_n_total += int(t_obj.get("cache_n") or 0)

        client = self._get_http_client()
        llm_t0 = time.perf_counter()
        try:
            resp = await client.post(
                self.settings.upstream_chat_completions_url,
                headers=self._headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Southbound LLM request failed: {exc}",
            ) from exc
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Southbound LLM returned HTTP {resp.status_code}.",
            )
        body = resp.json()
        if isinstance(body, Mapping):
            _accumulate_timings(body)

        max_tool_rounds = 3 if options.retrieval_strategy == "dynamic_fhir_tools" else 1
        followup_msgs = list(messages)
        for round_idx in range(max_tool_rounds):
            choice0 = ((body.get("choices") or [{}])[0]) if isinstance(body, Mapping) else {}
            msg0 = choice0.get("message") if isinstance(choice0, Mapping) else {}
            tool_calls = (msg0.get("tool_calls") or []) if isinstance(msg0, Mapping) else []
            if not tool_calls or not allow_tools:
                break
            normalized_calls = _normalize_tool_calls_with_ids(
                tool_calls, round_idx=round_idx
            )
            assistant_msg = dict(msg0)
            assistant_msg["tool_calls"] = normalized_calls
            followup_msgs.append(assistant_msg)
            tool_msgs, rec_delta = await asyncio.to_thread(
                _execute_tool_call_batch,
                ctx,
                options,
                normalized_calls,
                round_idx=round_idx,
                tool_calls_log=tool_calls_log,
            )
            overflow_recovered += rec_delta
            followup_msgs.extend(tool_msgs)
            next_tools = _OVERFLOW_TOOLS_SPEC if round_idx + 1 < max_tool_rounds else None
            _, extra_trimmed = _trim_history_to_fit(
                followup_msgs,
                tools=next_tools,
                tokenizer_family=spec.tokenizer_family,
                context_window=spec.context_window,
                max_output_tokens=effective_max_tokens,
            )
            history_trimmed_turns += extra_trimmed
            next_payload = _build_southbound_payload(
                upstream_model=upstream_model,
                messages=followup_msgs,
                req=req,
                spec=spec,
                effective_max_tokens=effective_max_tokens,
                stream=False,
                tools=next_tools,
            )
            try:
                resp_next = await client.post(
                    self.settings.upstream_chat_completions_url,
                    headers=self._headers(),
                    json=next_payload,
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Southbound LLM follow-up request failed: {exc}",
                ) from exc
            if resp_next.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"Southbound LLM follow-up returned HTTP {resp_next.status_code}.",
                )
            body = resp_next.json()
            if isinstance(body, Mapping):
                _accumulate_timings(body)

        upstream_llm_ms = round((time.perf_counter() - llm_t0) * 1000.0, 2)
        total_latency_ms = round(engine_prep_ms + upstream_llm_ms, 2)
        timings_envelope: Dict[str, Any] = {
            "engine_prep_ms": engine_prep_ms,
            "upstream_llm_ms": upstream_llm_ms,
            "total_latency_ms": total_latency_ms,
            "llm_rounds": llm_rounds,
            "prompt_n": prompt_n_total,
            "prompt_ms": round(prompt_ms_total, 2),
            "prompt_per_second": (
                round(prompt_n_total / (prompt_ms_total / 1000.0), 2)
                if prompt_ms_total > 0
                else 0.0
            ),
            "predicted_n": predicted_n_total,
            "predicted_ms": round(predicted_ms_total, 2),
            "predicted_per_second": (
                round(predicted_n_total / (predicted_ms_total / 1000.0), 2)
                if predicted_ms_total > 0
                else 0.0
            ),
            "cache_n": cache_n_total,
        }

        final_choice = ((body.get("choices") or [{}])[0]) if isinstance(body, Mapping) else {}
        final_finish_reason = str(
            (final_choice.get("finish_reason") if isinstance(final_choice, Mapping) else None)
            or "stop"
        )
        raw_text = (
            ((final_choice.get("message") if isinstance(final_choice, Mapping) else {}) or {}).get("content")
            or ""
        )
        cleaned_text = clean_model_text(str(raw_text))
        usage_obj = dict(body.get("usage") or {}) if isinstance(body, Mapping) and isinstance(body.get("usage"), Mapping) else {}
        upstream_prompt_tokens = int(usage_obj.get("prompt_tokens") or 0)
        suspected_truncation = _compute_suspected_truncation(
            llm_rounds=llm_rounds,
            upstream_prompt_tokens=upstream_prompt_tokens,
            exact_prompt_tokens=exact_prompt_tokens,
        )
        x_wahs = build_x_wahs_envelope(
            ctx=ctx,
            requested_model=requested_model,
            upstream_model=upstream_model,
            options=options,
            engine_version=self.settings.engine_version,
            overflow_recovered=overflow_recovered,
            kb_meta=kb_meta,
            note_rag_meta=note_rag_meta,
            temporal_digest_meta=temporal_digest_meta,
            tool_calls_log=tool_calls_log if (allow_tools or tool_calls_log) else None,
            timings=timings_envelope,
            exact_prompt_tokens=exact_prompt_tokens,
            upstream_prompt_tokens=upstream_prompt_tokens,
            suspected_upstream_truncation=suspected_truncation,
            history_trimmed_turns=history_trimmed_turns,
        )
        return {
            "id": body.get("id") or f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": body.get("created") or int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": cleaned_text},
                    "finish_reason": final_finish_reason,
                }
            ],
            "usage": usage_obj,
            "x_wahs": x_wahs,
        }

    async def start_stream(
        self,
        *,
        ctx: AssembledWahsContext,
        req: OpenAIChatCompletionRequest,
        options: WahsRequestOptions,
        assemble_ms: float = 0.0,
    ) -> AsyncIterator[str]:
        ctx = copy.copy(ctx)
        ctx.citation_map = copy.deepcopy(ctx.citation_map)
        emit_stream_events = bool(options.emit_stream_events)

        allow_tools, _, _ = _should_allow_overflow_tools(ctx, options)
        if allow_tools and options.retrieval_strategy == "dynamic_fhir_tools":
            completed = await self.complete(
                ctx=ctx,
                req=req.model_copy(update={"stream": False}),
                options=options,
                assemble_ms=assemble_ms,
            )

            async def _dynamic_tools_sse() -> AsyncIterator[str]:
                completion_id = str(completed.get("id") or f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}")
                created = int(completed.get("created") or time.time())
                requested_model = str(
                    completed.get("model") or (req.model or self.settings.default_model).strip()
                )
                choice0 = ((completed.get("choices") or [{}])[0])
                content_text = str((choice0.get("message") or {}).get("content") or "")
                tool_finish_reason = str(choice0.get("finish_reason") or "stop")
                role_chunk: Dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                if emit_stream_events:
                    role_chunk["x_wahs_stream"] = {
                        "event": "context_ready",
                        "prefix_cache_key_hash": ctx.prefix_cache_key_hash,
                        "citation_map": ctx.citation_map,
                    }
                yield f"data: {json.dumps(role_chunk)}\n\n"
                if emit_stream_events and completed["x_wahs"].get("tool_calls"):
                    tool_names = [
                        str(tc.get("name") or "chart_lookup")
                        for tc in (completed["x_wahs"].get("tool_calls") or [])
                        if isinstance(tc, Mapping)
                    ]
                    tool_event_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "x_wahs_stream": {
                            "event": "tool_round",
                            "round": 1,
                            "tools": tool_names,
                            "reset_content": False,
                            "citation_map": ctx.citation_map,
                        },
                    }
                    yield f"data: {json.dumps(tool_event_chunk)}\n\n"
                if content_text:
                    content_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {"content": content_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"
                final_chunk: Dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": tool_finish_reason}],
                    "x_wahs": completed["x_wahs"],
                }
                if completed.get("usage"):
                    final_chunk["usage"] = completed["usage"]
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return _dynamic_tools_sse()

        prep_t0 = time.perf_counter()
        requested_model = (req.model or self.settings.default_model).strip()
        spec = self.settings.resolve_model_spec(requested_model)
        upstream_model = spec.upstream_model
        effective_max_tokens = (
            int(req.max_tokens)
            if req.max_tokens is not None
            else int(spec.default_max_output_tokens)
        )
        tools_spec = _OVERFLOW_TOOLS_SPEC if allow_tools else None
        (
            messages,
            kb_meta,
            overflow_recovered,
            note_rag_meta,
            temporal_digest_meta,
            exact_prompt_tokens,
            history_trimmed_turns,
        ) = await asyncio.to_thread(
            prepare_upstream_messages,
            ctx=ctx,
            req=req,
            options=options,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            tools=tools_spec,
        )
        engine_prep_ms = round(assemble_ms + (time.perf_counter() - prep_t0) * 1000.0, 2)
        max_tool_rounds = 2 if allow_tools else 0
        current_tools0 = tools_spec if max_tool_rounds > 0 else None
        payload0 = _build_southbound_payload(
            upstream_model=upstream_model,
            messages=messages,
            req=req,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            stream=True,
            tools=current_tools0,
        )

        client = self._get_http_client()
        llm_t0 = time.perf_counter()
        req0 = client.build_request(
            "POST",
            self.settings.upstream_chat_completions_url,
            headers=self._headers(),
            json=payload0,
        )
        try:
            resp0 = await client.send(req0, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Southbound LLM stream request failed: {exc}",
            ) from exc
        if resp0.status_code >= 400:
            await resp0.aclose()
            raise HTTPException(
                status_code=502,
                detail=f"Southbound LLM stream returned HTTP {resp0.status_code}.",
            )

        async def _sse_generator() -> AsyncIterator[str]:
            nonlocal overflow_recovered, history_trimmed_turns
            completion_id = f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            upstream_finish_reason = "stop"
            upstream_usage: Dict[str, Any] = {}
            tool_calls_log: List[Dict[str, Any]] = []
            followup_msgs = list(messages)
            llm_rounds = 0
            resp0_open = True

            try:
                role_chunk: Dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                if emit_stream_events:
                    role_chunk["x_wahs_stream"] = {
                        "event": "context_ready",
                        "prefix_cache_key_hash": ctx.prefix_cache_key_hash,
                        "citation_map": ctx.citation_map,
                    }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                def _content_sse(text_piece: str) -> str:
                    out_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    return f"data: {json.dumps(out_chunk)}\n\n"

                for round_idx in range(max_tool_rounds + 1):
                    current_tools = tools_spec if round_idx < max_tool_rounds else None
                    stripper = _StreamingThinkStripper()
                    tool_calls_by_idx: Dict[int, Dict[str, Any]] = {}
                    round_raw_parts: List[str] = []
                    streaming_live = current_tools is None
                    buffered_pieces: List[str] = []
                    buffered_chars = 0
                    emitted_in_round = False
                    llm_rounds += 1

                    if round_idx == 0:
                        resp = resp0
                    else:
                        _, extra_trimmed = _trim_history_to_fit(
                            followup_msgs,
                            tools=current_tools,
                            tokenizer_family=spec.tokenizer_family,
                            context_window=spec.context_window,
                            max_output_tokens=effective_max_tokens,
                        )
                        history_trimmed_turns += extra_trimmed
                        payload = _build_southbound_payload(
                            upstream_model=upstream_model,
                            messages=followup_msgs,
                            req=req,
                            spec=spec,
                            effective_max_tokens=effective_max_tokens,
                            stream=True,
                            tools=current_tools,
                        )
                        next_req = client.build_request(
                            "POST",
                            self.settings.upstream_chat_completions_url,
                            headers=self._headers(),
                            json=payload,
                        )
                        resp = await client.send(next_req, stream=True)

                    try:
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_obj = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(chunk_obj, Mapping) and isinstance(chunk_obj.get("usage"), Mapping):
                                upstream_usage = dict(chunk_obj["usage"])
                            choice = (
                                ((chunk_obj.get("choices") or [{}])[0])
                                if isinstance(chunk_obj, Mapping)
                                else {}
                            )
                            if isinstance(choice, Mapping) and choice.get("finish_reason"):
                                upstream_finish_reason = str(choice["finish_reason"])
                            delta = (choice.get("delta") or {}) if isinstance(choice, Mapping) else {}
                            tc_deltas = delta.get("tool_calls") if isinstance(delta, Mapping) else None
                            if isinstance(tc_deltas, list):
                                _accumulate_stream_tool_calls(tool_calls_by_idx, tc_deltas)
                                if tool_calls_by_idx:
                                    streaming_live = False
                            raw_content = str(delta.get("content") or "") if isinstance(delta, Mapping) else ""
                            if raw_content:
                                round_raw_parts.append(raw_content)
                            piece = stripper.feed(raw_content)
                            if piece and not tool_calls_by_idx:
                                if streaming_live:
                                    yield _content_sse(piece)
                                    emitted_in_round = True
                                else:
                                    buffered_pieces.append(piece)
                                    buffered_chars += len(piece)
                                    if emit_stream_events and (
                                        buffered_chars >= PRE_TOOL_LOOKAHEAD_CHARS or "\n" in piece
                                    ):
                                        streaming_live = True
                                        for bp in buffered_pieces:
                                            yield _content_sse(bp)
                                        buffered_pieces.clear()
                                        buffered_chars = 0
                                        emitted_in_round = True
                    finally:
                        await resp.aclose()
                        if round_idx == 0:
                            resp0_open = False

                    tail = stripper.flush()
                    if tail and not tool_calls_by_idx:
                        if streaming_live:
                            yield _content_sse(tail)
                            emitted_in_round = True
                        else:
                            buffered_pieces.append(tail)

                    if tool_calls_by_idx and round_idx < max_tool_rounds:
                        assembled_calls = _normalize_tool_calls_with_ids(
                            [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)],
                            round_idx=round_idx,
                        )
                        followup_msgs.append(
                            {
                                "role": "assistant",
                                "content": "".join(round_raw_parts) or None,
                                "tool_calls": assembled_calls,
                            }
                        )
                        tool_msgs, rec_delta = await asyncio.to_thread(
                            _execute_tool_call_batch,
                            ctx,
                            options,
                            assembled_calls,
                            round_idx=round_idx,
                            tool_calls_log=tool_calls_log,
                        )
                        overflow_recovered += rec_delta
                        followup_msgs.extend(tool_msgs)
                        if emit_stream_events:
                            tool_names = [
                                str((tc.get("function") or {}).get("name") or "chart_lookup")
                                for tc in assembled_calls
                            ]
                            tool_event_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": requested_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                                "x_wahs_stream": {
                                    "event": "tool_round",
                                    "round": round_idx + 1,
                                    "tools": tool_names,
                                    "reset_content": bool(emitted_in_round),
                                    "citation_map": ctx.citation_map,
                                },
                            }
                            yield f"data: {json.dumps(tool_event_chunk)}\n\n"
                        continue
                    for buffered_piece in buffered_pieces:
                        yield _content_sse(buffered_piece)
                    break
            finally:
                if resp0_open:
                    await resp0.aclose()

            upstream_llm_ms = round((time.perf_counter() - llm_t0) * 1000.0, 2)
            upstream_prompt_tokens = int(upstream_usage.get("prompt_tokens") or 0)
            suspected_truncation = _compute_suspected_truncation(
                llm_rounds=max(1, llm_rounds),
                upstream_prompt_tokens=upstream_prompt_tokens,
                exact_prompt_tokens=exact_prompt_tokens,
            )
            x_wahs = build_x_wahs_envelope(
                ctx=ctx,
                requested_model=requested_model,
                upstream_model=upstream_model,
                options=options,
                engine_version=self.settings.engine_version,
                overflow_recovered=overflow_recovered,
                kb_meta=kb_meta,
                note_rag_meta=note_rag_meta,
                temporal_digest_meta=temporal_digest_meta,
                tool_calls_log=tool_calls_log if (allow_tools or tool_calls_log) else None,
                timings={
                    "engine_prep_ms": engine_prep_ms,
                    "upstream_llm_ms": upstream_llm_ms,
                    "total_latency_ms": round(engine_prep_ms + upstream_llm_ms, 2),
                    "llm_rounds": max(1, llm_rounds),
                },
                exact_prompt_tokens=exact_prompt_tokens,
                upstream_prompt_tokens=upstream_prompt_tokens,
                suspected_upstream_truncation=suspected_truncation,
                history_trimmed_turns=history_trimmed_turns,
            )
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": upstream_finish_reason}],
                "x_wahs": x_wahs,
            }
            if upstream_usage:
                final_chunk["usage"] = upstream_usage
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return _sse_generator()

    async def stream(
        self,
        *,
        ctx: AssembledWahsContext,
        req: OpenAIChatCompletionRequest,
        options: WahsRequestOptions,
        assemble_ms: float = 0.0,
    ) -> AsyncIterator[str]:
        sse_iter = await self.start_stream(
            ctx=ctx, req=req, options=options, assemble_ms=assemble_ms
        )
        async for chunk in sse_iter:
            yield chunk
