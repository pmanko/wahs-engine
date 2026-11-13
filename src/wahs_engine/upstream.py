"""Southbound OpenAI `/v1` inference client, Tier-0 overflow recovery, and SSE streamer."""

from __future__ import annotations

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
    build_wahs_hub_enrichment_block,
    get_lab_history,
    parse_omitted_clinical_items,
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
    if "<think>" in content and "</think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    if "<|channel>thought" in content and "<channel|>" in content:
        content = re.sub(r"<\|channel>thought.*?<channel\|>\s*", "", content, flags=re.DOTALL)
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
    tool_calls_log: Optional[List[Dict[str, Any]]] = None,
    timings: Optional[Dict[str, Any]] = None,
    exact_prompt_tokens: int = 0,
    upstream_prompt_tokens: int = 0,
    suspected_upstream_truncation: bool = False,
    history_trimmed_turns: int = 0,
) -> Dict[str, Any]:
    omitted_total, omitted_by_cat = parse_omitted_clinical_items(ctx.condensed.chart_text)
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


def _execute_local_overflow_tool(
    ctx: AssembledWahsContext,
    fn_name: str,
    fn_args: Mapping[str, Any],
    sharing_posture: str = "patient_self",
) -> Tuple[str, int]:
    if fn_name == "search_chart_entries":
        res = search_chart_entries(
            ctx.extracted,
            query=str(fn_args.get("query") or ""),
            section=fn_args.get("section"),
            sharing_posture=sharing_posture,
            limit=int(fn_args.get("limit") or 10),
        )
        _annotate_tool_results_with_citations(ctx, res, default_resource_type="Resource")
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "get_lab_history":
        res = get_lab_history(
            ctx.extracted,
            str(fn_args.get("analyte") or ""),
            limit=int(fn_args.get("limit") or 20),
            sharing_posture=sharing_posture,
        )
        _annotate_tool_results_with_citations(ctx, res, default_resource_type="Observation")
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "read_clinical_note":
        key = str(fn_args.get("note_id_or_query") or "")
        offset = max(0, int(fn_args.get("offset") or 0))
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
        raw_max_chars = fn_args.get("max_chars")
        if raw_max_chars is not None:
            max_chars = max(256, min(int(raw_max_chars), 24000))
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
    """Trim oldest conversation turns (preserving `messages[0]` verbatim and the latest user turn).

    Returns ``(exact_prompt_tokens, history_trimmed_turns)``. Raises HTTP 422 if the prompt still
    exceeds ``context_window - max_output_tokens`` after trimming all prior turns.
    """
    prompt_limit = int(context_window) - int(max_output_tokens)
    trimmed = 0
    while len(messages) > 2:
        exact_toks = count_chat_messages_tokens(
            messages, tools=tools, tokenizer_family=tokenizer_family
        )
        if exact_toks <= prompt_limit:
            return exact_toks, trimmed
        messages.pop(1)
        trimmed += 1
        if len(messages) > 2 and messages[1].get("role") == "assistant":
            messages.pop(1)
            trimmed += 1

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
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int, Optional[Dict[str, Any]], int, int]:
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

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ctx.system_prompt},
        *turns,
    ]

    kb_meta: Optional[Dict[str, Any]] = None
    note_rag_meta: Optional[Dict[str, Any]] = None
    overflow_recovered = 0
    effective_kb = bool(options.enable_kb) and options.retrieval_strategy != "chart_only"

    if effective_kb:
        try:
            enrichment = build_wahs_hub_enrichment_block(
                messages,
                extracted=ctx.extracted,
                max_tokens=800,
                sharing_posture=options.sharing_posture,
                token_counter=exact_counter,
            )
            kb_meta = enrichment.to_metadata_dict()
            overflow_recovered = int(kb_meta.get("overflow_items_recovered") or 0)
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
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": f"{messages[-1]['content'].rstrip()}\n\n{enrichment.enrichment_text}",
                }
        except Exception as exc:
            logger.warning("WAHS KB enrichment skipped: %s", exc)

    if options.retrieval_strategy == "note_rag":
        try:
            rag_res = retrieve_clinical_note_passages(
                ctx.extracted,
                raw_last_user_query,
                sharing_posture=options.sharing_posture,
                top_k=5,
            )
            passages = rag_res.get("results") or []
            rag_lines: List[str] = []
            rag_indices: List[int] = []
            for p in passages:
                if not isinstance(p, Mapping):
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
                if c_idx is not None and c_idx not in rag_indices:
                    rag_indices.append(c_idx)
                rag_lines.append(
                    f"{cite_prefix}({p.get('date') or 'undated'} | {p.get('display')} | "
                    f"source={p.get('source_org') or 'Chart'} | bm25={p.get('bm25_score')}): "
                    f"{p.get('passage_text')}"
                )
            note_rag_meta = {
                "total_chunks_indexed": int(rag_res.get("total_chunks_indexed") or 0),
                "passages_retrieved": len(passages),
                "cited_indices": rag_indices,
                "redacted_for_privacy_count": int(rag_res.get("redacted_for_privacy_count") or 0),
            }
            if rag_lines:
                rag_block = (
                    "=== SINGLE-PATIENT CLINICAL NOTE & REPORT PASSAGES (BM25 RAG — cite inline as [N]) ===\n"
                    + "\n".join(rag_lines)
                )
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": f"{messages[-1]['content'].rstrip()}\n\n{rag_block}",
                }
        except Exception as exc:
            logger.warning("WAHS Note RAG skipped: %s", exc)

    if options.retrieval_strategy == "dynamic_fhir_tools":
        tool_hint = (
            "=== DYNAMIC FHIR CHART TOOLS ACTIVE ===\n"
            "The Condensed Clinical Record above is a compact 2k-token IPS safety header. "
            "Whenever the question asks about prior hospital admissions, clinical notes, procedures, "
            "microbiology/pathology reports, or specific longitudinal lab dates not shown in full above, "
            "call `search_chart_entries`, `read_clinical_note`, or `get_lab_history` first, then synthesize "
            "your final patient-friendly answer citing each item's `[N]` (`citation_tag`) inline."
        )
        messages[-1] = {
            "role": messages[-1]["role"],
            "content": f"{messages[-1]['content'].rstrip()}\n\n{tool_hint}",
        }

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
        exact_prompt_tokens,
        history_trimmed_turns,
    )


class SouthboundOpenAIClient:
    """Plug-and-play OpenAI `/v1` client for any `LLM_BASE_URL` backend."""

    def __init__(
        self,
        settings: WahsEngineSettings,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings
        self._external_client = http_client

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

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.get(self.settings.upstream_models_url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json()
                for item in (data.get("data") or []) if isinstance(data, Mapping) else []:
                    if isinstance(item, Mapping) and item.get("id") and item["id"] not in seen:
                        seen.add(str(item["id"]))
                        models_out.append(dict(item))
        except Exception:
            pass
        finally:
            if owns_client:
                await client.aclose()

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
        effective_max_tokens = int(req.max_tokens or spec.default_max_output_tokens)
        omitted_total, _ = parse_omitted_clinical_items(ctx.condensed.chart_text)
        allow_tools = options.retrieval_strategy == "dynamic_fhir_tools" or (
            options.enable_overflow_tools
            and omitted_total > 0
            and options.retrieval_strategy != "chart_only"
        )
        tools_spec = _OVERFLOW_TOOLS_SPEC if allow_tools else None

        (
            messages,
            kb_meta,
            overflow_recovered,
            note_rag_meta,
            exact_prompt_tokens,
            history_trimmed_turns,
        ) = prepare_upstream_messages(
            ctx=ctx,
            req=req,
            options=options,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            tools=tools_spec,
        )
        engine_prep_ms = round(assemble_ms + (time.perf_counter() - prep_t0) * 1000.0, 2)

        payload: Dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
        }
        _apply_provider_southbound_params(payload, spec)
        if req.seed is not None:
            payload["seed"] = req.seed
        if tools_spec is not None:
            payload["tools"] = tools_spec

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

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds)
        llm_t0 = time.perf_counter()
        try:
            resp = await client.post(
                self.settings.upstream_chat_completions_url,
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
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
                followup_msgs.append(dict(msg0))
                for tc in tool_calls[:4]:
                    fn = (tc.get("function") or {}) if isinstance(tc, Mapping) else {}
                    fn_name = str(fn.get("name") or "")
                    try:
                        fn_args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        fn_args = {}
                    tool_out, rec_count = _execute_local_overflow_tool(
                        ctx, fn_name, fn_args, sharing_posture=options.sharing_posture
                    )
                    overflow_recovered += rec_count
                    tool_calls_log.append(
                        {
                            "round": round_idx + 1,
                            "name": fn_name,
                            "arguments": fn_args,
                            "results_returned": rec_count,
                        }
                    )
                    followup_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tc.get("id") or f"call_{round_idx + 1}"),
                            "name": fn_name,
                            "content": tool_out,
                        }
                    )
                next_tools = _OVERFLOW_TOOLS_SPEC if round_idx + 1 < max_tool_rounds else None
                next_payload: Dict[str, Any] = {
                    "model": upstream_model,
                    "messages": followup_msgs,
                    "temperature": req.temperature,
                    "max_tokens": effective_max_tokens,
                    "stream": False,
                }
                _apply_provider_southbound_params(next_payload, spec)
                if req.seed is not None:
                    next_payload["seed"] = req.seed
                if next_tools is not None:
                    next_payload["tools"] = next_tools
                round_toks = count_chat_messages_tokens(
                    followup_msgs, tools=next_tools, tokenizer_family=spec.tokenizer_family
                )
                if round_toks > (spec.context_window - effective_max_tokens):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Tool follow-up prompt ({round_toks} exact {spec.tokenizer_family} tokens) "
                            f"+ max_tokens ({effective_max_tokens}) exceeds model context_window ({spec.context_window})."
                        ),
                    )
                resp_next = await client.post(
                    self.settings.upstream_chat_completions_url,
                    headers=self._headers(),
                    json=next_payload,
                )
                resp_next.raise_for_status()
                body = resp_next.json()
                if isinstance(body, Mapping):
                    _accumulate_timings(body)
        finally:
            if owns_client:
                await client.aclose()

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

    async def stream(
        self,
        *,
        ctx: AssembledWahsContext,
        req: OpenAIChatCompletionRequest,
        options: WahsRequestOptions,
        assemble_ms: float = 0.0,
    ) -> AsyncIterator[str]:
        ctx = copy.copy(ctx)
        ctx.citation_map = copy.deepcopy(ctx.citation_map)

        if options.retrieval_strategy == "dynamic_fhir_tools":
            completed = await self.complete(
                ctx=ctx,
                req=req.model_copy(update={"stream": False}),
                options=options,
                assemble_ms=assemble_ms,
            )
            completion_id = str(completed.get("id") or f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}")
            created = int(completed.get("created") or time.time())
            requested_model = str(
                completed.get("model") or (req.model or self.settings.default_model).strip()
            )
            choice0 = ((completed.get("choices") or [{}])[0])
            content_text = str((choice0.get("message") or {}).get("content") or "")
            tool_finish_reason = str(choice0.get("finish_reason") or "stop")
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"
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
            return

        prep_t0 = time.perf_counter()
        requested_model = (req.model or self.settings.default_model).strip()
        spec = self.settings.resolve_model_spec(requested_model)
        upstream_model = spec.upstream_model
        effective_max_tokens = int(req.max_tokens or spec.default_max_output_tokens)
        (
            messages,
            kb_meta,
            overflow_recovered,
            note_rag_meta,
            exact_prompt_tokens,
            history_trimmed_turns,
        ) = prepare_upstream_messages(
            ctx=ctx,
            req=req,
            options=options,
            spec=spec,
            effective_max_tokens=effective_max_tokens,
            tools=None,
        )
        engine_prep_ms = round(assemble_ms + (time.perf_counter() - prep_t0) * 1000.0, 2)
        completion_id = f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        payload: Dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
        }
        _apply_provider_southbound_params(payload, spec)
        if req.seed is not None:
            payload["seed"] = req.seed

        # Standard initial assistant role chunk (`@ai-sdk/openai-compatible` compatible)
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        stripper = _StreamingThinkStripper()
        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds)
        llm_t0 = time.perf_counter()
        upstream_finish_reason = "stop"
        upstream_usage: Dict[str, Any] = {}
        try:
            async with client.stream(
                "POST",
                self.settings.upstream_chat_completions_url,
                headers=self._headers(),
                json=payload,
            ) as resp:
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
                    choice = ((chunk_obj.get("choices") or [{}])[0]) if isinstance(chunk_obj, Mapping) else {}
                    if isinstance(choice, Mapping) and choice.get("finish_reason"):
                        upstream_finish_reason = str(choice["finish_reason"])
                    delta = (choice.get("delta") or {}) if isinstance(choice, Mapping) else {}
                    piece = stripper.feed(str(delta.get("content") or ""))
                    if piece:
                        out_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": requested_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(out_chunk)}\n\n"

            tail = stripper.flush()
            if tail:
                tail_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tail_chunk)}\n\n"
        finally:
            if owns_client:
                await client.aclose()

        upstream_llm_ms = round((time.perf_counter() - llm_t0) * 1000.0, 2)
        upstream_prompt_tokens = int(upstream_usage.get("prompt_tokens") or 0)
        suspected_truncation = _compute_suspected_truncation(
            llm_rounds=1,
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
            timings={
                "engine_prep_ms": engine_prep_ms,
                "upstream_llm_ms": upstream_llm_ms,
                "total_latency_ms": round(engine_prep_ms + upstream_llm_ms, 2),
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
