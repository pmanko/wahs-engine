"""Southbound OpenAI `/v1` inference client, Tier-0 overflow recovery, and SSE streamer."""

from __future__ import annotations

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
    search_chart_entries,
)

from .assembler import AssembledWahsContext
from .config import WahsEngineSettings
from .schemas import OpenAIChatCompletionRequest, WahsRequestOptions

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
            "description": "Read the full deduplicated text of a clinical note (DocumentReference) by UUID or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id_or_query": {"type": "string"},
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


def build_x_wahs_envelope(
    *,
    ctx: AssembledWahsContext,
    requested_model: str,
    upstream_model: str,
    options: WahsRequestOptions,
    engine_version: str,
    overflow_recovered: int = 0,
    kb_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    omitted_total, omitted_by_cat = parse_omitted_clinical_items(ctx.condensed.chart_text)
    return {
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
            "actual_tokens": ctx.condensed.actual_tokens,
            "omitted_clinical_items": omitted_total,
            "omitted_by_category": omitted_by_cat,
            "overflow_recovered": overflow_recovered,
            "datasets_included": len(ctx.datasets_included),
            "prefix_cache_key_hash": ctx.prefix_cache_key_hash,
            "cache_hit": ctx.cache_hit,
            "dataset_cache_hits": ctx.dataset_cache_hits,
            "dataset_fetches": ctx.dataset_fetches,
        },
        "completeness_manifest": ctx.condensed.completeness_manifest,
        "privacy_notices": ctx.privacy_notices,
        "ai_provenance": {
            "resourceType": "Provenance",
            "profile": "http://hl7.org/fhir/uv/aitransparency/StructureDefinition/AI-Provenance",
            "recorded_model": upstream_model,
            "requested_model": requested_model,
            "engine": f"wahs-engine/{engine_version}",
            "kb_enabled": bool(options.enable_kb),
            "kb_enrichment": kb_meta,
            "human_review_required": True,
        },
    }


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
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "get_lab_history":
        res = get_lab_history(
            ctx.extracted,
            str(fn_args.get("analyte") or ""),
            limit=int(fn_args.get("limit") or 20),
            sharing_posture=sharing_posture,
        )
        return json.dumps(res, default=str), len(res.get("results") or [])
    if fn_name == "read_clinical_note":
        key = str(fn_args.get("note_id_or_query") or "")
        res = read_clinical_note(
            ctx.extracted,
            resource_uuid=key,
            sharing_posture=sharing_posture,
        )
        if not res.get("results") and key:
            res = read_clinical_note(
                ctx.extracted,
                query=key,
                sharing_posture=sharing_posture,
            )
        return json.dumps(res, default=str), len(res.get("results") or [])
    return json.dumps({"error": f"Unknown tool '{fn_name}'"}), 0


def prepare_upstream_messages(
    *,
    ctx: AssembledWahsContext,
    req: OpenAIChatCompletionRequest,
    options: WahsRequestOptions,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
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

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ctx.system_prompt},
        *turns,
    ]

    kb_meta: Optional[Dict[str, Any]] = None
    overflow_recovered = 0
    if options.enable_kb:
        try:
            enrichment = build_wahs_hub_enrichment_block(
                messages,
                extracted=ctx.extracted,
                max_tokens=800,
            )
            kb_meta = enrichment.to_metadata_dict()
            overflow_recovered = int(kb_meta.get("overflow_items_recovered") or 0)
            if enrichment.enrichment_text:
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": f"{messages[-1]['content'].rstrip()}\n\n{enrichment.enrichment_text}",
                }
        except Exception as exc:
            logger.warning("WAHS KB enrichment skipped: %s", exc)

    return messages, kb_meta, overflow_recovered


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
    ) -> Dict[str, Any]:
        requested_model = (req.model or self.settings.default_model).strip()
        upstream_model = self.settings.resolve_upstream_model(requested_model)
        messages, kb_meta, overflow_recovered = prepare_upstream_messages(
            ctx=ctx, req=req, options=options
        )
        omitted_total, _ = parse_omitted_clinical_items(ctx.condensed.chart_text)

        payload: Dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens or 1024,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if options.enable_overflow_tools and omitted_total > 0:
            payload["tools"] = _OVERFLOW_TOOLS_SPEC

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds)
        try:
            resp = await client.post(
                self.settings.upstream_chat_completions_url,
                headers=self._headers(),
                json=payload,
            )
            if resp.status_code in {400, 422} and (
                "tools" in payload or "chat_template_kwargs" in payload
            ):
                # Graceful fallback for strict OpenAI servers that reject optional extensions
                payload.pop("tools", None)
                payload.pop("chat_template_kwargs", None)
                resp = await client.post(
                    self.settings.upstream_chat_completions_url,
                    headers=self._headers(),
                    json=payload,
                )
            resp.raise_for_status()
            body = resp.json()

            # Single-round read-only Tier-0 overflow tool execution if model requested a lookup
            choice0 = ((body.get("choices") or [{}])[0]) if isinstance(body, Mapping) else {}
            msg0 = choice0.get("message") if isinstance(choice0, Mapping) else {}
            tool_calls = (msg0.get("tool_calls") or []) if isinstance(msg0, Mapping) else []
            if tool_calls and options.enable_overflow_tools:
                followup_msgs = [*messages, dict(msg0)]
                for tc in tool_calls[:3]:
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
                    followup_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tc.get("id") or "call_1"),
                            "name": fn_name,
                            "content": tool_out,
                        }
                    )
                second_payload = {
                    "model": upstream_model,
                    "messages": followup_msgs,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens or 1024,
                    "stream": False,
                }
                resp2 = await client.post(
                    self.settings.upstream_chat_completions_url,
                    headers=self._headers(),
                    json=second_payload,
                )
                resp2.raise_for_status()
                body = resp2.json()
        finally:
            if owns_client:
                await client.aclose()

        raw_text = (
            (((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        )
        cleaned_text = clean_model_text(str(raw_text))
        x_wahs = build_x_wahs_envelope(
            ctx=ctx,
            requested_model=requested_model,
            upstream_model=upstream_model,
            options=options,
            engine_version=self.settings.engine_version,
            overflow_recovered=overflow_recovered,
            kb_meta=kb_meta,
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
                    "finish_reason": "stop",
                }
            ],
            "usage": body.get("usage") or {},
            "x_wahs": x_wahs,
        }

    async def stream(
        self,
        *,
        ctx: AssembledWahsContext,
        req: OpenAIChatCompletionRequest,
        options: WahsRequestOptions,
    ) -> AsyncIterator[str]:
        requested_model = (req.model or self.settings.default_model).strip()
        upstream_model = self.settings.resolve_upstream_model(requested_model)
        messages, kb_meta, overflow_recovered = prepare_upstream_messages(
            ctx=ctx, req=req, options=options
        )
        completion_id = f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        payload: Dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens or 1024,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

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
        try:
            active_payload = dict(payload)
            for attempt in range(2):
                async with client.stream(
                    "POST",
                    self.settings.upstream_chat_completions_url,
                    headers=self._headers(),
                    json=active_payload,
                ) as resp:
                    if resp.status_code in {400, 422} and attempt == 0 and "chat_template_kwargs" in active_payload:
                        active_payload.pop("chat_template_kwargs", None)
                        continue
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
                        choice = ((chunk_obj.get("choices") or [{}])[0]) if isinstance(chunk_obj, Mapping) else {}
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
                    break

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

        x_wahs = build_x_wahs_envelope(
            ctx=ctx,
            requested_model=requested_model,
            upstream_model=upstream_model,
            options=options,
            engine_version=self.settings.engine_version,
            overflow_recovered=overflow_recovered,
            kb_meta=kb_meta,
        )
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "x_wahs": x_wahs,
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
