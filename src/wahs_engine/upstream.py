"""Stateless southbound OpenAI `/v1` inference client, SSE streamer, and `x_wahs` provenance builder."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from fastapi import HTTPException
import httpx

from .assembler import AssembledWahsContext, normalize_conversation_turns
from .config import ModelSpec, WahsEngineSettings
from .pipeline import EnricherPipeline
from .schemas import OpenAIChatCompletionRequest, WahsRequestOptions
from .tokenizers_registry import count_chat_messages_tokens, get_exact_token_counter


class _StreamingThinkStripper:
    """Strips `<think>...</think>` and `<|channel>thought...<channel|>` across SSE deltas."""

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
                    keep = len(self._in_tag) - 1
                    if len(self._buffer) > keep:
                        self._buffer = self._buffer[-keep:]
                    break
                self._buffer = self._buffer[end_idx + len(self._in_tag) :].lstrip("\n\r ")
                self._in_tag = None
                continue
            starts = [(self._buffer.find(o), c, len(o)) for o, c in (("<think>", "</think>"), ("<|channel>thought", "<channel|>")) if self._buffer.find(o) >= 0]
            if not starts:
                hold = 0
                for opener in ("<think>", "<|channel>thought"):
                    for k in range(min(len(opener) - 1, len(self._buffer)), 0, -1):
                        if self._buffer.endswith(opener[:k]):
                            hold = max(hold, k)
                            break
                if hold:
                    if len(self._buffer) > hold:
                        out.append(self._buffer[:-hold])
                    self._buffer = self._buffer[-hold:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break
            pos, close_marker, olen = min(starts, key=lambda x: x[0])
            if pos > 0:
                out.append(self._buffer[:pos])
            self._buffer = self._buffer[pos + olen :]
            self._in_tag = close_marker
        return "".join(out)

    def flush(self) -> str:
        if self._in_tag is not None:
            self._buffer, self._in_tag = "", None
            return ""
        rem, self._buffer = self._buffer, ""
        return rem


def clean_model_text(content: str) -> str:
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    if "<|channel>thought" in content:
        content = re.sub(r"<\|channel>thought.*?<channel\|>\s*", "", content, flags=re.DOTALL)
        content = re.sub(r"<\|channel>thought.*$", "", content, flags=re.DOTALL)
    return content.strip()


def build_x_wahs_envelope(
    *, ctx: AssembledWahsContext, requested_model: str, upstream_model: str, options: WahsRequestOptions,
    engine_version: str, enrichers: Sequence[Mapping[str, Any]] = (), timings: Optional[Dict[str, Any]] = None,
    exact_prompt_tokens: int = 0, upstream_prompt_tokens: int = 0, suspected_upstream_truncation: bool = False,
    history_trimmed_turns: int = 0, enrichment_tokens: int = 0, enrichment_text: str = "",
) -> Dict[str, Any]:
    manifest, enricher_list = ctx.condensed.completeness_manifest or {}, [dict(e) for e in enrichers]
    env: Dict[str, Any] = {
        "citation_map": ctx.citation_map,
        "scope_applied": {**ctx.scope_applied, "datasets_included": ctx.datasets_included},
        "condensation": {
            "tier": ctx.condensed.tier, "tier_name": ctx.condensed.tier_name,
            "profile_id": ctx.profile_id, "prompt_id": ctx.prompt_id, "prompt_sha256": ctx.prompt_sha256[:12],
            "enrichment_tokens": int(enrichment_tokens),
            "budget_tokens": ctx.condensed.target_tokens, "budget_effective": ctx.budget_effective or ctx.condensed.target_tokens,
            "budget_requested": ctx.budget_requested, "requested_budget_tokens": options.target_budget_tokens,
            "context_window": ctx.context_window, "preferred_chart_tokens": ctx.preferred_chart_tokens,
            "resolution_reason": ctx.clamp_reason, "clamp_reason": ctx.clamp_reason, "tokenizer_family": ctx.tokenizer_family,
            "actual_tokens": ctx.condensed.actual_tokens, "omitted_clinical_items": int(manifest.get("omitted_clinical_items_count") or 0),
            "omitted_by_section": dict(manifest.get("omitted_by_section") or {}), "datasets_included": len(ctx.datasets_included),
            "prefix_cache_key_hash": ctx.prefix_cache_key_hash, "cache_hit": ctx.cache_hit,
            "dataset_cache_hits": ctx.dataset_cache_hits, "dataset_fetches": ctx.dataset_fetches,
            "partial_fetch": ctx.partial_fetch, "partial_datasets": ctx.partial_datasets,
            "exact_prompt_tokens": exact_prompt_tokens, "upstream_prompt_tokens": upstream_prompt_tokens,
            "suspected_upstream_truncation": suspected_upstream_truncation, "history_trimmed_turns": history_trimmed_turns,
        },
        "completeness_manifest": manifest, "notices": list(ctx.notices), "enrichers": enricher_list,
        "enrichment_text": enrichment_text,
        "ai_provenance": {
            "resourceType": "Provenance", "profile": "http://hl7.org/fhir/uv/aitransparency/StructureDefinition/AI-Provenance",
            "recorded_model": upstream_model, "requested_model": requested_model,
            "engine": f"wahs-engine/{engine_version}",
            "profile_id": ctx.profile_id, "prompt_id": ctx.prompt_id, "prompt_sha256": ctx.prompt_sha256,
            "enrichment_tokens": int(enrichment_tokens),
            "enrichers": enricher_list, "human_review_required": True,
        },
    }
    if timings is not None:
        env["timings"] = timings
    return env


def _apply_provider_params(payload: Dict[str, Any], spec: ModelSpec) -> None:
    flavor = (spec.provider_flavor or "litellm").strip().lower()
    if flavor in {"litellm", "ollama"}:
        payload["num_ctx"] = int(spec.context_window)
        if spec.disable_thinking:
            payload["reasoning_effort"] = "none"
    elif flavor == "llamacpp" and spec.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}


def _trim_history_to_fit(messages: List[Dict[str, Any]], *, tokenizer_family: str, context_window: int, max_output_tokens: int) -> Tuple[int, int]:
    prompt_limit = int(context_window) - int(max_output_tokens)
    trimmed = 0
    while True:
        toks = count_chat_messages_tokens(messages, tokenizer_family=tokenizer_family)
        if toks <= prompt_limit:
            return toks, trimmed
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=1)
        if last_user <= 1:
            break
        messages.pop(1)
        trimmed += 1
        last_user -= 1
        while last_user > 1 and messages[1].get("role") != "user":
            messages.pop(1)
            trimmed += 1
            last_user -= 1
    toks = count_chat_messages_tokens(messages, tokenizer_family=tokenizer_family)
    if toks > prompt_limit:
        raise HTTPException(status_code=422, detail=f"Assembled prompt ({toks} tokens) + max_tokens ({max_output_tokens}) exceeds context_window ({context_window}).")
    return toks, trimmed


def prepare_upstream_messages(
    *,
    ctx: AssembledWahsContext,
    req: OpenAIChatCompletionRequest,
    spec: ModelSpec,
    effective_max_tokens: int,
    pipeline: Optional[EnricherPipeline] = None,
    max_enrichment_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int, str]:
    turns = normalize_conversation_turns(req.messages)

    if not turns or not any(t["role"] == "user" for t in turns):
        raise HTTPException(status_code=400, detail="messages must contain at least one non-empty user message.")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": ctx.system_prompt}, *turns]
    disclosures: List[Dict[str, Any]] = []
    enrichment_tokens = 0
    enrichment_text = ""

    active_ids = (
        ctx.active_enrichers
        if ctx.enrichment_resolved
        else (ctx.active_enrichers or (pipeline.enabled_ids if pipeline is not None else ()))
    )
    if pipeline is not None and active_ids:
        counter = get_exact_token_counter(spec.tokenizer_family)
        min_toks = count_chat_messages_tokens([messages[0], messages[-1]], tokenizer_family=spec.tokenizer_family)
        headroom = max(0, spec.context_window - effective_max_tokens - min_toks)
        last_q = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
        p_res = pipeline.run(
            extracted=ctx.extracted, condensed=ctx.condensed, citation_map=ctx.citation_map,
            question=last_q, conversation=turns, headroom_tokens=headroom, token_counter=counter,
            active_ids=active_ids, max_enrichment_tokens=max_enrichment_tokens,
        )
        disclosures = p_res.disclosures
        enrichment_tokens = p_res.used_tokens
        block = p_res.combined_block()
        if block:
            enrichment_text = block
            messages[-1] = {"role": messages[-1]["role"], "content": f"{messages[-1]['content']}\n\n{block}"}

    exact_toks, trimmed = _trim_history_to_fit(
        messages, tokenizer_family=spec.tokenizer_family, context_window=spec.context_window, max_output_tokens=effective_max_tokens
    )
    return messages, disclosures, exact_toks, trimmed, enrichment_tokens, enrichment_text


def _build_southbound_payload(*, upstream_model: str, messages: Sequence[Mapping[str, Any]], req: OpenAIChatCompletionRequest, spec: ModelSpec, effective_max_tokens: int, stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": upstream_model, "messages": list(messages), "temperature": req.temperature, "max_tokens": int(effective_max_tokens), "stream": stream}
    _apply_provider_params(payload, spec)
    for opt in ("top_p", "stop", "presence_penalty", "frequency_penalty", "seed", "user"):
        val = getattr(req, opt, None)
        if val is not None:
            payload[opt] = val
    if stream and isinstance(req.stream_options, Mapping) and req.stream_options:
        payload["stream_options"] = dict(req.stream_options)
    return payload


class SouthboundOpenAIClient:
    """Stateless OpenAI `/v1` client forwarding `messages[0]` chart prefix + conversation."""

    def __init__(self, settings: WahsEngineSettings, http_client: Optional[httpx.AsyncClient] = None, pipeline: Optional[EnricherPipeline] = None) -> None:
        self.settings = settings
        self._external_client = http_client
        self._shared_client: Optional[httpx.AsyncClient] = None
        self.pipeline = pipeline if pipeline is not None else EnricherPipeline(settings.enabled_enrichers, max_enrichment_tokens=settings.max_enrichment_tokens)

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
        h = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            h["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return h

    async def list_models(self) -> Dict[str, Any]:
        created = int(time.time())
        models_out = [{"id": m, "object": "model", "created": created, "owned_by": "wahs-engine"} for m in self.settings.available_models]
        seen = {m["id"] for m in models_out}
        try:
            resp = await self._get_http_client().get(self.settings.upstream_models_url, headers=self._headers(), timeout=10.0)
            if resp.status_code == 200:
                for item in (resp.json().get("data") or []) if isinstance(resp.json(), Mapping) else ():
                    if isinstance(item, Mapping) and item.get("id") and item["id"] not in seen:
                        seen.add(str(item["id"]))
                        models_out.append(dict(item))
        except Exception:
            pass
        return {"object": "list", "data": models_out}

    async def _prepare_turn(self, ctx: AssembledWahsContext, req: OpenAIChatCompletionRequest, *, stream: bool, assemble_ms: float) -> Tuple[AssembledWahsContext, str, ModelSpec, List[Dict[str, Any]], int, int, int, str, float, Dict[str, Any]]:
        prep_t0 = time.perf_counter()
        ctx_copy = copy.copy(ctx)
        ctx_copy.citation_map = copy.deepcopy(ctx.citation_map)
        requested_model = (req.model or self.settings.default_model).strip()
        spec = self.settings.resolve_model_spec(requested_model)
        eff_max = int(req.max_tokens) if req.max_tokens is not None else int(spec.default_max_output_tokens)
        messages, disclosures, exact_toks, trimmed, enr_toks, enr_text = await asyncio.to_thread(
            prepare_upstream_messages, ctx=ctx_copy, req=req, spec=spec, effective_max_tokens=eff_max,
            pipeline=self.pipeline, max_enrichment_tokens=self.settings.max_enrichment_tokens,
        )
        prep_ms = round(assemble_ms + (time.perf_counter() - prep_t0) * 1000.0, 2)
        payload = _build_southbound_payload(upstream_model=spec.upstream_model, messages=messages, req=req, spec=spec, effective_max_tokens=eff_max, stream=stream)
        return ctx_copy, requested_model, spec, disclosures, exact_toks, trimmed, enr_toks, enr_text, prep_ms, payload

    async def complete(self, *, ctx: AssembledWahsContext, req: OpenAIChatCompletionRequest, options: WahsRequestOptions, assemble_ms: float = 0.0) -> Dict[str, Any]:
        ctx, requested_model, spec, disclosures, exact_toks, trimmed, enr_toks, enr_text, prep_ms, payload = await self._prepare_turn(ctx, req, stream=False, assemble_ms=assemble_ms)

        llm_t0 = time.perf_counter()
        try:
            resp = await self._get_http_client().post(self.settings.upstream_chat_completions_url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Southbound LLM request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Southbound LLM returned HTTP {resp.status_code}.")

        llm_ms = round((time.perf_counter() - llm_t0) * 1000.0, 2)
        body = resp.json()
        choice0 = ((body.get("choices") or [{}])[0]) if isinstance(body, Mapping) else {}
        cleaned = clean_model_text(str(((choice0.get("message") if isinstance(choice0, Mapping) else {}) or {}).get("content") or ""))
        usage = dict(body.get("usage") or {}) if isinstance(body, Mapping) and isinstance(body.get("usage"), Mapping) else {}
        up_ptoks = int(usage.get("prompt_tokens") or 0)
        t_obj = body.get("timings") if isinstance(body, Mapping) and isinstance(body.get("timings"), Mapping) else {}
        prompt_n, prompt_ms = int(t_obj.get("prompt_n") or 0), float(t_obj.get("prompt_ms") or 0.0)
        pred_n, pred_ms = int(t_obj.get("predicted_n") or 0), float(t_obj.get("predicted_ms") or 0.0)

        x_wahs = build_x_wahs_envelope(
            ctx=ctx, requested_model=requested_model, upstream_model=spec.upstream_model, options=options,
            engine_version=self.settings.engine_version, enrichers=disclosures,
            timings={
                "engine_prep_ms": prep_ms, "upstream_llm_ms": llm_ms, "total_latency_ms": round(prep_ms + llm_ms, 2),
                "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 2), "prompt_per_second": round(prompt_n / (prompt_ms / 1000.0), 2) if prompt_ms > 0 else 0.0,
                "predicted_n": pred_n, "predicted_ms": round(pred_ms, 2), "predicted_per_second": round(pred_n / (pred_ms / 1000.0), 2) if pred_ms > 0 else 0.0,
                "cache_n": int(t_obj.get("cache_n") or 0),
            },
            exact_prompt_tokens=exact_toks, upstream_prompt_tokens=up_ptoks,
            suspected_upstream_truncation=bool(up_ptoks > 0 and exact_toks >= 1024 and up_ptoks < int(exact_toks * 0.85)),
            history_trimmed_turns=trimmed, enrichment_tokens=enr_toks, enrichment_text=enr_text,
        )
        return {
            "id": body.get("id") or f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": body.get("created") or int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": cleaned}, "finish_reason": str((choice0.get("finish_reason") if isinstance(choice0, Mapping) else None) or "stop")}],
            "usage": usage,
            "x_wahs": x_wahs,
        }

    async def start_stream(self, *, ctx: AssembledWahsContext, req: OpenAIChatCompletionRequest, options: WahsRequestOptions, assemble_ms: float = 0.0) -> AsyncIterator[str]:
        ctx, requested_model, spec, disclosures, exact_toks, trimmed, enr_toks, enr_text, prep_ms, payload = await self._prepare_turn(ctx, req, stream=True, assemble_ms=assemble_ms)

        client = self._get_http_client()
        llm_t0 = time.perf_counter()
        try:
            resp = await client.send(client.build_request("POST", self.settings.upstream_chat_completions_url, headers=self._headers(), json=payload), stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Southbound LLM stream request failed: {exc}") from exc
        if resp.status_code >= 400:
            await resp.aclose()
            raise HTTPException(status_code=502, detail=f"Southbound LLM stream returned HTTP {resp.status_code}.")

        async def _sse() -> AsyncIterator[str]:
            cid, created, finish_reason, up_usage = f"chatcmpl-wahs-{uuid.uuid4().hex[:12]}", int(time.time()), "stop", {}
            stripper = _StreamingThinkStripper()
            try:
                role_chunk: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": requested_model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
                if options.emit_stream_events:
                    role_chunk["x_wahs_stream"] = {"event": "context_ready", "prefix_cache_key_hash": ctx.prefix_cache_key_hash, "citation_map": ctx.citation_map}
                yield f"data: {json.dumps(role_chunk)}\n\n"
                async for line in resp.aiter_lines():
                    ln = line.strip()
                    if not ln.startswith("data:"):
                        continue
                    data_str = ln[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        cobj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(cobj, Mapping) and isinstance(cobj.get("usage"), Mapping):
                        up_usage = dict(cobj["usage"])
                    ch = ((cobj.get("choices") or [{}])[0]) if isinstance(cobj, Mapping) else {}
                    if isinstance(ch, Mapping) and ch.get("finish_reason"):
                        finish_reason = str(ch["finish_reason"])
                    delta = (ch.get("delta") or {}) if isinstance(ch, Mapping) else {}
                    piece = stripper.feed(str(delta.get("content") or "") if isinstance(delta, Mapping) else "")
                    if piece:
                        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': requested_model, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
                tail = stripper.flush()
                if tail:
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': requested_model, 'choices': [{'index': 0, 'delta': {'content': tail}, 'finish_reason': None}]})}\n\n"
            finally:
                await resp.aclose()

            llm_ms = round((time.perf_counter() - llm_t0) * 1000.0, 2)
            up_ptoks = int(up_usage.get("prompt_tokens") or 0)
            x_wahs = build_x_wahs_envelope(
                ctx=ctx, requested_model=requested_model, upstream_model=spec.upstream_model, options=options,
                engine_version=self.settings.engine_version, enrichers=disclosures,
                timings={"engine_prep_ms": prep_ms, "upstream_llm_ms": llm_ms, "total_latency_ms": round(prep_ms + llm_ms, 2)},
                exact_prompt_tokens=exact_toks, upstream_prompt_tokens=up_ptoks,
                suspected_upstream_truncation=bool(up_ptoks > 0 and exact_toks >= 1024 and up_ptoks < int(exact_toks * 0.85)),
                history_trimmed_turns=trimmed, enrichment_tokens=enr_toks, enrichment_text=enr_text,
            )
            fchunk: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": requested_model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "x_wahs": x_wahs}
            if up_usage:
                fchunk["usage"] = up_usage
            yield f"data: {json.dumps(fchunk)}\n\n"
            yield "data: [DONE]\n\n"

        return _sse()
