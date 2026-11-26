"""FastAPI application for slim baseline and modular profile-driven `wahs-engine`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import os
import time
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from .assembler import WAHSBundleAssembler, normalize_conversation_turns
from .config import WahsEngineSettings
from .jwt_claims import VerifiedCallerIdentity, decode_and_bind_jwt, extract_bearer_token
from .pipeline import EnricherPipeline
from .profile_registry import load_profile_registry, serialize_profiles_catalog
from .schemas import OpenAIChatCompletionRequest, WahsContextPreviewRequest, WahsRequestOptions
from .tokenizers_registry import count_chat_messages_tokens, get_exact_token_counter
from .upstream import SouthboundOpenAIClient, build_x_wahs_envelope


def _resolve_caller_identity(request: Request, options: WahsRequestOptions, settings: WahsEngineSettings) -> Optional[VerifiedCallerIdentity]:
    raw_token = extract_bearer_token(request.headers)
    if not raw_token or (raw_token == "local-demo-token" and settings.allow_local_bundle_mode and options.fhir_bundle is not None):
        if settings.allow_local_bundle_mode and options.fhir_bundle is not None:
            return None
        return decode_and_bind_jwt("")
    trusted_sub = request.headers.get(settings.trusted_sub_header) if settings.trusted_sub_header else None
    return decode_and_bind_jwt(
        raw_token, hs256_secret=settings.jwt_hs256_secret, jwks_json=settings.jwt_jwks_json,
        jwks_url=settings.jwt_jwks_url, expected_issuer=settings.jwt_issuer,
        expected_audience=settings.jwt_audience, trusted_header_sub=trusted_sub,
    )


def create_app(
    settings: Optional[WahsEngineSettings] = None,
    *,
    fhir_http_client: Optional[httpx.AsyncClient] = None,
    llm_http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    cfg = settings or WahsEngineSettings.from_env()
    pipeline = EnricherPipeline(cfg.enabled_enrichers, max_enrichment_tokens=cfg.max_enrichment_tokens)
    assembler = WAHSBundleAssembler(cfg, http_client=fhir_http_client)
    southbound = SouthboundOpenAIClient(cfg, http_client=llm_http_client, pipeline=pipeline)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await assembler.aclose()
            await southbound.aclose()

    application = FastAPI(
        title="WAHS Engine",
        version=cfg.engine_version,
        description="Zero-privilege OpenAI-compatible clinical FHIR R4 verbatim-first condensation and [N] provenance middleware.",
        lifespan=_lifespan,
    )
    application.state.settings = cfg
    application.state.assembler = assembler
    application.state.southbound = southbound
    application.state.pipeline = pipeline

    async def _assemble_request(req: Any, request: Request):
        options = req.resolved_wahs_options()
        identity = _resolve_caller_identity(request, options, cfg)
        requested_model = (req.model or cfg.default_model).strip()
        force_refresh = "no-cache" in request.headers.get("cache-control", "").lower()
        conv_msgs = [{"role": m.role, "content": m.content} for m in (getattr(req, "messages", None) or ())]
        ctx = await assembler.assemble(
            options=options,
            identity=identity,
            requested_model=requested_model,
            conversation_messages=conv_msgs,
            requested_max_tokens=getattr(req, "max_tokens", None),
            force_refresh=force_refresh,
        )
        return options, requested_model, ctx

    @application.get("/health")
    @application.get("/healthz")
    async def healthcheck() -> Dict[str, Any]:
        reg = load_profile_registry(cfg.profiles_path or None)
        default_prof = (cfg.default_profile or reg.default_profile).strip() or reg.default_profile
        return {
            "status": "ok", "service": "wahs-engine", "version": cfg.engine_version,
            "fhir_auth_base_url": cfg.fhir_auth_base_url, "llm_base_url": cfg.llm_base_url,
            "default_model": cfg.default_model, "default_profile": default_prof,
            "enabled_enrichers": list(cfg.enabled_enrichers),
            "allow_local_bundle_mode": cfg.allow_local_bundle_mode,
        }

    @application.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return await southbound.list_models()

    @application.get("/v1/wahs/profiles")
    async def list_wahs_profiles() -> Dict[str, Any]:
        try:
            return serialize_profiles_catalog(cfg)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/v1/wahs/context/preview")
    async def preview_context(req: WahsContextPreviewRequest, request: Request, response: Response) -> Dict[str, Any]:
        options, requested_model, ctx = await _assemble_request(req, request)
        spec = cfg.resolve_model_spec(requested_model)
        enrichers_meta = []
        enrichment_tokens = 0
        enrichment_text = ""
        active_ids = ctx.active_enrichers or pipeline.enabled_ids
        if active_ids:
            ctx = copy.copy(ctx)
            ctx.citation_map = copy.deepcopy(ctx.citation_map)
            turns = normalize_conversation_turns(req.messages)
            last_q = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
            current_toks = count_chat_messages_tokens([{"role": "system", "content": ctx.system_prompt}, *turns], tokenizer_family=spec.tokenizer_family)
            headroom = max(0, int(spec.context_window) - int(spec.default_max_output_tokens) - current_toks)
            p_res = await asyncio.to_thread(
                pipeline.run, extracted=ctx.extracted, condensed=ctx.condensed, citation_map=ctx.citation_map,
                question=last_q, conversation=turns, headroom_tokens=headroom,
                token_counter=get_exact_token_counter(spec.tokenizer_family),
                active_ids=active_ids, max_enrichment_tokens=cfg.max_enrichment_tokens,
            )
            enrichers_meta = p_res.disclosures
            enrichment_tokens = p_res.used_tokens
            enrichment_text = p_res.combined_block()
        response.headers["X-WAHS-Prefix-Hash"] = ctx.prefix_cache_key_hash
        return {
            "patient_id": ctx.patient_id, "patient_name": ctx.patient_name, "birth_date": ctx.birth_date,
            "prefix_cache_key_hash": ctx.prefix_cache_key_hash, "security_audit": ctx.security_audit,
            "chart_text": ctx.condensed.chart_text,
            "x_wahs": build_x_wahs_envelope(
                ctx=ctx, requested_model=requested_model, upstream_model=spec.upstream_model,
                options=options, engine_version=cfg.engine_version, enrichers=enrichers_meta,
                enrichment_tokens=enrichment_tokens, enrichment_text=enrichment_text,
            ),
        }

    @application.post("/v1/chat/completions")
    @application.post("/v1/wahs/chat/completions")
    async def chat_completions(req: OpenAIChatCompletionRequest, request: Request, response: Response) -> Any:
        t0 = time.perf_counter()
        options, _requested_model, ctx = await _assemble_request(req, request)
        assemble_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if req.stream:
            sse_iter = await southbound.start_stream(ctx=ctx, req=req, options=options, assemble_ms=assemble_ms)
            return StreamingResponse(
                sse_iter, media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-WAHS-Prefix-Hash": ctx.prefix_cache_key_hash},
            )
        result = await southbound.complete(ctx=ctx, req=req, options=options, assemble_ms=assemble_ms)
        response.headers["X-WAHS-Prefix-Hash"] = ctx.prefix_cache_key_hash
        return result

    return application


app = create_app()


def main() -> None:
    uvicorn.run("wahs_engine.app:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
