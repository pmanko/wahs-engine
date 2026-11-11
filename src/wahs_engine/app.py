"""FastAPI application for standalone `wahs-engine`."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from .assembler import WAHSBundleAssembler
from .config import WahsEngineSettings
from .jwt_claims import VerifiedCallerIdentity, decode_and_bind_jwt, extract_bearer_token
from .schemas import OpenAIChatCompletionRequest, WahsRequestOptions
from .upstream import SouthboundOpenAIClient, build_x_wahs_envelope


def _resolve_caller_identity(
    request: Request,
    options: WahsRequestOptions,
    settings: WahsEngineSettings,
) -> Optional[VerifiedCallerIdentity]:
    raw_token = extract_bearer_token(request.headers)
    if not raw_token:
        if settings.allow_local_bundle_mode and options.fhir_bundle is not None:
            return None
        # Trigger standard 401 via decode_and_bind_jwt
        return decode_and_bind_jwt("")

    trusted_sub = (
        request.headers.get(settings.trusted_sub_header)
        if settings.trusted_sub_header
        else None
    )
    return decode_and_bind_jwt(
        raw_token,
        hs256_secret=settings.jwt_hs256_secret,
        expected_user_sub=options.user_sub,
        trusted_header_sub=trusted_sub,
    )


def create_app(
    settings: Optional[WahsEngineSettings] = None,
    *,
    fhir_http_client: Optional[httpx.AsyncClient] = None,
    llm_http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    cfg = settings or WahsEngineSettings.from_env()
    assembler = WAHSBundleAssembler(cfg, http_client=fhir_http_client)
    southbound = SouthboundOpenAIClient(cfg, http_client=llm_http_client)

    application = FastAPI(
        title="WAHS Engine",
        version=cfg.engine_version,
        description=(
            "Zero-privilege OpenAI-compatible clinical FHIR R4 condensation, "
            "fhir-auth multi-dataset assembly, and [N] provenance middleware."
        ),
    )
    application.state.settings = cfg
    application.state.assembler = assembler
    application.state.southbound = southbound

    @application.get("/health")
    @application.get("/healthz")
    async def healthcheck() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "wahs-engine",
            "version": cfg.engine_version,
            "fhir_auth_base_url": cfg.fhir_auth_base_url,
            "llm_base_url": cfg.llm_base_url,
            "default_model": cfg.default_model,
            "allow_local_bundle_mode": cfg.allow_local_bundle_mode,
        }

    @application.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return await southbound.list_models()

    @application.post("/v1/wahs/context/preview")
    async def preview_context(
        req: OpenAIChatCompletionRequest,
        request: Request,
    ) -> Dict[str, Any]:
        options = req.resolved_wahs_options()
        identity = _resolve_caller_identity(request, options, cfg)
        ctx = await assembler.assemble(options=options, identity=identity)
        requested_model = (req.model or cfg.default_model).strip()
        return {
            "patient_id": ctx.patient_id,
            "patient_name": ctx.patient_name,
            "birth_date": ctx.birth_date,
            "prefix_cache_key_hash": ctx.prefix_cache_key_hash,
            "security_audit": ctx.security_audit,
            "chart_text": ctx.condensed.chart_text,
            "x_wahs": build_x_wahs_envelope(
                ctx=ctx,
                requested_model=requested_model,
                upstream_model=cfg.resolve_upstream_model(requested_model),
                options=options,
                engine_version=cfg.engine_version,
            ),
        }

    @application.post("/v1/chat/completions")
    @application.post("/v1/wahs/chat/completions")
    async def chat_completions(
        req: OpenAIChatCompletionRequest,
        request: Request,
        response: Response,
    ) -> Any:
        options = req.resolved_wahs_options()
        identity = _resolve_caller_identity(request, options, cfg)
        ctx = await assembler.assemble(options=options, identity=identity)

        if req.stream:
            return StreamingResponse(
                southbound.stream(ctx=ctx, req=req, options=options),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-WAHS-Prefix-Hash": ctx.prefix_cache_key_hash,
                },
            )

        result = await southbound.complete(ctx=ctx, req=req, options=options)
        response.headers["X-WAHS-Prefix-Hash"] = ctx.prefix_cache_key_hash
        return result

    return application


app = create_app()


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("wahs_engine.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
