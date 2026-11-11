"""Standalone OpenAI-compatible WAHS AI middleware (`wahs_engine`)."""

from __future__ import annotations

from .app import app, create_app
from .assembler import AssembledWahsContext, WAHSBundleAssembler, enforce_single_patient_scope
from .config import WahsEngineSettings
from .jwt_claims import VerifiedCallerIdentity, decode_and_bind_jwt, extract_bearer_token
from .schemas import ChartScope, OpenAIChatCompletionRequest, WahsRequestOptions
from .upstream import SouthboundOpenAIClient, build_x_wahs_envelope

__all__ = [
    "AssembledWahsContext",
    "ChartScope",
    "OpenAIChatCompletionRequest",
    "SouthboundOpenAIClient",
    "VerifiedCallerIdentity",
    "WAHSBundleAssembler",
    "WahsEngineSettings",
    "WahsRequestOptions",
    "app",
    "build_x_wahs_envelope",
    "create_app",
    "decode_and_bind_jwt",
    "enforce_single_patient_scope",
    "extract_bearer_token",
]
