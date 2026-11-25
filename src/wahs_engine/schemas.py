"""OpenAI-compatible request/response schemas + `wahs` chart scoping contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .config import parse_wahs_profile_spec


class ChartScope(BaseModel):
    """Declarative patient chart scoping controls (`wahs.chart_scope`)."""

    model_config = ConfigDict(extra="forbid")

    include_sources: List[str] = Field(default_factory=list)
    exclude_sources: List[str] = Field(default_factory=list)
    include_categories: List[str] = Field(default_factory=list)
    exclude_categories: List[str] = Field(default_factory=list)
    include_resource_types: List[str] = Field(default_factory=list)
    exclude_resource_types: List[str] = Field(default_factory=list)
    include_resource_ids: List[str] = Field(default_factory=list)
    exclude_resource_ids: List[str] = Field(default_factory=list)
    ips_bundle_id: Optional[str] = None
    date_from: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class WahsRequestOptions(BaseModel):
    """Optional `wahs` extension envelope on `/v1/chat/completions`."""

    model_config = ConfigDict(extra="forbid")

    target_budget_tokens: Optional[int] = Field(default=None, ge=512)
    sharing_posture: Literal["patient_self", "external_share"] = "patient_self"
    chart_scope: ChartScope = Field(default_factory=ChartScope)
    overlay_resources: List[Dict[str, Any]] = Field(default_factory=list)
    master_patient_id: Optional[str] = None
    user_sub: Optional[str] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    enable_kb: bool = True
    enable_temporal_digest: bool = Field(
        default=True,
        description=(
            "Inject deterministic chart-derived temporal anchor, encounter horizon, "
            "and longitudinal numeric series into Layer-3 enrichment (disabled in chart_only baseline)."
        ),
    )
    enable_overflow_tools: bool = True
    refresh: bool = False
    allow_partial_external_share: bool = False
    retrieval_strategy: Literal[
        "condensation_kb",
        "note_rag",
        "dynamic_fhir_tools",
        "chart_only",
    ] = "condensation_kb"


def apply_model_profile_defaults(
    options: WahsRequestOptions,
    requested_model: Optional[str],
) -> WahsRequestOptions:
    """Apply inferred retrieval strategy and `@<budget>` defaults from `model` unless explicitly overridden."""
    parsed = parse_wahs_profile_spec(requested_model)
    fields_set = set(options.model_fields_set or ())
    updates: Dict[str, Any] = {}

    effective_strategy = options.retrieval_strategy
    if parsed.inferred_strategy is not None and "retrieval_strategy" not in fields_set:
        effective_strategy = parsed.inferred_strategy  # type: ignore[assignment]
        updates["retrieval_strategy"] = effective_strategy

    if effective_strategy == "chart_only":
        for flag in ("enable_kb", "enable_temporal_digest", "enable_overflow_tools"):
            if flag not in fields_set:
                updates[flag] = False

    if parsed.explicit_budget_tokens is not None and "target_budget_tokens" not in fields_set:
        updates["target_budget_tokens"] = parsed.explicit_budget_tokens

    if not updates:
        return options
    return options.model_copy(update=updates)


def _resolve_wahs_options(
    wahs: Optional[WahsRequestOptions],
    model_extra: Optional[Dict[str, Any]],
    requested_model: Optional[str],
) -> WahsRequestOptions:
    if wahs is not None:
        base = wahs
    else:
        extra = model_extra or {}
        extra_body = extra.get("extra_body")
        if isinstance(extra_body, dict) and isinstance(extra_body.get("wahs"), dict):
            base = WahsRequestOptions.model_validate(extra_body["wahs"])
        else:
            base = WahsRequestOptions()
    return apply_model_profile_defaults(base, requested_model)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class OpenAIChatCompletionRequest(BaseModel):
    """Standard OpenAI `/v1/chat/completions` body with optional `wahs` scoping."""

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Any] = None
    user: Optional[str] = None
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self, default_model: Optional[str] = None) -> WahsRequestOptions:
        return _resolve_wahs_options(self.wahs, self.model_extra, self.model or default_model)


class WahsContextPreviewRequest(BaseModel):
    """Request body for `/v1/wahs/context/preview` (allows empty/omitted `messages`)."""

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self, default_model: Optional[str] = None) -> WahsRequestOptions:
        return _resolve_wahs_options(self.wahs, self.model_extra, self.model or default_model)
