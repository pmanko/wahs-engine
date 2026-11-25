"""OpenAI-compatible request schemas and pure `wahs` chart scoping contract."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    """Pure baseline `wahs` request options (zero enricher/pathway/privacy toggles on wire)."""

    model_config = ConfigDict(extra="ignore")

    master_patient_id: Optional[str] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    chart_scope: ChartScope = Field(default_factory=ChartScope)
    overlay_resources: List[Dict[str, Any]] = Field(default_factory=list)
    target_budget_tokens: Optional[int] = Field(default=None, ge=512, le=262144)
    refresh: bool = False
    emit_stream_events: bool = True


def _resolve_wahs_options(
    wahs: Optional[WahsRequestOptions],
    model_extra: Optional[Dict[str, Any]],
) -> WahsRequestOptions:
    if wahs is not None:
        return wahs
    extra_body = (model_extra or {}).get("extra_body")
    if isinstance(extra_body, dict) and isinstance(extra_body.get("wahs"), dict):
        return WahsRequestOptions.model_validate(extra_body["wahs"])
    return WahsRequestOptions()


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
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    seed: Optional[int] = None
    stop: Optional[Any] = None
    user: Optional[str] = None
    stream_options: Optional[Dict[str, Any]] = None
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self) -> WahsRequestOptions:
        return _resolve_wahs_options(self.wahs, self.model_extra)


class WahsContextPreviewRequest(BaseModel):
    """Request body for `/v1/wahs/context/preview`."""

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self) -> WahsRequestOptions:
        return _resolve_wahs_options(self.wahs, self.model_extra)
