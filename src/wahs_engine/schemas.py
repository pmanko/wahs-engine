"""OpenAI-compatible request/response schemas + `wahs` chart scoping contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    """Optional `wahs` extension envelope on `/v1/chat/completions`."""

    model_config = ConfigDict(extra="forbid")

    target_budget_tokens: int = Field(default=8192, ge=512, le=131072)
    sharing_posture: Literal["patient_self", "external_share"] = "patient_self"
    chart_scope: ChartScope = Field(default_factory=ChartScope)
    overlay_resources: List[Dict[str, Any]] = Field(default_factory=list)
    master_patient_id: Optional[str] = None
    user_sub: Optional[str] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    enable_kb: bool = True
    enable_overflow_tools: bool = True


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
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=16384)
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    user: Optional[str] = None
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self) -> WahsRequestOptions:
        if self.wahs is not None:
            return self.wahs
        extra = self.model_extra or {}
        extra_body = extra.get("extra_body")
        if isinstance(extra_body, dict) and isinstance(extra_body.get("wahs"), dict):
            return WahsRequestOptions.model_validate(extra_body["wahs"])
        return WahsRequestOptions()


class WahsContextPreviewRequest(BaseModel):
    """Request body for `/v1/wahs/context/preview` (allows empty/omitted `messages`)."""

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    wahs: Optional[WahsRequestOptions] = None

    def resolved_wahs_options(self) -> WahsRequestOptions:
        if self.wahs is not None:
            return self.wahs
        extra = self.model_extra or {}
        extra_body = extra.get("extra_body")
        if isinstance(extra_body, dict) and isinstance(extra_body.get("wahs"), dict):
            return WahsRequestOptions.model_validate(extra_body["wahs"])
        return WahsRequestOptions()
