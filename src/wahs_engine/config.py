"""Environment-driven configuration for standalone `wahs-engine`."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, Optional, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _infer_tokenizer_family(requested_model: str, upstream_model: str) -> str:
    combined = f"{requested_model} {upstream_model}".lower()
    if any(tok in combined for tok in ("gemma", "medgemma", "wahs-")):
        return "gemma4"
    return "o200k_base"


def _infer_provider_flavor(upstream_model: str, default_flavor: str) -> str:
    low = (upstream_model or "").strip().lower()
    if low.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return (default_flavor or "litellm").strip().lower() or "litellm"


@dataclass(frozen=True)
class ModelSpec:
    """Explicit southbound model capabilities and framing parameters."""

    upstream_model: str
    context_window: int = 32768
    chart_budget_ceiling: Optional[int] = 16384
    default_max_output_tokens: int = 1024
    tokenizer_family: str = "gemma4"
    provider_flavor: str = "litellm"
    disable_thinking: bool = True


@dataclass(frozen=True)
class WahsEngineSettings:
    """Immutable runtime settings for `wahs-engine`."""

    fhir_auth_base_url: str = "http://fhir-auth:8080/fhir"
    llm_base_url: str = "http://host.docker.internal:8079/v1"
    llm_api_key: str = ""
    default_model: str = "gemma4:12b"
    available_models: Tuple[str, ...] = (
        "wahs-12b-kb",
        "wahs-e4b-kb",
        "wahs-26b-kb",
        "gemma4:12b",
        "gemma4:e4b",
    )
    model_map: Dict[str, str] = field(default_factory=dict)
    model_specs: Dict[str, ModelSpec] = field(default_factory=dict)
    default_context_window: int = 32768
    default_chart_budget_ceiling: Optional[int] = 16384
    default_provider_flavor: str = "litellm"
    allow_local_bundle_mode: bool = False
    jwt_hs256_secret: str = ""
    trusted_sub_header: str = ""
    max_overlay_bytes: int = 5120
    max_everything_pages: int = 20
    everything_page_size: int = 1000
    max_everything_resources: int = 20000
    everything_concurrency: int = 4
    assembly_timeout_seconds: float = 45.0
    dataset_cache_max_entries: int = 256
    prefix_cache_max_entries: int = 128
    cache_ttl_seconds: float = 1800.0
    fhir_timeout_seconds: float = 25.0
    llm_timeout_seconds: float = 120.0
    engine_version: str = "0.1.0"

    def __post_init__(self) -> None:
        lowered = (self.fhir_auth_base_url or "").lower()
        if "fhir-internal" in lowered:
            raise ValueError(
                "Security invariant violation: wahs-engine must connect through "
                "zero-privilege fhir-auth (jwt-proxy), never directly to fhir-internal."
            )

    def resolve_upstream_model(self, requested_model: Optional[str]) -> str:
        model = (requested_model or self.default_model).strip() or self.default_model
        return self.model_map.get(model, model)

    def resolve_model_spec(self, requested_model: Optional[str]) -> ModelSpec:
        model = (requested_model or self.default_model).strip() or self.default_model
        if model in self.model_specs:
            return self.model_specs[model]
        upstream = self.resolve_upstream_model(model)
        if upstream in self.model_specs:
            return self.model_specs[upstream]
        return ModelSpec(
            upstream_model=upstream,
            context_window=self.default_context_window,
            chart_budget_ceiling=self.default_chart_budget_ceiling,
            default_max_output_tokens=1024,
            tokenizer_family=_infer_tokenizer_family(model, upstream),
            provider_flavor=_infer_provider_flavor(upstream, self.default_provider_flavor),
            disable_thinking=True,
        )

    @property
    def upstream_chat_completions_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1") or base.endswith("/api"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @property
    def upstream_models_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        if base.endswith("/v1") or base.endswith("/api"):
            return f"{base}/models"
        return f"{base}/v1/models"

    @classmethod
    def from_env(cls) -> "WahsEngineSettings":
        fhir_url = (
            os.environ.get("WAHS_FHIR_AUTH_BASE_URL")
            or os.environ.get("FHIR_AUTH_BASE_URL")
            or "http://fhir-auth:8080/fhir"
        ).strip()
        llm_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://host.docker.internal:8079/v1"
        ).strip()
        llm_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        default_model = (os.environ.get("WAHS_DEFAULT_MODEL") or "gemma4:12b").strip()

        raw_models = os.environ.get("WAHS_AVAILABLE_MODELS", "").strip()
        if raw_models:
            avail = tuple(m.strip() for m in raw_models.split(",") if m.strip())
        else:
            avail = (
                "wahs-12b-kb",
                "wahs-e4b-kb",
                "wahs-26b-kb",
                default_model,
            )
            avail = tuple(dict.fromkeys(avail))

        raw_map = os.environ.get("WAHS_MODEL_MAP_JSON", "").strip()
        parsed_map: Dict[str, str] = {}
        if raw_map:
            loaded = json.loads(raw_map)
            if isinstance(loaded, dict):
                parsed_map = {str(k): str(v) for k, v in loaded.items()}

        default_ctx = int(os.environ.get("WAHS_CONTEXT_WINDOW", "32768"))
        raw_ceil = os.environ.get("WAHS_CHART_BUDGET_CEILING")
        if raw_ceil is None:
            default_ceil: Optional[int] = 16384
        elif raw_ceil.strip().lower() in {"", "none", "null"}:
            default_ceil = None
        else:
            default_ceil = int(raw_ceil)
        default_flavor = (os.environ.get("WAHS_PROVIDER_FLAVOR") or "litellm").strip().lower() or "litellm"

        raw_specs = os.environ.get("WAHS_MODEL_SPECS_JSON", "").strip()
        parsed_specs: Dict[str, ModelSpec] = {}
        if raw_specs:
            loaded_specs = json.loads(raw_specs)
            if isinstance(loaded_specs, dict):
                for alias, entry in loaded_specs.items():
                    if not isinstance(entry, dict):
                        continue
                    up = str(entry.get("upstream_model") or parsed_map.get(str(alias), str(alias)))
                    ctx_w = int(entry.get("context_window") or default_ctx)
                    if "chart_budget_ceiling" in entry:
                        c_raw = entry["chart_budget_ceiling"]
                        c_val = int(c_raw) if c_raw is not None else None
                    else:
                        c_val = default_ceil
                    parsed_specs[str(alias)] = ModelSpec(
                        upstream_model=up,
                        context_window=ctx_w,
                        chart_budget_ceiling=c_val,
                        default_max_output_tokens=int(entry.get("default_max_output_tokens") or 1024),
                        tokenizer_family=str(
                            entry.get("tokenizer_family")
                            or _infer_tokenizer_family(str(alias), up)
                        ),
                        provider_flavor=str(
                            entry.get("provider_flavor")
                            or _infer_provider_flavor(up, default_flavor)
                        ),
                        disable_thinking=bool(entry.get("disable_thinking", True)),
                    )

        return cls(
            fhir_auth_base_url=fhir_url,
            llm_base_url=llm_url,
            llm_api_key=llm_key,
            default_model=default_model,
            available_models=avail,
            model_map=parsed_map,
            model_specs=parsed_specs,
            default_context_window=default_ctx,
            default_chart_budget_ceiling=default_ceil,
            default_provider_flavor=default_flavor,
            allow_local_bundle_mode=_env_bool("WAHS_ALLOW_LOCAL_BUNDLE_MODE", False),
            jwt_hs256_secret=os.environ.get("WAHS_JWT_HS256_SECRET", "").strip(),
            trusted_sub_header=os.environ.get("WAHS_TRUSTED_SUB_HEADER", "").strip(),
            max_overlay_bytes=int(os.environ.get("WAHS_MAX_OVERLAY_BYTES", "5120")),
            max_everything_pages=int(os.environ.get("WAHS_MAX_EVERYTHING_PAGES", "20")),
            everything_page_size=int(os.environ.get("WAHS_EVERYTHING_PAGE_SIZE", "1000")),
            max_everything_resources=int(os.environ.get("WAHS_MAX_EVERYTHING_RESOURCES", "20000")),
            everything_concurrency=int(os.environ.get("WAHS_EVERYTHING_CONCURRENCY", "4")),
            assembly_timeout_seconds=float(os.environ.get("WAHS_ASSEMBLY_TIMEOUT_SECONDS", "45.0")),
            dataset_cache_max_entries=int(os.environ.get("WAHS_DATASET_CACHE_MAX", "256")),
            prefix_cache_max_entries=int(os.environ.get("WAHS_PREFIX_CACHE_MAX", "128")),
            cache_ttl_seconds=float(os.environ.get("WAHS_CACHE_TTL_SECONDS", "1800.0")),
            fhir_timeout_seconds=float(os.environ.get("WAHS_FHIR_TIMEOUT_SECONDS", "25.0")),
            llm_timeout_seconds=float(os.environ.get("WAHS_LLM_TIMEOUT_SECONDS", "120.0")),
        )
