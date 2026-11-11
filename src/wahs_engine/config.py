"""Environment-driven configuration for standalone `wahs-engine`."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    allow_local_bundle_mode: bool = False
    jwt_hs256_secret: str = ""
    trusted_sub_header: str = ""
    max_overlay_bytes: int = 5120
    max_everything_pages: int = 20
    everything_page_size: int = 1000
    dataset_cache_max_entries: int = 256
    prefix_cache_max_entries: int = 128
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

    def resolve_upstream_model(self, requested_model: str) -> str:
        model = (requested_model or self.default_model).strip() or self.default_model
        return self.model_map.get(model, model)

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

        return cls(
            fhir_auth_base_url=fhir_url,
            llm_base_url=llm_url,
            llm_api_key=llm_key,
            default_model=default_model,
            available_models=avail,
            model_map=parsed_map,
            allow_local_bundle_mode=_env_bool("WAHS_ALLOW_LOCAL_BUNDLE_MODE", False),
            jwt_hs256_secret=os.environ.get("WAHS_JWT_HS256_SECRET", "").strip(),
            trusted_sub_header=os.environ.get("WAHS_TRUSTED_SUB_HEADER", "").strip(),
            max_overlay_bytes=int(os.environ.get("WAHS_MAX_OVERLAY_BYTES", "5120")),
            max_everything_pages=int(os.environ.get("WAHS_MAX_EVERYTHING_PAGES", "20")),
            everything_page_size=int(os.environ.get("WAHS_EVERYTHING_PAGE_SIZE", "1000")),
            dataset_cache_max_entries=int(os.environ.get("WAHS_DATASET_CACHE_MAX", "256")),
            prefix_cache_max_entries=int(os.environ.get("WAHS_PREFIX_CACHE_MAX", "128")),
            fhir_timeout_seconds=float(os.environ.get("WAHS_FHIR_TIMEOUT_SECONDS", "25.0")),
            llm_timeout_seconds=float(os.environ.get("WAHS_LLM_TIMEOUT_SECONDS", "120.0")),
        )
