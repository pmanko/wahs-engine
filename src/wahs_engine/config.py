"""Environment-driven configuration for slim baseline `wahs-engine`."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

DEFAULT_MODEL_ALIASES: Dict[str, str] = {
    "wahs-26b": "ollama/gemma4:26b",
    "wahs-12b": "ollama/gemma4:12b",
    "wahs-e4b": "ollama/gemma4:e4b",
}


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name)
        if val is not None and val.strip():
            return val.strip()
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _infer_tokenizer_family(requested_model: str, upstream_model: str) -> str:
    for raw_candidate in ((requested_model or "").lower(), (upstream_model or "").lower()):
        candidate = raw_candidate.strip()
        if not candidate:
            continue
        bare = candidate.rsplit("/", 1)[-1]
        if any(tok in bare for tok in ("gemma", "medgemma")) or any(
            tok in candidate for tok in ("wahs-26b", "wahs-12b", "wahs-e4b")
        ):
            return "gemma4"
        if any(
            tok in bare
            for tok in ("qwen", "llama", "hermes", "mistral", "deepseek", "gpt-", "claude", "o1", "o3", "o4")
        ):
            return "o200k_base"
    return "o200k_base"


def _infer_provider_flavor(upstream_model: str, default_flavor: str) -> str:
    if (upstream_model or "").strip().lower().startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return (default_flavor or "litellm").strip().lower() or "litellm"


@dataclass(frozen=True)
class ModelSpec:
    """Southbound model capability profile."""

    upstream_model: str
    context_window: int = 32768
    preferred_chart_tokens: Optional[int] = None
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
    default_model: str = "wahs-26b"
    available_models: Tuple[str, ...] = ("wahs-26b", "wahs-12b", "wahs-e4b")
    model_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))
    model_specs: Dict[str, ModelSpec] = field(default_factory=dict)
    default_context_window: int = 32768
    default_preferred_chart_tokens: Optional[int] = None
    default_provider_flavor: str = "litellm"
    default_profile: str = ""
    profiles_path: str = ""
    prompts_dir: str = ""
    enabled_enrichers: Tuple[str, ...] = ()
    allow_experimental_enrichers: bool = False
    max_enrichment_tokens: int = 2048
    allow_local_bundle_mode: bool = False
    jwt_hs256_secret: str = ""
    jwt_jwks_url: str = ""
    jwt_jwks_json: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = ""
    trusted_sub_header: str = ""
    max_overlay_bytes: int = 5120
    max_everything_pages: int = 20
    everything_page_size: int = 1000
    max_everything_resources: int = 20000
    everything_concurrency: int = 4
    assembly_timeout_seconds: float = 45.0
    dataset_cache_max_entries: int = 256
    dataset_cache_max_bytes: int = 268_435_456
    prefix_cache_max_entries: int = 128
    prefix_cache_max_bytes: int = 134_217_728
    cache_ttl_seconds: float = 1800.0
    fhir_timeout_seconds: float = 25.0
    llm_timeout_seconds: float = 120.0
    engine_version: str = "0.2.0"

    def __post_init__(self) -> None:
        raw_url = (self.fhir_auth_base_url or "").strip()
        host = (urlsplit(raw_url).hostname or "").lower()
        if "fhir-internal" in raw_url.lower() or host in {"fhir-internal", "hapi-fhir-internal", "fhir-internal.local"}:
            raise ValueError("Security invariant violation: wahs-engine must connect through zero-privilege fhir-auth (jwt-proxy), never directly to fhir-internal.")

    def resolve_upstream_model(self, requested_model: Optional[str]) -> str:
        model = (requested_model or self.default_model).strip() or self.default_model
        return self.model_map.get(model) or DEFAULT_MODEL_ALIASES.get(model, model)

    def resolve_model_spec(self, requested_model: Optional[str]) -> ModelSpec:
        model = (requested_model or self.default_model).strip() or self.default_model
        upstream = self.resolve_upstream_model(model)
        for key in (model, upstream):
            if key in self.model_specs:
                return self.model_specs[key]
        return ModelSpec(
            upstream_model=upstream,
            context_window=self.default_context_window,
            preferred_chart_tokens=self.default_preferred_chart_tokens,
            tokenizer_family=_infer_tokenizer_family(model, upstream),
            provider_flavor=_infer_provider_flavor(upstream, self.default_provider_flavor),
        )

    @property
    def upstream_chat_completions_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions" if (base.endswith("/v1") or base.endswith("/api")) else f"{base}/v1/chat/completions"

    @property
    def upstream_models_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        return f"{base}/models" if (base.endswith("/v1") or base.endswith("/api")) else f"{base}/v1/models"

    @classmethod
    def from_env(cls) -> "WahsEngineSettings":
        fhir_url = _env_str("WAHS_FHIR_AUTH_BASE_URL", "FHIR_AUTH_BASE_URL", default="http://fhir-auth:8080/fhir")
        llm_url = _env_str("LLM_BASE_URL", "OPENAI_BASE_URL", default="http://host.docker.internal:8079/v1")
        llm_key = _env_str("LLM_API_KEY", "OPENAI_API_KEY")
        default_model = _env_str("WAHS_DEFAULT_MODEL", default="wahs-26b")

        raw_models = _env_str("WAHS_AVAILABLE_MODELS")
        avail = tuple(m.strip() for m in raw_models.split(",") if m.strip()) if raw_models else tuple(dict.fromkeys(("wahs-26b", "wahs-12b", "wahs-e4b", default_model)))

        parsed_map = dict(DEFAULT_MODEL_ALIASES)
        if raw_map := _env_str("WAHS_MODEL_MAP_JSON"):
            loaded = json.loads(raw_map)
            if isinstance(loaded, dict):
                parsed_map.update({str(k): str(v) for k, v in loaded.items()})

        default_ctx = int(os.environ.get("WAHS_CONTEXT_WINDOW", "32768"))
        raw_pref = os.environ.get("WAHS_PREFERRED_CHART_TOKENS")
        default_pref: Optional[int] = (
            None
            if (raw_pref is None or raw_pref.strip().lower() in {"", "none", "null"})
            else int(raw_pref)
        )
        if default_pref is not None and default_pref <= 0:
            raise ValueError("WAHS_PREFERRED_CHART_TOKENS must be a positive integer when set.")
        default_flavor = _env_str("WAHS_PROVIDER_FLAVOR").lower() or ("llamacpp" if ":8077" in llm_url else "litellm")

        parsed_specs: Dict[str, ModelSpec] = {}
        if raw_specs := _env_str("WAHS_MODEL_SPECS_JSON"):
            loaded_specs = json.loads(raw_specs)
            if isinstance(loaded_specs, dict):
                for alias, entry in loaded_specs.items():
                    if not isinstance(entry, dict):
                        continue
                    up = str(entry.get("upstream_model") or parsed_map.get(str(alias), str(alias)))
                    p_val = (
                        (int(entry["preferred_chart_tokens"]) if entry["preferred_chart_tokens"] is not None else None)
                        if "preferred_chart_tokens" in entry
                        else default_pref
                    )
                    if p_val is not None and p_val <= 0:
                        raise ValueError(f"preferred_chart_tokens for '{alias}' must be a positive integer.")
                    parsed_specs[str(alias)] = ModelSpec(
                        upstream_model=up,
                        context_window=int(entry.get("context_window") or default_ctx),
                        preferred_chart_tokens=p_val,
                        default_max_output_tokens=int(entry.get("default_max_output_tokens") or 1024),
                        tokenizer_family=str(entry.get("tokenizer_family") or _infer_tokenizer_family(str(alias), up)),
                        provider_flavor=str(entry.get("provider_flavor") or _infer_provider_flavor(up, default_flavor)),
                        disable_thinking=bool(entry.get("disable_thinking", True)),
                    )

        raw_enrichers = _env_str("WAHS_ENABLED_ENRICHERS")
        enabled_enrichers = tuple(e.strip() for e in raw_enrichers.split(",") if e.strip()) if raw_enrichers else ()
        return cls(
            fhir_auth_base_url=fhir_url, llm_base_url=llm_url, llm_api_key=llm_key,
            default_model=default_model, available_models=avail, model_map=parsed_map, model_specs=parsed_specs,
            default_context_window=default_ctx, default_preferred_chart_tokens=default_pref,
            default_provider_flavor=default_flavor,
            default_profile=_env_str("WAHS_DEFAULT_PROFILE"),
            profiles_path=_env_str("WAHS_PROFILES_PATH"),
            prompts_dir=_env_str("WAHS_PROMPTS_DIR"),
            enabled_enrichers=enabled_enrichers,
            allow_experimental_enrichers=_env_bool("WAHS_ALLOW_EXPERIMENTAL_ENRICHERS", False),
            max_enrichment_tokens=max(64, int(os.environ.get("WAHS_MAX_ENRICHMENT_TOKENS", "2048"))),
            allow_local_bundle_mode=_env_bool("WAHS_ALLOW_LOCAL_BUNDLE_MODE", False),
            jwt_hs256_secret=_env_str("WAHS_JWT_HS256_SECRET"), jwt_jwks_url=_env_str("WAHS_JWT_JWKS_URL"),
            jwt_jwks_json=_env_str("WAHS_JWT_JWKS_JSON"), jwt_issuer=_env_str("WAHS_JWT_ISSUER"),
            jwt_audience=_env_str("WAHS_JWT_AUDIENCE"), trusted_sub_header=_env_str("WAHS_TRUSTED_SUB_HEADER"),
            max_overlay_bytes=int(os.environ.get("WAHS_MAX_OVERLAY_BYTES", "5120")),
            max_everything_pages=int(os.environ.get("WAHS_MAX_EVERYTHING_PAGES", "20")),
            everything_page_size=int(os.environ.get("WAHS_EVERYTHING_PAGE_SIZE", "1000")),
            max_everything_resources=int(os.environ.get("WAHS_MAX_EVERYTHING_RESOURCES", "20000")),
            everything_concurrency=int(os.environ.get("WAHS_EVERYTHING_CONCURRENCY", "4")),
            assembly_timeout_seconds=float(os.environ.get("WAHS_ASSEMBLY_TIMEOUT_SECONDS", "45.0")),
            dataset_cache_max_entries=int(os.environ.get("WAHS_DATASET_CACHE_MAX", "256")),
            dataset_cache_max_bytes=int(os.environ.get("WAHS_DATASET_CACHE_MAX_BYTES", "268435456")),
            prefix_cache_max_entries=int(os.environ.get("WAHS_PREFIX_CACHE_MAX", "128")),
            prefix_cache_max_bytes=int(os.environ.get("WAHS_PREFIX_CACHE_MAX_BYTES", "134217728")),
            cache_ttl_seconds=float(os.environ.get("WAHS_CACHE_TTL_SECONDS", "1800.0")),
            fhir_timeout_seconds=float(os.environ.get("WAHS_FHIR_TIMEOUT_SECONDS", "25.0")),
            llm_timeout_seconds=float(os.environ.get("WAHS_LLM_TIMEOUT_SECONDS", "120.0")),
        )
