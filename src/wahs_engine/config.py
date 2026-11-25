"""Environment-driven configuration for standalone `wahs-engine`."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit


MAX_EXPLICIT_BUDGET_TOKENS = 262_144
_BUDGET_SUFFIX_RE = re.compile(r"^(?P<base>.+?)@(?P<num>\d+)(?P<k>[kK])?$")

_HARNESS_ARM_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("wahs-engine-rag:", "note_rag"),
    ("wahs-engine-kb:", "condensation_kb"),
    ("wahs-engine-tools:", "dynamic_fhir_tools"),
    ("wahs-engine-chart:", "chart_only"),
    ("wahs-engine:", "condensation_kb"),
)

_COMPACT_PROFILE_SUFFIXES: Tuple[Tuple[str, str, str], ...] = (
    ("-note-rag", "note_rag", "-kb"),
    ("-rag", "note_rag", "-kb"),
    ("-tools", "dynamic_fhir_tools", "-kb"),
    ("-chart", "chart_only", "-kb"),
    ("-raw", "chart_only", "-kb"),
    ("-kb", "condensation_kb", "-kb"),
)

_CROSS_PROVIDER_STEM_TO_PROFILE_KEY: Dict[str, str] = {
    "gemma-26b": "wahs-26b-kb",
    "gemma4:26b": "wahs-26b-kb",
    "ollama/gemma4:26b": "wahs-26b-kb",
    "gemma-4-12b": "wahs-12b-kb",
    "gemma-4-12b-q4": "wahs-12b-kb",
    "gemma4:12b": "wahs-12b-kb",
    "ollama/gemma4:12b": "wahs-12b-kb",
    "gemma-e4b": "wahs-e4b-kb",
    "gemma-4-e4b": "wahs-e4b-kb",
    "gemma-e4b-q8": "wahs-e4b-kb",
    "gemma4:e4b": "wahs-e4b-kb",
    "ollama/gemma4:e4b": "wahs-e4b-kb",
    "qwen3-14b": "wahs-qwen3-14b-kb",
    "qwen3:14b": "wahs-qwen3-14b-kb",
    "ollama/qwen3:14b": "wahs-qwen3-14b-kb",
    "qwen3.5-9b": "wahs-qwen3-8b-kb",
    "qwen3-8b": "wahs-qwen3-8b-kb",
    "qwen3:8b": "wahs-qwen3-8b-kb",
    "ollama/qwen3:8b": "wahs-qwen3-8b-kb",
}


@dataclass(frozen=True)
class ParsedProfileSpec:
    """Normalized strategy, model lookup key, and optional `@<budget>` extracted from a model ID."""

    raw_model: str
    clean_model: str
    lookup_key: str
    inferred_strategy: Optional[str] = None
    explicit_budget_tokens: Optional[int] = None
    had_arm_prefix: bool = False


def parse_wahs_profile_spec(requested_model: Optional[str]) -> ParsedProfileSpec:
    """Parse harness eval arm IDs (`wahs-engine-rag:<m>@16k`) or compact UI profiles (`wahs-26b-rag@32k`)."""
    raw = str(requested_model or "").strip()
    clean = raw
    explicit_budget: Optional[int] = None

    m_budget = _BUDGET_SUFFIX_RE.match(raw)
    if m_budget:
        clean = m_budget.group("base").strip()
        num_val = int(m_budget.group("num"))
        if m_budget.group("k"):
            num_val *= 1024
        explicit_budget = min(max(512, num_val), MAX_EXPLICIT_BUDGET_TOKENS)

    low_clean = clean.lower()
    for pfx, strat in _HARNESS_ARM_PREFIXES:
        if low_clean.startswith(pfx):
            rest = clean[len(pfx) :].strip()
            return ParsedProfileSpec(
                raw_model=raw,
                clean_model=clean,
                lookup_key=rest or clean,
                inferred_strategy=strat,
                explicit_budget_tokens=explicit_budget,
                had_arm_prefix=True,
            )

    if low_clean.startswith("wahs-") and not low_clean.startswith("wahs-engine"):
        for sfx, strat, target_sfx in _COMPACT_PROFILE_SUFFIXES:
            if low_clean.endswith(sfx) and len(clean) > len(sfx):
                stem = clean[: len(clean) - len(sfx)]
                return ParsedProfileSpec(
                    raw_model=raw,
                    clean_model=clean,
                    lookup_key=f"{stem}{target_sfx}",
                    inferred_strategy=strat,
                    explicit_budget_tokens=explicit_budget,
                    had_arm_prefix=False,
                )

    return ParsedProfileSpec(
        raw_model=raw,
        clean_model=clean,
        lookup_key=clean,
        inferred_strategy=None,
        explicit_budget_tokens=explicit_budget,
        had_arm_prefix=False,
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _infer_tokenizer_family(requested_model: str, upstream_model: str) -> str:
    for candidate in ((requested_model or "").lower(), (upstream_model or "").lower()):
        if any(
            tok in candidate
            for tok in ("qwen", "llama", "mistral", "deepseek", "gpt-", "claude", "o1", "o3", "o4")
        ):
            return "o200k_base"
        if any(
            tok in candidate
            for tok in ("gemma", "medgemma", "wahs-26b", "wahs-12b", "wahs-e4b")
        ):
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
        "wahs-26b-rag",
        "wahs-12b-rag",
        "wahs-e4b-rag",
        "wahs-12b-kb",
        "wahs-e4b-kb",
        "wahs-26b-kb",
        "wahs-26b-tools",
        "wahs-12b-tools",
        "wahs-e4b-tools",
        "wahs-26b-chart",
        "wahs-12b-chart",
        "wahs-e4b-chart",
        "gemma4:12b",
        "gemma4:e4b",
    )
    model_map: Dict[str, str] = field(default_factory=dict)
    model_specs: Dict[str, ModelSpec] = field(default_factory=dict)
    default_context_window: int = 32768
    max_context_window: int = 131072
    default_chart_budget_ceiling: Optional[int] = 16384
    default_provider_flavor: str = "litellm"
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
    engine_version: str = "0.1.0"

    def __post_init__(self) -> None:
        raw_url = (self.fhir_auth_base_url or "").strip()
        lowered = raw_url.lower()
        parsed_url = urlsplit(raw_url)
        host = (parsed_url.hostname or "").lower()
        forbidden_hosts = {"fhir-internal", "hapi-fhir-internal", "fhir-internal.local"}
        if "fhir-internal" in lowered or host in forbidden_hosts:
            raise ValueError(
                "Security invariant violation: wahs-engine must connect through "
                "zero-privilege fhir-auth (jwt-proxy), never directly to fhir-internal."
            )

    def resolve_upstream_model(self, requested_model: Optional[str]) -> str:
        model = (requested_model or self.default_model).strip() or self.default_model
        parsed = parse_wahs_profile_spec(model)
        for key in (model, parsed.clean_model, parsed.lookup_key):
            if key and key in self.model_map:
                return self.model_map[key]
        cross_key = _CROSS_PROVIDER_STEM_TO_PROFILE_KEY.get(parsed.lookup_key.lower())
        if cross_key and cross_key in self.model_map:
            return self.model_map[cross_key]
        if parsed.had_arm_prefix and parsed.lookup_key:
            return parsed.lookup_key
        return parsed.clean_model or model

    def resolve_model_spec(self, requested_model: Optional[str]) -> ModelSpec:
        model = (requested_model or self.default_model).strip() or self.default_model
        parsed = parse_wahs_profile_spec(model)
        upstream = self.resolve_upstream_model(model)

        cross_key = _CROSS_PROVIDER_STEM_TO_PROFILE_KEY.get(parsed.lookup_key.lower())
        base_spec: Optional[ModelSpec] = None
        for key in (model, parsed.clean_model, parsed.lookup_key, cross_key, upstream):
            if key and key in self.model_specs:
                base_spec = self.model_specs[key]
                break

        if base_spec is None:
            base_spec = ModelSpec(
                upstream_model=upstream,
                context_window=self.default_context_window,
                chart_budget_ceiling=self.default_chart_budget_ceiling,
                default_max_output_tokens=1024,
                tokenizer_family=_infer_tokenizer_family(parsed.lookup_key or model, upstream),
                provider_flavor=_infer_provider_flavor(upstream, self.default_provider_flavor),
                disable_thinking=True,
            )

        if parsed.explicit_budget_tokens is not None:
            max_win = max(int(base_spec.context_window), int(self.max_context_window))
            req_b = min(int(parsed.explicit_budget_tokens), max(512, max_win // 2))
            eff_ctx = min(max_win, max(int(base_spec.context_window), req_b * 2))
            eff_ceiling = (
                max(int(base_spec.chart_budget_ceiling), req_b)
                if base_spec.chart_budget_ceiling is not None
                else None
            )
            return replace(
                base_spec,
                context_window=eff_ctx,
                chart_budget_ceiling=eff_ceiling,
            )
        return base_spec

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
                "wahs-26b-rag",
                "wahs-12b-rag",
                "wahs-e4b-rag",
                "wahs-12b-kb",
                "wahs-e4b-kb",
                "wahs-26b-kb",
                "wahs-26b-tools",
                "wahs-12b-tools",
                "wahs-e4b-tools",
                "wahs-26b-chart",
                "wahs-12b-chart",
                "wahs-e4b-chart",
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
        max_ctx = max(default_ctx, int(os.environ.get("WAHS_MAX_CONTEXT_WINDOW", "131072")))
        raw_ceil = os.environ.get("WAHS_CHART_BUDGET_CEILING")
        if raw_ceil is None:
            default_ceil: Optional[int] = 16384
        elif raw_ceil.strip().lower() in {"", "none", "null"}:
            default_ceil = None
        else:
            default_ceil = int(raw_ceil)
        raw_flavor = (os.environ.get("WAHS_PROVIDER_FLAVOR") or "").strip().lower()
        if raw_flavor:
            default_flavor = raw_flavor
        elif ":8077" in llm_url:
            default_flavor = "llamacpp"
        else:
            default_flavor = "litellm"

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
            max_context_window=max_ctx,
            default_chart_budget_ceiling=default_ceil,
            default_provider_flavor=default_flavor,
            allow_local_bundle_mode=_env_bool("WAHS_ALLOW_LOCAL_BUNDLE_MODE", False),
            jwt_hs256_secret=os.environ.get("WAHS_JWT_HS256_SECRET", "").strip(),
            jwt_jwks_url=os.environ.get("WAHS_JWT_JWKS_URL", "").strip(),
            jwt_jwks_json=os.environ.get("WAHS_JWT_JWKS_JSON", "").strip(),
            jwt_issuer=os.environ.get("WAHS_JWT_ISSUER", "").strip(),
            jwt_audience=os.environ.get("WAHS_JWT_AUDIENCE", "").strip(),
            trusted_sub_header=os.environ.get("WAHS_TRUSTED_SUB_HEADER", "").strip(),
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
