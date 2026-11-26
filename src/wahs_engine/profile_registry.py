"""Declarative `profiles.yaml` registry and zero-import enricher/prompt resolver."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException
import yaml

from .config import WahsEngineSettings
from .prompt_loader import load_prompt
from .schemas import WahsRequestOptions

_DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "profiles.yaml"

CORE_GENERALIZABLE_ENRICHERS: Tuple[str, ...] = (
    "temporal",
    "terminology_kb",
    "note_rag",
    "overflow_lookup",
)
ALL_GENERALIZABLE_ENRICHERS: Tuple[str, ...] = (
    "temporal",
    "terminology_kb",
    "note_rag",
    "overflow_lookup",
    "guideline_kb",
)
KNOWN_GENERALIZABLE_ENRICHERS = frozenset(
    (*ALL_GENERALIZABLE_ENRICHERS, "temporal_trends")
)
KNOWN_EXPERIMENTAL_ENRICHERS = frozenset(
    ("experimental.clinical_reconciliation",)
)


@dataclass(frozen=True)
class ProfileDefinition:
    id: str
    label: str
    prompt: str
    enrichers: Tuple[str, ...] = ()
    preferred_chart_tokens: Optional[int] = None


@dataclass(frozen=True)
class ProfileRegistry:
    default_profile: str
    profiles: Dict[str, ProfileDefinition]


@dataclass(frozen=True)
class ResolvedProfileConfig:
    profile_id: str
    label: str
    prompt_id: str
    prompt_text: str
    prompt_sha256: str
    active_enrichers: Tuple[str, ...]
    preferred_chart_tokens: Optional[int]


_REGISTRY_CACHE: Dict[Tuple[str, int, int], ProfileRegistry] = {}


def resolve_profiles_path(profiles_path: Optional[Path | str] = None) -> Path:
    if profiles_path is not None and str(profiles_path).strip():
        return Path(str(profiles_path).strip()).expanduser().resolve()
    env_p = os.environ.get("WAHS_PROFILES_PATH", "").strip()
    if env_p:
        return Path(env_p).expanduser().resolve()
    return _DEFAULT_PROFILES_PATH


def load_profile_registry(profiles_path: Optional[Path | str] = None) -> ProfileRegistry:
    path = resolve_profiles_path(profiles_path)
    if not path.is_file():
        raise FileNotFoundError(f"WAHS profiles file not found: {path}")
    st = path.stat()
    cache_key = (str(path), st.st_mtime_ns, st.st_size)
    if cache_key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[cache_key]

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
        raise ValueError(f"Invalid WAHS profiles YAML structure in {path}")

    profiles_out: Dict[str, ProfileDefinition] = {}
    for pid_raw, entry in raw["profiles"].items():
        if not isinstance(entry, dict):
            continue
        pid = str(pid_raw).strip()
        label = str(entry.get("label") or pid).strip()
        prompt = str(entry.get("prompt") or "wahs-default").strip()
        raw_enr = entry.get("enrichers") or []
        if isinstance(raw_enr, str):
            enr_list = [e.strip() for e in raw_enr.split(",") if e.strip()]
        else:
            enr_list = [str(e).strip() for e in raw_enr if str(e).strip()]
        pref_raw = entry.get("preferred_chart_tokens")
        pref_toks = int(pref_raw) if pref_raw is not None else None
        if pref_toks is not None and pref_toks <= 0:
            raise ValueError(f"preferred_chart_tokens for profile {pid!r} must be positive or null")
        profiles_out[pid] = ProfileDefinition(
            id=pid,
            label=label,
            prompt=prompt,
            enrichers=tuple(enr_list),
            preferred_chart_tokens=pref_toks,
        )

    default_prof = str(raw.get("default_profile") or "baseline").strip()
    if default_prof not in profiles_out:
        raise ValueError(f"default_profile {default_prof!r} missing from {path}")

    reg = ProfileRegistry(default_profile=default_prof, profiles=profiles_out)
    if len(_REGISTRY_CACHE) >= 8:
        _REGISTRY_CACHE.clear()
    _REGISTRY_CACHE[cache_key] = reg
    return reg


def expand_and_validate_enrichers(
    raw_ids: Sequence[str],
    *,
    allow_experimental: bool = False,
) -> Tuple[str, ...]:
    """Expand preset groups and validate enricher IDs without importing `wahs_extensions`."""
    expanded: List[str] = []
    for raw in raw_ids or ():
        for part in str(raw or "").split(","):
            eid = part.strip()
            if not eid:
                continue
            if eid == "core_generalizable":
                expanded.extend(CORE_GENERALIZABLE_ENRICHERS)
            elif eid == "all_generalizable":
                expanded.extend(ALL_GENERALIZABLE_ENRICHERS)
            elif eid.startswith("experimental."):
                if not allow_experimental:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Experimental enricher {eid!r} is disabled unless "
                            "WAHS_ALLOW_EXPERIMENTAL_ENRICHERS=true."
                        ),
                    )
                if eid not in KNOWN_EXPERIMENTAL_ENRICHERS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unknown experimental enricher ID: {eid!r}.",
                    )
                expanded.append(eid)
            elif eid in KNOWN_GENERALIZABLE_ENRICHERS:
                expanded.append(eid)
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unknown WAHS enricher ID {eid!r}; expected one of "
                        f"{sorted(ALL_GENERALIZABLE_ENRICHERS)} or presets 'core_generalizable'/'all_generalizable'."
                    ),
                )
    return tuple(dict.fromkeys(expanded))


def resolve_effective_profile(
    options: WahsRequestOptions,
    settings: WahsEngineSettings,
) -> ResolvedProfileConfig:
    try:
        reg = load_profile_registry(settings.profiles_path or None)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    default_pid = (settings.default_profile or reg.default_profile).strip() or reg.default_profile
    profile_id = (options.profile or default_pid).strip()
    prof = reg.profiles.get(profile_id)
    if prof is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown WAHS profile {profile_id!r}; available profiles: {sorted(reg.profiles)}.",
        )

    prompt_id = (options.prompt or prof.prompt).strip()
    if prompt_id.endswith(".txt"):
        prompt_id = prompt_id[:-4]
    try:
        prompt_text, prompt_sha = load_prompt(prompt_id, prompts_dir=settings.prompts_dir or None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if options.enrichers is not None:
        raw_enrichers: Sequence[str] = options.enrichers
    elif options.profile is not None:
        raw_enrichers = prof.enrichers
    else:
        raw_enrichers = prof.enrichers if prof.enrichers else settings.enabled_enrichers

    active_enrichers = expand_and_validate_enrichers(
        raw_enrichers,
        allow_experimental=settings.allow_experimental_enrichers,
    )
    return ResolvedProfileConfig(
        profile_id=prof.id,
        label=prof.label,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha,
        active_enrichers=active_enrichers,
        preferred_chart_tokens=prof.preferred_chart_tokens,
    )


def serialize_profiles_catalog(settings: WahsEngineSettings) -> Dict[str, Any]:
    reg = load_profile_registry(settings.profiles_path or None)
    default_pid = (settings.default_profile or reg.default_profile).strip() or reg.default_profile
    if default_pid not in reg.profiles:
        default_pid = reg.default_profile
    out_profiles: Dict[str, Dict[str, Any]] = {}
    for pid, prof in reg.profiles.items():
        _, prompt_sha = load_prompt(prof.prompt, prompts_dir=settings.prompts_dir or None)
        out_profiles[pid] = {
            "id": prof.id,
            "label": prof.label,
            "prompt": prof.prompt,
            "prompt_sha256": prompt_sha,
            "enrichers": list(prof.enrichers),
            "preferred_chart_tokens": prof.preferred_chart_tokens,
        }
    return {
        "default_profile": default_pid,
        "profiles": out_profiles,
    }
