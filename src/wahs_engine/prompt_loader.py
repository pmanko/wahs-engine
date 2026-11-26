"""File-backed system prompt loader with live reload and SHA-256 provenance."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Optional, Tuple

_SAFE_PROMPT_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def resolve_prompts_dir(prompts_dir: Optional[Path | str] = None) -> Path:
    if prompts_dir is not None and str(prompts_dir).strip():
        return Path(str(prompts_dir).strip()).expanduser().resolve()
    env_dir = os.environ.get("WAHS_PROMPTS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return _DEFAULT_PROMPTS_DIR


def load_prompt(
    prompt_id: str = "wahs-default",
    *,
    prompts_dir: Optional[Path | str] = None,
) -> Tuple[str, str]:
    """Read `<prompt_id>.txt` fresh from `prompts_dir`, returning `(text, sha256_hex)`.

    Raises `FileNotFoundError` if `prompt_id` is not a safe slug or `<prompt_id>.txt` is missing.
    """
    slug = (prompt_id or "").strip()
    if slug.endswith(".txt"):
        slug = slug[:-4]
    if not slug or not _SAFE_PROMPT_SLUG_RE.match(slug):
        raise FileNotFoundError(f"Invalid WAHS prompt slug: {prompt_id!r}")
    base_dir = resolve_prompts_dir(prompts_dir)
    target = (base_dir / f"{slug}.txt").resolve()
    if target.parent != base_dir or not target.is_file():
        raise FileNotFoundError(f"WAHS prompt file not found: {slug}.txt in {base_dir}")
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"WAHS prompt file is empty: {target}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest


__all__ = ["load_prompt", "resolve_prompts_dir"]
