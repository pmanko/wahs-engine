"""Slim baseline OpenAI-compatible runtime for WAHS Engine."""

from __future__ import annotations

from .config import DEFAULT_MODEL_ALIASES, ModelSpec, WahsEngineSettings

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODEL_ALIASES",
    "ModelSpec",
    "WahsEngineSettings",
    "__version__",
]
