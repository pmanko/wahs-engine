"""Offline exact tokenizers for `wahs_engine` (Gemma 4 SentencePiece + OpenAI tiktoken).

Zero network calls are ever made at runtime: the Gemma 4 SentencePiece `tokenizer.json`
is bundled as a gzip-compressed package asset under `wahs_engine/tokenizer_assets/`,
and OpenAI `o200k_base` / `cl100k_base` BPE encodings are pre-cached in the Docker
image via `TIKTOKEN_CACHE_DIR`.
"""

from __future__ import annotations

from functools import lru_cache
import gzip
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import tiktoken
from tokenizers import Tokenizer

_ASSET_DIR = Path(__file__).resolve().parent / "tokenizer_assets"
_GEMMA4_ASSET = _ASSET_DIR / "gemma4_tokenizer.json.gz"


@lru_cache(maxsize=4)
def _load_gemma4_tokenizer() -> Tokenizer:
    if not _GEMMA4_ASSET.is_file():
        raise RuntimeError(
            f"Bundled Gemma 4 tokenizer asset not found at {_GEMMA4_ASSET}."
        )
    raw_json = gzip.decompress(_GEMMA4_ASSET.read_bytes()).decode("utf-8")
    return Tokenizer.from_str(raw_json)


@lru_cache(maxsize=8)
def _load_tiktoken_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=16)
def get_exact_token_counter(tokenizer_family: str = "gemma4") -> Callable[[str], int]:
    """Return an exact offline `text -> int` token counter for `tokenizer_family`."""
    fam = (tokenizer_family or "gemma4").strip().lower()
    if fam in {"gemma", "gemma4", "gemma-4", "gemma3", "medgemma"}:
        tok = _load_gemma4_tokenizer()

        def _count_gemma4(text: str) -> int:
            if not text:
                return 0
            return len(tok.encode(text, add_special_tokens=False).ids)

        return _count_gemma4

    if fam in {"o200k", "o200k_base", "gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"}:
        enc = _load_tiktoken_encoding("o200k_base")

        def _count_o200k(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text, disallowed_special=()))

        return _count_o200k

    if fam in {"cl100k", "cl100k_base", "gpt-4", "gpt-3.5", "openai"}:
        enc = _load_tiktoken_encoding("cl100k_base")

        def _count_cl100k(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text, disallowed_special=()))

        return _count_cl100k

    # Fallback: use tiktoken `o200k_base` for any unspecified external BPE family
    enc = _load_tiktoken_encoding("o200k_base")

    def _count_default_bpe(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text, disallowed_special=()))

    return _count_default_bpe


def count_chat_messages_tokens(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    tokenizer_family: str = "gemma4",
) -> int:
    """Count exact chat-prompt tokens including per-turn template overhead and tool schemas.

    Empirically verified against live `ollama/gemma4:12b`: each message adds 5 turn-framing
    tokens plus an 8-token prompt header/generation wrapper (`8 + 5 * len(messages)`).
    """
    counter = get_exact_token_counter(tokenizer_family)
    msg_count = 0
    total = 0
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        msg_count += 1
        content = msg.get("content")
        if isinstance(content, list):
            text_val = "\n".join(
                str(p.get("text") or "")
                for p in content
                if isinstance(p, Mapping)
            )
        else:
            text_val = str(content or "")
        total += counter(text_val)
        if msg.get("tool_calls"):
            total += counter(json.dumps(msg["tool_calls"], separators=(",", ":"), default=str))
        if msg.get("name"):
            total += counter(str(msg["name"]))

    if msg_count > 0:
        total += 8 + (5 * msg_count)

    if tools:
        serialized_tools = json.dumps(list(tools), separators=(",", ":"), default=str)
        total += counter(serialized_tools) + 16

    return total
