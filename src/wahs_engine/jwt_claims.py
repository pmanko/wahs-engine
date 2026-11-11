"""Zero-privilege Bearer JWT claim extraction and subject-binding checks."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Mapping, Optional

from fastapi import HTTPException


@dataclass(frozen=True)
class VerifiedCallerIdentity:
    """Bound caller identity extracted from the Authorization Bearer token."""

    sub: str
    raw_token: str
    token_fingerprint: str
    email: Optional[str] = None
    claims: Dict[str, Any] = None  # type: ignore[assignment]


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def extract_bearer_token(headers: Mapping[str, str]) -> str:
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


def decode_and_bind_jwt(
    raw_token: str,
    *,
    hs256_secret: str = "",
    expected_user_sub: Optional[str] = None,
    trusted_header_sub: Optional[str] = None,
) -> VerifiedCallerIdentity:
    """Decode Bearer JWT claims, enforce optional HMAC/header checks, and bind `sub`.

    Note on zero-privilege security model:
    In production, `fhir-auth` (`uwcirg/jwt-proxy`) cryptographically validates
    `raw_token` against Keycloak's JWKS on every request (`MasterPatient` +
    dataset `Patient` stub checks) BEFORE `wahs-engine` reads any cached dataset.
    Additionally, cache keys include `token_fingerprint` when `hs256_secret` is
    unset so a forged token can never collide with another token's in-memory entry.
    """
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization: Bearer <keycloak_jwt> header.",
        )

    parts = raw_token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=401,
            detail="Malformed Authorization Bearer JWT (expected 3 segments).",
        )

    header_b64, payload_b64, sig_b64 = parts
    try:
        header_obj = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload_obj = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail="Malformed JWT header or payload encoding.",
        ) from exc

    if not isinstance(header_obj, dict) or not isinstance(payload_obj, dict):
        raise HTTPException(status_code=401, detail="Invalid JWT JSON structure.")

    if hs256_secret:
        if header_obj.get("alg") != "HS256":
            raise HTTPException(status_code=401, detail="Unsupported JWT signing algorithm.")
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            hs256_secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        try:
            actual_sig = _b64url_decode(sig_b64)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid JWT signature encoding.") from exc
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise HTTPException(status_code=401, detail="JWT signature verification failed.")

    exp = payload_obj.get("exp")
    if isinstance(exp, (int, float)) and exp < (time.time() - 30):
        raise HTTPException(status_code=401, detail="Bearer JWT has expired.")

    sub = str(payload_obj.get("sub") or "").strip()
    if not sub:
        raise HTTPException(status_code=401, detail="Bearer JWT is missing required 'sub' claim.")

    if trusted_header_sub and trusted_header_sub.strip() != sub:
        raise HTTPException(
            status_code=403,
            detail="Authenticated gateway subject mismatches Bearer JWT 'sub' claim.",
        )

    if expected_user_sub and expected_user_sub.strip() != sub:
        raise HTTPException(
            status_code=403,
            detail="Request wahs.user_sub mismatches authenticated Bearer JWT 'sub' claim.",
        )

    token_fp = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:24]
    email = str(payload_obj.get("email") or payload_obj.get("preferred_username") or "").strip() or None
    return VerifiedCallerIdentity(
        sub=sub,
        raw_token=raw_token,
        token_fingerprint=token_fp,
        email=email,
        claims=payload_obj,
    )
