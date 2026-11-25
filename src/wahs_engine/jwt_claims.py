"""Zero-privilege Bearer JWT claim extraction, RS256 JWKS verification, and subject binding."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
import urllib.request

from fastapi import HTTPException

# ASN.1 DER DigestInfo prefix for SHA-256 (RFC 8017 Section 9.2 Notes 1)
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")

_jwks_cache_lock = threading.Lock()
_jwks_url_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_JWKS_TTL_SECONDS = 300.0


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


def verify_rs256_jwk_signature(
    signing_input: bytes,
    signature_bytes: bytes,
    jwk: Mapping[str, Any],
) -> bool:
    """Verify an RSASSA-PKCS1-v1_5 SHA-256 (`RS256`) signature in pure Python stdlib (RFC 8017)."""
    if str(jwk.get("kty") or "").upper() != "RSA":
        return False
    use = str(jwk.get("use") or "sig").lower()
    if use not in {"sig", ""}:
        return False
    alg = str(jwk.get("alg") or "RS256").upper()
    if alg not in {"RS256", ""}:
        return False
    n_b64 = str(jwk.get("n") or "").strip()
    e_b64 = str(jwk.get("e") or "").strip()
    if not n_b64 or not e_b64:
        return False
    try:
        n_bytes = _b64url_decode(n_b64).lstrip(b"\x00")
        e_bytes = _b64url_decode(e_b64).lstrip(b"\x00")
    except Exception:
        return False
    k = len(n_bytes)
    if k < 256 or not e_bytes or len(signature_bytes) != k:
        return False
    n_int = int.from_bytes(n_bytes, "big")
    e_int = int.from_bytes(e_bytes, "big")
    if n_int % 2 == 0 or e_int < 3 or e_int >= n_int:
        return False
    sig_int = int.from_bytes(signature_bytes, "big")
    if sig_int >= n_int:
        return False
    em_int = pow(sig_int, e_int, n_int)
    em = em_int.to_bytes(k, "big")
    t_bytes = _SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(signing_input).digest()
    ps_len = k - len(t_bytes) - 3
    if ps_len < 8:
        return False
    expected_em = b"\x00\x01" + (b"\xff" * ps_len) + b"\x00" + t_bytes
    return hmac.compare_digest(em, expected_em)


def _parse_jwks_payload(raw: str) -> List[Dict[str, Any]]:
    loaded = json.loads(raw)
    if isinstance(loaded, Mapping):
        if isinstance(loaded.get("keys"), list):
            return [dict(k) for k in loaded["keys"] if isinstance(k, Mapping)]
        if loaded.get("kty"):
            return [dict(loaded)]
    elif isinstance(loaded, list):
        return [dict(k) for k in loaded if isinstance(k, Mapping)]
    return []


def _fetch_jwks_from_url(jwks_url: str, *, force_refresh: bool = False) -> List[Dict[str, Any]]:
    now = time.monotonic()
    with _jwks_cache_lock:
        cached = _jwks_url_cache.get(jwks_url)
        if cached is not None and not force_refresh and now < cached[0]:
            return list(cached[1])
    try:
        req = urllib.request.Request(jwks_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # nosec B310 - operator-configured JWKS URL
            body = resp.read().decode("utf-8")
        keys = _parse_jwks_payload(body)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load JWKS keys from configured WAHS_JWT_JWKS_URL: {exc}",
        ) from exc
    with _jwks_cache_lock:
        _jwks_url_cache[jwks_url] = (time.monotonic() + _JWKS_TTL_SECONDS, keys)
    return list(keys)


def _verify_against_jwks(
    header_obj: Mapping[str, Any],
    signing_input: bytes,
    sig_b64: str,
    *,
    jwks_json: str = "",
    jwks_url: str = "",
) -> None:
    if header_obj.get("alg") != "RS256":
        raise HTTPException(
            status_code=401,
            detail="Unsupported JWT signing algorithm (expected RS256 for JWKS verification).",
        )
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid JWT signature encoding.") from exc

    kid = str(header_obj.get("kid") or "").strip()
    keys: List[Dict[str, Any]] = []
    if jwks_json:
        try:
            keys.extend(_parse_jwks_payload(jwks_json))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Invalid WAHS_JWT_JWKS_JSON configuration.") from exc
    if jwks_url:
        keys.extend(_fetch_jwks_from_url(jwks_url, force_refresh=False))

    def _matches(candidate_keys: List[Dict[str, Any]]) -> bool:
        filtered = [k for k in candidate_keys if not kid or not k.get("kid") or str(k.get("kid")) == kid]
        for k in filtered:
            if verify_rs256_jwk_signature(signing_input, actual_sig, k):
                return True
        return False

    if _matches(keys):
        return
    # On key rotation (kid miss), refresh remote JWKS once before rejecting
    if jwks_url:
        refreshed = _fetch_jwks_from_url(jwks_url, force_refresh=True)
        if _matches(refreshed):
            return
    raise HTTPException(status_code=401, detail="JWT RS256 JWKS signature verification failed.")


def decode_and_bind_jwt(
    raw_token: str,
    *,
    hs256_secret: str = "",
    jwks_json: str = "",
    jwks_url: str = "",
    expected_issuer: str = "",
    expected_audience: str = "",
    expected_user_sub: Optional[str] = None,
    trusted_header_sub: Optional[str] = None,
) -> VerifiedCallerIdentity:
    """Decode Bearer JWT claims, enforce optional RS256 JWKS / HS256 / claim checks, and bind `sub`.

    Security model:
    1. When `WAHS_JWT_JWKS_URL` / `WAHS_JWT_JWKS_JSON` (RS256) or `WAHS_JWT_HS256_SECRET` is set,
       `wahs-engine` cryptographically verifies the signature locally before upstream calls.
    2. In all Mode-B deployments, `fhir-auth` (`uwcirg/jwt-proxy`) also cryptographically validates
       `raw_token` against Keycloak's JWKS on every request (`MasterPatient` + dataset `Patient`
       stub checks) before `wahs-engine` reads any cached dataset, and `token_fingerprint` prevents
       cache-key collisions when local signature keys are unset.
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

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    if jwks_json or jwks_url:
        _verify_against_jwks(
            header_obj,
            signing_input,
            sig_b64,
            jwks_json=jwks_json,
            jwks_url=jwks_url,
        )
    elif hs256_secret:
        if header_obj.get("alg") != "HS256":
            raise HTTPException(status_code=401, detail="Unsupported JWT signing algorithm.")
        expected_sig = hmac.new(
            hs256_secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        try:
            actual_sig = _b64url_decode(sig_b64)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid JWT signature encoding.") from exc
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise HTTPException(status_code=401, detail="JWT signature verification failed.")

    now = time.time()
    exp = payload_obj.get("exp")
    if isinstance(exp, (int, float)) and exp < (now - 30):
        raise HTTPException(status_code=401, detail="Bearer JWT has expired.")
    nbf = payload_obj.get("nbf")
    if isinstance(nbf, (int, float)) and nbf > (now + 30):
        raise HTTPException(status_code=401, detail="Bearer JWT is not yet valid (nbf).")

    if expected_issuer:
        iss = str(payload_obj.get("iss") or "").strip()
        if iss != expected_issuer.strip():
            raise HTTPException(status_code=401, detail="Bearer JWT issuer ('iss') mismatch.")

    if expected_audience:
        raw_aud = payload_obj.get("aud")
        aud_set = (
            {str(a).strip() for a in raw_aud if a}
            if isinstance(raw_aud, list)
            else ({str(raw_aud).strip()} if raw_aud else set())
        )
        if expected_audience.strip() not in aud_set:
            raise HTTPException(status_code=401, detail="Bearer JWT audience ('aud') mismatch.")

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
