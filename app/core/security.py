import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

try:
    from jose import JWTError, jwt
except ModuleNotFoundError:
    JWTError = ValueError
    jwt = None

try:
    from passlib.context import CryptContext
except ModuleNotFoundError:
    CryptContext = None

import structlog

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import UnauthorizedError

log = structlog.get_logger(__name__)

_TOKEN_BLACKLIST_PREFIX = "token_blacklist:"
_DEFAULT_BLACKLIST_TTL_SECONDS = 86400


class Role(StrEnum):
    user = "user"
    lawyer = "lawyer"
    admin = "admin"
    super_admin = "super_admin"


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") if CryptContext else None


def hash_password(password: str) -> str:
    if pwd_context is not None:
        return str(pwd_context.hash(password))
    salt = base64.urlsafe_b64encode(hashlib.sha256(settings.jwt_secret_key.encode()).digest()[:12]).decode()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    if pwd_context is not None:
        return bool(pwd_context.verify(password, password_hash))
    _, salt, expected = password_hash.split("$", 2)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return hmac.compare_digest(digest, expected)


def create_access_token(subject: str, role: Role, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {"sub": subject, "role": role.value, "type": "access", "exp": expire, "jti": uuid4().hex}
    return _encode_jwt(payload)


def create_refresh_token(subject: str, role: Role) -> str:
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload: dict[str, Any] = {"sub": subject, "role": role.value, "type": "refresh", "exp": expire, "jti": uuid4().hex}
    return _encode_jwt(payload)


def decode_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    try:
        payload = _decode_jwt(token)
    except (JWTError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise UnauthorizedError("Invalid or expired authentication token.") from exc
    if payload.get("type") != expected_type:
        raise UnauthorizedError("Invalid token type.")
    return payload


async def revoke_token(payload: dict[str, Any]) -> None:
    """Token revocation: no route ever checked this before -- a stolen or
    logged-out access token stayed valid for its full remaining lifetime
    (up to `access_token_expire_minutes`), since `decode_token` above only
    ever checks the signature and expiry, never any external state.

    Blacklists this specific token's `jti` (added to every token minted by
    `create_access_token`/`create_refresh_token` above) in Redis, keyed so
    `is_token_revoked` can check it on every subsequent authenticated
    request. TTL is set to the token's OWN remaining lifetime (from its
    `exp` claim) rather than a fixed duration -- once the token would have
    expired naturally anyway, the blacklist entry is redundant, so this
    never accumulates unbounded keys for old tokens. A token with no `jti`
    (shouldn't occur for anything minted by this module, but defensive
    against a hand-crafted/legacy token) is a no-op, matching this
    function's only caller (`POST /logout`) already raising on totally
    invalid tokens before ever reaching here.
    """
    jti = payload.get("jti")
    if not jti:
        return
    exp = payload.get("exp")
    ttl_seconds = _DEFAULT_BLACKLIST_TTL_SECONDS
    if exp:
        ttl_seconds = max(int(exp) - int(datetime.now(UTC).timestamp()), 1)
    await redis_client.client.set(f"{_TOKEN_BLACKLIST_PREFIX}{jti}", "1", ex=ttl_seconds)


async def is_token_revoked(payload: dict[str, Any]) -> bool:
    """Fails OPEN (not revoked) if Redis is unreachable/not connected --
    matching this app's existing convention elsewhere (e.g.
    `ConversationMemoryStore.clear`) that a Redis outage degrades a
    secondary feature rather than taking down the primary one. Revocation
    is defense-in-depth on top of signature+expiry (`decode_token` already
    ran and passed by the time this is called); the alternative -- treating
    Redis-down as "definitely revoked" -- would turn a cache outage into a
    total authenticated-traffic outage, a strictly worse failure mode for a
    check whose entire purpose is an EXTRA layer of safety.
    """
    jti = payload.get("jti")
    if not jti:
        return False
    try:
        return bool(await redis_client.client.exists(f"{_TOKEN_BLACKLIST_PREFIX}{jti}"))
    # matches RedisClient.ping()'s own defensive breadth
    except Exception as exc:  # noqa: BLE001 - a revocation check that cannot reach Redis fails closed to 'not revoked' and is logged; it must never break auth
        log.warning("token_revocation_check_unavailable", error=str(exc))
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return int(value.timestamp())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encode_jwt(payload: dict[str, Any]) -> str:
    if jwt is not None:
        return str(jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm))
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _b64(json.dumps(header, separators=(",", ":")).encode()),
            _b64(json.dumps(payload, default=_json_default, separators=(",", ":")).encode()),
        ]
    )
    signature = hmac.new(settings.jwt_secret_key.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


def _decode_jwt(token: str) -> dict[str, Any]:
    if jwt is not None:
        decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if not isinstance(decoded, dict):
            raise ValueError("Token payload is not a JSON object.")
        return decoded
    header_b64, payload_b64, signature_b64 = token.split(".")
    signing_input = f"{header_b64}.{payload_b64}"
    expected = _b64(hmac.new(settings.jwt_secret_key.encode(), signing_input.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(signature_b64, expected):
        raise ValueError("Invalid token signature.")
    payload = json.loads(_unb64(payload_b64))
    if not isinstance(payload, dict):
        # ValueError, not TypeError (TRY004): this is a malformed TOKEN, not a
        # programming error, and `decode_token` above catches exactly
        # `(JWTError, ValueError, KeyError, JSONDecodeError)` to turn it into a
        # 401. A TypeError here would escape that handler as a 500 and tell an
        # attacker probing with crafted tokens that they found something.
        raise ValueError("Token payload is not a JSON object.")  # noqa: TRY004
    if payload.get("exp") and int(payload["exp"]) < int(datetime.now(UTC).timestamp()):
        raise ValueError("Expired token.")
    return payload


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
