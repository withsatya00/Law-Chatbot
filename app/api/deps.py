from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, Header

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import Role, decode_token, is_token_revoked


async def get_current_user_claims(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Phase 1 security hardening: mandatory-auth counterpart to
    `get_current_user_id` below. No/malformed `Authorization` header or a
    token that fails `decode_token` -> 401 (never silently anonymous) --
    for routes where anonymous access must not be possible, like `/admin/*`.
    """
    if not authorization:
        raise UnauthorizedError("Authentication required.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("Authentication required.")
    payload = decode_token(token)
    if await is_token_revoked(payload):
        raise UnauthorizedError("This token has been revoked.")
    return payload


def require_roles(*roles: Role) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Route/router dependency factory: 401 with no valid token, 403 if the
    token's role isn't one of `roles`. Use via
    `APIRouter(dependencies=[Depends(require_roles(Role.admin))])` to guard
    every route on a router without touching each handler's signature.
    """
    allowed = {role.value for role in roles}

    async def _dependency(claims: dict[str, Any] = Depends(get_current_user_claims)) -> dict[str, Any]:
        if claims.get("role") not in allowed:
            raise ForbiddenError("Insufficient permissions for this operation.")
        return claims

    return _dependency


require_admin = require_roles(Role.admin, Role.super_admin)


async def get_current_user_id(authorization: str | None = Header(default=None)) -> str | None:
    """Part 46 "Authenticated User Ownership": the JWT-derived identity to use
    as an ownership boundary, reused by every route that needs it.

    No `Authorization` header at all -> `None` (anonymous; callers that allow
    anonymous access, like `/chat` and `/upload`, proceed with no owner
    identity). A header present but not a `Bearer` token -> also `None` (not
    a JWT presentation attempt at all). A header presenting an actual bearer
    token that fails `decode_token` (bad signature, expired, malformed) ->
    `decode_token` raises `UnauthorizedError` (`app/core/security.py`),
    which the app's existing exception handler already maps to 401 -- a bad
    credential is never silently downgraded to anonymous.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    payload = decode_token(token)
    if await is_token_revoked(payload):
        raise UnauthorizedError("This token has been revoked.")
    return payload.get("sub")
