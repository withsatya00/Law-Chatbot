from fastapi import APIRouter, Header

from app.core.exceptions import BadRequestError, UnauthorizedError
from app.core.security import (
    Role,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_revoked,
    revoke_token,
)
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from app.services.auth_service import AuthService

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest) -> TokenResponse:
    return await AuthService().register(request)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    return await AuthService().login(request)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest) -> TokenResponse:
    """Was entirely missing before -- `create_refresh_token` was minted at
    every login (see `AuthService._tokens`) but nothing ever redeemed one,
    so in practice every session was a bare access token with no renewal
    path at all (the client had no choice but to force a fresh `/login`
    once `access_token_expire_minutes` ran out). `decode_token` already
    401s on a malformed/expired/wrong-type token before this runs.

    Rotates on every use (mints a NEW refresh token and immediately
    revokes the one just presented) rather than reissuing the same one --
    standard refresh-rotation practice: a stolen refresh token is only
    ever usable once before `is_token_revoked` (`app/core/security.py`)
    catches the theft on the legitimate client's next refresh.
    """
    payload = decode_token(request.refresh_token, expected_type="refresh")
    if await is_token_revoked(payload):
        raise UnauthorizedError("This refresh token has already been used or revoked.")
    await revoke_token(payload)
    role = Role(payload["role"])
    subject = payload["sub"]
    return TokenResponse(
        access_token=create_access_token(subject, role),
        refresh_token=create_refresh_token(subject, role),
        role=role,
        user_id=subject,
    )


@router.post("/logout")
async def logout(
    body: LogoutRequest | None = None, authorization: str | None = Header(default=None)
) -> dict[str, str]:
    """Was previously a no-op that read/deleted nothing (see
    [[project_orphaned_prototype_cluster]]'s sibling memory) -- now actually
    ends the session by blacklisting the presented token(s)' `jti` via
    `revoke_token` (see `app/core/security.py`), checked by
    `get_current_user_id`/`get_current_user_claims` on every subsequent
    authenticated request. The access token comes from the `Authorization`
    header (never trust a client-supplied body for the token that's
    identifying the caller); the refresh token, if the caller wants it
    revoked too, is passed explicitly in the body since it isn't otherwise
    present on this request at all.
    """
    revoked_count = 0
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            await revoke_token(decode_token(token, expected_type="access"))
            revoked_count += 1
    if body and body.refresh_token:
        await revoke_token(decode_token(body.refresh_token, expected_type="refresh"))
        revoked_count += 1
    if revoked_count == 0:
        raise BadRequestError("No access token (Authorization header) or refresh_token provided to log out.")
    return {"status": "logged_out"}
