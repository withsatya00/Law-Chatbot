from pydantic import BaseModel, EmailStr, Field

from app.core.security import Role


class RegisterRequest(BaseModel):
    """Public self-registration. No `role` field: every account created
    through this endpoint is `Role.user` -- an unauthenticated caller must
    never be able to request an elevated role. Privileged accounts are
    provisioned out-of-band; see `scripts/create_privileged_user.py`.
    """

    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=2, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: Role
    user_id: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    """Optional -- the access token is taken from the `Authorization` header,
    not this body. Only needed to ALSO revoke the refresh token issued
    alongside it at login (`AuthService._tokens` always mints both)."""

    refresh_token: str | None = None
