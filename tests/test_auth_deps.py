"""Part 46 "Authenticated User Ownership": `get_current_user_id`, the one
JWT-verification dependency reused by every route that needs an ownership
identity. No header -> anonymous. A present-but-invalid bearer token is
never silently downgraded to anonymous -- it raises, matching the task's
"invalid JWT -> protected endpoints reject" requirement.
"""

import asyncio

import pytest

from app.api.deps import get_current_user_id
from app.core.exceptions import UnauthorizedError
from app.core.security import Role, create_access_token


def test_no_authorization_header_is_anonymous() -> None:
    assert asyncio.run(get_current_user_id(authorization=None)) is None


def test_non_bearer_scheme_is_treated_as_anonymous() -> None:
    assert asyncio.run(get_current_user_id(authorization="Basic dXNlcjpwYXNz")) is None


def test_valid_bearer_token_extracts_the_correct_user_id() -> None:
    token = create_access_token("user-42", Role.user)
    assert asyncio.run(get_current_user_id(authorization=f"Bearer {token}")) == "user-42"


def test_malformed_bearer_token_is_rejected_not_silently_ignored() -> None:
    with pytest.raises(UnauthorizedError):
        asyncio.run(get_current_user_id(authorization="Bearer not-a-real-jwt"))


def test_tampered_bearer_token_is_rejected() -> None:
    token = create_access_token("user-42", Role.user)
    tampered = token[:-4] + "abcd"
    with pytest.raises(UnauthorizedError):
        asyncio.run(get_current_user_id(authorization=f"Bearer {tampered}"))


def test_refresh_token_used_as_access_token_is_rejected() -> None:
    from app.core.security import create_refresh_token

    token = create_refresh_token("user-42", Role.user)
    with pytest.raises(UnauthorizedError):
        asyncio.run(get_current_user_id(authorization=f"Bearer {token}"))
