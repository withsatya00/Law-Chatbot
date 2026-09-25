"""`POST /refresh` and `POST /logout` (`app/api/auth.py`) -- both were either
missing entirely (`refresh`) or a no-op stub (`logout`) before this session's
JWT revocation work landed. Calls the route functions directly (same
pattern as `tests/test_auth_deps.py`), no HTTP client/live server needed.
"""

import asyncio

import pytest

from app.api.auth import logout, refresh
from app.cache.redis_client import redis_client
from app.core.exceptions import UnauthorizedError
from app.core.security import (
    Role,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_revoked,
)
from app.schemas.auth import LogoutRequest, RefreshRequest


class _FakeRedis:
    """Minimal in-memory stand-in -- real dict-backed `set`/`exists` (not a
    bare `AsyncMock`) so revocation tests verify actual before/after state,
    not just that a method was called. TTL is accepted and ignored (never
    relevant within one test's lifetime).
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis_client, "_client", _FakeRedis())


def test_refresh_issues_new_tokens_and_revokes_the_old_refresh_token() -> None:
    old_refresh = create_refresh_token("user-1", Role.user)

    response = asyncio.run(refresh(RefreshRequest(refresh_token=old_refresh)))

    assert response.user_id == "user-1"
    assert response.role == Role.user
    assert response.access_token
    assert response.refresh_token != old_refresh
    old_payload = decode_token(old_refresh, expected_type="refresh")
    assert asyncio.run(is_token_revoked(old_payload)) is True


def test_refresh_rejects_a_reused_token_on_the_second_attempt() -> None:
    old_refresh = create_refresh_token("user-2", Role.user)
    asyncio.run(refresh(RefreshRequest(refresh_token=old_refresh)))

    with pytest.raises(UnauthorizedError):
        asyncio.run(refresh(RefreshRequest(refresh_token=old_refresh)))


def test_refresh_rejects_an_access_token_presented_as_a_refresh_token() -> None:
    access = create_access_token("user-3", Role.user)

    with pytest.raises(UnauthorizedError):
        asyncio.run(refresh(RefreshRequest(refresh_token=access)))


def test_logout_revokes_the_bearer_access_token() -> None:
    token = create_access_token("user-4", Role.user)

    result = asyncio.run(logout(body=None, authorization=f"Bearer {token}"))

    assert result["status"] == "logged_out"
    assert asyncio.run(is_token_revoked(decode_token(token))) is True


def test_logout_with_no_token_at_all_is_rejected() -> None:
    from app.core.exceptions import BadRequestError

    with pytest.raises(BadRequestError):
        asyncio.run(logout(body=None, authorization=None))


def test_logout_also_revokes_an_explicitly_supplied_refresh_token() -> None:
    access = create_access_token("user-5", Role.user)
    refresh_token = create_refresh_token("user-5", Role.user)

    asyncio.run(logout(body=LogoutRequest(refresh_token=refresh_token), authorization=f"Bearer {access}"))

    assert asyncio.run(is_token_revoked(decode_token(refresh_token, expected_type="refresh"))) is True
