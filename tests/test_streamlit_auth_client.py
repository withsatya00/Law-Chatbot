from collections.abc import MutableMapping
from typing import Any, Self

import httpx

from streamlit_app import auth_client


def _response(status: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload or {},
        request=httpx.Request("POST", "http://api.test"),
    )


class _ScriptedClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, **_kwargs: Any) -> "_ScriptedClient":
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def _next(self, method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._next("POST", url, kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return self._next(method, url, kwargs)


def _signed_in_state() -> dict[str, Any]:
    return {
        auth_client.ACCESS_TOKEN_KEY: "old-access",
        auth_client.REFRESH_TOKEN_KEY: "old-refresh",
        auth_client.ROLE_KEY: "admin",
        auth_client.USER_ID_KEY: "admin-1",
        auth_client.EMAIL_KEY: "admin@example.com",
    }


def test_login_stores_identity_and_tokens_but_never_the_password() -> None:
    client = _ScriptedClient([
        _response(
            200,
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "role": "admin",
                "user_id": "admin-1",
            },
        )
    ])
    state: MutableMapping[str, Any] = {}

    ok, message = auth_client.login(
        state, "http://api.test", "admin@example.com", "secret", client_factory=client
    )

    assert ok is True and message == "Signed in."
    assert state == {
        "access_token": "access",
        "refresh_token": "refresh",
        "user_role": "admin",
        "user_id": "admin-1",
        "user_email": "admin@example.com",
    }
    assert "secret" not in repr(state)


def test_authenticated_request_forwards_bearer_and_custom_headers() -> None:
    client = _ScriptedClient([_response(200)])
    state = _signed_in_state()

    auth_client.request(
        state,
        "http://api.test",
        "POST",
        "/chat",
        headers={"X-Trace": "trace-1"},
        json={"question": "hello"},
        client_factory=client,
    )

    assert client.calls[0][2]["headers"] == {
        "X-Trace": "trace-1",
        "Authorization": "Bearer old-access",
    }


def test_401_rotates_once_and_retries_once_with_the_new_token() -> None:
    client = _ScriptedClient([
        _response(401),
        _response(
            200,
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "role": "admin",
                "user_id": "admin-1",
            },
        ),
        _response(200),
    ])
    state = _signed_in_state()

    response = auth_client.request(
        state,
        "http://api.test",
        "POST",
        "/chat",
        headers={"X-Trace": "trace-1"},
        json={"question": "hello"},
        client_factory=client,
    )

    assert response.status_code == 200
    assert [call[1] for call in client.calls] == [
        "http://api.test/chat",
        "http://api.test/refresh",
        "http://api.test/chat",
    ]
    assert client.calls[-1][2]["headers"] == {
        "X-Trace": "trace-1",
        "Authorization": "Bearer new-access",
    }
    assert state[auth_client.ACCESS_TOKEN_KEY] == "new-access"


def test_a_second_401_clears_the_local_session_and_does_not_loop() -> None:
    client = _ScriptedClient([
        _response(401),
        _response(
            200,
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "role": "admin",
                "user_id": "admin-1",
            },
        ),
        _response(401),
    ])
    state = _signed_in_state()

    response = auth_client.request(
        state, "http://api.test", "GET", "/protected", client_factory=client
    )

    assert response.status_code == 401
    assert len(client.calls) == 3
    assert state == {}


def test_logout_revokes_server_tokens_and_always_clears_local_identity() -> None:
    client = _ScriptedClient([_response(200)])
    state = _signed_in_state()

    auth_client.logout(state, "http://api.test", client_factory=client)

    assert state == {}
    assert client.calls[0][1] == "http://api.test/logout"
    assert client.calls[0][2]["headers"] == {"Authorization": "Bearer old-access"}
    assert client.calls[0][2]["json"] == {"refresh_token": "old-refresh"}


def test_super_admin_is_recognized_as_an_admin_identity() -> None:
    assert auth_client.is_admin({auth_client.ROLE_KEY: "super_admin"}) is True
