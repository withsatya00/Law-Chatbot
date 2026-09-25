"""Token handling for the Streamlit client.

The chat-first refactor deleted the login form along with the admin pages,
which left the client with no way to obtain a token at all: `POST /chat` went
out unauthenticated, `_try_refresh_admin_token` had tokens no UI flow ever
set, and two naming conventions (`admin_token` and `access_token`) were read
in different places. This module is the single owner of that state.

It is deliberately free of `streamlit`: every function takes the session-state
mapping as its first argument, so the refresh-and-retry rule is testable with
a plain dict and a stub transport rather than a running Streamlit server.
`streamlit_app/app.py` passes `st.session_state` straight through.

Two invariants:

* **One naming convention.** `access_token` / `refresh_token` / `user_role` /
  `user_email`. Nothing writes `admin_token`.
* **Nothing secret is ever returned for display.** Passwords are read once
  and passed to `POST /login`; tokens live in session state and are only ever
  rendered as the `Authorization` header. Error text shown to the user is
  generated here, never echoed from a response body that could contain a
  credential.
"""

from collections.abc import MutableMapping
from typing import Any, cast

import httpx

ACCESS_TOKEN_KEY = "access_token"
REFRESH_TOKEN_KEY = "refresh_token"
ROLE_KEY = "user_role"
USER_ID_KEY = "user_id"
EMAIL_KEY = "user_email"

# Every session key this module owns, so logout and a failed refresh clear
# exactly the same set -- a half-cleared session is how a stale role ends up
# gating a UI element after the token behind it is gone.
_SESSION_KEYS = (ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, ROLE_KEY, USER_ID_KEY, EMAIL_KEY)


def auth_headers(state: MutableMapping[str, Any]) -> dict[str, str]:
    """`Authorization: Bearer <access_token>`, or no header when signed out.

    An empty dict rather than an empty Bearer: sending `Bearer ` turns an
    anonymous-but-allowed call into a 401.
    """
    token = state.get(ACCESS_TOKEN_KEY)
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_authenticated(state: MutableMapping[str, Any]) -> bool:
    return bool(state.get(ACCESS_TOKEN_KEY))


def current_role(state: MutableMapping[str, Any]) -> str:
    return str(state.get(ROLE_KEY) or "")


def is_admin(state: MutableMapping[str, Any]) -> bool:
    return current_role(state) in {"admin", "super_admin"}


def clear_session(state: MutableMapping[str, Any]) -> None:
    for key in _SESSION_KEYS:
        state.pop(key, None)


def _store_tokens(state: MutableMapping[str, Any], payload: dict[str, Any]) -> None:
    state[ACCESS_TOKEN_KEY] = payload["access_token"]
    state[REFRESH_TOKEN_KEY] = payload["refresh_token"]
    if payload.get("role"):
        state[ROLE_KEY] = str(payload["role"])
    if payload.get("user_id"):
        state[USER_ID_KEY] = str(payload["user_id"])


def login(
    state: MutableMapping[str, Any], api_base_url: str, email: str, password: str,
    *, client_factory: Any = httpx.Client,
) -> tuple[bool, str]:
    """`POST /login`. Returns (ok, message-safe-to-display).

    The password is used here and nowhere else -- it is never written to
    session state, never logged, and never included in the returned message.
    """
    if not email or not password:
        return False, "Enter your email and password."
    try:
        with client_factory(timeout=20) as client:
            response = client.post(f"{api_base_url}/login", json={"email": email, "password": password})
    except httpx.HTTPError:
        return False, "Could not reach the server. Try again in a moment."
    if response.status_code != 200:
        # Deliberately identical for "no such account" and "wrong password":
        # a differentiated message is an account-existence oracle.
        return False, "Those credentials were not accepted."
    _store_tokens(state, response.json())
    state[EMAIL_KEY] = email
    return True, "Signed in."


def logout(
    state: MutableMapping[str, Any], api_base_url: str, *, client_factory: Any = httpx.Client,
) -> None:
    """Revokes the refresh token server-side, then clears the session.

    The session is cleared even when the call fails: a client that cannot
    reach the server must still be able to sign out of this browser.
    """
    refresh_token = state.get(REFRESH_TOKEN_KEY)
    if state.get(ACCESS_TOKEN_KEY) or refresh_token:
        try:
            with client_factory(timeout=20) as client:
                client.post(
                    f"{api_base_url}/logout",
                    json={"refresh_token": refresh_token},
                    headers=auth_headers(state),
                )
        except httpx.HTTPError:
            pass
    clear_session(state)


def refresh_tokens(
    state: MutableMapping[str, Any], api_base_url: str, *, client_factory: Any = httpx.Client,
) -> bool:
    """Rotates the token pair through `POST /refresh`. Returns whether it worked.

    The server revokes the old refresh token on use, so the new pair fully
    replaces the old one. A failure clears the session: continuing to hold a
    refresh token the server has already rejected only produces a second 401.
    """
    refresh_token = state.get(REFRESH_TOKEN_KEY)
    if not refresh_token:
        return False
    try:
        with client_factory(timeout=20) as client:
            response = client.post(f"{api_base_url}/refresh", json={"refresh_token": refresh_token})
    except httpx.HTTPError:
        clear_session(state)
        return False
    if response.status_code != 200:
        clear_session(state)
        return False
    _store_tokens(state, response.json())
    return True


def request(
    state: MutableMapping[str, Any],
    api_base_url: str,
    method: str,
    path: str,
    *,
    timeout: float = 60.0,
    client_factory: Any = httpx.Client,
    **kwargs: Any,
) -> httpx.Response:
    """One authenticated call, with the 401 rule applied exactly once.

    On 401: rotate through `POST /refresh` once, retry the original request
    once with the new token, and give up if either step fails -- at which
    point the session is already cleared and the UI shows the login form.
    Bounded on purpose: an unbounded retry against an expired session is an
    accidental credential-stuffing loop against our own API.
    """
    url = f"{api_base_url}{path}"
    base_headers = dict(kwargs.pop("headers", {}) or {})
    headers = {**base_headers, **auth_headers(state)}
    with client_factory(timeout=timeout) as client:
        response = cast(httpx.Response, client.request(method, url, headers=headers, **kwargs))
        if response.status_code != 401 or not state.get(REFRESH_TOKEN_KEY):
            return response
    if not refresh_tokens(state, api_base_url, client_factory=client_factory):
        return response
    retry_headers = {**base_headers, **auth_headers(state)}
    with client_factory(timeout=timeout) as client:
        retry = cast(httpx.Response, client.request(method, url, headers=retry_headers, **kwargs))
    if retry.status_code == 401:
        clear_session(state)
    return retry
