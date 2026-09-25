"""Regression tests for security finding C1 (session ownership / IDOR).

Before this fix, `GET/PATCH/DELETE /session/facts`, `PUT
/session/missing-details` and `GET /session` were gated only by
`ConversationMemoryStore.check_access`, which never claimed a session for
anyone until an authenticated `/chat` turn happened to do so -- so any
caller (a different authenticated account, or a fully unauthenticated one)
who knew or guessed a `session_id` could read/write its facts for as long
as it stayed unclaimed, which for a purely anonymous session was forever.

These tests exercise the real app (`TestClient(create_app())`) against the
real, locally-running Mongo/Redis this project's test suite already uses --
no mocking of the ownership gate itself, since that is exactly what is
under test.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import Role, create_access_token

OWNER_TOKEN_HEADER = "X-Session-Owner-Token"


@pytest.fixture(scope="module")
def client():
    # A bare app carrying only the router under test -- `app.main.create_app`'s
    # lifespan preloads the ~2GB embedding model and the BM25 index and starts
    # several background schedulers, none of which this router touches; going
    # through it here would make every test in this file pay that startup
    # cost for nothing. Mongo/Redis are the real, locally-running instances
    # this project's test suite already targets (see e.g.
    # `tests/test_legal_benchmark.py`), connected directly instead.
    from app.api import history
    from app.cache.redis_client import redis_client
    from app.core.exceptions import install_exception_handlers
    from app.database.mongodb import mongodb

    # Force a REAL connection regardless of what an earlier, unrelated test
    # file left behind: several unit tests in this suite monkeypatch
    # `redis_client._client`/`mongodb._client` with an in-memory fake for
    # their own isolation and don't restore it, and `connect()` alone is a
    # no-op once either is non-`None` -- confirmed live, a leaked
    # `SimpleNamespace` fake from an earlier test made every request here
    # fail with `AttributeError: 'SimpleNamespace' object has no attribute
    # 'get'` instead of ever reaching Redis.
    mongodb._client = None
    redis_client._client = None
    asyncio.run(mongodb.connect())
    asyncio.run(redis_client.connect())

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(history.router)

    with TestClient(app) as test_client:
        yield test_client

    # `mongodb`/`redis_client` are process-wide singletons shared with every
    # other test file in this session. `TestClient`'s own event loop is
    # closed by the time this `with` block exits, and both clients got
    # bound to it the moment a real query ran -- leaving them referenced
    # afterward would fail the NEXT test file to touch either with 'Event
    # loop is closed', for reasons that have nothing to do with it (this is
    # not hypothetical: the identical mistake in an earlier version of
    # `tests/test_account_data_erasure.py`'s fixture broke unrelated
    # `test_draft_export.py`/`test_draft_lifecycle.py` runs). `mongodb.
    # close()` is safe here (synchronous under the hood, no loop needed);
    # `redis_client.close()` awaits a real socket teardown that DOES want
    # its original loop, so it is allowed to fail -- either way, the stale
    # reference must not survive past this fixture.
    asyncio.run(mongodb.close())
    try:
        asyncio.run(redis_client.close())
    except Exception:  # noqa: BLE001 - best-effort graceful close; the reset below is what actually matters
        redis_client._client = None


def _auth(user_id: str) -> dict[str, str]:
    token = create_access_token(user_id, Role.user)
    return {"Authorization": f"Bearer {token}"}


def _sid() -> str:
    return f"c1-test-{uuid4()}"


# ---------------------------------------------------------------------------
# a. owner access -- an authenticated caller can read/write/delete their own
#    session's facts, and this is true on the very FIRST call (no prior
#    /chat turn needed to "claim" it first).
# ---------------------------------------------------------------------------


def test_authenticated_owner_can_read_facts_on_first_touch(client: TestClient) -> None:
    session_id = _sid()
    owner = _auth("user-a")

    response = client.get("/session/facts", params={"session_id": session_id}, headers=owner)

    assert response.status_code == 200
    assert response.json() == {
        "confirmed_facts": {}, "assumptions": {}, "missing_details": [], "corrections": [],
    }


def test_authenticated_owner_can_write_and_read_back_a_fact(client: TestClient) -> None:
    session_id = _sid()
    owner = _auth("user-b")

    write = client.patch(
        "/session/facts", params={"session_id": session_id}, headers=owner,
        json={"kind": "confirmed", "key": "city", "value": "Jaipur"},
    )
    assert write.status_code == 200
    assert write.json()["confirmed_facts"]["city"] == "Jaipur"

    # Independent re-fetch, not just the write response, per persistence-
    # verification requirement.
    readback = client.get("/session/facts", params={"session_id": session_id}, headers=owner)
    assert readback.status_code == 200
    assert readback.json()["confirmed_facts"]["city"] == "Jaipur"


def test_authenticated_owner_can_delete_a_fact(client: TestClient) -> None:
    session_id = _sid()
    owner = _auth("user-c")
    client.patch(
        "/session/facts", params={"session_id": session_id}, headers=owner,
        json={"kind": "confirmed", "key": "phone", "value": "9999999999"},
    )

    delete = client.delete("/session/facts/confirmed/phone", params={"session_id": session_id}, headers=owner)
    assert delete.status_code == 200
    assert "phone" not in delete.json()["confirmed_facts"]

    readback = client.get("/session/facts", params={"session_id": session_id}, headers=owner)
    assert "phone" not in readback.json()["confirmed_facts"]


def test_authenticated_owner_can_set_missing_details(client: TestClient) -> None:
    session_id = _sid()
    owner = _auth("user-d")

    response = client.put(
        "/session/missing-details", params={"session_id": session_id}, headers=owner,
        json={"details": ["landlord_name", "rent_amount"]},
    )
    assert response.status_code == 200
    assert response.json()["missing_details"] == ["landlord_name", "rent_amount"]

    readback = client.get("/session/facts", params={"session_id": session_id}, headers=owner)
    assert readback.json()["missing_details"] == ["landlord_name", "rent_amount"]


def test_authenticated_owner_can_retrieve_session(client: TestClient) -> None:
    session_id = _sid()
    owner = _auth("user-e")
    client.patch(
        "/session/facts", params={"session_id": session_id}, headers=owner,
        json={"kind": "confirmed", "key": "x", "value": "y"},
    )

    response = client.get("/session", params={"session_id": session_id}, headers=owner)
    assert response.status_code == 200
    body = response.json()
    assert body["owner_user_id"] == "user-e"
    assert body["confirmed_facts"]["x"] == "y"


# ---------------------------------------------------------------------------
# b. different-user access -- a DIFFERENT authenticated account must never
#    read or write another account's session, from the very first call.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_request",
    [
        lambda c, sid, h: c.get("/session/facts", params={"session_id": sid}, headers=h),
        lambda c, sid, h: c.patch(
            "/session/facts", params={"session_id": sid}, headers=h,
            json={"kind": "confirmed", "key": "k", "value": "v"},
        ),
        lambda c, sid, h: c.delete("/session/facts/confirmed/k", params={"session_id": sid}, headers=h),
        lambda c, sid, h: c.put(
            "/session/missing-details", params={"session_id": sid}, headers=h, json={"details": ["x"]},
        ),
        lambda c, sid, h: c.get("/session", params={"session_id": sid}, headers=h),
        lambda c, sid, h: c.delete("/session", params={"session_id": sid}, headers=h),
    ],
    ids=["get_facts", "patch_facts", "delete_fact", "put_missing_details", "get_session", "delete_session"],
)
def test_different_authenticated_user_is_forbidden(client: TestClient, make_request) -> None:
    session_id = _sid()
    owner = _auth("owner-user")
    stranger = _auth("stranger-user")

    # Owner claims the session first.
    first = client.get("/session/facts", params={"session_id": session_id}, headers=owner)
    assert first.status_code == 200

    response = make_request(client, session_id, stranger)
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# c. unauthenticated access -- once a session belongs to an authenticated
#    account, an unauthenticated caller must never read or write it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_request",
    [
        lambda c, sid: c.get("/session/facts", params={"session_id": sid}),
        lambda c, sid: c.patch(
            "/session/facts", params={"session_id": sid}, json={"kind": "confirmed", "key": "k", "value": "v"},
        ),
        lambda c, sid: c.delete("/session/facts/confirmed/k", params={"session_id": sid}),
        lambda c, sid: c.put("/session/missing-details", params={"session_id": sid}, json={"details": ["x"]}),
        lambda c, sid: c.get("/session", params={"session_id": sid}),
    ],
    ids=["get_facts", "patch_facts", "delete_fact", "put_missing_details", "get_session"],
)
def test_unauthenticated_caller_is_rejected_for_a_user_owned_session(client: TestClient, make_request) -> None:
    session_id = _sid()
    owner = _auth("owner-user-2")
    client.get("/session/facts", params={"session_id": session_id}, headers=owner)

    response = make_request(client, session_id)
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Anonymous sessions: session_id alone must never be a sufficient credential.
# ---------------------------------------------------------------------------


def test_anonymous_first_touch_mints_an_owner_token_distinct_from_session_id(client: TestClient) -> None:
    session_id = _sid()

    response = client.get("/session/facts", params={"session_id": session_id})

    assert response.status_code == 200
    minted = response.headers.get(OWNER_TOKEN_HEADER)
    assert minted, "an anonymous first touch must mint a server-generated ownership credential"
    assert minted != session_id


def test_anonymous_session_rejects_a_caller_with_no_credential_at_all(client: TestClient) -> None:
    session_id = _sid()
    client.get("/session/facts", params={"session_id": session_id})  # claims it anonymously

    response = client.get("/session/facts", params={"session_id": session_id})

    assert response.status_code == 401


def test_anonymous_session_rejects_a_wrong_owner_token(client: TestClient) -> None:
    session_id = _sid()
    client.get("/session/facts", params={"session_id": session_id})

    response = client.get(
        "/session/facts", params={"session_id": session_id},
        headers={OWNER_TOKEN_HEADER: "not-the-real-token"},
    )

    assert response.status_code == 403


def test_anonymous_session_accepts_the_correct_owner_token(client: TestClient) -> None:
    session_id = _sid()
    first = client.get("/session/facts", params={"session_id": session_id})
    token = first.headers[OWNER_TOKEN_HEADER]

    write = client.patch(
        "/session/facts", params={"session_id": session_id},
        headers={OWNER_TOKEN_HEADER: token},
        json={"kind": "confirmed", "key": "note", "value": "hello"},
    )
    assert write.status_code == 200

    readback = client.get(
        "/session/facts", params={"session_id": session_id}, headers={OWNER_TOKEN_HEADER: token},
    )
    assert readback.status_code == 200
    assert readback.json()["confirmed_facts"]["note"] == "hello"


def test_a_different_anonymous_caller_who_only_knows_session_id_is_rejected(client: TestClient) -> None:
    """The core C1 scenario: knowing/guessing `session_id` must not be
    enough on its own -- a second, unrelated anonymous caller (no token)
    must not be able to read the first caller's facts."""
    session_id = _sid()
    victim = client.get("/session/facts", params={"session_id": session_id})
    assert victim.status_code == 200

    attacker = client.get("/session/facts", params={"session_id": session_id})

    assert attacker.status_code == 401


def test_presenting_the_correct_owner_token_while_authenticated_upgrades_ownership(client: TestClient) -> None:
    session_id = _sid()
    anon = client.get("/session/facts", params={"session_id": session_id})
    token = anon.headers[OWNER_TOKEN_HEADER]

    upgraded = client.get(
        "/session/facts", params={"session_id": session_id},
        headers={**_auth("late-login-user"), OWNER_TOKEN_HEADER: token},
    )
    assert upgraded.status_code == 200

    # Now permanently owned by that account -- the old anonymous token no
    # longer works, and a different account is forbidden.
    stale_token_attempt = client.get(
        "/session/facts", params={"session_id": session_id}, headers={OWNER_TOKEN_HEADER: token},
    )
    assert stale_token_attempt.status_code == 401

    owner_attempt = client.get(
        "/session/facts", params={"session_id": session_id}, headers=_auth("late-login-user"),
    )
    assert owner_attempt.status_code == 200

    other_user_attempt = client.get(
        "/session/facts", params={"session_id": session_id}, headers=_auth("someone-else"),
    )
    assert other_user_attempt.status_code == 403
