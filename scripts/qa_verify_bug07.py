"""Live proof-of-fix for BUG-07: /draft-history session-scoped ownership."""
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
ts = int(time.time())

with httpx.Client() as c:
    def register(tag):
        email = f"qa.bug07.{tag}.{ts}@qalegalai-testdomain.com"
        r = c.post(f"{BASE_URL}/register", json={
            "email": email, "password": "QaTest!Pass1234", "full_name": f"Bug07 {tag}"
        }, timeout=30)
        return r.json()["access_token"], r.json()["user_id"]

    token_a, uid_a = register("a")
    token_b, uid_b = register("b")
    hdr_a = {"Authorization": f"Bearer {token_a}"}
    hdr_b = {"Authorization": f"Bearer {token_b}"}

    r = c.post(f"{BASE_URL}/chat", json={"question": "cheque bounce ho gaya hai, notice draft karna hai"}, headers=hdr_a, timeout=90)
    session_id = r.json()["session_id"]
    print("A: draft trigger:", r.status_code, "session:", session_id)

    # User B (wrong owner) tries to read A's session-scoped draft-history
    r = c.post(f"{BASE_URL}/draft-history", json={"session_id": session_id}, headers=hdr_b, timeout=30)
    print("B (wrong owner) draft-history for A's session ->", r.status_code, r.text[:200])
    assert r.status_code == 403, f"BUG-07 STILL OPEN: expected 403, got {r.status_code}"
    print(">>> BUG-07 CONFIRMED FIXED: cross-user session-scoped draft-history correctly blocked with 403.")

    # User A (real owner) must still be able to read their own session-scoped draft-history
    r = c.post(f"{BASE_URL}/draft-history", json={"session_id": session_id}, headers=hdr_a, timeout=30)
    print("A (real owner) draft-history for own session ->", r.status_code, r.text[:300])
    assert r.status_code == 200, "regression: owner's own access broke"
    print(">>> Owner's own access still works (no regression).")

    # Anonymous caller against an unclaimed session must still work (no forced login regression)
    r2 = c.post(f"{BASE_URL}/chat", json={"question": "RTI application chahiye passport delay ke liye"}, timeout=90)
    anon_session = r2.json()["session_id"]
    r3 = c.post(f"{BASE_URL}/draft-history", json={"session_id": anon_session}, timeout=30)
    print("anonymous draft-history for own unclaimed anonymous session ->", r3.status_code)
    assert r3.status_code == 200, "regression: anonymous unclaimed-session access broke"
    print(">>> Anonymous unclaimed-session access still works (no regression).")

    # cleanup
    c.delete(f"{BASE_URL}/session", params={"session_id": session_id}, headers=hdr_a, timeout=30)
    c.delete(f"{BASE_URL}/session", params={"session_id": anon_session}, timeout=30)
    c.delete(f"{BASE_URL}/me/data", headers=hdr_a, timeout=30)
    c.delete(f"{BASE_URL}/me/data", headers=hdr_b, timeout=30)
    print("\nDONE")
