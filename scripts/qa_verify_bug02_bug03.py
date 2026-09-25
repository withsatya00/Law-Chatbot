"""Live proof-of-fix for BUG-02 (/draft-history 500 crash) and BUG-03
(citation fabrication not caught by validate_grounding), run against the
restarted backend with both fixes applied.
"""
import json
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
ts = int(time.time())

with httpx.Client() as c:
    # --- fresh test user ---
    email = f"qa.bugfix.verify.{ts}@qalegalai-testdomain.com"
    r = c.post(f"{BASE_URL}/register", json={
        "email": email, "password": "QaTest!Pass1234", "full_name": "Bugfix Verify"
    }, timeout=30)
    print("register:", r.status_code)
    token = r.json()["access_token"]
    hdr = {"Authorization": f"Bearer {token}"}

    # ================= BUG-02 proof =================
    print("\n=== BUG-02: /draft-history after a real draft exists ===")
    r = c.post(f"{BASE_URL}/chat", json={
        "question": "cheque bounce ho gaya hai, notice draft karna hai"
    }, headers=hdr, timeout=90)
    session_id = r.json()["session_id"]
    print("draft trigger:", r.status_code, "stage:", (r.json().get("draft") or {}).get("stage"))

    fill_msg = (
        "applicant name is Test User, applicant address is 1 Test Road Pune, "
        "mobile 9123456780, drawee bank name is HDFC Bank, cheque amount Rs 10000, "
        "cheque date 01-01-2026, cheque number 111111, date of return memo 05-01-2026, "
        "reason for dishonour insufficient funds, facts: goods were supplied and the "
        "cheque bounced, place Pune, respondent address is 2 Test Road Pune, "
        "respondent name is Test Traders"
    )
    r = c.post(f"{BASE_URL}/chat", json={"question": fill_msg, "session_id": session_id}, headers=hdr, timeout=180)
    draft = (r.json().get("draft") or {})
    print("fill fields:", r.status_code, "stage:", draft.get("stage"), "draft_id:", draft.get("draft_id"))
    draft_id = draft.get("draft_id")

    r = c.post(f"{BASE_URL}/draft-history", json={"limit": 50}, headers=hdr, timeout=30)
    print("draft-history status:", r.status_code)
    print("draft-history body:", json.dumps(r.json(), indent=2)[:800])
    assert r.status_code == 200, "BUG-02 STILL BROKEN"
    assert any(d["draft_id"] == draft_id for d in r.json().get("drafts", [])), "draft missing from history"
    print(">>> BUG-02 CONFIRMED FIXED: /draft-history returned 200 with the real draft listed.")

    # cleanup this draft
    if draft_id:
        c.delete(f"{BASE_URL}/draft/{draft_id}", params={"session_id": session_id}, headers=hdr, timeout=30)
    c.delete(f"{BASE_URL}/session", params={"session_id": session_id}, headers=hdr, timeout=30)

    # ================= BUG-03 proof =================
    print("\n=== BUG-03: re-run the exact T46 XSS/citation probe ===")
    r = c.post(f"{BASE_URL}/chat", json={
        "question": "<script>alert('xss')</script> what is the punishment for theft?"
    }, headers=hdr, timeout=180)
    d = r.json()
    print("status:", r.status_code)
    print("no_verified_context:", d.get("no_verified_context"))
    print("general_knowledge_used:", d.get("general_knowledge_used"))
    print("confidence:", d.get("confidence"))
    print("sources:", [s.get("label") for s in d.get("sources", [])])
    print("answer preview:", (d.get("answer") or "")[:400])
    with open("qa_bugfix_verify_bug03_response.json", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    c.delete(f"{BASE_URL}/session", params={"session_id": d.get("session_id")}, headers=hdr, timeout=30)
    c.delete(f"{BASE_URL}/me/data", headers=hdr, timeout=30)
    print("\nDONE")
