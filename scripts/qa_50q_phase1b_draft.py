"""Continues the cheque-bounce drafting flow (session_draft3, the one draft
that actually entered 'collecting' stage in phase1) to get a real completed
draft, then exercises edit / translate / export / draft-history / delete on
it -- filling the T24-T26 gap left when T19/T20's Hindi/English drafting
triggers failed to enter drafting mode (BUG, documented separately).
"""
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-50q-full-20260924")
REQ = OUT / "requests"
RESP = OUT / "responses"
EXP = OUT / "exports"

state = json.load(open(OUT / "state.json", encoding="utf-8"))
results = json.load(open(OUT / "results.json", encoding="utf-8"))
hdr_a = {"Authorization": f"Bearer {state['user_a_token']}"}


def save():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUT / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def call(client, tid, method, path, **kw):
    started = time.time()
    url = f"{BASE_URL}{path}"
    resp = client.request(method, url, timeout=kw.pop("timeout", 180.0), **kw)
    entry = {"tid": tid, "method": method, "url": path, "latency_s": round(time.time() - started, 2),
              "http_status": resp.status_code}
    try:
        entry["body"] = resp.json()
        with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
            json.dump(entry["body"], f, ensure_ascii=False, indent=2)
    except Exception:
        entry["body_text"] = resp.text[:2000]
    with open(REQ / f"{tid}.json", "w", encoding="utf-8") as f:
        json.dump(kw.get("json", {}), f, ensure_ascii=False, indent=2, default=str)
    results[tid] = entry
    save()
    print(f"[{tid}] {method} {path} -> {entry['http_status']} ({entry['latency_s']}s)", flush=True)
    return entry


with httpx.Client() as client:
    fill_msg = (
        "Applicant name Priya Verma, address 22 Karve Road Pune, mobile 9123456780, "
        "drawee bank name HDFC Bank, cheque amount Rs 75000, cheque date 01-08-2026, "
        "cheque number 123456, date of return memo 10-08-2026, reason for dishonour "
        "insufficient funds, facts: I supplied goods to the respondent and received this "
        "cheque as payment which bounced due to insufficient funds despite my reminder, "
        "place Pune, respondent address 5 FC Road Pune, respondent name Suresh Traders"
    )
    r = call(client, "T22b_draft3_fill_fields", "POST", "/chat",
             json={"question": fill_msg, "session_id": state["session_draft3"]}, headers=hdr_a)
    draft = (r["body"] or {}).get("draft") or {}
    print("stage:", draft.get("stage"), "missing:", draft.get("missing_fields"), "draft_id:", draft.get("draft_id"))
    state["draft3_id"] = draft.get("draft_id")

    if not state["draft3_id"] and draft.get("missing_fields"):
        # try one more turn with any still-missing fields spelled out explicitly
        r2 = call(client, "T22c_draft3_fill_more", "POST", "/chat",
                   json={"question": "Please generate the draft now with the details I already gave.",
                         "session_id": state["session_draft3"]}, headers=hdr_a)
        draft = (r2["body"] or {}).get("draft") or {}
        print("stage2:", draft.get("stage"), "missing2:", draft.get("missing_fields"), "draft_id2:", draft.get("draft_id"))
        state["draft3_id"] = draft.get("draft_id") or state["draft3_id"]

    save()

    if state.get("draft3_id"):
        did = state["draft3_id"]
        r = call(client, "T23b_draft3_edit_command", "POST", "/chat",
                 json={"question": "change the cheque amount to Rs 85000",
                       "session_id": state["session_draft3"]}, headers=hdr_a)

        r = call(client, "T24b_draft3_translate_hindi", "POST", "/draft/translate",
                 json={"draft_id": did, "session_id": state["session_draft3"],
                       "target_language": "hindi", "source_language": "english"}, headers=hdr_a)

        r = call(client, "T25b_draft3_export_pdf", "POST", "/draft/export",
                 json={"draft_id": did, "session_id": state["session_draft3"], "format": "pdf"}, headers=hdr_a)
        if r["http_status"] == 200:
            resp_bytes_path = EXP / "T25b_draft3.pdf"
            # re-fetch raw bytes since call() only kept json
            resp = client.post(f"{BASE_URL}/draft/export",
                                json={"draft_id": did, "session_id": state["session_draft3"], "format": "pdf"},
                                headers=hdr_a, timeout=120.0)
            resp_bytes_path.write_bytes(resp.content)
            print("pdf size:", len(resp.content))

        r = call(client, "T26b_draft3_export_docx", "POST", "/draft/export",
                 json={"draft_id": did, "session_id": state["session_draft3"], "format": "docx"}, headers=hdr_a)
        if r["http_status"] == 200:
            resp = client.post(f"{BASE_URL}/draft/export",
                                json={"draft_id": did, "session_id": state["session_draft3"], "format": "docx"},
                                headers=hdr_a, timeout=120.0)
            (EXP / "T26b_draft3.docx").write_bytes(resp.content)
            print("docx size:", len(resp.content))

        r = call(client, "T27b_draft_history_user_a", "POST", "/draft-history",
                 json={"limit": 50}, headers=hdr_a)
    else:
        print("STILL no draft3_id -- drafting flow could not complete even on repeated fill attempts")

    save()
    print("PHASE1B DONE")
