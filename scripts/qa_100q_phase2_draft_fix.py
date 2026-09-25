"""Redo T039-T045 with the CORRECT field-fill message for the template that
actually entered drafting mode (cheque_bounce_notice, from T035 Hinglish
trigger) -- the original T039 sent consumer_complaint-shaped fields into a
cheque_bounce_notice session by mistake (test-script bug, not an app bug).
"""
import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-full-20260924")
REQ = OUT / "requests"
RESP = OUT / "responses"
EXP = OUT / "exports"

results = json.load(open(OUT / "results.json", encoding="utf-8"))
state = json.load(open(OUT / "state.json", encoding="utf-8"))


def save():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUT / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


async def call(client, tid, method, path, **kw):
    started = time.time()
    entry = {"tid": tid, "method": method, "url": path, "meta": {"feature": "drafting_lifecycle_fix"}}
    ext = kw.pop("_ext", "bin")
    try:
        resp = await client.request(method, f"{BASE_URL}{path}", timeout=kw.pop("timeout", 180.0), **kw)
        entry["http_status"] = resp.status_code
        entry["latency_s"] = round(time.time() - started, 2)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            entry["body"] = resp.json()
            with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
                json.dump(entry["body"], f, ensure_ascii=False, indent=2)
        else:
            path_out = EXP / f"{tid}.{ext}"
            path_out.write_bytes(resp.content)
            entry["saved_to"] = str(path_out)
            entry["body_bytes"] = len(resp.content)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    results[tid] = entry
    save()
    print(f"[{tid}] {method} {path} -> {entry.get('http_status')} ({entry.get('latency_s')}s) {entry.get('error','')}", flush=True)
    return entry


async def main():
    hdr_a = {"Authorization": f"Bearer {state['token_a']}"}
    lifecycle_session = state.get("lifecycle_session")
    print("lifecycle_session:", lifecycle_session)

    async with httpx.AsyncClient() as client:
        fill_msg = (
            "applicant name is Ravi Kumar. applicant address is 10 Anna Salai Chennai. "
            "mobile number is 9988776655. drawee bank name is HDFC Bank. "
            "cheque amount is Rs 45000. cheque date is 01-08-2026. cheque number is 445566. "
            "date of return memo is 10-08-2026. reason for dishonour is insufficient funds. "
            "facts is that goods were supplied to the respondent and the cheque received as "
            "payment was dishonoured. place is Chennai. respondent name is XYZ Traders. "
            "respondent address is 20 MG Road Chennai."
        )
        r = await call(client, "T039_draft_field_collection_v2", "POST", "/chat",
                        json={"question": fill_msg, "session_id": lifecycle_session}, headers=hdr_a, timeout=180)
        draft = (r.get("body") or {}).get("draft") or {}
        draft_id = draft.get("draft_id")
        print("stage:", draft.get("stage"), "missing:", draft.get("missing_fields"), "draft_id:", draft_id)

        if not draft_id and draft.get("missing_fields"):
            r2 = await call(client, "T039b_draft_field_collection_v2_followup", "POST", "/chat",
                             json={"question": "Please generate the draft now with the details already provided.",
                                   "session_id": lifecycle_session}, headers=hdr_a, timeout=180)
            draft = (r2.get("body") or {}).get("draft") or draft
            draft_id = draft.get("draft_id") or draft_id
            print("stage2:", draft.get("stage"), "draft_id2:", draft_id)

        state["lifecycle_draft_id"] = draft_id
        save()

        if draft_id:
            await call(client, "T040_draft_correction", "POST", "/chat",
                        json={"question": "change the cheque amount to Rs 55000", "session_id": lifecycle_session},
                        headers=hdr_a, timeout=180)

            await call(client, "T041_draft_regeneration", "POST", "/chat",
                        json={"question": "please regenerate the full draft with these updates", "session_id": lifecycle_session},
                        headers=hdr_a, timeout=180)

            await call(client, "T042_draft_version_history", "GET", f"/draft/{draft_id}/versions",
                        params={"session_id": lifecycle_session}, headers=hdr_a)

            await call(client, "T043_draft_export_pdf", "POST", "/draft/export",
                        json={"draft_id": draft_id, "session_id": lifecycle_session, "format": "pdf"}, headers=hdr_a, _ext="pdf")

            await call(client, "T044_draft_export_docx", "POST", "/draft/export",
                        json={"draft_id": draft_id, "session_id": lifecycle_session, "format": "docx"}, headers=hdr_a, _ext="docx")

            await call(client, "T045_draft_translate", "POST", "/draft/translate",
                        json={"draft_id": draft_id, "session_id": lifecycle_session,
                              "target_language": "tamil", "source_language": "english"}, headers=hdr_a, timeout=120)
        else:
            print("STILL no draft_id -- cannot run T040-T045")

    print("DRAFT FIX DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
