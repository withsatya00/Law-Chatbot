"""Second attempt: fresh session, single well-formed trigger+fill message in
one turn where possible, to reliably reach a completed draft for T040-T045.
"""
import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-retest-20260924")
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
    entry = {"tid": tid, "method": method, "url": path, "meta": {"feature": "drafting_lifecycle_fix2"}}
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

    async with httpx.AsyncClient() as client:
        # single combined trigger+fields message, all "is"/colon phrasing
        combined = (
            "cheque bounce ho gaya hai, notice draft karna hai. "
            "applicant name is Ravi Kumar. applicant address is 10 Anna Salai Chennai. "
            "mobile number is 9988776655. drawee bank name is HDFC Bank. "
            "cheque amount is Rs 45000. cheque date is 01-08-2026. cheque number is 445566. "
            "date of return memo is 10-08-2026. reason for dishonour is insufficient funds. "
            "facts is that goods were supplied to the respondent and the cheque received as "
            "payment was dishonoured. place is Chennai. respondent name is XYZ Traders. "
            "respondent address is 20 MG Road Chennai."
        )
        r = await call(client, "T039_draft_field_collection_v3", "POST", "/chat",
                        json={"question": combined}, headers=hdr_a, timeout=180)
        body = r.get("body") or {}
        session_id = body.get("session_id")
        draft = body.get("draft") or {}
        draft_id = draft.get("draft_id")
        print("session:", session_id, "stage:", draft.get("stage"), "missing:", draft.get("missing_fields"), "draft_id:", draft_id)
        state["lifecycle_session"] = session_id

        if not draft_id and draft.get("missing_fields"):
            still_missing = ", ".join(draft["missing_fields"])
            r2 = await call(client, "T039b_draft_field_collection_v3_followup", "POST", "/chat",
                             json={"question": f"generate the notice now, remaining fields not applicable or already given: {still_missing}",
                                   "session_id": session_id}, headers=hdr_a, timeout=180)
            draft = (r2.get("body") or {}).get("draft") or draft
            draft_id = draft.get("draft_id") or draft_id
            print("stage2:", draft.get("stage"), "draft_id2:", draft_id)

        state["lifecycle_draft_id"] = draft_id
        save()

        if draft_id:
            await call(client, "T040_draft_correction", "POST", "/chat",
                        json={"question": "change the cheque amount to Rs 55000", "session_id": session_id},
                        headers=hdr_a, timeout=180)

            await call(client, "T041_draft_regeneration", "POST", "/chat",
                        json={"question": "please regenerate the full draft with these updates", "session_id": session_id},
                        headers=hdr_a, timeout=180)

            await call(client, "T042_draft_version_history", "GET", f"/draft/{draft_id}/versions",
                        params={"session_id": session_id}, headers=hdr_a)

            await call(client, "T043_draft_export_pdf", "POST", "/draft/export",
                        json={"draft_id": draft_id, "session_id": session_id, "format": "pdf"}, headers=hdr_a, _ext="pdf")

            await call(client, "T044_draft_export_docx", "POST", "/draft/export",
                        json={"draft_id": draft_id, "session_id": session_id, "format": "docx"}, headers=hdr_a, _ext="docx")

            await call(client, "T045_draft_translate", "POST", "/draft/translate",
                        json={"draft_id": draft_id, "session_id": session_id,
                              "target_language": "tamil", "source_language": "english"}, headers=hdr_a, timeout=120)
        else:
            print("STILL no draft_id -- T040-T045 will remain unfulfilled; documenting as-is")

    print("DRAFT FIX2 DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
