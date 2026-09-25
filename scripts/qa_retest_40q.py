"""Retest driver for the qa-40q-multilingual master QA prompt, run against the
live local server after today's BUG-01..BUG-08 fixes. Reuses the EXACT
translated inputs from the original qa-40q-multilingual-20260921 run (stored
in that directory's requests/*.json) for a fair, apples-to-apples retest --
not re-translated, so any difference in outcome is attributable to the code
changes, not translation variance.

Writes requests/responses/exports into a fresh OUTPUT_DIR, plus a single
results.json summary, mirroring the original run's layout.
"""

import json
import time
import traceback
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
ORIGINAL_DIR = Path("C:/Law Chatbot/Law Chatbot/qa-40q-multilingual-20260921")
OUTPUT_DIR = Path("C:/Law Chatbot/Law Chatbot/qa-40q-retest-20260921")
REQUESTS_DIR = OUTPUT_DIR / "requests"
RESPONSES_DIR = OUTPUT_DIR / "responses"
EXPORTS_DIR = OUTPUT_DIR / "exports"

for d in (REQUESTS_DIR, RESPONSES_DIR, EXPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

QIDS = [f"Q{n:02d}" for n in range(1, 25)] + [
    "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31",
    "Q32", "Q33", "Q34",
    "Q35", "Q36", "Q37",
    "Q38a", "Q38b", "Q39a",
    "Q40",
]

# Session-group bookkeeping: which QIDs should inherit which prior QID's
# returned session_id, mirroring the original run's own grouping exactly.
SESSION_INHERITS_FROM = {
    "Q26": "Q25", "Q27": "Q25", "Q28": "Q25", "Q29": "Q25", "Q30": "Q25", "Q31": "Q25",
    "Q33": "Q32",
    "Q34": "Q25",  # explicit return to M1
    "Q36": "Q35", "Q37": "Q35",
    "Q38b": "Q38a", "Q39a": "Q38a",
}

results: dict[str, dict] = {}
session_ids: dict[str, str] = {}  # QID -> session_id returned for that QID's turn


def load_original_question(qid: str) -> dict:
    with open(ORIGINAL_DIR / "requests" / f"{qid}.json", encoding="utf-8") as f:
        return json.load(f)


def send_chat(qid: str, client: httpx.Client) -> None:
    payload = load_original_question(qid)
    payload.pop("session_id", None)
    parent = SESSION_INHERITS_FROM.get(qid)
    if parent:
        payload["session_id"] = session_ids[parent]

    with open(REQUESTS_DIR / f"{qid}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    started = time.time()
    entry = {
        "qid": qid, "sent_text": payload["question"], "http_status": None,
        "latency_s": None, "session_id": None, "detected_language": None,
        "answer_len": None, "answer_preview": None, "artifact": None,
        "draft": None, "missing_field": None, "no_verified_context": None,
        "error": None,
    }
    try:
        resp = client.post(f"{BASE_URL}/chat", json=payload, timeout=180.0)
        entry["http_status"] = resp.status_code
        entry["latency_s"] = round(time.time() - started, 2)
        if resp.status_code == 200:
            data = resp.json()
            with open(RESPONSES_DIR / f"{qid}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            entry["session_id"] = data.get("session_id")
            session_ids[qid] = data.get("session_id")
            entry["detected_language"] = data.get("detected_language")
            answer = data.get("answer") or ""
            entry["answer_len"] = len(answer)
            entry["answer_preview"] = answer[:220]
            entry["artifact"] = data.get("artifact")
            entry["draft"] = data.get("draft")
            entry["missing_field"] = data.get("missing_field")
            entry["no_verified_context"] = data.get("no_verified_context")
        else:
            entry["error"] = resp.text[:2000]
    except Exception as exc:  # noqa: BLE001
        entry["latency_s"] = round(time.time() - started, 2)
        entry["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"

    results[qid] = entry
    print(f"[{qid}] status={entry['http_status']} lang={entry['detected_language']} "
          f"latency={entry['latency_s']}s draft={'yes' if entry['draft'] else 'no'} "
          f"err={'yes' if entry['error'] else 'no'}", flush=True)

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def export_draft(qid: str, draft_id: str, session_id: str, fmt: str, client: httpx.Client) -> None:
    started = time.time()
    out = {"draft_id": draft_id, "format": fmt, "http_status": None, "latency_s": None,
           "size_bytes": None, "path": None, "error": None}
    try:
        resp = client.post(
            f"{BASE_URL}/draft/export",
            json={"draft_id": draft_id, "session_id": session_id, "format": fmt},
            timeout=120.0,
        )
        out["http_status"] = resp.status_code
        out["latency_s"] = round(time.time() - started, 2)
        if resp.status_code == 200:
            path = EXPORTS_DIR / f"{qid}_{draft_id}.{fmt}"
            path.write_bytes(resp.content)
            out["size_bytes"] = len(resp.content)
            out["path"] = str(path)
        else:
            out["error"] = resp.text[:2000]
    except Exception as exc:  # noqa: BLE001
        out["latency_s"] = round(time.time() - started, 2)
        out["error"] = f"{type(exc).__name__}: {exc}"
    results.setdefault(qid, {}).setdefault("exports", []).append(out)
    print(f"[{qid}] export {fmt} status={out['http_status']} size={out['size_bytes']} err={'yes' if out['error'] else 'no'}", flush=True)
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    with httpx.Client() as client:
        for qid in QIDS:
            send_chat(qid, client)
            # Export verification required for at least Q29 and Q38 (the
            # master prompt's own instruction). Q38's draft is finalized by
            # Q38b, so export after that turn, not Q38a.
            if qid == "Q29":
                draft = results[qid].get("draft") or {}
                draft_id = draft.get("draft_id")
                if draft_id:
                    export_draft(qid, draft_id, session_ids[qid], "pdf", client)
                    export_draft(qid, draft_id, session_ids[qid], "docx", client)
            if qid == "Q38b":
                draft = results[qid].get("draft") or {}
                draft_id = draft.get("draft_id")
                if draft_id:
                    export_draft(qid, draft_id, session_ids[qid], "pdf", client)
                    export_draft(qid, draft_id, session_ids[qid], "docx", client)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
