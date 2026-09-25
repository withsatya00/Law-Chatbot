"""Phase 3 (T068-T071): PostgreSQL persistence across a full backend
restart, for both chat history and the lifecycle draft. Call with
--pre before restarting the backend, and --post after restarting it.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-full-20260924")
REQ = OUT / "requests"
RESP = OUT / "responses"

results: dict = json.load(open(OUT / "results.json", encoding="utf-8"))
state: dict = json.load(open(OUT / "state.json", encoding="utf-8"))


def _save_results():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def call(client, tid, method, path, **kw):
    started = time.time()
    entry = {"tid": tid, "method": method, "url": path, "meta": {"feature": "postgres_persistence_restart"}}
    try:
        resp = await client.request(method, f"{BASE_URL}{path}", timeout=kw.pop("timeout", 60.0), **kw)
        entry["http_status"] = resp.status_code
        entry["latency_s"] = round(time.time() - started, 2)
        try:
            entry["body"] = resp.json()
        except Exception:
            entry["body_text"] = resp.text[:500]
    except Exception as exc:
        entry["http_status"] = None
        entry["error"] = f"{type(exc).__name__}: {exc}"
    if "body" in entry:
        with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
            json.dump(entry["body"], f, ensure_ascii=False, indent=2)
    results[tid] = entry
    _save_results()
    print(f"[{tid}] {method} {path} -> {entry.get('http_status')} ({entry.get('latency_s')}s)", flush=True)
    return entry


async def main(phase):
    hdr_a = {"Authorization": f"Bearer {state['token_a']}"}
    async with httpx.AsyncClient() as client:
        tag = "pre" if phase == "pre" else "post"
        tid_chat = "T068_history_pre_restart" if phase == "pre" else "T070_history_post_restart"
        r = await call(client, tid_chat, "GET", "/history", params={"session_id": state.get("multiturn_session")}, headers=hdr_a)
        msg_count = len((r.get("body") or {}).get("messages", []))
        print(f"{tag}-restart message count:", msg_count)

        if phase == "pre":
            state["pre_restart_msg_count"] = msg_count
        else:
            state["post_restart_msg_count"] = msg_count
            match = msg_count == state.get("pre_restart_msg_count") and msg_count > 0
            results["T070_history_post_restart"]["match_pre_restart"] = match
            _save_results()

        with open(OUT / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"PHASE3-{tag.upper()} DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["pre", "post"], required=True)
    args = p.parse_args()
    asyncio.run(main(args.phase))
