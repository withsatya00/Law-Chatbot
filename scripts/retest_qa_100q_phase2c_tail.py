"""Rerun of T095-T100, which were contaminated by T094's rate-limit burst
running before them in phase2b (all returned 429 instead of their real
behavior). Run only after the rate-limit window has cleared.
"""
import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-retest-20260924")
RESP = OUT / "responses"

results = json.load(open(OUT / "results.json", encoding="utf-8"))
state = json.load(open(OUT / "state.json", encoding="utf-8"))


def save():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def call(client, tid, method, path, **kw):
    started = time.time()
    entry = {"tid": tid, "method": method, "url": path, "meta": {"feature": "errors_concurrency_api_compat"}}
    try:
        resp = await client.request(method, f"{BASE_URL}{path}", timeout=kw.pop("timeout", 120.0), **kw)
        entry["http_status"] = resp.status_code
        entry["latency_s"] = round(time.time() - started, 2)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                entry["body"] = resp.json()
                with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
                    json.dump(entry["body"], f, ensure_ascii=False, indent=2)
            except Exception:
                entry["body_text"] = resp.text[:1000]
        else:
            entry["body_text"] = resp.text[:1000]
        entry["response_headers"] = dict(resp.headers)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    results[tid] = entry
    save()
    print(f"[{tid}] {method} {path} -> {entry.get('http_status')} ({entry.get('latency_s')}s) {entry.get('error','')}", flush=True)
    return entry


async def main():
    hdr_a = {"Authorization": f"Bearer {state['token_a']}"}
    async with httpx.AsyncClient() as client:
        await call(client, "T095_404_nonexistent_draft_template", "GET", "/draft-templates/does-not-exist-xyz")
        await call(client, "T096_404_nonexistent_endpoint", "GET", "/this-endpoint-does-not-exist")

        concurrent_session = state.get("multiturn_session")
        async def fire(q, tag):
            return await call(client, f"T097_concurrent_chat_{tag}", "POST", "/chat",
                               json={"question": q, "session_id": concurrent_session}, headers=hdr_a, timeout=150)
        await asyncio.gather(
            fire("concurrent question 1: what is limitation period for filing?", "a"),
            fire("concurrent question 2: kya mujhe lawyer chahiye is case ke liye?", "b"),
            fire("concurrent question 3: what documents should I carry?", "c"),
        )

        if state.get("lifecycle_draft_id"):
            did = state["lifecycle_draft_id"]
            lsession = state.get("lifecycle_session")
            async def edit(instr, tag):
                return await call(client, f"T098_concurrent_edit_{tag}", "POST", "/chat",
                                   json={"question": instr, "session_id": lsession}, headers=hdr_a, timeout=150)
            await asyncio.gather(
                edit("change the place to Mumbai", "a"),
                edit("change the place to Bengaluru", "b"),
            )
        else:
            results["T098_concurrent_edit_a"] = {"tid": "T098_concurrent_edit_a", "error": "SKIPPED: no draft_id"}
            save()

        await call(client, "T099_cors_preflight", "OPTIONS", "/chat",
                    headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                             "Access-Control-Request-Headers": "content-type,authorization"})
        await call(client, "T100_openapi_schema", "GET", "/openapi.json")

        # T094 rerun LAST this time so it doesn't contaminate anything else
        statuses = []
        started = time.time()
        for i in range(70):
            try:
                resp = await client.get(f"{BASE_URL}/draft-templates", timeout=15)
                statuses.append(resp.status_code)
                if resp.status_code == 429:
                    break
            except Exception:
                break
        results["T094_rate_limit_burst"] = {
            "tid": "T094_rate_limit_burst", "http_status": statuses[-1] if statuses else None,
            "latency_s": round(time.time() - started, 2), "total_requests": len(statuses),
            "hit_429": 429 in statuses, "status_counts": {str(s): statuses.count(s) for s in set(statuses)},
        }
        save()
        print(f"[T094_rate_limit_burst] sent={len(statuses)} hit_429={429 in statuses}", flush=True)

    print("TAIL RERUN DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
