"""Concurrency benchmark (2026-09-24, QA release pass): average/P95 latency
and timeout rate for concurrent `/chat` calls against the restarted backend,
with the FAISS local-ANN fix + lock-decoupled background rebuild in place.

Mirrors the prior QA passes' own methodology (N concurrent /chat calls,
measuring wall-clock per call) -- see config.py's
`_validate_retrieval_backend` docstring for the historical 6-concurrent-
request baseline (100-197s/call on the old brute-force scan).
"""

import asyncio
import statistics
import time

import httpx

QUESTIONS = [
    "mera landlord mera security deposit wapas nahi de raha hai",
    "what is anticipatory bail",
    "cheque bounce hone par kya karna chahiye",
    "consumer complaint deficiency in service",
    "admissibility of electronic evidence Bharatiya Sakshya Adhiniyam",
    "what is zero fir",
    "RTI application kaise file karte hain",
    "domestic violence ke against kya legal action le sakte hain",
    "GST registration kaise karein",
    "POSH act workplace sexual harassment",
]


async def _one_call(client: httpx.AsyncClient, question: str, idx: int) -> dict:
    started = time.perf_counter()
    try:
        response = await client.post("/chat", json={"question": question}, timeout=170)
        elapsed = time.perf_counter() - started
        return {"idx": idx, "question": question, "elapsed": elapsed, "status": response.status_code, "timeout": False}
    except (httpx.TimeoutException, httpx.ReadTimeout):
        elapsed = time.perf_counter() - started
        return {"idx": idx, "question": question, "elapsed": elapsed, "status": None, "timeout": True}


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(_one_call(client, q, i) for i, q in enumerate(QUESTIONS))
        )
        total_wall = time.perf_counter() - started

    print(f"=== {len(QUESTIONS)} concurrent /chat calls, total wall time {total_wall:.1f}s ===\n")
    for r in results:
        status = "TIMEOUT" if r["timeout"] else f"HTTP {r['status']}"
        print(f"  [{r['idx']}] {r['elapsed']:7.1f}s  {status}  {r['question'][:50]}")

    elapsed_values = [r["elapsed"] for r in results]
    timeouts = sum(1 for r in results if r["timeout"])
    elapsed_values.sort()
    p95_index = min(len(elapsed_values) - 1, int(len(elapsed_values) * 0.95))
    print("\n=== Summary ===")
    print(f"Average latency: {statistics.mean(elapsed_values):.1f}s")
    print(f"Median latency:  {statistics.median(elapsed_values):.1f}s")
    print(f"P95 latency:     {elapsed_values[p95_index]:.1f}s")
    print(f"Max latency:     {max(elapsed_values):.1f}s")
    print(f"Timeout rate:    {timeouts}/{len(results)} ({100*timeouts/len(results):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
