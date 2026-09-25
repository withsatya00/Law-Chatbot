# P2 — Real Redis concurrency and failure coverage (2026-09-14)

Status: implemented and executed. `tests/test_p2_real_redis.py`: **4 passed** against real disposable Redis at `127.0.0.1:36379`; existing security/production-reliability tests: **17 passed**; Ruff passed.

## Verified behavior

- Two distinct clients issue 30 concurrent requests each through the actual RateLimitMiddleware. With a limit of 7, each receives exactly 7 HTTP 200 and 23 HTTP 429 responses. Only 14 requests reach the route. Real Redis counters equal 30, with TTLs in (0, 60] and Retry-After headers on rejected responses.
- New minute admits requests into a fresh counter. Redis key expiration is verified using PEXPIRE and an actual delayed read. The implementation is a fixed-minute bucket, not a sliding-window algorithm; the test advances only the application's bucket clock rather than waiting a minute.
- Both disconnected TCP connections and unresponsive connections are exercised through a test-owned loopback proxy forwarding real Redis traffic. A real Redis client raises network errors (no monkeypatched Redis exceptions). Ten concurrent requests during each outage fail open as designed; health probes remain available.
- The same Redis connection pool reconnects after network recovery and again rejects the already-over-limit client, preserving the pre-outage counter.
- Semantic cache returns a cache miss and tolerates writes during a real network failure; the previously cached value is readable after recovery.

## Isolation and limits

No Redis service was stopped/restarted. No FLUSHDB/FLUSHALL or shared key deletion. Every test uses UUID-owned client/cache keys and deletes only its exact keys. No production .env changes or paid provider calls. HTTP requests use an in-process ASGI transport; Redis connections use real TCP. The proxy simulates loss of Redis connectivity, not Redis process restart/persistence. Test Redis socket timeouts are 0.3 seconds with retries disabled for bounded failure tests; these results do not establish production timeout latency under deployment settings.

These tests cover the requested pending Redis counter/concurrency and actual network-failure behavior. They retain the application's existing fail-open policy; no runtime limiter behavior changed.

## Run

The disposable Redis must be available on port 36379. If provisioning is needed, avoid colliding with the main compose project:

```powershell
docker compose -p legal_ai_phase1_test -f docker-compose.phase1-test.yml up -d redis
$env:PHASE1_LIVE_TESTS='1'
.venv/Scripts/python.exe -m pytest tests/test_p2_real_redis.py -q --basetemp=.pytest-local/p2-real-redis
```

Without PHASE1_LIVE_TESTS=1 these tests skip. When opted in, an unavailable disposable Redis fails the tests instead of silently skipping them.
