# Latency / Error Targets — Legal AI Assistant

No numeric SLA existed anywhere in this repo before this document (checked
`docs/deployment.md`, `ARCHITECTURE_REPORT.md`,
`docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md` — none define one; `deployment.md`'s
"Targets" section is about hosting platforms, not performance). Per an
explicit decision on 2026-09-14: rather than invent arbitrary numbers, this
formalizes the baseline that five live QA sessions
(`docs/qa/QA_TEST_MATRIX_20260911.md`) already independently, repeatedly
observed against the real Gemini provider and real local MongoDB/Redis — and
uses it as the standing target for grading load-test PASS/FAIL from here on.
Revise this document (not just the numbers in a session log) if a materially
different target is ever agreed.

## Chat (`/chat`, `/chat/stream`) and draft generate/regenerate

| Metric | Target | Basis |
|---|---|---|
| Typical response time | 15-90s | Observed baseline across sessions 1-5, dozens of live calls |
| Acceptable worst case (single request) | up to 250s | Observed spikes on complex regeneration/restyle and first-time draft generation (sessions 3-4); a request exceeding this may reasonably be client-timed-out |
| Data/session integrity on any failure | 100% — zero tolerance | A failed/timed-out request must never corrupt the draft, lose conversation memory, or crash the session (verified repeatedly, sessions 1-5) |
| Recoverability of a failed request | Must offer an explicit, working "retry" path | BUG-013a (still open) narrows this: only the literal word "retry" is guaranteed to work today — rephrasing the original request is not yet equivalent |

## Document upload / PDF processing (`/upload`, KB indexing)

No prior session measured this in isolation — this document establishes the
first baseline rather than comparing against history. Until a future
measurement session revises it:

| Metric | Target |
|---|---|
| Small text/PDF upload (a few KB, no OCR) | Under 10s |
| Scanned/OCR-requiring PDF | Under 60s per document (highly size/page-count dependent — see the session 5 startup incident, where large scanned PDFs took multiple minutes each under the incremental reindexer) |
| Upload failure behavior | Must return an explicit error, never a silent drop; the document must never appear "uploaded" in the UI without actually being retrievable |

## Error rate

No historic error-rate percentage was ever tracked (only individual live
incidents). Target, until superseded: **transient provider/DB errors must
always degrade to an explicit, recoverable fallback — never a raw 5xx with no
explanation, and never silent data loss.** A "no verified context" or
"temporary issue, say retry" response is an acceptable outcome under this
target; an unhandled exception, a crashed session, or a draft silently
corrupted/duplicated is not, regardless of how rarely it happens.

## Concurrency

No prior session tested more than one live call at a time. First concurrency
target, agreed 2026-09-14: the system must remain correct (no cross-request
data mixing, no crashed process) under **3-5 concurrent users** performing
chat/upload/drafting simultaneously on this single-instance local deployment.
Throughput/latency degradation under that load is expected and acceptable;
data corruption or a crashed server process is not. Testing beyond this level
(10+, 25+ concurrent) was explicitly deferred — see the session 6 load-test
write-up in `QA_TEST_MATRIX_20260911.md` for when/why to revisit.
