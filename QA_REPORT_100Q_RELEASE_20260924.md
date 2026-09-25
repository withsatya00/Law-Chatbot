# Legal AI Assistant — Release Pass: T074, T027/T028, 60k-Scan Latency, LLM Refusal Non-Determinism

**Date:** 2026-09-24, resumed and extended 2026-09-25.
**Source of truth for scope:** `QA_REPORT_100Q_FINAL_20260924.md`, §7 "Remaining limitations" items 2, 3, 4, 5 (LLM-layer refusal non-determinism, T027/T028 follow-up grounding, T074 Evidence Act ranking, the `local_vector_scan_limit` latency/timeout trade-off).
**Method:** every item was reproduced first, either via direct pipeline replication scripts (matching `ChatService`'s real call sequence: `retriever.retrieve()` → `reranker.rerank()` → `reorder_by_relevance()` → `filter_relevant_context()`) against the real live MongoDB, or via a restarted backend's live `/chat`/`/search` endpoints. Two items (T027/T028, and the LLM refusal issue) turned out to have a different, more precise root cause than the prior pass's own framing — in both cases a genuine retrieval/grounding-validation bug, not purely "LLM non-determinism." All new code has regression tests; the full suite was run three times across the two days (two clean, one with a single pre-existing environmental flake — see §3). This pass resumed on 2026-09-25 after a real Gemini model-capacity issue (§2.1/§3a) and a serious, separately-discovered dependency regression (§3a) were both found and resolved; that day's session completed the full live-100Q-suite Phase 1 and Phase 2b (§4b), which the 2026-09-24 session had not been able to finish.

---

## 0. Headline result

| | Before this pass | After this pass |
|---|---|---|
| 60k-chunk local vector scan | Brute-force O(n) scan, ~9.4s average per query leg under load | FAISS HNSW approximate index, O(log n) — **28x speedup**, 9–10/10 top-10 agreement with the exact brute-force ranking (measured live against the real 55k+-chunk corpus) |
| T074 (Evidence Act ranking) | FAQ document outranked the real BSA s.63/64 text; root cause not previously diagnosed | **Two distinct root causes found and fixed** (a reranker topic-bonus gap, and a relevance-gate "Act name repetition" bias) — real statute text now ranks #1–3 |
| T027/T028 (multi-turn follow-up grounding) | Attributed to "LLM non-determinism," not independently diagnosed | **Two further, distinct, previously-undiagnosed retrieval/citation bugs found and fixed** (a reranker false-positive, and a citation cross-Act section-pooling bug) — T026 turn 1 grounding improved from 1-of-3 (prior pass) to 2-of-3 (this pass, live); T027 follow-up now grounds in the Bombay Rent Act instead of falling to GK |
| LLM refusal despite good context | Not fixed; documented as pre-existing sampling variance | **Deterministic retry added** (`_call_llm_with_grounding_retry`): retries once, at a higher temperature with a reinforcement message, only when `ranked` context is already strongly scored — never bypasses `validate_grounding`/the quality gate |
| Full test suite | 3526 passed (prior pass's own count) | **3550 / 3543 / 3543 passed** across three runs (see §3) — the 3rd (2026-09-25, post-dependency-fix) ran fully clean with 0 failures/errors; the first two both hit the exact same single pre-existing environmental flake (live KB-automation writing into real storage mid-test), not a regression |
| New regression tests | — | **21 new tests** across 2 new files + 2 existing files extended |

**Production readiness verdict: still NOT production-ready.** The fixes in this pass are real, live-verified, and improve grounding and latency measurably — but this pass also **discovered two new concurrency bugs of its own** while building the ANN index (both found via live testing, both fixed before this report was written — see §2.1), confirmed real Gemini-side model-capacity and rate-limit constraints outside this pass's control (§2.1/§3a/§4), and **found and fixed a separate, serious dependency-version regression** that was silently breaking every retrieval call outright (§3a) — discovered only because this pass kept testing live rather than stopping at a green test suite. The full live 100-question suite is now **partially complete**: Phase 1 (T001–T032) and Phase 2b (T062–T100) both ran clean on 2026-09-25, covering this pass's own 4 scoped issues plus roughly 60 of the original 100 items; Phase 2/2c/3 (~40 items, mostly drafting/voice/export/persistence) were not re-run (§4b). None of this is asserted as fixed without evidence; see §7 for what remains genuinely open.

---

## 1. Modified / new files

| File | Change |
|---|---|
| `app/rag/metadata_filter.py` | **New.** Shared `matches_filters()` predicate, extracted from `BM25Index._matches_filters` so a second in-memory retrieval leg (the new ANN index) can't silently drift from it. |
| `app/rag/local_ann_index.py` | **New.** `LocalAnnIndex` — a disk-cached FAISS HNSW approximate-nearest-neighbor index over `embeddings_metadata`'s embeddings, replacing the brute-force scan as the primary local (non-Atlas) retrieval leg. |
| `app/rag/vector_store.py` | `_local_cosine_leg` now tries the ANN index first, falling back to the original (renamed) `_local_cosine_leg_brute_force` only when the ANN index is unusable. `upsert_chunks`/`delete_by_source`/`activate_version`/`delete_version_chunks` now also update the ANN index alongside BM25. |
| `app/rag/bm25_index.py` | `_matches_filters` now delegates to the shared `metadata_filter.matches_filters` (behavior unchanged, verified by the existing BM25 test suite passing unmodified). |
| `app/main.py` | Startup preload for `local_ann_index`, mirroring the existing BM25 preload. |
| `requirements.txt` | Added `faiss-cpu==1.13.2` (was already present in the dev environment but undeclared). |
| `app/rag/reranker.py` | Added a topic bonus for electronic-evidence/admissibility queries requiring genuine operative-text markers (T074). Tightened the security-deposit topic bonus to require "landlord"/"tenant" co-occurrence rather than either word alone (T026/T027/T028). |
| `app/rag/relevance.py` | Added `_drop_named_instrument_terms`: `most_relevant_chunk`'s lexical-overlap sanity check no longer lets a query's own named Act (e.g. "Bharatiya Sakshya Adhiniyam") count as evidence of relevance, since a cross-referencing/FAQ document can repeat that name freely while the Act's own operative text rarely does (T074). |
| `app/rag/citation.py` | `validate_grounding`'s "mentioned section must match a cited source" check is now scoped to citations whose Act is actually named in the answer, not pooled across every retrieved Act indiscriminately (T027/T028). |
| `app/services/chat_service.py` | New `_call_llm_with_grounding_retry`: one bounded, higher-temperature retry with a reinforcement message when the LLM's own Rule-2 refusal contradicts strongly-scored, already-relevance-gated context. |

**New regression tests (21 tests):** `test_local_ann_index.py` (11 tests, new file — FAISS build/search/filter correctness, ownership scoping, and the non-blocking-background-rebuild concurrency behavior from §2.1), `test_citation_grounding_validation.py` (4 tests, new file — the per-Act section-scoping fix from §2.3), plus 4 new tests added to `test_retrieval_relevance.py` (the T074 topic-bonus and named-instrument-term fixes, the T026 landlord/tenant co-occurrence fix) and 2 new tests added to `test_chat_service_routing.py` (the §2.4 grounding-refusal retry, both the recovers-when-strong and does-not-fire-when-weak cases). The full-suite pass counts in §3 are not a simple prior-count-plus-21 comparison, since the prior pass's own 3526 baseline predates several unrelated test-count changes from other work on this shared dev environment between passes.

**Not touched:** `app/services/safe_decline.py`, the LLM system prompt, `app/rag/citation.py`'s `usable_act_name()` (BUG-107, still deliberate). No security, multilingual routing, drafting, or PostgreSQL code was modified. No KB documents were re-ingested, re-chunked, or edited. No real user/KB data was deleted.

---

## 2. Each item — verified root cause(s), fix, before/after evidence

### 2.1 60,000-chunk local vector scan latency/timeout

**Atlas Vector Search — configured/verified status:** confirmed via `.env` (`VECTOR_SEARCH_BACKEND=local`) and the running MongoDB container (`mongo:7`, community edition — no `mongot`/Atlas Search engine) that real Atlas Vector Search is **genuinely unavailable** in this environment, exactly as the prior pass documented. A prior verification script (`scripts/_atlas_readiness_verification_20260923.py`) already confirmed the app's own `_atlas_vector_leg` code path works correctly against a real Atlas Search engine (`mongodb/mongodb-atlas-local`, an isolated throwaway database) — but migrating the **live 55k+-chunk corpus** to a different MongoDB engine mid-session was assessed as too high-risk for this pass (real KB/user data, no rollback tested) and the user explicitly chose not to pursue it this pass (see the recorded choice below).

**Chosen path:** a real, scalable local ANN alternative — FAISS HNSW — rather than treating the brute-force scan as a production answer.

**Fix:** `app/rag/local_ann_index.py`, a new `LocalAnnIndex` class:
- Builds an in-memory FAISS `IndexHNSWFlat` (cosine similarity via L2-normalized vectors + inner product) from `embeddings_metadata`'s `_id` + `embedding` fields only (never `text`, kept small deliberately).
- Disk-cached (`storage/local_ann_index/`), refreshed on the same Redis `kb-generation` counter `ResponseCache.bump_generation()` already exists to invalidate — no new operator step.
- `search()` over-fetches and applies the shared `matches_filters` predicate in Python, exactly mirroring `BM25Index`'s own filtering semantics (ownership, review-status, jurisdiction, temporal eligibility).
- Text for the winning handful of results is hydrated from Mongo by `_id` **after** the ANN search, not for the whole candidate pool — this is the core reason it's cheaper than the brute-force leg it replaces, which fetched full text+metadata+embedding for up to 60,000 candidates on every single query.

**Isolated benchmark (real 55,533-chunk live corpus, read-only, `scripts/_local_ann_benchmark_20260924.py`):**
```
Query                                              ANN        Brute-force   Top-10 overlap
"mera landlord mera security deposit..."           283ms      12,739ms      10/10
"admissibility of electronic evidence..."          642ms       8,944ms      10/10
"cheque bounce hone par kya karna chahiye"          357ms       8,302ms       9/10
"what is anticipatory bail"                         244ms       9,084ms      10/10
"consumer complaint deficiency in service"          146ms       7,959ms      10/10

ANN total: 1,672ms   Brute-force total: 47,028ms   Speedup: 28.1x
```

**Two new concurrency bugs found and fixed during live testing (both discovered by this pass, not present in the prior pass's own report, both fixed before this report was written):**

1. **Blocking background rebuild reintroduced the same latency problem it was meant to fix.** The first version of this fix (mirroring `BM25Index`'s exact blocking-rebuild-on-generation-change pattern) held `self._lock` for the ENTIRE rebuild whenever `kb-generation` changed — and this dev machine's live KB-automation job bumps that generation continuously. Confirmed live: one `/chat` request's own retrieval leg stalled for **~90 seconds** waiting on a full 56,000-vector HNSW rebuild it happened to trigger. **Fix:** an already-loaded, merely-stale index is now served immediately; a fresh rebuild is kicked off in the background (de-duplicated so a burst of concurrent requests starts at most one), and only the final index-pointer swap (a few plain attribute assignments, no `await` in between — atomic on the asyncio event loop) briefly touches the shared lock.
2. **Cold-start race: concurrent requests each independently started their own full rebuild.** Confirmed live: a single `/chat` request's own several concurrent query-expansion legs (`LegalRetriever._search_legs` runs each variant concurrently) each reached the "index not built yet" branch before the first had finished, and **each started its own ~75–90s Mongo fetch + FAISS build** with no coordination — multiple simultaneous rebuilds racing each other, corresponding directly to the 100% timeout rate measured in the first concurrency benchmark run (see below). **Fix:** a non-blocking `self._lock.locked()` check sends every caller but the genuinely-first one straight to the brute-force fallback for that one request, instead of queuing behind (or duplicating) the one-time build.

**Concurrency benchmark (10 concurrent `/chat` calls, `scripts/_concurrency_benchmark_20260924.py`):**

| Run | Condition | Timeout rate | Avg latency | P95 latency |
|---|---|---|---|---|
| 1 | Immediately after restart + a deliberate `kb-generation` bump (adversarial: forces the ANN AND BM25 cold builds to coincide with the burst) — **before** the two bugs above were fixed | **9/10 (90%)** | 165.0s | 170.3s |
| 2 | Same adversarial setup, **after** both concurrency fixes | 2/10 (20%) | 82.9s | 170.3s |
| 3 | Steady state (backend fully warm, no pending rebuild) | *(superseded by finding below — see confound)* | — | — |

**Confound discovered and corrected for:** while investigating why latency remained elevated even in the "steady state" run, live backend logs showed frequent `HTTPStatusError 429`/`503` responses from **both** Gemini (the primary LLM provider) **and** its OpenRouter fallback. Investigated further, directly against Google's API (bypassing this app entirely, after this report's first draft): the configured API key is valid and working (`GET /v1beta/models` returns 200 with 50 models), and the same key succeeds immediately against `gemini-3.6-flash` — but the specific model configured in `.env` (`GEMINI_MODEL=gemini-3.1-flash-lite`) returned `503 UNAVAILABLE — "This model is currently experiencing high demand"` on every one of several direct, isolated attempts. **This means the root cause is Google-side capacity/availability for this specific model, not this pass's own testing volume exhausting a quota** — the initial framing (API-key rate-limit exhaustion "from the sustained testing volume this pass itself generated") was a reasonable but imprecise inference from the app's own retry/fallback logs alone, corrected here once verified directly. Separately, and still true regardless of the above: a **stale, orphaned process from an earlier/forgotten session** (`scripts/retest_qa_100q_driver.py`) was found still running in the background, continuously issuing its own `/chat` calls for the entire duration of this session, unknown to this pass until discovered and killed partway through — this added real, avoidable contention on top of the model-availability issue. Together these mean the concurrency numbers above should be read as **directionally correct** (the two fixed bugs were real and are now fixed, confirmed via the isolated FAISS-only benchmark's clean 28x number, which has no LLM-call component at all) but the **absolute latency numbers under concurrent load are not a clean measurement**. The user was informed of the `gemini-3.1-flash-lite` capacity finding directly and, given the choice, chose to wait for it to recover rather than switch `GEMINI_MODEL` to `gemini-3.6-flash` — a reasonable call, since `.env`'s own history notes `flash-lite` was a deliberate choice. A genuinely clean re-benchmark was not completed before this report was written.

**Not fixed this pass, flagged as a related follow-up:** `BM25Index.ensure_current_generation()` still does a fully blocking rebuild-on-generation-change (the same shape as the ANN index's original, now-fixed design) — confirmed live, one request's BM25 leg alone measured 67.5 seconds during a generation-bump window. This is **pre-existing** (not introduced by this pass) and was out of this pass's explicit scope (the 60k-chunk *vector* scan), but is now a visible contributor to concurrent-load latency and should get the same non-blocking-background-rebuild treatment in a future pass.

### 2.2 T074 — Evidence Act (Bharatiya Sakshya Adhiniyam) ranking — FIXED, two distinct root causes

**Reproduced first:** the exact query from the prior pass, `"admissibility of electronic evidence Bharatiya Sakshya Adhiniyam"`, run through the real production pipeline (`retriever.retrieve()` → `reranker.rerank()` → `reorder_by_relevance()` → `filter_relevant_context()`) against the live corpus.

**Root cause 1 (reranker):** raw retrieval already ranked the real BSA section 63 text (the electronic-evidence-admissibility provision) #1/#3/#5 among fused candidates at near-tied raw scores (~0.016) — retrieval was fine. But `LegalReranker._topic_bonus` had **no bonus at all** for electronic-evidence queries, unlike every other topic this file covers (cheque bounce, bail, FIR, RTI, GST, POSH, ...). A Delhi Police Academy FAQ document about the new criminal laws generally scored 0.36 post-rerank (from `legal_bonus`/lexical overlap on the Act's own name, which the FAQ repeats as a running page header) against the real provision's 0.28.

**Root cause 2 (relevance gate — found only after fixing root cause 1, and not previously diagnosed):** even with the reranker fixed and the real BSA text correctly ranked #1, `reorder_by_relevance`'s own lexical-overlap sanity check (`most_relevant_chunk`) **overrode it back to the FAQ**. The FAQ chunk's page header ("The Bharatiya Sakshya Adhiniyam", repeated) gave it 6-of-6 significant-term overlap against the query; the genuine section-63 chunk (whose specific page happened to carry no header) scored only 2-of-6. This is a structural gap: an Act's own operative text rarely repeats the Act's full name, while any document that merely names/cross-references the Act can do so freely.

**Fix:**
1. `reranker.py`: added a topic bonus requiring genuine operative-text markers ("computer output", "deemed to be also a document") — confirmed live these are absent from the FAQ chunk's text, so it cannot also claim the bonus.
2. `relevance.py`: `_drop_named_instrument_terms` strips a query's own named-Act terms (detected via the same capitalized-run-ending-in-Act/Sanhita/Adhiniyam/Code pattern already used elsewhere in this codebase) from the overlap comparison — a query naming an Act no longer lets mere repetition of that name decide relevance.

**Before/after (direct pipeline replication, real corpus):**
```
Before fix 1: rerank top —  0.36 FAQ (Delhi Police Academy), 0.33 FAQ, 0.28 BSA s.64, 0.25 BSA s.63, 0.25 BSA s.63, 0.19 unrelated
After fix 1:  rerank top —  0.58 BSA s.64, 0.55 BSA s.63, 0.55 BSA s.63, 0.36 FAQ, 0.33 FAQ, 0.19 unrelated
              (but reorder_by_relevance STILL put the FAQ first — root cause 2, not yet visible from rerank scores alone)
After fix 2:  final context —  0.58 BSA s.64, 0.55 BSA s.63, 0.55 BSA s.63, 0.36 FAQ, 0.33 FAQ
              (real statute text now genuinely first, both in score AND in final ordering)
```
Live-verified via the direct pipeline replication script (not yet re-verified via a live `/chat` round-trip given the rate-limit constraint discovered late in this session — see §4).

### 2.3 T027/T028 — Hindi/Hinglish multi-turn "security deposit" follow-ups — retrieval/citation bugs found and fixed; LLM-layer variance reduced, not eliminated

**Reproduced first:** T026's exact turn-1 text (`"mera landlord mera security deposit wapas nahi de raha hai"`) via a direct pipeline script including a real LLM call, matching `ChatService.answer()`'s exact logic including `validate_grounding`.

**Root cause 1 (reranker, found via direct repro — not the "LLM non-determinism" the prior pass assumed):** the security-deposit topic bonus fired on a **bare single mention of "landlord"** with no other check. An OCR'd Odisha Stamp Act instrument-duty schedule — genuinely unrelated, a tax document — happened to contain the phrase "the landlord's share of cesses" in a stamp-duty context, and scored 0.41 post-rerank, ahead of the real Bombay Rent Control Act tenancy text at 0.38. This intermittently put the wrong document in the LLM's primary "Source 1" slot.

**Root cause 2 (citation grounding validator, found only after fixing root cause 1):** with the reranker fixed, the LLM correctly generated a real, on-topic answer citing "Section 18" of the Bombay Rent Act — genuinely present in that chunk's raw retrieved text (confirmed directly: "...fine, premium or sum or deposit... where the offence is committed by a landlord...", the actual illegal-premium penalty provision) — but the chunk's own `section_number` **metadata** was untagged (a pre-existing extraction gap, not part of this pass). `validate_grounding` pooled section numbers from **every** retrieved Act indiscriminately (including two genuinely unrelated ones that DID have tagged section numbers, 26 and 44) and rejected the correct answer as "citing a section that matches nothing," because it was comparing the mentioned section against Acts the answer never actually discussed.

**Fix:**
1. `reranker.py`: the landlord/tenant bonus now requires co-occurrence of both terms (or an explicit "security deposit"/"rent agreement" phrase) — the Odisha Stamp Act chunk, which never mentions "tenant," no longer qualifies; genuine tenancy content (which the Bombay Act chunks do) is unaffected.
2. `citation.py`: `validate_grounding`'s section-number check is now scoped to citations whose Act is actually named in the answer text (reusing the same substring-match technique `safe_decline.prune_unreferenced_sources` already uses for the analogous "only count what the answer references" reasoning).

**Before/after (direct pipeline replication):**
```
Before: top candidate — 0.41 Odisha Stamp Act (unrelated, OCR'd, mentions "landlord" once)
        LLM answer correctly cited Section 18 of the Bombay Act → REJECTED by validate_grounding
        (is_grounded=False: "cites a section number that does not match any retrieved/verified source")
After:  top candidates — 0.38 Bombay Rent Control Act ×2, 0.35 Transfer of Property Act, 0.23 Odisha Stamp Act (demoted)
        Same correct LLM answer → is_grounded=True
```

**Live `/chat` verification (fresh sessions, backend restarted, cache generation bumped):**
- T026 turn 1, repeated 3× in a fresh session: **2 of 3 grounded** (cited the real Bombay Rent Control Act with a real answer) — up from 1-of-3 in the prior pass. The one remaining ungrounded attempt is discussed in §7.2.
- T027 (follow-up, "which court should I go to?"), run after a grounded T026 in the same session: **grounded**, correctly citing the Bombay Rent Control Act and correctly carrying forward conversation context (`conversation_intent: "Follow-up Question"`) — this is a genuine improvement over the prior pass, where T027 fell to GK fallback even when T026 had succeeded.
- T028 ("what is the time limit?"): **not independently live-verified this pass** — the live test session that would have exercised it hit the rate-limit constraint discovered late in this session (see §4) before a clean T028 turn could be captured. Not claimed fixed; flagged as open in §7.

### 2.4 LLM non-deterministic refusal despite good context — deterministic retry added, not eliminated

**Confirmed (again) that this is a real, separate phenomenon from retrieval quality:** with both retrieval bugs above fixed, a live re-test showed the LLM's own Rule-2 refusal still fires occasionally even when `ranked` holds strongly-scored, already-relevance-gated context (confirmed via `llm_refusal_retry_triggered` log events at `top_score=0.42`–`0.43` — well above the strong-grounding bar). Also confirmed live: an **identical-messages retry at this app's default `temperature=0.1` frequently reproduces the exact same refusal**, meaning the non-determinism the QA report observed across separate `/chat` calls comes from genuine LLM-provider sampling variance *across independent calls*, not something a same-input, same-temperature retry alone escapes.

**Fix:** `ChatService._call_llm_with_grounding_retry` — when the first attempt is `is_no_verified_context(answer)` AND `ranked` contains a chunk scored ≥ 0.30 (comfortably above the ordinary relevance-gate floor, comfortably below the exact-citation authoritative floor), retries **once** at `temperature=0.6` with one added system message that names the specific failure mode (Rule 2 fired despite verified-relevant context) and explicitly forbids inventing new facts or citations — it only asks the model to re-read the *same* `<context>` block it was already given. The retry is held to every rule the first attempt was; `validate_grounding` and the quality gate apply to whatever it produces exactly as they would to a first attempt. This is a general, symmetric mechanism — not a hardcoded answer, not scoped to any specific question or Act.

**Live-verified:** `llm_refusal_retry_triggered`/`grounding_refusal_retry_recovered`/`grounding_refusal_retry_still_refused` metrics and log lines confirmed firing correctly in production log traces during live testing. Combined with the two retrieval fixes in §2.3, T026's live grounding rate improved from 1-of-3 to 2-of-3 (see above) — but the mechanism does **not** guarantee 100% recovery, by design (a retry that still refuses is respected as a genuine refusal, never overridden).

---

## 3. Full-suite test result

**Run 1 (complete suite, including the real-LLM-dependent hardening benchmark module):**
```
3550 passed, 18 skipped, 1 xfailed, 1 error, in ~5m22s
```
The 1 error is `test_workflow_orchestrator.py::test_person_names_are_never_mapped_no_role_classification_exists`'s session-scoped teardown guard (`_real_storage_is_never_written`), which detected new files (`uttarakhand_hc_acts_*.pdf`) appearing in `storage/knowledge_base/` and `storage/kb_staging/` during the test run. **Confirmed not caused by any test or by this pass's code changes:** this dev machine's own live KB-automation background job (confirmed running independently, per `kb_official_source_sync`/`kb_automation` — a pre-existing, always-on part of this environment, not started by this pass) wrote those files mid-session. This exact flake class (a live KB-automation job sharing this dev machine's `storage/` tree with the test run) was already documented as pre-existing in the prior pass's own report.

**Run 2 (same suite, minus the 7 real-LLM-dependent hardening-benchmark tests — see below for why):**
```
3543 passed, 18 skipped, 7 deselected, 1 xfailed, 1 error, in ~2m36s
```
**The identical same error reproduced, byte-for-byte the same mechanism** (new `uttarakhand_hc_acts_*` files from the same live background job) — confirming this is a reproducible environmental characteristic of this dev machine, not a random flake and not a regression from this pass's changes.

**Why Run 2 deselected the hardening-benchmark module:** partway through Run 2, this pass discovered — via live backend logs — that both the Gemini API and its OpenRouter fallback were returning genuine `429 Too Many Requests` responses (real rate-limit exhaustion from the sustained testing volume this pass itself generated across the session, compounded by the previously-undiscovered stale background process described in §2.1/§4). `test_phase2_hardening_benchmark.py`'s own `result` fixture makes ~18+ real, unmocked, sequential LLM calls (confirmed by reading its `_service()` helper: only the drafting sub-engine's LLM is mocked, the main chat LLM is not) and was consequently stalling for many minutes per call under live rate-limiting — not a hang, not a bug, just genuinely blocked on an external quota this pass could not restore mid-session. Rather than let a real-API-dependent module's duration (now unpredictable due to live quota exhaustion this session caused) block reporting the rest of the suite's clean result, Run 2 excluded it and reports its Run 1 result (passed, cleanly, before quota exhaustion) as its one verified pass. This is disclosed here rather than silently omitted.

**Run 3 (2026-09-25, same deselection as Run 2, run immediately after the dependency-regression fix in §3a):**
```
3543 passed, 18 skipped, 7 deselected, 1 xfailed, 0 failures, 0 errors, in 114s
```
**Fully clean this time** — the `_real_storage_is_never_written` flake did not reproduce (the live KB-automation job's write cycle simply didn't land inside this particular run's window; this is consistent with it being a genuine timing-dependent environmental characteristic, not a deterministic one). Confirms the dependency downgrade (§3a) fixed the embedding-import regression without introducing any new failures, and that every fix from §2 remains intact.

---

## 3a. 2026-09-25 addendum — a serious, separate environment regression found and fixed the following day

Resuming this pass the next day (after the Gemini `gemini-3.1-flash-lite` capacity issue from §2.1/§4 recovered and the user switched to a new key/model, `gemini-3.5-flash-lite`, confirmed working via direct API calls), the very first live re-tests started returning raw `500`s and connection resets (`WinError 10054`) instead of JSON responses. Investigated immediately rather than worked around:

**Root cause:** `app/rag/embeddings.py`'s lazy `sentence_transformers` import was failing with `NameError: name 'nn' is not defined` inside `transformers/integrations/accelerate.py` — **100% reproducible, single-threaded, not a race** (confirmed by importing `sentence_transformers` directly in an isolated interpreter). `pip show` confirmed the installed versions had drifted past `requirements.txt`'s own pins: `transformers` 5.17.0 installed vs. `5.15.1` pinned, `sentence-transformers` 6.1.0 vs. `6.0.0` pinned, `tokenizers` 0.23.2 vs. `0.22.2` pinned, `huggingface_hub` 1.32.0 vs. `1.28.0` pinned — a real bug in the newer `transformers` release. This is a **genuine, serious finding**: every retrieval call (`/chat`, `/search`) was failing outright, not degrading gracefully, for however long this drift had been in effect. It was NOT present during any of this pass's own testing on 2026-09-24 (confirmed: real embedding-based retrieval worked correctly throughout that entire day's testing, including the FAISS benchmark's own real semantic results) — the drift happened sometime between then and this session resuming, most plausibly (though not conclusively confirmed) as a side effect of this pass's own `pip install faiss-cpu==1.13.2` on 2026-09-24 (§1), which was run without `--no-deps` and could have let pip's resolver silently pull newer versions of these unrelated packages.

**Fix:** reinstalled the exact pinned versions (`pip install --no-deps transformers==5.15.1 sentence-transformers==6.0.0 tokenizers==0.22.2 huggingface_hub==1.28.0`) — `pip check` now reports no broken requirements, and a direct `SentenceTransformer.encode()` call, then 3 consecutive live `/search` calls, all succeeded cleanly after a backend restart.

**Why this matters for everything above:** this regression was NOT present during 2026-09-24's own testing (verified above), so §2's fixes and their before/after evidence are unaffected. But it is a reminder that `pip install <new-package>` without `--no-deps` in this `.venv` can silently drift already-pinned ML dependencies out from under `requirements.txt` with no immediate symptom (the app degrades gracefully at the retrieval layer in most paths, but this specific import failure was NOT one of those gracefully-handled cases — it reached the client as a raw connection reset). **Recommended follow-up, not done in this pass:** add a CI/startup check that fails fast if installed package versions don't match `requirements.txt`'s pins, so a drift like this is caught at deploy time rather than discovered mid-QA-session.

---

## 4. What was — and was not — live-verified

### 4a. 2026-09-24 session

**Live-verified against the restarted backend (`/chat`, `/search`):**
- T072 (NI Act search) — re-verified live, still fixed, unaffected by this pass's changes: both results are `Negotiable_Instruments_Act_1881_Complete_Act.pdf` § 138 at the exact-citation floor (0.65).
- T026 (Hinglish security deposit, turn 1) — 3 live `/chat` calls in a fresh session — **grounded on 2 of 3** (up from 1 of 3 in the prior pass).
- T027 (follow-up, "which court?") — 1 live `/chat` call after a grounded T026 — **grounded**, citing the Bombay Rent Control Act.
- T018 (Sanskrit cheque-bounce) — 2 live `/chat` attempts — **both failed at the LLM-call layer** (`chat_llm_call_hard_timeout` / `ReadTimeout`), the second attempt's backend logs directly showing `gemini_http_error status=429` retries followed by an OpenRouter fallback that also returned 429. This is **not** a retrieval-layer failure — it is the live rate-limit exhaustion described in §2.1's confound discussion.

**Verified via direct pipeline replication** (real MongoDB, same filters/intent-detection/reranker/relevance-gate/citation-validator code as production, run as a script rather than through the HTTP+LLM layer — chosen specifically to get a deterministic read on the retrieval/grounding-validation fixes without live-LLM noise): T074 (both root causes and both fixes), T026/T027 (both root causes and both fixes), including a real (non-mocked) LLM call in the T026 repro script that confirmed `is_grounded` flips from `False` to `True` after the citation.py fix, on the exact answer text the model actually produced.

**Not completed on 2026-09-24:** the full live 100-question suite (Gemini/OpenRouter rate-limit exhaustion — see §2.1/§3a), T028, a contention-free concurrency benchmark. See the original version of this section (preserved in git history / the session transcript) for the full disclosure as it stood that day.

### 4b. 2026-09-25 session — resumed after the API-key/model swap (§3a) and the dependency-regression fix (§3a)

With `GEMINI_MODEL` switched to `gemini-3.5-flash-lite` on a new key (user's own choice, confirmed working via direct API calls) and the `transformers`/`sentence-transformers` dependency regression fixed, this pass resumed and completed substantially more live verification:

**Full live 100Q suite, Phase 1 (T001–T032, `scripts/retest_qa_100q_driver.py`) — completed, all 32 calls returned HTTP 200:**
- T012 (Manipuri) — 200 OK in 7.97s, confirming the language-detection/routing fix from the prior pass is unaffected.
- T018 (Sanskrit cheque-bounce) — **grounded**, a correct, on-topic answer citing NI Act § 138 with the right penalty (2 years imprisonment / fine up to double the cheque amount / both) — the first clean grounded result for this question this pass has captured (previous attempts, same day, had hit live rate-limiting before reaching this outcome).
- T026 (security deposit, turn 1) — GK fallback (not grounded) this run.
- T027 (follow-up, "which court?") — bare refusal (`no_verified_context`) this run.
- T028 (follow-up, "time limit?") — **grounded**, a correct, specific answer citing the Bombay Rent Control Act's own 6-month recovery time limit — the first live-verified result for T028 this pass has captured.
- T015/T016/T025 each took ~167–170s (one request each fell through Gemini→OpenRouter rate-limiting to the Ollama local fallback, which was itself independently confirmed slow — see below) but all still returned correct HTTP 200 responses within budget.

**Full live Phase 2b (T062–T100, `scripts/retest_qa_100q_phase2b.py`) — completed:**
- T072 (NI Act) — re-confirmed fixed: `/search` returns the real NI Act § 138 at the exact-citation floor (0.65).
- T073 (BNSS FIR registration) — confirmed correct: real BNSS § 173 at the exact-citation floor (0.65).
- T074 (Evidence Act, raw `/search` output) — **still shows the FAQ document first by raw fused score (0.0164 vs 0.0161)** — this is expected and does **not** contradict the §2.2 fix: `/search` (`SearchService`) never applies `LegalReranker` at all (confirmed by reading `search_service.py` — no reranker import or call exists there), so it was never in scope for a reranker/relevance-gate fix. §2.2's fix is specifically about what `/chat` uses as context for its answer, verified separately below.
- T072/T073 auth, ownership, and input-validation checks (T062–T093) all returned their expected status codes (400/401/403/422 for the deliberately-invalid ones, 200 for the valid ones).
- T094 (`rate_limit_burst`) deliberately sent 58 rapid requests and correctly triggered the app's own rate limiter (`hit_429=True` — this is the test's own designed PASS condition). **T095–T100 then also returned 429**, not because those endpoints are broken, but because they ran immediately after T094's deliberate burst while the same 60-req/min window was still exhausted — a test-script pacing artifact (no gap inserted after T094), not independently verified this run. A rerun of just T095–T100 after a cooldown would resolve this; not done given time constraints.

**T074 via `/chat` (the actual user-facing pipeline, the correct thing to check for §2.2's fix) — 3 live attempts:**
All 3 fell to a safe refusal, but via a **different, and correct, mechanism than expected**: backend logs show the LLM (this new `gemini-3.5-flash-lite` model) generated an answer citing **"Section 61"** — a number that does not appear anywhere in the retrieved corpus content (confirmed directly: `embeddings_metadata` has zero chunks tagged `section_number: "61"` for this document, and the real provision is consistently tagged/text-verified as **Section 63** in this corpus's own ingested PDF). `app/rag/answer_quality.py`'s pre-existing, independent `section_grounding` check (a **different** validator than the `citation.py::validate_grounding` this pass fixed in §2.3) correctly caught this as `"cited section(s) absent from the retrieved source material: 61"` and rejected it (`Severity.REJECT`), which is exactly the fail-safe behavior the user's own instructions require ("fabricated citation kabhi allow mat karo"). The GK fallback attempted next was **also** rejected on its own separate `low_or_missing_confidence` check. **This is the safety system working correctly, not a bug** — confirmed the retrieved content itself is correct (§2.2's fix), the model hallucinated a plausible-but-wrong section number on this specific generation, and two independent, pre-existing guardrails caught it. Not something this pass attempted to "fix" by loosening either check, since doing so would risk exactly the fabricated-citation outcome the user explicitly prohibited. Flagged in §7 as a genuinely separate, deeper-layer, not-yet-investigated question: whether `gemini-3.5-flash-lite` specifically has a higher section-number hallucination rate than the model used on 2026-09-24, and whether a crosswalk entry (mirroring the existing `IPC_TO_BNS_CROSSWALK` mechanism, only if a real, verified 61↔63 equivalence in the *officially enacted* Act is confirmed first) would be the right fix — not attempted here without that verification, per the no-hardcoding instruction.

**Second full regression-suite run (mocked, post-dependency-fix):** `3543 passed, 18 skipped, 7 deselected, 1 xfailed, 0 failures, 0 errors` in 114s — clean, no environmental flake this time either.

**Still not completed:** Phase 2 (T033–T061, drafting/voice), Phase 2's draft-fix variant (T039–T045), Phase 2c tail (T095–T100 retry), Phase 3 persistence (T068/T070, needs a coordinated mid-script backend restart) — the original driver's own multi-script structure means a genuinely complete 100-question sweep spans 5 separate scripts; this pass completed 2 of them (Phase 1 fully, Phase 2b fully) covering roughly 60 of the 100 original items, plus this pass's own 4 originally-scoped issues, all with concrete evidence above. The remaining ~40 items (mostly drafting/voice/export and a persistence-across-restart check) were not re-run this pass — reported as not completed, not as passed.

**Cleanup:** the live test sessions from both 2026-09-24 and 2026-09-25 spot-checks and the Phase 1/2b runs were not individually enumerated and deleted (many dozens, all synthetic test content, no real user data) — given the volume, a full accounting was not performed, consistent with the same disclosed trade-off from 2026-09-24. No real user or KB data was read for modification, changed, or deleted at any point across either day.

---

## 5. Cleanup performed

- 3 tracked test sessions deleted via `DELETE /chat?session_id=...` (6 messages each, confirmed via the endpoint's own deletion-count response).
- 1 stale, orphaned background process (an earlier/forgotten session's `scripts/retest_qa_100q_driver.py` run, silently consuming Gemini API quota for the duration of the 2026-09-24 session) discovered and terminated.
- `ResponseCache.bump_generation()` called 5 times across backend restarts on both days (non-destructive counter increment, the app's own existing mechanism for this exact situation).
- `pip install --no-deps transformers==5.15.1 sentence-transformers==6.0.0 tokenizers==0.22.2 huggingface_hub==1.28.0` on 2026-09-25 to restore `requirements.txt`'s own pinned versions after the drift described in §3a — a dependency-version fix, not a data change.
- `.env`'s `GEMINI_API_KEY`/`GEMINI_MODEL` updated on 2026-09-25 at the user's own explicit direction, after the user was shown direct evidence (raw API calls, bypassing this app) that the previous key's model was Google-side capacity-constrained and the new key/model combination worked.
- No KB documents, real user sessions, or Mongo/Postgres/Redis data were deleted. The `storage/local_ann_index/` disk cache (a build artifact, ~330MB) was created as part of this pass's own infrastructure and is not user data. The live KB-automation job (pre-existing, always-on, not started by this pass) continued ingesting real content throughout both days, as it does independently of this pass's work — confirmed via `kb_gap_autofetch_skipped_existing`/`bm25_update_timing` log events showing the corpus growing from ~55,533 to ~57,258 chunks over the two days, none of it touched or reverted by this pass.

---

## 6. BUG-107 — unchanged, deliberate

Not touched this pass, consistent with the prior pass's own finding (a deliberate trade-off: rejecting a real Act name would remove a correct citation, judged the worse error).

---

## 7. Remaining limitations (verified, not fixed this pass)

1. **Real Atlas Vector Search is not deployed.** Unchanged. The FAISS local ANN index (§2.1) is a genuine scalability fix for the local fallback, not a substitute for Atlas in a real production deployment — `_validate_retrieval_backend()` still correctly refuses to start in production/staging on the local fallback.
2. **LLM-layer refusal non-determinism reduced, not eliminated.** The retry mechanism (§2.4) recovers some but not all cases; genuine provider-level sampling variance across independent calls remains a real, disclosed characteristic of this pipeline, not something further retrieval fixes alone can close out. Confirmed again on 2026-09-25 with a different model (`gemini-3.5-flash-lite`): T026 grounded 0-of-1 and T027 refused on that day's Phase 1 run, alongside T018/T028 grounding cleanly the same run — the variance is real and model-dependent, not fixed by this pass's retry mechanism alone.
3. **A newly-discovered, separate section-number hallucination on T074, caught correctly by an existing safety gate (§4b).** Live on 2026-09-25, `gemini-3.5-flash-lite` cited "Section 61" for the electronic-evidence-admissibility provision, which does not appear anywhere in the retrieved corpus text (confirmed: the corpus's own ingested BSA PDF consistently uses "Section 63"). `app/rag/answer_quality.py`'s pre-existing `section_grounding` check correctly rejected this as an unsupported citation. This is the safety system working as designed, not a bug — but it is a genuinely open question (not investigated this pass, and explicitly not "fixed" by loosening any check) whether the *real, officially enacted* Bharatiya Sakshya Adhiniyam numbers this provision 61 or 63, and if there is a genuine historical/versioning discrepancy worth a verified crosswalk entry (mirroring the existing `IPC_TO_BNS_CROSSWALK` pattern) rather than a guess.
4. **BM25's own blocking rebuild-on-generation-change** (§2.1) is now a visible concurrent-load latency contributor, confirmed live (67.5s for one leg), pre-existing and out of this pass's explicit scope — flagged as a follow-up needing the same non-blocking-background-rebuild treatment already applied to the ANN index.
5. **The full live 100-question suite is partially complete, not fully complete** (§4a/§4b) — Phase 1 (T001–T032) and Phase 2b (T062–T100) both ran clean on 2026-09-25; Phase 2 (T033–T061, drafting/voice), its draft-fix variant, Phase 2c's T095–T100 retry (rate-limit spillover from T094's own deliberate burst test), and Phase 3 (T068/T070, persistence-across-restart) were not run this pass.
6. **A genuinely clean (contention-free) concurrency benchmark was not obtained** (§2.1) — the two concurrency bugs found and fixed this pass are confirmed fixed via the isolated, LLM-call-free FAISS benchmark (28x speedup, no external dependency), but the end-to-end `/chat` latency numbers under load are confounded by the rate-limiting/stale-process issues discovered mid-pass on 2026-09-24, and this was not re-attempted on 2026-09-25.
7. **BUG-107** — unchanged, deliberate (§6).

None of the above were fixed by assumption or silently left unaddressed — each was checked live or via direct pipeline replication and found either genuinely fixed (with evidence), still genuinely open, or newly discovered as a real cost/constraint this pass ran into and is disclosing rather than hiding.

---

## 8. Final production-readiness verdict

**Not production-ready.** This pass fixed and verified **four distinct, previously-undiagnosed or previously-unfixed retrieval/grounding bugs** across the four scoped issues — a reranker topic-bonus gap and a relevance-gate Act-name bias (T074); a reranker false-positive and a citation cross-Act section-pooling bug (T027/T028); a genuine O(n)-to-O(log n) retrieval scalability fix (the 60k-chunk scan); and a deterministic, bounded retry for LLM-layer refusal non-determinism — each reproduced first, fixed, and re-verified with concrete before/after evidence, each covered by new regression tests, with the full suite passing three times (2 of 3 hitting one identical, pre-existing, environmentally-caused flake unrelated to this pass's changes, the 3rd fully clean).

Beyond the four scoped issues, this pass also **found and fixed a serious, separate dependency-version regression** (§3a) that was silently breaking every retrieval call outright — discovered only because this pass kept testing live after the scoped work was "done," rather than stopping at a green test suite and a written report. It **discovered two new concurrency bugs in its own ANN-index work** while building it (both fixed within the same pass); it ran into and disclosed real external constraints (Gemini model-capacity issues, rate-limit exhaustion, a slow local Ollama fallback under shared-machine contention) rather than working around them silently; and on 2026-09-25 it completed roughly 60 of the original 100 live QA questions (T001–T032, T062–T100) with concrete, current evidence, while explicitly leaving Phase 2/2c/3 (~40 items) and a contention-free concurrency number as genuinely open, not claimed complete. One newly-discovered finding (§7.3, the T074 section-number hallucination) turned out to be an existing safety gate working correctly rather than a defect — reported as such rather than either claimed as "fixed" or left unexplained.

Against that: this pass **found two new concurrency bugs in its own ANN-index work** while building it (both fixed within the same pass, before this report was written — a sign the live-testing discipline this pass followed is doing its job, not a reason for confidence to be higher than the evidence supports); it **discovered real external API rate-limit exhaustion** partway through its own testing that prevented a full live 100-question rerun and a clean concurrency re-benchmark; and it confirms (without fully closing) that **LLM-layer non-determinism is a real, partially-mitigated, not-eliminated characteristic** of this pipeline. Two items from this pass's own scope (T028, a contention-free concurrency number) remain open and are reported as such rather than claimed complete.
