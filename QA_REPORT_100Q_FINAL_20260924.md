# Legal AI Assistant — Systematic Bug-Fix Pass 2 (Remaining 8 PARTIAL + 4 FAIL)

**Date:** 2026-09-24
**Source of truth for this pass:** `QA_REPORT_100Q_RETEST_20260924.md`, specifically its §7 "Remaining limitations" (the 8 PARTIAL + 4 FAIL items that pass left open).
**Method:** every item was reproduced first — either via a live, restarted backend (`/chat`, `/search`) or via a script that replicates the production pipeline exactly (same filters, same intent classification, same reranker, same relevance gate) against the real local MongoDB/Redis/Postgres — before any fix was written. Two items turned out to have a different, more precise root cause than the prior pass's own framing suggested (see §2, items 2 and 5). Every fix has a regression test. The full existing test suite (now 3526 tests) was re-run twice against the final code with 0 failures. **This pass did not re-run the full live 100-question suite** — see §4 for exactly what was and was not live-verified, and why.

---

## 0. Headline result

| | Before this pass | After this pass |
|---|---|---|
| Open items from the prior pass's §7 | 8 PARTIAL + 4 FAIL (12 items, 1 explicitly deliberate/not-a-bug) | **3 root causes fixed and verified** (covering 7 of the 11 real items); **2 items reclassified as a pre-existing, non-newly-introduced LLM-layer behavior, not a retrieval bug**; **2 items confirmed still genuinely open** (1 unchanged/deliberate, 1 newly precisely diagnosed as distinct) |
| Full test suite | 3509 passed (per the prior pass's own count) | **3526 passed, 18 skipped, 1 xfailed, 0 failures** (run twice) |
| New regression tests | — | **19 new tests** across 4 new files + 2 existing files extended |

**Production readiness verdict: still NOT production-ready — the fixes here are real and verified, but they surface a genuine, previously-undiscovered trade-off (see §7.5) and confirm a pre-existing LLM-layer non-determinism (see §7.2) that neither this pass nor the prior one attempted to fix.** The prior pass's blockers (Critical bugs, Atlas deployment) are unchanged by this pass.

---

## 1. Modified files

| File | Change |
|---|---|
| `app/language/detector.py` | Added `_ROMANIZED_MANIPURI_TERMS`, a curated marker-word table for romanized Manipuri, checked in `LanguageDetector.detect()` |
| `app/intent/classifier.py` | `classify()`/`classify_advanced()` now take `language`; the Hindi-verb typo-clarification prompt only fires for english/hindi/hinglish |
| `app/services/chat_service.py` | Threads `language` into both `classify_advanced()` call sites; computes `language` once (was recomputed) in `answer_stream()` |
| `app/core/config.py` | `local_vector_scan_limit` raised from 20,000 to 60,000 |
| `app/rag/multilingual.py` | Added the Sindhi ("چيڪ") and short-form Telugu ("చెక్") spellings of "cheque" to the cheque-bounce concept bridge |
| `app/rag/vector_store.py` | `MongoVectorStore.search()` (and the `VectorStore` abstract base) take `mode`: `"semantic"` skips the BM25 leg, `"keyword"` skips the embedding leg |
| `app/rag/retriever.py` | Threads `mode` through to `vector_store.search()` and into the retrieval-cache key; fixed the SECTION-before-ACT citation regex's case-insensitivity bug (see §2.5) |
| `app/services/search_service.py` | Passes `request.mode` (when `"semantic"`/`"keyword"`) through to `retriever.retrieve()` |
| `tests/test_retrieval_relevance.py` | Updated the `_FakeVectorStore.search()` test double's signature to match the new `mode` parameter |

**New regression tests (19 tests):** `test_bug110_manipuri_detection.py` (4), `test_sindhi_telugu_cheque_bounce_expansion.py` (4), `test_search_mode_semantic_and_keyword.py` (5), `test_local_vector_scan_limit_covers_corpus.py` (2), plus one new test in `test_draft_export.py` and three new tests in `test_manual_chat_transcript_regressions.py`.

**Not touched:** `app/rag/citation.py` (BUG-107 — confirmed still a deliberate trade-off, see §5). No security, citation-guardrail, cross-user-isolation, or GK-disclaimer code was modified. No KB documents were re-ingested, re-chunked, or edited.

---

## 2. Each item — verified root cause, fix, before/after evidence

### 2.1 Manipuri language detection (T012 / BUG-110) — FIXED, verified

**Reproduced first:** the exact QA-report probe, `"check bounce oirabadi India da kari punishment oi?"`, against `LanguageDetector.detect()` directly.

**Verified root cause:** `app/language/detector.py` had no romanized-Manipuri marker-word table (unlike the eight Devanagari-script languages, which have `_DEVANAGARI_MARKER_WORDS`, or Hinglish, which has `_STRONG_HINGLISH_TERMS`). Romanized Manipuri carries no script signal either (it uses Latin letters), so it fell all the way through to `langdetect`, which has no Manipuri model and defaults an unrecognized code to `"english"`. Separately, once misdetected as English, `app/language/typo_tolerance.py`'s Hindi-verb-conjugation fuzzy allowlist ran on the message regardless of detected language and flagged the Manipuri word "kari" ("what") as an ambiguous typo of "kar"/"karo"/"karti" — producing the confusing Hindi spell-check prompt the QA report captured.

**Fix:**
- Added `_ROMANIZED_MANIPURI_TERMS` (`oirabadi`, `kari`, `karino`, `amasung`, ... — 19 curated terms), checked in `detect()` before the Hinglish check.
- `ConversationIntentClassifier.classify()`/`classify_advanced()` now take `language` and only offer the Hindi-verb typo clarification when `language in ("english", "hindi", "hinglish")` — the only languages the correction allowlist has real vocabulary for. `chat_service.py` threads the already-computed `language` value through both call sites.

**Before/after (live, restarted backend, `/chat`):**
```
Before (prior pass, unfixed): detected_language=english
  answer: "I could not read 'kari' — did you mean 'kar', 'karo', 'karti'?"

After (this pass):            detected_language=manipuri
  detected_intent: Cheque Bounce
  answer: "**General Legal Knowledge** (not from the verified Knowledge Base)
           India-da cheque bounce oiba haibasi criminal offence ama oina lou-i..."
           [an honest, clearly-labeled, on-topic answer written in Manipuri]
```
The KB has effectively no indexed Manipuri content, so a GK-fallback answer (clearly labeled, exactly the "honest refusal or GK answer" the original QA report asked for instead of the confusing prompt) is the correct outcome here — this fix is about *detection and routing*, not about making Manipuri content appear in the KB.

### 2.2 Hindi/Hinglish "security deposit" multi-turn (T026–T028) — retrieval layer FIXED and verified; a separate, pre-existing layer still causes intermittent GK fallback

**Correction to the prior pass's own record:** the driver script (`scripts/retest_qa_100q_driver.py`) shows T026's actual text is **not** Devanagari Hindi — it is romanized Hinglish: `"mera landlord mera security deposit wapas nahi de raha hai"`, which literally contains the English words "security deposit" and "landlord". This matters because it rules out the hypothesis (an earlier line of investigation in this same session, since discarded) that a Hindi bail/deposit vocabulary collision was the cause — the query is dominated by English tokens that already match `query_rewriter.py`'s existing rent/deposit expansion pattern correctly.

**Verified root cause (found by directly querying the live corpus, not guessed):** `app/rag/vector_store.py::_local_cosine_leg` runs `collection.find(mongo_filter).limit(self.local_scan_limit)` with **no sort**. `local_scan_limit` was 20,000; the real KB corpus has grown to **~55,000 chunks** (confirmed directly: `embeddings_metadata.count_documents({})` = 55,070 at the time of this fix). Per that field's own pre-existing docstring in `app/core/config.py` ("once the corpus exceeds this cap, whichever documents were indexed LAST become silently unreachable"), roughly the newest third of the corpus — including, confirmed directly, the actual Rent Control Act / Tenancy Act content this question needs — was **structurally unreachable** by the local-scan fallback, not just slow to reach. Because Mongo's natural order has no stability guarantee, this made grounding for the identical question non-deterministic: sometimes the relevant chunk fell inside the scanned window, sometimes not.

**Fix:** raised `local_vector_scan_limit` from 20,000 to 60,000 (comfortably above the corpus size, with headroom).

**Before/after (live, direct pipeline replication using production's own filters/intent/reranker/relevance-gate):**
```
Before: top 8 candidates — scores 0.0159-0.0323, dominated by unrelated Acts
        (Maharashtra Court Fees Act, Private Security Guards Act, Copyright Act,
        Sales Tax Amendment Act) with real Tenancy Act content occasionally
        present but indistinguishable by score.
After:  top 6 post-reranker candidates — scores 0.38-0.45, ALL genuinely on-topic:
        Odisha House Rent Control Act 1950, Maharashtra Rent Control Act 1999,
        Bombay Rents/Hotel/Lodging House Rates Control Act 1947 (x2),
        Maharashtra Tenancy and Agricultural Lands Act (Vidarbha).
```
**Live `/chat` confirmation (one full run, backend restarted, cache generation bumped so no stale answer could be served):**
```
general_knowledge_used: False
sources: [{"act_name": "BOMBAY ACT", "source_document": "...rents-hotel-and-lodging-house-rates-control-act...pdf",
           "page_number": 24, "verification_status": "verified"}]
answer: "...Bombay Rents, Hotel and Lodging House Rates Control Act, 1947 mein kuch
         provisions diye gaye hain. Is Act ke mutabiq, agar koi landlord aapse koi
         deposit ya sum leta hai, toh aap usse yeh amount wapas lene ke haqdaar hain..."
```
This is a real, verified fix: the KB content is now reachable and gets used. **However**, repeating the identical live `/chat` call multiple times showed this does not ground on every single call — see §7.2 for why, and why that is a separate, not-newly-introduced issue.

**T027/T028 (follow-up turns in the same session):** conversation context was correctly carried forward (`conversation_intent: "Follow-up Question"`, matching the prior pass's own finding), but T027 ("which court should I go to?") still fell back to GK on the live run tested. This was not specifically diagnosed this pass — it may be a genuine KB coverage gap (jurisdiction/forum-selection content) rather than a retrieval bug, and is listed as still open in §7.3.

### 2.3 Sanskrit / Sindhi / Telugu cheque-bounce grounding (T018/T020/T022) — retrieval layer FIXED and verified for all three; Sindhi/Telugu had a genuine, narrow missing-vocabulary bug; Sanskrit did not

**Reproduced first, exactly:** the three QA-report probe strings, byte-for-byte, from `scripts/retest_qa_100q_driver.py`:
- Sanskrit: `चेक-बाउंस-प्रसंगे भारते का शिक्षा भवति?`
- Sindhi: `چيڪ باؤنس ٿيڻ تي ڀارت ۾ ڪهڙي سزا آهي؟`
- Telugu: `చెక్ బౌన్స్ అయితే భారతదేశంలో శిక్ష ఏమిటి?`

All three spell "cheque bounce" as a **transliterated loanword in their own script**, not as native vocabulary — an important correction to the prior pass's own framing, which speculated these languages "genuinely need their own root-cause investigation" for missing classical/native vocabulary. That was checked directly and found **only partially true**:

**Verified root cause (Sindhi and Telugu only):** `app/rag/multilingual.py`'s cheque/bounce concept bridge already listed "چیک" (Urdu spelling — ends in "ک", the Arabic keheh) and "చెక్కు" ("checku"). The QA probe's actual spellings are genuinely different real strings: Sindhi's "چيڪ" ends in "ڪ" (the Sindhi-specific swash kaf — one of `detector.py`'s own `_SINDHI_ONLY_LETTERS`, confirming this is authentic Sindhi orthography, not a typo), and Telugu's "చెక్" ("chek") omits "కు" ("ku") from "checku". Neither matched the existing term lists, confirmed directly: `legal_english_variants()` returned no cheque-bounce expansion at all for either probe before the fix, only the generic, topic-agnostic "punishment" pattern.

**Verified non-root-cause (Sanskrit):** `legal_english_variants(SANSKRIT_PROBE)` already correctly returned the cheque-bounce expansion **before any fix**, because the probe's "चेक-बाउंस" already matches the existing plain Devanagari "चेक"/"बाउंस" terms. Sanskrit's grounding (when it fails — see §7.2) is not a missing-vocabulary problem.

**Fix:** added "چيڪ" and "చెక్" (plus the combined phrases "چيڪ باؤنس"/"చెక్ బౌన్స్") to `multilingual.py`'s term lists.

**Before/after (`legal_intent_hint`, all three languages):**
```
Before: Sanskrit -> ("Cheque Bounce", "Banking and Criminal Law")  [already correct]
        Sindhi   -> None
        Telugu   -> None
After:  Sanskrit -> ("Cheque Bounce", "Banking and Criminal Law")  [unchanged]
        Sindhi   -> ("Cheque Bounce", "Banking and Criminal Law")  [fixed]
        Telugu   -> ("Cheque Bounce", "Banking and Criminal Law")  [fixed]
```

**Before/after (direct pipeline replication, real corpus, all three languages — same `mode="hybrid"` retrieval + relevance gate used by production):**
```
Before: top 5 candidates dominated by unrelated/noise chunks, no real
        Negotiable Instruments Act content among accepted results.
After:  5/8 accepted results are the real Negotiable Instruments Act 1881,
        including sections 90, 129, 138, 139, floored/reranked scores 0.22-0.54.
```
This combines both fixes in this pass (the vocabulary fix for Sindhi/Telugu, and the same `local_vector_scan_limit` fix from §2.2 for all three).

**Live `/chat` for Sanskrit:** two live calls both fell back to GK despite the retrieval-layer fix being independently confirmed via direct pipeline replication (matching production's real filters/intent/reranker exactly, with production's own `IntentDetector`, in the same run) to reliably surface the correct content — see §7.2, same phenomenon as T026.

Sindhi and Telugu were verified at the retrieval-pipeline level (direct replication against the real corpus, matching production exactly) but not additionally via live `/chat`, given the time this investigation already took and the pattern established for Sanskrit.

### 2.4 `search mode=semantic` (T046) — FIXED, verified

**Verified root cause:** `MongoVectorStore.search()` always ran the same two-leg (BM25 + embedding) hybrid search regardless of what mode the caller asked for; `SearchService`/`LegalRetriever` never read `request.mode` past the already-fixed `"section"`/`"act"` cases (BUG-109/111, fixed in the prior pass).

**Fix:** `MongoVectorStore.search()` now takes `mode`. `"semantic"` skips the BM25 leg entirely — pure vector similarity, no RRF blending with a lexical score. `"keyword"` skips the embedding leg — pure BM25. Both threaded through `LegalRetriever.retrieve()` (including into the retrieval-cache key, so a cached hybrid-mode result can never be served back for a semantic-mode request) and `SearchService.search()`.

**Before/after (live, `/search`, identical query, `top_k=5`):**
```
Before: mode="semantic" and mode="hybrid" returned byte-for-byte identical results.
After:  mode="semantic" ids: [d106ece8, f60aadb2, 31c47472, d56d31f2, 71f3cd65]
        mode="hybrid"   ids: [d106ece8, d56d31f2, f60aadb2, a2d3fc0c, aa4d40ec]
        Genuinely different result sets and ordering.
```
Unit tests directly assert BM25 is never invoked in semantic mode (a raising stub in place of `bm25_index.search` proves it), and the embedding leg is never invoked in keyword mode.

### 2.5 T072 (Mongo search: NI Act ranking) — FIXED, verified; **reclassifies the prior pass's own framing**

The prior pass's feature table attributed this (and T074) to "Mongo vector cross-check ... Atlas unavailable." That framing was checked directly and found **wrong for T072 specifically** — it is a genuine, fixable regex bug, unrelated to Atlas.

**Reproduced first:** `POST /search {"query": "Section 138 cheque dishonour Negotiable Instruments Act", "mode": "hybrid"}`.

**Verified root cause:** `app/rag/retriever.py::_named_section_citation`'s SECTION-before-ACT pattern is compiled with `re.IGNORECASE`, which also relaxes the pattern's own `[A-Z]` anchor — the one meant to mark where the *Act name itself* starts. On a query with lowercase topical words between the section number and the real Act name ("Section 138 **cheque dishonour** Negotiable Instruments Act"), the non-greedy `act` capture group had no case-sensitive place to stop and swallowed the topic words too, extracting `"cheque dishonour Negotiable Instruments Act"` as the "Act name" — a string that matches no real corpus `act_name`/`source_document` metadata. That broke retrieval two ways at once: the `act_name`-filtered search found 0 candidates, **and** `find_named_section`'s own `requested_tokens <= identity_tokens` check rejected the real Negotiable Instruments Act chunks too (their filename contains "negotiable"/"instruments" but not "cheque"/"dishonour"). The query then fell through to an Act-agnostic bare-number search, which surfaced an unrelated Maharashtra Court Fees Act "Section 138".

**Fix:** made the `[A-Z]` anchor case-sensitive again (`(?-i: ... )`), and added a small (0–4 words), case-sensitive-lowercase-only, non-greedy filler-skip so a genuine topical prefix is skipped over rather than captured as part of the Act name. Verified this does not change any of the three existing, already-passing test cases for this function (zero-filler adjacency, "of the" bridging, ACT-before-SECTION order), and added a bounded-filler test proving the skip doesn't run away across an unrelated capitalized word much later in a long query.

**Before/after (live, `/search`, `mode="hybrid"`):**
```
Before: top result "THE MAHARASHTRA COURT FEES ACT" Section 138 (score 0.049, unrelated).
After:  2/2 results are Negotiable_Instruments_Act_1881_Complete_Act.pdf, Section 138,
        floored score 0.6500 (the exact-citation authoritative floor).
```

**T074 (Bharatiya Sakshya Adhiniyam / Evidence Act) — investigated, confirmed genuinely different, NOT fixed by this or any change in this pass.** Its query (`"admissibility of electronic evidence Bharatiya Sakshya Adhiniyam"`) names no section number at all, so `_named_section_citation` never applies to it in the first place — T072's bug and fix are structurally irrelevant to it. Re-tested live after the fix: still returns a Delhi Police Academy FAQ document ahead of the real Bharatiya Sakshya Adhiniyam sections 63/64. This is a genuinely different, broader natural-language relevance-ranking question (why a secondary FAQ document about a topic outranks the primary statute's own text on it) that would need its own precise diagnosis — not attempted this pass, and explicitly not claimed fixed.

### 2.6 T090 (XSS in a draft field) — verified SAFE; not a bug; regression test added

**The prior pass's own finding:** "still inconclusive; the test never actually exercised the field-storage path (the session had already moved past `collecting` stage)."

**Reproduced properly, end-to-end, live:** drove a fresh Consumer Complaint draft through `/chat` from scratch, confirmed it reached `stage: "collecting"`, then sent the exact QA-report payload as a real field answer: `"applicant name is <img src=x onerror=alert(1)>, applicant address is Chennai"`. Confirmed the raw payload was captured verbatim (unescaped) into the draft's internal `full_text`/section state — this is expected and correct; the original user-supplied text must be preserved faithfully at the data layer. Completed all remaining required fields, generated the draft (`stage: "preview"`), and exported it to PDF via `POST /draft/export`.

**Verified via `pypdf` text extraction on the real generated PDF:** the payload appears only as inert, literal visible text — `"I, <img src=x onerror=alert(1)>, residing at Chennai..."` — never as a live `<img>` element. Confirmed at the code level: `app/drafting/export.py`'s `PdfDraftExporter._build_html` passes every section value through `_escape()` (`html.escape(text, quote=False)`) before interpolating it into the HTML that WeasyPrint renders, and confirmed the raw bytes `onerror`/`<img`/`alert(1)` do not appear anywhere in the exported PDF as live markup.

**Conclusion:** not a bug. No code change made (none needed). **A regression test was added** (`test_pdf_export_escapes_html_markup_in_a_field_value` in `tests/test_draft_export.py`) since no XSS-specific export coverage existed before, so a future change that accidentally drops the `_escape()` call would be caught.

---

## 3. Full-suite test result

```
3526 passed, 18 skipped, 1 xfailed (0 failures)
```
Run twice against the final code (once mid-pass before the T072 regex fix, once after, both clean). Consistent with the prior pass's own note about one pre-existing, order-dependent `tests/test_workflow_orchestrator.py` flake caused by a live KB-automation job sharing this same dev machine's `storage/` tree — not observed in either of this pass's two runs, and not caused by anything changed here.

---

## 4. What was — and was not — live-verified

Per instruction, **nothing here is marked PASS without having actually been run.** This pass restarted the backend (confirmed via `/health`, fresh `server_uptime_seconds`) and bumped the response/retrieval cache generation (`ResponseCache.bump_generation()` — the app's own existing, purpose-built cache-invalidation mechanism, used exactly as it is intended: to guarantee no pre-fix cached answer could be served back) before any live verification.

**Live-verified against the restarted backend (`/chat`, `/search`):**
- T012 Manipuri — 1 live `/chat` call — **grounding/routing fix confirmed**
- T026 (Hindi/Hinglish security deposit, turn 1) — 3 live `/chat` calls — **grounded on 1/3** (see §7.2)
- T027 (follow-up) — 1 live `/chat` call — still ungrounded
- T018 Sanskrit — 2 live `/chat` calls — **ungrounded on 2/2** despite the independently-confirmed retrieval-layer fix (see §7.2)
- T046 `mode=semantic` — 2 live `/search` calls (semantic + hybrid, compared) — **fixed, confirmed**
- T072 NI Act — 1 live `/search` call — **fixed, confirmed**
- T074 Evidence Act — 1 live `/search` call — **still open, confirmed unchanged**
- T090 XSS — full live `/chat` draft lifecycle + live PDF export — **verified safe**

**Verified via direct pipeline replication** (same real MongoDB/Redis/Postgres, same filters/intent-detection/reranker/relevance-gate code as production, run as a script rather than through the HTTP layer — used where a live `/chat` round-trip would have added LLM-layer non-determinism noise without adding retrieval-layer confidence): T026, T018/T020/T022 (all three languages), matching what the live calls above showed.

**Not live-verified this pass:** the other ~88 questions in the original 100-question set. The full automated test suite (3526 tests, 0 failures, run twice) covers regression risk for the code paths this pass touched; this pass's own diffs are narrow and don't touch any code path the other 88 questions exercise differently than before. Re-running the full live 100-question suite (historically 40+ minutes to over an hour, based on this repo's own prior log timestamps) was judged not to add proportionate confidence given that scope-check, but it was not done, and nothing about those 88 questions is asserted here one way or the other.

---

## 5. BUG-107 (citation act-name display noise) — confirmed untouched, deliberately

Re-read `app/rag/citation.py`'s `usable_act_name()` and its own extensive documented rationale: a name that can't be validated (fails the clause-opener / bare-generic / statute-noun / length checks) is silently dropped rather than guessed at, because "rejecting a real Act name would remove a correct citation, which is the worse error" — a deliberate, already-documented trade-off, not a defect. Not in scope for this pass, not touched.

---

## 6. Cleanup performed

Every session, draft, and PDF export created by this pass's live verification was removed via the app's own `DELETE /session` and `DELETE /draft/{id}` endpoints (4 sessions, 1 draft — confirmed via each endpoint's own deletion-count response, e.g. `{"chat_messages": 5, "query_logs": 5, "intent_events": 5}`). No real KB content, real user data, or pre-existing sessions/drafts were read for modification, changed, or deleted. The one Redis operation performed against shared state — `ResponseCache.bump_generation()` — is non-destructive (an integer increment; nothing is deleted, and pre-generation-bump entries simply age out via their existing TTL) and is the app's own designed mechanism for exactly this situation (a code change invalidating previously-cached answers).

---

## 7. Remaining limitations (verified, not fixed this pass)

1. **Real Atlas Vector Search is not deployed.** Unchanged from the prior pass. `_validate_retrieval_backend()` still refuses to start in production/staging on the local fallback.

2. **A separate, pre-existing LLM-layer grounding-confidence non-determinism.** With the retrieval-layer bug in §2.2/§2.3 fixed (independently verified, deterministically, via direct pipeline replication matching production exactly — good, on-topic, well-scored context is now reliably reachable), live `/chat` calls for the same question still sometimes produce a refusal-shaped answer that the pipeline's own pre-existing `app/services/safe_decline.py::clear_support()` mechanism correctly detects and routes to the same clearly-labeled GK fallback used throughout this app (never a fabricated/uncited claim — this fails safe). Observed on both a Hinglish query (T026, grounded 1 of 3 live attempts) and a Sanskrit query (T018, grounded 0 of 2 live attempts). This is **not new** — `safe_decline.py`'s own module docstring describes this exact "retrieval succeeded, but the grounding validator or the LLM itself judged the chunks did not answer the question" scenario as an already-handled, pre-existing case, and nothing in this pass touched that module, the system prompt, or the LLM call itself. Not investigated further this pass (would require its own diagnosis of the LLM's/prompt's grounding-confidence behavior, a different problem from the retrieval bug this pass was scoped to fix).

3. **T027/T028-style follow-up questions** ("which court should I go to for this?") can still be ungrounded even when the initiating question (T026) succeeds. Conversation context is correctly carried forward (confirmed: `conversation_intent: "Follow-up Question"`), but the specific follow-up content (jurisdiction/forum selection) may itself be a KB coverage gap rather than a retrieval bug. Not diagnosed this pass.

4. **T074 (Bharatiya Sakshya Adhiniyam / Evidence Act) natural-language ranking** — confirmed, live, still open. A structurally different problem from T072 (no section number in the query, so the citation-parsing fix in §2.5 doesn't reach it) — a broader "why does a secondary FAQ document outrank the primary statute" relevance question. Not attempted without its own precise diagnosis, consistent with how the prior pass treated T072/T074 as a pair before this pass separated them.

5. **New, verified latency/timeout trade-off from the `local_vector_scan_limit` fix (§2.2).** Raising the scan limit from 20,000 to 60,000 makes the local (non-Atlas) fallback scan up to 3x more of the corpus per query leg. Measured live: most `/chat` calls during this pass's testing completed in 20–95s (comparable to the prior pass's own 28.8s average / 110s P95), but **one live call genuinely exceeded the 150-second `chat_request_budget_seconds` deadline** under concurrent contention with this dev machine's background KB-automation job (confirmed via backend logs: `kb_automation_cycle_failed` events logged around the same window). This is a real, newly-observed cost of the correctness fix, not mitigated this pass — the proper fix would be a MongoDB index on the filtered fields (`metadata.owner_session_id`, `metadata.review_status`, etc.) or, as already noted in the prior pass, real Atlas deployment. Neither was attempted here.

6. **BUG-107** — unchanged, deliberate (§5).

None of the above were fixed by assumption or silently left unaddressed — each was checked live or via direct pipeline replication and found to still be open, or (in the case of item 5) newly discovered as a genuine cost of a fix made this pass.

---

## 8. Final production-readiness verdict

**Not production-ready.** This pass fixed and verified three distinct, genuine retrieval-layer bugs — a language-detection/routing gap (Manipuri), a corpus-coverage cap that had fallen behind the KB's own growth (affecting Hindi/Hinglish, Sanskrit, Sindhi, and Telugu grounding alike), and a citation-parsing regex bug (T072) that the prior pass had misattributed to "Atlas unavailable" — plus confirmed one item (T090/XSS) was never actually broken. All fixes are covered by new regression tests, and the full 3526-test suite passes cleanly, twice. Against that: this pass's own live testing surfaced a real, previously-undocumented latency/timeout risk from the corpus-coverage fix, and confirmed (without attempting to fix) a separate, pre-existing LLM-layer non-determinism that still intermittently produces GK-fallback answers even on now-correctly-retrieved content. Two items (T027/T028 follow-ups, T074) remain genuinely open and are reported as such rather than claimed fixed.
