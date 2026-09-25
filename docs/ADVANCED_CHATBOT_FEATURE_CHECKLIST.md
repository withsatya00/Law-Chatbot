# Advanced Chatbot Feature Checklist

Status legend: `[x]` implemented and tested · `[~]` partially implemented ·
`[ ]` not implemented · `[!]` blocked by external credential/service.

Verified against actual code on 2026-08-22 (do not trust prior status
summaries without re-checking file:line references below). Run tests via
the project venv, not the system Python:
`".venv/Scripts/python.exe" -m pytest -q --basetemp=<writable-dir>` — a
stale, ACL-broken `%TEMP%\pytest-of-Satya` directory on this machine
causes spurious `PermissionError`s under the default basetemp; this is a
pre-existing OS issue, not a code defect (see "Known environment issues").

## Phase 1 — Core Chatbot Intelligence

- [x] **Correction-based re-routing** — `app/intent/classifier.py`
  (`_extract_correction`, `classify_advanced`) now detects "No, I meant X",
  Hinglish "X nahi, Y", and signal-only corrections ("you misunderstood"),
  and `app/services/chat_service.py` (`_dispatch_conversation_intent`)
  recurses into the real `answer()` handler with the corrected text —
  retrieval/entity extraction/answer generation now see only the corrected
  request, never the original wrong one. Original text is preserved once in
  an audit intent-event tagged `is_correction`; no duplicate writes.
  Also verified to never corrupt draft state (correction reroute recurses
  through `answer()` before any draft-mode branch runs) and covered over
  `/chat/stream` (Draft-Generation-classified corrections fall back to the
  full pipeline, where the actual reroute happens, same as `/chat`).
  Tests: `tests/test_advanced_intent_classification.py`
  (`test_correction_*`, `test_hinglish_correction_*`,
  `test_ordinary_message_is_not_flagged_*`),
  `tests/test_chat_service_routing.py`
  (`test_correction_reroutes_to_corrected_request_without_duplicate_intent_event`,
  `test_correction_without_extractable_text_is_not_rerouted`,
  `test_streaming_correction_reroutes_to_the_corrected_request`).
- [x] **Multi-intent workflow orchestration** — new
  `app/services/workflow_orchestrator.py`: a fixed `ALLOWED_CHAINS`
  allowlist (`MAX_CHAIN_LENGTH = 2`) covering all four required chains
  (Document Analysis → Draft Generation, Document Analysis → Legal
  Research, Legal Research → Draft Generation, Translation → Response
  Modification — "Legal Research" is the RAG-fallthrough intent family,
  since the classifier has no single intent by that name), plus a
  conservative, deterministic `map_facts_to_draft_fields` that copies only
  unambiguous single-candidate values from a document analysis into a
  draft template's fields (never a person name — no role classification
  exists yet, so applicant-vs-respondent can't be told apart; left for the
  user to answer directly rather than guessed). `detect_chain` is a pure
  function over `classify_advanced`'s own `detected_intents` — no extra LLM
  call, and it's the *only* source of truth for what a workflow is; nothing
  executes an arbitrary LLM-proposed sequence.

  Only "Document Analysis → Draft Generation" (the flagship "review this
  PDF, flag risky clauses, draft a notice based on it" case) has real
  cross-step automation, in `ChatService._run_document_analysis_then_draft`:
  resolves the document (same order as a standalone analysis turn),
  re-verifies ownership via the unmodified `DocumentService.analyze` (raises
  `ForbiddenError`/`BadRequestError` exactly as before — no workflow-specific
  bypass), analyzes once (no duplicate LLM call), maps facts, then seeds
  `DraftConversationEngine`'s existing collecting state so its own
  already-tested "ask only what's missing" / "auto-preview once nothing's
  missing" logic runs completely unchanged — reused, not reimplemented.
  Never reaches approved/locked/exported automatically:
  `_process_collecting_message` only ever advances to `preview_ready`; the
  user's own explicit next message is required for every step after that,
  completely untouched by this workflow. `ChatResponse` gained
  `workflow_chain`/`workflow_status` fields (empty/`None` on every ordinary
  turn) exposing chain-execution status in response metadata. Detecting the
  chain deterministically (not just via the LLM multi-intent path, which a
  confident "Draft Generation" match short-circuits past — see
  `protected_intents` in `classify_advanced`) required a small classifier
  change: `classify()`'s Draft Generation branch now also populates
  `detected_intents` when the message independently matches the existing
  `_DOCUMENT_ANALYSIS_VERB_PATTERN`/`_DOCUMENT_ANALYSIS_NOUN_PATTERN` regex
  pair, without changing `intent`/`confidence`/`ambiguous` — every existing
  draft-routing test is unaffected. Also added an explicit
  `classifier_source` field to `ConversationIntentMatch` (was previously
  inferred from `detected_intents` being non-empty, an assumption this
  change broke) and, while touching that code, fixed a real pre-existing
  duplicate intent-history write: `answer_stream`'s non-streaming-intent
  fallback path logged its own classification event against an unpersisted
  preview *and then* called `answer()`, which logs its own — now only the
  genuinely-streamed path logs directly, and the fallback relies on
  `answer()`'s own (accurate, real-memory) write.

  The other three allowlisted chains are recognized and logged (a
  `classifier_source: "workflow_chain"` intent-history event) for the
  "workflow-chain usage" analytics this feeds, but have no bespoke
  cross-step automation beyond that — deliberately: "Legal Research →
  Draft Generation" and "Document Analysis → Legal Research" already work
  turn-by-turn through existing, unmodified routing (a RAG answer this
  turn, an ordinary `DraftIntentDetector` trigger the next), and
  "Translation → Response Modification" already works via
  `_respond_with_response_modification` chaining onto the latest version.
  Manufacturing custom automation for those three would be speculative
  complexity with no missing capability behind it.

  Field-mapping coverage is intentionally thin beyond `facts`/narrative
  fields (seeded from the analysis's own executive summary) and
  single-candidate amount/case-number/date fields — person-name role
  classification (Phase 2 #6, still `[~]`) is the real blocker to mapping
  applicant/respondent names, and guessing would violate "never fabricate
  entities."
  Tests: `tests/test_workflow_orchestrator.py` (13 pure-function tests:
  chain allowlist matching/ordering/rejection, fact-mapping
  conservativeness), `tests/test_chat_service_routing.py`
  (`test_document_analysis_then_draft_generation_asks_only_for_missing_fields`,
  `test_document_analysis_then_draft_generation_reaches_preview_never_auto_approves`,
  `test_workflow_never_hijacks_an_already_active_draft_session`,
  `test_workflow_with_no_uploaded_document_asks_for_one_without_analyzing`,
  `test_workflow_denies_analysis_of_a_document_owned_by_someone_else`,
  `test_streaming_fallback_to_answer_does_not_double_log_the_intent_event`).
- [x] **General Clarification Mode** — extends the existing
  `pending_clarification` pattern (was only translation-target/
  response-modification-target) with a third case: `classify()`'s
  Draft Generation branch now distinguishes an unambiguous chain request
  ("review this PDF... draft a notice **based on it**") from a genuinely
  ambiguous one with no linking phrase ("review the PDF and draft a
  notice") via `_WORKFLOW_LINK_PATTERN`. The ambiguous case returns a new
  `"Workflow Clarification"` intent (protected from LLM override — 0.6
  confidence is below the 0.70 LLM threshold, so without protection a
  genuinely ambiguous case could get silently resolved to a guess) and
  `ChatService._respond_with_workflow_clarification` asks which one, storing
  `pending_clarification="workflow_intent"` plus the original question.
  Resolved next turn by `_resolve_workflow_clarification` via a fixed
  keyword set (`_resolve_workflow_clarification_choice` — "pdf/analysis/
  pehle/first" → run the chain, "direct/seedha/skip" → plain drafting) —
  deliberately does NOT recurse through re-classification (the original
  question still lacks the link phrase that made it ambiguous, so
  reclassifying would just ask again); directly invokes the already-built
  chain-execution/draft-turn methods with a synthetic, resolved match
  instead. An unrelated reply expires the pending state (one-turn-only
  TTL, same as the pre-existing translation-target case) and falls through
  to normal classification — never force-interpreted, never treated as a
  legal fact.
  Tests: `tests/test_advanced_intent_classification.py`
  (`test_ambiguous_document_and_draft_phrasing_asks_for_clarification`,
  `test_linked_document_and_draft_phrasing_does_not_ask_for_clarification`,
  `test_clarification_intent_is_protected_from_llm_override`),
  `tests/test_chat_service_routing.py`
  (`test_ambiguous_workflow_phrasing_asks_for_clarification_and_stores_pending_state`,
  `test_clarification_reply_choosing_analysis_runs_the_full_chain`,
  `test_clarification_reply_choosing_direct_starts_plain_drafting`,
  `test_unrelated_reply_to_clarification_expires_it_without_treating_reply_as_a_fact`).
- [x] **Intent feedback** ("Wrong intent" / "I wanted legal research" /
  "This is document analysis") — new `classify()` check
  (`_extract_intent_feedback`, `_INTENT_FEEDBACK_ALIASES`) returns a
  protected `"Intent Feedback"` intent, resolving a named category via a
  small alias map (capped at 5 words specifically so a real question
  phrased as "I wanted X" isn't swallowed as feedback about a prior turn —
  "I wanted legal research on cheque bounce cases in detail" stays a real
  question). `ChatService._respond_with_intent_feedback` stores
  `{session_id, owner_user_id, message_text, original_intent,
  corrected_intent}` in a new `intent_feedback` Mongo collection
  (`IntentFeedbackRepository`) — `original_intent` read from this
  session's own `intent_history` (the previous turn's classification,
  never guessed), `owner_user_id` from the JWT-verified identity
  (never a client-supplied one). Deliberately a separate collection from
  both `intent_events` (silent per-turn telemetry, below) and
  `app/memory/entity_memory.py`'s legal-facts store — this is metadata
  about routing, never treated as a fact about the user's situation, and
  the feedback message itself never reaches retrieval/the LLM. Bare "you
  misunderstood my request" (no category named) is intentionally left to
  the pre-existing `is_correction` signal-only path rather than
  duplicated here — same underlying signal, one handler.
  Tests: `tests/test_advanced_intent_classification.py` (7 tests: generic/
  named/unrecognized/long-phrase/LLM-protection cases),
  `tests/test_chat_service_routing.py` (6 tests: original+corrected intent
  stored, bare-feedback stores no correction, authenticated-vs-anonymous
  ownership, never reaches retrieval, write-failure never breaks the turn).
- [x] **Intent analytics** — new cross-session `intent_events` collection
  (`IntentEventRepository`), written via a single new call site
  (`ChatService._log_intent_event`, replacing 4 duplicated
  `append_intent_event`-only call sites) that dual-writes to both the
  pre-existing session-scoped `intent_history` (capped at 50, per-session
  state) and this new unbounded, cross-session collection — the capped
  per-session list alone couldn't answer "most common intent this month."
  Best-effort (try/except, warns and continues on failure, matching
  `ConversationMemoryStore._persist`'s own resilience pattern) so an
  analytics hiccup never breaks a real chat turn.
  `AnalyticsService.dashboard()` gained an `intent_analytics` section:
  most-common intents, low-confidence intents, correction rate, multi-intent
  frequency, workflow-chain usage, LLM-vs-deterministic classifier-source
  breakdown, and wrong-intent-feedback rate (cross-referencing the new
  `intent_feedback` collection above). Exposed via the existing
  `/admin/analytics/dashboard` endpoint (already admin-only,
  `require_admin`) — no new endpoint needed.
  Tests: `tests/test_analytics_recommendations.py`
  (`test_intent_analytics_aggregates_all_required_fields`).

## Phase 2 — Document Intelligence

- [~] **Document entity recognition** — `app/entity_extraction/extractor.py`
  is regex-only: `case_number`, `section_number`, `amount`, `date`,
  `vehicle_number`, `act_name` (fixed list), `state` (11 hardcoded states).
  No person/company names, roles (applicant/respondent/advocate), court
  names, addresses, obligations, or deadlines. No LLM-based extraction
  path with Pydantic validation + deterministic fallback exists yet.
- [~] **Timeline extraction** — `DocumentService._extract_timeline`
  (`app/services/document_service.py:217-238`) returns
  `{date, event_description, source_text}`
  (`app/schemas/document.py:37-40`). Supports numeric dates, English-month
  dates, and 11 Indian-language months via
  `app/drafting/localized_dates.py`. Missing: normalized ISO date field,
  event category, responsible party, page number, deadline/expiry flag,
  chronological sorting, and relative-duration phrases ("within 30 days",
  "before expiry", "on or before").
- [~] **Document-type-specific missing-clause detection** — clause/risk
  analysis exists (LLM-backed with a deterministic keyword-scan fallback,
  `document_service.py:36-54,198-215`) and `missing_information` is
  returned, but it's one generic free-text list from the LLM prompt, not
  per-document-type checklists (rental/employment/NDA/sale/service/
  partnership/notice/affidavit/complaint), and doesn't separate "missing
  clause" vs "missing fact" vs "not applicable" vs "unable to determine".
- [ ] **Page-level evidence** — no page-number tracking anywhere.
  `SourceCitation` (`app/schemas/common.py:7-14`) has no page field; no
  `page_number` in `app/rag/chunker.py` / `metadata.py` / `loader.py`.
- [ ] **Contract comparison** across two documents — does not exist.

## Phase 3 — Legal Research and Answer Quality

- [x] **Grounding validation activated** — `LegalCitationEngine.validate_grounding()`
  (`app/rag/citation.py`) existed but was completely dead code (not even
  `citations_from_chunks` was used anywhere; `chat_service.py` had its own
  parallel `_citation_from_chunk`). Now invoked in the real RAG answer path
  (`app/services/chat_service.py`, in the main `answer()` LLM branch) right
  after the LLM response is generated: if the answer has zero real source
  citations behind it, it's replaced with the existing safe
  `NO_VERIFIED_CONTEXT_MESSAGE` and confidence forced to 0 (never cached,
  `is_cacheable` already requires `confidence >= threshold`), instead of
  presenting unsupported LLM prose as verified law. Deliberately scoped to
  only the genuine-LLM-generation branch, not the deterministic
  `_fallback_answer` path (used when the LLM call itself errors) — that
  text is quoted directly out of the retrieved chunk, inherently grounded
  regardless of whether a formal `source_document` tag is present; treating
  it as ungrounded broke a real existing regression test
  (`test_rag_llm_error_fallback_prefers_relevant_chunk_over_top_ranked`)
  during this work and was reverted to the narrower scope.
  Tests: `tests/test_chat_service_routing.py`
  (`test_ungrounded_answer_is_downgraded_to_insufficient_context`,
  `test_grounded_answer_with_real_citations_is_not_downgraded`).
- [~] **Structured chatbot responses** — `ChatResponse`
  (`app/schemas/chat.py`) already exposes answer/sources/confidence/
  confidence_reason/disclaimer; explanation/applicable-law/risks/next-steps
  are folded into `answer` text via prompt structure rather than separate
  fields. Not changed this session — pre-existing, functional, not
  reworked given budget.
- [~] **Confidence handling** — single blended score
  (`_confidence()` in `chat_service.py`, retrieval+intent+entity weighted),
  not split into model-confidence vs source-grounded-confidence. Grounding
  validation (this session) now feeds into it for downgrade-on-unsupported
  behavior, but the two confidence axes aren't separately surfaced yet.
- [!] **Case-law / amendment feeds** — no live case-law/amendment API
  integration exists. Requires an external legal database subscription
  (e.g. Manupatra/SCC/Indian Kanoon API access) and credentials; blocked
  until such a service is provisioned. Local KB (BM25 + vector over
  ingested Acts) is real and already used correctly.

## Phase 4 — Security and Privacy

- [x] **Ownership checks on drafting endpoints** — `app/api/drafting.py`
  had **zero** ownership/auth checks on any route; any authenticated or
  anonymous caller could act on any draft by id. Fixed: added
  `_ensure_draft_access()` (mirrors `DocumentService._ensure_document_access`'s
  three-way rule: no owner -> open; `user_id`-owned -> that exact
  authenticated user only; `session_id`-owned -> that exact session only)
  and wired `get_current_user_id` + the fetched draft record into
  edit/export/approve/lock/unlock/rollback/versions/translate. Added
  `session_id` to the request schemas that lacked it
  (`DraftEditRequest`/`DraftExportRequest`/`DraftLifecycleRequest`/
  `DraftRollbackRequest`/`DraftTranslateRequest`) so the check has something
  to verify against. `draft-history` no longer trusts a client-supplied
  `user_id` — an authenticated caller's JWT identity always wins, an
  anonymous caller's claimed `user_id` is dropped.
  Tests: `tests/test_draft_ownership.py` (13 tests: pure-function ownership
  matrix + per-route cross-user/cross-session denial + history spoofing).
- [x] **Feedback endpoint ownership** — `app/api/feedback.py` had no auth
  dependency, no message-ownership check (any caller could rate/comment on
  any message by guessing `message_id`), and `rating` had no range
  validation. Fixed: `FeedbackRequest.rating` now `Field(ge=1, le=5)`
  (matches the existing 1-5 scale `NEGATIVE_RATING_THRESHOLD`/
  `rating_summary` assume elsewhere); `/feedback` now looks up the target
  query-log entry and verifies its `session_id` matches the request before
  calling `attach_feedback`.
  Tests: `tests/test_feedback_ownership.py` (5 tests: rating bounds,
  same-session success, cross-session denial, missing-message 404).
- [ ] **Privacy controls** — no PII masking, no log redaction, no secure
  document deletion, no user-data-deletion flow, no document
  retention/expiry configuration. Not attempted this session (large,
  cross-cutting; needs its own pass).
- [!] **Refresh-token revocation** — JWTs are stateless with no blacklist;
  `/logout` (`app/api/auth.py:19-21`) is a no-op. A real revocation list
  needs shared storage (Redis/Mongo) keyed by token id/jti plus an
  is-revoked check in the auth dependency — architecturally straightforward
  but not implemented; flagged rather than attempted given scope.
- [~] **Voice endpoint security** — `app/api/voice_router.py` had no auth
  dependency, no audio size limit (only checked non-empty), and no
  MIME-type validation (client-claimed `content_type` forwarded straight to
  Gemini). Fixed: `/voice/chat` now takes `get_current_user_id` and forwards
  it as `authenticated_user_id` to `ChatService.answer` (same as `/chat`);
  audio is read in bounded 1 MB chunks capped at
  `settings.voice_max_audio_mb` (default 10 MB, new config); content-type is
  checked against an explicit allowlist of formats a browser/mobile recorder
  actually produces, rejecting missing/unrecognized types (415) instead of
  guessing `audio/wav`. `/voice/speak`'s `SpeakRequest.text` now has a
  20,000-char cap. Marked partial, not `[x]`, because true audio *duration*
  validation (as opposed to size, which is only a proxy) would need an audio
  probing library (ffmpeg/mutagen) not currently a dependency — not added
  per "do not install unnecessary dependencies"; the size cap is the
  practical bound in place instead. Browser microphone permission policy is
  a frontend (Streamlit) concern, out of scope for this backend.
  Tests: `tests/test_voice_security.py` (8 tests: MIME allowlist, missing
  content-type, size limit, empty audio, auth propagation, text-length cap).

## Phase 5 — Human Review and Chatbot UX

- [ ] **Advocate review workflow** — does not exist (the only
  `review_status` in the codebase is an unrelated admin
  "unanswered-question queue" feature in
  `app/repositories/analytics.py:147-184`).
- [ ] **Workflow progress events** — no orchestrator exists to emit them
  (depends on Phase 1's multi-intent orchestration being built first).
- [~] **Lawyer recommendation** — real code path
  (`app/recommendation/engine.py`), but a hardcoded ~11-category mapping
  to generic lawyer *types*, not a real directory (no names, no contact
  info, no location/availability/verification data) — correctly does not
  invent real lawyers. A clean provider interface for a real directory
  would need actual data; not attempted this session.
- [~] **Multilingual behavior** — solid existing coverage: English/Hindi/
  Hinglish detection (`app/language/`), 11+ Indian languages for dates and
  draft export (`test_multilingual_draft_audit.py`), Act/Section names
  preserved through translation per existing drafting glossary/translation
  memory work. Not re-verified exhaustively this session beyond what the
  existing 719-test suite already covers.

## Session summary (2026-08-22)

**Turn 1** — fixed and tested: correction-based re-routing (Phase 1 #1),
drafting-endpoint ownership (Phase 4 #14), feedback-endpoint ownership +
rating validation (Phase 4 #15), grounding validation activation
(Phase 3 #10), voice-endpoint size/MIME/auth hardening (Phase 4 #17,
partial). 28 new tests; suite went from a 719-test clean baseline to 747.

**Turn 2** ("phase 1") — completed the rest of Phase 1: multi-intent
workflow orchestration (#2, the flagship "review PDF → flag risks → draft
notice" chain, previously the single biggest gap), General Clarification
Mode (#3), Intent Feedback (#4), Intent Analytics (#5) — all `[x]` now, see
their entries above for what each actually covers and where it's
intentionally scoped down (workflow chains: 1 of 4 chains has real
cross-step automation, the other 3 are recognized/logged but rely on
already-working turn-by-turn routing; fact-mapping: conservative, no
person-name guessing until Phase 2 entity extraction gets role
classification). Along the way: fixed a real pre-existing duplicate
intent-history write in the streaming fallback path, added an explicit
`classifier_source` field (was inferred incorrectly), added a
`_WORKFLOW_LINK_PATTERN` distinction between an unambiguous chain request
and a genuinely ambiguous one. 42 more tests; suite now at 789, all
passing.

Files changed (turn 2, additive to turn 1's list below):
- `app/intent/classifier.py` — `classify()`'s Draft Generation branch gained
  workflow-chain detection (`_DOCUMENT_ANALYSIS_VERB/NOUN_PATTERN` +
  `_WORKFLOW_LINK_PATTERN`) and a new `"Workflow Clarification"` intent for
  the ambiguous case; new `_extract_intent_feedback`/
  `_INTENT_FEEDBACK_ALIASES` for `"Intent Feedback"`; `ConversationIntentMatch`
  gained `classifier_source`, `is_intent_feedback`, `feedback_corrected_intent`;
  both new intents added to `protected_intents`.
- `app/services/workflow_orchestrator.py` (new) — `ALLOWED_CHAINS` allowlist,
  `detect_chain`, `map_facts_to_draft_fields`.
- `app/services/chat_service.py` — `_dispatch_workflow_chain`/
  `_run_document_analysis_then_draft` (the flagship chain execution),
  `_respond_with_workflow_clarification`/`_resolve_workflow_clarification`,
  `_respond_with_intent_feedback`, `_log_intent_event` (replaces 4 duplicated
  `append_intent_event` call sites, dual-writes to `intent_events`); fixed
  the `answer_stream` duplicate-intent-event-write bug in the same pass.
- `app/repositories/analytics.py` — new `IntentEventRepository`,
  `IntentFeedbackRepository`.
- `app/services/analytics_service.py` — `dashboard()` gained `intent_analytics`.
- `app/schemas/chat.py` — `ChatResponse` gained `workflow_chain`/`workflow_status`.
- `app/models/collections.py` — `INTENT_EVENTS`, `INTENT_FEEDBACK`.
- New tests: `tests/test_workflow_orchestrator.py`; extended
  `tests/test_advanced_intent_classification.py`,
  `tests/test_chat_service_routing.py`, `tests/test_analytics_recommendations.py`.

Files changed (turn 1):
- `app/intent/classifier.py` — `ConversationIntentMatch` gained
  `is_correction`/`corrected_text`; `_extract_correction` now returns both
  and recognizes the Hinglish "X nahi, Y" pattern and signal-only
  corrections; both `classify_advanced` return paths propagate the fields.
- `app/services/chat_service.py` — correction reroute branch in
  `_dispatch_conversation_intent`; `LegalCitationEngine` wired in as
  `self.citation_engine` and invoked in the RAG answer path for grounding
  validation.
- `app/api/drafting.py` — `_ensure_draft_access` helper; ownership checks on
  edit/export/approve/lock/unlock/rollback/versions/translate; `draft-history`
  no longer trusts client-supplied `user_id`.
- `app/api/feedback.py` — message-ownership check before `attach_feedback`.
- `app/api/voice_router.py` — MIME allowlist, bounded chunked read + size
  cap, `get_current_user_id` propagation, `SpeakRequest` length cap.
- `app/schemas/drafting.py` — added `session_id` to
  `DraftEditRequest`/`DraftExportRequest`/`DraftLifecycleRequest`/
  `DraftRollbackRequest`/`DraftTranslateRequest`.
- `app/schemas/history.py` — `FeedbackRequest.rating` now bounded 1-5.
- `app/core/config.py` — `voice_max_audio_mb` setting + `voice_max_audio_bytes`.
- New tests: `tests/test_draft_ownership.py`, `tests/test_feedback_ownership.py`,
  `tests/test_voice_security.py`.

Not attempted (still `[ ]`/`[~]` above), roughly in order of value:
LLM-based entity extraction beyond regex + role classification
(Phase 2 #6 — also the real blocker to richer workflow-chain fact-mapping),
page-level citation evidence (Phase 2 #8), contract comparison (Phase 2 #9),
privacy controls / PII masking / data-deletion (Phase 4 #16), advocate
review workflow (Phase 5 #18), workflow progress events (Phase 5 #19).

Remaining risks worth flagging explicitly:
- `app/api/drafting.py`'s `/draft` (conversational `DraftMessageRequest`)
  endpoint still has no ownership check tying `session_id` to the
  authenticated caller — lower risk than the fixed routes since it only
  operates through session memory state, not an arbitrary `draft_id`, but
  not verified this session.
- Refresh tokens are still unrevocable (stateless JWT, no blacklist) — flagged
  `[!]`, needs shared storage (Redis/Mongo) keyed by token id.
- No privacy/PII controls at all yet — full legal documents and user text
  currently flow into logs/DB without redaction.

## Known environment issues (not code defects)

- `%TEMP%\pytest-of-Satya` has a broken ACL (`Get-Acl` itself raises
  `UnauthorizedAccessException`) dated 2026-07-28, unrelated to any change
  in this session. Work around with `--basetemp=<writable dir>` rather than
  attempting to repair OS-level ACLs.
- The system `python` on `PATH` (3.11, `AppData\Local\Programs\Python\Python311`)
  does not have this project's dependencies (`weasyprint`, etc.) installed.
  Always use the project's `.venv\Scripts\python.exe` (3.12.7) to run tests.
- **Resolved (Phase 1).** `app/api/chat_router.py`,
  `app/api/draft_chat_router.py` and `app/rag/hybrid_legal_retriever.py`
  were dead/unwired code — never registered in `app/main.py`, and each
  containing a literal `raise NotImplementedError(...)` setup stub. All
  three have been removed. The real chat path is
  `app/api/chat.py` -> `app/services/chat_service.py`, and the real hybrid
  retrieval path is `app/rag/retriever.py` (`LegalRetriever`) over
  `app/rag/vector_store.py`'s dense + `app/rag/bm25_index.py` sparse legs,
  fused by `app/rag/fusion.py` and reranked by `app/rag/reranker.py`.
  `tests/test_no_setup_stubs.py` now fails the build if a setup
  `NotImplementedError` is reintroduced.
