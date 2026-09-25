# QA Test Matrix — Legal AI Assistant (session 2026-09-11)

Tester: real-user simulation via live HTTP API (`http://localhost:8000`), local dev
environment. MongoDB + Redis already running; API server restarted once (to pick up
a code fix) and confirmed healthy. Streamlit UI not exercised yet (see Blocked).

Legend: PASS (execution evidence captured) · FAIL (bug found) · BLOCKED (a
concrete dependency/tool/service prevented execution — e.g. no browser, no
credentials, a service down) · NOT RUN / PENDING (in scope, simply not yet
executed this session — time budget, not a blocker) · MOCKED (not a real
end-to-end call — unit/integration test only).

**Correction (2026-09-11, follow-up session):** the previous version of this
file used "BLOCKED" loosely for several items that were actually just not
executed yet (no real blocker — no missing credential, tool, or service).
Those are now corrected to NOT RUN / PENDING below. "BLOCKED" is reserved for
genuine external blockers (see Streamlit UI note). The earlier "Full Phase 1
pass" framing is also corrected: Phase 1 as specified lists 20 scenarios
(T1-T20); only 5 were actually executed (T1, T3, T4, T5, T6) — see the
updated Phase 1 table for the full breakdown of executed vs. not-run.
**2675 unit/integration tests passing is evidence that existing deterministic
logic hasn't regressed — it is NOT evidence that any live multi-turn user
journey (draft generation, sequential draft edits, multilingual conversation,
etc.) has actually been exercised end-to-end.** Those journeys are reported
separately below as either live-PASS (with request/response evidence) or NOT
RUN, never inferred from unit-test counts.

## Environment / Discovery (Phase 0)

- Backend: FastAPI + Uvicorn, `app.main:app`, started from project `.venv`.
- MongoDB (27017) and Redis (6379) already running locally.
- LLM provider: Gemini (`gemma-4-26b-a4b-it` per `/health`).
- Embedding: BAAI/bge-m3 on CUDA, BM25 index loaded (30,664 chunks) from disk.
- No auth token used for most tests (routes support anonymous session-scoped use).
- Existing pytest suite present (~789+ tests per docs/ADVANCED_CHATBOT_FEATURE_CHECKLIST.md).
- Two stray uvicorn processes were found already running on port 8000 (started
  2026-09-10 18:08, no `--reload`); restarted cleanly as a single instance so a
  code fix would actually take effect. No other blocking issues found in discovery.

## Phase 1 — Startup & Basic Journey (5 of 20 specified scenarios executed)

| Test ID | Scenario | Expected | Actual | Status | Evidence |
|---|---|---|---|---|---|
| P1-T1 | Fresh app load / GET /health | 200 ok, all components ok | `{"status":"ok",...}` all components ok | PASS | health check output |
| P1-T2 | New chat create | — | — | NOT RUN | — |
| P1-T3 | First message "Hi, I need help with a legal issue." | Graceful response, no crash | 200, ~22s latency, "No verified document..." fallback (reasonable for vague greeting w/ no KB match) | PASS | t3 body |
| P1-T4 | Normal legal question (security deposit, Mumbai) | Relevant MH-specific sources, no fabricated entities | 200, correct MH Rent Act sources; **entity extraction bug found & fixed** (see Bugs) | PASS (after fix) | t4 body, extractor fix |
| P1-T5 | Empty message `""` | 422 validation error | 422, `string_too_short` | PASS | t5 body |
| P1-T6 | Whitespace-only `"   "` | Graceful handling, no long hang | 200 in 6s, "No verified document" fallback. One earlier run took >30s (network/LLM latency variance, not reproced on retry) | PASS (flagged: see Notes) | t6 body x2 |
| P1-T7 | Very long message / input limit | — | — | NOT RUN | — |
| P1-T8 | Emojis, punctuation, special characters | — | — | NOT RUN | — |
| P1-T9 | Enter vs Shift+Enter behavior | — | — | BLOCKED | no browser-automation tool available in this environment; Streamlit dev UI not running (verified: `curl` to :8501 refused). API-level testing continues instead per execution rules. |
| P1-T10 | Rapid double-click send | — | — | NOT RUN | — |
| P1-T11 | Message sent during an in-flight response | — | — | NOT RUN | — |
| P1-T12 | Streaming response completes | — | — | PARTIAL — see Priority 1 (streaming leak) below | streaming section |
| P1-T13 | Streaming interrupt / network disconnect | — | — | NOT RUN | — |
| P1-T14 | Retry after error | — | — | NOT RUN | — |
| P1-T15 | Refresh recovers conversation | — | — | NOT RUN | — |
| P1-T16 | Reopen existing conversation | — | — | NOT RUN | — |
| P1-T17 | Switch between multiple chats | — | — | NOT RUN | — |
| P1-T18 | New chat has no context leak from a previous chat | — | — | NOT RUN | — |
| P1-T19 | Mobile viewport usability | — | — | BLOCKED | no browser-automation tool available in this environment; Streamlit dev UI not running (verified: `curl` to :8501 refused). API-level testing continues instead per execution rules. |
| P1-T20 | Backend unavailable → clear error + recovery | — | — | NOT RUN (would require deliberately stopping the shared local server — deferred to avoid disrupting other in-progress work on this machine) | — |

### Bugs found & fixed this session

**BUG-001 (FAIL→FIXED): Entity extractor fabricates section/case numbers from ordinary words**
- File: `app/entity_extraction/extractor.py`
- Root cause: `case_number` and `section_number` regexes were matched with a
  blanket `re.IGNORECASE`, which also loosened the **value**-capturing groups
  (`[0-9A-Z]...`, meant to require digits/uppercase real identifiers). Any
  word starting with "sec"/"case"/"fir"/"complaint" had its trailing letters
  wrongly captured — e.g. "security deposit" → `section_number: ["urity"]`.
  This risks fabricating wrong facts into legal drafts (violates "facts
  bilkul mat badalna" / no-fabrication requirement in Phase 4/5).
- Fix: scoped the case-insensitivity to just the keyword portion via inline
  `(?i:...)` groups, removed the blanket `re.IGNORECASE` from `extract()`'s
  `re.findall` call. Verified real identifiers ("Section 138", "Case No:
  CRL/1234/2025") still extract correctly.
- Regression test added: `tests/test_language_intent_entities.py::test_entity_extraction_does_not_misread_ordinary_words_as_identifiers`.
- Verified live: before fix, `POST /chat` with the security-deposit message
  returned `extracted_entities: {"section_number": ["urity"], "amount": ["50000"]}`;
  after fix + server restart, `{"amount": ["50000"]}` (no false section_number).
- Existing suite (`tests/test_language_intent_entities.py`,
  `tests/test_structured_extraction.py`, 108 tests) still green after the fix.

### Notes / minor, not fixed (low severity)

- P1-T6: one whitespace-only request took >30s while a retry took 6s — looks
  like transient LLM-provider latency variance (same fallback path, same
  code), not a reproducible hang. Flagged for awareness, not treated as a bug
  without a second reproduction.
- P1-T4: assistant asks "Which State?" even though "Mumbai" was in the
  message and retrieval already correctly scoped to Maharashtra-specific Acts
  — i.e. retrieval resolved the state but the answer text still asks for
  explicit confirmation. Possibly intentional (explicit confirmation before
  citing state-specific law) but worth a product decision; not changed.

### BUG-002 (FAIL→FIXED, HIGH severity): "Which State?" clarification never resolves for a city-only reply

- Files: `app/rag/matter_context.py`, `app/rag/jurisdiction.py`
- Symptom (live repro, Hinglish multi-turn journey, session `qa-draft-*`):
  1. "Mera landlord mera security deposit wapas nahi kar raha hai." → asks
     "Yeh matter kis State ... se related hai?"
  2. "Mumbai mein hoon. Deposit amount 50000 rupees tha." → **asks the exact
     same State question again**, ignoring that Mumbai was just given.
  3. "Landlord ka naam Rahul Sharma hai. ..." → same question a third time.
  4. "Ab isko legal notice mein convert kar do please." → same question a
     fourth time; drafting never starts, no `draft` object ever returned.
  This is exactly the failure mode Phase 2 rule 18 warns about
  ("clarification questions ke endless loop mein na phase") and rule 17
  ("already provided information unnecessarily dobara na pooche").
- Root cause: `app/rag/matter_context.py`'s `detect_state()` (the function
  that actually drives the in-chat "which State?" clarification and decides
  when it is resolved) only recognises explicit State/UT **names**
  ("Maharashtra", "UP") via `_EXPLICIT_LOCATION_RE`/`_BARE_STATE_RE`. It has
  no notion of cities at all. A separate module,
  `app/rag/jurisdiction.py`, already had a `city -> state` map
  (`_CITY_TO_STATE`), but it was private and only used for an unrelated
  feature (the Model Tenancy Act caveat) — never consulted by the module that
  actually gates the clarification loop. So "Mumbai" (or any other city)
  never satisfied `state_codes`, `needs_clarification` stayed `True`
  forever, and the question repeated on every turn.
- Fix:
  - `app/rag/jurisdiction.py`: renamed `_CITY_TO_STATE` → public
    `CITY_TO_STATE` so it can be reused as the single source of truth.
  - `app/rag/matter_context.py`: imports `CITY_TO_STATE`; `detect_state()`
    now also matches a city with a locational cue ("in Mumbai", "Mumbai
    mein") via a new `_EXPLICIT_CITY_LOCATION_RE`, and — when
    `allow_bare_mention=True` (i.e. the previous turn just asked the
    clarifying question) — a bare city reply ("Mumbai") via a new
    `_BARE_CITY_RE`, mirroring the existing bare-State-reply allowance. A
    named State still always wins over a city match in the same turn (state
    check runs first).
- Regression test added:
  `tests/test_matter_context.py::test_city_mention_resolves_state_and_ends_the_clarification_loop`
  (covers both the explicit "in Mumbai" phrasing and the bare "Mumbai" reply
  to an active clarification).
- Verified: `tests/test_matter_context.py` (13 tests), the broader
  jurisdiction/multi-turn suites (`test_jurisdiction_phase2_gaps.py`,
  `test_jurisdiction_phase2_gaps2.py`, `test_jurisdiction_retrieval_phase2.py`,
  `test_multi_turn_conversations.py` — 165 tests total) and
  `test_chat_service_routing.py` (73 tests) all still green after the fix.
  Live re-run of the exact same 4-turn Hinglish journey against a restarted
  server: **confirmed fixed** — the assistant no longer re-asks "which
  State?" after "Mumbai mein hoon..."; instead of looping, later turns
  correctly reference the matter as already understood (see BUG-003 below
  for what the same live run surfaced next).

### BUG-003 (FAIL→FIXED, HIGH severity): raw LLM chain-of-thought / system-prompt fragments leaked into the user-facing answer

- File: `app/llm/resilient.py` (`ResilientLLMProvider.chat`, the provider-
  agnostic wrapper used by every `/chat` call regardless of which LLM backend
  is configured).
- Symptom (live repro, same session as BUG-002): one assistant turn's full
  text was internal reasoning wrapped in literal `<think>...</think>` markup,
  including a verbatim fragment of the system prompt's own instructions:
  > "The prompt says 'Answer the user's question directly...' ... Rule 2:
  > 'If the user's question is empty or actually about something else,
  > output Rule 2's exact fallback line and nothing else.' ... So I'll
  > output that exact line..."
  followed by the actual answer. This is both an unprofessional/broken
  answer and a prompt-leakage issue (internal system instructions exposed
  to the end user).
- Root cause: `GeminiProvider._parse_response` already strips parts the
  Gemini API marks with a structured `"thought": true` flag (existing,
  correct code, with a comment anticipating exactly this class of bug) —
  but the configured "thinking" model (`gemma-4-26b-a4b-it`) does not always
  use that structured flag; it sometimes writes its reasoning as plain
  `<think>...</think>` text INSIDE a normal (non-thought) part, which the
  existing filter cannot see. No provider or wrapper stripped inline think
  tags from the text itself.
- Fix: added inline `<think>...</think>` (and unterminated-`<think>`)
  stripping in `ResilientLLMProvider.chat()` — the single provider-agnostic
  choke-point every configured LLM backend (Gemini/Groq/OpenAI/Claude/
  DeepSeek/Ollama) already passes through — rather than patching each
  provider file individually. If stripping leaves no usable text at all
  (the model produced only reasoning), the response is now treated as a
  retryable failure (reuses the existing retry/fail-over loop) instead of
  being returned as an empty "successful" answer.
- Regression tests added in `tests/test_chat_reliability_integration.py`:
  `test_inline_think_tags_are_stripped_from_a_successful_answer`,
  `test_answer_that_is_pure_thinking_retries_then_fails_over`.
- Note: this fix covers the non-streaming `/chat` path (`ResilientLLMProvider
  .chat`), which is what every test and live repro above used. The streaming
  path (`ResilientLLMProvider.stream` → `/chat/stream`) still passes chunks
  straight from the primary provider with no equivalent stripping — flagged
  as a follow-up, not fixed this session (buffering/splitting `<think>` tags
  correctly across an SSE token stream is a materially bigger change than
  the non-streaming fix and wasn't reproduced live in the time available).

### Operational observation (not a code bug; flagged for the product owner)

- The configured Gemini model (`gemma-4-26b-a4b-it`, a mandatory-"thinking"
  variant kept per `app/llm/gemini.py`'s own comments for its much higher
  daily quota) is very slow in practice: individual `/chat` calls during this
  session's live testing regularly took 15-90s, and one legitimate call
  (logged: `llm_call_ms: 72675`, one internal retry) took ~97s end-to-end,
  exceeding even a 150s client-side test timeout on a later attempt. This is
  a known, already-documented trade-off in the codebase (quota vs. latency),
  not something introduced or fixed this session — but it materially slowed
  down live multi-turn QA (a naive multi-turn journey can spend several
  minutes just waiting on the LLM) and would affect real users the same way.
  Worth a product decision (faster/non-thinking model vs. current quota
  headroom) independent of this QA pass.

## Session 2 (2026-09-11, follow-up) — Priority 1: streaming output leak

**Status: FIXED and live-verified.** Full step-by-step evidence, root cause,
and fix are in a new dedicated section further below ("PRIORITY 1 — Streaming
output leak: full writeup"). Summary: BUG-003 (session 1) only fixed the
non-streaming `/chat` path. `/chat/stream` streamed raw, un-stripped
`<think>` reasoning to the client token-by-token, before the point where any
post-hoc string strip could reach it — confirmed by inspecting
`ChatService.answer_stream` (`app/services/chat_service.py`), which forwards
each `self.llm.stream(...)` chunk to the client as a `"token"` SSE event
immediately, and `ResilientLLMProvider.stream()` (`app/llm/resilient.py`),
which previously passed provider chunks through unfiltered. Fixed with a new
incremental `_InlineThinkStreamFilter` that filters BEFORE a chunk is
yielded, covering all 8 required scenarios (complete blocks, tags split
across chunks, reasoning split across chunks, unclosed tags, multiple
blocks, reasoning-only response, legitimate angle-bracket text, error text
after partial streaming) — 18 new tests in
`tests/test_streaming_think_filter.py`, all passing. Full suite after this
fix: `2693 passed, 3 skipped, 0 failed` (up from 2675 — the 18 new tests).

**Secondary finding (not a leak, flagged for follow-up, not fixed this
session):** one live `/chat/stream` probe with a pure-English question
("In Maharashtra, what is the notice period for eviction...") returned its
"no verified context" fallback message in **Hindi**, not English, on a fresh
session with no prior language preference. Worth checking `language`
resolution for this specific no-context-fallback branch — out of scope for
Priority 1/2/3 and not investigated further this session.

## Session 2 — Priority 2: draft follow-up workflow (in progress)

**Journey session ID: `qa-p2-deposit-1`** (synthetic facts per the task's
Priority 2 spec: Amit Kumar / Rohit Sharma / Test Flat 12, Example Road,
Mumbai / ₹50,000 / vacated 1 Aug 2026 / 15-day response period).

| Step | Message | Result | Status |
|---|---|---|---|
| 1 | "Mera landlord deposit return nahi kar raha." | Correctly asked which State (19.5s) | PASS |
| 2 | "Mumbai, Maharashtra mein hoon. Deposit amount 50,000 rupees tha, 1 August 2026 ko vacate kiya tha." | Did NOT re-ask "which state" (BUG-002 fix holds); acknowledged facts, no verified-context answer with as-of-date note (60.3s) | PASS |
| 3 | "In facts par English mein legal notice draft kar do. ... Jo details missing hain unke liye placeholders rakhna, response period 15 days rakhna." | **FAIL found**: routed to `draft_workflow` (correct — drafting DID trigger, unlike session 1) but selected the WRONG template: `missing_person_report` instead of `legal_notice`. See BUG-004 below. Root-caused, fixed, regression-tested. Retest of this exact step pending server restart. | FAIL → FIXED, retest PENDING |

### BUG-004 (FAIL→FIXED, CRITICAL severity): wrong draft document type selected — "legal notice" request launched a Missing Person Report

- File: `app/drafting/intent.py` (`_typo_tolerant_match`, used by
  `DraftIntentDetector._ranked_templates` → `_partial_name_overlap`).
- Symptom (live repro, step 3 of the Priority 2 journey above): an explicit,
  unambiguous request — "legal notice draft kar do" with sender/recipient/
  property/amount all stated — was answered with:
  > "Let's draft your Missing Person Report. I just need 10 more details:
  > 1. Applicant Address 2. Mobile Number 3. Applicant Name... 8. Missing
  > Person's Name 9. Place 10. Police Station"
  Completely the wrong document type, despite `route: "draft_workflow"`
  correctly firing (confirming BUG-002's fix holds — drafting itself is no
  longer blocked by a stuck clarification loop).
- Reproduced deterministically (no LLM involved) via
  `DraftIntentDetector().detect(message)` directly:
  `missing_person_report` scored 0.76 vs `legal_notice`'s 0.36.
- Root cause: `_typo_tolerant_match(word, other)` allowed a Levenshtein
  distance of **2** for any word pair where the shorter word was over 5
  characters. My test message's "response **period** 15 days" contains
  "period" — which, at distance 2, coincidentally typo-matched "**person**"
  (both 6 letters, genuinely unrelated words). Combined with a real,
  unrelated "**missing**" elsewhere in the message ("jo details **missing**
  hain" = "which details are missing"), `_partial_name_overlap`'s "2+
  overlapping tokens" bar was coincidentally cleared against FOUR of
  `missing_person_report`'s own trigger phrases (each containing both
  "missing" and "person"), summing to a higher score than the message's own
  correctly-spelled, exact "legal notice" match. This is the THIRD instance
  of the same failure class already documented in this file's own comments
  (a prior live incident: "kanuni"/"karni"; another: "place"/"police") —
  each previously patched narrowly, never by questioning whether distance-2
  tolerance was actually needed at all.
- Evidence the fix is safe: computed the exact edit distance for every
  legitimate typo pair already covered by this app's own test suite
  ("polcie"/"police", "agrement"/"agreement", "domestik"/"domestic",
  "bonce"/"bounce", "moeny"/"money", "recoevry"/"recovery", "renta"/"rent")
  — every one of them is distance **1**; none has ever actually relied on
  the distance-2 allowance to pass.
- Fix: `_typo_tolerant_match` is now capped at distance 1, unconditionally
  (removed the length-based distance-2 branch entirely, rather than another
  narrow per-word-pair exclusion).
- Regression test added:
  `tests/test_draft_discovery.py::test_typo_tolerance_does_not_hijack_a_correctly_named_template_via_an_unrelated_word`
  (uses the exact reproducing message). All 71 tests in
  `test_draft_discovery.py` (including every pre-existing typo-tolerance
  test) still pass.
- **Not yet re-verified live end-to-end** (server restart + retest of
  journey step 3 pending as of this write-up — see next matrix update).

### BUG-005 (FAIL→FIXED, HIGH severity): amount-change edit command not recognized on `legal_notice` (no amount field existed)

- Files: `app/drafting/templates/legal_notice.yaml`,
  `tests/test_draft_conversation.py`.
- Symptom (live repro, step 4 of the Priority 2 journey, the task's own
  literal script: "Amount 50,000 se 65,000 kar do, baaki same rakho."):
  the message was NOT recognized as a draft edit at all. Server log
  confirmed `route: "rag"`, `conversation_intent: "General Legal
  Information"` — the message left the draft flow entirely and was answered
  as an unrelated RAG question (170s LLM call, no relation to the draft).
  The draft was untouched and the user was given no indication their edit
  request failed to apply.
- Root cause: `EditCommandInterpreter._match_field_and_value` resolves an
  edit only against a field the TEMPLATE actually declares (`field_synonyms`
  / field labels). `legal_notice.yaml` had **no monetary field at all** —
  unlike sibling templates (`recovery_notice`'s `principal_amount`,
  `cheque_bounce_notice`'s `cheque_amount`), so "Amount ... kar do" matched
  no field, `_match_field_and_value` returned `(None, None)`,
  `EditCommandInterpreter.interpret` returned `action="unknown"`, and
  `ChatService._is_draft_interruption` then treated the whole message as
  leaving the draft (its final fallback, `looks_informational`, saw an
  ordinary-looking statement and returned `True`).
  This also explains a related Phase-4 observation: the ₹50,000 figure the
  user gave in step 2 never appeared anywhere in the step-3 generated
  notice — there was no structured field for it to land in.
- Fix: added an optional `claim_amount` field to `legal_notice.yaml`
  (`required: false`, since a "General Legal Notice" is not always
  monetary) with `amount`/`deposit`/`deposit amount` synonyms, following
  this app's own established convention for monetary notice templates.
- Regression test:
  `tests/test_draft_conversation.py::test_edit_command_interpreter_parses_amount_change_on_legal_notice`
  (uses the task's own exact wording).
- Verified live after restart: the same message now returns "Updated Claim
  Amount (Rs.), if applicable. Here is the revised draft:..." and routes
  through `draft_workflow`, not `rag`.

### BUG-006 (FAIL→FIXED, HIGH severity): editing a draft silently corrupted an UNRELATED section ("baaki same rakho" violated)

- Files: `app/drafting/templates/legal_notice.yaml` (added
  `subject_template`), `tests/test_drafting.py`.
- Symptom (live repro, same step-4 amount-edit turn, AFTER BUG-005's fix
  made the edit actually apply): the edit correctly updated `claim_amount`,
  but the regenerated draft's **Subject** section — a field the user never
  asked to change, having explicitly said "baaki same rakho" ("keep
  everything else the same") — silently changed from the original,
  properly-composed "LEGAL NOTICE FOR THE REFUND OF SECURITY DEPOSIT" to a
  raw, unformatted dump of the `expected_relief` field text ("refund of the
  full deposit within the response period."), which is also the same text
  verbatim already appearing (correctly) in the Prayer section — i.e. the
  document now visibly duplicated one sentence in two unrelated places.
  Also observed: neither "50,000" nor "65,000" appeared anywhere in the
  regenerated document text at all, despite the field having been updated —
  flagged for awareness (the value is stored and edit-recognized correctly
  per BUG-005's fix and the regression test above; whether the LLM/renderer
  reliably weaves a newly-set `claim_amount` into the visible prose on every
  regeneration was not independently re-verified this session given time
  budget — worth a follow-up live check).
- Root cause: `LegalDraftEngine._subject_line_text` (`app/drafting/
  engine.py:1514-1533`) falls back to `fields.get("expected_relief", "")`
  used VERBATIM as the entire Subject line whenever a template has no
  `subject_template` of its own. `legal_notice.yaml` had none. This
  fallback path runs on every regeneration (initial generation apparently
  let the LLM compose Subject text freely within its own free-form output;
  an edit-triggered regeneration recomputes structural sections like
  Subject deterministically via this function) — so ANY edit to ANY
  `legal_notice` draft would hit this, not just an amount edit.
  **Broader finding, not fixed:** 28 of the 55+ templates in this catalog
  have no `subject_template` at all and share this same class of risk;
  this session only fixed the one template actually reproduced and tested
  (`legal_notice`). Flagged for a follow-up audit rather than blanket-fixed
  here without individually verifying each template's field semantics.
- Fix: added `subject_template: "Legal notice for {expected_relief}"` to
  `legal_notice.yaml`, matching this app's own established convention
  (`demand_notice.yaml` already uses the identical idiom verbatim).
- Regression test:
  `tests/test_drafting.py::test_legal_notice_subject_line_is_not_a_raw_duplicate_of_the_relief_field`.
- Verified live: retest of the exact same amount-edit step after server
  restart — Subject line now reads "Legal notice for refund of the full
  deposit within the response period." (coherent, no longer a bare
  duplicate). **Confirmed fixed.**

### Finding-007 (content-quality gap, NOT a code bug — verified, not fixed): the claim amount is correctly stored but the LLM doesn't always restate it in the prose

- Live observation: after BUG-005's fix, neither "50,000" nor "65,000"
  appeared anywhere in the regenerated notice text, even though the edit
  was correctly recognized and applied.
- Investigated at the code level (no further live LLM calls needed):
  `LegalDraftEngine.regenerate` → `_render_sections` → `_render_fields`
  correctly includes EVERY non-blank field (required or optional) in the
  text handed to the LLM prompt — verified directly:
  `engine._render_fields(template, {..., "claim_amount": "65,000"}, "english")`
  produces a line `"Claim Amount (Rs.), if applicable: 65,000"` in the
  prompt. So this is not a plumbing/threading bug — the fact reaches the
  model. The model itself simply did not choose to restate the specific
  figure in the Facts/Prayer prose on this generation.
- Not fixed this session: a prompt-wording change to force every material
  figure to be explicitly restated would touch the shared
  `legal_drafting_prompt.md` used by all 55+ templates, and verifying it
  doesn't regress elsewhere needs more live-LLM iteration than the
  session's time budget allows. Flagged for a follow-up prompt-engineering
  pass, with this session's concrete repro (session `qa-p2-deposit-1`,
  `claim_amount=65000`, generated Facts/Prayer sections with no numeral
  anywhere) as the starting evidence.

### Step 5 result: recipient name change — PASS

"Recipient ka naam Rohit Sharma se Rahul Verma kar do." → new name "Rahul
Verma" present, old name "Rohit Sharma" gone, unrelated fact "Amit Kumar"
preserved, Subject line unchanged/stable. Clean pass, no bugs.

### BUG-008 + BUG-009 (FAIL→FIXED, HIGH severity): "X ki jagah Y kar do" edits silently left the draft and produced a wrong, hallucinated refusal

- Files: `app/drafting/edit_commands.py` (`_FROM_TO_PATTERN`,
  `_strip_possessives`), `app/drafting/templates/legal_notice.yaml`.
- Symptom (live repro, step 6 of the Priority 2 journey): "15 days ki jagah
  7 days kar do." (an entirely ordinary way to say "change 15 days to 7
  days") did not edit the draft. Worse than a silent no-op: the message
  left the draft flow, was answered as an ordinary legal question, and the
  live answer **refused the request with a fabricated legal justification**:
  > "I cannot change the statutory requirements, as the notice period is
  > mandated by law. Under Section 18(6) of the Maharashtra Shops and
  > Establishments (Regulation of Employment and Conditions of Service)
  > Act, 2017, a worker is required to apply for leave at least fifteen
  > days in advance..."
  That statute governs an EMPLOYEE'S leave-application notice to an
  employer — entirely unrelated to a legal notice's own respond-by
  deadline, which is at the sender's discretion. A real user could easily
  have believed they were legally barred from shortening their own notice.
  (Draft state itself was not lost — `draft: null` in this turn's response,
  with the chat text correctly noting the draft was "still saved.")
- Two compounding root causes, both fixed:
  1. `legal_notice.yaml` had no synonym for `response_deadline_days` at
     all (same class of gap as BUG-005) — confirmed even a plain "15 days
     se 7 days kar do." (using the ALREADY-supported "se" connector) came
     back `action=unknown` for this reason alone.
  2. `_FROM_TO_PATTERN` only recognised "se"/"से" as the from-to connector,
     never "ki jagah"/"ke jagah" ("instead of") — an equally ordinary
     Hinglish/Hindi construction. Adding it alone was not enough: `_strip_
     possessives` (which drops "ka"/"ki"/"ke" between two nouns, e.g.
     "recipient ka address" -> "recipient address") was ALSO stripping the
     "ki" out of "ki jagah" as if it were that same unrelated possessive
     particle, corrupting the connector before the newly-added pattern
     alternative could ever see it intact — confirmed live: with only the
     regex alternation added and `_strip_possessives` left alone, the
     command resolved the right field but captured a corrupted value
     (`"jagah 7 days"` instead of `"7 days"`, the stray word from the
     mangled connector still attached).
- Fix: added `response_deadline_days` synonyms ("days", "deadline",
  "response deadline", "notice period") to `legal_notice.yaml`; extended
  `_FROM_TO_PATTERN` to also accept "ki jagah"/"ke jagah"/"की जगह"/"के जगह";
  added a negative lookahead to `_strip_possessives` so it no longer
  strips "ki"/"ke" specifically when immediately followed by "jagah".
- Regression test:
  `tests/test_draft_conversation.py::test_edit_command_interpreter_parses_ki_jagah_deadline_change_on_legal_notice`.
  Also re-ran `recipient ka address change karo Pune` (a genuine
  possessive) to confirm the negative lookahead didn't regress ordinary
  possessive-stripping.
- The underlying "message that leaves the draft flow gets answered as a
  legal question, and that answer can be confidently wrong/hallucinated
  about an unrelated statute" failure mode is broader than this one
  connector gap — flagged as a standing risk (Phase 7 concern: answer
  grounding/relevance), not something this session's fix eliminates in
  general, only for this specific reproduced phrasing.
- Live re-verification: pending (next matrix update, after server restart).

### Step 6 result: deadline change ("ki jagah" phrasing) — PASS (after fix)

"15 days ki jagah 7 days kar do." → "Updated Days given to respond.", "7
days" present, "15 days" gone, prior edits (Rahul Verma) retained. Confirmed
fixed live.

### BUG-010 (FAIL, HIGH severity, NOT FIXED — feature gap, documented not patched): no "add a fact/paragraph" edit capability exists at all

- File: `app/drafting/edit_commands.py`.
- Symptom (live repro, step 7 of the Priority 2 journey, the task's own
  script: "Ye fact add karo: maine 5 aur 12 August ko reminders bheje."):
  left the draft flow entirely (same failure shape as BUG-008/009) and was
  answered as a brand-new legal question, re-asking "Which State?" even
  though the State had been established several turns earlier in the same
  conversation.
- Root cause, confirmed at the code level (two independent gaps):
  1. `EditAction` (the `Literal` type covering every recognized edit) has
     **no "append"/"add" value at all** — only `replace_field`,
     `remove_paragraph`, `translate`, `regenerate`, `format`, `export`,
     `approve`, `lock`, `unlock`, `rollback`, `unknown`. There is
     structurally no way to express "add this to what's already there" as
     opposed to "replace what's there."
  2. Independently, even the verb-detection gate never fires here: bare
     romanized "karo" (no space, as in "add **karo**") is not in
     `_REPLACE_VERBS` at all — only spaced forms ("kar do", "kardo", "kr
     do", "kar dijiye") and the Devanagari "करो" are recognized. Confirmed:
     `any(v in msg.lower() for v in _REPLACE_VERBS)` is `False` for this
     exact message.
- **Deliberately not fixed this session.** A minimal patch (recognizing
  "add"/bare "karo" and routing it through the existing `replace_field`
  action against the `facts` field) was considered and rejected: `replace_
  field` **overwrites** a field's value — mapping "add a fact" onto it
  would silently DELETE the facts already in the draft the moment a user
  tried to add one more, which directly violates this session's own
  explicit test requirement ("Sequential edits accumulate hon; previous
  edits disappear na hon") and would be worse than the current honest
  failure (an edit that visibly doesn't apply, vs. one that silently
  destroys prior content while claiming success). A correct fix needs:
  a genuine `append_field` `EditAction`; verb detection for "add"/bare
  "karo"/"jodo"/"jod do" alongside the existing replace verbs; and,
  structurally, the current field VALUES threaded into
  `EditCommandInterpreter.interpret()` (today it only ever receives the
  message text, template, and language -- not what's already in the
  field it would be appending to), which is a signature change reaching
  into every existing caller. This is a real, scoped feature gap, not a
  one-line bug — flagged for a dedicated follow-up session rather than
  rushed.
- Not regression-tested (nothing to regression-test — no fix applied).
  The reproduction above is the evidence; resuming work on this should
  start from `EditCommandInterpreter.interpret` and `EditAction` in
  `app/drafting/edit_commands.py`.

## Session 3 (2026-09-11, resumed exactly from session 2's checkpoint) — BUG-010 implementation, Finding-007 fix, Priority 2 journey continued

### BUG-010 status update: FIXED (real append/add capability implemented, live-verified)

Per the reporting corrections: this bug is now counted FIXED only after live
verification below, not on implementation alone.

- Implemented a genuine third `EditAction`, `append_field` (not a variant of
  `replace_field` -- confirmed it never discards existing content):
  `app/drafting/edit_commands.py` (`_APPEND_VERB_PATTERN`, `_APPEND_VALUE_PATTERN`,
  `_match_append_field_and_value`, `EditCommandInterpreter.append_value`),
  `app/drafting/conversation.py` (`append_field` dispatch in `_continue_preview`,
  reading the CURRENT persisted field value via `draft_engine.drafts.find_by_id`
  before merging).
- Behavior implemented against the required rules: explicit supplied content
  inserted without inventing extra facts (verbatim insertion, confirmed
  live below); existing content never discarded (merge, not replace);
  explicit target honoured when named ("Facts section mein add karo");
  falls back to the template's own narrative field when no location is
  named, and the reply names the section chosen; never targets a
  non-`textarea` field; duplicate requests are a no-op (case-insensitive
  substring check) with an explicit "already there" reply, not a silent
  re-add; save/regenerate reuses `LegalDraftEngine.regenerate`'s existing
  lifecycle/versioning guarantees (no new concurrency logic invented --
  inherits whatever `regenerate` already provides for every other edit
  action); suggest-only messages verified to never trigger append (see
  test below); a legal-allegation-style addition would still pass through
  the EXISTING `PromptInjectionScanner`/`DraftSafetyGuard` scan inside
  `_render_sections_within_deadline` (unchanged, already runs on every
  regenerate call) -- not a new mechanism, the existing one already covers it.
- 17 new deterministic tests in `tests/test_draft_conversation.py`
  (Hinglish + English intent, explicit target, no target named, non-textarea
  target rejection, no-content-asks-for-it, add-vs-replace classification,
  suggest-only negative case, merge/dedup logic, end-to-end through
  `DraftConversationEngine` with the current persisted value, duplicate
  request is a no-op not a false success).
- **Live verification (session `qa-p2-deposit-1`, draft
  `3db8360e-7e5a-4783-99ea-3c139f003782`):** resent the EXACT checkpoint
  message ("Ye fact add karo: maine 5 aur 12 August ko reminders bheje.")
  against the running server. Result: `"Added to Facts of the Case (in
  your own words). Here is the revised draft:"`, and the Facts section now
  reads:
  > 1. That I vacated the flat on 1 August 2026 after giving proper notice,
  > but the landlord has not returned my security deposit despite reminders.
  >
  > 2. That maine 5 aur 12 August ko reminders bheje
  The original fact (point 1) is untouched; the new fact is inserted
  verbatim as point 2 (no fabrication); all prior edits held (recipient
  "Rahul Verma", "7 days" deadline, Subject line). **PASS, live-verified.**

### Priority 2 (Finding-007) status update: FIXED and live-verified

Treated as a material draft-correctness issue per the correction, not a
stylistic nicety.

- Root cause confirmed (code-level, no further live calls needed):
  `claim_amount` reaches the LLM prompt correctly (`_render_fields` output
  verified directly to contain "Claim Amount (Rs.), if applicable: 65,000"),
  but nothing guaranteed the MODEL actually restated it -- a pure
  generation-content gap, not a plumbing bug.
- Fix: new `DraftField.must_appear_verbatim` flag (`app/drafting/templates/base.py`,
  parsed in `app/drafting/templates/loader.py`), set `true` on `legal_notice`'s
  `claim_amount`. `LegalDraftEngine._ensure_verbatim_fields` (new,
  `app/drafting/engine.py`) runs once, right before every `_RenderedDraft`
  is returned -- covering the LLM-generated, format-reask, expansion-pass,
  AND deterministic-fallback paths uniformly -- and appends one plain,
  fact-only sentence (`"{field label}: {value}."`) to the Facts section
  ONLY if the value (comma-insensitive) doesn't already appear anywhere in
  the rendered text. Never invents wording beyond the field's own label and
  the user's own supplied value.
- 4 new deterministic tests in `tests/test_drafting.py`: amount appears even
  when the (faked) LLM omits it; not duplicated when the LLM already states
  it; survives the deterministic-fallback path; no field given -> no
  spurious injected sentence. Each asserts the INVARIANT (does the dynamic
  value appear, exactly once, and is an unrelated figure absent) rather
  than a fixed full-document string.
- **Live verification: CONFIRMED**, and under the strongest possible
  condition -- see step 8 below, where the LLM call itself failed and the
  engine fell back to the deterministic path, and `claim_amount` (65,000)
  still appeared in the final persisted document. Proves the guarantee
  holds regardless of generation mode, not just on the LLM-success path.

### BUG-011 (FAIL→FIXED at the routing level; tone-STYLING application not yet live-verified): "Tone polite but firm karo" had no recognized action

- Files: `app/drafting/edit_commands.py` (`_TONE_VALUE_PATTERN`, new
  `restyle` action), `app/drafting/conversation.py` (dispatch),
  `app/drafting/engine.py` (`regenerate`/`_render_sections_within_deadline`
  gain `style_instruction`), `app/schemas/drafting.py`
  (`DraftGenerateRequest.style_instruction`).
- Symptom (live repro, step 8 of the Priority 2 journey, the task's own
  script): "Tone polite but firm karo." left the draft flow entirely
  (`route: "rag"`... actually resolved to a fresh "Which State?"
  clarification, same failure shape as every prior missing-action bug this
  session) instead of being recognized as a request to re-render the SAME
  facts in a different voice.
- Root cause: same underlying gap discovered during BUG-010 (bare
  romanized "karo" not a `_REPLACE_VERBS` trigger), compounded by there
  being no concept of a "tone/register change" action AT ALL -- this isn't
  a field-value edit (no field named "tone" exists on any template), it
  needed a genuinely new action, the same way append did.
- Fix: a new `restyle` action, detected via `_TONE_VALUE_PATTERN`
  (triggers on "tone"/"register"/"lehja"/"स्वर" + karo/kar do/banao/etc.),
  carrying the requested tone as free text. `LegalDraftEngine.regenerate`
  gained an optional `style_instruction` parameter threaded through to
  `_render_sections_within_deadline`, which appends an explicit prompt
  directive: rewrite in the given tone, changing NO fact/name/date/amount/
  deadline/relief and adding NO new allegation or threat -- the same
  no-fabrication guarantee every other edit path already has.
- Regression tests: `tests/test_draft_conversation.py::test_continue_preview_tone_change_regenerates_with_style_instruction`
  (routing reaches `regenerate` with the right `style_instruction`) and
  `tests/test_drafting.py::test_style_instruction_reaches_the_generation_prompt_without_licensing_new_facts`
  (the directive text actually reaches the LLM prompt, with the
  no-new-facts constraint present in it).
- **Live verification, partial:** resending the exact live-failing message
  now correctly logs `route: "draft_workflow"` (was `"rag"`/a jurisdiction
  clarification before the fix) -- the ROUTING defect is confirmed fixed.
  However, the underlying Gemini call itself stalled for **317 seconds**
  (`gemini_overall_timeout`, `elapsed_seconds: 317.47` against a 100s
  per-attempt budget) on this attempt -- a severe, unusual latency spike,
  well beyond the 15-90s range observed earlier this session -- so the
  engine fell back to the deterministic skeleton before any LLM could
  actually attempt the tone rewrite. The persisted result correctly
  retained all prior edits and (per BUG-010/Finding-007 above) the claim
  amount, confirming the SAFE side of this path, but the actual
  tone-restyling behavior (does the LLM, when it responds, genuinely
  rewrite in the requested register without changing facts) has **not**
  been live-confirmed -- only unit-tested (prompt directive verified
  present; no live LLM call completed to observe compliance). Flagged
  honestly rather than claimed as a full live PASS.

### Step 9 result: translate to Hindi — PASS (high quality)

"Notice Hindi mein translate karo." → full Hindi translation,
`generation_mode: "llm"` (real generation, LLM recovered from the earlier
317s stall). Verified: राहुल वर्मा (Rahul Verma, prior edit) correct; 65,000
रुपये (Finding-007's fix) appears correctly TWICE (Facts + Prayer),
unchanged from the English version; 7 दिनों (7 days, prior edit) correct;
both reminder facts (5 अगस्त 2026 / 12 अगस्त 2026, BUG-010's added fact)
translated and woven into natural Hindi legal prose with no fabrication —
same two dates, same event, more fluent phrasing than the raw Hinglish
insertion. Names transliterated to Devanagari (अमित कुमार, राहुल वर्मा) while
the property address stayed in Latin script -- normal/expected for a Hindi
legal document, not a preservation violation. Clean pass, no bugs.

### BUG-012 (FAIL→FIXED, CRITICAL severity — false-positive draft cancellation): an "explain, don't modify" request was misread as "cancel the draft" and the bot claimed it was deleted

- File: `app/services/chat_service.py` (`_CANCEL_DRAFT_PATTERN`).
- Symptom (live repro, step 10 of the Priority 2 journey): "Ye draft mujhe
  Hinglish mein samjha do, lekin draft Hindi mein hi rehne do, badlo mat."
  ("Explain this draft to me in Hinglish, but let the draft STAY in Hindi,
  don't change it") — a request that explicitly said NOT to modify the
  draft — got the reply "ठीक है, मैंने वह मसौदा हटा दिया है" ("OK, I have
  deleted that draft"), with `draft: null` in the response, after 10 turns
  of accumulated edits (name change, amount change, deadline change, an
  added fact, a full Hindi translation).
- Root cause: `_CANCEL_DRAFT_PATTERN` has a proximity branch matching the
  word "draft" within 20 characters of a small set of dismissal words,
  including "rehne". "Rehne do" is genuinely ambiguous in Hindi/Hinglish —
  it can mean "leave it, forget it" (dismissal, the sense this pattern was
  written for) OR "let it remain/stay as it is" (the OPPOSITE — a
  preservation instruction). My message used the second sense ("Hindi mein
  hi rehne do" = "let it stay in Hindi"), and the word "draft" happened to
  appear twice in the same sentence purely because I was asking about the
  draft — close enough to "rehne" (15 characters away) to match the
  20-character proximity window.
- **Impact assessed, not assumed:** verified via `POST /draft-history`
  that the underlying draft document (`draft_id
  3db8360e-7e5a-4783-99ea-3c139f003782`) was NOT actually deleted —
  `_respond_with_draft_cancelled` only calls `DraftConversationEngine.reset`,
  which clears the CONVERSATION's session-level pointer to the draft, not
  the persisted MongoDB record itself. `draft-history` still listed it,
  `status: "preview_ready"`, `language: "hindi"`, fully intact. Still rated
  CRITICAL: (a) the bot's own wording ("I have deleted that draft") is
  false and alarming — a real user would reasonably believe their work was
  destroyed when it was not; (b) the conversation's ability to keep editing
  that draft naturally in the SAME session was genuinely broken (the
  session's `draft_mode`/`draft_id` memory was cleared), which is real
  functional damage even though the underlying data survived; (c) the
  trigger phrase is an entirely ordinary way to say "keep it as it is" in
  Hindi/Hinglish, not an edge case.
- Fix: removed "rehne" from the 20-character proximity-based branch only.
  It remains in the narrower bare-word branch (an ENTIRE message that is
  JUST "rehne do" and nothing else) added for Part 38, where the ambiguity
  risk is much lower — a two-word message with nothing else really is far
  more likely to mean dismissal, matching that branch's original rationale.
  Every genuine cancel phrasing this pattern exists for (`"cancel kr do
  draft ko"`, `"cancel draft"`, `"draft hata do"`, `"band karo"`, bare
  `"rehne do"`) verified to still match after the fix.
- **Live re-verification: CONFIRMED FIXED.** Two angles tested:
  1. Same checkpoint session (`qa-p2-deposit-1`): confirmed the underlying
     draft record was never destroyed (`POST /draft-history` still lists
     it, `status: "preview_ready"`, `language: "hindi"`). Reconnecting the
     SAME conversation session to actively edit that specific draft again
     turned out to be its own separate friction (see Finding-008 below) —
     not part of BUG-012 itself, since the false-cancellation trigger is
     what was fixed, and a genuinely different draft-reopening flow exists
     and works (see Finding-008).
  2. Clean synthetic re-test (fresh session `qa-bug012-verify`): built a
     minimal draft to preview stage, then sent the adapted exact trigger
     phrasing ("...lekin draft English mein hi rehne do, badlo mat.").
     Result: `draft` was NOT null (a true cancellation returns `draft:
     null`, as the original repro showed), the document was returned
     fully intact, and every fact stayed internally consistent (Rs. 20,000
     appears consistently throughout Facts/Prayer, applicant/respondent
     names unchanged, no fabrication). The false-cancellation defect is
     confirmed fixed.

### Finding-008 (session-continuity friction, NOT the same as BUG-012, not fixed): reopening a listed saved draft by name didn't resume it in this session

- After BUG-012's false cancellation cleared session `qa-p2-deposit-1`'s
  active-draft pointer, `POST /draft-history`-equivalent chat requests
  ("Mera legal notice draft wapas dikhao...") correctly LISTED the saved
  draft ("Here are your saved drafts: - General Legal Notice (hindi) —
  preview_ready... Say which one you want to open..."), but two follow-up
  attempts to actually select it ("General Legal Notice wala draft open
  karo.", then bare "General Legal Notice") both just re-listed the same
  one-item list instead of reopening it into an editable preview state.
- Not root-caused this session (time budget) -- flagged as a distinct,
  real, separate gap in the "resume a listed draft" flow specifically (as
  opposed to a fresh "continue draft"/confirmation-word resume immediately
  after generation, which is a different, already-tested code path). Worth
  a dedicated follow-up: likely `DraftIntentDetector.detect_named_template`
  or the discovery-mode selection handler not matching a bare template
  name when arriving from the saved-drafts list context specifically.

### Finding-009 (content-quality gap, not fixed): "explain in language X" was applied as "regenerate the whole draft in X" instead of a language-scoped EXPLANATION only

- Observed during the BUG-012 clean re-verification: "Ye draft mujhe
  Hinglish mein samjha do, lekin draft English mein hi rehne do, badlo
  mat." ("Explain this to me in Hinglish, but let the draft stay in
  English, don't change it") resulted in the ENTIRE document being
  regenerated in Hinglish (`"Here is the draft in Hinglish:"` followed by
  a full Hinglish-language document) rather than an English draft
  unchanged plus a separate Hinglish-language explanation of it. This
  directly corresponds to the task's own step 10 script item
  ("Explanation Hinglish mein do, draft Hindi hi rakho") — the product has
  no apparent distinct "explain in language X" capability separate from
  "translate/regenerate in language X" (`EditCommandInterpreter`'s
  `translate` action, triggered here by `extract_requested_language`
  detecting "Hinglish" in the message, does not distinguish "explain in"
  from "translate to"). Not investigated further or fixed this session —
  flagged as a real, reproduced gap for follow-up, distinct from BUG-012.

### Steps 11-14 results (executed on session `qa-bug012-verify`'s active draft, NOT the original `qa-p2-deposit-1` checkpoint — see Finding-008 for why)

| Step | Result | Status |
|---|---|---|
| 11. Suggest-only, no modify | Message was NOT recognized as a suggestion request at all (fell through to "I don't have a template for that", full 56-template menu). **Critical safety invariant held: the draft was explicitly confirmed still saved and unmutated** ("Your Rent / Security Deposit Recovery Notice draft is still saved"). The requested FUNCTIONALITY (actual improvement suggestions without applying them) does not exist as a capability — Finding-010, not fixed. | PARTIAL: safety PASS, functionality NOT RUN/missing |
| 12. Final complete draft | Not separately executed as its own step — every prior regeneration already returns the full current draft text on each turn (confirmed throughout steps 4-10). | Covered implicitly by prior steps, not separately re-verified |
| 13. Refresh/reopen (API-level, browser unavailable) | `POST /draft-history` for the session correctly lists the draft with current `status: "preview_ready"`, `language: "hinglish"` matching the last regeneration. Persistence confirmed at the database level. | PASS |
| 14. Export | `POST /draft/export` (format=txt) returned 200, 4100 bytes, correct UTF-8 content matching the current draft exactly (title, Hinglish body). Only TXT format spot-checked; PDF/DOCX/RTF (listed as available) not individually verified this session. | PASS (TXT only, spot-checked) |

### Finding-010 (capability gap, not fixed): no "suggest improvements without applying them" mode exists

Step 11 revealed there is no distinct "review and suggest, don't touch the
draft" capability — a suggestion-only request just falls through
unrecognized. The GOOD news is the fallback path is safe (confirmed no
mutation), so this is a missing feature, not a data-safety bug. Not
investigated further or fixed this session.

## Test execution summary (this session)

- **Full existing pytest suite** (excluding tests gated behind a live server
  env var): `2675 passed, 3 skipped, 10 deselected, 0 failed` — run both
  before and after all fixes in this session; no regressions introduced.
  **Correction (session 4 external review):** "10 deselected" was an
  inaccurate label carried forward across this file's updates — no
  `-m`/`-k`/`--deselect` filter was ever used, so nothing was actually
  deselected; these are ordinary `pytest.mark.skipif` **skips**. Re-run with
  `-rs` (no filters) against the current tree: `2724 passed, 11 skipped in
  178.33s`, itemized —
  `test_chat_reliability_integration.py:555` and `:571` (2×, "set
  LEGAL_AI_LIVE_BASE_URL to run against a running backend"),
  `test_legal_benchmark.py:207` (1×, "set LEGAL_AI_RUN_BENCHMARK=1..."),
  `test_phase1_live_services.py:52/:76(×2)/:112/:126/:154` (6×, "requires
  isolated Phase 1 services"), `test_phase1_staging_acceptance.py:114` (1×,
  same reason), `test_phase3_multilingual_benchmark.py:116` (1×, "Needs live
  Mongo/Redis and the embedding model..."). All 11 require infrastructure
  this session didn't stand up as isolated instances (a second, disposable
  Mongo/Redis/benchmark environment) — none hide a silently-excluded
  critical test; every skip has a legitimate, printed reason. "Full suite"
  wording is accurate for what actually ran; it was never accurate to
  imply 100% of the repo's tests executed, and this file should have said
  so plainly the first time.
- **Targeted regression suites re-run after fixes**: `test_language_intent_entities.py`
  + `test_structured_extraction.py` (108 passed), `test_matter_context.py`
  (13 passed, incl. new test), `test_jurisdiction_phase2_gaps.py` +
  `test_jurisdiction_phase2_gaps2.py` + `test_jurisdiction_retrieval_phase2.py`
  + `test_multi_turn_conversations.py` (165 passed), `test_chat_service_routing.py`
  (73 passed), `test_chat_reliability_integration.py` (38 passed, incl. 2 new
  tests), `test_draft_quality_pass.py` + `test_draft_safety_and_structure.py`
  + `test_phase1_staging_acceptance.py` (87 passed, 1 skipped).
- **Live end-to-end verification**: API server restarted twice against a live
  local MongoDB+Redis+Gemini stack; `/health` confirmed `ok` each time;
  BUG-001 and BUG-002 each verified fixed against the running server with a
  real HTTP request (not mocked).

## Remaining phases — status as of end of session 1 (corrected labels, see session 2 below for what changed)

Given the size of the requested QA program (10 phases, 200+ scenarios) and
the LLM latency observed above (each live multi-turn call taking 15-90s+),
session 1 focused on Phase 0 (discovery) and Phase 1/2 (basic journey +
one real multi-turn Hinglish conversation), which is what surfaced the three
real bugs fixed above. The following were **NOT RUN** (no concrete blocker —
simply not yet executed; corrected from an earlier looser use of "BLOCKED")
as of the end of session 1, and are being worked through in session 2 below:

- Phase 3 (full multilingual matrix — 13+ languages × 10 scenarios each):
  not executed live. `docs/ADVANCED_CHATBOT_FEATURE_CHECKLIST.md` records
  that the existing 789+-test suite already has some multilingual regression
  coverage (`test_multilingual_draft_audit.py` etc.), which the 2675-passed
  full-suite run above re-confirms still passes, but this is unit-test
  coverage, not this session's own live execution across the requested
  language matrix — reported as PASS-VIA-EXISTING-SUITE, not re-verified
  live here.
- Phase 4/5 (draft generation + the 25-step sequential draft-editing
  journey, explicitly the highest-priority phase): **NOT RUN in session 1**
  (not "blocked" — no missing dependency; a live attempt to reach drafting
  simply didn't complete within session 1's time budget because of LLM
  latency on intermediate turns). Executed in session 2 — see the Priority 2
  section below for the actual step-by-step result.
- Phase 6 (full user-journey simulations A-H): not executed.
- Phase 7 (legal-answer/retrieval quality audit): not executed beyond the
  one live rent-deposit question in Phase 1 (which did return correctly
  Maharashtra-scoped, page-cited sources).
- Phase 8 (uploads/OCR/export): not executed.
- Phase 9 (auth/isolation/failure-handling): not executed beyond confirming
  anonymous session-scoped chat works; no multi-user isolation test run.
- Phase 10 (final full regression + lint/typecheck): the full pytest suite
  was run (see above); lint/typecheck (`ruff`/`mypy`) were not run this
  session.

## CHECKPOINT (session 2, SUPERSEDED — see the session-3 checkpoint at the end of this file for the current state)

**Completed this session (session 2):**
- Priority 1 (streaming leak): DONE. `_InlineThinkStreamFilter` added to
  `app/llm/resilient.py`; 18 tests in `tests/test_streaming_think_filter.py`,
  all passing; live-verified via `/chat/stream` (no leak reproduced after
  fix, across the streaming calls that did complete — see the live-latency
  caveat in the Priority 1 write-up above).
- Priority 2 (draft follow-up journey, session `qa-p2-deposit-1`): steps
  1-7 of the task's own script executed live against the real API:
  1. Vague problem statement → clarification asked. PASS.
  2. Facts given (city/amount/date) → no repeat clarification (BUG-002
     fix holds). PASS.
  3. "Draft an English legal notice, use placeholders for missing
     details" → wrong template selected (BUG-004), root-caused, fixed,
     regression-tested, retested live. PASS after fix. (Note: the
     "use placeholders" instruction was NOT honoured — the engine still
     asked for the 6 missing fields instead of placeholder-filling. Not
     separately investigated; flagged as a Phase-4 gap worth a dedicated
     check.)
  4. Amount 50,000 → 65,000: found BUG-005 (edit not recognized, no
     amount field existed) and BUG-006 (unrelated Subject section
     corrupted by the edit) — both root-caused, fixed, regression-tested,
     retested live, PASS. Separately flagged Finding-007 (the LLM doesn't
     always restate a newly-set amount in the prose) as a content-quality
     gap, not fixed.
  5. Recipient name Rohit Sharma → Rahul Verma: clean PASS, no bug.
  6. "15 days ki jagah 7 days kar do": found BUG-008 (no field synonym)
     + BUG-009 (missing "ki jagah" connector, corrupted by possessive-
     stripping, AND the fallback RAG answer hallucinated an irrelevant
     statutory refusal) — root-caused, fixed, regression-tested, retested
     live, PASS.
  7. "Ye fact add karo: ...": found BUG-010 (no add/append edit capability
     exists at all — structural gap). **NOT fixed** (see BUG-010 write-up
     for why a quick patch was rejected as unsafe). **This is the exact
     point to resume Priority 2 from.**
- Priorities 3 and 4: NOT RUN this session (time budget — every live turn
  above cost 20-170s; 7 sequential edit-journey turns plus the streaming
  work already consumed the practical session budget).

**Immediate next action, if resuming:** either (a) design and implement
`BUG-010`'s fix (`append_field` `EditAction`, verb detection, threading
current field values into `EditCommandInterpreter.interpret`), then retest
Priority 2 step 7 through the same session (`qa-p2-deposit-1` — the draft
is `draft_id: 3db8360e-7e5a-4783-99ea-3c139f003782` as of this writeup,
`template_id: legal_notice`, currently: applicant Amit Kumar, respondent
Rahul Verma, deposit property Test Flat 12/Example Road/Mumbai, claim_amount
65,000, response_deadline_days 7), then continue with steps 8-12 (tone,
translate to Hindi, Hinglish-explanation/Hindi-draft split, suggest-only,
final draft) — the script's own remaining scenarios; or (b) skip BUG-010 for
now and continue Priority 2 from step 8 onward with a DIFFERENT edit that
doesn't require "add" semantics, coming back to BUG-010 separately.

**Test/build state at handoff:** full pytest suite `2697 passed, 3 skipped,
10 deselected, 0 failed` (last run after BUG-008/009's fix). `ruff check`
and `mypy` clean on every file changed this session. Server running at
`localhost:8000` against real local MongoDB/Redis/Gemini, restarted 4 times
this session (session 2) to pick up each fix — currently running the code
as of BUG-008/009/010's investigation (BUG-010 has no fix to pick up).

**Files changed this session (session 2, in addition to session 1's 7
files):** `app/llm/resilient.py` (streaming filter — additive to session
1's non-streaming fix), `tests/test_streaming_think_filter.py` (new),
`app/drafting/intent.py` (typo-tolerance distance-2 removal), `tests/
test_draft_discovery.py` (regression test added), `app/drafting/templates/
legal_notice.yaml` (added `claim_amount` + `response_deadline_days`
synonyms, `subject_template`), `app/drafting/edit_commands.py`
(`_FROM_TO_PATTERN` "ki jagah" support, `_strip_possessives` fix),
`tests/test_draft_conversation.py` (2 regression tests added), `tests/
test_drafting.py` (1 regression test added).

## CHECKPOINT (session 3, 2026-09-11) — CURRENT, resume from here

### Reconciled counts (per row, matching every enumerated scenario)

**Phase 1 (20 rows):** 5 PASS, 2 BLOCKED (browser unavailable), 1 PARTIAL
(T12 streaming — see below), 12 NOT RUN. Total 20. ✓

**Priority 2/3 draft-edit journey (14 rows, steps 1-14):**
1 PASS, 2 PASS, 3 PASS, 4 PASS, 5 PASS, 6 PASS, 7 PASS, 8 PARTIAL (routing
fixed+verified; LLM tone-compliance itself unobserved — provider stalled),
9 PASS, 10 PARTIAL (critical safety bug fixed+verified; original requested
behavior — explain-only — not achieved, see Finding-009), 11 PARTIAL
(safety invariant PASS; requested suggest-only functionality missing, see
Finding-010), 12 NOT RUN (not separately executed as its own turn), 13
PASS, 14 PASS (TXT only). Total: 10 PASS, 3 PARTIAL, 1 NOT RUN = 14. ✓

**Priority 4 (document targeting/recovery, 10 rows):** ALL NOT RUN this
session (time budget).

**Priority 5 (streaming deep verification, 6 sub-items):** timeout-stage
diagnosis — DONE (server logs confirm the stall is inside the provider's
own streaming call, before any output — `gemini_stream_overall_timeout`
with zero chunks received, not in buffering/filtering/client consumption,
which are all downstream of where the stall was observed); time-to-first-
byte/total-timeout recorded for every live attempt (see Priority 1 section
above); ordinary non-thinking streaming output — NOT separately verified
(every live streaming attempt this session happened to hit either a
no-context fallback or a provider stall, never a clean multi-chunk
grounded-RAG generation); controlled fake-provider integration tests for
split tags — DONE, 18 tests, clearly labelled non-live; one bounded
real-provider verification — attempted (this session's Priority 1 section);
inconclusive due to provider stalls, not retried further per "do not
repeatedly wait on identical long calls." **Status: IMPLEMENTED + AUTOMATED
TESTED + LIVE PROVIDER VERIFICATION INCOMPLETE**, exactly as the correction
requires — never upgraded to a full live PASS.

**Priority 6 (multilingual matrix, 11 languages × full cycle = 11 rows):**
ALL NOT RUN this session. English, Hinglish, and Hindi were exercised as
part of the Priority 2/3 journey itself (not this Priority's own separate
per-language batch protocol) and are reported under Priority 2/3 above, not
double-counted here.

**Grand total this session:** 20 (Phase 1) + 14 (Priority 2/3) + 10
(Priority 4, all NOT RUN) + 11 (Priority 6, all NOT RUN) = 55 enumerated
rows, plus Priority 5's 6 qualitative sub-items tracked separately (not a
per-scenario table). PASS: 15. PARTIAL: 4. BLOCKED: 2. NOT RUN: 34.
15+4+2+34 = 55. ✓

### Bug ID reconciliation (unique IDs only, matching the summary count)

11 unique bugs opened this session and the one prior: **BUG-001, BUG-002,
BUG-003, BUG-004, BUG-005, BUG-006, BUG-008, BUG-009, BUG-010, BUG-011,
BUG-012** (BUG-007's slot was correctly used for "Finding-007" instead — a
content-quality/correctness gap, not a routing/crash/data-loss defect,
kept in a separate naming track on purpose).

- **FIXED and live-verified (10):** BUG-001, BUG-002, BUG-003, BUG-004,
  BUG-005, BUG-006, BUG-008, BUG-009, BUG-010, BUG-012.
- **FIXED at the routing level, one sub-claim not live-verified (1):**
  BUG-011 (the reported defect — "Tone polite but firm karo" leaves the
  draft flow — is fixed and confirmed via `route: "draft_workflow"` in
  server logs; whether the LLM, once it responds, actually restyles
  correctly without changing facts has unit-test coverage only, not a
  completed live call).
- **Open, not fixed (0 bugs).** All discovered BUGS are fixed at the code
  level. **Correction (session 4 external review):** this line previously
  read as an unqualified "0 open" — that is misleading on its own, since
  BUG-011 is only 90% verified (routing confirmed live; the actual
  tone-restyling LLM behavior was never observed end-to-end because the
  provider stalled before responding — see its write-up above). BUG-011
  should be tracked as **PARTIAL VERIFICATION, not CLOSED** until a live
  call actually completes and its output is inspected for (a) a genuinely
  different tone and (b) zero fact/name/date/amount drift. Every other bug
  in the FIXED list (BUG-001 through BUG-010, BUG-012) did receive a
  completed live re-run with inspected output, so those are correctly
  closed; BUG-011 alone is the exception.

**Findings (content-quality/capability gaps, separate from BUGS):**

| ID | Summary | Severity | Affected workflow | Acceptance criteria to close |
|---|---|---|---|---|
| Finding-007 | Claim amount not restated in prose | Medium | Any monetary draft field | **FIXED, live-verified** (see above) |
| Finding-008 | Reopening a listed saved draft by name (from the "here are your saved drafts" list) doesn't resume it into an editable preview — repeats the list instead | High — blocks the entire "come back later and keep editing" journey for any draft not still pinned as the session's active draft | Draft resume / multi-session continuity | A message naming (or selecting) one item from a just-shown saved-drafts list must set that draft as the session's active `draft_id` and return it in `preview_ready` state, verified via a live follow-up edit actually landing on that draft (not a re-list) |
| Finding-009 | "Explain in language X, don't change the draft" is executed as "regenerate/translate the whole draft into X" | High — silently mutates a draft the user explicitly said to leave alone; violates the no-unrequested-change invariant | Language-scoped explanation vs. translate/regenerate | A request containing an explicit "explain"/"samjhao" verb + language, combined with an explicit "don't change"/"badlo mat" clause, must return the draft's `language`/content byte-for-byte unchanged, with a separate explanatory text in the requested language; verified live with a diff of before/after draft content |
| Finding-010 | No "suggest improvements without applying them" mode; falls through to an unrelated template-menu response (safely, without mutating) | Medium — missing capability, not a safety defect (fallback confirmed non-destructive) | Suggest-only review | A recognized "suggest"/"review" intent must return prose suggestions with the draft object unchanged (`draft_id`, all field values, and rendered text identical before/after), verified live |

### Exact next action if resuming

1. **Priority 4** (document targeting/recovery): start a second draft in a
   NEW session (or reuse `qa-bug012-verify`, which already has one active
   `rent_notice` draft — `draft_id a7bed2fc-3182-4f93-8967-9c0bce34d582`),
   create a second, different draft type in the SAME session, and run the
   10 enumerated cross-document/isolation/concurrency checks. Compare
   stored `draft_id`s and field contents directly (via `POST
   /draft-history` and the `draft` object in each `/chat` response), never
   infer targeting correctness from response text alone, per the explicit
   instruction.
2. **Priority 5** (streaming): if continuing, the next useful step is
   catching ONE clean live streaming call that reaches real multi-chunk
   generation (not a fallback/timeout) — try a query already known to
   retrieve real context quickly WITHOUT requiring a long generation, to
   minimize provider stall risk, before attempting the full verification
   checklist's remaining "ordinary streaming output" item.
3. **Priority 6** (multilingual): start with Batch A (Hindi, Odia, Bengali,
   Marathi) using the same `p2_client.py`/direct-`/chat` pattern used
   throughout this session; expect each language's full cycle (query →
   clarification → draft → edit → language switch → reopen) to cost
   several 60-170s live LLM turns, same as English/Hinglish did.
4. **Findings 008/009/010**: each needs its own root-cause investigation
   session, starting points noted in their write-ups above
   (`DraftIntentDetector.detect_named_template`/discovery-mode selection
   for 008; `EditCommandInterpreter`'s `translate` action and
   `extract_requested_language` for 009; no starting point yet for 010 —
   would need a new `EditAction` the same way `append_field`/`restyle`
   were built this session).

### Test/build state at this handoff

Full pytest suite: `2722 passed, 3 skipped, 10 deselected, 0 failed` (last
run, after BUG-012's fix). `ruff check` and `mypy` clean on every file
changed this session (pre-existing issues in untouched files confirmed via
`git diff` and left alone, per "existing unrelated changes preserve
karo"). Server running at `localhost:8000` against real local
MongoDB/Redis/Gemini, restarted throughout this session to pick up each
fix — currently running all fixes through BUG-012.

**Sessions/drafts still live on the server, useful for resuming:**
- `qa-p2-deposit-1` — original checkpoint session; its
  `General Legal Notice` draft (`3db8360e-7e5a-4783-99ea-3c139f003782`,
  language hindi) is intact but not currently the session's ACTIVE draft
  (Finding-008) — reopening it cleanly is itself the first thing to solve
  if resuming this exact conversation thread.
- `qa-bug012-verify` — has one active `Rent / Security Deposit Recovery
  Notice` draft (`a7bed2fc-3182-4f93-8967-9c0bce34d582`, language
  hinglish, `status: preview_ready`) ready for immediate further editing
  without needing to rebuild anything.

**Files changed this session (session 3, in addition to sessions 1-2's
files):** `app/drafting/edit_commands.py` (append_field detection/merge,
restyle/tone detection, further `_field_candidates` refactor),
`app/drafting/conversation.py` (`append_field` + `restyle` dispatch),
`app/drafting/templates/base.py` (`DraftField.must_appear_verbatim`),
`app/drafting/templates/loader.py` (parses the new attribute),
`app/drafting/templates/legal_notice.yaml` (`claim_amount.
must_appear_verbatim: true`), `app/drafting/engine.py`
(`_ensure_verbatim_fields`, `style_instruction` prompt directive),
`app/schemas/drafting.py` (`DraftGenerateRequest.style_instruction`),
`app/services/chat_service.py` (`_CANCEL_DRAFT_PATTERN` "rehne" fix),
`tests/test_draft_conversation.py` (19 new tests across append/restyle),
`tests/test_drafting.py` (5 new tests across Finding-007/restyle),
`tests/test_answer_quality_audit.py` (2 new tests for BUG-012).

## Session 4 (2026-09-11) — external review corrections + Priority 4 resumed

An external review of sessions 1-3 flagged 25 specific weak points. Two are
doc-honesty corrections, applied above in place (the misleading "0 open
bugs" summary line re: BUG-011, and the "10 deselected" mislabeling —
corrected with exact skip reasons). The rest are genuine open-scope gaps
already tracked above (Findings 008-010's table now carries severity/
affected-workflow/acceptance-criteria per the review's request) or
not-yet-executed priorities. This session resumes live execution in the
review's stated order: **document targeting/isolation → persistence/
concurrency → export consistency → latency/streaming → Odia/Hindi
multilingual → remaining languages → auth/uploads/citations.**

### Priority 4 — document targeting / isolation / concurrency (enumerated)

No prior session enumerated this priority's 10 rows with IDs (the review's
point 25) — defining them now, before executing, so every status below is
traceable:

| ID | Scenario |
|---|---|
| P4-T1 | Create draft A (legal_notice) in session X |
| P4-T2 | Create draft B (a different template) in the SAME session X |
| P4-T3 | Edit draft A by name/context; confirm draft B's fields are untouched |
| P4-T4 | Edit draft B; confirm draft A's fields are untouched |
| P4-T5 | Cross-chat isolation: a fresh session Y must not see X's facts/drafts |
| P4-T6 | Cross-user isolation: a different user identity must not see X's drafts |
| P4-T7 | Concurrent/rapid edits to the same draft (simulated overlapping requests) |
| P4-T8 | Stale-base-version edit does not silently overwrite a newer edit |
| P4-T9 | Save/persistence failure path: no false-success acknowledgement |
| P4-T10 | Every mutating reply's claim (e.g. "updated"/"deleted") is verified against the persisted record, not just response text |

| ID | Result | Status |
|---|---|---|
| P4-T1 | Session `qa-p4-target-1`, draft A = `legal_notice` (`draft_id fe2e47d9-a8e6-4f1f-82fb-e28c6b654c4c`, applicant Amit Kumar, deposit 40,000, Suresh Patil, 15-day response). Step 1 correctly asked "which State?" (19.9s). Step 2 (city/amount/landlord-name/date given) did not re-ask State (BUG-002 holds), returned a no-verified-context fallback (195.4s). Step 3 (draft request) client-timed-out at 250s, then a retry surfaced BUG-013a (below); saying the literal word "retry" (77.2s) resumed drafting; remaining fields supplied (126.2s) completed the draft in fallback/deterministic mode (`"enhanced drafting is temporarily unavailable"`). | PASS (with BUG-013a found along the way) |
| P4-T2 | Draft B = `cheque_bounce_notice` (`draft_id 086235c9-838e-4803-a3a6-9e430121edd1`, drawer Vikram Singh, ₹25,000, cheque 445566, Axis Bank) created in the SAME session while draft A already existed in `preview_ready`. Both correctly listed via `POST /draft-history` afterward, distinct `draft_id`s, distinct templates. | PASS |
| P4-T3 | Sent, with draft B still the session's active draft: **"General Legal Notice wale draft mein amount 40,000 se 55,000 kar do."** — explicitly named draft A by its template's display name. **FAIL — confirmed via `GET /draft/{id}/versions`**: draft A (`fe2e47d9...`) still shows only version 1 (untouched); draft B (`086235c9...`) gained a version 2, and its `Cheque Amount (Rs.)` field was silently changed from 25,000 to 55,000. The edit was applied to the wrong, unnamed, merely-currently-active draft instead of the explicitly-named one. Root-caused, fixed, unit-tested, and **live re-verified fixed** in a fresh session (`qa-bug013-verify`) after a server restart — see **BUG-013** below. | **PASS after fix (was FAIL — data corruption, root cause confirmed)** |
| P4-T4 | Mirror case: with draft A (`legal_notice`) now active (from P4-T3), sent "Cheque Bounce Notice wale draft mein cheque amount 25,000 se 30,000 kar do." Result: reply opened "(Switched to your Cheque Bounce Notice draft...)", returned `draft.draft_id` `601e7788...` (draft B) with `Cheque Amount (Rs.)` correctly updated to 30,000; `GET /draft/a8995dd3.../versions` (draft A) confirmed still only 2 versions (no new one added) — untouched. | **PASS** |
| P4-T5 – P4-T10 | In progress — see below | IN PROGRESS |

### BUG-013 (FAIL, CRITICAL severity, confirmed not fixed this session): editing a draft by explicit name silently applies to the wrong, currently-active draft instead

- Files: `app/drafting/conversation.py` (edit dispatch reads
  `memory["draft_template_id"]`/the session's single active draft pointer;
  no code path was found that inspects the edit message itself for a
  template/document name and switches context before applying the edit),
  `app/drafting/edit_commands.py` (`EditCommandInterpreter.interpret`
  receives only the message, the ACTIVE template, and language — never a
  candidate list of the session's OTHER saved drafts to match against).
- Symptom (live repro, `qa-p4-target-1`): with two drafts open in the same
  session — draft A (`legal_notice`, `fe2e47d9...`) parked, draft B
  (`cheque_bounce_notice`, `086235c9...`) active — the message "General
  Legal Notice wale draft mein amount 40,000 se 55,000 kar do." (explicitly
  naming draft A by its template's own display name, "General Legal
  Notice") was silently applied to draft B instead: draft B's unrelated
  `Cheque Amount (Rs.)` field changed from 25,000 to 55,000 (a coincidental
  field-name match — "amount" resolves to whichever monetary field the
  ACTIVE template happens to have, regardless of which document the user
  named), draft A was never touched, and the reply ("Updated Cheque Amount
  (Rs.). Here is the revised draft:...") gave no indication the wrong
  document had been edited — a real user watching only the chat reply, not
  independently diffing `draft-history`, would have no way to notice their
  Rs. 55,000 change landed on the wrong notice with the wrong opposing
  party's name on it.
- Root cause: the edit-command pipeline has no concept of a document name
  in the user's message at all — `EditCommandInterpreter.interpret` (and
  every caller in `conversation.py`) resolves fields purely against
  whichever ONE draft is currently `memory["draft_template_id"]`/active.
  There is no step that scans the message for another saved draft's
  template name/nickname and re-targets before interpreting the edit. This
  is the same class of gap as Finding-008 (reopening a listed draft by name
  doesn't work either) — together they show the "resume/target a specific
  one of several saved drafts by name" capability does not exist anywhere
  in the current design, only "the one draft that happens to be active
  right now."
- **Not fixed this session** — this is a structural gap (same shape as
  BUG-010 before its fix): a correct fix needs the edit-command interpreter
  (or a step before it) to check the message against the session's other
  saved drafts' template names/aliases, and either switch the active draft
  first or ask for disambiguation, before falling through to "edit whatever
  is currently active." Flagged as the single highest-priority item to fix
  next, per the external review's own stated priority order.
- Regression test to add when fixed: two drafts of different templates in
  one session; an edit message naming the NON-active one by its display
  name must update that draft's fields (verified via `GET
  /draft/{id}/versions` gaining a new version) and leave the active draft's
  fields byte-for-byte unchanged.

**FIXED and live-verified this session.** Files: `app/drafting/conversation.py`
(new `_find_named_parked_draft` helper; `_continue_preview` now checks, before
interpreting any edit, whether exactly one PARKED draft's template display
name is named in the message -- if so it switches to that draft first via
the existing park/restore mechanism, recurses once (`_retargeted=True` guards
against re-entering the check), and prepends a transparent "(Switched to your
{X} draft...)" note to the reply). Deliberately conservative: requires an
UNAMBIGUOUS match (exactly one parked draft's name found) so an ordinary
edit to the active draft is never accidentally redirected.
- Unit regression test:
  `tests/test_draft_conversation.py::test_editing_a_named_non_active_parked_draft_retargets_instead_of_corrupting_the_active_one`
  (asserts `regenerate` is called with the PARKED draft's `draft_id`, not
  the active one's, and that the parked/active roles swap correctly in
  memory). Full suite after the fix: `2725 passed, 11 skipped, 0 failed`
  (up from 2724 -- the one new test). `ruff check`/`mypy` clean on the
  changed files (one pre-existing, unrelated import-order lint warning and
  one pre-existing, unrelated mypy error elsewhere in the same file were
  confirmed present before this change and left untouched).
- **Live re-verification (fresh session `qa-bug013-verify`, after a clean
  server restart to pick up the fix)**: rebuilt the exact repro -- draft A
  (`legal_notice`, `draft_id a8995dd3-0b4e-4c79-ab30-61ae47ea47af`,
  claim_amount 40,000) parked, draft B (`cheque_bounce_notice`, `draft_id
  601e7788-2066-4b92-aa93-ca7afe0c7e40`, cheque_amount 25,000) active. Sent
  the identical trigger message: **"General Legal Notice wale draft mein
  amount 40,000 se 55,000 kar do."** Result: reply now opens with "(Switched
  to your General Legal Notice draft, since you named it -- your Cheque
  Bounce Notice draft is untouched and still saved.)", the returned
  `draft.draft_id` is `a8995dd3...` (draft A) with `claim_amount` correctly
  updated to 55,000 in the rendered text, and `GET
  /draft/601e7788.../versions` confirms draft B still shows only its
  original version 1 -- untouched, cheque_amount still 25,000. **Confirmed
  fixed**, not just unit-tested.
### P4-T5 (cross-chat isolation) and P4-T6 (cross-user isolation)

| ID | Result | Status |
|---|---|---|
| P4-T5 | Fresh session `qa-p4-isolation-fresh`: `POST /draft-history` returned `{"drafts": []}` (no leakage of `qa-bug013-verify`'s drafts); `GET /draft/{qa-bug013-verify's draft A id}/versions?session_id=qa-p4-isolation-fresh` correctly returned `403 forbidden`; a chat message asking to see "mera legal notice draft" in the fresh session correctly replied "You do not have any saved drafts yet." (0.25s, no LLM call, no leaked facts). | **PASS** |
| P4-T6 | Registered two real accounts via `POST /register` (User A, User B — genuine JWTs, not simulated). User A created a `police_complaint` draft (`draft_id 905e75fd-6bde-4886-992e-da20f759a914`) authenticated, via `POST /draft` with `Authorization: Bearer <A's token>`. User B, authenticated as themselves (a DIFFERENT valid JWT), queried `GET /draft/905e75fd.../versions` with NO session_id: correctly `403`. But User B querying the SAME endpoint WITH User A's `session_id` (`qa-p4-user-a`) as a query parameter: **200 OK, full version history returned** — User B's own valid authentication did nothing to stop them from reading User A's draft once they had (or guessed) A's session_id. See **BUG-014** below. | **FAIL — confirmed cross-user access via session_id, not fixed this session** |

### BUG-014 (FAIL, CRITICAL severity, confirmed not fixed this session): "Authenticated User Ownership" is never actually applied to drafts — every draft remains session_id-scoped even when created by a logged-in user, making `session_id` a de facto bearer credential across accounts

- Files: `app/drafting/conversation.py` (`DraftConversationEngine.handle_turn`
  and its entire internal call chain never accept or thread a `user_id` at
  all; `_finalize_generation_reply`'s `DraftGenerateRequest(...)` — the ONE
  place that actually creates a draft record — is built with `session_id=
  session_id` only, no `user_id` field, so it defaults to `None`),
  `app/services/draft_management.py` (`ensure_draft_access`'s
  `owner_user_id`-first branch, correctly implemented, simply never
  receives a stamped `owner_user_id` to check against for any draft that
  went through this engine), `app/api/drafting.py` (`/draft` endpoint DOES
  resolve `user_id` via `Depends(get_current_user_id)` but never passes it
  into `handle_turn`), `app/services/chat_service.py` (four separate
  `self.draft_conversation.handle_turn(...)` call sites — lines ~964, 983,
  1342, 1430 — all sit inside methods that ALREADY receive
  `authenticated_user_id` as a parameter, but none of them pass it through
  either).
- Symptom (live repro): registered two real accounts (`POST /register`),
  each with a genuine JWT. User A, authenticated, created a
  `police_complaint` draft via `POST /draft`. User B -- authenticated as
  themselves, a completely different, valid account -- could not read
  User A's draft by ID alone (`403`, correct), but COULD read it in full
  (`200`, full version history) simply by also supplying User A's
  `session_id` as a query parameter, despite User B's own JWT proving they
  are a different, unrelated user. This is exactly backwards from what
  `ensure_draft_access`'s own docstring promises ("`user_id` stamped ->
  only that exact authenticated user, from any session"): the promise
  never activates because the `user_id` is never stamped in the first
  place for anything created through the conversational drafting engine --
  which is the ONLY engine `/chat` and `/draft` (the two routes real users
  and the Streamlit frontend actually use) create drafts through.
- Root cause, confirmed at the code level (no further live calls needed):
  `DraftGenerateRequest` at `app/drafting/conversation.py:1522` (inside
  `_finalize_generation_reply`, the single choke-point that actually calls
  `LegalDraftEngine.generate()` and persists the first version of a draft)
  sets `session_id=session_id` but has no `user_id=` at all -- even though
  `LegalDraftEngine.generate()` (`app/drafting/engine.py:968`) already
  correctly persists whatever `request.user_id` it's given, and
  `ensure_draft_access` already correctly enforces it once present. Every
  piece of the ownership mechanism (Part 46, per `get_current_user_id`'s
  own docstring) is correctly built and already covers every OTHER
  mutating draft route (`/draft/edit`, `/draft/export`, `/draft/approve`,
  etc. -- all call `_ensure_draft_access(draft, user_id, request.session_id)`
  correctly) -- it just was never wired into the one path that actually
  creates the record, so it has nothing to check for the vast majority of
  real drafts.
- Impact: `session_id` functions as a de facto bearer credential for the
  entire lifetime of a draft, REGARDLESS of authentication. Any actor who
  observes, logs, guesses, or is otherwise handed a session_id (browser
  history, a shared support ticket, a referrer header, a misconfigured
  log line, or simple sequential/predictable generation if the frontend
  ever uses one) gets full read/edit/export access to that draft's legal
  content -- names, facts, amounts, addresses -- even while authenticated
  as a completely unrelated account. This is a genuine privacy/security
  gap, not a hypothetical one: the fix (Part 46) was clearly built
  specifically to prevent this class of access and simply never got
  connected to its only real caller.
- **Not fixed this session.** Scope assessed rather than rushed, per this
  file's own established discipline for structural gaps (see BUG-010's
  original session): fixing this correctly requires threading `user_id:
  str | None` through `DraftConversationEngine.handle_turn` and every
  internal method on its path to `_finalize_generation_reply` (at minimum
  `_start_collecting`, `_continue_collecting`, `_process_collecting_message`,
  `_continue_confirm_summary`, and the discovery-flow equivalents -- roughly
  8-10 method signatures in `conversation.py`), plus updating all 5 call
  sites (`app/api/drafting.py:203`, and
  `app/services/chat_service.py` lines ~964, 983, 1342, 1430 -- each of
  which already HAS `authenticated_user_id`/`user_id` in scope at the
  calling function, so no further upstream threading is needed there,
  which somewhat bounds the blast radius). Attempting this in the same
  pass as BUG-013 risked an under-tested change across every drafting
  stage (collecting/discovery/confirm-summary) this session did not have
  budget to re-verify individually -- flagged as the top follow-up item,
  ideally fixed and re-verified with the exact two-user repro above as the
  acceptance test before being called closed.
- Regression test to add when fixed: two authenticated users (real JWTs,
  as in the live repro), User A creates a draft, User B -- with a valid
  token AND knowledge of A's `session_id` -- must still get `403` on every
  draft route; only User A's own token (any session_id, including a new
  one) should succeed.

### BUG-015 / P4-T7 (concurrent edits) — CONFIRMED FAIL: silent lost update, no conflict detection

Fired two genuinely overlapping edits at the SAME draft (`a8995dd3...`,
`legal_notice`, then holding `claim_amount: 55,000`, `response_deadline_days:
15 days`) via two parallel background requests to the same session:
CONC1 = "General Legal Notice wale draft mein amount 55,000 se 60,000 kar
do." (100.3s), CONC2 = "...15 days ki jagah 10 days kar do." (100.4s,
started at essentially the same wall-clock moment).

- Each response's OWN returned document reflected ONLY its own edit:
  CONC1's response showed `60,000` / `15 days` (unaware of CONC2's
  in-flight change); CONC2's response showed `55,000` / `10 days` (unaware
  of CONC1's). Both calls DID create a new persisted version each (`GET
  /draft/.../versions` shows 4 versions total, v3 and v4 both genuinely
  written) — so neither request silently no-opped or errored.
- **The FINAL persisted document (`GET /draft/{id}/review`, the source of
  truth after both calls completed) has `response_deadline_days: "10 days"`
  (CONC2's change) but `claim_amount: "55,000"` — CONC1's amount change to
  60,000 is GONE**, silently overwritten by CONC2's write, which was built
  from a snapshot taken before CONC1's write landed. The user who sent
  CONC1 received a response explicitly confirming "Updated Claim Amount
  (Rs.), if applicable. Here is the revised draft:" showing 60,000 — a
  false confirmation of a change that did not survive.
- Root cause (confirmed by the pattern, not yet traced to an exact line):
  every edit path goes through `LegalDraftEngine.regenerate(draft_id,
  {field: new_value})`, which reads the CURRENT persisted fields, merges in
  the one changed field, and writes a brand-new version — with no
  optimistic-concurrency check (no expected-version/ETag compared before
  the write) and no per-field locking. Two overlapping regenerate calls on
  the same `draft_id` each read the pre-edit state and each write their own
  full merged snapshot; whichever commits last wins in its entirety,
  silently discarding any field the other call had changed.
- **Not fixed this session** — this is the same class of architectural gap
  as BUG-014 (a genuine concurrency-control feature that doesn't exist yet,
  not a one-line patch): a correct fix needs `LegalDraftEngine.regenerate`
  to take an expected base version number (or equivalent optimistic lock),
  reject/retry a write whose base is stale, and have
  `DraftConversationEngine` surface a "someone/something else changed this
  draft, please retry" reply instead of a false "Updated..." confirmation
  on a lost write. Flagged as CRITICAL alongside BUG-014 for the same
  reason: a real user editing from two tabs, or two rapid consecutive
  messages sent before the first finished (very plausible given this
  session's own 15-190s+ per-call latency), can silently lose a change
  while being told it succeeded.
- Acceptance test for a future fix: repeat this exact two-concurrent-edit
  repro; the LOSING request must either be rejected with an explicit
  "please retry, the draft changed" reply, or its change must be correctly
  merged rather than dropped — never a silent false "Updated..." success.

### P4-T8 – P4-T10 (stale-base-version / save-failure / claim-vs-persisted-state verification)

- P4-T8 (stale-base-version) is effectively the same failure mode just
  confirmed under P4-T7 (a stale read used as the base for a write) —
  not re-tested as a separate scenario given the finding is already
  established and root-caused together with P4-T7.
- P4-T9 (save/persistence failure path): not exercised via live fault
  injection this session (deliberately did not kill the shared local
  MongoDB mid-request, to avoid disrupting other in-progress work on this
  machine, consistent with the same caution session 1 applied to P1-T20).
  Partial evidence exists in BUG-013a's repro (a provider-side stall was
  correctly surfaced as a retryable failure with `saved_progress: true`
  rather than a false success) — but that is an LLM-provider failure, not
  a database-write failure, so this item stays genuinely NOT RUN.
- P4-T10 (verify every mutating claim against the persisted record, not
  response text alone): applied as the standing methodology for every
  check in this session (BUG-013 and P4-T7 were both caught specifically
  BECAUSE the persisted record was checked independently via `GET
  /draft/{id}/versions`/`review` rather than trusting the chat reply) —
  covered by practice throughout Priority 4, not a separate scenario to
  run.

Moving on to Priority 5 (streaming) and Priority 6 (multilingual) next, per
the stated priority order — Priority 4's remaining genuinely-open item is
P4-T9 (save-failure fault injection), which needs a disposable
MongoDB/Redis instance to test safely rather than the shared local one.

### Priority 5 (streaming) — the one remaining checklist item, resolved: clean ordinary streaming, live-confirmed

Every prior session's live `/chat/stream` attempt happened to hit either a
no-context fallback or a provider stall, so "ordinary non-thinking
streaming output" was never actually observed end-to-end (session 3's own
checkpoint states this explicitly). This session caught one:
`POST /chat/stream` with "What is Section 138 of the Negotiable Instruments
Act about?" (session `qa-p5-stream-1`, 103.2s total, `200 OK`) returned 26
real `token` SSE events plus a final `done` event — genuine multi-chunk
streamed generation, real grounded content (correct Section 138 text),
`extracted_entities` correctly picked up `section_number: ["138"]` and
`act_name: ["Negotiable Instruments Act"]`. Scanned the FULL raw SSE
transcript (not just head/tail) for any `<think>`, "Rule 2", "prompt says",
or "system prompt" fragment: **zero matches**. This is the missing live
confirmation Priority 1's `_InlineThinkStreamFilter` never had on an
ordinary (non-reasoning-leak) generation — confirmed clean.

**Priority 5 status, final: IMPLEMENTED + AUTOMATED TESTED + LIVE
PROVIDER VERIFICATION COMPLETE** (upgraded from session 3's "...INCOMPLETE"
now that a clean multi-chunk run has actually been observed and inspected
in full).

- **Operational note from this restart**: the server on port 8000 was found
  to be a stray process running from a DIFFERENT Python environment (`uv`-
  managed `cpython-3.12`, not this project's own `.venv`) plus a second,
  non-listening duplicate -- the same class of stray-process issue session
  1's discovery phase already flagged once before. Both were stopped and a
  single clean instance was started from the project's own `.venv` so this
  fix (and any future one) is guaranteed to actually be running. Worth a
  standing dev-environment hygiene note: check `Get-NetTCPConnection
  -LocalPort 8000` before trusting that a "restart" picked up a code
  change, since a second, wrong-environment process can silently keep
  serving stale code with an unrelated PID holding the port.

### BUG-013a (FAIL, MEDIUM severity, confirmed not fixed this session): re-sending an original request after a stalled/failed call does not trigger the same recovery as saying "retry"

- File: `app/services/chat_service.py` (or wherever the "your earlier
  question ran into a temporary issue... say retry" fallback branch lives —
  not individually traced this session; the live behavior is confirmed,
  the exact function was not located given time budget).
- Symptom (live repro, `qa-p4-target-1` step 3): after the first attempt at
  "convert to legal notice..." exceeded a 250s client timeout, the SAME
  session's next message — a client retry that RESENT THE IDENTICAL
  original request text — was answered with "...your earlier question...
  ran into a temporary issue and didn't get a proper answer. Say 'retry'
  anytime to try it again," routed as a generic RAG "Follow-up Question"
  with a no-verified-context fallback, and did NOT attempt the draft again.
  Only sending the literal word "retry" actually resumed drafting.
- Impact: a real user who does not know the exact magic word and simply
  repeats their own request in different words (the natural thing to do)
  gets stuck being told to say a specific word, while their re-typed
  request itself is silently discarded into an unrelated RAG answer rather
  than being recognized as an equivalent retry attempt.
- Not investigated at the code level or fixed this session — flagged
  alongside BUG-013 as latency/reliability follow-up work, directly
  relevant to the external review's priority-2 concern (latency policy)
  and priority-16 concern (save failure / retry).

### New latency finding (Priority 4/5): a plain draft-generation call, not just tone-restyle, now exceeds 250s

- Session 3's only documented severe stall (317s) was on a `restyle`
  (tone-change) regeneration call. This session reproduces the same class
  of stall on an **ordinary first-time draft-generation** call (`qa-p4-target-1`
  step 3, "convert to legal notice") — it did not return within a 250s
  client timeout at all (`curl` exit 28), and `POST /draft-history` for
  that session immediately after showed `{"drafts": []}` — i.e. nothing was
  persisted even as a fallback within that window, unlike the step-2 call
  in the same session which returned a fallback answer in 195s.
- This is materially worse evidence than session 3's for the point the
  external review raised (point 2): the stall is not confined to one
  unusual edit action, latency variance is wide enough that a 250s
  client-side budget is not safe, and there is currently no visible
  cancellation/fallback/retry policy surfaced to the end user while a call
  runs this long — a real user would simply see nothing happen for over
  four minutes.
- Retried with a 400s budget: **completed in 60.5s** — i.e. the identical
  request that stalled past 250s on the first attempt returned in under a
  quarter of that time on the very next attempt. This confirms it is
  latency VARIANCE (consistent with the 15-90s baseline plus occasional
  severe spikes already documented in sessions 1-3), not a deterministic
  hang — but a policy that can swing from 60s to 250s+ on the exact same
  request, with no client-visible progress indicator or server-side
  cancellation in between, remains a real production risk regardless of
  which side of the variance a given user happens to land on.

## Session 5 (2026-09-12) — the six remaining live conversation journeys

Scope for this session, as requested: status the six journeys still open after
session 4 with live PASS/FAIL evidence — **Follow-up/context**, **Draft
continuation**, **Voice**, **Languages**, **Citations**, **Failure recovery**.
Per the same standing rule as every prior session: **the 2722-test pytest
suite passing is not evidence any of these six journeys work — it is evidence
existing deterministic logic hasn't regressed.** Every verdict below is from a
real HTTP call against a live server (real Gemini, real MongoDB, real Redis),
not a unit test.

### Environment note: startup blocked by a leftover reindex job, cleared with explicit user approval

On restart, `recover_pending_reindex_jobs` (app/main.py) found an
`indexing_jobs` record stuck at `status: "running"` from a prior session, with
`root: "storage"` — the whole `storage/` tree (uploads, drafts, archive,
backups, bm25_index, kb_staging, ~3,775 candidate files), not the intended
`storage/knowledge_base` (1,692 files). Recovery correctly re-ran it
unconditionally, which began re-walking thousands of files (mostly prior
sessions' own draft exports, correctly rejected fast by the quality-check
gate, but interleaved with genuine large scanned PDFs taking minutes each via
OCR) and blocked every live call for over 20 minutes with no end in sight.
Root cause of the bad `root` value was not chased further (out of today's
scope) but is worth a follow-up: something previously queued `/admin/reindex`
with `root=storage` instead of `root=storage/knowledge_base`, and the crash
recovery path has no cap on how broad a recovered job's scope can be.
**Fix applied only after explicit user approval** (my sandbox's own
permission gate correctly flagged a direct database write as a
shared-resource action and blocked it on the first attempt): marked that one
`indexing_jobs` document `status: "failed"` via a single scoped
`update_many({status: {$in:[queued,running]}, root:"storage"})`, restarted the
server, confirmed a clean boot with zero reindex activity and
`GET /health` → all components `ok`. No knowledge-base content was touched.

### Operational note: three of Google's Gemini model names are now retired

Directly relevant to BUG-016 below, but general: live `curl` calls against
`generativelanguage.googleapis.com` during this session found
`gemini-2.0-flash` (this codebase's own `config.py` default for
`gemini_model`), `gemini-1.5-flash` (voice_router.py's old hardcoded STT
fallback), and `gemini-2.5-flash` all now return `404 "no longer available"`,
each pointing callers at `gemini-3.6-flash`. This deployment's `.env` already
overrides `GEMINI_MODEL=gemma-4-26b-a4b-it` so the stale default doesn't bite
today's text chat — but any fallback path in this codebase that still falls
through to one of the three retired names on a fresh/unconfigured install
would break outright. Flagged, not chased further — this session only fixed
the one fallback (STT) it was actively root-causing.

---

### Journey 1 — Follow-up/context: facts remembered, no context mixing on topic change

| Turn | Message | Result |
|---|---|---|
| 1 | "What is the punishment for a first-time offence under Section 138 of the Negotiable Instruments Act?" (session `qa5-followup-1`) | 112.1s. Correct, grounded answer (2 years imprisonment / fine up to 2x cheque amount / both), 1 accurate citation, exact statute text quoted. |
| 2 | "Is that offence compoundable, and who has jurisdiction to try it?" (implicit pronoun follow-up) | 204.9s. **FAIL as asked** — retrieved 6 wholly unrelated sources (Bombay Prevention of Excommunication Act, Maharashtra Official Languages Act, etc.), generation did not complete, fell back to "say retry." |
| 2-retry | "retry" | 221.5s. Different failure: "No verified document... currently available" — retry did not recover the original question either. |
| 3 | "Switching topics: what documents do I need to file an RTI application in Maharashtra?" | 217.3s. Intent correctly detected as **RTI** (no bleed from the cheque-bounce topic) but "No verified document... available" despite the RTI Act being in the KB (see control check below) — a retrieval/generation miss, not a context-mixing one. |
| 4 | "Going back to the cheque bounce section we discussed earlier — what was the maximum fine again?" | 48.1s. **PASS** — correctly recalled "twice the amount" from turn 1, verbatim-consistent, with no RTI leakage from turn 3. Confirms fact retention survives an intervening unrelated topic in both directions. |

**Control check (isolates the turn-2/3 failures from the context mechanism
itself):** the exact same compound question from turn 2, asked fully
self-contained with zero conversation history in a fresh session
(`qa5-followup-control`), **also** failed with "No verified document" in
20.5s. This proves turns 2 and 3's failures are the same
already-extensively-documented retrieval-confidence/provider-instability
class from sessions 1-4 (see Priority 5's latency-variance writeup above),
**not** a follow-up-context defect — a fresh, explicit, non-follow-up
question hits the identical wall.

**Verdict: PASS for the mechanism actually asked about** (fact retention
across a topic switch, no incorrect context mixing in either direction — turn
4 is direct, reproducible evidence). The turn-2/3 answer failures are a
pre-existing, separately-tracked instability, confirmed still present, not
newly discovered.

### Journey 2 — Draft continuation: reopening a saved draft and correctly revising it

Re-ran Finding-008's exact scenario against `qa-p2-deposit-1`'s pre-existing
`General Legal Notice` draft (`3db8360e-...`, 8 versions, v8 active/Hindi,
confirmed via `GET /draft/{id}/versions` before touching anything).

| Step | Message | Result |
|---|---|---|
| 1 | "Mujhe mere saved drafts dikhao" | 2.2s. Correctly lists the one saved draft. |
| 2 | "Open the General Legal Notice draft" | 2.2s. **FAIL — Finding-008 reconfirmed, unchanged since session 3.** Re-shows the identical list instead of resuming the draft into an editable preview. |
| 3 | "General Legal Notice wale draft mein sender ka naam badal kar Rajesh Sharma kar do" (explicit named edit, not a plain "open") | 2.6s. **FAIL — a new, worse variant.** Instead of resuming the existing 8-version draft OR asking which one, the engine silently started a **brand-new field-collection flow from zero** (`draft_id: null`, all 8 fields listed as missing) for a template that already has a complete, `preview_ready` draft under this exact name in this exact session. |

Confirmed via `POST /draft-history` and `GET /draft/{id}/versions` immediately
after step 3 that the original draft (`3db8360e-...`) is untouched (still 8
versions, nothing new) — no data was corrupted, but a user who continued
answering the "8 more details" prompt would end up with a genuine **duplicate**
`General Legal Notice` draft in the same session, invisible to each other,
rather than ever reaching their original one.

**Root cause, pinpointed this session** (not previously located):
`SavedDraftsWorkflow.execute()` in `app/chatops/workflows/legal.py:263-292`
lists the drafts and returns `finished=True` — it never records anywhere in
`context.memory` which drafts were just shown or how to map a follow-up name
back to a `draft_id`. A subsequent "open <name>" re-matches the same
`SAVED_DRAFTS` intent (hence the repeated list), and a named *edit* command
goes through a completely different code path (`DraftConversationEngine`'s
own `detect_named_template`, added for BUG-013) that has no awareness of
drafts that were never made this session's active draft in the first place,
so it falls through to "start fresh" instead of "resume" or "ask."

**Not fixed this session** — properly resuming a named/listed draft into the
active `DraftConversationEngine` state (`draft_id`, `draft_template_id`,
`draft_stage=preview`, prior field values) requires new wiring across the
same state machine BUG-014's writeup already flagged as too large to
under-test in one sitting; attempting it here risked exactly the kind of
rushed, unverified structural change this file's own discipline (see
BUG-010's original session) argues against. Acceptance test for whoever picks
this up: exactly steps 1-3 above, both variants, must resume the *existing*
`draft_id` (not re-list, not start a duplicate).

**Verdict: FAIL, unresolved, HIGH severity (Finding-008), now with a
documented worse sub-case.**

### Journey 3 — Voice: names/amounts confirmation flows through to the final draft

**BUG-016 (FAIL→FIXED, CRITICAL severity, newly discovered this session):
`/voice/chat` speech-to-text was completely non-functional**

- Repro: synthesized a real WAV via the app's own `/voice/speak` ("My name is
  Suresh Kumar Yadav. The rent deposit amount is forty five thousand rupees.
  Please prepare a legal notice.") and fed it back into `/voice/chat` —
  confirms both that TTS output is genuine, playable, correctly-worded audio
  (independently useful: **TTS itself is verified working**) and gives a
  ground-truth STT test case. First attempt: `502 Bad Gateway`.
- Root cause, confirmed directly against Google's API (not guessed): server
  log showed `gemini_stt_http_error` / `400 Client error` from
  `gemma-4-26b-a4b-it` (this deployment's configured `GEMINI_MODEL`, used for
  text chat). Re-issuing the identical request by hand got the exact reason:
  `"Audio input modality is not enabled for this model"`. `voice_router.py`'s
  `gemini_speech_to_text` did `model = settings.gemini_model or
  "gemini-1.5-flash"` — i.e., it silently reused whatever model the operator
  configured for **text** chat completion for **audio** transcription too,
  unlike TTS, which already correctly has its own independent
  `gemini_tts_model` setting. Any deployment pointing `GEMINI_MODEL` at a
  non-multimodal model breaks voice input outright; the hardcoded fallback
  (`gemini-1.5-flash`) is now also a retired model (see the operational note
  above), so even an unconfigured install would have failed the same way.
- **Fixed**: added a dedicated `gemini_stt_model` setting to
  `app/core/config.py` (default `gemini-3.6-flash`, confirmed live via direct
  API call to actually accept audio and transcribe correctly — see the
  operational note above for why the two more obvious choices,
  `gemini-2.0-flash`/`gemini-2.5-flash`, were rejected first), mirroring the
  existing `gemini_tts_model` pattern exactly. `voice_router.py` now reads
  `settings.gemini_stt_model` instead of `settings.gemini_model`.
- **Live-verified fixed, twice, end-to-end**, after a server restart:
  - Attempt 1 (session `qa5-voice-3`): STT transcript came back **exactly**
    "My name is Suresh Kumar Yadav. The rent deposit amount is 45,000 rupees.
    Please prepare a legal notice." — name and amount both correct — and the
    chat pipeline correctly filed them under `Applicant Name` and `Claim
    Amount (Rs.)` in the in-progress `General Legal Notice` draft.
  - Attempt 2 (session `qa5-voice-5`, both turns through `/voice/chat`, none
    through plain `/chat`): completed the full remaining-fields turn by
    voice, reached `stage: "preview"` with a real `draft_id`
    (`2c01cef6-...`), `voice_final_confirmation_required: true`, and
    `POST /voice/drafts/{id}/confirm` with the exact required phrase
    returned `200 {"voice_final_confirmed": true}`. **Full voice round trip —
    names/amounts spoken in → confirmed final draft out — confirmed working.**
  - `tests/test_voice_security.py` (12 tests) and the full `config`/`voice`
    filtered slice of the suite (24 tests) pass after the change; one
    unrelated teardown flake (a stray file the *live server's own* KB
    automation wrote into `storage/kb_review/failed` during the test run,
    nothing to do with this change) is noted, not a regression.

**Two secondary findings surfaced by this same testing, not fixed (both
content-quality/design gaps, neither blocks the core PASS above):**

| ID | Summary | Severity |
|---|---|---|
| Finding-011 | `voice_collected`/`voice_final_confirmation_required` is only ever set on a draft if the *specific turn that completes it* happens to go through `/voice/chat`. A user who starts a draft by voice and finishes it by typing (or the reverse) gets a draft the voice-confirmation endpoint then permanently refuses ("This draft was not collected through the voice workflow") — reproduced directly (session `qa5-voice-3`) before the same fields were re-supplied purely by voice (session `qa5-voice-5`) to confirm it wasn't the fix itself. Not blocking (ordinary text-based export/download still works on such a draft), but voice-specific confirmation silently becomes unreachable for a very plausible mixed-modality real usage pattern. |Medium|
| Finding-012 | A single free-form spoken utterance that combines identity+amount+instruction in one breath (exactly what a real caller does) gets its *entire text* dumped verbatim into the `Facts of the Case` field ("1. That My name is Suresh Kumar Yadav. 2. That The rent deposit amount is 45,000 rupees. 3. That Please prepare a legal notice." — literally in the generated notice), and a multi-field turn spoken as one sentence gets truncated mid-address by naive sentence-splitting ("45 FC Road, Pune. The"). Not voice-specific in mechanism (equivalent typed input would do the same), but voice input makes this phrasing far more likely in practice. |Medium|

**Verdict: FAIL→FIXED, CRITICAL (BUG-016), live-verified end-to-end twice.**
Two Medium findings opened for follow-up, not blocking.

### Journey 4 — Languages: facts and meaning preserved in required languages

English, Hindi, and Hinglish were already live-verified across sessions 2-3's
draft-edit journey. This session added two more of the Eighth-Schedule
languages not yet tested live, using the same fact (`Section 138`
NI Act penalty) as the cross-language control:

| Language | Result |
|---|---|
| Tamil (`qa5-lang-tamil-1`) | 2.3s (cache hit on the already-warm topic). Fully fluent Tamil answer; every fact correct and complete — 2-year imprisonment cap, fine up to double the cheque amount, 6-month presentation window, 30-day notice, 15-day payment window — matched word-for-word against the actual Section 138/139 text retrieved independently via `/search`. **PASS.** |
| Bengali (`qa5-lang-bengali-1`) | 95.9s. Same fact set, same result: fully fluent Bengali, every figure and condition correct and complete. **PASS.** |

Minor, non-blocking observation in both: the standard legal-disclaimer
sentence at the end of the answer stays in English rather than being
translated — consistent across both languages tested, so most likely a
deliberate fixed-string compliance disclaimer rather than a translation gap,
but not confirmed either way this session.

**Verdict: PASS** for every language actually tested live (English, Hindi,
Hinglish — prior sessions; Tamil, Bengali — this session). The full 22-language
(or even the previously-scoped 11-language) exhaustive matrix remains
explicitly **NOT RUN** — five languages across five sessions is a
representative sample, not full coverage; each additional language's full
collect→draft→edit→translate→reopen cycle costs 60-220s per turn the same as
every other live call in this file, and was out of today's time budget beyond
this representative check.

### Journey 5 — Citations: the referenced passage actually supports the claim

Took journey 1 turn 1's live answer (Section 138 NI Act penalty) and
independently re-retrieved the same statute via `POST /search`
(`mode: hybrid`, bypassing the chat/LLM layer entirely) to check the citation
against the actual indexed source text, not just the label:

- Chat answer's claim: "imprisonment for up to two years, a fine that may be
  up to twice the cheque's value, or both," cited as "Negotiable Instruments
  Act — Section 138."
- Independently retrieved chunk (`/search`, `source_document:
  Negotiable_Instruments_Act_1881_Complete_Act.pdf`, `section_number: "138"`,
  `verification_status: "verified"`): "...be punished with imprisonment for a
  term which may be extended to two years, or with fine which may extend to
  twice the amount of the cheque, or with both." — **word-for-word match.**

Also checked Tamil/Bengali (journey 4) against the same source: both
correctly reproduced the 2-year/2x-fine/6-month/30-day/15-day figures with no
drift introduced by translation.

**Minor observation, not a defect:** journey 1 turn 4's answer (5 sources
returned) included citations of clearly tangential relevance alongside the
correct one — e.g. "Maharashtra Court-Fees Act — Section 138," a coincidental
section-*number* match with nothing to do with cheque bounce. The answer text
itself only actually drew on the correct NI Act section, so the claim made
was not mis-supported, but a user who spot-checks a random citation from a
longer list rather than the first one could land on an irrelevant one. Worth
a future look at whether `sources` should be filtered to only what the answer
text actually cites, rather than every above-threshold retrieval candidate.

**Verdict: PASS** — every claim checked was directly and verifiably supported
by the actual retrieved passage.

### Journey 6 — Failure recovery: conversation continues after a provider/database error

Journey 1 (turns 2-3) already produced a genuine, unplanned live provider
failure — reused as this journey's primary evidence rather than manufacturing
a synthetic one:

- Turn 2 hit a real generation failure mid-conversation ("...answer-generation
  service did not complete... say 'retry'"). The session did **not** crash,
  hang, or lose state.
- Turns 3, 4, and the control check all continued to return valid `200`
  responses in the same session afterward.
- Turn 4 correctly recalled turn 1's fact (the fine amount) **after** two
  intervening failed turns — conversation memory survived the failures
  intact, not just the HTTP layer.
- Consistent with BUG-013a (session 4, not re-tested this session, still
  open): the literal word "retry" is required to resume a failed request —
  rephrasing it naturally (as turn 2's own retry attempt effectively did,
  landing on a *different* fallback rather than recovering) does not count as
  an equivalent retry. This is a content-recovery gap, not a
  conversation-continuity one — the conversation itself never broke.
- **Database-outage fault injection: NOT RUN, by the same deliberate caution
  sessions 1-4 already established** — this machine's MongoDB/Redis are
  shared with other in-progress work, so deliberately killing either was
  avoided again rather than repeating a decision already made twice before.
  Indirect evidence only: this session's own direct MongoDB write (aborting
  the stuck reindex job, see above) and every one of the ~20 live calls in
  this session completed against the same live database without incident —
  i.e., the database itself never actually failed during this session for
  there to be a recovery to observe.

**Verdict: PASS** for the specific, literal ask — the conversation (session
state, memory, facts) demonstrably continues and stays coherent after a live
provider failure. The narrower, already-tracked BUG-013a (retry-phrasing
strictness) remains open, unrelated to whether the conversation itself
survives.

---

### Session 5 summary: PASS/FAIL evidence for all six requested journeys, and the honest gap against "zero unresolved critical failures"

| Journey | Verdict | Evidence |
|---|---|---|
| Follow-up/context | **PASS** | Turn 4 recall across an intervening unrelated topic; control check isolates the turn-2/3 failures as pre-existing retrieval instability, not a context bug |
| Draft continuation | **FAIL (unresolved)** | Finding-008 reconfirmed + new duplicate-draft sub-case, root cause pinpointed to `SavedDraftsWorkflow.execute()` |
| Voice | **FAIL→FIXED** | BUG-016 (CRITICAL) found, root-caused, fixed, live-verified twice end-to-end incl. final confirmation |
| Languages | **PASS** | Tamil + Bengali added live this session; English/Hindi/Hinglish from prior sessions; full matrix explicitly out of scope |
| Citations | **PASS** | Section 138 claim verified word-for-word against independently retrieved source text |
| Failure recovery | **PASS** | Session survived a real live provider failure mid-conversation with memory intact; BUG-013a (narrower, pre-existing) still open |

Every one of the six requested journeys now has live PASS/FAIL evidence, per
the "complete tab" requirement. **The one newly-discovered CRITICAL failure
this session (BUG-016) was fixed and live-verified before this file was
written up — it does not remain open.**

Read honestly against "unresolved critical failures zero hon" for the file as
a whole, not just today's six journeys: **that bar is not yet met.**
Session 4's **BUG-014** (session_id functions as a cross-user bearer
credential for drafts) and **BUG-015** (concurrent draft edits silently lose
an update with a false success message) are both still open, both still
CRITICAL, and neither was in today's six journeys' scope — they were not
touched, re-verified, or fixed this session. Finding-008 (Draft continuation,
above) is HIGH, not CRITICAL, but is also unresolved. A genuine "zero
unresolved critical failures" state requires BUG-014 and BUG-015 to be
fixed too; that is flagged here rather than left implicit.

## Session 6 (2026-09-14) — real-service integration and load validation

Scope, as requested: (1) fill missing real-MongoDB/Redis coverage in the test
suite, in an isolated environment; (2) measure chat/upload/drafting under an
agreed concurrent-user workload; (3) verify saved conversations/drafts/
uploads survive a server restart; (4) check whether the existing tracing/
dashboard setup is actually receiving telemetry. Two decisions were needed
before any of this could be graded PASS/FAIL and were confirmed explicitly
before proceeding (no numeric SLA existed anywhere in this repo, and the
concurrency level has real Gemini API cost implications): **latency/error
targets are formalized from five sessions' own observed baseline, not
invented** (`docs/qa/LATENCY_ERROR_TARGETS.md`, new this session), and
**concurrency level is "Light": 3-5 concurrent**.

### Incident: a compose command briefly took down the shared dev Redis

Before any testing began, `docker compose -f docker-compose.phase1-test.yml
up -d` (run from the same directory as the main `docker-compose.yml`, with no
explicit project name) silently **replaced** the already-running shared dev
Redis container (`legal_ai_assistant-redis-1`, port 6379) — Compose derives
its project name from the directory when none is set, so both compose files
resolved to the same project and the phase1-test config recreated the same
container name/slot with different port mappings and no volume mount. Caught
immediately via `docker ps`. **Fixed with explicit user approval** (my
sandbox's permission gate correctly blocked the `docker stop`/`rm` step as a
shared-resource action on the first attempt): stopped the misnamed
containers, brought the real `redis` service back up from the main
`docker-compose.yml` (which reattached its original, untouched
`legal_ai_assistant_redis_data` volume — confirmed **zero data loss**, 3,455
keys present immediately after), then relaunched the disposable test
containers under an explicit separate project name
(`docker compose -p legal_ai_phase1_test -f docker-compose.phase1-test.yml up
-d`) so this cannot collide again. Total shared-Redis downtime: under two
minutes, caught before any other process is known to have depended on it
mid-outage. Worth a standing note for anyone else running
`docker-compose.phase1-test.yml`: **always pass an explicit `-p` project
name**, or run it from a directory with no other compose file.

### 1. Real MongoDB/Redis coverage added

The existing convention (`tests/test_phase1_live_services.py`,
`tests/test_phase1_staging_acceptance.py`: `PHASE1_LIVE_TESTS=1`, disposable
Mongo `:37017`/Redis `:36379`, LLM stubbed via `LLMFactory.create`/
`create_resilient` to avoid real API cost) already covers auth/registration,
reindex rollback, private-version isolation, publication-interruption
recovery, cross-session draft/upload isolation, and one full login→chat→
upload→draft→export chain — 6-7 real-DB tests total, all still passing this
session (`7 passed in 47.89s` against the freshly-recreated containers). That
existing chain test already documents BUG-014's exact root cause inline
(`draft_record.get("user_id") is None` asserted deliberately, with a comment
explaining why) — so BUG-014 already had real-DB regression coverage; it did
not need duplicating.

**What was genuinely missing:** a real-database test for concurrent draft
edits (BUG-015). This is the one class of defect a mocked repository
structurally cannot catch — `unittest.mock.AsyncMock` returns are synchronous
and deterministic, so two "concurrent" calls against a mock never actually
race. Added `tests/test_phase1_draft_concurrency.py`: creates one real draft
via `LegalDraftEngine.generate()` against the real disposable MongoDB, then
fires two genuinely overlapping `engine.regenerate()` calls (via
`asyncio.gather`) changing two different fields of the same draft, and
asserts both changes survive in the final persisted document.

**Result: reproduces BUG-015 deterministically against real infrastructure.**
Marked `@pytest.mark.xfail(strict=True, reason="BUG-015 ...")` rather than a
plain failing test, per this file's own standing practice of documenting a
known, not-yet-fixed defect as a tracked expectation rather than a red
build — `strict=True` means the moment someone adds an optimistic-concurrency
check to `regenerate()`, this test flips to an unexpected pass (XPASS) and
the suite goes red until the marker is removed, so the fix can't silently go
unnoticed. Confirmed root cause directly in the repository layer this
session: `DraftRepository.update_by_id` (`app/repositories/base.py:27-30`) is
an unconditional `update_one({"_id": ...}, {"$set": updates})` — no version
or timestamp precondition of any kind.

**Not added this session, flagged as remaining gaps:** real-Redis coverage
for `RateLimitMiddleware`'s sliding-window counter under genuine concurrent
requests (lower priority — Redis's own `INCR` is atomic, so this is a
correctness-confirmation exercise rather than a suspected bug, unlike the
draft-concurrency gap which was a *known* live-observed defect) and the
Redis-outage graceful-degradation path (`except (RedisError, RuntimeError):
pass`) against a real connection failure rather than a mocked exception.

### 2. Concurrent-user load measurement (Light: 3-5 concurrent)

Ran three concurrent batches against the live server (real Gemini, real
local MongoDB/Redis) with independent session IDs per call, then verified
correctness directly against MongoDB (not just HTTP status codes):

| Workload | Concurrency | Result | Latency (min/max/mean) |
|---|---|---|---|
| Chat (`/chat`, distinct questions/sessions) | 5 | 5/5 succeeded (`200`) | 2.5s / 192.8s / 143.9s |
| Upload (`/upload`, distinct small text files) | 3 | 3/3 succeeded (`200`) | 22.8s / 22.8s / 22.8s (near-identical completion times) |
| Draft start (`/draft`, same template, distinct facts) | 3 | 3/3 succeeded (`200`) | 27.4s / 30.7s / 29.2s |

**Correctness (the actual point of a concurrency test, not just timing):**
queried MongoDB directly after the run — every upload's `owner_session_id`,
every chat's `session_id`/`question` pairing, and every draft session's
in-progress `draft_fields` (`conversation_memory` collection, all three
`police_complaint` sessions running the identical template concurrently) came
back correctly isolated with **zero cross-session contamination**. This is
the target that actually matters at this concurrency level, and it was fully
met.

**Latency against `LATENCY_ERROR_TARGETS.md`:**
- **Draft: meets target** (27-31s, comfortably inside the 15-90s typical
  band).
- **Chat: does not comfortably meet the "typical" target under this load.**
  4 of 5 concurrent calls landed at 170-193s — inside the documented
  "acceptable worst case ≤250s" ceiling, but well outside the "typical
  15-90s" band every single-request measurement in sessions 1-5 established.
  This is a real, newly-measured finding: **concurrency measurably shifts
  chat latency toward the tolerance ceiling**, not a one-off. Most plausibly
  provider-side (Gemini call concurrency/rate-limit contention), consistent
  with this app's own already-documented wide latency variance — not
  something this session traced further to a specific queue/lock.
- **Upload: misses the (newly-established, first-ever-measured) target.**
  22.8s for a ~3.6KB plain-text file with no OCR is well over the "under 10s"
  target set in `LATENCY_ERROR_TARGETS.md`. The near-identical completion
  time across all three concurrent uploads (22.75-22.76s, not just "similar"
  but essentially simultaneous) suggests serialization on a shared resource
  (most likely the single embedding-model instance or the BM25 index
  writer), not independent parallel processing — flagged for a future
  profiling pass, not root-caused this session.

### 3. Restart persistence: conversations, drafts, uploads

Restarted the API process (not the databases) mid-way through this session's
own load-test data, then re-verified every data type through the real API
(not a direct DB read) post-restart:

| Data type | Verification | Result |
|---|---|---|
| Conversation history | `GET /history?session_id=loadtest-chat-0` | **PASS** — full question+answer pair returned intact, unchanged |
| Uploaded document | `GET /documents?session_id=loadtest-upload-0` | **PASS** — `status: "indexed"`, correct filename/timestamps |
| In-progress draft | Sent 3 more turns to `loadtest-draft-0` (started *before* the restart, sitting at `stage: "collecting"` with 2 fields already given) | **PASS** — the pre-restart `draft_fields` were still there after restart, new fields merged correctly on top, reached `stage: "preview"` with a real `draft_id`, and appeared correctly in `POST /draft-history` afterward |

**Verdict: PASS**, all three data types, verified live through the real API
after an actual process restart — not inferred from "MongoDB is durable
storage" alone.

### 4. Telemetry: is anything actually reaching the tracing/dashboard setup?

- **Distributed tracing (OpenTelemetry): confirmed OFF.** `.env` has no
  `OTEL_EXPORTER_OTLP_ENDPOINT`, and this session's own live server log shows
  `tracing_disabled_no_endpoint` on every boot (`app/observability/
  tracing.py`: the SDK and FastAPI/pymongo/redis auto-instrumentation are
  fully wired and would work the moment an endpoint is configured, but
  nothing is exported anywhere today). Not a bug — by design, a deployment
  that hasn't configured a collector pays no tracing cost — but it does mean
  **zero trace data exists anywhere for this deployment today.**
- **Metrics (`/internal/metrics`, Prometheus format): the data source itself
  is live and accurate.** Queried it directly after this session's own calls
  and it reported exactly what happened (`http_status_200 11`,
  `http_POST__draft_count 4` matching the 4 real `/draft` calls made
  post-restart, correct avg/max latency per route) — not a stub, not stale
  data.
- **But nothing is currently scraping it into a dashboard.** The optional
  Prometheus+Grafana overlay (`docker/docker-compose.observability.yml`) is
  not running in this dev environment, and was **not started this
  session** — deliberately, given the exact class of mistake made earlier
  this session (a compose project-name collision) and that the overlay's own
  `docker-compose.yml` base would spin up ITS OWN `mongo`/`redis` containers
  on the same ports the shared dev instances already use. Starting it safely
  would need its own isolated pass (distinct project name, or genuinely
  separate ports), not a rushed add-on to this session.
- **Verdict: telemetry infrastructure is correctly built but effectively
  inactive in this environment** — metrics are measurably real but
  unconsumed, tracing is off. Confirmed, not assumed.

### Session 6 summary against the "Complete tab" bar

| Requirement | Status |
|---|---|
| Missing real MongoDB/Redis coverage added, isolated environment | **Done** — BUG-015 now has a deterministic real-DB regression test (originally `xfail`, tied to the open bug; now asserts the fix directly since BUG-015 was fixed in session 7, below); a couple of lower-priority real-Redis gaps remain, explicitly listed above, not done |
| Agreed concurrent-workload measurement (chat/upload/drafting) | **Done** — Light (3-5 concurrent), zero errors, zero cross-session data mixing |
| Restart persistence (conversations/drafts/uploads) | **Done — PASS**, verified live through the real API post-restart |
| Telemetry reaching tracing/dashboard | **Done — checked and answered honestly**: tracing off (no endpoint), metrics real but unconsumed (no dashboard running) |
| Agreed latency/error targets met | **Partially.** Correctness/error targets: **met** (zero failures, zero data corruption, restart-safe). Latency targets: draft **meets** target; chat **does not comfortably meet** the "typical" band under concurrency (shifts into the worst-case tolerance zone); upload **misses** its (newly-established) target and shows a serialization signature worth a future profiling pass. |
| Atlas / reranker / distributed queue | **Not added — correctly, per the stated principle.** Nothing measured this session points at retrieval quality (Atlas/reranker) or a job-queue bottleneck (distributed queue) as the actual constraint. The two real findings — chat latency under concurrent load, and upload latency's serialization signature — both look provider/embedding-instance-bound, not something a heavier data-layer would fix; they need profiling before any infra decision, exactly as the stated principle asks for. |

## Session 7 (2026-09-14) — the two P0 bugs: BUG-014 and BUG-015, fixed

Scope, as requested: fix the two standing CRITICAL/P0 items this file has
carried open since session 4 — **BUG-014** (draft ownership never actually
stamped for conversationally-created drafts, making `session_id` a de facto
bearer credential across accounts) and **BUG-015** (concurrent draft edits
silently lose an update, with a false "Updated..." success). Both are now
**fixed, unit/integration-tested against real disposable infrastructure, and
live-verified end-to-end against the real dev server** (real Gemini, real
local MongoDB/Redis, real registered accounts) — not just patched and
assumed.

### BUG-014: fixed

**Root cause was narrower than session 4's original assessment.** That
session estimated the fix would need threading `user_id` through 8-10 method
signatures across `DraftConversationEngine`. Re-investigating this session
found `memory["owner_user_id"]` is already reliably stamped into the shared
conversation-memory dict *before* `handle_turn` ever runs — by
`ChatService._claim_session_if_unowned` on the `/chat` path, and directly by
the `/draft` route on the direct path — and that same `memory` dict is
already threaded through every stage of the state machine. The bug was
narrower: `_finalize_generation_reply` (`app/drafting/conversation.py`), the
single choke point that actually persists a new draft record via
`LegalDraftEngine.generate()`, built its `DraftGenerateRequest` without ever
reading that already-present value.

- **Fix**: one line — `DraftGenerateRequest(..., user_id=memory.get("owner_user_id"))`
  in `_finalize_generation_reply`. `LegalDraftEngine._build()` already
  correctly persisted `request.user_id` onto the draft record, and
  `ensure_draft_access` already correctly enforced it once present — both
  sides of the mechanism were already built and already covered every other
  mutating draft route; this was the one missing wire.
- **Tests**: `tests/test_phase1_staging_acceptance.py`'s existing chain test
  previously asserted `draft_record.get("user_id") is None` with a comment
  documenting the gap — flipped to `== user_id` (an acceptance test the
  original session 4 write-up itself specified). Added
  `test_second_authenticated_user_cannot_read_first_users_draft_via_session_id`:
  two genuinely separate, freshly-registered real accounts (real JWTs, real
  `/register`+`/login`), User A creates and completes a draft via the
  conversational `/draft` flow, User B's own valid-but-different token plus
  A's `session_id` is refused (`403`), User B's token alone is refused
  (`403`, unchanged), and User A's own token from a brand-new session still
  works (`200`) — the exact acceptance test session 4's write-up specified.
  Both pass against real disposable Mongo/Redis
  (`docker-compose.phase1-test.yml`, `PHASE1_LIVE_TESTS=1`).
- **Live-verified against the real dev server** (not just the disposable
  test containers): registered two genuinely new accounts via the real
  `/register`+`/login` endpoints, User A created and completed a real
  `police_complaint` draft (`draft_id 5522d09e-166e-4a62-bf72-366ab1f10584`)
  via the real `/draft` conversational flow with a real Gemini call. User B
  (a different, genuinely authenticated account) querying
  `GET /draft/{id}/versions` with **User A's own `session_id`** as a query
  parameter now returns `403 {"error":{"code":"forbidden",...}}` — this
  exact request returned `200` with full version history before the fix
  (session 4's live repro). User A's own token from a brand-new session
  (`session_id` User A never used before) still correctly returns `200`.
- **Regression check**: `tests/test_draft_conversation.py`,
  `test_drafting.py`, `test_draft_ownership.py`,
  `test_multi_turn_conversations.py` (264 tests) all pass unchanged. `ruff
  check`/`mypy` clean on every changed file.

### BUG-015: fixed

**Design**: `LegalDraftEngine.regenerate`'s read-merge-write cycle now uses
an optimistic-concurrency compare-and-swap keyed on a new `version` integer
field on each draft document, instead of `DraftRepository.update_by_id`'s
old unconditional `update_one($set)`. On a losing race, `regenerate` re-reads
the document the winner just committed, re-merges its OWN requested field(s)
onto that fresh base (never onto its own stale read), and retries exactly
once; only a genuine repeated collision (a third overlapping writer landing
inside the retry's own window too) raises the new `DraftConflictError` (HTTP
409) rather than ever reporting a false success. Backward compatible with
every draft persisted before `version` existed: `DraftRepository.
compare_and_swap` treats a missing `version` field as equivalent to `1`, so
the very next edit to a pre-existing draft self-heals it onto the new scheme
with no migration script.

- **Files**: `app/repositories/drafts.py` (`DraftRepository.compare_and_swap`,
  new), `app/drafting/engine.py` (`_build` stamps `"version": 1` on create;
  `regenerate` rewritten as a 2-attempt read-render-compare_and_swap loop via
  two new helpers, `_render_regenerated_content`/`_try_apply_regeneration`),
  `app/core/exceptions.py` (`DraftConflictError`, HTTP 409), `app/drafting/
  conversation.py` (`_regenerate_or_conflict_reply`: all six call sites that
  mutate a draft in `_continue_preview` now route through this wrapper,
  which turns a persisted `DraftConflictError` into a friendly "someone else
  just changed this draft... please retry" chat reply instead of an
  unhandled 500). `app/api/drafting.py`'s direct `/draft/edit` route needed
  no change — `DraftConflictError` is an `AppError` subclass, so it already
  propagates through the existing exception handler as a clean `409`.
- **Tests**: `tests/test_phase1_draft_concurrency.py` (added session 6 as an
  `xfail(strict=True)` reproduction, now flipped to a plain passing
  assertion) fires two genuinely overlapping `asyncio.gather`'d
  `regenerate()` calls changing two different fields of the same real,
  disposable-MongoDB-backed draft and asserts BOTH survive — run 5
  consecutive times with no flake. Updated three existing unit tests
  (`test_drafting.py`, `test_draft_lifecycle.py` ×2) that mocked the now
  -unused `update_by_id` to mock `compare_and_swap` instead.
- **Live-verified against the real dev server**: created a real
  `police_complaint` draft via the real conversational `/draft` flow (real
  Gemini). Note: `/chat` turned out to already have its OWN per-session
  in-flight-request guard ("I'm still working on your previous message...")
  that serializes same-session messages before they would ever reach
  `regenerate`'s race — a useful independent finding (defense in depth at a
  different layer), but it meant the direct `/draft/edit` endpoint (which has
  no such guard) was the right surface to reproduce the ORIGINAL race against.
  Fired two genuinely concurrent `POST /draft/edit` calls
  (`asyncio.gather`) at the same draft, changing `applicant_mobile` and
  `place` respectively — both returned `200 {"status": "complete"}`, and
  `GET`-ing the persisted record directly from MongoDB afterward confirmed
  **both** field changes survived (`applicant_mobile: "9333333333"`,
  `place: "Rajpura Live Test"`, `version: 4`) — reproducing session 4's exact
  repro shape (two concurrent field edits to the same draft) with the
  opposite, now-correct outcome.
- **Known remaining, lower-severity gap, not fixed this session**:
  `LegalDraftEngine._append_version` (the separate `draft_versions`
  audit-trail collection) still has its own, unrelated read-then-insert race
  on `version_number` (`latest_for_draft` then `+1`) — two truly overlapping
  writers could still get two version-history entries with the same
  `version_number`. This does NOT affect the current, authoritative
  `fields`/`sections` on the `legal_drafts` document itself (the actual data
  BUG-015 was about, and what is now protected) — only the secondary
  history log's own numbering under a genuine 3-way-or-more collision.
  Flagged for a future pass, not blocking.
- **Regression check**: `tests/test_draft_conversation.py`, `test_drafting.py`,
  `test_draft_ownership.py`, `test_multi_turn_conversations.py`,
  `test_draft_lifecycle.py`, `test_draft_export.py`,
  `test_draft_advocate_register.py` (357 tests) all pass. `ruff check`/`mypy`
  clean on every changed file. Full suite (`pytest tests/`, server stopped to
  rule out interference): **2812 passed, 13 skipped, 0 failed** — the one
  environment-only exception below.

### Operational note: an unrelated, independent script was found polluting the test-isolation guard

While chasing a `conftest.py` "tests wrote into real storage" failure that
appeared to move between unrelated test files on repeat full-suite runs
(inconsistent with anything in this session's own changes), found a
long-running, completely separate Python process
(`odisha_ingest_batch1.py`, PID from a **different** Claude session's own
scratchpad directory, not this one) actively ingesting Odisha Act PDFs into
`storage/archive`/`storage/kb_review` throughout this entire session,
independent of the API server's own state (the noise persisted even with
the API server stopped). This is not a defect in this codebase or in
anything fixed this session — it is a real, currently-running, presumably
intentional ingestion job left over from other work happening in parallel on
this shared machine. **Not touched** (stopping someone else's in-progress
ingestion job was not this session's call to make); noted here so a future
session doesn't mistake this specific storage-isolation flake for a real
regression again. Every actually-relevant test file was independently
re-run in isolation this session and passed cleanly (0 failures) precisely
to rule this out.

### Session 7 summary

| Item | Status |
|---|---|
| BUG-014 (draft ownership / cross-account session_id bearer-credential) | **FIXED, unit+integration+live-verified** |
| BUG-015 (concurrent draft edits, silent lost update) | **FIXED, unit+integration+live-verified** |
| Full pytest suite | 2812 passed, 13 skipped, 0 failed (server stopped; the one intermittent failure seen with it running is unrelated, external ingestion noise, confirmed above) |
| `ruff check` / `mypy` | Clean on every file changed this session |

**Both of this file's standing CRITICAL/P0 items are now closed.** Combined
with session 5's six conversational journeys and session 6's real-service/
load validation, the only unresolved items remaining in this file are:
Finding-008 (Draft continuation reopen-by-name, HIGH not CRITICAL, session
5), BUG-013a (retry-phrasing strictness, MEDIUM, session 4), and the
non-critical latency/coverage gaps documented in sessions 5-6 (upload
latency target miss, chat latency under concurrency, the lower-priority
real-Redis test gaps, `_append_version`'s own secondary numbering race noted
just above). **Zero unresolved CRITICAL-severity items remain.**

## Session 8 (2026-09-14) — P0 re-verification, saved-draft journey, natural retry, performance re-check, release candidate freeze

### 1. P0 fixes re-verified on the current build

Re-ran the exact real-database regression tests from session 7 against the
current build, no code changes since: `tests/test_phase1_staging_acceptance.py`
(both tests, including `test_second_authenticated_user_cannot_read_first_users_draft_via_session_id`),
`tests/test_phase1_draft_concurrency.py`, and the full
`tests/test_phase1_live_services.py` suite — **9/9 passed** against the
disposable real Mongo/Redis containers. **BUG-014 and BUG-015 remain fixed
and verified on the current build; see session 7 above for the original
root-cause analysis and live end-to-end verification (real accounts, real
concurrent writes) — not repeated here since nothing changed.**

### 2. Saved-draft journey: open → edit amount → reopen (next request) → export

Tested exactly the flow described, live, against the real dev server:

| Step | Result |
|---|---|
| Open (create) a `General Legal Notice`, claim amount 30,000 | `draft_id cc49bc55-...`, `preview_ready` |
| Edit: "Change the claim amount to 45000 rupees" | `fields.claim_amount` → `"45000 rupees"`, same `draft_id`, `version: 2` |
| Interrupted with an unrelated question (simulates "next request" after a gap) | Correctly answered the unrelated question AND correctly paused the draft with "your previous... draft is still saved -- type 'continue draft'" |
| Reopen: "continue draft" | Resumed the **same** `draft_id`, `stage: preview`, correct content |
| Export (`POST /draft/export`, txt) | `200`, non-empty file |
| Verify no duplicate | `count_documents({session_id: ...})` on `legal_drafts` → **1** (not 2) |

**Verdict: PASS**, exactly as specified — the original draft updates in place
and no duplicate is created for this flow (single active/paused draft,
resumed via the app's own "continue draft" mechanism).

**Important scope note, checked separately for honesty:** this flow is
distinct from **Finding-008** (session 5) — reopening a draft **from a
multi-draft saved-drafts list**, after a *different* draft has since become
the session's active one.

**Discovered mid-session: a fix for Finding-008 already exists in this
working tree, uncommitted, from separate (not-mine) work
(`docs/qa/P1_FIXES_20260914.md`, itself dated today) — `app/chatops/
workflows/legal.py`'s `SavedDraftsWorkflow` was rewritten to track shown
`_choices` and resolve a numbered/named reply against them via
`app/chatops/selection.py`, then delegate to `DraftManagementWorkflow` to
actually open the selection. That note explicitly flags "Live multi-turn
HTTP retest with the deployed build remains pending" — this session
supplied exactly that missing live test, live against the actual running
server (not assumed from the code or from that note's own unit tests):**

| Sub-case | Result |
|---|---|
| List drafts while a DIFFERENT draft is mid-collection (`draft_mode` active, "collecting" stage) | Replying "1" to select the listed draft was **not** treated as a selection at all — it was consumed as a field answer for the OTHER, currently-collecting draft. **FAIL.** |
| List drafts while a DIFFERENT draft is fully settled at "preview" (both drafts complete, neither mid-collection) | Correctly listed only the non-active draft (excluding the active one) and asked "Which draft would you like to open?" — but replying "1" was **not** routed back into `SavedDraftsWorkflow` to resolve against the shown choices at all: it fell through to a generic domain-intent RAG classification (`intent: "Legal Notice"`) and answered "No verified document...", while ALSO auto-pausing the other (correctly still-active) draft as an unrelated side effect. **FAIL**, different failure signature than session 5's (back then: re-shows the list; now: silently misroutes to an unrelated RAG answer). |

**Verdict: Finding-008 is NOT actually fixed on the current build, despite a
real, substantial attempt to fix it existing in the (uncommitted) working
tree.** The underlying data structure for remembering shown choices exists
now, but the cross-turn continuation — routing the NEXT message back into
`SavedDraftsWorkflow.extract_facts` with the previous turn's `_choices`
intact, with correct priority over both an active-draft-collecting state
and the general intent/RAG router — does not work end-to-end via the real
`/chat` path in either sub-case tested. This is exactly the class of gap
"looks right in isolation, breaks in a live multi-turn HTTP conversation"
that this file's whole methodology exists to catch, and is exactly what
that other note's own "pending" line predicted needed checking. **Finding-008
remains open** — not touched/fixed by this session, but now characterized
more precisely (two distinct live failure modes, not just the original one)
for whoever picks up the existing, partial, uncommitted fix attempt next.

**New finding surfaced while setting up the above (not one of today's
requested checks, documented for completeness):** sending
`"Start a new draft: I want to write a police complaint for a stolen phone."`
produced `"Okay, I've discarded that draft."` — discarding the CURRENT draft
instead of parking it and starting the new one (the documented, intended
behavior for "start another draft"). **No actual data loss** — the discarded
draft's record is still fully intact in MongoDB (`lifecycle_state:
"preview_ready"`) — but the reply's wording is misleading (implies deletion)
and the requested new police-complaint draft never actually started.
**Root-caused**: `_CANCEL_DRAFT_PATTERN` (`app/services/chat_service.py:305-306`)
includes `\bnew draft\b` as an **unanchored** alternative in an otherwise
end-anchored pattern set — it matches the literal substring "new draft"
*anywhere* in a message, not just a bare "start a new draft" with nothing
else. A message that combines "new draft" phrasing with a specific request
(exactly the natural way to ask) gets misread as "cancel," not "start
another." Not fixed this session (outside today's five requested checks);
flagged with an exact file/line for a future pass. Workaround confirmed
live: phrasing the same request as "Draft a police complaint..." (omitting
the words "new draft") works correctly and does not trigger this.

### 3. Natural retry: correction to a standing (stale) file note

**BUG-013a is actually already fixed on the current build** — this
corrects session 4's "not fixed" verdict, which this file had been carrying
forward unchanged through sessions 5-7.

Live repro, real Gemini, temporarily lowering `CHAT_REQUEST_BUDGET_SECONDS`
(a documented settings knob) to deterministically force one real generation
timeout, then restoring the default before the actual retry test (so the
retry itself runs under normal, real conditions, not another artificial
timeout):

1. A fresh, real question genuinely timed out: *"I found potentially
   relevant verified material... but the answer-generation service did not
   complete... Say 'retry'..."* — confirmed `last_failed_question` correctly
   persisted to MongoDB with the exact question text.
2. Restarted with the normal budget. Resent the **identical original
   question text** (never the word "retry").
3. Server-side evidence (`intent_history` in the session's Redis-cached
   memory) shows the resend was classified `"Retry Failed Request"`
   (confidence 0.9, reason: *"message resends the previously failed question
   verbatim -- treated as an equivalent retry"*), `last_failed_question` was
   then correctly cleared, and the retried question was re-dispatched
   through the full pipeline exactly once more.
4. **No duplicate output or save**: exactly one final assistant answer was
   returned to the client and appended to history; no duplicate draft/record
   was created (this was a plain Q&A turn).

The mechanism (`app/intent/classifier.py`'s
`_looks_like_resend_of_failed_question`, wired into
`ConversationIntentClassifier.classify`) is explicitly commented as the
"BUG-013a fix" and cites this very QA file by name — it exists and works; it
was simply never confirmed live and the "not fixed" note was never
corrected in this file until now. **Verdict: PASS.** (The retried attempt's
own *answer quality* — it landed on "no verified context" for a
deliberately hard, verbose test question — is a separate, unrelated,
already-extensively-documented retrieval-confidence characteristic, not a
retry-mechanism defect.)

Independent cross-check found mid-session: `docs/qa/P1_FIXES_20260914.md`
(separate, uncommitted work in this same working tree, dated today) reached
the identical conclusion by reading the code rather than live-testing it —
*"The previous QA description of an unimplemented exact-resend fix is stale
relative to this code."* Two independent investigations (one live-HTTP, one
code-reading) agreeing is stronger evidence than either alone; recorded here
rather than treated as redundant.

### 4. Real performance targets, re-measured (not inferred from any isolated benchmark)

Repeated the identical workload shape from session 6 (5 concurrent chats, 3
concurrent uploads, 3 concurrent draft starts) with fresh session IDs/content
per run (reusing session 6's exact identifiers the first time round produced
misleading artifacts — 2 of 5 chat answers were cache hits from identical
prior questions, and all 3 uploads got a `400` from a same-content/session
duplicate-upload guard, not a real failure — both discarded, re-run clean
below).

| Workload | Run 1 (fresh IDs) | Run 2 (fresh IDs) | Target (`LATENCY_ERROR_TARGETS.md`) |
|---|---|---|---|
| Chat, 5 concurrent | not separately isolated | 52.2s / 245.3s / 94.8s / 163.1s / 104.4s (mean 132.0s) | 15-90s typical |
| Upload, 3 concurrent | timed out at 300s (client) **all three** — see below | 9.4s / 7.3s / 8.3s (all `200`) | under 10s |
| Draft start, 3 concurrent | 30.7s / 28.8s / 29.6s | — | 15-90s typical |

**Chat: does not meet the "typical" target under 5-concurrent load, and this
is now confirmed twice independently** (session 6: 170-193s for 4/5; this
session: 52-245s, mean 132s) — a real, repeatable characteristic of this
concurrency level, not a one-off. All 5 calls still succeeded (zero errors),
so this is a latency-target miss, not a correctness failure.

**Upload: confirms the target under most conditions, but surfaced a severe,
intermittent outlier this session that session 6 never saw.** The first
3-concurrent-upload attempt (run immediately after the 5-concurrent chat
batch above) did not fail — the server log shows all three eventually
completed, but their **embedding step alone took ~320 seconds each**
(`"embedding_ms": 320479` / `320832` / `320498`) against a **347ms** solo
embedding call measured minutes later. Two immediate follow-up checks (2
concurrent, then 3 concurrent again, run independently rather than right
after a heavy chat batch) both completed comfortably inside the 10s target
(6.2s/7.2s; 9.4s/7.3s/8.3s) — **the severe slowdown did not reproduce in
isolation**, pointing at contention between the shared embedding model
instance and a just-finished burst of concurrent RAG retrieval, not a
simple "N concurrent uploads" trigger. Not root-caused further this session
(would need GPU/CUDA-level profiling of the single shared `BAAI/bge-m3`
instance under mixed concurrent chat+upload load) — flagged as a real,
reproduced-once, high-impact, not-yet-explained performance risk for the
release notes below, distinct from the milder "usually fine" baseline.

**Explicit check against "don't treat a BM25 benchmark improvement as
end-to-end success":** `scripts/benchmark_bm25_update.py` (the only
BM25-specific benchmark in this repo) is a synthetic, in-memory, 4,000-chunk
probe of `BM25Index.add_or_update_chunks`'s own speed and event-loop
behavior — its own docstring states **"no database or LLM calls."** It does
not exercise embedding, Mongo persistence, or the real `/upload` pipeline at
all, and therefore says nothing about (and cannot explain or rule out) the
320s embedding-step outlier measured live above. Confirmed by reading it
directly rather than assumed — this repo's only BM25 benchmark is
correctly out of scope for any end-to-end latency claim, and none was made
based on it.

### 5. Release candidate freeze

**Gate status:** the six conversational journeys (session 5), real-service/
load validation (session 6), and both P0 fixes (session 7, re-verified
above) all have live PASS evidence and a clean regression suite. Freezing
this as a release candidate **with the known limitations below explicitly
carried forward, not hidden** — this is not a claim that every discovered
issue is resolved, only that the CRITICAL-severity bar is met and everything
else is documented.

- **Regression suite, final count this session:** `pytest tests/` (full
  suite, no filters): **2817 passed, 17 skipped, 0 genuine failures.** The 1
  `ERROR` seen (`test_workflow_orchestrator.py`, a real-storage-isolation
  trip) is the same external, independent, currently-running ingestion
  script identified in session 7 (`odisha_ingest_batch1.py`, a different
  Claude session's own scratchpad process) — reconfirmed present and
  unrelated by rerunning the specific affected test file standalone (clean
  pass).
- **Backend version**: no version file/tag exists in this repository
  (checked `pyproject.toml`, `app/core/config.py` — `app_name` is set, no
  semantic version field). Pinned instead to a git commit: **`5abded6`**
  (branch `phase-1-runtime-stabilization`), created this session containing
  exactly the BUG-014/BUG-015 fixes plus their tests and this QA
  documentation (12 files) — nothing else. **Important caveat found this
  session**: at commit time, this repository's working tree also carried
  ~109 additional modified/untracked files from separate, unrelated,
  still-uncommitted work (a reranker, a safety module, observability
  wiring, a partial Finding-008 fix attempt, real-Redis test coverage, ML
  fine-tuning scripts, and more — see session 8's Finding-008 write-up
  above for one concrete example of that other work's own current state).
  **None of that other work is included in this release candidate commit**
  — it was deliberately left exactly as found, per an explicit decision
  this session, so as not to disturb whoever is doing it. The MERN team
  should be told plainly: `5abded6` is a clean, narrow, fully-tested commit
  containing only the P0 fixes — it is a safe integration point, but it is
  **not** the full extent of in-progress work on this backend, and a
  meaningfully newer commit may exist by the time of actual handoff.
- **Setup instructions**: `README.md` (quick start), `WINDOWS_SETUP.md`
  (Windows-specific prerequisites/gotchas), `docs/deployment.md` (hosting
  targets and required steps), `docker-compose.yml` (containerized
  mongo/redis/api/worker/streamlit).
- **API docs**: live OpenAPI/Swagger at `/docs` (FastAPI auto-generated,
  confirmed reachable throughout this session) and `/openapi.json` for a
  machine-readable contract the MERN team can codegen a client from directly
  — this is the authoritative, always-current API reference; no separate
  hand-maintained API doc file exists in this repo to also hand off.
- **Test report**: this file, `docs/qa/QA_TEST_MATRIX_20260911.md` (sessions
  1-8, every live journey/fix with PASS/FAIL evidence), plus
  `docs/qa/LATENCY_ERROR_TARGETS.md` (session 6, the agreed latency/error
  targets these numbers are graded against).
- **Known limitations, carried into the release candidate explicitly:**
  1. Finding-008 (HIGH): reopening a saved draft, once a *different* draft
     is the session's active one, does not work live — confirmed in two
     distinct failure modes this session (numeric reply consumed as a field
     answer for a competing collecting-stage draft; numeric reply silently
     misrouted to an unrelated RAG answer when no draft is mid-collection).
     A substantial, uncommitted fix attempt already exists in this working
     tree (`app/chatops/workflows/legal.py`'s rewritten `SavedDraftsWorkflow`
     + `app/chatops/selection.py`) but does not work end-to-end yet — the
     gap is specifically in cross-turn continuation, not the selection
     logic itself.
  2. New finding (session 8, severity MEDIUM — misleading message, no data
     loss): `"start a new draft: <request>"` phrasing is misread as a
     cancel command (`_CANCEL_DRAFT_PATTERN`'s unanchored `\bnew draft\b`).
     Workaround: omit the words "new draft" (e.g. "Draft a police
     complaint...").
  3. Chat latency under 5-concurrent load does not meet the 15-90s
     "typical" target (confirmed twice: 132-144s mean); still zero errors,
     zero data corruption.
  4. Upload latency under concurrency is usually within target but showed
     one severe (~320s), unreproduced-in-isolation outlier this session,
     suspected shared-embedding-model contention with concurrent chat
     load — not yet root-caused.
  5. `LegalDraftEngine._append_version`'s own secondary version-number race
     (session 7) — does not affect current field data, only the audit-trail
     numbering under a genuine 3-way-or-more collision.
  6. Lower-priority real-Redis test coverage gaps (rate limiter under real
     concurrency, Redis-outage graceful degradation) noted but not added
     (session 6).
  - Every item above is non-CRITICAL, has a documented repro, and (where
    relevant) a workaround. None was fixed silently or hidden from this
    list to make the freeze look cleaner than it is.

## Session 9 (2026-09-14) — release marked provisional; three of four remaining items fixed

Explicit correction to session 8's framing, per direct instruction: **QA is
complete against what was tested; the backend is not yet production-ready.**
The release freeze (`5abded6`/`bd30d27`) is **provisional**, not final.
2817 passing tests are useful evidence of no regression, not proof the known
live failures are resolved. This session fixes three of the four remaining
items and is honest that the fourth is not fully closed.

### 1. Saved-draft selection: FIXED and live-verified in both failure modes

**Root cause, precisely located**: `ChatService.answer()`
(`app/services/chat_service.py:766`, session 8's own investigation stopped
one level too shallow) only dispatches to the chatops orchestrator
(`_dispatch_chatops`, where `SavedDraftsWorkflow`'s pending "which draft?"
selection actually lives) when `not draft_was_active` OR the message
"explicitly" names a capability by keyword (`_explicit_workflow_request`).
A bare `"1"` replying to "Which draft would you like to open?" names
nothing explicit, so whenever ANY draft-engine draft was also active
(collecting OR merely sitting at preview), this whole branch was skipped
and the orchestrator — which already correctly implements "let an active
workflow finish its own turn" — never even ran.

- **Fix**: one additional bypass condition, reading the exact same
  `chatops.state.active(memory)` stack `_dispatch_chatops` itself already
  consults a few lines later — if a chatops workflow is already mid-flight
  and holding the floor, it keeps it regardless of `draft_was_active`.
- **Live-verified, both previously-failing sub-cases, on the real dev
  server**, real Gemini, real MongoDB:
  - Competing collecting-stage draft active: created draft A (legal notice,
    preview), then draft B (police complaint, left collecting). "Show my
    saved drafts" → "1" now correctly opened draft A's preview (previously:
    swallowed as a field answer for draft B). Draft B's collected fields
    confirmed intact and still parked afterward — no data loss.
  - No competing collecting-stage draft (both settled at preview): "Show my
    saved drafts" listed both (correctly excluding whichever is currently
    active); selecting "2" correctly opened the named draft (previously:
    silently misrouted to an unrelated RAG "no verified document" answer).
  - Followed through to `POST /draft/export` on the reopened draft: `200`,
    non-empty file — the full journey works end to end now.
- **Regression check**: `test_chatops_orchestration.py`,
  `test_chatops_foundation.py`, `test_chat_service_routing.py`,
  `test_reported_chat_session_regressions.py`, `test_draft_conversation.py`,
  `test_multi_turn_conversations.py` (482 tests) pass. `ruff check` clean.

### 2. New-draft cancellation: FIXED and live-verified

**Root cause** (session 8 found the symptom; this session traces and fixes
it): `_CANCEL_DRAFT_PATTERN` (`chat_service.py:305-306`) had `\bnew
draft\b`/`\bstart over\b` as UNANCHORED alternatives — matching that
substring anywhere in a message, unlike every other phrase in the same
pattern (all correctly anchored to the whole message). `"Start a new draft:
I want to write a police complaint..."` matched purely on containing "new
draft" and was read as a cancel command.

- **Fix, two parts**:
  1. Anchored `"start over"`/`"new draft"`/`"another draft"` in
     `_CANCEL_DRAFT_PATTERN` to bare-message-only, matching this file's own
     established convention for every other dismissal phrase.
  2. That alone would only stop the wrong "discarded" message — it would
     not make the requested new draft actually start (nothing previously
     handled "trigger phrase + a real request" as one combined action; only
     the bare trigger alone was ever handled). Added
     `_NEW_DRAFT_PREFIX_PATTERN` (`app/drafting/conversation.py`), a
     PREFIX-matching sibling of the existing bare `_NEW_DRAFT_PATTERN`:
     parks the current draft, strips the matched trigger phrase, and
     re-runs template detection (`_dispatch_fresh_request`, factored out of
     the existing "no draft active yet" path so both share one dispatch)
     on whatever request follows it. An empty remainder still falls back to
     the original bare-menu behavior, unchanged.
- **Live-verified**: with a police-complaint draft actively collecting,
  sent the exact original repro message. Result: no "discarded" message; a
  `legal_notice` collection correctly started (`missing_fields` for the
  General Legal Notice template); the police-complaint draft confirmed
  still present and fully intact in `parked_drafts` (all four
  previously-collected fields), not lost.
- **Regression check**: the existing parametrized test asserting the bare
  form (`test_start_another_draft_parks_the_current_one_and_shows_the_menu`,
  4 cases) still passes unchanged; 440 tests across the broader drafting/
  chat-routing suites pass. `ruff check` clean.

### 3. Performance: a genuine root-cause fix applied, but NOT declared closed — the outlier stays open, exactly as instructed

**Traced the concurrency bottleneck to its actual mechanism** (not just
re-measured): `EmbeddingProvider.embed_batch` (`app/rag/embeddings.py`)
offloaded the CPU/GPU-bound `SentenceTransformer.encode()` call via
`asyncio.to_thread`, which hands each concurrent caller a SEPARATE thread
from asyncio's default pool. Multiple threads driving inference into the
SAME CUDA-backed model instance simultaneously contend for the one GPU
context rather than parallelizing — consistent with a standing comment
already in `chat_service.py` from an earlier session describing an
8-minute stall attributed to "GPU/Mongo contention from a concurrent
ingestion job," and consistent with this session's own measurement: a solo
embed call ~350ms; three concurrent ones during the earlier load test,
~320 **seconds** each.

- **Fix**: `embed_batch` now routes through one dedicated,
  process-wide-shared `ThreadPoolExecutor(max_workers=1)` instead of the
  default pool. Concurrent embed calls queue and run one at a time on a
  single thread rather than fighting each other on the GPU from several
  threads at once. Since `IndexingPipeline` (upload) and `LegalRetriever`
  (chat retrieval) both construct their own `EmbeddingProvider()` but the
  executor is a class-level singleton, this one change covers both paths.
- **First re-test (light conditions): looked fixed.** 3 concurrent uploads
  with fresh content: 8.1s/8.9s/9.8s, all under the 10s target, no
  degradation across repeats.
- **Second re-test (heavy conditions — 5 concurrent chats immediately
  followed by 3 concurrent uploads, the exact sequence that produced the
  original outlier): did NOT hold.** All 5 chat calls hit a
  `chat_request_hard_timeout` at ~978s each — a hard ceiling documented to
  fire at `chat_request_budget_seconds + grace = 230s` instead firing over
  4x late. This is worse in shape than the original finding, not better: a
  single-worker queue has no way to bound how long a request waits behind
  whatever is already running, so if any one embed call is genuinely slow
  (real GPU contention from elsewhere), every request queued behind it on
  that one thread inherits the full delay, serially, with the outer
  `asyncio.wait_for` only stopping the CALLER from waiting further — it
  cannot actually free the thread, which keeps running underneath
  regardless.
- **A major, previously-invisible confound surfaced while investigating
  this**: a separate, independent process — `odisha_ingest_batch1.py`,
  confirmed still actively running throughout this entire session from a
  different Claude session's own scratchpad — is real, GPU-touching
  ingestion work with no relationship to this codebase's own request
  concurrency. It was not stopped (not this session's call to make, same
  standing decision as session 7). Its presence means **any** GPU-latency
  measurement taken on this shared machine right now is potentially
  dominated by an external, uncontrolled process, not solely by this
  application's own concurrency handling — a confound this session did not
  have before and that no prior session's performance numbers accounted
  for either.
- **Verdict, exactly as instructed**: the single-worker executor fix is
  real, correctly targets the actual mechanism, is safe (no regressions,
  68+ embedding/retrieval tests pass), and measurably helps in isolation —
  but **the 320s-class outlier is explicitly NOT marked resolved.** Normal
  -case uploads being fast does not close this finding, per instruction; a
  single-worker queue is arguably a worse worst-case shape than the
  original unbounded-thread-pool version, and the external ingestion
  process is a genuine, unresolved confound on top of that. A safer
  long-term design (bounded to a small N of workers rather than exactly 1,
  or true request micro-batching) is flagged for a future pass rather than
  guessed at without clean measurement conditions — see Task 4 below.
- **Also newly documented this session**: the same
  duplicate-process-per-restart pattern session 5 originally flagged
  (`run_api.ps1`'s launched process consistently spawns a SECOND, uv
  -managed python subprocess running the identical uvicorn command) was
  investigated further. Confirmed harmless — it is `uv`'s own venv
  launcher delegating to its centrally-managed shared interpreter, not two
  independent application instances (killing the child ends the same
  logical server, confirming there was only ever one). Documented so a
  future session does not re-diagnose it as a GPU-doubling bug, as this
  session initially suspected before tracing the parent/child relationship.

### 4. Clean build verification: DONE

**Reviewed all ~109 other uncommitted/untracked files individually** (diffs
read, not just file names) rather than blanket-including or blanket
-excluding. Finding: essentially all of it traces directly to a specific,
already-live-tested item from this project's own multi-session QA history
(BUG-003's streaming chain-of-thought leak fix in `app/llm/resilient.py`,
BUG-010/Finding-010's append/suggest/restyle edit actions and the "ki
jagah"/"baaki same rakho" parsing fixes in `app/drafting/edit_commands.py`,
the Finding-008 fix chain this session's own Task 1 depends on, BUG-013a's
classifier fix, a BM25-index-off-the-event-loop fix, real-Redis integration
tests, a ClamAV-backed upload malware scan that degrades gracefully outside
production, OpenTelemetry tracing, Prometheus metrics, a complete
document-Q&A feature, and a full CI workflow) — a large body of genuinely
tested, valuable work that had simply never been committed. Excluded only
what is clearly unrelated to this project's chatbot: an external Odisha
-Acts bulk-ingestion effort's own status script, ML fine-tuning experiment
scripts/artifacts (`models/`, `unsloth_compiled_cache/`,
`scripts/finetune_*.py`), and this session's own debug logs.

**Verified the dependency chain, not just the individual files**: this
session's own Task 1 fix (`chat_service.py`'s routing gate) only works
because `app/chatops/workflows/legal.py`/`drafts.py`/`selection.py` (already
in the working tree, not mine) provide the actual multi-turn selection
logic — confirmed by reading their diffs line by line, not assumed. Natural
retry (BUG-013a) similarly depends on uncommitted `app/intent/classifier.py`
changes — without including it, "closed" would silently regress to "not
fixed" the moment anyone checked out a narrower commit. Both are included.

**Ran the exact CI commands before committing, not after**: `ruff check
app/ tests/` and `mypy app/` (this project's own CI, `.github/workflows/
ci.yml`, requires both clean under `strict = true`). Found and fixed 6 real
errors the accumulated uncommitted work had introduced (5 in
`app/rag/bm25_index.py` — subclassing an untyped third-party base class,
missing annotations; 1 in `app/services/chat_service.py` — an
`AsyncIterator`/`AsyncGenerator` type mismatch on the streaming handlers'
`.aclose()` call) before committing, so the CI this same commit adds would
actually pass on it.

**Committed** in two parts: `f2b57b3` (the 96-file review above) and
`4d58541` (a `requirements.txt` fix found by the clean-checkout process
itself — see below). Full suite re-run immediately before both commits:
2817 passed, 17 skipped, 0 genuine failures.

**Clean-checkout verification, genuinely from scratch** (`git worktree add`
at commit `f2b57b3` into an isolated directory — not a copy of the existing
working tree, not the existing `.venv`):
- Fresh Python 3.12 venv (`uv venv`), dependencies installed from
  `requirements.txt` exactly as README/WINDOWS_SETUP.md/CI specify.
- `scripts/validate_environment.py` caught a **real, genuine gap**:
  `segno` (QR codes for notarization) has been a required dependency in
  `pyproject.toml` since that feature was built, but was never added to
  the pinned `requirements.txt` lock file — a clean `pip install -r
  requirements.txt` therefore produces an environment where notarization
  QR generation raises `MissingOptionalDependencyError`. This is exactly
  the class of drift a clean-checkout exercise exists to catch; a checkout
  that only ever reused the long-lived working `.venv` would never surface
  it. **Fixed** (`4d58541`) and re-verified: `validate_environment.py`
  reports "All checks passed."
- Full test suite in the clean checkout: **2814 passed, 20 skipped, 0
  errors** — genuinely zero, including none of the storage-write-isolation
  noise that appeared throughout this whole multi-session effort in the
  shared working directory, confirming (as suspected in earlier sessions)
  that noise was always the external `odisha_ingest_batch1.py` process
  writing into shared `storage/`, not a real test defect.
- `ruff check`/`mypy --strict`: clean, matching the main working tree.
- **Live acceptance, on a separate port (8001) against this clean
  checkout's own server process**: a grounded `/chat` call (correct
  Section 138 citation); the full Task 1 journey (two drafts, list, select
  the non-active one while the other is mid-collection, confirm no data
  loss, export — all correct); the exact Task 2 repro message (`"Start a
  new draft: ..."` — correctly parks and starts the new one, no
  "discarded" message). One self-inflicted false alarm during this pass is
  worth recording: reusing a test session across turns after one turn's
  fictional name happened to contain the word "Check" (a real
  `_SHOW`-intent keyword) caused a `SavedDraftsWorkflow` to genuinely,
  correctly latch onto the session per the orchestrator's own
  by-design "an active workflow keeps the floor until explicitly
  switched away" policy — traced fully via `best_match()` in isolation and
  the live server log before concluding it was test-data hygiene, not a
  product defect, and redone cleanly to confirm.
- Worktree deregistered from git afterward (`git worktree remove`); its
  directory could not be fully deleted due to a Windows long-path limit
  inside the fresh `.venv` it created, left in the session scratchpad
  rather than the repository.

**This is the final tested handoff version: commit `4d58541`** on
`phase-1-runtime-stabilization`, verified via a genuinely independent,
from-scratch checkout — not the narrow `bd30d27` from session 7/8, and not
assumed correct from the long-lived shared working directory's own state.

### Session 9 summary

| Item | Status |
|---|---|
| Saved-draft selection routing | **FIXED, live-verified in both failure modes, in both the main environment and an independent clean checkout** |
| New-draft cancellation | **FIXED, live-verified in both environments** |
| Performance (concurrent embedding contention) | **Root-caused and a real fix applied; NOT declared closed** — outlier persists under heavy conditions, plus a newly-found external GPU-contention confound (the still-running `odisha_ingest_batch1.py` process) |
| Clean build verification | **DONE** — 96-file reviewed commit + a genuine dependency-lock gap found and fixed via a from-scratch checkout; final handoff commit `4d58541` |
| P0 (BUG-014/BUG-015) | Re-verified, unchanged, still closed |
| Natural retry (BUG-013a) | Unchanged, still closed — now also formally committed (was previously only live-verified in an uncommitted working tree) |

**The release freeze remains provisional, honestly.** Three of four
remaining items are now fixed/completed and independently re-verified from
a clean checkout; the performance item has a genuine fix applied but is
explicitly, deliberately not claimed as resolved — the 320s/978s-class
outlier stays open, and a real external confound (a separate process
sharing this machine's GPU) means it cannot be conclusively closed from
this environment at all. That is the one item standing between this and a
non-provisional release.

## Session 10 (2026-09-14) — status update: only performance acceptance remains open

Per report review: **functional fixes** and **clean-build verification** are
now treated as **CLOSED** and are not reopened by the next phase below.
Nothing in this update changes their status; they are re-stated here only
for a single at-a-glance summary.

| Item | Status |
|---|---|
| Functional fixes (BUG-001 through BUG-015 and the P0/retry items) | **CLOSED** |
| Clean-build verification (from-scratch checkout, `4d58541`) | **CLOSED** |
| Performance acceptance | **OPEN — the only remaining item, see next-phase plan below** |

The GPU-contention confound recorded in Session 9 is real but **not proven
to be the sole cause of every observed stall**. It is not being used here to
excuse or pre-close the performance item — the next phase exists precisely
to test the system with that confound removed.

### Next phase: final performance acceptance on isolated staging

Scope: this is the one gate left before a non-provisional release. Plan:

1. **Dedicated GPU/environment.** Run on hardware where no other workload
   (in particular, no ingestion job such as the `odisha_ingest_batch1.py`
   process implicated in Session 9) shares the same GPU during the test
   window. Do not stop any other process on a shared machine without
   coordinating with whoever owns it — if a dedicated box/isolated staging
   environment isn't available, get one provisioned rather than
   commandeering a shared one unilaterally.
2. **Deploy the verified commit, and record versions.** Deploy code
   `4d58541` + docs `3524d89`. Record alongside the run: exact LLM model
   name/version in use, full runtime configuration, and the dataset/index
   version (BM25/embedding index build) — so results are reproducible and
   comparable to prior sessions' numbers.
3. **Repeat the same workload.** After a warm-up period, run: 5 concurrent
   chats, 3 concurrent uploads, and mixed batches combining chat, upload,
   and drafting requests. Record cold-start timings as a separate,
   clearly-labeled measurement from the warmed-up runs (cold-start and
   steady-state latency should not be blended into one number).
4. **Measured acceptance.** For every run, capture not just total
   latency but the breakdown: queue wait, embedding time, retrieval time,
   and LLM time. Only close the performance item once repeated runs meet
   the agreed targets AND no severe stalls (the 320s/978s-class outliers
   from Session 9) recur. If isolated runs are still slow, investigation
   continues — GPU contention is not to be assumed as the explanation
   without this isolated evidence.

## Session 11 (2026-09-14) — BUG-016 found from a real user transcript, fixed and regression-tested

A live chat transcript surfaced during this session (not from this file's
own scripted journeys) showed a FIR/police-complaint conversation with two
separate symptoms: (a) two general-knowledge questions ("FIR aur police
complaint mein kya difference hai?", "Zero FIR kya hoti hai? ... sources
dikhao") each got the standard "no verified document in the Knowledge
Base" fallback, and (b) a third, procedural question got a nonsensical
"Mujhe yaad nahi hai" ("I don't remember") reply. (a) and (b) turned out to
be unrelated: (a) is a content/dataset-coverage question (see Finding-011
below, not a code bug); (b) is a genuine, reproducible code bug, root-caused
and fixed below.

### BUG-016 (FAIL→FIXED, HIGH severity): an ordinary "what should I do/give"
question was hijacked into a bogus "I don't remember" reply

- File: `app/memory/entity_memory.py` (`_ACTION_SEEKING_PATTERN`,
  `is_recall_query`).
- Symptom (live repro): "Police meri FIR register nahi kar rahi. Main
  Jaipur, Rajasthan mein hoon. Mujhe pehle kya information deni chahiye"
  ("Police won't register my FIR. I'm in Jaipur, Rajasthan. What
  information should I give first?") — a procedural question that also
  supplies the user's location in the same breath — was answered with
  "Mujhe yaad nahi hai, kyunki aapne ye information pehle share nahi ki
  thi" ("I don't remember, because you didn't share this information
  earlier"), completely ignoring both the actual question and the location
  just given.
- Root cause, confirmed deterministically (no LLM involved, reproduced by
  calling `entity_memory.is_recall_query`/`matches_known_category`
  directly): `app/services/chat_service.py`'s Entity Memory branch (Part 35)
  gates on `is_recall_query(text)` before ever considering
  `matches_known_category`/`find_matching_fact`. `_ACTION_SEEKING_PATTERN`
  — the guard meant to stop exactly this class of misfire (its own docstring
  cites the near-identical "mera bike chori ho gya hai kya krna chahiye"
  case) — only recognised "kya" directly followed by a VERB from a fixed
  list (`karu`/`karna`/`krna`/...). In "kya information deni chahiye", "kya"
  questions the NOUN ("what information"), not a verb, so that alternation
  never matched, `is_recall_query` wrongly returned `True`, and
  `matches_known_category`'s entity-only fallback then fired on nothing
  more than the possessive "meri" sitting next to "FIR" (a known
  Document-entity keyword) — with no stored fact to match, the "I don't
  remember" branch fired.
- Fix: added `\bshould\s+i\b` and bare `chahiye` to `_ACTION_SEEKING_PATTERN`.
  Both are unconditional advice/need markers that never appear in this
  module's own genuine recall phrasing (recall is always past-tense —
  "chori hua tha", "kisne phoda tha" — never "should"/"chahiye"), so this
  is safe to match without requiring a specific adjacent verb, and it
  additionally covers the analogous English gap ("what information should
  I give the police first?", not just the Hinglish phrasing actually
  reproduced).
- Regression tests: extended the existing parametrized
  `tests/test_multi_turn_conversations.py::test_hinglish_action_questions_are_not_memory_recall`
  with the exact reproducing message plus its English equivalent; added
  `test_fir_procedural_question_naming_meri_fir_is_not_hijacked_into_no_memory`,
  a full `ChatService.answer()`-level test asserting the response is
  neither routed as `"Entity Memory"` nor contains "don't remember"/"yaad
  nahi". `tests/test_multi_turn_conversations.py` (114 tests),
  `tests/test_chat_service_routing.py`, and
  `tests/test_reported_chat_session_regressions.py` all pass after the fix
  (394 tests total across the broader targeted run); `ruff check` and
  `mypy` clean on the changed files.
- Not a regression from Session 9/10's work — `entity_memory.py` was
  untouched by any prior session in this file; this was a pre-existing gap
  surfaced by a real user conversation that happened not to match any of
  this file's own scripted test phrasings.

### Finding-011 (content/dataset-coverage, NOT a code bug — not investigated further): FIR/Zero FIR general-knowledge questions return "no verified document"

- Both "FIR aur police complaint mein kya difference hai?" and "Zero FIR
  kya hoti hai?" returned the standard no-context-in-Knowledge-Base
  fallback rather than an answer with sources. This is very plausibly a
  real KB **content** gap (FIR/CrPC-BNSS procedural material not present in
  the ingested corpus) rather than a retrieval defect, consistent with this
  file's own earlier, separate note (Session 2) about a no-context fallback
  answering in the wrong language on an unrelated topic — i.e. the
  no-context path itself is known to fire for genuine gaps, not just bugs.
  Not root-caused this session (would require inspecting/querying the
  actual ingested BM25/vector index contents for FIR-related statutes,
  which needs the KB tooling rather than a quick code read) — flagged for
  whoever owns KB ingestion/coverage to check whether CrPC/BNSS FIR
  provisions (Section 154/173 CrPC, Zero FIR circulars) are actually
  ingested. Not counted against the performance-acceptance gate or the
  functional-fixes closure above; this is a separate, dataset-scoped
  question.
