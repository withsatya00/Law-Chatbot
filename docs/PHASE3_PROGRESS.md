# Phase 3 progress

Advanced document intelligence and a complete chat-first product experience.

```
PHASE 3 PROGRESS
[####################] 100%
Completed: A, B, C, D, E, F, G, H
Current:   Phase 3 complete
Next:      Phase 4 remains deferred until explicitly requested
```

Baseline carried in from Phase 2 (verified, not re-audited): 1767 passed /
0 failed / 3 skipped, Ruff clean, strict mypy clean over 194 files, index
drift `false`, grounding 1.00, safe-decline 1.00, source precision 0.57.
Working tree is the uncommitted Phase 2 tree on branch
`phase-1-runtime-stabilization`. Nothing is committed in this phase.

---

## Milestone A — REST-to-chat capability inventory (+5%)

Evidence: `app/api/*.py` route decorators (124 routes), the nine classes
registered in `app/chatops/registry.WORKFLOWS`, the conversation-intent
branches in `ChatService._dispatch_conversation_intent`, and
`streamlit_app/app.py`. Claims in comments were not taken as evidence; every
"available in chat" row below names the workflow class or the
`_respond_with_*` branch that actually serves it.

### Status legend

| Code | Meaning |
|---|---|
| 1 | Fully available through chat |
| 2 | Partially available through chat |
| 3 | REST-only (gap to close in this phase) |
| 4 | Admin-only (reachable only with an admin/notary role) |
| 5 | Intentionally unavailable in chat |
| 6 | External integration required |

### Registered chat workflows found (9)

`cyber_fraud`, `jurisdiction`, `saved_drafts`, `draft_export`,
`notarization_prepare`, `notarization_status`, `notarization_verify`,
`notary_queue`, `notary_admin`.

Conversation-intent branches that also serve user capabilities without being
workflows: Draft Generation (`DraftConversationEngine`), Document Analysis,
Translation, Summarization, Conversation Memory, Lawyer Recommendation,
Capability Question, Response Modification, Intent Feedback, Retry Failed
Request.

### Matrix

Ownership rule column: `owner` = owner_user_id scoped in the service;
`session` = session-scoped fallback for anonymous users; `admin` = role-gated
only; `public` = deliberately unauthenticated.

#### Chat and retrieval

| REST route | Service | Chat intent / workflow | Role | Confirm | Ownership | Status |
|---|---|---|---|---|---|---|
| `POST /chat` | `ChatService.answer` | (the chat itself) | any | no | session/owner | 1 |
| `POST /chat/stream` | `ChatService` | streaming transport | any | no | session/owner | 1 |
| `DELETE /chat` | `ConversationMemoryStore` | "clear this conversation" | any | yes | session | 3 |
| `POST /search` | `SearchService` | ordinary question | any | no | public corpus | 1 |
| `POST /summarize` | summarizer | Summarization intent | any | no | session | 2 |
| `POST /intent` | `IntentDetector` | introspection only | any | no | n/a | 5 |
| `POST /entities` | `EntityExtractor` | introspection only | any | no | n/a | 5 |
| `POST /feedback` | `IntentFeedbackRepository` | Intent Feedback intent | any | no | session | 2 |
| `GET /history` | `ChatRepository` | Conversation Memory intent | any | no | session/owner | 2 |
| `GET /session` | memory store | Conversation Memory intent | any | no | session | 2 |
| `DELETE /session` | memory store | — | any | yes | session | 3 |
| `DELETE /me/data` | user data purge | — | user | strong | owner | 3 |

#### Auth

| REST route | Service | Chat | Role | Status |
|---|---|---|---|---|
| `POST /auth/register` | `AuthService` | — | anon | 5 |
| `POST /auth/login` | `AuthService` | — | anon | 5 |
| `POST /auth/refresh` | `AuthService` | — | anon | 5 |
| `POST /auth/logout` | `AuthService` | — | user | 5 |

Credentials are deliberately never collected in a chat transcript: the
transcript is stored, logged and summarised. Category 5, not a gap.

#### Documents

| REST route | Service | Chat workflow | Role | Confirm | Ownership | Status |
|---|---|---|---|---|---|---|
| `POST /upload` | `DocumentService.upload_and_index` | composer attachment | any | no | owner/session | 1 |
| `POST /document-analysis` | `DocumentService.analyze` | Document Analysis intent | any | no | owner/session | 2 |
| document summary | `DocumentService.analyze` | — | any | no | owner/session | 3 |
| risky clauses | `DocumentService.analyze` | — | any | no | owner/session | 3 |
| choose between uploads | — | — | any | no | owner/session | 3 |
| document-specific review checklist | *(does not exist)* | — | any | no | owner/session | 3 |
| two-document comparison | *(does not exist)* | — | any | no | owner/session | 3 |

#### Drafting (19 routes)

| REST route | Service | Chat workflow | Confirm | Ownership | Status |
|---|---|---|---|---|---|
| `GET /draft-templates` | `LegalDraftEngine.list_templates` | Draft Generation intent | no | public | 2 |
| `GET /draft-templates/{id}` | `get_template_detail` | — | no | public | 3 |
| `POST /draft` | `DraftConversationEngine` | Draft Generation intent | no | owner/session | 1 |
| `POST /draft/edit` | `LegalDraftEngine.regenerate` | draft conversation edit | no | owner | 2 |
| `POST /draft/export` | `LegalDraftEngine.export` | `draft_export` | no | owner | 1 |
| `DELETE /draft/{id}` | drafts repo | — | yes | owner | 3 |
| `POST /draft/audit` | `fact_audit` | — | no | owner | 3 |
| `POST /draft/approve` | `approve` | — | yes | owner | 3 |
| `POST /draft/lock` | `lock` | — | yes | owner | 3 |
| `POST /draft/unlock` | `unlock` | — | yes | owner | 3 |
| `POST /draft/rollback` | `rollback` | — | yes | owner | 3 |
| `GET /draft/{id}/versions` | `list_versions` | — | no | owner | 3 |
| `POST /draft/translate` | `translate` | Translation intent | no | owner | 2 |
| `POST /legal-terms` | `legal_terms` | glossary answers | no | public | 2 |
| `POST /draft-history` | `history` | `saved_drafts` | no | owner/session | 1 |
| `GET /draft/{id}/review` | review builder | — | no | owner | 3 |
| `GET /draft/{id}/compare` | version compare | — | no | owner | 3 |
| `POST /draft/{id}/duplicate` | duplicate | — | no | owner | 3 |
| `POST /draft/{id}/conflicts/resolve` | conflict resolver | — | no | owner | 3 |

#### Cases (17 routes) — no chat coverage at all today

`POST /cases`, `GET /cases`, `GET /cases/upcoming-hearings`,
`GET /cases/hearing-reminders`, `GET /cases/{id}`, `PATCH /cases/{id}`,
`DELETE /cases/{id}`, `POST /cases/{id}/hearings`,
`POST /cases/{id}/hearings/{hid}/acknowledge-reminder`,
`POST /cases/{id}/notes`, `POST /cases/{id}/documents`,
`POST /cases/{id}/evidence/upload`, `POST /cases/{id}/drafts`,
`POST /cases/{id}/tasks`, `POST /cases/{id}/timeline`,
`POST /cases/{id}/conflicts/resolve`, `GET /cases/{id}/lawyer-summary`.

All backed by `CaseService`, all owner-scoped through
`CaseService._get_owned_case`. Status **3** for every one. Delete requires
confirmation.

#### Phase 2 legal workflows

| REST route | Service | Chat workflow | Status |
|---|---|---|---|
| `POST /cyber-fraud` | `cyber_fraud_workflow` | `cyber_fraud` | 1 |
| `POST /jurisdiction` | `jurisdiction_check` | `jurisdiction` | 1 |
| `POST /evidence/organize` | `organize_evidence` | — | 3 |
| `POST /timeline` | `analyze_timeline` | — | 3 |

#### Assistant / user features (`/assistant/*`, 17 routes)

| REST route | Service | Chat workflow | Confirm | Ownership | Status |
|---|---|---|---|---|---|
| `POST /assistant/follow-up` | `next_follow_up` | used internally | no | n/a | 1 |
| `GET /assistant/preferences` | `PreferenceService.get` | — | no | owner | 3 |
| `PATCH /assistant/preferences` | `PreferenceService.update` | — | no | owner | 3 |
| `DELETE /assistant/preferences` | `PreferenceService.delete` | — | yes | owner | 3 |
| `POST /assistant/forms` | `FormWorkflowService.create` | — | no | owner | 3 |
| `GET /assistant/forms` | `.list` | — | no | owner | 3 |
| `GET /assistant/forms/{id}` | `.get` | — | no | owner | 3 |
| `PATCH /assistant/forms/{id}` | `.update` | — | no | owner | 3 |
| `POST /assistant/forms/{id}/confirm` | `.confirm` | — | yes | owner | 3 |
| `POST /assistant/jobs` | `BackgroundJobService.create` | — | no | owner | 3 |
| `GET /assistant/jobs` | `.list` | — | no | owner | 3 |
| `GET /assistant/jobs/{id}` | `.get` | — | no | owner | 3 |
| `POST /assistant/jobs/{id}/retry` | `.retry` | — | yes | owner | 3 |
| `GET /assistant/downloads` | download artifacts | — | no | owner | 3 |
| `GET /assistant/downloads/{id}` | download artifact | — | no | owner | 3 |
| `POST /assistant/legacy-reference` | `LegalUpdateService.map_legacy` | — | no | public | 3 |
| `GET /assistant/search` | search | ordinary question | no | owner | 2 |

#### Lawyer recommendation

| REST route | Service | Chat | Status |
|---|---|---|---|
| `POST /recommend-lawyer` | `LawyerRecommendationEngine.recommend` | Lawyer Recommendation intent | 2 |

Partial: the engine returns only `category`/`confidence`/`reason`. No
specialization rationale, urgency, jurisdiction note, documents-to-carry,
questions-to-ask, or directory-availability signal. No verified directory
provider exists — names/numbers/ratings must never be generated (category 6
for any real directory lookup).

#### Notarization (15 routes)

| REST route | Service | Chat workflow | Role | Status |
|---|---|---|---|---|
| `POST /notarization/prepare` | `NotarizationService` | `notarization_prepare` | user | 1 |
| `GET /notarization/eligibility` | service | `notarization_prepare` | user | 2 |
| `GET /notarization/signing/providers` | e-sign registry | — | user | 6 |
| `POST /notarization/signing/initiate` | e-sign provider | `notarization_prepare` (secure url) | user | 6 |
| `POST /notarization/signing/callback` | provider webhook | — | provider | 6 |
| `POST /notarization/requests` | service | `notarization_prepare` | user | 1 |
| `GET /notarization/requests` | service | `notary_queue` | notary | 4 |
| `POST /.../start-review` | service | `notary_queue` | notary | 4 |
| `POST /.../decision-token` | tokens | `notary_queue` | notary | 4 |
| `POST /.../approve` | service | *(human notary only)* | notary | 4 |
| `POST /.../reject` | service | *(human notary only)* | notary | 4 |
| `POST /.../revoke` | service | `notary_admin` | admin | 4 |
| `GET /notarization/documents/{id}/status` | service | `notarization_status` | owner | 1 |
| `GET /notarization/documents/{id}/download` | service | artifact link | owner | 1 |
| `GET /verify/{token}` | service | `notarization_verify` | public | 1 |

#### Admin knowledge base (`/admin/*`, 12 routes) — no chat coverage

`GET /admin/knowledge-base/status`, `GET /admin/logs/status`,
`POST /admin/reindex`, `GET /admin/reindex/{job_id}`,
`POST /admin/cache/flush`, `POST /admin/knowledge-base/upload`,
`POST /admin/knowledge-base/backfill-uploads`,
`POST /admin/knowledge-base/reconcile-staging`,
`GET /admin/knowledge-base/dashboard`, `GET /admin/documents/unowned`,
`POST /admin/documents/{source}/assign-owner`,
`GET /admin/knowledge-base/staging`. All `require_admin`. Status **3**
(reachable only via REST today; target is 4 = admin-only *in chat*).
Reindex, cache flush, reconcile-apply and owner assignment require
confirmation.

#### Admin Phase-3 governance (`/admin/phase3/*`, 10 routes) — no chat coverage

`GET /dashboard`, `GET /metrics`, `GET /audit-logs`,
`POST /knowledge-gaps/{message_id}/source`, `GET /legal-sources`,
`POST /legal-sources/{id}/verify`, `POST /legal-sources/{id}/review`,
`POST /legal-sources/{id}/supersede`, `POST /legal-sources/{id}/link-document`,
`POST /evaluations/run`. All `require_admin`, status **3**. Verify/review/
supersede are governance state changes and require confirmation. Automatic
verification of a legal source is forbidden.

#### Analytics (4 routes) — no chat coverage

`GET /analytics/dashboard`, `GET /analytics/cache`,
`GET /analytics/unanswered-queue`, `POST /analytics/unanswered-queue/{id}/review`.
`require_admin`, status **3**.

#### Health / ops (4 routes)

`GET /health`, `/health/live`, `/health/ready`, `/health/index-drift`.
Operational probes, category **5** for chat (an admin can still ask for KB
status, which is the meaningful subset).

#### Voice (3 routes)

| REST route | Chat | Status |
|---|---|---|
| `POST /voice/chat` | composer microphone | 1 |
| `POST /voice/speak` | 🔊 replay control | 1 |
| `POST /voice/drafts/{id}/confirm` | voice-draft confirmation | 2 |

### Summary of coverage before Phase 3

| Status | Count |
|---|---|
| 1 — fully in chat | 18 |
| 2 — partial | 15 |
| 3 — REST-only (gap) | 78 |
| 4 — admin/notary-only, in chat | 7 |
| 5 — intentionally unavailable | 9 |
| 6 — external integration | 3 |

Explicitly missing chat workflows (the Milestone C/D/E backlog): all 17 case
routes; 13 of 19 draft-management routes; document summary / risky-clause /
multi-document selection / review checklist / comparison; evidence
organisation and timeline; preferences; background jobs; downloads;
personal-data deletion; conversation clearing; all 12 admin KB routes; all 10
admin governance routes; all 4 analytics routes; legacy-reference lookup.

### Verified corrections to existing claims

* `streamlit_app/app.py` claims every capability is reachable by talking.
  False as written — nine workflows exist; cases, draft management, admin,
  preferences and jobs have no workflow. The sidebar cleanup it describes is
  real and already done.
* `app/chatops/__init__.py` claims "admin actions" are reachable. False — no
  admin workflow other than `notary_admin` is registered.

Focused acceptance for A: the matrix covers all 124 user-facing routes, names
the backing service for each, and lists the missing workflows explicitly.

---

## Milestone B — orchestration foundation (+15%)

Extended the existing `app/chatops` machinery; nothing was replaced.

Files: `app/chatops/intents.py`, `app/chatops/state.py`, `app/chatops/base.py`,
`app/chatops/orchestrator.py`, `app/schemas/chat.py`,
`app/services/chat_service.py`, `tests/test_chatops_foundation.py` (new).

What was added:

* **Resume** (`intents.RESUME`, `state.resume`) — "continue" / "aage badho" /
  "jaari rakho" brings a parked workflow back with its facts. Anchored to the
  start of the message so "the lease continues until March" is not a control
  verb.
* **Fact correction** (`intents.CORRECTION`, `state.forget_fact`) — "change
  the alpha to X" clears exactly the named field. Which field is named is
  decided by matching the field LABEL, never by asking a model.
* **Stale-state expiry** (`state.STALE_AFTER`, 45 min, `expire_stale`) — a
  half-finished workflow older than the window is dropped before anything
  reads it, so yesterday's half-typed address is never reused as if it had
  just been said.
* **Ambiguity** (`_ambiguity_turn`, `_resolve_pending_choice`) — two
  capabilities scoring identically produce a numbered list of REGISTERED
  workflow titles; the reply is resolved against that list, expires after one
  turn, and an unrelated reply falls through as a fresh message.
* **Interruption** (`_is_interruption`) — an ordinary legal question asked
  mid-workflow parks it and returns `None`, so `ChatService` answers normally
  and the facts survive for "continue". A message that matches another
  workflow is a task switch, not an interruption.
* **Idempotency** (`ChatWorkflow.idempotency_key`, `state.already_executed` /
  `mark_executed`) — the ledger lives in conversation memory, not on the
  workflow entry, because the entry is popped the moment a workflow completes
  which is exactly when a replay becomes possible. Only successful runs are
  recorded, so a genuine retry after a failure still executes. Bounded to 50.
* **Structured progress** — `workflow_name`, `current_step`,
  `completed_steps`, `total_steps`, `progress_percentage`, `required_field`
  on both `WorkflowTurn` and `ChatResponse`. Derived from `required_fields`,
  so a workflow gets a progress bar by declaring what it needs and the
  reported progress cannot drift from the facts actually held. Every existing
  response field is untouched.

Authorization was already class-declared and orchestrator-checked; the inline
refusal was factored into `_forbidden_turn` and now carries `workflow_name`
without leaking it into the user-facing text.

Focused tests: `tests/test_chatops_foundation.py` — 27 passed. Regression
sweep over `test_chatops_orchestration`, `test_chat_service_routing`,
`test_multi_turn_conversations`, `test_reported_chat_session_regressions`,
`test_workflow_orchestrator`, `test_response_shape` — 360 passed. Ruff clean,
strict mypy clean on the changed modules.

---

## Milestone C — complete chat coverage (+20%)

Registered workflows: **9 → 26**. New: `document_summary`, `risky_clauses`,
`document_choose`, `draft_manage`, `case_create`, `case_list`, `case_manage`,
`evidence_organize`, `lawyer_summary`, `preferences`, `downloads`,
`background_jobs`, `clear_conversation`, `delete_my_data`,
`admin_knowledge_base`, `admin_analytics`, `admin_sources`.

### Service extractions (business logic moved OUT of route bodies)

A chat workflow cannot call a route, so logic living in a route body can only
be reused by re-implementing it — and a second implementation of "who may
delete this draft" is how cross-owner leaks happen. Four extractions, all
behaviour-preserving, both callers now sharing one implementation:

| New service | Was in | Now used by |
|---|---|---|
| `app/services/draft_management.py` | `app/api/drafting.py` route bodies | `/draft/*` routes + `draft_manage` |
| `app/services/user_data.py` | `app/api/history.py` route bodies | `DELETE /session`, `DELETE /me/data` + `clear_conversation`, `delete_my_data` |
| `app/services/admin_operations.py` | `app/api/admin.py` route bodies | `/admin/*` routes + `admin_knowledge_base` |
| `ensure_document_access` (module-level in `document_service.py`) | private method | `DocumentService` + `document_insight.load_document` |

Also new: `app/chatops/selection.py` (pick one record by number, ordinal or
name — never by id), `app/services/document_insight.py` (owner-checked
document loading with page provenance), `AuditLogRepository.find_recent`.

### Supporting changes

* `DocumentService.upload_and_index` now appends to `uploaded_documents` in
  conversation memory. That key was declared and persisted but never written;
  without the LIST, a second upload silently replaced the first and neither
  document selection nor two-document comparison was possible.
* `chatops_stack` and `chatops_executed` are now persisted by
  `ConversationMemoryStore`, so "continue" survives a Redis eviction.
* `ChatWorkflow.needs_confirmation(context)` — per-run rather than per-class,
  because listing a case's hearings and deleting the case are the same
  workflow and only one may proceed on a bare request.
* The ambiguity question is filtered by role BEFORE it is asked: offering
  "notary administration" as an option would itself disclose a privileged
  capability to an ordinary user.
* One real multilingual bug fixed: `\b` is defined in terms of `\w`, which
  excludes Indic combining marks, so `हटा\b` never matched — "ड्राफ्ट हटा दो"
  silently did not route. Same class of failure `confirm.py` documents.

### Confirmation matrix (chat)

| Confirmed | Not confirmed |
|---|---|
| draft delete / approve / lock / unlock / rollback | draft open, versions, compare, duplicate, translate, review |
| case delete, case close | case list, open, notes, tasks, hearings, timeline, evidence |
| account data erasure (typed phrase **and** yes) | preferences, downloads, job listing |
| clear conversation | — |
| job retry | — |
| re-index, cache flush, reconciliation apply, owner assignment | KB status, dashboard, staging, unowned report |
| legal-source review, evaluation run | source listing, audit log |

Account erasure requires the phrase **"delete my data"** typed in full before
the yes/no turn is even reached: a bare "yes" is one keystroke and the action
spans every session the account ever had.

Focused tests: `tests/test_chatops_coverage.py` — 69 passed (happy paths,
missing information, numbered selection, role denial, cross-owner denial,
confirmation, cancellation, duplicate-retry prevention, no outbound network
call from any workflow module). Ruff clean over `app` and `tests`; strict
mypy clean over 204 source files.

---

## Milestone D — entities and timeline (+20%)

New: `app/schemas/extraction.py`, `app/entity_extraction/rules.py`,
`app/entity_extraction/timeline.py`, `app/entity_extraction/hybrid.py`,
`LoadedDocument.page_index()`, `EntityExtractor.extract_structured()`,
`document_timeline` workflow, `tests/test_structured_extraction.py`.

### The extraction order, enforced

1. deterministic rules — the matched span IS the evidence;
2. conversation facts — what the user already told us beats any reading;
3. an optional model pass, skipped when no provider is configured, when the
   text is under 400 chars, or when the request deadline is nearly spent;
4. strict Pydantic validation with a **document-containment gate**: a value
   that does not literally appear in the text is dropped, which is what makes
   a fabricated party name structurally impossible;
5. deterministic fallback — every failure leaves the rule results standing.

**A model may never confirm a legal role.** Its role proposals are capped at
0.55 confidence, land in `unresolved_roles`, and become questions. A rule that
found a role from a label in the text always wins.

### Four real extraction defects found and fixed while building this

| Symptom | Cause |
|---|---|
| Party named "Vikram Singh FIR No" | `\s+` in the name pattern crossed the newline into the next form field |
| A person named "shall vacate the premises", role `tenant` | `re.IGNORECASE` on the whole pattern turned `[A-Z]` into "any letter"; labels are now scoped `(?i:...)` |
| Section "s. 45" invented from "Rs. 45,000" | the `s\.` abbreviation had no leading `\b` |
| Address reported as "Rahul Sharma, 12 MG Road, Pune 411001" | the address pattern started at any capital and swallowed the party name; now anchored on a leading number and trailing PIN |

### Timeline

Dated events sorted chronologically; undated events kept in their own list
(a placeholder date puts them in a false order that reads as fact);
contradictions reported, never resolved. Relative deadlines are kept as what
they are — phrase, day count, and what they run from — and resolved to a date
**only when the text supplies the anchor**. "Within 30 days from the date of
receipt" with no stated receipt date produces a question, not a limitation
date. Understood in English, Hindi and Hinglish, including number words.
Dates are date-only: a hearing has a date, not an instant, and attaching a
time would invent both precision and a timezone.

`POST /entities` gains `structured`, `unresolved_roles`, `conflicts`,
`timeline`, `undated_events`, `date_contradictions` and `questions` — all
additive, with the flat `entities` map byte-identical to before (pinned by a
test). No model call is made from that route.

Focused tests: `tests/test_structured_extraction.py` — 55 passed. Regression
sweep 243 passed. Ruff clean over `app` and `tests`; strict mypy clean over
208 source files.

---

## Milestone E — document review and comparison (+15%)

New: `app/schemas/document_review.py`, `app/services/document_review.py`,
`app/services/document_comparison.py`, `document_review` and
`document_compare` workflows, `tests/test_document_review_and_comparison.py`.

### Review

Validated checklists for all ten required types: rental agreement,
employment agreement, NDA, service agreement, sale agreement, partnership
deed, legal notice, affidavit, complaint, power of attorney. Each clause
carries the wording that evidences it, whether its absence is critical, and
the question to ask about it.

**Three outcomes, never two.** `present` (quoted, with page), `missing`
(genuinely absent from a document that extracted cleanly), and
`unable_to_determine` (the extraction is too thin to support either claim).
`_extraction_quality` gates the absence claim on ≥600 characters and ≥55%
alphabetic content: telling somebody their agreement has no termination
clause when really the OCR failed is a confident falsehood about a document
they are about to sign. A thin extraction reports `risk_level: unknown`, not
`low` — an unreadable document is not a safe one.

An unrecognised document type runs **no** checklist and says so. Running an
employment checklist over a sale deed produces a page of confident nonsense.

### Comparison

Aspect-based over the fifteen aspects the milestone lists, each located,
quoted with its page in both documents, and classified added / removed /
changed / unchanged. Both documents are loaded through
`document_insight.load_document`, so a document belonging to someone else
raises before any of its text is read — its contents, filename and existence
all stay invisible. Documents are addressed by id throughout; nothing is
matched by filename.

### Three real defects found by the tests

| Symptom | Cause | Fix |
|---|---|---|
| "courts at Pune" → "courts at Mumbai" compared as **unchanged** | a 0.92 similarity threshold: a one-word substitution is ~97% similar as a string and completely different as a term | materiality is not a matter of degree — compare normalised token sequences |
| A re-typeset document reported 5 spurious changes | quotes were fixed character windows, and re-spacing shifts every offset | quote the sentence containing the cue; a sentence boundary does not move when whitespace does |
| A rent rise 25,000 → 32,000 compared as **unchanged** | the sentence splitter broke on the period in "Rs.", truncating the quote one word before the amount | protect abbreviation and decimal periods before splitting |

Focused tests: `tests/test_document_review_and_comparison.py` — 39 passed
(identical documents, modified clause, added clause, removed clause, page
evidence, scanned document, low-quality extraction, cross-owner denial,
multilingual request). Regression sweep 297 passed. Ruff clean; strict mypy
clean over 211 source files.

---

## Milestone F — lawyer recommendation and multilingual UX (+10%)

New: `app/recommendation/directory.py` and
`tests/test_lawyer_and_multilingual_ux.py`. Modified:
`app/recommendation/engine.py`, `app/schemas/common.py`,
`app/api/recommend.py`, `app/services/chat_service.py`, and
`app/chatops/orchestrator.py`.

`LawyerDirectory` is the only boundary through which a person's name,
contact, enrolment, location, or profile may enter a recommendation. The
default `NullLawyerDirectory` returns no people. A directory outage degrades
to a specialization suggestion, and the response states whether a directory
was connected instead of filling the gap with model output. Recommendations
now include deterministic urgency, jurisdiction cautions, documents to take,
and questions to ask.

English, Hindi, and Hinglish renderers cover the recommendation. An explicit
language request wins over ambient detection. A standalone language switch
during an active workflow is conversation control, not the answer to a legal
field: workflow facts survive and the preference carries into the next turn.
Translation still reads its source language from the prior conversation, so
"translate into Hindi" cannot accidentally issue a Hindi-to-Hindi prompt.

Focused verification: recommendation, ChatOps foundation/coverage, chat
routing, multi-turn, greeting, and language/entity regressions — **394
passed**. Ruff and strict mypy are clean on the seven changed source modules.

---

## Milestone G — integration, security and conversational evaluation (+10%)

New: `app/chatops/evaluation.py`,
`tests/benchmarks/phase3_conversations_v1.json`, and
`tests/test_phase3_conversational_evaluation.py`.

The new benchmark contains inputs and expectations only. Observed language
and workflow are produced by the real `LanguageDetector` and registered
ChatOps matcher; unlike the old Phase-3 fixture, it cannot award itself a
perfect score using hand-written `observed` objects.

Its first run exposed three real routing collisions: a Hindi saved-draft list
request entered lifecycle management; an explicitly risky-clause request was
misclassified by an over-broad review expectation; and "timeline from this
document" entered case management. Specificity and precedence were corrected,
without lowering the benchmark floor.

Metrics over 20 deterministic conversations: routing accuracy **1.00**,
language accuracy **1.00**, ordinary-question workflow safety **1.00**.
The focused integration/security sweep covering ChatOps, recommendation,
review/comparison, structured extraction, ownership, and security passed
**234/234**. Targeted Ruff and strict mypy are clean.

---

## Milestone H — documentation, UI verification and final gates (+5%)

The final registry contains **29 workflows** (not the earlier informal claim
of 31). The user guide now describes the product that actually exists: the
sidebar contains conversations and minimal settings, while cases, drafts,
downloads, document intelligence, notarization preparation/status, and
role-gated operations are requested in chat. The Phase-3 guide records the
real 20-case evaluation and its measured dimensions.

The production `EvaluationService` now runs that real benchmark too. Its old
default fixture carried both `expected` and hand-written `observed` objects,
so the admin endpoint could report perfect citation, grounding, export, and
latency scores without executing those systems. It now reports only what it
actually measures: routing accuracy, language accuracy, and ordinary-question
workflow safety. Moving this evaluator behind the service call boundary also
avoids a ChatOps/service import cycle. A natural Hindi saved-drafts phrase
found by the production-service test was added to the deterministic matcher.

### Final gates (2026-09-03)

| Gate | Result |
|---|---|
| Strict environment validator | pass; all reported checks green, including MongoDB, Redis, Gemini, Poppler, Tesseract, WeasyPrint, QR, PDF/DOCX/RTF/TXT exports |
| Ruff (`app streamlit_app scripts tests`) | pass |
| Strict mypy (`app`) | pass, 213 source files |
| Complete pytest suite | **1970 passed, 0 failed, 3 skipped** |
| Phase-3 conversation benchmark | 20 cases; routing 1.00, language 1.00, ordinary-question safety 1.00 |
| Phase-2 live retrieval benchmark | source precision 0.57 (4/7), section accuracy 1.00 (1/1), grounding 1.00 (16/16), 2 disclosed corpus gaps |
| Mongo/BM25 reconciliation dry-run | 2079 / 2079 chunks; 0 stale, 0 missing, 0 ownership problems; `drifted: false` |
| FastAPI startup + live health | lifespan start/stop pass; `GET /health/live` returned 200 / `alive` |
| Streamlit startup smoke | real server started headless; root returned HTTP 200 with Streamlit marker |

Phase 3 is complete at **100%**. No commit was created. Human source review
and the two already-disclosed corpus ingestion gaps remain operational data
work rather than unfinished Phase-3 code.
