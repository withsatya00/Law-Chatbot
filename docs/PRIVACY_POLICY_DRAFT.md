# Privacy policy / terms of service -- draft for legal counsel review

**Status: DRAFT. Not published, not legally reviewed, not user-facing.**
Every claim below is grounded in what the code actually does as of
2026-09-07, cited by file path -- nothing here is aspirational or invented.
Sections marked **NOT YET IMPLEMENTED** describe real gaps: no code, no
policy, no user-facing text exists for them today. This document is a
starting point for a qualified lawyer to turn into an actual, publishable
policy -- it is not that policy.

## 1. What this product is (in-product disclaimers, verbatim)

Every substantive legal answer already carries this disclaimer
(`app/core/constants.py`):

> "This information is provided for educational purposes only and should
> not be considered legal advice. Please consult a qualified advocate for
> legal advice specific to your situation."

Every generated draft document carries this one:

> "This document is an AI-generated draft based on the information provided
> by the user. It is intended only as a drafting aid. Before submitting it
> before any Court, Government Authority, Police Department, Tribunal, or
> any legal forum, it must be reviewed and approved by a qualified
> advocate."

The lawyer-recommendation feature additionally discloses: "No verified
lawyer directory is connected. This is a specialization suggestion, not a
lawyer listing." (`app/schemas/common.py`). A published policy should
restate these three disclaimers rather than contradict or soften them --
they are load-bearing for the product's own liability position.

## 2. Data this product holds, and for how long

- **Chat history** (`chats` collection): kept indefinitely by default. The
  code deliberately excludes chat history from automatic deletion
  (`app/database/mongodb.py`'s retention-collections list, with the comment
  "that IS the user's own conversation history... expiring it out from
  under them would be data loss, not data hygiene"). A published policy
  must state plainly that conversations are retained until the user deletes
  them.
- **Analytics** (`query_logs`, `intent_events`): retained per
  `ANALYTICS_RETENTION_DAYS` (`app/core/config.py`), **default 0, meaning
  kept forever** unless the deploying operator sets a positive value. If
  launch requires a bounded analytics retention window, that must be set
  explicitly before go-live -- it is not on by default.
- **Uploaded documents and generated drafts**: retained until the user
  deletes them (see below) or an administrator does so.

## 3. Deletion rights, as actually implemented

Two distinct, real user-facing workflows exist today
(`app/chatops/workflows/account.py`, `app/services/user_data.py`):

- **"Delete this conversation"**: erases one session's chat history and
  associated logs only. Drafts and cases from that session are untouched.
- **"Delete my data" / "erase my data"** (also recognized in Hindi/Hinglish):
  requires the user to type the exact confirmation phrase, not a bare yes.
  Deletes chats, query logs, intent events, feedback, drafts and all draft
  versions, cases, preferences, form workflows, background jobs, and
  exported files on disk -- and reports per-collection deleted counts back
  to the user.
  - **Explicit limitation, stated in the product's own confirmation
    message**: this does **not** delete the user's login/account itself.
    An account deletion request today requires an administrator, since no
    self-service "close my account" flow exists. A published policy must
    disclose this gap rather than imply full self-service erasure.

## 4. NOT YET IMPLEMENTED -- open items for legal counsel and product

None of the following exist anywhere in the codebase as of 2026-09-07. Each
needs either a real implementation, a documented policy decision, or both
before a privacy policy referencing them could be accurate:

- **Consent capture.** There is no consent checkbox, banner, or recorded
  consent event anywhere in the frontend (`streamlit_app/`) or backend. If
  the target jurisdiction(s) require affirmative consent before processing
  (e.g. for analytics or third-party LLM calls), that flow does not exist
  and must be built, not merely documented.
  Depends on: [[source-verification-checklist]] area of legal ops if the
  same reviewer handles both.
- **Cookie policy.** No cookie-consent mechanism was found. If the
  Streamlit frontend or any future web frontend sets cookies, a cookie
  policy and consent flow are both outstanding work.
- **Third-party data-sharing disclosure.** The product calls out to
  external LLM providers (Groq, Gemini, OpenAI, Claude, DeepSeek --
  whichever `LLM_PROVIDER`/`LLM_FALLBACK_PROVIDERS` select in
  `docker-compose.production.yml`). User questions and uploaded-document
  text are sent to whichever provider is active. A published policy must
  name the actual provider(s) in production use and describe what data
  reaches them -- this draft cannot do that because provider selection is a
  deployment-time configuration choice, not a fixed fact of the code.
- **Self-service account closure.** As noted in section 3, only an
  administrator can currently close an account. Decide whether launch
  requires a self-service path, or whether "contact an administrator" is
  acceptable to state in the policy.
- **A finalized retention window for analytics.** Section 2's default is
  "forever" -- someone (product + legal) must pick a real number before the
  policy can state one.

## 5. What this draft is not

This is not a substitute for `docs/SOURCE_VERIFICATION_CHECKLIST.md` (legal
*content* verification) or `docs/ADMIN_RUNBOOK.md`'s new TLS/secrets section
(operational security) -- it covers only user-data privacy/consent/retention
commitments. All three should be reviewed together before launch, since a
privacy policy that promises data handling the infrastructure doesn't yet
enforce (e.g. TLS everywhere, encrypted backups) would itself be a
compliance risk.
