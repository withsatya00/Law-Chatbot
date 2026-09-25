# Phase 4 progress

State/UT coverage infrastructure and the operational gaps it exposed.

```
PHASE 4 PROGRESS
[####################] 100% (infra milestones)
Completed: A, B, C, D, E, F, G
Next:      Human review of the 31 real UP ordinances now queued; re-check
           up_acts (timeout) and bombay_high_court_judgments (unreachable
           from this environment) from an unrestricted network; find a
           correct listing endpoint for delhi_high_court_judgments; onboard
           the remaining 31 State/UT and ~23 High Court portals once real
           verified URLs exist
```

## Nationwide rollout increment (September 2026)

- Added verified latest-judgment adapters for the Delhi High Court and Bombay
  High Court. Bombay candidates bind Maharashtra, Goa, and Dadra and Nagar
  Haveli and Daman and Diu.
- Added an all-State/UT registry. A jurisdiction without a verified adapter is
  reported as `verification_required`; it is never counted as covered merely
  because it is in scope.
- Added `GET /admin/phase3/kb-rollout/readiness`. Each jurisdiction must have a
  verified source, healthy adapters, approved chunks, a passing retrieval
  benchmark, and no manual-access exception before it reports `ready`.
- Added `KB_AUTOMATION_ADAPTER_ALLOWLIST` for controlled canary rollout. An
  unknown adapter name fails service construction instead of silently disabling
  ingestion.
- Updated dashboard discovery links from the retired India Code host to the
  current `indiacode.gov.in` statistics and legislation-count pages. The legacy
  India Code catalogue adapter remains fail-closed until the replacement site
  exposes a stable document catalogue route; portal migration must not be
  treated as successful ingestion.

This increment does not claim nationwide corpus completion. Uttar Pradesh has
State-law adapters; Delhi and Bombay have verified High Court adapters. Every
remaining source needs one official portal verification before activation.

Baseline carried in from Phase 3 / post-Phase-3 hardening (verified, not
re-audited): full pytest suite passing, Ruff clean, strict mypy clean over
`app`. Nothing is committed in this phase.

---

## Milestone A — Jurisdiction-aware candidates

`CatalogueConfig` gained `jurisdiction_code` (default `"IN"`, unchanged
behaviour for every existing Central adapter) and `applicable_state_codes`.
`OfficialCatalogueAdapter._parse` now tags a candidate with its real
jurisdiction instead of hardcoding `"IN"`. `_process_claimed`'s
`normalize_jurisdiction` call now passes `applicability: specific_states` and
the real state code(s) for a non-Central candidate, instead of `unknown`/`[]`
for every automated candidate regardless of what the adapter already knew.
`verification_status` is untouched — still `unverified` until a human reviews
it; per `_review_reasons`, supplying the real applicability only removes one
review-blocking reason among several, never `approved`s a record by itself.

Covered by `test_state_candidate_gets_real_applicability_not_unknown` in
`tests/test_kb_automation.py`.

## Milestone B — Manual-access exceptions and portal layout-change detection

Two real gaps closed:

1. A CAPTCHA/login-wall page was previously quarantined identically to a
   transient network error — nothing told an admin "this one needs a human."
   `ManualAccessRequiredError` now routes it to a distinct terminal status
   (`manual_access_required`), alerted as `kb_manual_access_required`, and
   `retry()` explicitly refuses it (its query only matches `retry`/
   `quarantined`).
2. A redesigned single-URL portal returning valid HTML with zero matched
   links was previously indistinguishable from "nothing new was published
   today." `PortalLayoutChangedError` now flags it distinctly
   (`kb_adapter_layout_changed`) when the adapter previously found candidates.
   Multi-URL paginated adapters (India Code's ~34 offset pages) are excluded
   from this check — their per-page counts legitimately vary, confirmed by
   `test_layout_change_check_skips_paginated_multi_url_adapters`.

Covered by `test_captcha_page_becomes_manual_access_required_not_quarantined`,
`test_retry_refuses_manual_access_required_jobs`,
`test_downloader_treats_403_as_manual_access_required`,
`test_downloader_rejects_captcha_marker_distinctly`,
`test_layout_change_on_previously_productive_single_url_adapter`, and the
pagination-exclusion test above — all in `tests/test_kb_automation.py`.

## Milestone C — Confidence scoring and amendment/repeal classification

New `app/rag/source_confidence.py`: `score_candidate` (0-1, from a real date
match, Act number, `.pdf` link, matching official domain, and title quality —
never from a model) and `classify_change_type` (deterministic English/Hindi
keyword match: amendment/संशोधन, repeal/rescind/निरसन, ordinance/अध्यादेश).
Both are advisory-only: `list_jobs(order_by_confidence=True)` can sort the
review queue by score, but `_process_claimed`'s jurisdiction metadata never
carries `confidence_score` or `change_type` — confirmed by the same
Milestone-A test asserting those keys are absent from the metadata dict.

## Milestone D — State and High Court adapter modules

`app/services/kb_state_adapters.py`: `UttarPradeshActsAdapter` and
`UttarPradeshOrdinancesAdapter`, over the already-verified `upvidhai.gov.in`
listings (same URLs as the existing `config/law_monitors.json` entries — this
complements, not replaces, those content-hash monitors). Registered in
`KnowledgeBaseAutomationService`'s default adapter list alongside the eleven
existing Central adapters.

`app/services/kb_high_court_adapters.py`: the generalized framework
(multi-state `applicable_state_codes`) with **zero** instantiated High
Courts — no High Court portal URL has been verified in this codebase, and per
`docs/SOURCE_GOVERNANCE.md` this project does not invent one. The module
docstring shows the exact shape to fill in once an operator supplies and
verifies a real URL.

**Honest count (updated after the nationwide-rollout increment): 5 of 16
registered catalogue adapters carry a non-Central jurisdiction** —
`up_acts`/`up_ordinances` (UP), `delhi_high_court_judgments` (DL), and
`bombay_high_court_judgments` (MH, plus GA and DH via
`applicable_state_codes`). **2 of roughly 25 High Courts are configured**
(Delhi, Bombay). That covers **5 of the 36 States/UTs**; the other 31 remain
`verification_required` in `app/services/kb_source_registry.py` — correctly,
since no URL exists for them and this project does not invent official URLs.

A live check of `GET /admin/phase3/kb-rollout/readiness` against the running
database (2026-09-07) shows the honest current state plainly:
**`production_ready: false`, 0 of 36 jurisdictions `ready`**. Even the 5
jurisdictions with a registered adapter show `adapter_healthy: false`
(`never_run` — `discover()` has not actually been executed against the live
portal in this environment) and `approved_corpus_present: false` /
`retrieval_benchmark_passed: false` (no real document from any of these
portals has been fetched, reviewed and approved yet). The adapters currently
pass only their unit tests against synthetic HTML fixtures
(`tests/test_kb_state_adapters.py`), not a real network run. Getting a
jurisdiction to `ready` requires, in order: an actual `discover()`/`run_cycle()`
pass against the live portal, a human reviewing and approving what it finds,
and then the automatic retrieval benchmark passing — the first step is a
network action an operator should schedule deliberately, and the second step
cannot be automated at all by design (see `docs/SOURCE_GOVERNANCE.md`).

### Live verification against the real portals (2026-09-07)

Ran `KnowledgeBaseAutomationService.run_cycle` scoped to
`jurisdiction_codes=("UP","DL","MH","GA","DH")` against the actual live
government sites. Result: 2 adapters errored, 2 succeeded but found zero
candidates. Investigating each honestly, rather than accepting the aggregate
number:

- **Real, reproducible defect found and fixed.** The shared
  `OfficialCatalogueAdapter._parse` (used by every catalogue adapter,
  Central/State/High-Court alike) required an anchor's own visible text to be
  non-empty before considering it a candidate. `upvidhai.gov.in`'s real
  Ordinances page links every one of its PDFs through an icon-only anchor
  (`<a href="...pdf"><img alt=""/></a>` — no visible text), so **all 31 real
  ordinances on that live page were silently discarded**, every single run,
  regardless of `include_pattern`. Fixed to require the surrounding row/cell
  text (`record_text`) instead, with the candidate's title falling back to
  that row text when the anchor itself has none. Re-run against the live
  page after the fix: **31 real candidates found** (verified by URL, e.g.
  `https://upvidhai.gov.in/Upload/Ordinance/2025-1.pdf`). This affects any
  future portal using the same common icon-only-download-link pattern, not
  only UP.
- **`up_acts` (`Act-hi.aspx`)**: `ReadTimeout` on two consecutive live
  attempts, while `Ordinance-hi.aspx` on the same domain loads fine — page-
  specific slowness on the UP side, not a code defect. Needs a later re-check
  rather than a timeout bump made without more evidence.
- **`delhi_high_court_judgments`**: the configured URL loads (real HTTP 200,
  real HTML, correct domain) but contains **zero PDF or judgment links** in
  the raw response — inspecting it shows a JS-application shell
  (`ajax`/ ` api/` markers present, no static content) rather than a
  server-rendered listing. The chosen URL is structurally the wrong endpoint.
  This is **not** patched by adjusting `include_pattern` and no replacement
  URL was substituted here, per `docs/SOURCE_GOVERNANCE.md` — inventing one
  would be exactly the fabrication that document forbids. Note this is
  already reported honestly downstream without any code change needed: the
  live `coverage()` state shows this adapter as `last_discovered_count: 0`,
  which `KnowledgeBaseAutomationService.coverage()` reports as adapter status
  `"empty"`, and `rollout_readiness`'s `adapter_healthy` gate is correctly
  `false` for DL as a result — `official_source_verified: true` for the
  domain is not being confused with the adapter actually working.
- **`bombay_high_court_judgments`**: `ConnectTimeout` on two consecutive
  attempts, including to the bare domain root (`https://bombayhighcourt.nic.in/`)
  — this environment could not establish a TCP connection to the host at
  all. Recorded as inconclusive rather than "URL is wrong": this could be the
  real site being down, or a network restriction specific to this sandboxed
  environment. Needs re-verification from an unrestricted network before any
  conclusion is drawn either way.

Net effect: real code defect fixed and verified with live evidence (UP
Ordinances); two adapters have real, specific, honestly-recorded operational
blockers that remain open (not silently worked around); `production_ready`
is still `false` after this — finding 31 real PDFs still only reaches the
`discovered` stage, not `approved`/`tested`, which still needs a human to
review them.

## Milestone E — Coverage matrix stage extension + monthly reconciliation

`KnowledgeBaseCoverageService.merge_automation` (pure, unit-tested without
Mongo — mirrors `from_grouped_chunks`'s existing style) folds
`discovered`/`downloaded`/`tested_passed`/`tested_total`/
`manual_access_required` onto each jurisdiction row, fed by the new
`KnowledgeBaseAutomationService.per_jurisdiction_summary()`. A row's
`complete` only requires a passing benchmark when one actually ran.
`manual_access_exceptions` is reported as its own list rather than silently
folded into "incomplete."

`scripts/reconcile_state_coverage.py` — read-only, classifies every
incomplete jurisdiction by furthest stage reached, writes a timestamped
report under `storage/operations/state_coverage_reconciliation/`.

**Registered and verified 2026-09-07** (with explicit user confirmation, since
provisioning a scheduled task is a system-level change): Windows task
`LawChatbot-StateCoverageReconciliation`, hidden
`scripts/run_state_coverage_reconciliation.ps1` (mirrors
`check_law_monitor.ps1`'s pattern), monthly (every 4 weeks, Monday 3am),
alongside the existing 15-minute `LawChatbot-LawMonitor` task. Started once
manually to confirm it end-to-end: exit code `0`, a real report written to
`storage/operations/state_coverage_reconciliation/reconciliation_20260907T121558Z.json`.
See `docs/LAW_UPDATE_MONITORING.md`'s "Local activation" section for the
registration command and management commands.

Covered by 4 new tests in `tests/test_kb_coverage.py` (benchmark-required-only-
when-run, benchmark-passes, manual-access-exception surfaced without forcing
incompleteness, reconciliation stage classification) plus the 2 pre-existing
tests, all passing.

## Milestone F — Ops dashboard + Streamlit "Complete" tab

`GET /admin/phase3/state-coverage` composes the enriched coverage matrix,
`KnowledgeBaseAutomationService.coverage()` (now reporting
`dead_letter: {quarantined, manual_access_required, oldest_manual_access_required,
layout_changed_adapters}`), the open manual-access-required jobs, and recent
`kb_adapter_layout_changed` events.

`streamlit_app/coverage_admin_page.py` — originally three tabs, now five
(📊 By stage / ⚠️ Exceptions / ✅ Complete / 🚀 Readiness / 🔌 Sources),
following `kb_admin_page.py`'s exact shape and admin-only render guard. Wired
into `streamlit_app/app.py`'s sidebar next to the existing KB Review link. The
two later tabs were added once `GET /admin/phase3/kb-rollout/readiness`,
`GET /admin/phase3/kb-production/readiness`, and the `/admin/phase3/kb-sources`
onboarding lifecycle (register → probe → activate, from
`app/services/kb_source_onboarding.py`) existed with no UI surface at all —
an admin previously had to script raw API calls to use them. "Readiness"
shows both the Phase-4 per-jurisdiction rollout gates and the Phase-6
production-release gates (`app/services/kb_production_release.py`) side by
side; "Sources" exposes the full register/probe/activate lifecycle as a form
plus per-source action buttons, with `authority_confirmed` enforced in the UI
before a registration request is even sent — the same manual-verification
requirement `docs/SOURCE_GOVERNANCE.md` already requires of a human.

Covered by `tests/test_state_coverage_admin_route.py` (route registration and
admin gating; response composition from faked services). The two new tabs
call routes already covered by `tests/test_kb_phase4_rollout.py` and
`tests/test_kb_source_onboarding.py`; no new backend behavior was added here,
only its UI.

## Milestone G — Tests, docs, final gates

### Final gates

Re-verified 2026-09-07, after the nationwide-rollout increment, the
Streamlit Readiness/Sources tabs, the live-portal verification pass, and the
icon-only-anchor parser fix it produced:

| Gate | Result |
|---|---|
| Focused Phase-4 tests (`test_kb_automation.py`, `test_kb_coverage.py`, `test_kb_state_adapters.py`, `test_law_monitoring.py`, `test_state_coverage_admin_route.py`, `test_source_governance.py`, `test_kb_phase4_rollout.py`, `test_kb_source_onboarding.py`) | **84 passed** |
| Keyword regression sweep (`-k "kb_ or coverage or automation or monitor or adapter or rollout or onboarding"`) | **288 passed** |
| Complete pytest suite | **2532 passed, 0 failed, 11 skipped** |
| Ruff (`app streamlit_app scripts tests`) | pass on every file this phase (and this increment) touched; 6 pre-existing findings remain, all in untouched `scripts/kb_audit_phase1.py`/`kb_audit_phase2.py` |
| Strict mypy (`app`) | pass on every file this phase touched, 235 source files checked; 4 pre-existing errors remain, all in the untouched `app/rag/matter_context.py` |
| Live `GET /admin/phase3/kb-rollout/readiness` against the running database | `production_ready: false`, **0/36 ready** — see the honest count and live-verification notes under Milestone D; this is the true current state, not the aspirational one |
| Live `run_cycle` against real UP/Delhi-HC/Bombay-HC portals | 1 real defect found and fixed (see Milestone D); UP Ordinances now discovers 31 real candidates; `up_acts` and `bombay_high_court_judgments` have open, honestly-recorded operational blockers (timeout/connect-timeout) |

No commit was created; the working tree already contains the ongoing Phase 4
work.
