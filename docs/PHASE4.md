# Phase 4 implementation

Phase 4 extends the Phase-3 official-source acquisition engine
(`app/services/kb_automation.py`) with a real State/UT/High-Court identity,
closes three operational gaps found while extending it, and reports the
result as a per-jurisdiction coverage matrix and admin dashboard. It does not
change any Phase 1-3 API contract; every new route is additive and
`require_admin`-gated like the rest of `/admin/phase3/*`.

## Components

- **Jurisdiction-aware candidates**: `CatalogueConfig.jurisdiction_code` and
  `applicable_state_codes` (a High Court often covers several States/UTs from
  one portal). A discovered State/UT candidate now lands with
  `applicability: specific_states` and the real state code instead of
  `unknown` — `verification_status` still starts `unverified`; this only
  supplies a structural fact the adapter already has evidence for.
- **Manual-access exceptions**: `ManualAccessRequiredError`, raised on a
  CAPTCHA/login-wall marker, HTTP 403/429, or an explicit access-denied page.
  Routed to a distinct terminal job status `manual_access_required` (not
  `quarantined`) — retrying a CAPTCHA wall on a timer cannot succeed, and
  `retry()` explicitly refuses this status. An admin fetches the document by
  hand and uploads it through the existing knowledge-gap source flow instead.
- **Portal layout-change detection**: `PortalLayoutChangedError`. A
  single-URL catalogue adapter that previously discovered candidates and now
  discovers zero from an otherwise-successful fetch is flagged distinctly
  (`kb_adapter_layout_changed`) rather than silently recorded as a clean,
  empty run. Multi-URL paginated adapters (e.g. India Code's offset pages)
  are excluded — their per-page counts vary legitimately.
- **Confidence scoring**: `app/rag/source_confidence.py::score_candidate` —
  deterministic points for a real date, an Act number, a `.pdf` link, and a
  matching official domain. Advisory review-queue ordering only
  (`list_jobs(order_by_confidence=True)`); it never touches
  `verification_status`.
- **Amendment/repeal classification**: `SourceCandidate.change_type`
  (`amendment` / `repeal` / `ordinance` / `unclassified`), from a
  deterministic English/Hindi keyword match on the listing text. Advisory
  only — `amends`/`supersedes` linkage stays a human decision through
  `MonitorReviewRequest`.
- **State and High Court adapters**: `app/services/kb_state_adapters.py`
  (real: Uttar Pradesh Acts/Ordinances, over the already-verified
  `upvidhai.gov.in` listings) and `app/services/kb_high_court_adapters.py`
  (framework only — zero High Courts registered; no High Court portal URL has
  been verified in this codebase, and this project does not invent official
  URLs — see `docs/SOURCE_GOVERNANCE.md`).
- **Coverage matrix stage extension**:
  `KnowledgeBaseCoverageService.merge_automation` folds discover/download/test
  counts onto the existing catalogued/indexed/approved matrix. A
  jurisdiction's `complete` flag requires a passing benchmark only when one
  actually ran (`tested_total > 0`) — a jurisdiction reached purely through
  ordinary human KB upload isn't penalized for a benchmark it was never
  eligible for. `manual_access_exceptions` lists jurisdictions with an open
  CAPTCHA/access-blocked job, reported rather than hidden.
- **Monthly reconciliation**: `scripts/reconcile_state_coverage.py` — read-only,
  classifies every incomplete jurisdiction by the furthest stage it reached
  (`tested_failing`, `manual_access_exception`, `downloaded_not_indexed`,
  `discovered_not_downloaded`, `not_configured`), writes a timestamped JSON
  report under `storage/operations/state_coverage_reconciliation/`. Never
  mutates anything — approval/verification stay admin actions.
- **Ops dashboard**: `GET /admin/phase3/state-coverage` composes the
  coverage matrix, automation/adapter health (including the dead-letter
  counts), the open manual-access-required jobs, and recent layout-change
  alerts in one response. `streamlit_app/coverage_admin_page.py` renders it as
  three tabs (By stage / Exceptions / Complete), reachable from the admin
  sidebar next to the existing KB Review page.

## Honest scope limit

Only Uttar Pradesh has a verified, currently-monitored state portal. No High
Court URL has been verified in this codebase. The coverage matrix reports the
other 27 States/UTs and every High Court as `catalogued_only` /
`not_configured` — exactly like the existing MP-404-disabled precedent in
`docs/LAW_UPDATE_MONITORING.md` — rather than claiming coverage that does not
exist. See `docs/PHASE4_PROGRESS.md` for the exact count.

## Testing

```bash
pytest -q tests/test_kb_automation.py tests/test_kb_state_adapters.py tests/test_kb_coverage.py tests/test_law_monitoring.py tests/test_state_coverage_admin_route.py
pytest -q
```
