# Official law update monitoring

This worker detects changes to explicitly configured official HTTPS URLs. It
does not crawl all Indian law, interpret changes as commencement, or approve
content automatically. Existing documents and historical versions are retained.
No sources or production jobs are enabled merely by deploying this code.

## Local activation — 5 September 2026

`config/law_monitors.json` is registered in the configured database. Three
monitors are enabled: MHA's new-criminal-laws table, UP's Acts listing first
page, and UP's ordinance listing. All check every 24 hours. MP Code's official
landing URL returned HTTP 404 and is registered **disabled**; MP legislation
monitoring is not active. This is limited initial coverage, not all India.

Windows task `LawChatbot-LawMonitor` runs the hidden
`scripts/check_law_monitor.ps1` every 15 minutes to process due checks. It runs
as the current limited user while logged in; the PC must be on, with network
and MongoDB available. This is local activation, not a production-server
deployment. Redis was restored by starting the installed Docker backend.

The same task runs conservative KB machine verification after monitoring.
Verification is rate-limited to once per 24 hours and currently covers BNS and
BSA only. It requires byte-for-byte equality with the configured MHA Act PDF,
mandatory identity/applicability text, and a matching official Gazette
commencement notification. Failure after publication automatically returns the
document to `needs_review`. BNS's chunk containing the uncommenced section
106(2) is excluded as a whole because the legacy index is not subsection-aware.
Machine-verified answers disclose that no human legal review occurred.

BNSS, Consumer Protection, NI Act, IT Act and RTI remain excluded until an
equally strict automated policy can resolve their version, commencement and
applicability constraints. Never broaden `machine_verified` to mean merely
"download succeeded" or "official-looking domain".

Live verification: three initial baselines, then three unchanged repeat fetches,
three total pending-review events, valid raw snapshot hashes, zero publications.
The scheduled task completed with exit code 0. Evidence is in
`storage/law_monitor/live_verification.json`; task output is appended to
`storage/law_monitor/worker.log`.

```powershell
Get-ScheduledTaskInfo -TaskName 'LawChatbot-LawMonitor'
Start-ScheduledTask -TaskName 'LawChatbot-LawMonitor'
Disable-ScheduledTask -TaskName 'LawChatbot-LawMonitor'
```

Reconfigure explicitly with `python -m scripts.configure_law_monitors
config/law_monitors.json --actor YOUR_OPERATOR_ID`.

Optional `content_selector` hashes only a reviewed HTML section, so visitor
counters outside legal tables do not generate alerts. Missing/empty selected
content fails the check instead of accepting a blank baseline. `checksum` is
the selected-content fingerprint; `snapshot_checksum` is the SHA-256 of the
full raw snapshot. Keep both for audit. These listing monitors detect changed
listing content/links, not edits inside unchanged linked PDFs or all pagination.

## Configure and run

All endpoints below require existing admin authentication, under `/admin/phase3`.

1. `PUT /law-monitors` with `title`, `url`, `topic`, optional `state_code`,
   `interval_hours` (1–168, default 24), and `enabled`. The URL identifies the
   monitor; the same request updates it. Only HTTPS `.gov.in`/`.nic.in` sources
   are supported. The administrator must verify that the page belongs to the
   relevant issuing authority; domain validation alone is not legal verification.
2. Run `python -m scripts.run_law_monitor --once` from the project root, or run
   `python -m scripts.run_law_monitor` under a supervised service. The worker
   provisions indexes idempotently. `--once` can be invoked by an external
   scheduler. Do not launch a new worker inside every web process.
3. `GET /law-monitors/coverage` shows configured state/topic monitoring, errors,
   overdue checks and pending reviews. `POST /law-monitors/{id}/check` requests
   one immediate check. First successful fetch creates a baseline review too.
4. `GET /law-updates?status=pending_review` lists detected changes; `GET
   /law-updates/{id}/snapshot` downloads the exact bytes and the list exposes
   their SHA-256. Snapshots never enter the answer KB automatically.

Example monitor (verify the target page before enabling it):

```json
{"title":"Relevant department notifications", "url":"https://department.gov.in/notifications", "state_code":"MP", "topic":"property", "interval_hours":24, "enabled":false}
```

This is a placeholder URL, not an installed live source. Register official
notification/Act listing pages to notice new links, and individual PDFs to
notice changed bytes. A listing change is queued for manual link discovery;
this release does not automatically follow newly discovered links. Dynamic
timestamps/navigation changes can produce false-positive review items.

## Review and publish

Compare the snapshot and issuing authority's actual instrument. Determine
publication and effective dates separately; check affected sections, state/local
applicability and transition/savings provisions. Upload any new official document
through the existing admin KB upload flow with unverified metadata first.

`POST /law-updates/{id}/review` accepts:

```json
{
  "decision": "publish",
  "notes": "Describe the instrument, affected sections and version comparison.",
  "evidence_url": "https://department.gov.in/official-instrument.pdf",
  "affected_versions_reviewed": true,
  "publications": [
    {"document_id": "existing-indexed-id", "jurisdiction_metadata": {}}
  ]
}
```

Replace the empty object with the complete Phase 1 jurisdiction metadata. The
endpoint requires approved-quality applicability, official source URL and an
effective-from date. It stamps the authenticated reviewer and actual review time;
it does not accept a caller's claimed reviewer identity. Include full corrected
metadata for affected older documents in the same plan (including section
overrides and amendment links). `effective_to` is inclusive in current retrieval:
when an old version stops at midnight before a successor takes effect, set the
old end to the prior day. Do not expire an entire Act for a section-only change.
Dates/relationships are explicit reviewer decisions, not inferred from detection.
Future-effective versions remain subject to Phase 2 date filtering.

Use `decision: "dismiss"`, evidence URL and notes for page-only changes, with
an empty publications list. Dismissal does not verify a law or refresh its legal
verification date. Last successful monitoring check is also not legal review.

Publication reuses the metadata correction service: no re-embedding, propagation
to chunks, and shared generation invalidation for both caches. Each API worker's
keyword index rebuilds from Mongo when it observes a new generation, so dates
and approval metadata cannot remain stale in its in-memory/disk corpus. If Redis
cannot provide freshness, the keyword leg falls back to the vector leg. Monitored
publication requires successful cache invalidation before and after changes;
Redis failure is not treated as successful publication. Affected documents
are quarantined before publication. Failures attempt to re-quarantine the whole
plan and retain `publish_failed`; retry after inspecting and fixing the cause.
Review inputs, actor, timestamp and document IDs remain in the durable event.

## Operational limits and recovery

- Monitor only configured URLs: absence of changes is not proof of current or
  complete nationwide coverage. Add every supported state/topic's official
  listings and inspect missing coverage before claiming support.
- Requests have a total deadline, public-address check, 4 MiB decoded-byte cap,
  no redirects or inherited proxies. Redirects/CAPTCHA/unsupported bodies fail
  visibly and require a corrected URL or manual checking. Use outbound network
  controls as well: DNS is checked before HTTPX resolves the host for connection.
- Errors preserve the last successful baseline and use exponential retries;
  successful unchanged checks never reset legal verification dates.
- Publication spans multiple Mongo records and is not an atomic database
  transaction. Serialize admin publications for overlapping documents. A process
  crash can leave an event `publishing`; inspect its saved plan, quarantine the
  affected documents through the existing metadata endpoint, then have an
  operator reset the event to `publish_failed` for explicit retry. Previous
  metadata is retained on the event for rollback. Inspect any
  `quarantine_failures` immediately. Never mark a failed event published manually.
- To roll back metadata, reapply the previously reviewed metadata through the
  existing jurisdiction correction endpoint (which invalidates caches). Keep
  original documents and snapshots. Monitoring snapshots have no automatic
  retention deletion; size storage according to configured sources/check cadence.
- This framework does not automatically consolidate amendments into rewritten
  Act text or decide transition/savings applicability. Those remain reviewed
  source/metadata work, with Phase 2 historical limitations retained.

Technical references: [HTTPX streaming and redirect behavior](https://www.python-httpx.org/quickstart/),
[HTTPX timeouts](https://www.python-httpx.org/advanced/timeouts/).

## Configured official-document acquisition

`python -m scripts.sync_official_kb_sources` downloads the explicitly configured
priority PDFs, checks PDF bytes and document identity, and sends new content
through the normal shared-KB ingestion pipeline. Exact canonical files are an
idempotent no-op. The scheduled worker runs this command with
`--if-due-hours 24`, so the 15-minute monitor task performs acquisition at most
once per day.

Acquisition proves source provenance and document identity only. New documents
therefore remain `needs_review` unless a separate law-specific machine policy
also proves applicability, commencement and relevant exclusions. A timeout,
404, non-PDF response or failed identity token is reported as `held` and never
indexed. The latest run is written to
`storage/kb_audit/official_source_sync_latest.json`; admins can inspect live
document/chunk/publication status at `GET /admin/phase3/official-sources/coverage`.

## Phase 4: State/UT catalogue adapters, exceptions and reconciliation

`GET /admin/phase3/state-coverage` composes the per-jurisdiction discover/
download/index/test matrix (`app/services/kb_coverage.py`), adapter/dead-letter
health, open manual-access exceptions and recent layout-change alerts in one
response; `streamlit_app/coverage_admin_page.py` renders it as three tabs
(By stage / Exceptions / Complete) from the admin sidebar.

**Manual-access exceptions.** A portal behind a CAPTCHA, login wall, or an
explicit HTTP 403/429 block is never put on the retry/backoff schedule --
retrying it on a timer cannot succeed. It lands in a distinct terminal job
status, `manual_access_required`, alerted as `kb_manual_access_required`.
`POST /admin/phase3/kb-automation/jobs/{job_id}/retry` will not requeue one of
these (its query only matches `retry`/`quarantined`); resolve it by fetching
the document by hand and uploading it through the existing
`POST /admin/phase3/knowledge-gaps/{message_id}/source` flow instead. This is
the intended, and only intended, place manual downloading happens -- a portal
that is not CAPTCHA/access-blocked should progress through automation, not a
human re-downloading it routinely.

**Portal layout-change alerts.** A single-URL catalogue adapter that
previously discovered candidates and now discovers zero from an otherwise-
successful fetch is flagged as `kb_adapter_layout_changed` rather than
silently recorded as a clean, empty run -- almost always the portal was
redesigned and the adapter's selectors no longer match. Multi-URL paginated
adapters (e.g. India Code's offset pages) are excluded, since their per-page
counts vary legitimately. Check `adapters[].status == "layout_changed"` in the
coverage response and update the adapter's `include_pattern`/markup
assumptions in `app/services/kb_central_adapters.py`,
`app/services/kb_state_adapters.py` or `app/services/kb_high_court_adapters.py`.

**Adding the next State/UT or High Court adapter.** Per `docs/SOURCE_GOVERNANCE.md`,
verify the portal URL manually first (an admin actually opening the page and
confirming it belongs to the issuing authority -- domain validation alone is
not verification, exactly as this document already states for law monitors).
Only then add a `CatalogueConfig` with the real `jurisdiction_code` to
`app/services/kb_state_adapters.py::state_source_adapters()` or
`app/services/kb_high_court_adapters.py::high_court_source_adapters()`. An
unverified URL is not registered at all -- mirroring the existing MP Code
entry above, which is registered but `enabled: false` after returning HTTP 404.

**Monthly full reconciliation.** `python -m scripts.reconcile_state_coverage`
is read-only: it classifies every incomplete jurisdiction by the furthest
pipeline stage it reached (`tested_failing`, `manual_access_exception`,
`downloaded_not_indexed`, `discovered_not_downloaded`, `not_configured`) and
writes a timestamped report under
`storage/operations/state_coverage_reconciliation/`. It never approves,
verifies or re-indexes anything.

## Local activation — 7 September 2026

Windows task `LawChatbot-StateCoverageReconciliation` is registered (hidden
`scripts/run_state_coverage_reconciliation.ps1`, mirroring
`check_law_monitor.ps1`'s pattern), triggered every 4 weeks on Monday at
3am. It was started once manually to verify it end-to-end: exit code `0`,
report written to
`storage/operations/state_coverage_reconciliation/reconciliation_20260907T121558Z.json`,
log appended to `storage/operations/state_coverage_reconciliation/task.log`.
Registered with:

```powershell
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "<repo>\scripts\run_state_coverage_reconciliation.ps1"' -WorkingDirectory '<repo>'
$trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 4 -DaysOfWeek Monday -At 3am
Register-ScheduledTask -TaskName 'LawChatbot-StateCoverageReconciliation' -Action $action -Trigger $trigger
```

```powershell
Get-ScheduledTaskInfo -TaskName 'LawChatbot-StateCoverageReconciliation'
Start-ScheduledTask -TaskName 'LawChatbot-StateCoverageReconciliation'
Disable-ScheduledTask -TaskName 'LawChatbot-StateCoverageReconciliation'
```
