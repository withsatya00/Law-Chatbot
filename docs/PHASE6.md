# Phase 6 — Production corpus rollout

Phase 6 turns the acquisition system into a measurable production rollout. It
does not auto-approve legal material or deploy the application.

## Implemented first increment: fail-closed release gate

`GET /admin/phase3/kb-production/readiness` returns `go` only when all of the
following are true:

- MongoDB and Redis are reachable;
- autonomous KB automation is enabled;
- every jurisdiction in `KB_PRODUCTION_REQUIRED_JURISDICTIONS` passes the
  Phase-4 source, adapter, approved-corpus, retrieval-benchmark and
  manual-access gates;
- a completed conversation evaluation exists, is no older than
  `KB_RELEASE_EVALUATION_MAX_AGE_HOURS`, has no recorded failures, and every
  score is at least `KB_RELEASE_MIN_EVALUATION_SCORE`.

The response contains exact blocking gate names and
`jurisdiction_not_ready:<code>` entries. It is read-only and cannot publish,
approve, or deploy anything.

The default release scope is the current verified pilot (`UP,DL,MH,GA,DH`).
Operators expand this comma-separated list as new source batches pass Phase 5.
This prevents an early pilot from being labelled nationwide-ready while still
allowing a controlled production launch.

## Additional controls implemented

- `POST /admin/phase3/kb-automation/run?jurisdictions=UP,DL` bounds discovery,
  change checks, downloads, publication benchmarks and canary rechecks to an
  explicit jurisdiction batch. Invalid codes are rejected.
- `POST /admin/phase3/kb-sources/{name}/pause` stops future discovery without
  deleting source, job or corpus records.
- Published automated documents receive recurring retrieval canaries. A
  regression quarantines their chunks, invalidates response caches and emits a
  `kb_canary_rollback` event.
- Amendments, repeals and commencement notifications cannot pass automated
  publication until an admin records reviewed relationships through
  `POST /admin/phase3/kb-automation/jobs/{id}/relationship-review`.
- `GET /admin/phase3/kb-operations/status` reports overdue legal review,
  relationship backlog, manual-access exceptions, low OCR quality, unhealthy
  adapters and per-jurisdiction/language publication counts.
- `POST /admin/phase3/kb-operations/audit` emits deduplicated backlog alerts;
  the scheduler runs this audit after every acquisition cycle.
- `GET /admin/phase3/kb-benchmarks/jurisdictions` exposes State/UT retrieval
  benchmark totals.
- `scripts/backup_mongo.py` creates checksummed BSON backups and
  `scripts/verify_mongo_backup.py` verifies checksums, BSON decoding and record
  counts offline before a restore drill.
- Strict mypy and Ruff now pass over the complete application/source tree.

## External rollout still required

- verify and activate the official portals that are not yet configured;
- complete human legal review for staged Acts, amendments and commencement
  relationships;
- run the checksummed backup/restore drill against an isolated staging database;
- run a production-network soak and retain the resulting readiness evidence.
