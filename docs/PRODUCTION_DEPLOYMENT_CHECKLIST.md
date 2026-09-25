# Production deployment checklist

## Before deployment

- [ ] Review the threat model, retention period, privacy notice, and incident contacts.
- [ ] Store `JWT_SECRET_KEY`, `SECRETS_ENCRYPTION_KEY`, database credentials, provider keys, signing-key passphrases, Sentry DSN, and OTLP credentials in the deployment secret manager. Do not bake them into images or commit `.env`.
- [ ] Set `ENVIRONMENT=production`, explicit HTTPS CORS origins, production MongoDB/Redis endpoints, request limits, and analytics retention.
- [ ] Run `python scripts/validate_environment.py`; production startup must reject weak JWT/encryption keys.
- [ ] Pin and scan container images and dependencies. Restrict runtime filesystem and network permissions.
- [ ] Run `python scripts/migrate_phase2.py`, `python scripts/migrate_phase3.py`, and `python scripts/create_indexes.py` against a backup-tested staging copy.
- [ ] Run the entire regression suite and `tests/test_phase3.py` benchmark tests.
- [ ] Run PDF rendering tests inside the exact API image and visually inspect English, Devanagari, and RTL samples. Confirm DOCX/TXT/RTF fallback.
- [ ] Verify that current official legal sources are indexed, marked verified, linked to chunks, and show correct “current as of” dates.
- [ ] Resolve or accept all stale/superseded source alerts before release.

## Infrastructure

- [ ] Run API and worker as separate services; scale workers independently.
- [ ] Use managed MongoDB with encryption at rest, TLS, least-privilege credentials, point-in-time recovery, and tested restore procedures.
- [ ] Use Redis authentication/TLS, persistence appropriate to the environment, and memory/eviction alerts.
- [ ] Put the API behind TLS, a WAF/reverse proxy, request body limits, and trusted proxy configuration.
- [ ] Persist `storage/` on encrypted storage; apply malware scanning to uploads and private object ACLs.
- [ ] Send structured logs, metrics, trace IDs, and error reports to access-controlled observability systems. Scrub PII at ingestion and exporter boundaries. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable distributed tracing (`app/observability/tracing.py`); an optional Prometheus + Grafana stack is provided at `docker/docker-compose.observability.yml` (`docker compose -f docker-compose.yml -f docker/docker-compose.observability.yml up`) with a starter dashboard at `docker/observability/grafana/dashboards/`. Its Prometheus scrape target is the API's unauthenticated `/internal/metrics` -- keep it inside the deployment's own network/reverse-proxy trust boundary, never expose it publicly.
- [ ] Alert on readiness failures, high 5xx rate, job backlog/age, export failures, OCR/index failures, latency, no-source rate, and stale verified sources.
- [ ] Exercise backup and restore using `scripts/restore_mongo_backup.py`; record RPO/RTO evidence.

## Release and rollback

- [ ] Deploy to staging, run health checks at `/health/live` and `/health/ready`, then smoke-test login, owner isolation, chat, source citations, voice confirmation, forms, draft review, and every export format.
- [ ] Run `POST /admin/phase3/evaluations/run`; block release if a required dimension regresses.
- [ ] Use a canary or rolling deployment. Watch errors, latency, safety routing, language failure rates, and background queue age.
- [ ] Keep the prior image and reversible migration procedure available. Database migrations are additive/idempotent.
- [ ] Record release version, prompt/model/embedding versions, benchmark run ID, source verification date, and approver in the change log.

## Post-deployment

- [ ] Verify rate limiting, RBAC, audit-log access, account deletion, artifact deletion, and cross-owner denial tests.
- [ ] Confirm that form and voice workflows stop at review/confirmation and never submit externally.
- [ ] Confirm lawyer summaries require the user to choose export/share and trigger no automatic transfer.
- [ ] Schedule periodic source verification, dependency patching, restore drills, access reviews, and benchmark regression runs.

## Phase 2 gates

- [ ] `scripts\create_indexes.py` run (adds the governance indexes).
- [ ] `scripts\migrate_phase2_governance.py` dry-run reviewed, then `--apply`.
- [ ] `scriptseconcile_indexes.py` reports `drifted: false`.
- [ ] `stale_private_records` is `0`.
- [ ] Every source presented as current law has been reviewed with an evidence
      URL; unreviewed sources remain `unverified` and are disclosed as such in
      `currency_notice`.
- [ ] `GET /admin/phase3/dashboard` -> `source_governance_summary` reviewed;
      `repealed_or_superseded` sources have their replacement linked.
- [ ] Benchmark run and its `corpus coverage gaps` list is empty, or each gap is
      an accepted, documented omission.
