# Phase 3 admin runbook

## Daily checks

Open the admin dashboard and inspect readiness, job/error metrics, failed draft/export events, low-confidence/no-source answers, language failure rates, wrong-intent feedback, and the oldest queued job. Escalate any apparent cross-owner access, external submission, source fabrication, or unredacted sensitive log as a security incident.

## Resolve a knowledge gap

1. Open **Unanswered Questions Queue** and review the masked example question.
2. Find the controlling material on an official or otherwise authoritative source. Check jurisdiction, commencement/effective date, amendments, repeal status, and whether rules/notifications are also required.
3. Upload it from the gap entry. Record title, official URL, jurisdiction, Act, section, and source version. Upload creates `pending_review`, never `verified`.
4. Wait for indexing. Investigate failed staging/index jobs; do not mark the gap resolved merely because upload succeeded.
5. Link the indexed document to the legal source record.
6. Independently verify it, set the last-verified date and current status, and retain audit evidence.
7. Re-run representative questions in every relevant language and run the internal evaluation suite. Mark the gap resolved only after citations point to the expected source.

## Legal update handling

- Review `GET /admin/phase3/legal-sources?stale_only=true` at least monthly and after major legal changes.
- A source is flagged when it is older than the configured review expectation or marked repealed/superseded.
- Never treat the legacy-reference endpoint as an automatic amendment engine. Its small curated map is a discovery aid and always requires verification against current official text and the facts.
- When replacing a source, retain its version/status history, mark superseded material, link the new indexed document, and regression-test citations.

## Evaluation and release gate

Run `POST /admin/phase3/evaluations/run` using a benchmark stored under `tests/benchmarks`. Review every failed dimension, not only the aggregate. Add reviewed multilingual cases for every corrected production incident. Do not place real user facts in benchmark data.

## Queue operations

The `worker` service claims jobs atomically. Check job status and attempts in `background_jobs`. For a failed job, correct the underlying dependency or source record and create a new job; retain the failed record for audit. Never edit a job owner or use a raw filesystem path as a payload.

## Incident response

1. Preserve audit logs, request/trace IDs, relevant versions, and deployment metadata under the incident policy.
2. Contain the affected route/provider without disabling ownership or grounding controls.
3. Rotate exposed credentials and invalidate tokens when required.
4. Notify the security/privacy/legal contacts under the approved incident plan.
5. Add a sanitized regression case, correct the source or routing logic, rerun all safety and ownership tests, and document the release.

## Backup and restore

Use encrypted, access-controlled backups and the repository restore script in an isolated staging environment. Validate document/source links, unique owner indexes, TTL indexes, cases, drafts/versions, jobs, and audit records after restore. Never test a restore by overwriting the only production copy.

## TLS, domain, and secrets

This section exists because neither TLS termination nor a secrets rotation
procedure was documented anywhere in this repository as of 2026-09-07 --
`docker-compose.production.yml` binds the API to `127.0.0.1:8000` only
(line 25), so TLS and the public domain are entirely the deploying
operator's responsibility, not something this codebase provides.

**Required secrets** (compose fails to start without these --
`docker-compose.production.yml:8-13`):

| Variable | Purpose |
|---|---|
| `JWT_SECRET_KEY` | Signs/verifies access and refresh tokens. |
| `SECRETS_ENCRYPTION_KEY` | Encrypts sensitive fields at rest. |
| `MONGODB_URI` | Production MongoDB connection string. |
| `REDIS_URL` | Production Redis connection string. |
| `LLM_PROVIDER` | Primary LLM provider selector. |
| `API_CORS_ORIGINS` | Explicit HTTPS origin allowlist -- never `*` in production. |

**Optional secrets** (`docker-compose.production.yml:14-20`, `.env.example`):
`LLM_FALLBACK_PROVIDERS`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`CLAUDE_API_KEY`, `DEEPSEEK_API_KEY`, `OLLAMA_BASE_URL`, `SENTRY_DSN`,
`OTEL_EXPORTER_OTLP_ENDPOINT`.

**Rotation procedure, per secret:**

1. Generate the new value out-of-band (never in a shell history or ticket).
2. Write it to the deployment secret manager under a new version.
3. Roll the API and worker services so every process picks up the new value
   -- both read secrets from the environment at startup only, per
   `scripts/validate_environment.py`'s own checks; neither hot-reloads.
4. For `JWT_SECRET_KEY` specifically: rotating it invalidates every
   outstanding access/refresh token immediately (no dual-key grace period
   exists in this codebase today) -- treat it as a forced global logout, and
   schedule it accordingly rather than rotating it reactively mid-incident
   unless the key itself is the thing that was exposed.
5. For `SECRETS_ENCRYPTION_KEY` specifically: this repository has no
   re-encryption/key-rolling tool. Rotating it without first decrypting and
   re-encrypting every existing at-rest ciphertext under the new key will
   make that data unreadable. Treat this as **not yet safely rotatable** --
   verify against the actual encryption-at-rest code before attempting it in
   production, and do not rotate it under incident pressure without that
   verification.
6. Never bake any secret into a container image or commit `.env`
   (`.dockerignore`/`.gitignore` already exclude it -- do not override that).

**TLS and domain:** put a reverse proxy (nginx, Caddy, or the platform's
managed load balancer) in front of the API's `127.0.0.1:8000` bind,
terminating TLS there. This repository provides no certificate
provisioning, renewal automation, or domain configuration -- treat DNS,
certificate issuance/renewal (e.g. ACME/Let's Encrypt), and HSTS/security
headers as the operator's setup, tracked outside this codebase, and confirm
`API_CORS_ORIGINS` is updated to the real HTTPS origin before go-live.

---

## Phase 2: source governance and index drift

See [SOURCE_GOVERNANCE.md](SOURCE_GOVERNANCE.md) for the full guide.

| Task | Command / endpoint |
|---|---|
| Check BM25 vs MongoDB drift | `GET /health/index-drift`, or `scriptseconcile_indexes.py` |
| Prune stale index entries | `scriptseconcile_indexes.py --apply` |
| Rebuild the sparse index | `scriptseconcile_indexes.py --rebuild-missing` |
| Backfill governance fields | `scripts\migrate_phase2_governance.py --apply` |
| Review / verify a source | `POST /admin/phase3/legal-sources/{id}/review` |
| Record an amendment | `POST /admin/phase3/legal-sources/{id}/supersede` |
| Governance counts | `GET /admin/phase3/dashboard` -> `source_governance_summary` |

**Escalate immediately** if `stale_private_records` is non-zero: those are
retrievable copies of private uploads MongoDB no longer holds.

**Never** mark a source verified without an evidence URL pointing at the issuing
authority's own publication — the API enforces this, and the enforcement is the
point.

---

## Phase 1: jurisdiction-aware Knowledge Base metadata

See `app/rag/kb_jurisdiction.py` for the field vocabulary and validation rules.

| Task | Command / endpoint |
|---|---|
| Upload with jurisdiction metadata | `POST /admin/knowledge-base/upload` (multipart `file` + optional `jurisdiction_metadata` JSON form field) |
| Correct/publish an already-indexed document | `PUT /admin/knowledge-base/documents/{document_id}/jurisdiction` (JSON body, same shape as `jurisdiction_metadata` above) |
| Backfill the existing corpus (dry run) | `python scripts\backfill_kb_jurisdiction.py` |
| Backfill the existing corpus (apply) | `python scripts\backfill_kb_jurisdiction.py --apply` |

**Explicit approval is required, always — there is no legacy pass-through.**
`review_status` must equal `approved` exactly for a document to reach shared
retrieval; a document with no `review_status` field at all (every KB document
indexed before this phase) is excluded the moment this code is deployed,
whether or not the backfill script has ever been run. Private, owner-scoped
documents are unaffected — see `ChatService._prepare_rag_context`'s shared
vs. ownership branch split. Running `--apply` does not change what is
retrievable; it stamps each excluded document with why
(`review_reasons`, e.g. `applicability=unknown`) so there is a queue to work
through, via either a corrected re-upload or the correction endpoint above —
never guess a State/date to make the count go down.

**Correcting a wrong State or date on an already-indexed document** does not
require re-uploading the file: `PUT /admin/knowledge-base/documents/{document_id}/jurisdiction`
re-validates the full jurisdiction metadata and writes it directly onto the
document record and every one of its chunks (`kb_jurisdiction.
propagate_jurisdiction_metadata`) — no re-embedding, and it never touches
`DocumentQualityChecker`'s content-hash dedup check (a metadata-only change is
not new content). Supplying `verification_status: "verified"` with
`verified_by` set is what actually publishes the document (flips
`review_status` to `approved`) — omitting it, or leaving `applicability`/
`issuing_level` unknown, leaves it `needs_review`.

**Rollback:** the backfill only ever writes the fields listed in
`kb_jurisdiction.JURISDICTION_FIELDS` (plus `jurisdiction_schema_version` and
`section_overrides`) onto `uploaded_documents.metadata` and
`embeddings_metadata.metadata` — never text, chunk IDs, embeddings, or
`document_versions`. To revert a run, `$unset` those same keys on the
documents/chunks named in that run's report (`documents_updated`/
`chunks_updated` in its printed summary; pass `--apply` again after a code fix
to re-run — it only ever touches rows still missing
`jurisdiction_schema_version`, so a partial rollback is always safe to re-apply).

**Never** let an ingestion or backfill path mark AI/regex-inferred metadata as
`verified` — `metadata_provenance="inferred"` caps `verification_status` at
`inferred`, enforced in `kb_jurisdiction.normalize_jurisdiction` regardless of
what the caller asks for.
# Official law update monitoring

For scheduled official-source checks, change snapshots, admin review/publication,
coverage and recovery, see [LAW_UPDATE_MONITORING.md](LAW_UPDATE_MONITORING.md).
Live sources and the worker must be configured explicitly; deploying the code
does not imply nationwide monitoring is active.
