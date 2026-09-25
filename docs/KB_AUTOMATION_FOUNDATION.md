# KB automation foundation

The automation worker discovers reviewed manifest entries, persists one job per
source/version, downloads official PDFs and passes them through the existing
failure-safe KB ingestion pipeline. It is disabled by default.

## Production setup

1. Review `config/kb_sources.json`. Only HTTPS `.gov.in` and `.nic.in` URLs are
   accepted. Increment `version` when an official source publishes a new legal
   version.
2. Run `python scripts/create_indexes.py` during deployment.
3. Ensure `clamscan` has current signature databases. The image contains the
   scanner; signature updates remain an infrastructure operation. Missing or
   unusable scanning fails closed and the job enters retry/quarantine.
4. Set `KB_AUTOMATION_ENABLED=true` on exactly one scheduler instance. The
   durable Mongo lease makes interrupted jobs reclaimable, but a single
   scheduler avoids unnecessary duplicate portal traffic.
5. Inspect `GET /admin/phase3/kb-automation/status` and the jobs endpoint.

## Flow and gates

`manifest discovery -> durable job -> official URL/DNS/redirect checks ->
bounded PDF download -> MIME/magic/EOF validation -> SHA-256 -> malware scan ->
canonical identity/version lookup -> staging -> OCR/parser/chunker -> canary
chunk validation -> atomic activation or rollback`

Downloaded documents use `verification_status=unverified`. Successful indexing
therefore ends as `quarantined`: chunks exist but the shared retrieval gate
cannot return them. A separate machine policy or human legal review must prove
applicability and commencement before publication. This prevents an official
PDF from being treated as currently binding law merely because it downloaded
successfully.

## Admin endpoints

- `POST /admin/phase3/kb-automation/run?limit=20`
- `GET /admin/phase3/kb-automation/status`
- `GET /admin/phase3/kb-automation/jobs?status=quarantined`
- `POST /admin/phase3/kb-automation/jobs/{job_id}/retry`

Retry is available for transient/security failures. A successfully indexed
document waiting for legal verification is not replayed; it must go through the
existing KB review/publication workflow.
