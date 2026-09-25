# Central laws automation

Phase 2 extends the durable acquisition engine with named official catalogue
adapters for India Code, the Legislative Department, eGazette, Supreme Court,
RBI, SEBI, MCA, IRDAI, CBDT, CBIC/GST and selected ministries.

## Scheduled behavior

Each enabled cycle performs four independent stages:

1. **Discovery:** catalogue pages produce stable source identities. Catalogue
   row dates/text form a discovery version; a changed record creates a new job.
   India Code pages rotate through the Central Acts catalogue and resolve item
   pages to actual PDF links. Multi-URL ministry catalogues also rotate through
   a persistent checkpoint.
2. **Change detection:** known official document URLs are periodically fetched
   and hashed. Changed bytes create a new version job linked to the prior job;
   unchanged bytes do not re-index.
3. **Acquisition:** PDF or substantive HTML passes official-domain, public-DNS,
   redirect, size, MIME/signature, malware and checksum gates before the
   existing OCR/parser/chunker/canary activation flow.
4. **Publication benchmark:** acquisition remains hidden as `quarantined`
   while legal metadata is unverified. Once the existing human or strict
   machine-verification workflow approves every chunk, the scheduler runs the
   real hybrid retriever with the document title and source filter. Only a
   returned source changes the automation job to `published`; failures retain
   the prior active KB and store benchmark evidence.

Hindi and English links discovered in the same catalogue record share a parent
identity while retaining their own source URL, checksum, language and version.
Acts, rules, regulations, notifications, circulars/orders and case law retain
their source type and issuing authority in acquisition evidence.

## Operational controls

- `KB_AUTOMATION_ENABLED=true` enables scheduled cycles.
- `KB_AUTOMATION_INTERVAL_MINUTES` controls catalogue discovery.
- `KB_AUTOMATION_SOURCE_REFRESH_HOURS` controls byte-level change checks.
- `GET /admin/phase3/kb-automation/status` reports every adapter's last
  success/failure, checkpoint and discovered count.
- `POST /admin/phase3/kb-automation/run` runs a bounded cycle on demand.

Official portals with CAPTCHA/login or client-only rendering fail into adapter
telemetry or the exception queue. They never bypass validation or silently
count as corpus coverage.

## Verification

- Central automation unit/integration tests: 9 passed.
- Related ingestion, monitoring, retrieval and legal benchmark tests: 79 passed
  with 1 environment-dependent test skipped.
- Full repository regression: 2494 passed, 11 skipped.
- Ruff and targeted Mypy: passed.
