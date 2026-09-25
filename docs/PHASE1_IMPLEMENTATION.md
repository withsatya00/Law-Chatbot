# Phase 1: reliability and security implementation

Scope: existing functionality and repository cleanup, not nationwide collection (Phase 2) or public launch certification (Phase 3). Existing uncommitted work was preserved. No production database records, user documents, KB sources or backups were deleted or migrated.

## Feature inventory and evidence

"Working" below means exercised by the stated tests, not a claim that all real model outputs are correct.

| Feature | Status | Evidence / limitation |
|---|---|---|
| Registration, login, token revocation | Working in isolated live services | `test_phase1_live_services.py` uses actual MongoDB and Redis |
| Chat, intent routing, follow-ups and corrections | Working under regression tests; live-model acceptance pending | `test_multi_turn_conversations.py`, `test_manual_chat_transcript_regressions.py`, `test_post_phase3_hardening.py` |
| Private uploads, document questions/review/comparison | Working under service tests; ownership checked at HTTP boundary | `test_document_service_ownership.py`, `test_document_review_and_comparison.py`, live cross-account session tests |
| Draft creation, edits, lifecycle and export | Working under regression tests; live-model quality acceptance pending | `test_draft_conversation.py`, `test_draft_export.py`, `test_draft_ownership.py`, `test_draft_lifecycle.py` |
| PDF page evidence | Working under real PDF loader/pipeline regression tests | `test_pipeline_page_citation_regression.py`, `test_page_level_evidence.py`; existing indexed sources need deliberate reindex to gain repaired metadata |
| KB reindexing | Working under staged-write, partial-activation and bookkeeping failure tests | Real MongoDB/BM25 rollback and retry tested; publication uses compensating writes, not a cross-system atomic transaction |
| Interrupted reindex recovery | Working through explicit maintenance command | Writers must be stopped before applying; dry-run is the default; crash recovery tested against real services |
| KB queue | Working for the supported single API consumer deployment | Pending/in-flight IDs deduplicated, task restarts, shutdown cancellation and persisted-job recovery; distributed job leasing is a Phase 3 scaling gate |
| Jurisdiction/source governance | Working under regression tests | National source coverage and scheduled freshness acceptance remain Phase 2 work |
| Multilingual conversations/drafts | Working for tested cases | Existing language/benchmark tests; not certification of every language |
| Deployment | Hardened configuration prepared | Production definition uses external private services and one API process; HTTPS ingress and real secret/provider configuration are operator-owned |
| Deprecated prototype | Unused; removed | 10 Git-tracked files, no live source/script/template/deployment references; recoverable from Git history |
| Generated/private/runtime data | Retained | Inventory counts only; no inferred deletions |

## Changes

- Optional Windows native-tool discovery tolerates inaccessible installation directories.
- Readiness returns HTTP 503 when MongoDB or Redis is unavailable.
- Rate limiting returns HTTP 429 from middleware and excludes readiness/liveness probes.
- Production settings reject example secrets and wildcard CORS origins.
- Production uploads require an authenticated owner. `/upload` and `/draft` check session ownership before accessing or mutating memory.
- Upload files use exclusive creation; oversize, disconnect and cancelled writes remove only their own partial file, after closing the handle.
- Version lookup and indexing locks distinguish private owners with identical filenames.
- Failed activation/bookkeeping restores previous searchable chunks before cleanup. The source lock is released after publication bookkeeping. Failed restoration preserves records for recovery rather than deleting potentially serving evidence.
- KB queue coalesces duplicate pending/in-flight jobs and explicitly cancels its consumer before dependencies close.
- Docker development bindings are localhost-only; the runtime is non-root. Build context excludes secrets, private storage and generated clutter. Configuration and test storage isolation are copied into the image.
- Standalone `docker-compose.production.yml` does not publish databases or Streamlit. It expects private authenticated external services, explicit CORS and real secrets.

## Repeatable validation

Completed on 7 September 2026:

- Full Windows suite: **2,479 passed, 9 skipped**. Skips are explicit optional/live-service gates; result XML: `output/phase1-release-results.xml`.
- Disposable MongoDB/Redis integration suite: **6 passed**, including auth/revocation, cross-user denial, reindex rollback/retry and interrupted-process recovery.
- Non-root Linux container suite with networking disabled: **133 passed**, covering PDF/page extraction, draft PDF/DOCX/TXT/RTF export, upload/storage isolation and P0 publication behavior.
- Ruff: changed application/tests/recovery/inventory files clean. Mypy: Phase 1 changed modules clean; full `app` still reports seven pre-existing errors in `kb_jurisdiction.py`, `matter_context.py`, `law_monitoring.py` and `admin_phase3.py`.
- Docker image build, production Compose parsing and UID/GID 10001 runtime check passed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/phase1-tests
.\.venv\Scripts\python.exe -m scripts.phase1_inventory
docker compose -p legal-phase1-test -f docker-compose.phase1-test.yml up -d
$env:PHASE1_LIVE_TESTS='1'
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_live_services.py -q --basetemp=tmp/phase1-live
Remove-Item Env:PHASE1_LIVE_TESTS
docker compose -p legal-phase1-test -f docker-compose.phase1-test.yml down
```

The live suite hardcodes disposable ports 37017/36379, generates its own database name, and drops only that database. It uses deterministic embedding doubles and does not validate a paid LLM or Atlas Search.

File inventory: `output/phase1-inventory.json`. No secret values or private document contents are read. Static import counts do not establish that a module is unused; dynamic imports, route registration, templates and entrypoints must also be checked.

## Deployment and recovery gates

1. Back up data and run `python -m scripts.verify_p0_reindex_migration` read-only. Investigate missing active statuses and legacy ownership metadata before deployment.
2. Create required indexes with the existing `scripts.create_indexes` procedure, including `uniq_reindex_lock`, before allowing concurrent requests. No live index migration was applied in this implementation.
3. For a killed indexing process, stop API and all indexing writers. Run `python -m scripts.recover_interrupted_reindexes` to inspect. Then use `--apply --writers-stopped` to restore previous content, clear abandoned new chunks and release the source lock. Records with missing previous evidence are held for review. Retry the source through its normal reviewed ingestion path.
4. Configure production secrets/endpoints/model settings, TLS ingress, storage permissions and backups. The `storage`/`models` volumes must be writable by UID/GID 10001. Existing root-owned model caches may need an operator-reviewed ownership migration; the code does not change host ACLs.
5. Use `docker compose -f docker-compose.production.yml` as a standalone definition, not merged with development Compose. It starts with separate named volumes: explicitly plan data migration before switching; do not assume existing data appears automatically.
6. Existing running containers retain their old port mappings until deliberately recreated. Editing Compose does not close ports on already running services. This task did not restart those services.
7. Complete live-provider acceptance of login → conversation → follow-up → document question → draft → edit → export using synthetic cases and the intended model. Measure factual accuracy separately from successful HTTP responses.

## Limits that remain explicit

- MongoDB and BM25 cannot be atomically switched together by the current design. A crash during publication can briefly expose both versions; the tested recovery command restores a consistent version. Do not enable multiple API replicas until distributed writer coordination and publication semantics are validated.
- Existing page-less indexed chunks are not repaired without reindexing their sources. Bulk production reindexing was not performed.
- Passing tests do not establish complete legal coverage, current-law accuracy or every-language support.
