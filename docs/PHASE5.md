# Phase 5 — Nationwide source onboarding

Phase 5 removes the code-release dependency from State/UT and High Court
onboarding while preserving the source-governance gates.

## Implemented first increment

An admin can now register any confirmed official `.gov.in`/`.nic.in`
catalogue, probe its real layout, and activate it only after the probe finds at
least one legal document:

1. `PUT /admin/phase3/kb-sources` registers or updates a source as
   `pending_probe`. Updating an active source disables it until it is probed
   again.
2. `POST /admin/phase3/kb-sources/{name}/probe` performs a bounded fetch and
   parses candidates. It does not activate, download, index, or publish them.
3. `POST /admin/phase3/kb-sources/{name}/activate` succeeds only for the exact
   record whose latest probe passed.
4. `GET /admin/phase3/kb-sources` lists rollout state and probe evidence.

Active records are loaded into the existing scheduled acquisition engine.
They inherit its official-domain checks, download limits, malware scan,
deduplication, quarantine, approval, retrieval benchmark, canary publication,
and rollback behavior. Registration, probe, and activation actions emit
attributed operational audit events.

`GET /admin/phase3/kb-rollout/readiness` also consumes active dynamic sources,
so a newly onboarded jurisdiction appears in the nationwide matrix without a
code change. It still does not become `ready` until its adapter is healthy,
approved chunks exist, the retrieval benchmark passes, and no manual-access
exception is open.

## Remaining Phase 5 rollout work

- Verify and register each remaining State/UT Gazette and legislature portal.
- Verify and register the remaining High Court listings.
- Replace the legacy India Code adapter only after the migrated portal exposes
  a stable official catalogue/feed that passes the same probe.
- Run sources in jurisdiction batches and approve their staged documents after
  legal review.

CAPTCHA/login-wall portals remain explicit `manual_access_required` exceptions;
the system does not bypass their access controls.
