# Phase 2 implementation status

Updated: 2026-09-07

This document tracks the expanded Phase 2 requested after the original
retrieval/source-governance milestone.  A configured source is not treated as
downloaded coverage, and an indexed source is not treated as verified law.

## Implemented and verified

- Source governance records canonical identity, versions, hashes, official
  links, effective dates, human verification evidence and supersession.
- Retrieval combines dense vectors and BM25, uses legal reranking, preserves
  exact section/article candidates, checks jurisdiction/date/currency and
  returns page-level evidence when the source provides real pages.
- Large statutory sections now use boundary-aware splits.  Every continuation
  carries a deterministic `parent_section_id`, heading, part index/count and
  inherited section/article provenance.
- Legal answers expose separate retrieval, grounding and model confidence;
  missing or conflicting evidence can trigger a safe decline instead of an
  unsupported answer.
- Multi-turn follow-ups, corrections, topic changes, code switching,
  transliteration, draft collection/edit/export and document comparison have
  regression coverage in the existing test suite.
- Conversation memory now stores `confirmed_facts`, `assumptions` and
  `missing_details` separately.  Corrections retain an audit trail.  Owner
  checked APIs let a user inspect, update and delete individual facts.
- The admin coverage matrix includes Central plus every 28 State and 8 Union
  Territory entry.  It reports `catalogued_only`, `indexed_needs_review` and
  `approved` separately, with unmapped legacy chunks visible.

## Operator APIs

- `GET /admin/phase3/knowledge-base/coverage`
- `GET /admin/phase3/official-sources/coverage`
- `GET /admin/phase3/law-monitors/coverage`
- `GET /session/facts?session_id=...`
- `PATCH /session/facts?session_id=...`
- `DELETE /session/facts/{confirmed|assumption}/{key}?session_id=...`
- `PUT /session/missing-details?session_id=...`

## Verification

- Full automated suite: **2484 passed, 9 skipped**.
- Focused hierarchy, coverage and memory tests: **10 passed**.
- Ruff on all files changed for this increment: **passed**.

## Work that requires corpus operations or expert review

Phase 2 is not operationally complete merely because the software paths exist.
The coverage endpoint must be run against production MongoDB and its matrix
must drive these remaining tasks:

1. Discover each Act, rule, amendment and notification from the configured
   official catalogues; add state Gazette/department sources only after their
   domains and document identity rules are verified.
2. Download and hash the files, ingest them into `needs_review`, resolve exact
   duplicates and link new versions to the canonical document identity.
3. Have a qualified reviewer approve applicability, commencement, amendment
   status and source identity.  Unreviewed documents remain outside shared
   retrieval.
4. Expand the versioned lawyer benchmark by jurisdiction and topic, then record
   retrieval precision, section accuracy, citation support and safe-decline
   results.  The last live benchmark had source precision **0.57 (4/7)**, so
   the lawyer-reviewed quality gate is not yet met.
5. Run every advertised language through input, retrieval, follow-up, drafting
   and export tests, and publish a per-language result.  Unit coverage alone is
   not a language certification.
6. Benchmark a learned semantic reranker against the current hybrid retrieval
   and heuristic legal reranker before enabling it.  Model downloads and a
   production latency budget are deployment decisions; no unbenchmarked model
   is enabled by this change.

India Code and the Legislative Department catalogue are discovery inputs, not
proof that the local KB contains every law.  The matrix remains `complete=false`
until every jurisdiction has indexed documents and all of its indexed chunks
are approved; topic-level completeness still needs the external inventory and
lawyer benchmark described above.
