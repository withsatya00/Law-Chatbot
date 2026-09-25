# Phase 2 Progress

<!-- Machine-readable. A resumed session reads this instead of re-auditing the repo. -->

```yaml
phase: 2
percent: 100
baseline_commit: 8c2cc7f
baseline_branch: phase-1-runtime-stabilization
milestones:
  A_baseline_architecture: done      # 5%
  B_source_governance: done          # 15%
  C_page_level_citations: done       # 20%
  D_index_reconciliation: done       # 15%
  E_source_currency: done            # 15%
  F_retrieval_confidence: done       # 15%
  G_legal_eval_suite: done           # 10%
  H_migration_final_gates: done      # 5%
next_action: "Phase 2 complete. Not committed (no commit was requested). Human review items: re-ingest Hindu Marriage Act + RTI Act; review the 25 indexed sources for verification."
```

## Baseline counts (recorded 2026-09-03, live MongoDB)

| Metric | Value |
|---|---|
| knowledge-base files | 76 |
| Mongo chunks (`embeddings_metadata`) | 2079 |
| BM25 chunks (on-disk index) | **5598** |
| distinct `source_document` values | 25 |
| `documents` collection records | 0 |
| legal-source registry records | 0 |
| verified records | 0 |
| records with `last_verified_date` | 0 |
| chunks with `page_number` | **0** |
| chunks carrying `owner_user_id` | 1505 |

**Confirmed drift:** BM25 holds 5598 chunks against Mongo's 2079 — 3519 stale
entries the index never dropped. This is Milestone D's target; do not "fix" it
by rebuilding blindly, because 1505 of the Mongo chunks are owner-scoped private
uploads and a naive rebuild would decide their visibility.

**No page evidence exists anywhere yet** (`page_number` count is 0), so
Milestone C is new capability, not a repair. Nothing may be backfilled onto the
2079 existing chunks: their page identity was never captured and is not
recoverable from stored text.

## Architecture map (traced, not assumed)

| Concern | Owner |
|---|---|
| Load | `app/rag/loader.py` — `DocumentLoader.load()`; `_load_pdf` joins all pages into ONE string, so page identity is lost at load time |
| OCR | `app/rag/ocr.py` — `OcrEngine.extract_from_pdf_pages()` returns one concatenated string |
| Chunk | `app/rag/chunker.py` — `SectionAwareChunker.chunk()`; `_make_chunk` copies `document.metadata` verbatim |
| Metadata | `app/rag/metadata.py` — `MetadataExtractor` (act/section/provenance) |
| Types | `app/rag/types.py` — `LoadedDocument`, `DocumentChunk` |
| Index | `app/rag/pipeline.py` → `app/rag/vector_store.py` (`MongoVectorStore`, dense) + `app/rag/bm25_index.py` (sparse, pickled to disk) |
| Fuse/rank | `app/rag/fusion.py` (RRF) → `app/rag/reranker.py` (`LegalReranker`) |
| Retrieve | `app/rag/retriever.py` — `LegalRetriever.retrieve()` |
| Citations | `app/schemas/common.py` — `SourceCitation` (already has `verification_status`, `current_as_of`, `amendment_status`; **no page fields**) |
| Statute currency | `app/rag/statute_currency.py`, `app/rag/jurisdiction.py` |
| Source registry | `app/schemas/phase3.py` `LegalSourceMetadata` / `LegalSourceResponse`; `app/services/phase3.py` `LegalSourceService`; `app/repositories/phase3.py` `LegalSourceRepository` |
| Governance API | `app/api/admin_phase3.py` — router-level `Depends(require_admin)`; audit via `AuditLogRepository` |
| Ingestion | `app/services/kb_ingestion_service.py`, `app/rag/incremental.py` |

### Existing governance skeleton (extend, do not replace)

`LegalSourceService` already has `create` / `list` / `verify` / `link_document`,
`SourceStatus` and `VerificationStatus` literals, and admin-only routes with
audit-log writes. Milestone B extends this — it does not build a second registry.

Gaps to close in B: no `official_title` / `issuing_authority` / `publication_date`
/ `checksum` / `ingestion_version` / `supersedes` fields; `verify()` accepts a
verified status with **no supporting evidence**; no reject/supersede operations;
no chunk-id linkage; no indexes or migration.

## Notes

- `LegalSourceService.link_document` already propagates governance metadata onto
  `embeddings_metadata` chunks — reuse this seam for chunk-level linkage.
- `SourceCitation` gained currency fields in an earlier phase; page fields are
  additive and must stay backward-compatible with the 2079 page-less chunks.


## Milestone B — done (+15%)

Files: `app/schemas/phase3.py`, `app/services/phase3.py`, `app/api/admin_phase3.py`,
`scripts/create_indexes.py`, `scripts/migrate_phase2_governance.py`,
`tests/test_source_governance.py` (19 tests).

- Schema gained `official_title`, `source_type`, `issuing_authority`,
  `publication_date`, `checksum`, `ingestion_version`, `supersedes`,
  `superseded_by`, `chunk_ids`, `reviewed_by/at`, `review_notes`, `evidence_url`.
- Default `verification_status` moved `pending_review` -> `unverified`.
- `create()` refuses a caller-supplied `verified`; only `review()` grants it,
  and only with an evidence URL, review notes and a non-future date.
- `supersede()` requires the replacement to be registered; lineage written both ways.
- `link_document()` now records the governed `chunk_ids`.
- Migration is dry-run by default and returns a pre-Phase-2 `verified` record
  with no evidence to `pending_review` rather than keeping an unauditable claim.

## Milestone C — done (+20%)

Files: `app/rag/types.py` (`LoadedPage`, `LoadedDocument.pages`), `app/rag/ocr.py`
(`extract_pdf_pages_individually`), `app/rag/loader.py`, `app/rag/chunker.py`
(`_PageMap`), `app/schemas/common.py` (`EvidencePage`, citation page fields),
`app/schemas/chat.py` (`evidence_pages`), `app/rag/citation.py`,
`app/services/chat_service.py`, `tests/test_page_level_evidence.py` (17 tests).

- `_load_pdf` returns per-page text; `joined_text` is byte-identical to before,
  so no existing chunk boundary or embedding shifts.
- An empty/unreadable page keeps its slot (`extraction_method="none"`), so a
  failure can never renumber the pages after it.
- `_PageMap` verifies the pages reconstruct `document.text` before attributing
  anything; on mismatch it attributes no page at all.
- Non-paginated formats and the 2079 pre-Phase-2 chunks carry no page keys.


## Milestone D — done (+15%)

Files: `app/rag/reconciliation.py` (new), `app/rag/bm25_index.py`
(`remove_chunk_ids`), `scripts/reconcile_indexes.py` (new), `app/api/health.py`
(`/health/index-drift`), `tests/test_index_reconciliation.py` (20 tests).

**Root cause of the 5598-vs-2079 drift:** `BM25Index` had no remove-by-chunk-id
path. `add_or_update_chunks` only adds; `remove_by_source` keys on
`source_document`. Re-indexing a document mints fresh `uuid4` chunk ids, so
every prior generation stayed in the pickle forever.

**Live result (applied):**

| | before | after |
|---|---|---|
| Mongo chunks | 2079 | 2079 |
| BM25 chunks | 5598 | **2079** |
| stale BM25 records | 4097 | 0 |
| missing from BM25 | 578 | 0 |
| stale PRIVATE records | 0 | 0 |
| ownership problems | 0 | 0 |

Two operator actions, deliberately separate: `--apply` prunes only entries
absent from Mongo; `--rebuild-missing` re-tokenizes the whole corpus from Mongo
and is the fix for the 578 chunks the index never received. Dry-run is default
and writes nothing. `/health/index-drift` reports drift without failing.


## Milestone E — done (+15%)

Files: `app/rag/statute_currency.py` (`status_ranking_adjustment`,
`asks_about_a_past_incident`, `currency_notice`), `app/rag/reranker.py`,
`app/schemas/chat.py` (`currency_notice`), `app/services/chat_service.py`
(`_currency_notice_for`), `app/api/admin_phase3.py` (dashboard split),
`tests/test_source_currency.py` (20 tests).

- IPC/BNS, CrPC/BNSS, IEA/BSA mappings already existed; Phase 2 adds the
  retrieval preference and the disclosure.
- Ranking adjustment is bounded (|max| 0.20) and is a PREFERENCE: repealed law
  stays retrievable because it is the correct answer for a pre-July-2024 incident.
- An ungoverned chunk scores 0.0 adjustment — most of the corpus is unverified
  and penalising it would rank by review coverage, not relevance.
- `currency_notice` discloses repealed/superseded, unknown status, and
  unverified sources, and adds a non-retrospectivity warning when the question
  names a past date. A verified in-force source produces NO notice (otherwise
  every answer carries one and readers stop reading them).
- Admin dashboard now separates `stale_sources`, `outdated_sources`,
  `unverified_sources` and a `source_governance_summary` count block.


## Milestone F — done (+15%)

Files: `app/rag/confidence.py` (new), `app/schemas/chat.py`,
`app/services/chat_service.py` (`_confidence_fields`, `_applicable_law_from`,
`_ConfidenceFields`), `tests/test_retrieval_confidence.py` (23 tests).

- Three dimensions reported separately: `retrieval_confidence`,
  `source_grounding_confidence`, `model_confidence`, plus `overall_confidence`.
- **Grounding is a hard ceiling, not a weighted term.** `overall` can never
  exceed `source_grounding_confidence`, so strong retrieval + a confident
  classifier cannot paper over an answer nobody can check.
- `confidence` keeps its exact Phase 1 meaning and every Phase 1 rule that caps
  it still decides the headline number; the dimensions explain it, never raise it.
- `confidence_reason` names the WEAKEST dimension, so a reader learns what to
  do (rephrase / upload / distrust) rather than just that the number is low.
- `applicable_law` is built from citation METADATA, never parsed from the
  answer text -- otherwise a model naming "Section 420 IPC" in prose would be
  laundered into a structured field that looks authoritative.
- Additive only: `risks`/`next_steps`/`applicable_law` default to `[]`.


## Milestone G — done (+10%)

Files: `tests/benchmark/legal_benchmark_v1.yaml` (new, versioned),
`tests/test_legal_benchmark.py` (16 tests), `app/rag/reranker.py`
(colloquial cheque terms).

Metrics (live, 2026-09-03, through retrieve -> rerank, i.e. the real chat path):

| metric | value |
|---|---|
| cases scored | 16 |
| source precision | 0.57 (4/7) |
| section accuracy | 1.00 (1/1) |
| grounding pass rate | 1.00 (16/16) |
| safe-decline coverage | 1.00 (7/7) |
| corpus coverage gaps | 2 |

**Two real findings, neither worked around:**

1. *Colloquial phrasing missed the right section.* "What can I do if a cheque
   given to me bounced?" matched no cheque trigger (the list required the noun
   phrase "cheque bounce"), so NI Act s.138 sat at rank ~12 and never reached
   the top 8 — correct Act, wrong provision. Trigger list widened to the forms
   people actually use; section accuracy went 0.00 -> 1.00.
2. *Two Acts are absent from MongoDB entirely* — Hindu Marriage Act and RTI
   Act. 76 files sit in the knowledge-base directory but only 25 distinct
   sources are indexed. These are reported as `corpus coverage gaps` and
   EXCLUDED from source precision: a case whose Act is not in the corpus cannot
   test ranking, and counting it as a precision miss would make the metric
   measure ingestion while tempting someone to lower the precision floor for
   the wrong reason. **Needs re-ingestion — human action, listed under
   remaining review items.**

The benchmark was also corrected to run `retrieve() -> rerank()` rather than
`retrieve()` alone; measuring only the first stage scores material the user
never sees.

`expected_section` may only be set for sections on the verified whitelist in
`_VERIFIED_SECTIONS`, so an unverified expectation cannot be added silently.


## Milestone H — done (+5%)

Files: `docs/SOURCE_GOVERNANCE.md` (new), `README.md`, `docs/ADMIN_RUNBOOK.md`,
`docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md`, this file.

### Final gates (all run once, 2026-09-03)

| Gate | Result |
|---|---|
| `validate_environment.py --strict` | exit 0 |
| `ruff check app streamlit_app scripts tests` | All checks passed |
| `mypy app` (strict) | Success, 194 source files |
| `pytest` (full) | **1767 passed, 0 failed, 3 skipped** |
| benchmark | precision 0.57, section 1.00, grounding 1.00, safe-decline 1.00 |
| `reconcile_indexes.py` dry-run | `drifted: false` |
| `migrate_phase2_governance.py` dry-run | 0 records needing update |

Working tree contains Phase 2 changes only; no secrets, generated files or test
artifacts are staged. **Not committed** — no commit was requested.

## Remaining human legal-review items

1. **Re-ingest two Acts.** Hindu Marriage Act 1955 and Right to Information Act
   2005 are in the knowledge-base directory but absent from MongoDB. 76 files
   present, 25 indexed.
2. **Review the 25 indexed sources.** All are `unverified`; none may be
   presented as verified law until a reviewer records an evidence URL. This is
   working as designed, not a defect.
3. **Link replacements for repealed sources** once IPC/CrPC/Evidence Act records
   are registered, so `supersedes`/`superseded_by` lineage is complete.
