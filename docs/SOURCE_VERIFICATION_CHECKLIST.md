# Source verification checklist

Audience: the admin/lawyer reviewer who signs off on legal sources before they
reach shared retrieval. This tracks exactly the same act names as
`storage/kb_audit/master_acts_checklist.json` -- keep both in sync.

## Why this exists

`review_status` on a chunk can only become `approved` (the single value
`LegalRetriever` requires before a chunk is ever returned to a user -- see
`app/rag/kb_jurisdiction.py`'s `REVIEW_APPROVED` comment) when
`verification_status` is `verified` or `machine_verified`. There is no
"provisionally visible but unverified" state -- the codebase deliberately
does not allow one. As of 2026-09-07, a live check of this project's MongoDB
found **0 of 3,139 indexed chunks were `approved`** -- every source ever
ingested was sitting at `needs_review`, meaning the assistant could not cite
any law for any question. That snapshot is out of date: as of 2026-09-21 the Acts in the table below are `approved`
(see the correction there).

## How to review a source

Two paths exist, both documented in `docs/SOURCE_GOVERNANCE.md`:

1. **Machine verification** (`scripts/run_kb_machine_verification.py`) --
   fail-closed: downloads the exact official PDF, byte-hash-matches it
   against the locally indexed file, and checks identity/applicability/
   commencement text tokens. Only publishes (`verification_status:
   machine_verified`) if every check passes. Currently only has policies for
   BNS and BSA (`app/services/kb_machine_verification.py`'s `POLICIES`
   tuple). Extending it to another Act requires: the official Act PDF URL,
   a genuine official commencement-notification PDF URL, and confirming the
   locally indexed file is byte-identical to the official one -- if any of
   those don't exist (see notes below), this path is not available and #2
   is required instead.
2. **Human/lawyer review** -- call `POST /admin/phase3/legal-sources/{id}/review`
   with `verification_status: "verified"`, a real `evidence_url`, and
   `review_notes`. Rejects a future `last_verified_date`. This is required
   for every Act where machine verification cannot apply.

## Status by Act (corrected 2026-09-21)

The 2026-09-07 version of this table was stale and called several indexed Acts
"not present" or "unverified". Everything below was re-checked against live
MongoDB (`embeddings_metadata`, by `metadata.source_document`). The "Verified by"
detail lives in each chunk's `verified_by` / `last_verified_at`; this file does
not record approval and cannot grant it.

| Act | Chunks | Status | Next step |
|---|---|---|---|
| Bharatiya Nyaya Sanhita, 2023 (BNS) | 237 | Approved (machine_verified) | None. ss. 303/304/318 retrievable. |
| Bharatiya Sakshya Adhiniyam, 2023 (BSA) | 79 | Approved (machine_verified) | None. |
| Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) | 405 + 374 (two copies) | Approved (human-verified 2026-09-08 / 2026-09-18) | None for retrieval. s.173 incl. Zero FIR/SP escalation is indexed. A lawyer may still want to rule on consolidated vs as-enacted text. |
| Consumer Protection Act, 2019 | 127 | Approved (human-verified 2026-09-18) | None for retrieval. Staggered commencement remains a legal-policy question. |
| Negotiable Instruments Act, 1881 | 47 | Approved (human-verified 2026-09-08) | None. s.138 indexed. |
| Right to Information Act, 2005 | 47 + 73 (two copies) | Approved (human-verified 2026-09-08 / 2026-09-18) | None. ss. 6, 7, 19 indexed. |
| Indian Contract Act, 1872 | 183 | Approved (human-verified 2026-09-08) | None. s.10 indexed. |
| Code of Civil Procedure, 1908 | 91 | Approved (human-verified 2026-09-18) | None. s.80 indexed; a bridge rule for it was added 2026-09-21. |
| RBI circular: limiting liability of customers (2017) | 8 | Approved (human-verified 2026-09-18) | None. |
| Payment of Wages Act, 1936 (Maharashtra-hosted, State-modified copy) | 42 | Approved (human-verified 2026-09-09) | None. Retained for periods before the Code on Wages applied. |
| Code on Wages, 2019 | 45 | Approved (human-verified 2026-09-18) | None. Already ingested 2026-08-27 as `2589gi_P65_6.pdf` (the registry looked for a different file name, so it read as "never ingested"; fixed with `OfficialSource.kb_filenames`). Preferred central source for unpaid wage/salary retrieval. Its stored file hash differs from the registry PDF fetched 2026-09-18; a reviewer may confirm the indexed edition is the intended one. |
| Hindu Marriage Act, 1955 | 24 | Approved (human-verified 2026-09-18) | None. No longer a gap. |
| Information Technology Act, 2000 | 66 | Approved (human-verified 2026-09-18) | **Original 2000 text only.** Amended provisions (ss. 43A, 66C, 66D, 66E, 67A, 67B, 69A) are not available from it. |
| **Information Technology (Amendment) Act, 2008** | 0 | **Real gap - not indexed** | Scanned official MeitY PDF; see below. |

### Real gap: IT (Amendment) Act, 2008

The only official amended text found is MeitY's `IT_amendment_act2008-1_0.pdf`
(20 pages, image-only, 0 extractable characters). Automated sync cannot read
it, so `scripts/sync_official_kb_sources.py` reports it as `manual_ocr_required`
without downloading it (registry key `IT_ACT_2008_AMENDMENT`, `requires_ocr=True`).

1. `python scripts/ingest_scanned_official_pdf.py --law IT_ACT_2008_AMENDMENT --dry-run`
   OCRs it, checks it identifies itself, and writes an evidence manifest to
   `storage/kb_audit/scanned_ingestion/` (per-page OCR confidence and character
   counts, SHA-256, source URL/type). No database writes.
2. Rerun without `--dry-run` to index it. It is **always** `needs_review` /
   `unverified`: not retrievable until a human verifies it.
3. Human reviewer: compare the OCR text with the page images (the dry run of
   2026-09-21 read section 67B as "678"), then verify via
   `POST /admin/phase3/legal-sources/{id}/review` with a real `evidence_url`.
   This is the amending Act, not a consolidated IT Act; only after this review
   are ss. 43A/66C/66D/66E/67A/67B/69A answerable from the KB.

Tesseract on Windows is not on PATH by default; the script adds
`C:\Program Files\Tesseract-OCR` for its own process only.

### Not gaps: how to tell a gap from a retrieval miss

A refusal for a question about an Act in the table above is a retrieval/bridge
miss, not missing law. `GapAutoFetchService` now checks the KB
(`app/services/kb_presence.py`) before queueing an Act, and closes stale queued
jobs for indexed Acts as `skipped_existing`. Add a new Act's aliases and KB file
names to `KNOWN_KB_LAWS` there (or register it in `kb_official_source_sync.SOURCES`,
with `kb_filenames` if it was uploaded under another name).

## State/UT-level Acts

No state-specific Act has ever been sourced. `app/services/kb_official_source_sync.py`
has zero state-law URLs configured. Launch scope covers all 28 states + 8 UTs
per `storage/kb_audit/master_acts_checklist.json`'s `states_and_union_territories`
section -- every row there is an unstarted placeholder, not a claim that any
sourcing has begun. This is a substantial follow-on body of work (source
discovery + ingestion + review, per state) beyond what this checklist can
close out in one pass.
