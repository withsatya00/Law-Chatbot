# Legal Source Governance, Page Evidence and Index Reconciliation

Phase 2 operator guide. Companion to `docs/ADMIN_RUNBOOK.md`.

---

## 1. What "verified" means here

A source is `verified` **only** when a human compared it against the issuing
authority's own publication and recorded what they compared. Nothing else grants
it:

- a plausible or official-looking filename does not;
- being present in the knowledge base does not;
- `create()` refuses a caller-supplied `verification_status: "verified"` and
  downgrades it to `pending_review`;
- the Phase 2 migration returns any pre-existing `verified` record with no
  evidence to `pending_review` rather than keeping an unauditable claim.

`verification_status` values: `unverified` (default) → `pending_review` →
`verified` | `rejected`.

`status` (legal currency) values: `in_force`, `amended`, `repealed`,
`superseded`, `unknown` (default).

### Reviewing a source

```
POST /admin/phase3/legal-sources/{source_id}/review
{
  "verification_status": "verified",
  "last_verified_date": "2026-01-15",
  "evidence_url": "https://www.indiacode.nic.in/handle/123456789/2000",
  "review_notes": "Section text compared line by line against the gazette PDF.",
  "status": "in_force"
}
```

`verified` requires `evidence_url` **and** `review_notes`, and rejects a future
`last_verified_date`. `rejected` and `pending_review` do not require an evidence
URL — requiring one to record that a source is unusable would discourage
recording it at all.

Every call is admin-only (router-level `require_admin`) and written to the audit
log with the actor id. Reviewer identity is stored as an account id, never an
email.

### Recording an amendment

```
POST /admin/phase3/legal-sources/{source_id}/supersede
{ "superseded_by_source_id": "<id>", "effective_date": "2024-07-01" }
```

The replacement must already be registered. Lineage is written both ways
(`superseded_by` on the old record, `supersedes` on the new one).

---

## 2. Page-level evidence

PDFs are loaded page by page. Each chunk carries `page_number`, `page_start`,
`page_end`, optional `page_range`, and `extraction_method`
(`embedded_text` | `ocr` | `mixed`).

Rules that hold everywhere:

- **A page number is the page's real number in the file, or it is absent.** An
  empty or OCR-failed page keeps its slot, so a failure can never renumber the
  pages after it.
- Non-paginated formats (TXT, DOCX, HTML, RTF, ODT) carry **no** page keys.
  `page_number=None` means "this source has no page identity", never "page 1".
- If the per-page texts do not reconstruct the document text exactly, no page is
  attributed at all rather than a guessed one.
- **Page numbers are never backfilled.** The chunks indexed before Phase 2 never
  had their page identity captured and it is not recoverable from stored text.
  Re-ingest a document if you need page evidence for it.

Responses expose `evidence_pages`. An empty list means no page evidence is
available for that answer — not that the answer is unsourced.

---

## 3. Index reconciliation

MongoDB is the source of truth; the BM25 index is a derived on-disk copy.

```powershell
.venv\Scripts\python.exe scripts\reconcile_indexes.py                    # dry run
.venv\Scripts\python.exe scripts\reconcile_indexes.py --apply            # prune stale entries
.venv\Scripts\python.exe scripts\reconcile_indexes.py --rebuild-missing  # rebuild from Mongo
```

Two deliberately separate actions:

- `--apply` removes only BM25 entries whose chunk id is absent from Mongo. It
  never writes to Mongo, never deletes an uploaded file, never edits ownership
  metadata, and is idempotent.
- `--rebuild-missing` re-tokenizes the whole corpus from Mongo. This is the fix
  for chunks that exist in Mongo but were never added to the index. It is a
  much larger action, so it is opt-in rather than folded into `--apply`.

`GET /health/index-drift` reports the same numbers without writing. Drift is
`degraded`, not a failure — retrieval still answers.

**Watch `stale_private_records`.** A stale entry carrying `owner_user_id` is a
retrievable copy of a private upload that MongoDB no longer has. That is a
privacy consequence, not a ranking one.

Ownership metadata problems are **reported, never repaired**: deciding an
ambiguous chunk is public, or assigning it an owner, is a visibility decision no
automated pass should make.

---

## 4. Source currency in answers

Retrieval **prefers** current, verified sources; it does not filter on them. A
repealed provision is still the correct answer for an incident that happened
while it was in force, so it stays reachable and is simply down-ranked.

`ChatResponse.currency_notice` discloses, in plain words:

- a cited source recorded as `repealed`/`superseded` — "treat as the law that
  WAS in force";
- an `unknown` amendment status;
- an unverified source;
- for a question naming a past date, that the BNS/BNSS/BSA took effect on
  1 July 2024 and are not retrospective, so which law governs depends on when
  the incident occurred.

A verified, in-force source produces **no** notice. If every answer carried one,
readers would stop reading them.

---

## 5. Confidence dimensions

`ChatResponse` reports `retrieval_confidence`, `source_grounding_confidence`,
`model_confidence` and `overall_confidence` alongside the original `confidence`
field, which keeps its exact previous meaning.

**Grounding is a hard ceiling.** `overall` can never exceed
`source_grounding_confidence`, so strong retrieval plus a confident classifier
cannot inflate an answer nobody can check. `confidence_reason` names the
*weakest* dimension, so a reader learns whether to rephrase, upload a document,
or distrust the answer.

`applicable_law` is built from citation metadata, never parsed out of the
generated text — otherwise a model naming a section in prose would be laundered
into a structured field that looks authoritative.

---

## 6. Migrations

```powershell
.venv\Scripts\python.exe scripts\migrate_phase2_governance.py            # dry run
.venv\Scripts\python.exe scripts\migrate_phase2_governance.py --apply
.venv\Scripts\python.exe scripts\create_indexes.py                       # idempotent
```

The governance migration adds missing structural fields only. It never invents
an official URL, a publication date or a verification date, never grants
verification, and never deletes anything.

---

## 7. Benchmark

```powershell
$env:LEGAL_AI_RUN_BENCHMARK=1
.venv\Scripts\python.exe -m pytest tests\test_legal_benchmark.py -q -s
```

Opt-in because it needs a populated MongoDB and loads the ~2.3 GB embedding
model. The dataset-integrity tests run always and guard the benchmark against
the failure that makes benchmarks worthless: expectations quietly relaxed until
whatever the system emits counts as correct.

`expected_section` may only be set for sections on the verified whitelist in
`tests/test_legal_benchmark.py::_VERIFIED_SECTIONS`.

A case whose expected Act is absent from the corpus is reported as a **corpus
coverage gap** and excluded from source precision — it cannot test ranking, and
counting it as a precision miss would turn the metric into a measure of
ingestion coverage.
