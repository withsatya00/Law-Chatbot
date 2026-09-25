# Phase 3 implementation

Phase 3 adds production-oriented intelligence and operations without changing the Phase 1 or Phase 2 API contracts. New APIs are additive. Existing safety prompts, grounded citation validation, owner checks, draft fact audits, contradiction blocks, and jurisdiction warnings remain active.

## Components

- Admin knowledge dashboard: question frequency, no-source questions, low confidence, language failure rate, intent corrections, feedback trends, missing documents, operational failures, and stale legal sources.
- Legal source registry: jurisdiction, Act, section, version, effective/update/verification dates, amendment status, owner, indexed document, and verification status.
- Legacy reference helper: a deliberately small curated IPC/CrPC to BNS/BNSS map. Unknown references are never guessed and every mapping requires verification against the current official text.
- Voice workflow: existing multilingual STT/TTS now marks voice-created drafts as unconfirmed. Final export is blocked until the owner reviews the facts and explicitly confirms them.
- Explanation modes: `simple`, `detailed`, and `advocate`, with identical citation and safety requirements.
- Proactive follow-ups: one missing question at a time, deduplicated against chat, documents, structured facts, and previously asked fields. English, Hindi, and Hinglish are supported.
- Form assistance: owner-scoped police, cybercrime, RTI, and consumer complaint pre-fill records. Review and explicit confirmation are mandatory; no external submission is implemented.
- Safe preferences: language, explanation level, voice output, and export format only. Case facts remain in owner-scoped cases and drafts.
- Evaluation: a versioned 20-case English/Hindi/Hinglish conversation benchmark. Inputs and expectations are stored, while language and routing observations are produced by the real detector and registered workflow matcher. It reports routing accuracy, language accuracy, and ordinary-question workflow safety; it cannot self-score from fixture-supplied observations.
- Operations: durable MongoDB jobs for ingestion/OCR/indexing hand-offs, exports, and analysis; a separate worker; structured request/trace IDs; metrics; audit logs; health/readiness endpoints; rate limits; environment validation; Docker deployment.
- UX: one chat-first interface with no separate feature pages, plus mobile styling, conversation search, inline workflow progress/downloads, and categorized feedback. The 29 registered workflows are invoked through ordinary English, Hindi, or Hinglish requests and remain role- and owner-gated.

## Main API examples

All owner-scoped routes require `Authorization: Bearer <access-token>`.

Select an explanation mode:

```http
POST /chat
Content-Type: application/json

{"question":"Explain anticipatory bail","language":"hinglish","explanation_mode":"simple"}
```

Get exactly one relevant follow-up:

```http
POST /assistant/follow-up
Content-Type: application/json

{
  "workflow":"cyber_fraud",
  "messages":[{"role":"user","content":"PhonePe par kal 10:30 ko fraud hua"}],
  "known_fields":{"bank":"HDFC"},
  "asked_fields":[],
  "language":"hinglish"
}
```

Create and confirm an RTI pre-fill:

```http
POST /assistant/forms
Authorization: Bearer <token>
Content-Type: application/json

{"form_type":"rti","facts":{"applicant_name":"Asha","applicant_address":"Pune"}}
```

After correcting all missing fields shown by the response:

```http
POST /assistant/forms/<workflow-id>/confirm
Authorization: Bearer <token>
Content-Type: application/json

{"confirmation_text":"I CONFIRM THE REVIEWED FACTS"}
```

This confirms reviewed data only. The response always reports `external_submission: false`.

Queue an export and poll it:

```http
POST /assistant/jobs
Authorization: Bearer <token>
Content-Type: application/json

{"job_type":"export","payload":{"draft_id":"<id>","format":"docx","watermark":true}}
```

Use `GET /assistant/jobs/<job-id>`, `GET /assistant/downloads`, and `GET /assistant/downloads/<artifact-id>`.

Admin source upload uses multipart form data at `POST /admin/phase3/knowledge-gaps/<message-id>/source`: `file` contains the document and `metadata_json` contains `LegalSourceMetadata`. Verification and document linking use:

- `POST /admin/phase3/legal-sources/<source-id>/verify`
- `POST /admin/phase3/legal-sources/<source-id>/link-document`
- `POST /admin/phase3/evaluations/run`
- `GET /admin/phase3/dashboard`

## Data and migrations

New collections are `legal_sources`, `operational_events`, `audit_logs`, `background_jobs`, `user_preferences`, `form_workflows`, `evaluation_runs`, and `download_artifacts`. Run:

```bash
python scripts/migrate_phase3.py
python scripts/create_indexes.py
```

The migration is idempotent and backfills source-version fields on existing document versions. Unique owner indexes prevent preferences from crossing accounts; every form, job, download, case, document, and draft access is checked against its owner.

## Testing

```bash
pytest -q tests/test_phase3.py
pytest -q
pytest -q tests/test_phase2_pdf_runtime.py tests/test_draft_export.py
```

Run PDF tests separately in the production container because WeasyPrint depends on native Pango/GTK libraries. A missing runtime yields an actionable `UnsupportedExportError`; DOCX/TXT/RTF stay usable.
