# Phase 2 workflow assistant

Phase 2 is additive: all Phase 1 routes and request/response fields remain available. New workflow routes are deterministic and do not require an LLM. Private case routes require the existing bearer-token authentication and remain owner-scoped.

## API examples

Start cyber-fraud triage:

```bash
curl -X POST http://localhost:8000/workflows/cyber-fraud \
  -H "Content-Type: application/json" \
  -d '{
    "language":"hinglish",
    "narrative":"PhonePe UPI se Rs 45000 fraud hua on 26/08/2026",
    "amount":"45000",
    "transaction_time":"26/08/2026 10:30",
    "transaction_id":"HDFC1234567890",
    "bank":"HDFC Bank",
    "platform":"PhonePe",
    "evidence":[{
      "evidence_id":"ev-1",
      "document_name":"upi_receipt.pdf",
      "text":"Debit Rs 45,000 on 26/08/2026 UTR HDFC1234567890"
    }]
  }'
```

The response contains sourced immediate actions, missing information, annexures, chronology, contradictions, an export block flag, three editable drafts, and an opt-in lawyer-escalation recommendation. `1930` and `cybercrime.gov.in` are shown only with official MHA/I4C sources.

Organize evidence and check jurisdiction:

```bash
curl -X POST http://localhost:8000/workflows/evidence/organize \
  -H "Content-Type: application/json" \
  -d '{"series":"A","evidence":[{"evidence_id":"ev-1","document_name":"receipt.png","text":"Rs 45,000 on 26/08/2026 UTR HDFC1234567890"}]}'

curl -X POST http://localhost:8000/workflows/jurisdiction \
  -H "Content-Type: application/json" \
  -d '{"matter_type":"cyber_complaint","complainant_location":"Pune, Maharashtra","online_transaction":true}'
```

Case dashboard (bearer token required):

```bash
curl -X POST http://localhost:8000/cases \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"case_number":"CYBER/2026/001","title":"UPI fraud","legal_category":"Cyber","parties":["Applicant","Unknown beneficiary"],"next_action":"Report to bank and 1930"}'

curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/cases/CASE_ID/lawyer-summary

curl -X POST http://localhost:8000/cases/CASE_ID/evidence/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@upi_receipt.pdf" -F "description=Bank transaction receipt"
```

The evidence upload uses the existing parser/OCR pipeline, extracts dates, amounts, parties and transaction references, assigns a stable annexure number, and stores the result only inside the authenticated owner's case.

Draft review workspace:

```bash
curl "http://localhost:8000/draft/DRAFT_ID/review?session_id=SESSION_ID"
curl "http://localhost:8000/draft/DRAFT_ID/compare?original_version=1&revised_version=2&session_id=SESSION_ID"
curl -X POST http://localhost:8000/draft/DRAFT_ID/duplicate \
  -H "Content-Type: application/json" -d '{"session_id":"SESSION_ID"}'
```

The existing edit, regenerate/translate, lifecycle, delete, history/switch, and PDF/DOCX/TXT/RTF export routes are unchanged. Export now accepts `watermark_text` and `include_header_footer`; PDF failure returns an actionable error while DOCX/TXT/RTF remain usable.

## UI

Open **Legal Workflows** in the Streamlit sidebar. The cyber workspace collects transaction details and evidence, shows verified escalation, flags conflicts, and exposes editable/downloadable complaint drafts. The jurisdiction tab always displays the verify-before-filing warning.

## Migration and indexes

```bash
python scripts/migrate_phase2.py
python scripts/create_indexes.py
```

The migration is idempotent and only fills absent fields. Existing cases and Phase 1 API clients continue to work.

## Verification

```bash
pytest -q tests/test_phase2_workflows.py
pytest -q tests/test_phase2_pdf_runtime.py
pytest -q
```

PDF runtime behavior is deliberately isolated from ordinary chat/workflow tests.
