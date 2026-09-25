# Handwriting and layout-aware document extraction

In Streamlit, open **Extract handwriting / document text**, upload a file, and select **Extract text**. Review the structured preview, then download editable Word or UTF-8 text.

This is a separate transcription workflow. It does not automatically index the original upload into chat or publish it to the shared knowledge base. To ask the chatbot about the transcription, attach the downloaded file in the regular chat uploader.

## Formats and fidelity

- PDF: renders every page, including mixed printed/handwritten pages, with original page numbering.
- PNG, JPEG, BMP, WebP and TIFF: vision transcription; multi-frame TIFF retains every frame.
- DOCX: reads paragraphs/headings/tables in document order and transcribes embedded paragraph/table images. DOCX does not have reliable physical page numbers; page numbers are not invented. Floating-image positions, text boxes, headers/footers, nested tables and exact font/spacing reproduction are not supported in this initial implementation.
- TXT: preserves supplied UTF-8 text.

Headings, paragraphs, list markers and table cells remain editable in Word. Original handwriting, exact coordinates, fonts, complex merged cells and original pagination after text reflow are not reproduced. Unreadable words should be marked `[illegible]`; users must check names, dates and amounts against the source. OCR is probabilistic, not guaranteed for every handwriting/script or scan quality.

## Configuration

Uses the existing `GEMINI_API_KEY`. It is independent of the model chosen for legal chat:

```dotenv
HANDWRITING_OCR_MODEL=gemini-3.6-flash
HANDWRITING_OCR_MAX_PAGES=20
HANDWRITING_OCR_TIMEOUT_SECONDS=120
HANDWRITING_OCR_DOCUMENT_TIMEOUT_SECONDS=600
```

`OCR_ENABLED=false` disables vision transcription. PDF rendering uses `pypdfium2` from project dependencies, without a separate Poppler install. Install updated dependencies and restart backend/Streamlit to load this feature.

Images are sent to the configured Gemini service for transcription. Backend temporary files are deleted after the request, including failures; extracted content is returned to the caller and not written to shared KB storage or a global result cache. Existing upload size/malware checks apply. Staging/production require authentication. Word/TXT without images do not need a vision call.

Files above the page/frame limit are rejected rather than silently truncated. Provider errors, blocked/truncated responses, corrupt/encrypted PDFs and empty extractions return explicit errors rather than an apparently complete document.

## API

`POST /documents/extract`, multipart field `file`; use the existing bearer authentication in staging/production.

Returns `filename`, `pages` (ordered blocks and per-page warnings), `text`, `docx_base64`, `warnings`, and `extraction_method`. The DOCX is supplied in the response, not through a public download URL.

## Validation

Follow-up live upload repair (2026-09-16): reproduced provider HTTP 503 being misreported as client HTTP 400, then a second failure from global duplicate-hash rejection. OCR now retries transient HTTP/transport failures up to three attempts with bounded backoff and reports exhausted availability failures as HTTP 503. Private attachments may reuse the same bytes in another session or on retry; shared KB duplicate checks remain and exclude private records. Chat attachment errors explain temporary service unavailability without exposing provider bodies. The API was restarted and the user's Hindi screenshot returned HTTP 200 with one indexed chunk (24.2 seconds); evidence is `output/handwriting-smoke/live-upload-final.json`. The subsequent live `/chat` request also returned HTTP 200 and correctly summarized the court-order application template; evidence is `output/handwriting-smoke/live-summary-final.json`. The focused regression suite passed 68 tests.

Private chat attachments now use the handwriting engine for images and scanned PDFs as well. They are indexed in the existing conversation/user scope; the separate extraction panel remains an ephemeral preview/export workflow. Shared KB ingestion retains its existing loader. A failed attachment (including one failure in a multi-file turn) stops the chat request, so missing content is not replaced by a conversation summary. Retry the failed file before requesting a combined review. Attachment-only turns explicitly request a document summary.

The two user-supplied screenshots from 2026-09-16 were transcribed through the private loader using the actual OCR provider: English cursive (198 characters) and Hindi (768 characters). Both exceeded the configured text-quality threshold; this is not a claim of perfect transcription. The Hindi image has cropped line endings. Evidence is saved locally under `output/handwriting-smoke/user-screenshots/`. Regression tests also cover failed and partially failed uploads, scanned PDFs, frame order, empty extraction and document-summary routing. The running shared server was not restarted or tested end to end for attachment indexing after this patch.

`tests/test_handwriting_extraction.py` covers the provider contract, incomplete responses, configuration errors, private error handling, Word tables/headings, page/frame identity, embedded DOCX images, limits, scanner rejection, temporary cleanup and authentication.

Live synthetic cursive-image smoke testing validates provider connectivity and transcription/export. Synthetic cursive text is not evidence of accuracy on arbitrary real handwriting; real samples and native-language review remain necessary for an OCR accuracy benchmark.

Verified on 2026-09-16: image transcription and an in-process HTTP request to the extraction route using a real rasterized PDF returned the expected name, amount, date and list; the route returned HTTP 200 with a readable DOCX. No running shared backend was restarted. Streamlit AppTest covers preview/table/download rendering and clearing results on identity changes. Word contents, headings, tables and page breaks were checked structurally; visual DOCX rendering could not run because LibreOffice is not installed on this host.
