# Legal AI Assistant

Phase 2 practical workflows, API examples, migration notes, and verification commands are documented in [docs/PHASE2.md](docs/PHASE2.md).

Enterprise-oriented FastAPI backend for a multilingual Indian Legal AI Assistant with RAG, document ingestion, document analysis, lawyer recommendation, a conversational AI legal drafting engine, MongoDB persistence, Redis caching, and a Streamlit testing console.

## Run Locally

```bash
cp .env.example .env
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Run the developer UI:

```bash
streamlit run streamlit_app/app.py
```

## Run With Docker

```bash
cp .env.example .env
docker compose up --build
```

API: `http://localhost:8000`

Streamlit test console: `http://localhost:8501`

## Main APIs

- `POST /register`
- `POST /login`
- `POST /chat`
- `POST /upload`
- `POST /search`
- `POST /document-analysis`
- `POST /intent`
- `POST /entities`
- `POST /recommend-lawyer`
- `POST /feedback`
- `GET /history`
- `GET /session`
- `GET /health`
- `DELETE /session`

### Conversational AI Legal Drafting Engine

Drafting happens entirely inside the normal chat conversation — there is no separate drafting
page or static form. `POST /chat` detects drafting intent automatically (English, Hindi, and
Hinglish: "I want to write a police complaint", "RTI application banana hai", "cyber complaint
likhni hai", ...), extracts whatever information is already in the message, asks only for what's
still missing, and lets you keep editing/translating/exporting the draft through further chat
messages ("change the police station to X", "translate into hindi"). See `app/drafting/`,
particularly `conversation.py` (the state machine), `intent.py`, `field_extraction.py`,
`validation.py`, and `edit_commands.py`.

Backend surface used internally by `/chat` (also directly callable for testing/automation — the
Streamlit frontend only ever calls `/chat`):

- `GET /draft-templates` / `GET /draft-templates/{draft_id}` — list templates and their field schemas
- `POST /draft` — send a message into the drafting state machine directly
- `POST /draft/edit` — apply a structured or free-text edit to an existing draft
- `POST /draft/export` — `{draft_id, format}` → PDF / DOCX / TXT
- `POST /draft/translate`
- `POST /legal-terms` — Legal Hindi Knowledge Library lookup
- `POST /draft-history`

Templates are pure data: `app/drafting/templates/*.yaml`, one file per draft type (fields, act/
section hints, drafting notes, natural-language `trigger_phrases`, and `field_synonyms` used to
interpret in-chat edit commands). Adding a new draft type is a YAML-only change — no Python code
changes anywhere. The YAML files currently present in this directory are the authoritative template
catalog; call `GET /draft-templates` instead of relying on a documentation count.

Entity extraction from free text is regex/heuristic-based (dates, amounts, phone numbers, email, a
known bank/city list, "my name is X" and explicit "label: value" / "label is X" phrasing) so it
works without any LLM configured; when an LLM key is set, a second structured-JSON extraction pass
runs on top and generally captures the rest (e.g. freeform fields like "facts" or "police
station"). It's best-effort NLU, not guaranteed-perfect — anything missed is simply asked for
directly, like a field the user never mentioned.

Every generated draft carries the mandatory AI-generated-draft disclaimer and is screened by both
the existing prompt-injection scanner and a dedicated `DraftSafetyGuard` before generation, to
block requests for forged, fabricated, or fake content. In-chat edits are versioned (`DRAFT_VERSIONS`
collection, `LegalDraftEngine.regenerate`) using the same old/new-version-chain pattern as document
versioning. PDF export renders Devanagari via a Unicode TTF (`PDF_UNICODE_FONT_PATH`, auto-detected
on Windows via `Nirmala.ttc`; on Linux/Docker install `fonts-noto-core`, already added to the
provided `Dockerfile`).

## Configuration

All runtime settings are environment-driven. See `.env.example` for MongoDB, Redis, JWT, LLM providers, embedding model, upload limits, and CORS.

## Architecture

The app follows clean boundaries:

- `api`: FastAPI routers
- `services`: business orchestration
- `rag`: loaders, metadata extraction, chunking, embeddings, vector store, retrieval, reranking
- `llm`: provider abstractions, factory, prompt files
- `memory`: session memory backed by Redis
- `repositories`: MongoDB persistence
- `language`, `intent`, `entity_extraction`, `recommendation`: independent AI support services
- `drafting`: independent AI legal draft generation engine, glossary, and PDF/DOCX/TXT export
- `streamlit_app`: developer-only testing interface

## Production Notes

Use MongoDB Atlas Vector Search in production by replacing the local cosine fallback in `MongoVectorStore.search` with an Atlas `$vectorSearch` aggregation using the configured index. The rest of the application depends only on the `VectorStore` interface.

Set a strong `JWT_SECRET_KEY`, configure provider API keys, restrict CORS origins, enforce HTTPS at the load balancer, and run `scripts/create_indexes.py` during deployment.

## Source governance, page evidence and index reconciliation

Phase 2 added a legal-source registry with human review, page-level citations,
and a deterministic MongoDB/BM25 reconciler. Operator guide:
[docs/SOURCE_GOVERNANCE.md](docs/SOURCE_GOVERNANCE.md).

```powershell
.venv\Scripts\python.exe scriptseconcile_indexes.py                    # index drift, dry run
.venv\Scripts\python.exe scriptseconcile_indexes.py --apply            # prune stale BM25 entries
.venv\Scripts\python.exe scripts\migrate_phase2_governance.py            # governance backfill, dry run
```

A source is only `verified` when a human recorded the evidence for it; an
official-looking filename confers nothing. A page number shown to a user is the
page the text is actually on, or it is absent — page numbers are never guessed
and never backfilled onto chunks indexed before Phase 2.

## Phase 2 and Phase 3 guides

- [Phase 2 workflows](docs/PHASE2.md)
- [Phase 3 implementation and API examples](docs/PHASE3.md)
- [Production deployment checklist](docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md)
- [Admin runbook](docs/ADMIN_RUNBOOK.md)
- [Phase 3 user guide](docs/USER_GUIDE_PHASE3.md)
