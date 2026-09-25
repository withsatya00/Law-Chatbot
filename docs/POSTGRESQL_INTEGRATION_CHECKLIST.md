# PostgreSQL Integration Checklist

## Deliverable

Run `database/postgresql/001_legal_ai_schema.sql` against a new PostgreSQL 15+
database. It creates Legal AI-owned users, chats, memory, drafts, documents,
generated-file metadata, feedback, preferences, cases, indexing jobs, outbox and
audit tables. It does not store embeddings or file bytes.

## Required from the website/app backend

- An authenticated internal profile endpoint that returns a verified email
  and/or E.164 phone number. Never trust email/phone sent directly in a public
  request body.
- A service-to-service authentication mechanism for that endpoint.
- Clear behavior for email/phone changes, account blocking and account deletion.
- Confirmation that every non-null email and phone is unique in the source
  system.
- A webhook/event with a unique event ID for profile updates and deletions.

Example internal response:

```json
{
  "email": "user@gmail.com",
  "email_verified": true,
  "phone": "+919876543210",
  "phone_verified": true,
  "status": "active"
}
```

## Required infrastructure

- PostgreSQL 15+ database and least-privilege application role.
- `pgcrypto` and `citext` extension permission during migration.
- TLS-enabled `POSTGRESQL_URL`, supplied through a secret manager or `.env`;
  never commit its value.
- Connection pooling and separate migration/application credentials.
- Object storage (preferred) or a durable shared filesystem for original and
  generated files. PostgreSQL stores `storage_key`, not file bytes.
- MongoDB Atlas for `embeddings_metadata` chunks/vectors and Redis for ephemeral
  cache, locks and rate limits.

## Identity rules

1. Normalize email by trimming whitespace and lowercasing. Do not apply
   Gmail-specific dot or plus-address rewriting.
2. Normalize phone to E.164 before lookup.
3. Look up email and phone separately. If they resolve to two different
   `ai_users`, reject the request as an identity conflict; never auto-merge.
4. Create `ai_users` only from verified data returned by the trusted backend.
5. Store `ai_users.id` in every chat/draft/document row. Never use email/phone
   as a foreign key.
6. On a verified profile change, update the same `ai_users` row transactionally.

## Data boundaries

| Store | Data |
| --- | --- |
| PostgreSQL | AI users, chat sessions/messages, memory, drafts/versions, document metadata, generated-file metadata, feedback, cases, jobs/outbox/audit |
| MongoDB Atlas | Chunk text, embeddings, legal metadata and vector-search filters; private chunks carry PostgreSQL `ai_user_id` and `document_id` |
| Object/filesystem storage | Original uploads and generated PDF/DOCX/TXT/RTF bytes |
| Redis | Temporary context/cache, rate limits and distributed locks |

## Application work still required

The SQL schema alone does not switch the current application from MongoDB.
Implementation must add a PostgreSQL driver/pool, PostgreSQL repositories,
trusted profile lookup, transactional outbox worker, Mongo-to-PostgreSQL data
migration, deletion coordination across stores and ownership regression tests.

Recommended cutover order:

1. Apply schema to development PostgreSQL.
2. Implement and test identity resolution.
3. Implement chat/session repositories.
4. Implement draft/version repositories.
5. Implement document metadata, indexing jobs and outbox worker.
6. Backfill existing Mongo business data with an idempotent migration.
7. Reconcile row counts and ownership before switching reads.
8. Run cross-user isolation, deletion, retry and rollback tests.
9. Deploy to staging before production.
