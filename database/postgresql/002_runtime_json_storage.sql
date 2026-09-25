BEGIN;

-- These tables preserve the application's current document contracts while
-- transactional chat/draft storage moves to PostgreSQL. MongoDB remains only
-- for RAG/vector and the subsystems not yet migrated.
CREATE TABLE IF NOT EXISTS ai_chat_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ai_chat_turns_session ON ai_chat_turns (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_ai_chat_turns_user ON ai_chat_turns (user_id);

CREATE TABLE IF NOT EXISTS ai_conversation_states (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    user_id TEXT,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ai_conversation_states_user ON ai_conversation_states (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS ai_legal_drafts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ai_legal_drafts_session ON ai_legal_drafts (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_ai_legal_drafts_user ON ai_legal_drafts (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_legal_draft_versions (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_legal_draft_version
    ON ai_legal_draft_versions ((payload->>'draft_id'), ((payload->>'version_number')::INTEGER));

COMMIT;
