-- Legal AI application database schema
-- Target: PostgreSQL 15+
-- Identity arrives from the existing website/app backend as a verified email
-- and/or E.164 phone number. This database generates its own stable UUID and
-- never uses mutable PII as a foreign key.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE OR REPLACE FUNCTION legal_ai_set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS ai_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email CITEXT,
    phone VARCHAR(16),
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    phone_verified BOOLEAN NOT NULL DEFAULT FALSE,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'blocked', 'deleted')),
    profile_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT ai_users_identity_required CHECK (email IS NOT NULL OR phone IS NOT NULL),
    CONSTRAINT ai_users_email_nonempty CHECK (email IS NULL OR length(trim(email::text)) > 3),
    CONSTRAINT ai_users_phone_e164 CHECK (phone IS NULL OR phone ~ '^\+[1-9][0-9]{7,14}$'),
    CONSTRAINT ai_users_verified_email_present CHECK (NOT email_verified OR email IS NOT NULL),
    CONSTRAINT ai_users_verified_phone_present CHECK (NOT phone_verified OR phone IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_users_email
    ON ai_users (email) WHERE email IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_users_phone
    ON ai_users (phone) WHERE phone IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_ai_users_status ON ai_users (status);

CREATE TABLE IF NOT EXISTS ai_chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    language VARCHAR(32),
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived', 'deleted')),
    last_message_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ai_chat_sessions_user_updated
    ON ai_chat_sessions (ai_user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS ai_chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    parent_message_id UUID REFERENCES ai_chat_messages(id) ON DELETE SET NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    detected_language VARCHAR(32),
    detected_intent VARCHAR(120),
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'complete'
        CHECK (status IN ('pending', 'streaming', 'complete', 'failed', 'cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT ai_chat_messages_content_nonempty CHECK (length(trim(content)) > 0)
);
CREATE INDEX IF NOT EXISTS ix_ai_chat_messages_session_timeline
    ON ai_chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_ai_chat_messages_user_created
    ON ai_chat_messages (ai_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_conversation_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL UNIQUE REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    confirmed_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    assumptions JSONB NOT NULL DEFAULT '{}'::jsonb,
    missing_details JSONB NOT NULL DEFAULT '[]'::jsonb,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_conversation_memory_user
    ON ai_conversation_memory (ai_user_id);

CREATE TABLE IF NOT EXISTS ai_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES ai_chat_sessions(id) ON DELETE SET NULL,
    draft_type VARCHAR(120) NOT NULL,
    title VARCHAR(255),
    language VARCHAR(32),
    fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    sections JSONB NOT NULL DEFAULT '{}'::jsonb,
    full_text TEXT NOT NULL DEFAULT '',
    status VARCHAR(24) NOT NULL DEFAULT 'collecting'
        CHECK (status IN ('collecting', 'draft', 'ready', 'approved', 'locked', 'cancelled', 'deleted')),
    current_version INTEGER NOT NULL DEFAULT 1 CHECK (current_version > 0),
    approved_at TIMESTAMPTZ,
    locked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ai_drafts_user_updated ON ai_drafts (ai_user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_ai_drafts_session ON ai_drafts (session_id);

CREATE TABLE IF NOT EXISTS ai_draft_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    draft_id UUID NOT NULL REFERENCES ai_drafts(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    sections JSONB NOT NULL DEFAULT '{}'::jsonb,
    full_text TEXT NOT NULL,
    change_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (draft_id, version_number)
);
CREATE INDEX IF NOT EXISTS ix_ai_draft_versions_history
    ON ai_draft_versions (draft_id, version_number DESC);

CREATE TABLE IF NOT EXISTS ai_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES ai_chat_sessions(id) ON DELETE SET NULL,
    original_filename VARCHAR(255) NOT NULL,
    storage_key TEXT NOT NULL,
    mime_type VARCHAR(120) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    sha256_hash CHAR(64) NOT NULL,
    detected_language VARCHAR(32),
    indexing_status VARCHAR(24) NOT NULL DEFAULT 'pending'
        CHECK (indexing_status IN ('pending', 'processing', 'indexed', 'failed', 'deleting', 'deleted')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ai_documents_user_created
    ON ai_documents (ai_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_ai_documents_indexing_status
    ON ai_documents (indexing_status, created_at);
CREATE INDEX IF NOT EXISTS ix_ai_documents_hash
    ON ai_documents (ai_user_id, sha256_hash);

CREATE TABLE IF NOT EXISTS ai_generated_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    draft_id UUID REFERENCES ai_drafts(id) ON DELETE CASCADE,
    document_id UUID REFERENCES ai_documents(id) ON DELETE CASCADE,
    format VARCHAR(12) NOT NULL CHECK (format IN ('pdf', 'docx', 'txt', 'rtf', 'json')),
    storage_key TEXT NOT NULL,
    media_type VARCHAR(120) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    sha256_hash CHAR(64) NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_generated_files_source_required CHECK (draft_id IS NOT NULL OR document_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_ai_generated_files_user_created
    ON ai_generated_files (ai_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES ai_chat_sessions(id) ON DELETE SET NULL,
    message_id UUID REFERENCES ai_chat_messages(id) ON DELETE SET NULL,
    rating SMALLINT CHECK (rating BETWEEN 1 AND 5),
    category VARCHAR(50),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_feedback_value_required CHECK (rating IS NOT NULL OR category IS NOT NULL OR comment IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_ai_feedback_user_created
    ON ai_feedback (ai_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_user_preferences (
    ai_user_id UUID PRIMARY KEY REFERENCES ai_users(id) ON DELETE CASCADE,
    preferred_language VARCHAR(32),
    explanation_level VARCHAR(24),
    preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID NOT NULL REFERENCES ai_users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    case_number VARCHAR(120),
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    next_hearing_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ai_cases_user_updated ON ai_cases (ai_user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_ai_cases_upcoming ON ai_cases (ai_user_id, next_hearing_at)
    WHERE next_hearing_at IS NOT NULL AND deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS ai_indexing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES ai_documents(id) ON DELETE CASCADE,
    job_type VARCHAR(24) NOT NULL DEFAULT 'index'
        CHECK (job_type IN ('index', 'reindex', 'delete')),
    status VARCHAR(24) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'indexed', 'failed', 'retrying', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 4 CHECK (max_attempts > 0),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_indexing_jobs_claim
    ON ai_indexing_jobs (status, next_attempt_at, created_at);

CREATE TABLE IF NOT EXISTS ai_outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    aggregate_id UUID NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'processed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_outbox_claim
    ON ai_outbox_events (status, available_at, created_at);

CREATE TABLE IF NOT EXISTS ai_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ai_user_id UUID REFERENCES ai_users(id) ON DELETE SET NULL,
    actor_type VARCHAR(24) NOT NULL DEFAULT 'user',
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    request_id VARCHAR(100),
    ip_hash VARCHAR(128),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_audit_logs_user_created
    ON ai_audit_logs (ai_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_ai_audit_logs_resource
    ON ai_audit_logs (resource_type, resource_id, created_at DESC);

DROP TRIGGER IF EXISTS trg_ai_users_updated_at ON ai_users;
CREATE TRIGGER trg_ai_users_updated_at BEFORE UPDATE ON ai_users
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_chat_sessions_updated_at ON ai_chat_sessions;
CREATE TRIGGER trg_ai_chat_sessions_updated_at BEFORE UPDATE ON ai_chat_sessions
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_conversation_memory_updated_at ON ai_conversation_memory;
CREATE TRIGGER trg_ai_conversation_memory_updated_at BEFORE UPDATE ON ai_conversation_memory
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_drafts_updated_at ON ai_drafts;
CREATE TRIGGER trg_ai_drafts_updated_at BEFORE UPDATE ON ai_drafts
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_documents_updated_at ON ai_documents;
CREATE TRIGGER trg_ai_documents_updated_at BEFORE UPDATE ON ai_documents
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_user_preferences_updated_at ON ai_user_preferences;
CREATE TRIGGER trg_ai_user_preferences_updated_at BEFORE UPDATE ON ai_user_preferences
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_cases_updated_at ON ai_cases;
CREATE TRIGGER trg_ai_cases_updated_at BEFORE UPDATE ON ai_cases
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();
DROP TRIGGER IF EXISTS trg_ai_indexing_jobs_updated_at ON ai_indexing_jobs;
CREATE TRIGGER trg_ai_indexing_jobs_updated_at BEFORE UPDATE ON ai_indexing_jobs
    FOR EACH ROW EXECUTE FUNCTION legal_ai_set_updated_at();

COMMIT;
