-- Read-only verification after applying 001_legal_ai_schema.sql.

WITH expected(table_name) AS (
    VALUES
        ('ai_users'),
        ('ai_chat_sessions'),
        ('ai_chat_messages'),
        ('ai_conversation_memory'),
        ('ai_drafts'),
        ('ai_draft_versions'),
        ('ai_documents'),
        ('ai_generated_files'),
        ('ai_feedback'),
        ('ai_user_preferences'),
        ('ai_cases'),
        ('ai_indexing_jobs'),
        ('ai_outbox_events'),
        ('ai_audit_logs')
), actual AS (
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
)
SELECT
    expected.table_name,
    CASE WHEN actual.table_name IS NULL THEN 'MISSING' ELSE 'OK' END AS status
FROM expected
LEFT JOIN actual USING (table_name)
ORDER BY expected.table_name;

SELECT extname AS installed_extension
FROM pg_extension
WHERE extname IN ('pgcrypto', 'citext')
ORDER BY extname;
