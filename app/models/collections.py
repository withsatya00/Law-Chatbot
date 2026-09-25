USERS = "users"
CHATS = "chats"
SESSIONS = "sessions"
CONVERSATION_MEMORY = "conversation_memory"
UPLOADED_DOCUMENTS = "uploaded_documents"
EMBEDDINGS_METADATA = "embeddings_metadata"
LAWYER_CATEGORIES = "lawyer_categories"
LAWYER_RECOMMENDATIONS = "lawyer_recommendations"
PROMPT_LOGS = "prompt_logs"
QUERY_LOGS = "query_logs"
FEEDBACK = "feedback"
SYSTEM_LOGS = "system_logs"
DOCUMENT_VERSIONS = "document_versions"
INDEXING_JOBS = "indexing_jobs"
KB_STAGING_RECORDS = "kb_staging_records"
KB_AUTOMATION_JOBS = "kb_automation_jobs"
KB_AUTOMATION_STATE = "kb_automation_state"
KB_GAP_AUTOFETCH_QUEUE = "kb_gap_autofetch_queue"
KB_SOURCE_ADAPTERS = "kb_source_adapters"
MODEL_REGISTRY = "model_registry"
SEMANTIC_CACHE = "semantic_cache"
OBSERVABILITY_EVENTS = "observability_events"
LEGAL_DRAFTS = "legal_drafts"
DRAFT_VERSIONS = "draft_versions"
CASES = "cases"
INTENT_EVENTS = "intent_events"
INTENT_FEEDBACK = "intent_feedback"
LEGAL_SOURCES = "legal_sources"
OPERATIONAL_EVENTS = "operational_events"
AUDIT_LOGS = "audit_logs"
BACKGROUND_JOBS = "background_jobs"
USER_PREFERENCES = "user_preferences"
FORM_WORKFLOWS = "form_workflows"
EVALUATION_RUNS = "evaluation_runs"
DOWNLOAD_ARTIFACTS = "download_artifacts"

# --- E-Notarization ---------------------------------------------------------
# One row per notarizable document VERSION (`notarization_documents`), so a
# hash, a signature and a notarization always refer to specific bytes.
NOTARIZATION_DOCUMENTS = "notarization_documents"
NOTARY_ACCOUNTS = "notary_accounts"
NOTARIZATION_REQUESTS = "notarization_requests"
ESIGN_SESSIONS = "esign_sessions"
# Append-only. See `app/notarization/audit.py`.
NOTARIZATION_AUDIT_EVENTS = "notarization_audit_events"

# One row per uploaded document's most recent professionally-formatted
# retype (`DocumentTranscribeWorkflow`), keyed by `document_id` so a later
# PDF/DOCX/TXT export request re-renders the SAME already-transcribed text
# instead of re-running the LLM transcription on every download.
DOCUMENT_TRANSCRIPTS = "document_transcripts"

