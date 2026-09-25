import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import AnyHttpUrl, BaseModel, Field

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings, SettingsConfigDict
else:
    # `pydantic-settings` is a declared dependency; this is a last-resort
    # runtime guard so a partially-installed host degrades to plain-BaseModel
    # settings instead of failing to import the whole app. Type checking runs
    # against the real classes above -- checking against these stand-ins would
    # verify nothing about the settings model that actually ships.
    try:
        from pydantic_settings import BaseSettings, SettingsConfigDict
    except ModuleNotFoundError:
        BaseSettings = BaseModel

        class SettingsConfigDict(dict):
            pass


_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

# Phase 1 security hardening: known-insecure JWT_SECRET_KEY values -- the
# code default, the .env.example placeholder, and other common placeholders
# -- that must never reach a production/staging deployment, since anyone who
# has read either file can forge tokens signed with them.
_INSECURE_JWT_SECRETS = {
    "",
    "change-this-secret-in-production",
    "secret",
    "changeme",
    "change-me",
    "your-secret-key",
    "replace_with_a_strong_random_secret_min_32_chars",
    "replace_with_a_secret_manager_value_min_32_chars",
}
_MIN_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Legal AI Assistant"
    environment: Literal["development", "test", "staging", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[AnyHttpUrl | str] = Field(
        default_factory=lambda: cast(list[AnyHttpUrl | str], ["http://localhost:8501"])
    )

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "legal_ai_assistant"
    postgresql_enabled: bool = False
    postgresql_url: str = "postgresql://legal_ai:legal_ai@localhost:5432/legal_ai"
    postgresql_pool_min_size: int = Field(default=1, ge=1, le=20)
    postgresql_pool_max_size: int = Field(default=10, ge=1, le=100)
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-this-secret-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14

    llm_provider: str = "ollama"
    # Ordered, comma-separated backup providers used by
    # `LLMFactory.create_resilient()` when the primary returns an error --
    # e.g. "groq,gemini,ollama". Empty (the default) keeps single-provider
    # behaviour, with retries still applied to the primary.
    llm_fallback_providers: str = ""
    llm_max_retry_attempts: int = 3
    llm_retry_backoff_seconds: float = 0.75
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Separate from `gemini_model` (the chat-completion model) on purpose --
    # Gemini's native-audio TTS models are a distinct model family, not a
    # capability of the chat model, so this stays independently configurable.
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    gemini_tts_voice: str = "Kore"
    # Same reasoning as `gemini_tts_model`: speech-to-text needs a model with
    # audio INPUT support, which is not guaranteed of whatever `gemini_model`
    # is configured to for text chat -- confirmed live (QA session 5,
    # BUG-016) that pointing `gemini_model` at a Gemma variant breaks
    # `/voice/chat` outright ("Audio input modality is not enabled for this
    # model"), since `gemini_speech_to_text` previously reused `gemini_model`
    # directly. Kept independently configurable so an operator can pick a
    # cheaper/faster text model without silently breaking voice input.
    gemini_stt_model: str = "gemini-3.6-flash"
    claude_api_key: str = ""
    claude_model: str = "claude-opus-5"
    claude_max_tokens: int = 4096
    deepseek_api_key: str = ""
    # Both configurable rather than hardcoded to DeepSeek's own API: a
    # "DeepSeek" model is also commonly reached through a third-party
    # OpenAI-compatible host (e.g. NVIDIA NIM's "deepseek-ai/..." catalog
    # names, served from integrate.api.nvidia.com, not api.deepseek.com) --
    # an API key issued by that host would silently 401 against DeepSeek's
    # own endpoint. Defaults preserve DeepSeek's own official API for anyone
    # who already has a native "sk-..." key and never sets these.
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    # OpenRouter: a single OpenAI-compatible endpoint that proxies many
    # providers' models by "provider/model" id. Free-tier models are marked
    # with a ":free" suffix (not "-free" -- confirmed live against
    # OpenRouter's own /models endpoint; "z-ai/glm-5.3-free" doesn't exist
    # and 400s). Defaults to the same model already configured for the
    # `gemini` provider (`gemini_model`), just reached through OpenRouter's
    # free tier instead of Google's API directly.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    llm_timeout: int = 120
    # --- Request budget --------------------------------------------------
    # The chat UI's HTTP timeout, and the server's own ceiling underneath it.
    # These are the two halves of one contract and are deliberately declared
    # together, because getting them out of step is exactly what produced the
    # reported "Sorry, I couldn't reach the assistant just now": the server
    # had no total budget at all, so a drafting turn could run ~450s
    # (extraction call + 3 retried generation calls at `llm_timeout` each)
    # while the client gave up at 180s. The user saw a transport error for a
    # request the backend was still working on, and lost everything they had
    # typed.
    #
    # `chat_request_budget_seconds` is bound as a deadline for the whole of
    # `POST /chat` (see `app/llm/deadline.py`); every LLM call clamps itself
    # to what remains. It MUST stay meaningfully below
    # `client_request_timeout_seconds` so there is time left to build and
    # send a real degraded response -- `Settings._validate_budgets()` enforces
    # that at startup rather than letting a bad .env recreate the bug.
    client_request_timeout_seconds: int = 180
    chat_request_budget_seconds: int = 150
    llm_max_tokens: int = 4096
    conversation_intent_llm_enabled: bool = False
    conversation_intent_llm_min_confidence: float = 0.70
    conversation_intent_llm_max_history: int = 8
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_version: str = "2026-08-03"
    embedding_device: str = ""
    embedding_dimensions: int = 1024
    vector_store_provider: str = "mongodb"
    vector_search_backend: Literal["atlas", "local"] = "atlas"
    # `MongoVectorStore._local_cosine_leg`'s fallback scan cap, used whenever
    # `vector_search_backend="local"` or Atlas `$vectorSearch` errors out.
    # `.find().limit()` with no explicit sort returns Mongo's natural/
    # insertion order, so once the corpus exceeds this cap, whichever
    # documents were indexed *last* become silently unreachable by this leg
    # -- confirmed directly at 1439 chunks with a limit of 1000. Configurable
    # so a deployment that can't run Atlas can raise it without a code
    # change; still bounded so a pathologically large corpus can't blow up
    # per-query latency.
    # QA retest 2026-09-24 (Hindi/Sanskrit/Sindhi/Telugu grounding
    # investigation): the corpus has grown to ~55,000 chunks, past this cap's
    # old value of 20,000 -- meaning roughly the newest third of the corpus
    # (whichever documents were indexed last, per this field's own docstring
    # above) was silently unreachable by every local-scan query, not just a
    # slow one. Live-reproduced: a Hindi/Hinglish security-deposit question
    # ("mera landlord mera security deposit wapas nahi de raha hai")
    # correctly expanded to the right search vocabulary and correctly passed
    # `is_relevant_chunk`'s lexical gate on a genuinely on-topic Tenancy Act
    # chunk -- when that chunk happened to fall inside the scanned window.
    # Across repeated identical calls it sometimes did and sometimes didn't
    # (`.find().limit()` with no sort has no stability guarantee), so
    # grounding for the same question was non-deterministic, not just
    # degraded -- confirmed directly against the live corpus (55,070 chunks
    # at time of fix). Raised to comfortably exceed the corpus so the local
    # fallback covers all of it deterministically, matching what Atlas
    # `$vectorSearch` already does; still bounded (not removed) so a much
    # larger future corpus can't blow up per-query latency unboundedly, and
    # still gated by `local_vector_scan_max_concurrency` (BUG-101) so this
    # heavier per-query cost can't pile up under concurrent load.
    local_vector_scan_limit: int = 60000
    # QA session 2026-09-24 ("BUG-101"): with no cap, 6 concurrent `/chat`
    # requests each independently ran a full `local_vector_scan_limit`
    # (20,000-document, full-embedding-vector) fetch+score at once, live-
    # measured at 100-197s per call under that pile-up (vs. single-digit
    # seconds uncontended) -- the requests weren't failing, they were all
    # fighting the same Mongo collection and CPU-bound scoring thread pool
    # for the same scarce resource simultaneously. Bounding how many of
    # these heavyweight scans run at once turns that pile-up into a queue:
    # later callers wait instead of all degrading together. This is a
    # mitigation for the local/non-Atlas fallback specifically (see
    # `_validate_retrieval_backend`) -- the real fix for production is Atlas
    # Vector Search, which doesn't have this problem at all (~10ms/query,
    # confirmed via `scripts/_atlas_readiness_verification_20260923.py`).
    local_vector_scan_max_concurrency: int = Field(default=2, ge=1, le=50)
    vector_schema_version: str = "v1"
    mongodb_vector_index: str = "legal_chunks_vector_index"
    # Off by default: `LegalReranker`'s heuristic bonuses (definition-query
    # source-type weighting, topic bonuses, section provenance, TOC penalty,
    # etc.) are individually hand-tuned against real benchmark queries across
    # this project's retrieval-quality test suite. A cross-encoder is a
    # genuinely stronger relevance signal in general, but blending it in
    # shifts every one of those tuned scores at once and needs its own
    # benchmark pass before it's trusted -- so it ships here as an optional
    # extra signal a deployment can opt into, never a silent behavior change.
    use_neural_reranker: bool = False
    neural_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How much weight the cross-encoder signal gets against the existing
    # heuristic score once enabled: `final = (1 - w) * heuristic + w * neural`.
    # Additive, not a replacement, so the tuned heuristics still dominate.
    neural_reranker_weight: float = 0.25
    # Part 40 "Hybrid Legal Retrieval": Reciprocal Rank Fusion's smoothing
    # constant (default 60, per the spec) and a debug-mode toggle that adds
    # the raw per-leg rankings (query text, top BM25/embedding hits, RRF
    # ranking) to the hybrid-retrieval log line -- left off by default so
    # production logs stay minimal and don't dump full query text.
    retrieval_rrf_k: int = 60
    retrieval_debug: bool = False
    prompt_version: str = "v1"
    model_version: str = "v1"
    indexing_batch_size: int = 32
    min_ocr_quality_score: float = 0.55
    min_metadata_quality_score: float = 0.45
    ocr_enabled: bool = True
    ocr_language: str = "eng+hin"
    # Tesseract only ships trained data for `ocr_language` above (English +
    # Hindi) -- a scanned document in any other script (Tamil, Bengali,
    # Urdu, ...) still runs through that same model and produces confidently
    # wrong text with no signal that anything went wrong, since Tesseract
    # doesn't know it was given the wrong alphabet. Below `ocr_min_confidence`
    # (Tesseract's own 0-100 word-confidence scale, mean across the
    # document), `DocumentLoader._load_pdf` retries the scan through the
    # Gemini vision OCR path (`handwriting_ocr_model`), which reads any
    # script, before falling back to keeping the low-confidence Tesseract
    # text with an `ocr_degraded_reason` so the ingestion caller can flag it.
    ocr_min_confidence: float = Field(default=45.0, ge=0, le=100)
    ocr_low_confidence_gemini_fallback: bool = True
    handwriting_ocr_model: str = "gemini-3.6-flash"
    handwriting_ocr_max_pages: int = Field(default=20, ge=1, le=100)
    handwriting_ocr_timeout_seconds: int = Field(default=120, ge=10, le=300)
    handwriting_ocr_document_timeout_seconds: int = Field(default=600, ge=30, le=1800)

    max_upload_mb: int = 25
    # Phase 4 "Voice chatbot security": bounds how much audio `/voice/chat`
    # will read into memory and forward to Gemini STT -- previously
    # unbounded (only an empty-file check existed). 10 MB comfortably covers
    # a multi-minute voice question at typical mobile-recorded bitrates
    # while still capping worst-case memory/upstream-API cost per request.
    voice_max_audio_mb: int = 10
    # C9 "VAD before STT": energy-based silence detection/trimming, applied
    # to `audio/wav` uploads directly and to `audio/webm`/`audio/opus`
    # uploads via a bounded PyAV decode -- see `app/services/voice_vad.py`'s
    # module docstring. A quiet-frame threshold, not a hard mute detector --
    # real recordings have room noise/mic hiss, not literal digital silence.
    voice_vad_enabled: bool = True
    voice_vad_silence_threshold_dbfs: float = -40.0
    voice_vad_frame_ms: int = 30
    voice_vad_padding_ms: int = 150
    # Security bound for decoding a compressed (webm/opus) upload before VAD:
    # Opus compresses so well that a file well under `voice_max_audio_mb` can
    # still decode to a wildly disproportionate PCM duration/size (a
    # decompression-bomb shape), so decoding is capped by BOTH a wall-clock
    # timeout (a corrupt/adversarial stream that stalls the decoder) and a
    # maximum decoded duration (a stream that decodes fine but keeps
    # producing frames far past any plausible voice question) -- either one
    # tripping aborts the decode and fails open (`applicable=False`), the
    # same as an unparseable WAV.
    voice_vad_decode_timeout_seconds: float = 8.0
    voice_vad_max_decode_seconds: float = 180.0
    upload_storage_dir: Path = Path("./storage/uploads")
    knowledge_base_dir: Path = Path("./storage/knowledge_base")
    # Admin Knowledge Base Ingestion Pipeline: separate from
    # `upload_storage_dir` (general per-user chat uploads, owner-scoped and
    # never promoted anywhere) -- this is a temporary staging area for admin
    # documents on their way into `knowledge_base_dir`. A file lives here only
    # between upload and either a successful transfer or a failed/pending
    # indexing outcome.
    kb_staging_dir: Path = Path("./storage/kb_staging")
    # Duplicates detected during ingestion are moved here (never deleted) so
    # the original bytes + filename stay recoverable for an admin to review.
    archive_dir: Path = Path("./storage/archive")
    # Documents that cannot be indexed (corrupt/failed) or whose provenance is
    # unverified (orphaned staging files, `storage/uploads` backfill candidates)
    # are quarantined here for an admin decision -- never auto-indexed into the
    # shared Knowledge Base, and never deleted.
    kb_review_dir: Path = Path("./storage/kb_review")
    # Autonomous KB acquisition is opt-in. Production also requires a real
    # malware scanner; a missing scanner quarantines the job instead of
    # allowing unchecked official bytes into parsing/indexing.
    kb_automation_enabled: bool = False
    kb_automation_interval_minutes: int = 60
    kb_automation_manifest: Path = Path("./config/kb_sources.json")
    kb_automation_max_download_mb: int = 25
    kb_automation_max_attempts: int = 4
    kb_automation_source_refresh_hours: int = 24
    # A chat answer that comes back "out of KB" can queue the question's
    # likely Act for an automatic fetch+verify+index attempt (see
    # `app/services/kb_gap_autofetch.py`). Opt-in for the same reason as
    # `kb_automation_enabled` above: this reaches out to the internet based
    # on live, unauthenticated user input, so an operator must deliberately
    # turn it on rather than have it start firing silently after an upgrade.
    kb_gap_autofetch_enabled: bool = False
    kb_gap_autofetch_interval_minutes: int = 15
    kb_gap_autofetch_max_attempts: int = 3
    # When the strict-RAG path finds nothing `review_status=approved`, this
    # additionally looks (read-only, same query) among `needs_review` chunks
    # for a possibly-relevant excerpt and discloses it -- quoted verbatim,
    # never composed or interpreted by the LLM, with the refusal and an
    # explicit "not verified" warning kept in front of it. Never upgrades
    # anything to `approved`/`verified` and never affects the confident-answer
    # path; see `ChatService._unverified_candidate_excerpt`. Exists so a
    # large, human-review-limited backlog (many States' worth of scraped
    # sources) fails toward "here is an unverified excerpt, confirm it
    # yourself" rather than either a bare refusal or a false "verified"
    # claim -- see the corrected KB audit's Real Gap discussion.
    chat_unverified_disclosure_enabled: bool = True
    # Controlled General Knowledge (GK) fallback (`app.core.gk_fallback`):
    # when the strict-RAG guardrail finds zero verified KB chunks, this lets
    # the LLM answer from general legal knowledge instead of the bare
    # refusal -- clearly labeled, disclaimed, never for a question asking
    # for a specific provision or the current/latest law, and discarded
    # outright if the model's answer still contains a citation-shaped
    # fragment or a self-rated LOW confidence. Opt-in and off by default for
    # the same reason `kb_gap_autofetch_enabled` is: this is new,
    # user-visible behavior on live, unauthenticated input, and an operator
    # must deliberately turn it on rather than have it change existing
    # refusal responses after an upgrade.
    general_knowledge_fallback_enabled: bool = False
    # Optional comma-separated adapter names for a controlled rollout. Empty
    # means every registered adapter; unknown names fail startup construction.
    kb_automation_adapter_allowlist: str = ""
    # Keeps the ~80-Act official-source registry (`kb_official_source_sync.
    # SOURCES`) current without an operator having to remember to re-run
    # `scripts/sync_official_kb_sources.py` -- confirmed live 2026-09-22: the
    # registry had gone stale for 4 days with nobody noticing, because nothing
    # was checking it automatically. Enabled by default, unlike
    # `kb_automation_enabled`/`kb_gap_autofetch_enabled` above: it never acts
    # on live/untrusted input, only re-checks a small, fixed, admin-reviewed
    # list of official .gov.in/.nic.in URLs, and every newly-indexed source
    # still lands `needs_review` like every other path in this file.
    official_source_sync_enabled: bool = True
    official_source_sync_interval_hours: int = 24
    # Runs `kb_machine_verification.MachinePolicy` checks (BNS/BSA/Contract
    # Act/RTI/CPC, ...) periodically. Safe to enable by default for the SAME
    # reason `official_source_sync_enabled` is: the fail-closed byte-hash +
    # identity/applicability/commencement evidence bar is what does the
    # "reviewing" here, not a person clicking a button -- see `_review_
    # reasons` in `app/rag/kb_jurisdiction.py`. It can only ever raise a
    # document to `machine_verified`, never to plain `verified` (that
    # remains exclusively a human action -- `_resolve_verification`'s Rule
    # 3), and it already skips anything already human-`verified` (see
    # `MachineVerificationService.verify`'s guard, added 2026-09-22 after
    # this scheduler's manual predecessor was found overwriting one).
    # Extending coverage to a NEW Act still requires a human to source and
    # confirm the official commencement-notification URL -- this only
    # automates RE-checking Acts a human has already set up a policy for.
    kb_machine_verification_enabled: bool = True
    kb_machine_verification_interval_hours: int = 24
    # Phase-6 release scope. Expand this list in controlled batches; readiness
    # is evaluated against these jurisdictions rather than claiming all-India
    # readiness before the nationwide corpus exists.
    kb_production_required_jurisdictions: str = "UP,DL,MH,GA,DH"
    kb_release_evaluation_max_age_hours: int = 168
    kb_release_min_evaluation_score: float = 0.95
    kb_review_sla_hours: int = 72
    kb_canary_recheck_hours: int = 24
    kb_malware_scan_command: str = "clamscan"
    # Operational output (reconciliation manifests). Under `storage/`, which is
    # gitignored in full -- these manifests record real filenames and hashes.
    operations_output_dir: Path = Path("./storage/operations")
    session_ttl_seconds: int = 86_400
    rate_limit_per_minute: int = 60
    # Empty by default -- `RateLimitMiddleware` then keys strictly on the TCP
    # peer address, which is correct for a directly-exposed API but collapses
    # every real client onto one bucket behind a reverse proxy/load balancer
    # (the peer is always the proxy). Set to the proxy's/LB's own IP(s) to
    # make the middleware trust *that* peer's `X-Forwarded-For` header for
    # the real client IP -- never trust it from an unlisted peer, since any
    # client can otherwise forge the header to bypass or smear rate limits.
    trusted_proxy_ips: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    sentry_dsn: str = ""
    otel_exporter_otlp_endpoint: str = ""
    secrets_encryption_key: str = ""
    background_job_poll_seconds: int = 1
    backup_dir: Path = Path("./storage/backups")
    # Part 58 "Answer Quality Audit" issue 25: how long the two analytics
    # collections that carry user message text (`query_logs`, `intent_events`)
    # are kept. Both are unbounded by construction -- they exist to be queried
    # across sessions -- so without a retention bound a drafting-heavy
    # deployment accumulates chat transcripts indefinitely. Redaction
    # (`app/utils/pii.py`) already strips identifiers from what goes in;
    # this bounds how long even the redacted record survives.
    #
    # Defaults to 0 = "keep forever", so an existing deployment's behaviour is
    # unchanged until an operator sets a policy. Applied at startup as a
    # MongoDB TTL index (see `ensure_retention_indexes`), so expiry happens in
    # the database rather than needing a sweeper process.
    analytics_retention_days: int = 0

    # --- E-Notarization ---------------------------------------------------
    # Names a provider registered in `app/notarization/esign/registry.py`.
    # "manual" (the default) applies NO electronic signature and says so; it
    # never reports a signature that did not happen.
    esign_provider: str = "manual"
    # Minutes an e-sign session stays valid before it is treated as expired.
    esign_session_ttl_minutes: int = 60
    # Seconds a signed notary approve/reject/revoke action token stays valid.
    # Short by design: it is minted only after a fresh re-authentication.
    notary_action_token_ttl_seconds: int = 300
    # Per-IP ceilings for the two abusable notarization flows, per minute.
    # The public verification endpoint is unauthenticated and enumerable in
    # principle, so it is limited independently of the global API limit.
    verification_rate_limit_per_minute: int = 30
    signing_rate_limit_per_minute: int = 10
    # Public base URL a notarized document's QR code points at.
    verification_base_url: str = "http://localhost:8000"

    draft_output_dir: Path = Path("./storage/drafts")
    draft_max_facts_chars: int = 6000
    # Wall-clock ceiling for ONE draft generation, across the first drafting
    # call and every expansion pass after it (`LegalDraftEngine
    # ._render_sections`). Nothing bounded that total before: with
    # `llm_timeout` at 120s, three sequential calls -- each retried up to
    # `llm_max_retry_attempts` times by `ResilientLLMProvider` -- could keep a
    # single `POST /chat` running for many minutes. Held below the client
    # timeout so the request returns a real (if shorter) draft instead of
    # being cut off. Expansion is what gets sacrificed -- the first, complete
    # draft is always kept.
    draft_generation_budget_seconds: int = 100
    # Grounds draft generation in the ACTUAL text of the Acts/sections a
    # template hints at (`applicable_acts_hint`/`applicable_sections_hint`),
    # retrieved from the same shared Knowledge Base chat retrieval already
    # uses, instead of the model drafting from the hint names alone plus its
    # own parametric knowledge -- the gap that made a hallucinated or
    # substituted section number possible in the first place (see
    # `citation_audit.py`, which only catches that failure after the fact).
    # Advisory/best-effort: retrieval failing or timing out never blocks
    # drafting, it just falls back to hints-only, exactly as before this
    # existed.
    draft_legal_context_enabled: bool = True
    draft_legal_context_top_k: int = Field(default=3, ge=1, le=10)
    draft_legal_context_max_chars: int = Field(default=2500, ge=0)
    draft_legal_context_timeout_seconds: int = Field(default=12, ge=1, le=60)
    # "CONFIDENTIAL" doesn't fit a document meant to be signed and handed to a
    # landlord/tenant/police station/court -- it reads as an internal-document
    # label, not a legal-drafting-aid marker, and looked especially out of
    # place stamped across the signature block. "DRAFT" says what's actually
    # true (this hasn't been reviewed by an advocate yet, matching the
    # disclaimer already appended to every export) and is the conventional
    # watermark word for a document pending review/signature.
    draft_watermark_text: str = "DRAFT"
    draft_watermark_image_path: str = ""
    draft_signature_image_path: str = ""
    draft_signing_certificate_path: str = ""
    draft_signing_key_path: str = ""
    draft_signing_key_passphrase: str = ""
    draft_signing_reason: str = "Legal draft approval"
    document_analysis_max_chars: int = 30_000
    pdf_unicode_font_path: str = ""
    # Part 57 "Drafting Lifecycle Redesign": per-script-family overrides,
    # keyed by the internal script family name used in
    # `app/drafting/export.py` (e.g. "ol_chiki", "meetei_mayek",
    # "perso_arabic") -- lets an operator drop in Noto Sans Ol Chiki/Meetei
    # Mayek or Noto Nastaliq Urdu without any code change, unlike
    # `pdf_unicode_font_path` above (a single global override). Set via a
    # JSON object env var, e.g.
    # PDF_SCRIPT_FONT_PATHS='{"ol_chiki": "/fonts/NotoSansOlChiki-Regular.ttf"}'.
    pdf_script_font_paths: dict[str, str] = Field(default_factory=dict)
    # Part 53 "PDF Hindi Rendering + Professional Layout Audit": on Windows,
    # WeasyPrint's native GTK3 dependencies (Pango/Cairo/GDK-Pixbuf/
    # fontconfig) must be discoverable via `os.add_dll_directory()`, not just
    # `PATH` -- Python 3.8+'s safer default DLL search on Windows no longer
    # searches `PATH` to resolve a loaded DLL's OWN transitive dependencies.
    # Only needed on Windows; a Linux/Docker deployment installs these as
    # regular shared libraries via the system package manager instead, where
    # the dynamic linker's normal search path already covers this. Blank by
    # default (no directory added) so a host without GTK3 installed at all
    # still gets WeasyPrint's own clear "could not import external
    # libraries" error rather than a confusing path-not-found from this
    # setting pointing nowhere.
    gtk_runtime_bin_path: str = ""

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def voice_max_audio_bytes(self) -> int:
        return self.voice_max_audio_mb * 1024 * 1024

    def llm_fallback_provider_list(self) -> list[str]:
        """`llm_fallback_providers` parsed into ordered, lowercased names.

        Blank entries and duplicates are dropped so a stray trailing comma or
        a repeated name in the env var can't produce an empty provider name or
        a pointlessly retried provider.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for raw in (self.llm_fallback_providers or "").split(","):
            name = raw.strip().lower()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def __init__(self, **data: Any) -> None:
        """`**data: Any`, not `**data: object`: these are field VALUES forwarded
        to `BaseSettings.__init__`, whose signature declares each settings key
        with its own type. `object` made every one of them a mismatch (11 errors
        for one call) while checking nothing -- the real validation is pydantic's
        own, at runtime, against the field declarations above.
        """
        if BaseSettings is BaseModel:
            data = self._env_overrides() | data
        super().__init__(**data)
        self._validate_security()
        self._validate_budgets()
        self._validate_retrieval_backend()

    def _validate_budgets(self) -> None:
        """Keep the server deadline safely inside the caller's timeout.

        A bad deployment override must fail at startup instead of silently
        recreating the failure where Streamlit disconnects while `/chat` is
        still spending time on provider retries.
        """
        if self.chat_request_budget_seconds <= 0:
            raise RuntimeError("CHAT_REQUEST_BUDGET_SECONDS must be greater than zero.")
        if self.client_request_timeout_seconds <= 0:
            raise RuntimeError("CLIENT_REQUEST_TIMEOUT_SECONDS must be greater than zero.")
        if self.chat_request_budget_seconds > self.client_request_timeout_seconds - 5:
            raise RuntimeError(
                "CHAT_REQUEST_BUDGET_SECONDS must be at least 5 seconds below "
                "CLIENT_REQUEST_TIMEOUT_SECONDS so the API can return a graceful response."
            )

    def _validate_retrieval_backend(self) -> None:
        """QA session 2026-09-24 ("BUG-101"): live-confirmed that
        `vector_search_backend="local"` -- `MongoVectorStore._local_cosine_leg`,
        a full unindexed scan of up to `local_vector_scan_limit` (20,000)
        candidate documents including their full embedding vectors, scored in
        a worker thread -- collapses under even light concurrency (6
        simultaneous `/chat` calls measured at 100-197s each, with the BM25/
        Redis leg and long-term-memory Postgres writes timing out alongside
        it; see `CONCURRENCY_BUG_EVIDENCE_6way.log`). It exists as a
        degrade-gracefully fallback for local/community MongoDB during
        development (no Atlas Search index available), not as a production
        retrieval path -- the real fix is `vector_search_backend="atlas"`
        (`MongoVectorStore._atlas_vector_leg`, confirmed correct end-to-end
        against a genuine `$vectorSearch` engine via
        `scripts/_atlas_readiness_verification_20260923.py`: ~10ms per query
        vs. the local leg's 100,000+ms under the same load).

        Mirrors `_validate_security`'s pattern exactly: development/test stay
        unvalidated so local workflows keep working unchanged; production/
        staging refuse to start on the slow, concurrency-fragile fallback
        rather than silently shipping it.
        """
        if self.environment not in ("production", "staging"):
            return
        if self.vector_search_backend != "atlas":
            raise RuntimeError(
                f"Refusing to start with environment={self.environment!r}: "
                f"VECTOR_SEARCH_BACKEND={self.vector_search_backend!r}, not 'atlas'. "
                "The local vector-scan fallback is a development convenience, not a "
                "production retrieval path -- it collapses under light concurrent load "
                "(verified: 100+ second retrieval latency, BM25/Redis timeouts, and "
                "failed long-term-memory writes at just 6 concurrent requests). Configure "
                "a real Atlas Search vector index (see scripts/create_indexes.py and "
                "docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md) and set VECTOR_SEARCH_BACKEND=atlas "
                "before starting in this environment."
            )

    def _validate_security(self) -> None:
        """Phase 1 security hardening: refuses to start a production/staging
        process with a missing, placeholder, or too-short JWT_SECRET_KEY --
        a weak secret lets anyone forge access/refresh tokens for any role.
        Development/test are left unvalidated so local workflows and the
        existing test suite keep working unchanged (backward compatible).
        """
        if self.environment not in ("production", "staging"):
            return
        if self.jwt_secret_key.strip().lower() in _INSECURE_JWT_SECRETS or len(
            self.jwt_secret_key
        ) < _MIN_JWT_SECRET_LENGTH:
            raise RuntimeError(
                f"Refusing to start with environment={self.environment!r}: JWT_SECRET_KEY is missing, "
                f"a known placeholder, or shorter than {_MIN_JWT_SECRET_LENGTH} characters. Set a strong, "
                "unique JWT_SECRET_KEY (e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`) "
                "via the environment or .env before starting in this environment."
            )
        if (len(self.secrets_encryption_key.strip()) < 32
                or self.secrets_encryption_key.strip().lower() in _INSECURE_JWT_SECRETS):
            raise RuntimeError(
                "Refusing to start production/staging without SECRETS_ENCRYPTION_KEY of at least 32 characters. "
                "Load it from a deployment secret manager; never commit it to .env or source control."
            )
        if any("*" in str(origin) for origin in self.api_cors_origins):
            raise RuntimeError("Production/staging CORS origins must be explicit; wildcards are forbidden.")

    def _env_overrides(self) -> dict[str, object]:
        overrides: dict[str, object] = {}
        for field_name in self.model_fields:
            env_name = field_name.upper()
            if env_name not in os.environ:
                continue
            value: object = os.environ[env_name]
            if field_name == "api_cors_origins":
                value = json.loads(str(value))
            elif field_name in {
                "api_port",
                "access_token_expire_minutes",
                "refresh_token_expire_days",
                "max_upload_mb",
                "session_ttl_seconds",
                "rate_limit_per_minute",
                "indexing_batch_size",
                "embedding_dimensions",
                "claude_max_tokens",
                "draft_max_facts_chars",
                "draft_generation_budget_seconds",
                "draft_legal_context_top_k",
                "draft_legal_context_max_chars",
                "draft_legal_context_timeout_seconds",
                "document_analysis_max_chars",
                "llm_timeout",
                "llm_max_retry_attempts",
                "client_request_timeout_seconds",
                "chat_request_budget_seconds",
                "llm_max_tokens",
                "top_k",
                "retrieval_rrf_k",
                "analytics_retention_days",
                "background_job_poll_seconds",
            }:
                value = int(str(value))
            elif field_name in {
                "min_ocr_quality_score", "min_metadata_quality_score", "temperature", "top_p",
                "conversation_intent_llm_min_confidence",
                "llm_retry_backoff_seconds", "ocr_min_confidence",
            }:
                value = float(str(value))
            elif field_name in {
                "ocr_enabled", "retrieval_debug", "conversation_intent_llm_enabled",
                "ocr_low_confidence_gemini_fallback", "draft_legal_context_enabled",
            }:
                value = str(value).strip().lower() in {"1", "true", "yes", "on"}
            overrides[field_name] = value
        return overrides


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
