import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    admin,
    admin_phase3,
    analytics,
    auth,
    cases,
    chat,
    document_analysis,
    drafting,
    feedback,
    health,
    history,
    intent,
    notarization,
    phase2,
    phase3,
    recommend,
    search,
    summarize,
    upload,
    voice_router,
)
from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import install_exception_handlers
from app.core.logger import configure_logging
from app.core.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.database.mongodb import ensure_retention_indexes, mongodb
from app.database.postgresql import postgresql
from app.notarization.migrations import ensure_notarization_indexes
from app.observability.metrics import metrics
from app.observability.tracing import setup_tracing
from app.rag.bm25_index import bm25_index
from app.rag.incremental import recover_pending_reindex_jobs
from app.rag.local_ann_index import local_ann_index
from app.services.kb_automation import kb_automation_scheduler
from app.services.kb_gap_autofetch import gap_autofetch_scheduler
from app.services.kb_indexing_queue import kb_indexing_queue, recover_pending_jobs
from app.services.kb_machine_verification import machine_verification_scheduler
from app.services.kb_official_source_sync import official_source_sync_scheduler

# NOTE: startup deliberately does NOT scan `settings.upload_storage_dir`.
# Until this was removed, every API start kicked off a backfill that copied
# every PDF in the users' private uploads directory into the shared Knowledge
# Base and indexed it -- hundreds of files, on every restart, with nobody
# authorizing any of it. Promoting an upload is now an explicit, admin-only
# action (`POST /admin/knowledge-base/backfill-uploads?confirm=true`), and even
# that only quarantines candidates as `needs_review`. See
# `KnowledgeBaseIngestionService.backfill_from_uploads`.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log = structlog.get_logger(__name__)
    await mongodb.connect()
    await postgresql.connect()
    await redis_client.connect()
    try:
        # Part 40 section 13: loaded once here so no request pays the cost
        # of building it -- `ensure_loaded()` is also called defensively on
        # every search, so a failure here just means the first real query
        # loads it instead of blocking startup.
        await bm25_index.ensure_loaded()
    except Exception as exc:  # noqa: BLE001 - a BM25 preload failure just defers the load to the first query; it must not stop the app booting
        log.warning("bm25_index_startup_load_failed", error=str(exc))
    try:
        # QA pass 2026-09-24 (60k-chunk local-scan latency): same reasoning
        # as the BM25 preload above, for the local ANN fallback -- a cold
        # build (or disk-cache load) here instead of inside the first real
        # `/chat` request. `_local_cosine_leg` still falls back to the exact
        # brute-force scan if this preload fails, so a failure here degrades
        # latency, not correctness.
        await local_ann_index.ensure_current_generation()
    except Exception as exc:  # noqa: BLE001 - see the BM25 preload comment immediately above; same reasoning
        log.warning("local_ann_index_startup_load_failed", error=str(exc))
    try:
        # Same reasoning as the BM25 preload immediately above, for the other
        # half of retrieval: `EmbeddingProvider._get_model()` downloads/loads
        # the sentence-transformers model (BAAI/bge-m3, ~2GB) into memory on
        # first use and caches it at the class level for the rest of the
        # process -- but nothing forced that first use to happen here, so it
        # happened inside the first real `/chat` request instead. Confirmed
        # live: a fresh restart's first request measured 17s inside
        # `entities_retrieval_recommendation_ms` alone for
        # `embedding_model_loading` -> `embedding_model_loaded`, on top of
        # ordinary retrieval time, for a user who did nothing but be first.
        # `asyncio.to_thread` because loading a torch model is synchronous,
        # CPU/GPU-bound work that would otherwise block the startup event
        # loop for the same 17s.
        from app.rag.embeddings import EmbeddingProvider

        await asyncio.to_thread(EmbeddingProvider()._get_model)
    except Exception as exc:  # noqa: BLE001 - an embedding-model preload failure just defers the load to the first query; it must not stop the app booting
        log.warning("embedding_model_startup_load_failed", error=str(exc))
    try:
        # Part 58 issue 25: applies `analytics_retention_days` as a TTL index
        # on the two collections that retain user message text. No-op unless
        # an operator has configured a window.
        retained = await ensure_retention_indexes()
        if retained:
            log.info(
                "analytics_retention_indexes_applied",
                collections=retained,
                retention_days=settings.analytics_retention_days,
            )
    except Exception as exc:  # noqa: BLE001 - index setup is idempotent and retried next boot; it must not stop the app booting
        log.warning("analytics_retention_index_setup_failed", error=str(exc))
    try:
        # E-Notarization indexes, including the UNIQUE constraint on
        # `verification_token` that keeps one QR code resolving to exactly one
        # document. Additive: creates only new collections' indexes.
        notarization_indexes = await ensure_notarization_indexes()
        if notarization_indexes:
            log.info("notarization_indexes_applied", indexes=notarization_indexes)
    except Exception as exc:  # noqa: BLE001 - index setup is idempotent and retried next boot; it must not stop the app booting
        log.warning("notarization_index_setup_failed", error=str(exc))
    try:
        # Background indexing queue recovery: requeues kb_staging_records left
        # at status="processing" by an interrupted process, reusing the
        # existing record + staged file rather than re-uploading.
        requeued = await recover_pending_jobs()
        if requeued:
            log.info("kb_indexing_queue_recovered", requeued=requeued)
    except Exception as exc:  # noqa: BLE001 - queue recovery is best-effort; it must not stop the app booting
        log.warning("kb_indexing_queue_recovery_failed", error=str(exc))
    try:
        # `/admin/reindex` jobs run on FastAPI `BackgroundTasks`, which is
        # in-process only -- a crash/restart mid-job otherwise leaves it
        # stuck at status="queued"/"running" forever with nothing to resume
        # it. Safe to just re-run: see `recover_pending_reindex_jobs`.
        reindex_recovered = await recover_pending_reindex_jobs()
        if reindex_recovered:
            log.info("incremental_reindex_recovered", recovered=reindex_recovered)
    except Exception as exc:  # noqa: BLE001 - reindex recovery is best-effort; it must not stop the app booting
        log.warning("incremental_reindex_recovery_failed", error=str(exc))
    kb_automation_scheduler.start()
    gap_autofetch_scheduler.start()
    official_source_sync_scheduler.start()
    machine_verification_scheduler.start()
    log.info("application_started", app=settings.app_name, environment=settings.environment)
    try:
        yield
    finally:
        await machine_verification_scheduler.close()
        await official_source_sync_scheduler.close()
        await gap_autofetch_scheduler.close()
        await kb_automation_scheduler.close()
        await kb_indexing_queue.close()
        await redis_client.close()
        await postgresql.close()
        await mongodb.close()
        log.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Enterprise Legal AI Assistant backend with RAG and document analysis.",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    async def api_root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "status": "running",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/internal/metrics", include_in_schema=False)
    async def internal_metrics() -> Response:
        """Unauthenticated Prometheus scrape target -- deliberately separate
        from the admin-guarded `GET /admin/phase3/metrics/prometheus`
        (same underlying counters, `app.observability.metrics.metrics`).
        Prometheus scraping a short-lived admin JWT on every scrape interval
        isn't practical, so this exists instead -- SECURITY: never expose
        this path outside the deployment's own network/reverse-proxy trust
        boundary (Docker Compose's default bridge network already keeps it
        unreachable from outside the host unless a proxy explicitly forwards
        it; see docker/docker-compose.observability.yml).
        """
        return Response(content=metrics.to_prometheus_text(), media_type="text/plain; version=0.0.4")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.api_cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(admin_phase3.router)
    app.include_router(analytics.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(upload.router)
    app.include_router(search.router)
    app.include_router(history.router)
    app.include_router(summarize.router)
    app.include_router(recommend.router)
    app.include_router(document_analysis.router)
    app.include_router(intent.router)
    app.include_router(feedback.router)
    app.include_router(drafting.router)
    app.include_router(voice_router.router)
    app.include_router(cases.router)
    app.include_router(phase2.router)
    app.include_router(phase3.router)
    # E-Notarization. The public `/verify/{token}` endpoint lives on the
    # same router; `notarization.admin_router` is separately admin-guarded.
    app.include_router(notarization.router)
    app.include_router(notarization.admin_router)
    setup_tracing(app)
    return app


app = create_app()
