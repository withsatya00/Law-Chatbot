import time
from typing import Any

import structlog
from fastapi import APIRouter, Response

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.optional_deps import missing_optional_dependencies
from app.database.mongodb import mongodb
from app.database.postgresql import postgresql
from app.llm.factory import LLMFactory

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])
STARTED_AT = time.perf_counter()


@router.get("/health")
async def health() -> dict[str, Any]:
    started = time.perf_counter()
    db_status = await mongodb.ping()
    postgres_status = await postgresql.ping()
    redis_status = await redis_client.ping()
    provider = LLMFactory.create()
    llm_started = time.perf_counter()
    llm_status = await provider.health()
    llm_latency_ms = round((time.perf_counter() - llm_started) * 1000, 2)
    return {
        "status": "ok" if db_status and postgres_status and redis_status else "degraded",
        "components": {
            "database": "ok" if db_status else "unavailable",
            "postgresql": "ok" if postgres_status else "unavailable",
            "redis": "ok" if redis_status else "unavailable",
            "vector_db": "ok" if db_status else "unavailable",
            "llm": "ok" if llm_status else "not_configured",
            "embedding": "ok",
        },
        "llm_provider": {
            "name": settings.llm_provider,
            "model": getattr(provider, "model", None),
            "status": "ok" if llm_status else "unavailable",
            "latency_ms": llm_latency_ms,
        },
        # Phase 1 item 8: which optional document-format libraries this host
        # is missing. Chat and drafting no longer break when one is absent
        # (they are bound lazily now), so the only way an operator would
        # otherwise learn that PDF upload or DOCX export is unavailable here
        # is a user hitting it. Each entry names the feature and the install
        # command. An empty object means everything is present.
        "optional_features_unavailable": missing_optional_dependencies(),
        "server_uptime_seconds": round(time.perf_counter() - STARTED_AT, 2),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/index-drift")
async def index_drift() -> dict[str, Any]:
    """Whether the BM25 index still agrees with MongoDB.

    Read-only and safe to poll. Drift here is not a service outage -- retrieval
    still answers -- so it is reported as `degraded` rather than failing the
    endpoint. It matters because a stale BM25 entry is a retrievable copy of
    text MongoDB no longer has, and `stale_private_records` counts the subset
    of those that carry owner metadata, which is the privacy-relevant case.

    Fix with `scripts/reconcile_indexes.py --apply`.
    """
    from app.rag.reconciliation import IndexReconciler

    try:
        report = (await IndexReconciler().analyze()).as_dict()
    except Exception as exc:  # noqa: BLE001 - a diagnostic endpoint must not 500
        log.warning("index_drift_check_failed", error=str(exc), error_type=type(exc).__name__)
        return {"status": "unknown", "error": type(exc).__name__}
    return {"status": "degraded" if report["drifted"] else "ok", **report}


@router.get("/health/ready")
async def readiness(response: Response) -> dict[str, Any]:
    database = await mongodb.ping()
    postgres = await postgresql.ping()
    redis = await redis_client.ping()
    if not (database and postgres and redis):
        response.status_code = 503
    ready = database and postgres and redis
    return {"status": "ready" if ready else "not_ready", "database": database, "postgresql": postgres, "redis": redis}
