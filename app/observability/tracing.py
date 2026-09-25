"""OpenTelemetry distributed tracing setup.

Off by default: `settings.otel_exporter_otlp_endpoint` already existed as a
config field before this module did, but nothing anywhere read it -- the
OpenTelemetry SDK wasn't even a dependency. Tracing only activates when an
endpoint is actually configured, so a deployment that hasn't set one up
pays no cost and sends no data anywhere; `instrument_app` (FastAPI/pymongo/
redis auto-instrumentation) still runs either way so spans exist locally,
they just have nowhere to export to until an endpoint is set.
"""
from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

log = structlog.get_logger(__name__)

_instrumented = False


def setup_tracing(app: "FastAPI") -> None:
    """Called once from `create_app()`. Idempotent so repeated app
    construction in tests doesn't double-instrument pymongo/redis, which
    OpenTelemetry's instrumentors raise on.
    """
    global _instrumented
    if _instrumented:
        return
    _instrumented = True

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: settings.app_name}))
    if settings.otel_exporter_otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
        )
        log.info("tracing_enabled", endpoint=settings.otel_exporter_otlp_endpoint)
    else:
        log.info("tracing_disabled_no_endpoint")
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    # Motor (this project's Mongo driver) wraps pymongo, and pymongo's own
    # instrumentation already patches at the level Motor's calls pass
    # through -- no separate Motor instrumentor exists or is needed.
    PymongoInstrumentor().instrument(tracer_provider=provider)
    RedisInstrumentor().instrument(tracer_provider=provider)
