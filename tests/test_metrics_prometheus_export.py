"""`MetricStore` was in-memory-only with no way to export it -- counters
reset on every restart and there was no persistent history anywhere
(`GET /admin/phase3/metrics` just returns the same in-process snapshot as
JSON). `to_prometheus_text()` doesn't add persistence to this store itself;
it exposes the same counters in Prometheus's text exposition format so an
actual Prometheus server (or compatible agent) can scrape and retain history
externally -- the standard fix for "metrics don't survive a restart".
"""
import asyncio

from app.observability.metrics import MetricStore, _prometheus_metric_name


def test_metric_names_with_dots_and_slashes_are_sanitized() -> None:
    # Prometheus names must match [a-zA-Z_:][a-zA-Z0-9_:]* -- this
    # codebase's own metric names ("http.GET./chat/stream") use both
    # characters that pattern forbids.
    assert _prometheus_metric_name("http.GET./chat/stream") == "http_GET__chat_stream"


def test_a_name_starting_with_a_digit_gets_a_valid_prefix() -> None:
    # Prometheus names may not start with a digit either -- "200.ok" would
    # sanitize its dot fine but still be an invalid name without this check.
    name = _prometheus_metric_name("200.ok")
    assert name[0].isalpha() or name[0] in "_:"


def test_counters_render_as_prometheus_counter_lines() -> None:
    store = MetricStore()
    store.increment("http.status.200", 5)

    text = store.to_prometheus_text()

    assert "# TYPE http_status_200 counter" in text
    assert "http_status_200 5" in text


def test_timings_render_as_count_avg_and_max_gauges() -> None:
    store = MetricStore()
    store.observe_ms("chat.latency", 10.0)
    store.observe_ms("chat.latency", 30.0)

    text = store.to_prometheus_text()

    assert "chat_latency_count 2" in text
    assert "chat_latency_avg_ms 20.00" in text
    assert "chat_latency_max_ms 30.00" in text


def test_empty_store_still_produces_valid_trailing_newline() -> None:
    assert MetricStore().to_prometheus_text() == "\n"


def test_prometheus_metrics_route_returns_the_same_underlying_counters() -> None:
    from app.api import admin_phase3

    admin_phase3.metrics.increment("route_smoke_test_counter")

    response = asyncio.run(admin_phase3.application_metrics_prometheus())

    assert response.media_type == "text/plain; version=0.0.4"
    assert "route_smoke_test_counter" in response.body.decode()
