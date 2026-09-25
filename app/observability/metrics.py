import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Self

# Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]* -- this
# codebase's own metric names use "." and "/" freely (e.g.
# "http.GET./chat/stream"), so they need sanitizing before they can be
# exposed in Prometheus text format at all.
_INVALID_PROMETHEUS_CHARS = re.compile(r"[^a-zA-Z0-9_:]")


def _prometheus_metric_name(name: str) -> str:
    sanitized = _INVALID_PROMETHEUS_CHARS.sub("_", name)
    return sanitized if re.match(r"^[a-zA-Z_:]", sanitized) else f"_{sanitized}"


@dataclass
class MetricStore:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe_ms(self, name: str, value: float) -> None:
        self.timings[name].append(value)
        self.timings[name] = self.timings[name][-500:]

    def snapshot(self) -> dict[str, Any]:
        timing_snapshot = {}
        for name, values in self.timings.items():
            if not values:
                continue
            timing_snapshot[name] = {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 2),
                "max_ms": round(max(values), 2),
            }
        return {"counters": dict(self.counters), "timings": timing_snapshot}

    def to_prometheus_text(self) -> str:
        """Prometheus exposition format over this same in-process store.

        This does NOT make metrics survive a restart by itself -- counters
        still reset to zero on every process start, same as `snapshot()`.
        What it fixes is "no persistent export": scrape this with an actual
        Prometheus server (or any compatible agent) on an interval, and
        history lives in ITS time-series storage, external to and surviving
        this process's restarts -- the standard way this problem is solved,
        rather than teaching this in-process store to persist itself.
        """
        lines: list[str] = []
        for name, value in sorted(self.counters.items()):
            metric = _prometheus_metric_name(name)
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        for name, values in sorted(self.timings.items()):
            if not values:
                continue
            metric = _prometheus_metric_name(name)
            lines.append(f"# TYPE {metric}_count gauge")
            lines.append(f"{metric}_count {len(values)}")
            lines.append(f"# TYPE {metric}_avg_ms gauge")
            lines.append(f"{metric}_avg_ms {sum(values) / len(values):.2f}")
            lines.append(f"# TYPE {metric}_max_ms gauge")
            lines.append(f"{metric}_max_ms {max(values):.2f}")
        return "\n".join(lines) + "\n"


metrics = MetricStore()


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "Self":
        self.started = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        metrics.observe_ms(self.name, (perf_counter() - self.started) * 1000)
