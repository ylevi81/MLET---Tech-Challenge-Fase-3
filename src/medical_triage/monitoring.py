"""HTTP metrics with bounded labels and one registry per application."""

from __future__ import annotations

import time

from prometheus_client import CollectorRegistry, Counter, Histogram


class HTTPMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        labels = ("method", "route", "status_code")
        self.requests = Counter(
            "http_requests_total", "Completed HTTP requests", labels, registry=self.registry
        )
        self.duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration including response transmission",
            labels,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )


class MetricsMiddleware:
    """Record errors too, excluding scrapes and arbitrary URL labels."""

    def __init__(self, app, metrics: HTTPMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/metrics":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        status = 500

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"]
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                method = "OTHER"
            labels = (method, route, str(status))
            self.metrics.requests.labels(*labels).inc()
            self.metrics.duration.labels(*labels).observe(time.perf_counter() - started)
