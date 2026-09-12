"""Measure end-to-end HTTP latency for the prediction endpoint.

The script is intentionally not run during installation or tests. A result file is
created only when this command is executed against a live API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_TEXT = (
    "The patient presented with persistent chest discomfort and shortness of breath. "
    "Electrocardiography and laboratory testing were requested for further evaluation."
)


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without external dependencies."""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    """Build stable latency summary statistics in milliseconds."""
    return {
        "min_ms": round(min(values), 3),
        "mean_ms": round(fmean(values), 3),
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "max_ms": round(max(values), 3),
    }


def predict(url: str, text: str, timeout: float) -> tuple[float, dict[str, Any]]:
    """Send one prediction request and return client latency plus decoded JSON."""
    body = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} returned by {url}: {details}") from exc
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if "classification" not in payload or "probabilities" not in payload:
        raise RuntimeError("prediction response does not match the documented API schema")
    return elapsed_ms, payload


def run_benchmark(
    *, url: str, text: str, requests_count: int, warmup: int, timeout: float
) -> dict[str, Any]:
    """Run sequential warm-up and measured requests against a live endpoint."""
    if requests_count < 1:
        raise ValueError("requests_count must be at least 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    for _ in range(warmup):
        predict(url, text, timeout)

    client_latencies: list[float] = []
    server_latencies: list[float] = []
    classifications: dict[str, int] = {}
    started = time.perf_counter()
    for _ in range(requests_count):
        client_ms, response = predict(url, text, timeout)
        client_latencies.append(client_ms)
        server_ms = response.get("latency_ms")
        if isinstance(server_ms, int | float):
            server_latencies.append(float(server_ms))
        classification = str(response["classification"])
        classifications[classification] = classifications.get(classification, 0) + 1
    wall_seconds = time.perf_counter() - started

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_url": url,
        "mode": "sequential",
        "warmup_requests": warmup,
        "measured_requests": requests_count,
        "client_latency": summarize(client_latencies),
        "throughput_requests_per_second": round(requests_count / wall_seconds, 3),
        "classifications": classifications,
    }
    if server_latencies:
        result["server_inference_latency"] = summarize(server_latencies)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--requests", type=int, default=100, dest="requests_count")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        url=args.url,
        text=args.text,
        requests_count=args.requests_count,
        warmup=args.warmup,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
