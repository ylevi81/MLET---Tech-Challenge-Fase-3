"""Exercise the running Compose stack and record evidence from its HTTP APIs."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch(url: str, *, payload: dict | None = None, auth: str | None = None):
    headers = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if auth:
        headers["Authorization"] = f"Basic {auth}"
    with urlopen(Request(url, data=data, headers=headers), timeout=15) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/stack-validation.json"))
    args = parser.parse_args()
    health = fetch("http://127.0.0.1:8000/health")
    assert health["model_loaded"], health
    for _ in range(40):
        prediction = fetch(
            "http://127.0.0.1:8000/predict",
            payload={"text": "A patient with chest discomfort underwent cardiac evaluation."},
        )
        assert prediction["artifact_id"] == health["artifact_id"]
        time.sleep(0.5)  # Allow Prometheus to scrape while requests are generated.
    try:
        fetch("http://127.0.0.1:8000/predict", payload={"text": ""})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Expected validation error")
    time.sleep(6)

    queries = {
        "scrape_up": 'up{job="medical-api"}',
        "successful_predictions": 'sum(http_requests_total{route="/predict",status_code="200"})',
        "validation_errors": 'sum(http_requests_total{route="/predict",status_code="422"})',
        "mean_seconds": 'sum(rate(http_request_duration_seconds_sum{route="/predict"}[1m]))'
        ' / sum(rate(http_request_duration_seconds_count{route="/predict"}[1m]))',
        "p95_seconds": "histogram_quantile(0.95, sum by (le) "
        '(rate(http_request_duration_seconds_bucket{route="/predict"}[1m])))',
    }
    measured = {}
    for name, query in queries.items():
        result = fetch("http://127.0.0.1:9090/api/v1/query?" + urlencode({"query": query}))
        assert result["status"] == "success" and result["data"]["result"], result
        value = float(result["data"]["result"][0]["value"][1])
        assert math.isfinite(value) and value > 0, (name, value)
        measured[name] = value

    user = os.environ.get("GRAFANA_ADMIN_USER", "admin")
    password = os.environ.get("GRAFANA_ADMIN_PASSWORD", "admin")
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    dashboard = fetch("http://127.0.0.1:3000/api/dashboards/uid/medical-api", auth=auth)
    datasource = fetch("http://127.0.0.1:3000/api/datasources/uid/prometheus/health", auth=auth)
    assert datasource["status"] == "OK", datasource
    assert dashboard["meta"]["provisioned"], dashboard["meta"]
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "api_health": health,
        "prometheus": measured,
        "grafana": {
            "dashboard_uid": dashboard["dashboard"]["uid"],
            "panels": len(dashboard["dashboard"]["panels"]),
            "provisioned": True,
            "datasource_status": datasource["status"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
