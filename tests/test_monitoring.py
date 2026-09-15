from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from medical_triage.api import create_app


def samples(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]


def test_metrics_count_success_validation_and_missing_routes(trained_bundle):
    with TestClient(create_app(bundle=trained_bundle)) as client:
        assert client.post("/predict", json={"text": "cardiac heart disease"}).status_code == 200
        assert client.post("/predict", json={"text": ""}).status_code == 422
        client.get("/patient/private-id")
        client.get("/patient/another-private-id")
        observed = samples(client)
        counts = {
            (s.labels["route"], s.labels["status_code"]): s.value
            for s in observed
            if s.name == "http_requests_total"
        }
        assert counts == {("/predict", "200"): 1, ("/predict", "422"): 1, ("unmatched", "404"): 2}
        durations = [s for s in observed if s.name == "http_request_duration_seconds_count"]
        assert sum(s.value for s in durations) == 4
        assert samples(client) == observed  # Scrapes do not count themselves.


def test_metrics_remain_available_without_model(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"text": "heart disease"}).status_code == 503
        observed = samples(client)
        assert sum(s.value for s in observed if s.name == "http_requests_total") == 2


def test_unhandled_error_is_counted_and_registries_are_isolated(trained_bundle):
    application = create_app(bundle=trained_bundle)

    @application.get("/failure")
    def fail():
        raise RuntimeError("test failure")

    with TestClient(application, raise_server_exceptions=False) as client:
        assert client.get("/failure").status_code == 500
        assert any(
            s.name == "http_requests_total" and s.labels["status_code"] == "500"
            for s in samples(client)
        )
    with TestClient(create_app(bundle=trained_bundle)) as client:
        assert not any(s.name == "http_requests_total" for s in samples(client))
