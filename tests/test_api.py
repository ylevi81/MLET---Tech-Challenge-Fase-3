from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from medical_triage.api import create_app


def test_health_and_prediction_contract(trained_bundle: dict) -> None:
    with TestClient(create_app(bundle=trained_bundle)) as client:
        health = client.get("/health")
        response = client.post(
            "/predict",
            json={"text": "Cardiac artery and vascular heart findings were evaluated."},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"] in payload["probabilities"]
    assert payload["predicted_labels"]
    assert set(payload["probabilities"]) == {
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    }
    assert payload["latency_ms"] >= 0
    assert payload["model_version"]


def test_api_rejects_blank_text(trained_bundle: dict) -> None:
    with TestClient(create_app(bundle=trained_bundle)) as client:
        response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422


def test_api_is_not_ready_without_artifact(tmp_path: Path) -> None:
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"text": "Valid medical abstract text."})

    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert prediction.status_code == 503

