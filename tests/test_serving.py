import numpy as np
import pytest
from fastapi.testclient import TestClient

from medical_triage.api import create_app
from medical_triage.serving import prepare


@pytest.fixture(scope="module")
def serving_models(trained_artifacts, tmp_path_factory):
    output = tmp_path_factory.mktemp("serving-models")
    prepare(trained_artifacts[0], output)
    return output


@pytest.mark.parametrize(
    "backend,filename,tolerance",
    [
        ("onnx_fp32", "model.onnx", 1e-4),
        ("onnx_int8", "model_int8.onnx", 0.03),
    ],
)
def test_endpoints_use_selected_runtime(
    trained_artifacts, serving_models, backend, filename, tolerance
):
    model_path = trained_artifacts[0]
    text = "The cardiac patient has heart disease and vascular inflammation"
    with TestClient(create_app(model_path=model_path, backend="sklearn")) as baseline:
        expected = baseline.post("/predict", json={"text": text, "threshold": 1}).json()
    application = create_app(
        model_path=model_path, backend=backend, onnx_path=serving_models / filename
    )
    with TestClient(application) as client:
        assert client.get("/health").json()["inference_backend"] == backend
        response = client.post("/predict", json={"text": text, "threshold": 1})
        assert response.status_code == 200
        actual = response.json()
        assert actual.keys() == expected.keys()
        assert actual["artifact_id"] == expected["artifact_id"]
        assert actual["classification"] == expected["classification"]
        assert actual["predicted_labels"] == expected["predicted_labels"]
        assert actual["threshold"] == 1
        np.testing.assert_allclose(
            list(actual["probabilities"].values()),
            list(expected["probabilities"].values()),
            atol=tolerance,
        )
        assert (
            application.state.model_store.bundle["pipeline"].__class__.__name__ == "ONNXPredictor"
        )


@pytest.mark.parametrize(
    "backend,filename",
    [
        ("onnx_int8", "missing.onnx"),
        ("onnx_int8", "model.onnx"),  # Wrong backend metadata must not be accepted.
        ("invalid", "model.onnx"),
    ],
)
def test_invalid_runtime_fails_closed(trained_artifacts, serving_models, backend, filename):
    with TestClient(
        create_app(
            model_path=trained_artifacts[0], backend=backend, onnx_path=serving_models / filename
        )
    ) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"text": "cardiac disease"}).status_code == 503
        assert client.get("/metrics").status_code == 200


def test_stale_model_is_rejected(trained_bundle, serving_models):
    from medical_triage.serving import configure_backend

    stale = {**trained_bundle, "artifact_id": "different-training-run"}
    with pytest.raises(ValueError, match="metadata"):
        configure_backend(stale, "onnx_int8", serving_models / "model_int8.onnx")


def test_prepare_reuses_matching_artifacts(trained_artifacts, serving_models):
    before = (serving_models / "model_int8.onnx").stat().st_mtime_ns
    prepare(trained_artifacts[0], serving_models)
    assert (serving_models / "model_int8.onnx").stat().st_mtime_ns == before
