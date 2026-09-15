"""FastAPI application for low-latency medical abstract classification."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from medical_triage import __version__
from medical_triage.model import load_model, predict_abstracts, validate_model_bundle
from medical_triage.monitoring import HTTPMetrics, MetricsMiddleware
from medical_triage.serving import configure_backend

DEFAULT_MODEL_PATH = "artifacts/model.joblib"


class PredictRequest(BaseModel):
    """A single abstract and an optional multilabel decision threshold."""

    text: Annotated[str, Field(min_length=1, max_length=20_000)]
    threshold: Annotated[float | None, Field(ge=0.0, le=1.0)] = None

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value


class PredictedLabel(BaseModel):
    """One selected medical condition."""

    id: int
    name: str
    probability: float


class PredictResponse(BaseModel):
    """Multilabel inference result."""

    artifact_id: str
    model_version: str
    classification: str
    predicted_labels: list[str]
    probabilities: dict[str, float]
    labels: list[PredictedLabel]
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    """Readiness state exposed to orchestrators and load balancers."""

    status: str
    service: str
    version: str
    model_loaded: bool
    inference_backend: str = "sklearn"
    artifact_id: str | None = None
    labels: int | None = None
    error: str | None = None


class ModelStore:
    """Thread-safe holder for a read-only model bundle."""

    def __init__(
        self,
        model_path: Path,
        bundle: dict[str, Any] | None = None,
        *,
        backend: str = "sklearn",
        onnx_path: Path,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.onnx_path = onnx_path
        self._lock = RLock()
        self._bundle: dict[str, Any] | None = None
        self.error: str | None = None
        if bundle is not None:
            self._bundle = configure_backend(validate_model_bundle(bundle), backend, onnx_path)

    @property
    def bundle(self) -> dict[str, Any] | None:
        with self._lock:
            return self._bundle

    def load(self) -> None:
        with self._lock:
            try:
                self._bundle = configure_backend(
                    load_model(self.model_path), self.backend, self.onnx_path
                )
                self.error = None
            except Exception as exc:
                self._bundle = None
                self.error = f"{type(exc).__name__}: {exc}"


def create_app(
    *,
    model_path: str | Path | None = None,
    bundle: dict[str, Any] | None = None,
    backend: str | None = None,
    onnx_path: str | Path | None = None,
) -> FastAPI:
    """Create an application; tests may inject a validated in-memory bundle."""

    configured_path = Path(
        model_path or os.environ.get("MEDICAL_TRIAGE_MODEL_PATH", DEFAULT_MODEL_PATH)
    ).expanduser()
    configured_backend = backend or os.environ.get("MEDICAL_TRIAGE_BACKEND", "sklearn")
    default_onnx = "model_int8.onnx" if configured_backend == "onnx_int8" else "model.onnx"
    configured_onnx = Path(
        onnx_path
        or os.environ.get("MEDICAL_TRIAGE_ONNX_PATH", str(configured_path.with_name(default_onnx)))
    ).expanduser()
    store = ModelStore(
        configured_path, bundle=bundle, backend=configured_backend, onnx_path=configured_onnx
    )
    metrics = HTTPMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if store.bundle is None:
            store.load()
        yield

    application = FastAPI(
        title="Medical Abstracts Classifier",
        description="Classificação multilabel de resumos médicos.",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.model_store = store
    application.add_middleware(MetricsMiddleware, metrics=metrics)

    @application.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        current = store.bundle
        if current is None:
            payload = HealthResponse(
                status="not_ready",
                service="medical-abstracts-classifier",
                version=__version__,
                model_loaded=False,
                inference_backend=configured_backend,
                error=store.error or "Model has not been loaded",
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        return HealthResponse(
            status="ok",
            service="medical-abstracts-classifier",
            version=__version__,
            model_loaded=True,
            inference_backend=configured_backend,
            artifact_id=str(current["artifact_id"]),
            labels=len(current["label_binarizer"].classes_),
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        current = store.bundle
        if current is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        started = time.perf_counter()
        prediction = predict_abstracts(
            current,
            [request.text],
            threshold=request.threshold,
        )[0]
        latency_ms = (time.perf_counter() - started) * 1_000
        return PredictResponse(
            artifact_id=str(current["artifact_id"]),
            model_version=str(current["artifact_id"]),
            classification=str(prediction["classification"]),
            predicted_labels=[str(value) for value in prediction["predicted_labels"]],
            probabilities={
                str(key): float(value) for key, value in prediction["probabilities"].items()
            },
            labels=[PredictedLabel(**label) for label in prediction["labels"]],
            threshold=float(prediction["threshold"]),
            latency_ms=round(latency_ms, 3),
        )

    return application


app = create_app()
