"""Select an inference runtime while preserving the API's bundle contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medical_triage.model import load_model

BACKENDS = {"sklearn", "onnx_fp32", "onnx_int8"}


class ONNXPredictor:
    """Implement predict_proba for the shared multilabel postprocessing."""

    def __init__(self, path: Path, bundle: dict, backend: str) -> None:
        from medical_triage.optimize import make_session

        self.session = make_session(path)
        metadata = self.session.get_modelmeta().custom_metadata_map
        expected = {
            "artifact_id": str(bundle["artifact_id"]),
            "class_ids": json.dumps([int(x) for x in bundle["label_binarizer"].classes_]),
            "backend": backend,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("ONNX metadata does not match the model bundle/backend; export again")
        self.n_classes = len(bundle["label_binarizer"].classes_)

    def predict_proba(self, texts) -> np.ndarray:
        from medical_triage.optimize import predict_proba_onnx

        probabilities = predict_proba_onnx(self.session, texts)
        if probabilities.shape != (len(texts), self.n_classes):
            raise ValueError("Unexpected ONNX probability shape")
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite ONNX probabilities")
        return probabilities


def configure_backend(bundle: dict, backend: str, onnx_path: Path) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown inference backend: {backend}")
    if backend == "sklearn":
        return bundle
    return {**bundle, "pipeline": ONNXPredictor(onnx_path, bundle, backend)}


def prepare(model_path: Path, output_dir: Path) -> None:
    """Export when absent/stale; preserve the original Joblib bundle for metadata."""

    from medical_triage.optimize import optimize_bundle

    bundle = load_model(model_path)
    fp32 = output_dir / "model.onnx"
    int8 = output_dir / "model_int8.onnx"
    try:
        ONNXPredictor(fp32, bundle, "onnx_fp32")
        ONNXPredictor(int8, bundle, "onnx_int8")
    except Exception:
        optimize_bundle(
            bundle,
            ["The patient has cardiac disease", "A tumor was found after surgery"],
            onnx_path=fp32,
            int8_path=int8,
        )
        ONNXPredictor(fp32, bundle, "onnx_fp32")
        ONNXPredictor(int8, bundle, "onnx_int8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
