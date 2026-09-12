"""Command-line interface for download, training, prediction, and benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medical_triage.data import download_dataset
from medical_triage.model import load_model, predict_abstracts, train_and_save


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_dataset(force_download=args.force)
    _print_json({"dataset_dir": str(path)})
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else download_dataset()
    metrics = train_and_save(
        dataset_dir=dataset_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        random_state=args.random_state,
        test_size=args.test_size,
        threshold=args.threshold,
        max_features=args.max_features,
    )
    _print_json(
        {
            "model_path": str(Path(args.model_path).resolve()),
            "metrics_path": str(Path(args.metrics_path).resolve()),
            "artifact_id": metrics["artifact_id"],
            "evaluation": metrics["evaluation"],
        }
    )
    return 0


def _read_prediction_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.text_file).read_text(encoding="utf-8")


def _cmd_predict(args: argparse.Namespace) -> int:
    bundle = load_model(args.model_path)
    result = predict_abstracts(
        bundle,
        [_read_prediction_text(args)],
        threshold=args.threshold,
    )[0]
    _print_json({"artifact_id": bundle["artifact_id"], **result})
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    bundle = load_model(args.model_path)
    text = args.text or (
        "A patient with chest pain, hypertension and abnormal cardiac rhythm "
        "underwent cardiovascular evaluation."
    )
    for _ in range(args.warmup):
        predict_abstracts(bundle, [text])
    durations_ms = []
    started = time.perf_counter()
    for _ in range(args.iterations):
        request_started = time.perf_counter()
        predict_abstracts(bundle, [text])
        durations_ms.append((time.perf_counter() - request_started) * 1_000)
    elapsed = time.perf_counter() - started
    _print_json(
        {
            "artifact_id": bundle["artifact_id"],
            "iterations": args.iterations,
            "warmup": args.warmup,
            "latency_ms": {
                "mean": round(float(np.mean(durations_ms)), 3),
                "p50": round(float(np.percentile(durations_ms, 50)), 3),
                "p95": round(float(np.percentile(durations_ms, 95)), 3),
                "max": round(float(np.max(durations_ms)), 3),
            },
            "throughput_requests_per_second": round(args.iterations / elapsed, 3),
        }
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["MEDICAL_TRIAGE_MODEL_PATH"] = str(Path(args.model_path).resolve())
    uvicorn.run(
        "medical_triage.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="medical-triage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="download the Kaggle corpus")
    download_parser.add_argument("--force", action="store_true", help="force a fresh download")
    download_parser.set_defaults(handler=_cmd_download)

    train_parser = subparsers.add_parser("train", help="train and persist the model")
    train_parser.add_argument(
        "--dataset-dir",
        help="existing CSV directory; omit to download with KaggleHub",
    )
    train_parser.add_argument("--model-path", default="artifacts/model.joblib")
    train_parser.add_argument("--metrics-path", default="artifacts/metrics.json")
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--max-features", type=_positive_int, default=100_000)
    train_parser.set_defaults(handler=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="classify one medical abstract")
    predict_parser.add_argument("--model-path", default="artifacts/model.joblib")
    text_group = predict_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text")
    text_group.add_argument("--text-file")
    predict_parser.add_argument("--threshold", type=float)
    predict_parser.set_defaults(handler=_cmd_predict)

    benchmark_parser = subparsers.add_parser("benchmark", help="benchmark local inference")
    benchmark_parser.add_argument("--model-path", default="artifacts/model.joblib")
    benchmark_parser.add_argument("--text")
    benchmark_parser.add_argument("--iterations", type=_positive_int, default=100)
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.set_defaults(handler=_cmd_benchmark)

    serve_parser = subparsers.add_parser("serve", help="start the FastAPI service")
    serve_parser.add_argument("--model-path", default="artifacts/model.joblib")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--workers", type=_positive_int, default=1)
    serve_parser.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and convert expected operational failures into exit code 1."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
