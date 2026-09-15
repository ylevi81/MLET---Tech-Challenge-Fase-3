"""Multilabel model training, persistence, evaluation, and inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from medical_triage.data import load_grouped_corpus, normalize_abstract, split_grouped_corpus

BUNDLE_FORMAT_VERSION = 1
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_FEATURES = 100_000


def build_pipeline(*, random_state: int = 42, max_features: int = DEFAULT_MAX_FEATURES) -> Pipeline:
    """Build the deterministic TF-IDF plus one-vs-rest baseline."""

    if max_features < 1:
        raise ValueError("max_features must be positive")
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    # strip_accents fica em normalize_abstract: skl2onnx so converte
                    # CountVectorizer com strip_accents=None (ver data.strip_accents).
                    strip_accents=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    # max_df=0.98 removia apenas os unigramas 'of' e 'the',
                    # mas deixava 11.422 bigramas sem os tokens que os compoem.
                    # O ONNX nao consegue formar um n-grama cujo token nao esta
                    # no pool, o que quebrava 11% do vocabulario na exportacao.
                    max_features=max_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )


def _threshold_probabilities(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix")
    indicator = (matrix >= threshold).astype(np.int8)
    empty_rows = np.flatnonzero(indicator.sum(axis=1) == 0)
    if len(empty_rows):
        indicator[empty_rows, matrix[empty_rows].argmax(axis=1)] = 1
    return indicator


def _evaluate(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    label_ids: Sequence[int],
    label_names: dict[int, str],
) -> dict[str, Any]:
    predictions = _threshold_probabilities(probabilities, threshold)
    per_label_precision = precision_score(targets, predictions, average=None, zero_division=0)
    per_label_recall = recall_score(targets, predictions, average=None, zero_division=0)
    per_label_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    supports = targets.sum(axis=0)
    per_label = {}
    for index, label_id in enumerate(label_ids):
        per_label[str(label_id)] = {
            "name": label_names[int(label_id)],
            "precision": round(float(per_label_precision[index]), 6),
            "recall": round(float(per_label_recall[index]), 6),
            "f1": round(float(per_label_f1[index]), 6),
            "support": int(supports[index]),
        }
    return {
        "threshold": threshold,
        "subset_accuracy": round(float(accuracy_score(targets, predictions)), 6),
        "hamming_loss": round(float(hamming_loss(targets, predictions)), 6),
        "jaccard_samples": round(
            float(jaccard_score(targets, predictions, average="samples", zero_division=0)), 6
        ),
        "precision_micro": round(
            float(precision_score(targets, predictions, average="micro", zero_division=0)), 6
        ),
        "recall_micro": round(
            float(recall_score(targets, predictions, average="micro", zero_division=0)), 6
        ),
        "f1_micro": round(float(f1_score(targets, predictions, average="micro")), 6),
        "f1_macro": round(float(f1_score(targets, predictions, average="macro")), 6),
        "f1_weighted": round(float(f1_score(targets, predictions, average="weighted")), 6),
        "predicted_cardinality_mean": round(float(predictions.sum(axis=1).mean()), 6),
        "true_cardinality_mean": round(float(targets.sum(axis=1).mean()), 6),
        "per_label": per_label,
    }


def train_model(
    dataset_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    threshold: float = DEFAULT_THRESHOLD,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and evaluate a grouped multilabel model without writing files."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")
    started = time.perf_counter()
    corpus = load_grouped_corpus(dataset_dir, minimum_rows=2_000)
    split = split_grouped_corpus(corpus, test_size=test_size, random_state=random_state)
    label_ids = tuple(sorted(corpus.label_names))
    binarizer = MultiLabelBinarizer(classes=label_ids)
    binarizer.fit(corpus.label_sets)
    train_targets = binarizer.transform(split.train_labels)
    test_targets = binarizer.transform(split.test_labels)
    if (train_targets.sum(axis=0) == 0).any():
        raise ValueError("The training split does not contain every configured label")

    pipeline = build_pipeline(random_state=random_state, max_features=max_features)
    fit_started = time.perf_counter()
    pipeline.fit(split.train_texts, train_targets)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    probabilities = np.asarray(pipeline.predict_proba(split.test_texts), dtype=float)
    predict_seconds = time.perf_counter() - predict_started
    evaluation = _evaluate(
        test_targets,
        probabilities,
        threshold=threshold,
        label_ids=label_ids,
        label_names=corpus.label_names,
    )

    artifact_id = uuid4().hex
    trained_at = datetime.now(UTC).isoformat()
    metrics: dict[str, Any] = {
        "artifact_id": artifact_id,
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "trained_at": trained_at,
        "dataset": corpus.audit.to_dict(),
        "split": {
            "strategy": split.strategy,
            "random_state": random_state,
            "test_size": test_size,
            "train_abstracts": len(split.train_texts),
            "test_abstracts": len(split.test_texts),
        },
        "model": {
            "type": "TF-IDF + OneVsRest LogisticRegression",
            "max_features": max_features,
            "ngram_range": [1, 2],
            "class_weight": "balanced",
            "labels": [
                {"id": label_id, "name": corpus.label_names[label_id]} for label_id in label_ids
            ],
        },
        "evaluation": evaluation,
        "timings_seconds": {
            "fit": round(fit_seconds, 6),
            "predict_test": round(predict_seconds, 6),
            "total_before_persistence": round(time.perf_counter() - started, 6),
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    bundle: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "artifact_id": artifact_id,
        "trained_at": trained_at,
        "pipeline": pipeline,
        "label_binarizer": binarizer,
        "label_names": corpus.label_names,
        "default_threshold": threshold,
        "training_metrics": metrics,
    }
    return bundle, metrics


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def _fsync(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_bundle(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    *,
    model_path: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    """Publish model and metrics via same-filesystem atomic replacements.

    Both files are fully staged before publication. The model is replaced first
    and metrics last, making metrics the commit marker. The model also embeds a
    metrics copy, so it remains self-describing if publication is interrupted.
    """

    final_model = Path(model_path).expanduser().resolve()
    final_metrics = Path(metrics_path).expanduser().resolve()
    if final_model == final_metrics:
        raise ValueError("model_path and metrics_path must be different")
    model_temp = _temporary_path(final_model)
    metrics_temp = _temporary_path(final_metrics)
    try:
        joblib.dump(bundle, model_temp, compress=3)
        _fsync(model_temp)
        published_metrics = dict(metrics)
        published_metrics["model_sha256"] = _sha256(model_temp)
        metrics_temp.write_text(
            json.dumps(published_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _fsync(metrics_temp)
        os.replace(model_temp, final_model)
        os.replace(metrics_temp, final_metrics)
    finally:
        model_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)
    return published_metrics


def train_and_save(
    dataset_dir: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    random_state: int = 42,
    *,
    test_size: float = 0.2,
    threshold: float = DEFAULT_THRESHOLD,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> dict[str, Any]:
    """Train, atomically save the model bundle, and return published metrics."""

    bundle, metrics = train_model(
        dataset_dir,
        random_state=random_state,
        test_size=test_size,
        threshold=threshold,
        max_features=max_features,
    )
    return save_model_bundle(
        bundle,
        metrics,
        model_path=model_path,
        metrics_path=metrics_path,
    )


def validate_model_bundle(bundle: Any) -> dict[str, Any]:
    """Reject malformed or unsupported artifacts before inference."""

    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must contain a dictionary bundle")
    required = {
        "format_version",
        "artifact_id",
        "pipeline",
        "label_binarizer",
        "label_names",
        "default_threshold",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {', '.join(sorted(missing))}")
    if bundle["format_version"] != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported model format version: {bundle['format_version']}")
    classes = tuple(int(value) for value in bundle["label_binarizer"].classes_)
    if set(classes) != set(int(value) for value in bundle["label_names"]):
        raise ValueError("Model classes and label names do not match")
    return bundle


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local Joblib model artifact."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return validate_model_bundle(joblib.load(path))


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Public shorthand for loading a persisted model artifact."""

    return load_model(model_path)


def predict_abstracts(
    bundle: dict[str, Any],
    texts: Sequence[str],
    *,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Predict one or more abstracts, returning sorted labels and probabilities."""

    validate_model_bundle(bundle)
    if not texts:
        raise ValueError("At least one abstract is required")
    normalized = [normalize_abstract(text) for text in texts]
    selected_threshold = (
        float(bundle["default_threshold"]) if threshold is None else float(threshold)
    )
    probabilities = np.asarray(bundle["pipeline"].predict_proba(normalized), dtype=float)
    indicators = _threshold_probabilities(probabilities, selected_threshold)
    classes = [int(value) for value in bundle["label_binarizer"].classes_]
    label_names = {int(key): value for key, value in bundle["label_names"].items()}
    results: list[dict[str, Any]] = []
    for row_probabilities, row_indicator in zip(probabilities, indicators, strict=True):
        all_probabilities = {
            label_names[label_id]: round(float(row_probabilities[index]), 6)
            for index, label_id in enumerate(classes)
        }
        labels = [
            {
                "id": classes[index],
                "name": label_names[classes[index]],
                "probability": round(float(row_probabilities[index]), 6),
            }
            for index in np.flatnonzero(row_indicator)
        ]
        labels.sort(key=lambda item: item["probability"], reverse=True)
        top_index = int(row_probabilities.argmax())
        results.append(
            {
                "classification": label_names[classes[top_index]],
                "predicted_labels": [str(item["name"]) for item in labels],
                "probabilities": all_probabilities,
                "labels": labels,
                "threshold": selected_threshold,
            }
        )
    return results


def predict(
    bundle: dict[str, Any],
    texts: Sequence[str],
    *,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Public shorthand for batched abstract prediction."""

    return predict_abstracts(bundle, texts, threshold=threshold)
