from __future__ import annotations

import json
from pathlib import Path

from medical_triage.model import predict


def test_training_persists_real_bundle_and_multilabel_metrics(
    trained_artifacts: tuple[Path, Path, dict[str, object]],
) -> None:
    model_path, metrics_path, returned_metrics = trained_artifacts

    assert model_path.is_file() and model_path.stat().st_size > 0
    assert metrics_path.is_file() and metrics_path.stat().st_size > 0
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted == returned_metrics
    assert persisted["dataset"]["source_rows"] == 2_000
    assert persisted["dataset"]["grouped_abstracts"] == 1_000
    assert len(persisted["model"]["labels"]) == 5
    assert "f1_micro" in persisted["evaluation"]
    assert "hamming_loss" in persisted["evaluation"]
    assert len(persisted["model_sha256"]) == 64


def test_predict_returns_at_least_one_label(trained_bundle: dict) -> None:
    [result] = predict(
        trained_bundle,
        ["Cardiac artery heart circulation findings in a clinical cohort."],
    )

    assert result["labels"]
    assert all(0 <= item["probability"] <= 1 for item in result["labels"])
    assert 0 <= result["threshold"] <= 1

