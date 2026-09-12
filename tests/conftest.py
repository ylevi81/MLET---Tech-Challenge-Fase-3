from __future__ import annotations

import csv
from pathlib import Path

import pytest

LABELS = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

TOPIC_WORDS = {
    1: "tumor oncology chemotherapy biopsy malignancy",
    2: "digestive intestinal gastric liver abdominal",
    3: "neurological brain seizure neuron cognitive",
    4: "cardiac vascular artery heart circulation",
    5: "general pathology inflammation clinical systemic",
}


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 2,000-row offline corpus with 1,000 unique multilabel texts."""
    root = tmp_path_factory.mktemp("medical-corpus")
    _write_csv(
        root / "medical_tc_labels.csv",
        ("condition_label", "condition_name"),
        [(label_id, name) for label_id, name in LABELS.items()],
    )

    train_rows: list[tuple[object, ...]] = []
    test_rows: list[tuple[object, ...]] = []
    for index in range(1_000):
        primary = index % 5 + 1
        secondary = primary % 5 + 1
        text = (
            f"Study cohort case{index:04d} reports {TOPIC_WORDS[primary]} "
            f"findings and outcome subgroup{index % 23}."
        )
        train_rows.append((primary, text))
        # Whitespace differences exercise normalization before grouping.
        test_rows.append((secondary, f"  {text}  "))

    _write_csv(
        root / "medical_tc_train.csv",
        ("condition_label", "medical_abstract"),
        train_rows,
    )
    _write_csv(
        root / "medical_tc_test.csv",
        ("condition_label", "medical_abstract"),
        test_rows,
    )
    return root


@pytest.fixture(scope="session")
def trained_artifacts(
    corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, dict[str, object]]:
    from medical_triage.model import train_and_save

    output = tmp_path_factory.mktemp("model-artifacts")
    model_path = output / "model.joblib"
    metrics_path = output / "metrics.json"
    metrics = train_and_save(
        dataset_dir=corpus_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=42,
        max_features=500,
    )
    return model_path, metrics_path, metrics


@pytest.fixture(scope="session")
def trained_bundle(trained_artifacts: tuple[Path, Path, dict[str, object]]) -> dict:
    from medical_triage.model import load_artifact

    return load_artifact(trained_artifacts[0])
