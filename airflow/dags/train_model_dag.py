"""Airflow DAG for downloading the corpus and publishing a trained model.

All operations that access the network or train a model live inside TaskFlow
tasks. Importing this module is therefore safe for the Airflow scheduler.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from airflow.decorators import dag, task

DAG_ID = "train_medical_abstracts_classifier"
DATASET_HANDLE = "saharalaa/medical-abstracts-tc-corpus"
TRAIN_SCHEDULE = os.environ.get("AIRFLOW_TRAIN_SCHEDULE") or None


@dag(
    dag_id=DAG_ID,
    description="Download the five-class medical corpus and publish a multilabel model",
    schedule=TRAIN_SCHEDULE,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "fiap-mlet",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["fiap", "medical-abstracts", "training"],
)
def train_medical_abstracts_classifier():
    """Create the runtime-only download, training, and promotion workflow."""

    @task(task_id="download_corpus")
    def download_corpus() -> str:
        """Download the Kaggle dataset at task runtime and return its local path."""
        from pathlib import Path

        from medical_triage.data import download_dataset

        dataset_dir = Path(download_dataset()).resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Kaggle dataset directory was not found: {dataset_dir}")
        if not any(dataset_dir.rglob("*.csv")):
            raise FileNotFoundError(f"No CSV file was found under {dataset_dir}")
        return str(dataset_dir)

    @task(task_id="train_and_promote")
    def train_and_promote(dataset_dir: str) -> dict[str, str]:
        """Train all five labels in staging and atomically publish each artifact."""
        import json
        import shutil
        from pathlib import Path
        from uuid import uuid4

        from medical_triage.model import train_and_save

        source_dir = Path(dataset_dir).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory is unavailable: {source_dir}")

        artifact_root = Path(
            os.environ.get("MEDICAL_TRIAGE_ARTIFACT_DIR", "artifacts")
        ).resolve()
        staging_dir = artifact_root / ".staging" / uuid4().hex
        staged_model = staging_dir / "model.joblib"
        staged_metrics = staging_dir / "metrics.json"
        final_model = artifact_root / "model.joblib"
        final_metrics = artifact_root / "metrics.json"
        final_manifest = artifact_root / "promotion.json"

        staging_dir.mkdir(parents=True, exist_ok=False)
        artifact_root.mkdir(parents=True, exist_ok=True)

        try:
            train_and_save(
                dataset_dir=source_dir,
                model_path=staged_model,
                metrics_path=staged_metrics,
                random_state=42,
            )

            if not staged_model.is_file() or staged_model.stat().st_size == 0:
                raise RuntimeError("Training did not produce a non-empty model bundle")
            if not staged_metrics.is_file() or staged_metrics.stat().st_size == 0:
                raise RuntimeError("Training did not produce metrics metadata")

            metrics = json.loads(staged_metrics.read_text(encoding="utf-8"))
            labels = metrics.get("classes")
            if labels is None:
                labels = metrics.get("labels")
            model_metadata = metrics.get("model")
            if labels is None and isinstance(model_metadata, dict):
                labels = model_metadata.get("labels")
            if not isinstance(labels, list | tuple | dict):
                raise RuntimeError("Training metadata does not identify the trained classes")
            if len(labels) != 5:
                raise RuntimeError(f"Expected five trained classes, found {len(labels)}")

            # os.replace keeps each published file atomic. The manifest is replaced last,
            # so consumers can use it as the commit marker for a complete promotion.
            os.replace(staged_model, final_model)
            os.replace(staged_metrics, final_metrics)

            manifest = {
                "dataset_handle": DATASET_HANDLE,
                "model_path": str(final_model),
                "metrics_path": str(final_metrics),
                "promoted_at": datetime.now(UTC).isoformat(),
            }
            staged_manifest = staging_dir / "promotion.json"
            staged_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(staged_manifest, final_manifest)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        return {
            "model_path": str(final_model),
            "metrics_path": str(final_metrics),
            "manifest_path": str(final_manifest),
        }

    train_and_promote(download_corpus())


train_model_dag = train_medical_abstracts_classifier()
