"""Dependency-light smoke test runnable with the Python standard library."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from medical_triage.api import create_app
from medical_triage.data import load_corpus, split_grouped_corpus
from medical_triage.model import load_artifact, predict, train_and_save


class OfflinePipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        labels = {
            1: "neoplasms",
            2: "digestive system diseases",
            3: "nervous system diseases",
            4: "cardiovascular diseases",
            5: "general pathological conditions",
        }
        topics = {
            1: "tumor oncology chemotherapy biopsy malignancy",
            2: "digestive intestinal gastric liver abdominal",
            3: "neurological brain seizure neuron cognitive",
            4: "cardiac vascular artery heart circulation",
            5: "general pathology inflammation clinical systemic",
        }
        cls._write(
            cls.root / "medical_tc_labels.csv",
            ("condition_label", "condition_name"),
            list(labels.items()),
        )
        train_rows = []
        test_rows = []
        for index in range(1_000):
            primary = index % 5 + 1
            secondary = primary % 5 + 1
            text = f"Case{index:04d} {topics[primary]} outcome group{index % 23}."
            train_rows.append((primary, text))
            test_rows.append((secondary, f" {text} "))
        cls._write(
            cls.root / "medical_tc_train.csv",
            ("condition_label", "medical_abstract"),
            train_rows,
        )
        cls._write(
            cls.root / "medical_tc_test.csv",
            ("condition_label", "medical_abstract"),
            test_rows,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    @staticmethod
    def _write(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def test_end_to_end_offline_training(self) -> None:
        corpus = load_corpus(self.root)
        self.assertEqual(corpus.audit.source_rows, 2_000)
        self.assertEqual(corpus.audit.grouped_abstracts, 1_000)
        split = split_grouped_corpus(corpus, random_state=42)
        self.assertTrue(set(split.train_texts).isdisjoint(split.test_texts))

        artifact = self.root / "model.joblib"
        metrics_file = self.root / "metrics.json"
        metrics = train_and_save(
            self.root,
            artifact,
            metrics_file,
            random_state=42,
            max_features=500,
        )
        self.assertTrue(artifact.is_file())
        self.assertEqual(json.loads(metrics_file.read_text(encoding="utf-8")), metrics)
        self.assertEqual(len(metrics["model"]["labels"]), 5)

        bundle = load_artifact(artifact)
        [result] = predict(bundle, ["Cardiac artery heart circulation findings."])
        self.assertTrue(result["labels"])

        with TestClient(create_app(bundle=bundle)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.post(
                "/predict",
                json={"text": "Cardiac artery heart circulation findings."},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["classification"], payload["probabilities"])
        self.assertEqual(len(payload["probabilities"]), 5)
        self.assertGreaterEqual(payload["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
