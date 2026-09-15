"""Tests for the ONNX export and INT8 quantisation (stage 4).

These use a tiny in-memory pipeline rather than the published artifact, so they
run offline in CI without Kaggle access and without a trained model.

Every test here pins a failure that actually happened while building the stage,
not a hypothetical one. Three of them would have caught silent corruption:
the conversion that changed 0.9% of labels, the quantisation that returned a
byte-identical file, and the sublinear term-frequency formula that ONNX
computed as ``log(1 + tf)`` instead of ``1 + log(tf)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from medical_triage.data import normalize_abstract, strip_accents
from medical_triage.optimize import (
    benchmark_latency,
    export_onnx,
    make_session,
    predict_proba_onnx,
    quantize,
    verify_parity,
)

CORPUS = [
    "the cardiac patient had chest pain and a severe arrhythmia after surgery",
    "a malignant tumor of the colon was resected with clear surgical margins",
    "the infant developed a seizure and the neurologic exam showed a deficit",
    "renal failure progressed and the patient required dialysis for a week",
    "acute myocardial infarction was treated with thrombolytic therapy",
    "gastric ulcer bleeding required endoscopic intervention and transfusion",
    "a benign lesion of the nervous system was observed on the scan",
    "chronic obstructive disease of the lung worsened after a viral infection",
]
TARGETS = [[1, 4], [0], [2], [4], [3], [1], [2], [4]]


@pytest.fixture(scope="module")
def bundle() -> dict:
    """A small multilabel pipeline shaped like the production one."""

    labels = np.zeros((len(TARGETS), 5), dtype=int)
    for row, target in enumerate(TARGETS):
        labels[row, target] = 1

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(solver="liblinear", random_state=42)
                ),
            ),
        ]
    )
    pipeline.fit(CORPUS, labels)
    return {"pipeline": pipeline}


@pytest.fixture(scope="module")
def exported(bundle, tmp_path_factory) -> tuple:
    directory = tmp_path_factory.mktemp("onnx")
    onnx_path = directory / "model.onnx"
    int8_path = directory / "model_int8.onnx"
    export_onnx(bundle["pipeline"], onnx_path)
    quantize(onnx_path, int8_path)
    return onnx_path, int8_path


def test_accent_folding_matches_the_vectorizer_option():
    """normalize_abstract must reproduce strip_accents="unicode"."""

    assert strip_accents("pré-eclâmpsia") == "pre-eclampsia"
    assert normalize_abstract("  neonatal   jaundice ") == "neonatal jaundice"


def test_onnx_probabilities_match_sklearn(bundle, exported):
    """A conversion that changes predictions is not a conversion.

    The bigram path is what broke here originally: ONNX kept one-character
    tokens that scikit-learn's token pattern drops, so bigrams spanning them
    silently diverged.
    """

    onnx_path, _ = exported
    expected = bundle["pipeline"].predict_proba(CORPUS)
    session = make_session(onnx_path)
    actual = predict_proba_onnx(session, CORPUS)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_sublinear_tf_uses_one_plus_log(bundle, exported):
    """Guards the ``1 + log(tf)`` rewrite.

    skl2onnx emits ``log(1 + tf)``. With a repeated term the two formulas
    disagree well beyond float tolerance, so a regression here fails loudly
    instead of quietly shifting probabilities.
    """

    onnx_path, _ = exported
    repeated = ["the patient the patient the patient had chest pain chest pain"]
    expected = bundle["pipeline"].predict_proba(repeated)
    actual = predict_proba_onnx(make_session(onnx_path), repeated)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_quantised_model_is_materially_smaller(exported):
    """If INT8 does not shrink the file, quantisation silently did nothing.

    That was the original behaviour: the weights sat inside
    ``ai.onnx.ml.LinearClassifier`` attributes, where ``quantize_dynamic``
    cannot see them, and the output was byte-identical to the input.
    """

    onnx_path, int8_path = exported
    assert int8_path.stat().st_size < onnx_path.stat().st_size * 0.85


def test_quantised_model_keeps_labels(bundle, exported):
    onnx_path, int8_path = exported
    report = verify_parity(
        bundle, CORPUS, onnx_path=onnx_path, int8_path=int8_path
    )
    assert report["onnx_fp32"]["label_agreement"] == 1.0
    assert report["onnx_int8"]["label_agreement"] >= 0.95


def test_benchmark_reports_every_variant(bundle, exported):
    onnx_path, int8_path = exported
    results = benchmark_latency(
        bundle,
        CORPUS,
        onnx_path=onnx_path,
        int8_path=int8_path,
        batch_sizes=(1,),
        runs=5,
        warmup=1,
    )
    assert set(results) == {"sklearn", "onnx_fp32", "onnx_int8"}
    for tag, stats in results.items():
        assert stats["batch_1"]["mean_ms"] > 0
        if tag != "sklearn":
            assert "speedup_vs_sklearn" in stats["batch_1"]
