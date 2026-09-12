from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from medical_triage.data import (
    DATASET_HANDLE,
    download_dataset,
    load_corpus,
    split_grouped_corpus,
)


def test_download_uses_exact_kaggle_handle(
    monkeypatch, corpus_dir: Path
) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_download(handle: str, **kwargs) -> str:
        calls.append((handle, kwargs))
        return str(corpus_dir)

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=fake_download),
    )

    assert download_dataset() == corpus_dir.resolve()
    assert calls == [("saharalaa/medical-abstracts-tc-corpus", {})]
    assert DATASET_HANDLE == "saharalaa/medical-abstracts-tc-corpus"


def test_loader_groups_duplicate_abstracts_and_aggregates_labels(corpus_dir: Path) -> None:
    corpus = load_corpus(corpus_dir)

    assert corpus.audit.source_rows == 2_000
    assert corpus.audit.grouped_abstracts == 1_000
    assert corpus.audit.cross_split_abstracts == 1_000
    assert corpus.audit.multilabel_abstracts == 1_000
    assert len(corpus.texts) == len(set(corpus.texts)) == 1_000
    assert all(len(labels) == 2 for labels in corpus.label_sets)
    assert set(corpus.label_names.values()) == {
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    }


def test_new_split_has_no_text_overlap(corpus_dir: Path) -> None:
    corpus = load_corpus(corpus_dir)
    split = split_grouped_corpus(corpus, random_state=42)

    assert set(split.train_texts).isdisjoint(split.test_texts)
    assert len(split.train_texts) + len(split.test_texts) == len(corpus.texts)

