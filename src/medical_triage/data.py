"""Download, validate, and group the Medical Abstracts TC Corpus."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_HANDLE = "saharalaa/medical-abstracts-tc-corpus"
LABELS_FILENAME = "medical_tc_labels.csv"
TRAIN_FILENAME = "medical_tc_train.csv"
TEST_FILENAME = "medical_tc_test.csv"
EXPECTED_FILES = (LABELS_FILENAME, TRAIN_FILENAME, TEST_FILENAME)


@dataclass(frozen=True)
class DatasetAudit:
    """Serializable facts collected while loading the source CSVs."""

    dataset_handle: str
    source_rows: int
    grouped_abstracts: int
    collapsed_rows: int
    repeated_text_label_rows: int
    multilabel_abstracts: int
    cross_split_abstracts: int
    trailing_whitespace_rows: int
    source_split_rows: dict[str, int]
    source_label_rows: dict[str, dict[str, int]]
    label_cardinality: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class GroupedCorpus:
    """One row per normalized abstract, with all applicable labels."""

    texts: tuple[str, ...]
    label_sets: tuple[tuple[int, ...], ...]
    label_names: dict[int, str]
    audit: DatasetAudit


@dataclass(frozen=True)
class CorpusSplit:
    """A deterministic, abstract-level train/test split."""

    train_texts: tuple[str, ...]
    test_texts: tuple[str, ...]
    train_labels: tuple[tuple[int, ...], ...]
    test_labels: tuple[tuple[int, ...], ...]
    strategy: str


def download_dataset(*, force_download: bool = False) -> Path:
    """Download the exact Kaggle dataset requested by the challenge.

    The import is lazy so offline commands using ``--dataset-dir`` never need
    KaggleHub credentials or network access.
    """

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on runtime state
        raise RuntimeError(
            "kagglehub is required to download the dataset; install project dependencies"
        ) from exc

    kwargs = {"force_download": True} if force_download else {}
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE, **kwargs)).resolve()
    validate_dataset_directory(dataset_dir)
    return dataset_dir


def validate_dataset_directory(dataset_dir: str | Path) -> dict[str, Path]:
    """Resolve and validate the three expected CSV files under a directory."""

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    resolved: dict[str, Path] = {}
    for filename in EXPECTED_FILES:
        direct = root / filename
        matches = [direct] if direct.is_file() else list(root.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Required dataset file not found: {filename}")
        if len(matches) > 1:
            paths = ", ".join(str(path) for path in matches)
            raise ValueError(f"Multiple copies of {filename} found: {paths}")
        resolved[filename] = matches[0]
    return resolved


def strip_accents(value: str) -> str:
    """Remove combining marks, reproducing TfidfVectorizer's ``strip_accents="unicode"``.

    This lives here, ahead of the vectorizer, because ONNX cannot represent the
    vectorizer's own accent folding: skl2onnx converts ``CountVectorizer`` only
    when ``strip_accents=None``. Normalising before the pipeline keeps the
    scikit-learn and ONNX paths byte-identical instead of letting them diverge
    on accented input.

    Measured on the corpus: none of its 14.438 abstracts contain a non-ASCII
    character, so this is a no-op for the published metrics. It is kept for
    inputs the API may receive at inference time.
    """

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_abstract(value: str) -> str:
    """Trim and collapse whitespace without changing the abstract's words."""

    if not isinstance(value, str):
        raise TypeError("medical_abstract must be a string")
    normalized = strip_accents(re.sub(r"\s+", " ", value).strip())
    if not normalized:
        raise ValueError("medical_abstract cannot be blank")
    return normalized


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")


def _integer_labels(series: pd.Series, filename: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{filename} contains null condition_label values")
    numeric = pd.to_numeric(series, errors="raise")
    if not (numeric % 1 == 0).all():
        raise ValueError(f"{filename} contains non-integer condition_label values")
    return numeric.astype(int)


def load_grouped_corpus(
    dataset_dir: str | Path,
    *,
    minimum_rows: int = 1,
) -> GroupedCorpus:
    """Load both source splits and collapse repeated abstracts into label sets."""

    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")

    files = validate_dataset_directory(dataset_dir)
    labels_frame = pd.read_csv(files[LABELS_FILENAME], encoding="utf-8")
    _require_columns(labels_frame, {"condition_label", "condition_name"}, LABELS_FILENAME)
    if labels_frame[["condition_label", "condition_name"]].isna().any().any():
        raise ValueError(f"{LABELS_FILENAME} contains null values")
    labels_frame = labels_frame[["condition_label", "condition_name"]].copy()
    labels_frame["condition_label"] = _integer_labels(
        labels_frame["condition_label"], LABELS_FILENAME
    )
    labels_frame["condition_name"] = labels_frame["condition_name"].astype(str).str.strip()
    if labels_frame["condition_name"].eq("").any():
        raise ValueError(f"{LABELS_FILENAME} contains blank condition names")
    if labels_frame["condition_label"].duplicated().any():
        raise ValueError(f"{LABELS_FILENAME} contains duplicate condition labels")
    if labels_frame["condition_name"].duplicated().any():
        raise ValueError(f"{LABELS_FILENAME} contains duplicate condition names")
    label_names = dict(
        zip(
            labels_frame["condition_label"].tolist(),
            labels_frame["condition_name"].tolist(),
            strict=True,
        )
    )

    source_frames: list[pd.DataFrame] = []
    source_label_rows: dict[str, dict[str, int]] = {}
    trailing_whitespace_rows = 0
    for split_name, filename in (("train", TRAIN_FILENAME), ("test", TEST_FILENAME)):
        frame = pd.read_csv(files[filename], encoding="utf-8")
        _require_columns(frame, {"condition_label", "medical_abstract"}, filename)
        frame = frame[["condition_label", "medical_abstract"]].copy()
        frame["condition_label"] = _integer_labels(frame["condition_label"], filename)
        if frame["medical_abstract"].isna().any():
            raise ValueError(f"{filename} contains null medical_abstract values")
        unknown = sorted(set(frame["condition_label"]).difference(label_names))
        if unknown:
            raise ValueError(f"{filename} contains unknown labels: {unknown}")
        raw_text = frame["medical_abstract"].astype(str)
        trailing_whitespace_rows += int(raw_text.ne(raw_text.str.strip()).sum())
        frame["medical_abstract"] = raw_text.map(normalize_abstract)
        frame["source_split"] = split_name
        source_label_rows[split_name] = {
            str(key): int(value)
            for key, value in frame["condition_label"].value_counts().sort_index().items()
        }
        source_frames.append(frame)

    combined = pd.concat(source_frames, ignore_index=True)
    if len(combined) < minimum_rows:
        raise ValueError(
            f"Dataset contains {len(combined)} rows; at least {minimum_rows} are required"
        )
    repeated_text_label_rows = int(
        combined.duplicated(subset=["medical_abstract", "condition_label"]).sum()
    )
    texts: list[str] = []
    label_sets: list[tuple[int, ...]] = []
    cross_split_abstracts = 0
    for text, group in combined.groupby("medical_abstract", sort=False, observed=True):
        texts.append(str(text))
        label_sets.append(tuple(sorted(set(int(value) for value in group["condition_label"]))))
        cross_split_abstracts += int(group["source_split"].nunique() > 1)

    cardinality = Counter(len(labels) for labels in label_sets)
    audit = DatasetAudit(
        dataset_handle=DATASET_HANDLE,
        source_rows=len(combined),
        grouped_abstracts=len(texts),
        collapsed_rows=len(combined) - len(texts),
        repeated_text_label_rows=repeated_text_label_rows,
        multilabel_abstracts=sum(count for size, count in cardinality.items() if size > 1),
        cross_split_abstracts=cross_split_abstracts,
        trailing_whitespace_rows=trailing_whitespace_rows,
        source_split_rows={"train": len(source_frames[0]), "test": len(source_frames[1])},
        source_label_rows=source_label_rows,
        label_cardinality={str(size): count for size, count in sorted(cardinality.items())},
    )
    return GroupedCorpus(tuple(texts), tuple(label_sets), label_names, audit)


def load_corpus(dataset_dir: str | Path, *, minimum_rows: int = 1) -> GroupedCorpus:
    """Public shorthand for :func:`load_grouped_corpus`."""

    return load_grouped_corpus(dataset_dir, minimum_rows=minimum_rows)


def split_grouped_corpus(
    corpus: GroupedCorpus,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CorpusSplit:
    """Split unique abstracts while approximately preserving label combinations."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(corpus.texts) < 2:
        raise ValueError("At least two grouped abstracts are required")

    indices = list(range(len(corpus.texts)))
    signatures = ["|".join(map(str, labels)) for labels in corpus.label_sets]
    counts = Counter(signatures)
    strata = [
        signature if counts[signature] >= 2 else str(corpus.label_sets[index][0])
        for index, signature in enumerate(signatures)
    ]
    strata_counts = Counter(strata)
    class_count = len(strata_counts)
    expected_test = round(len(indices) * test_size)
    expected_train = len(indices) - expected_test
    can_stratify = (
        min(strata_counts.values()) >= 2
        and expected_test >= class_count
        and expected_train >= class_count
    )
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata if can_stratify else None,
    )
    strategy = "label-combination-stratified" if can_stratify else "random-group"
    return CorpusSplit(
        train_texts=tuple(corpus.texts[index] for index in train_indices),
        test_texts=tuple(corpus.texts[index] for index in test_indices),
        train_labels=tuple(corpus.label_sets[index] for index in train_indices),
        test_labels=tuple(corpus.label_sets[index] for index in test_indices),
        strategy=strategy,
    )
