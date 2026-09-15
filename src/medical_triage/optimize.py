"""Export the trained pipeline to ONNX Runtime and quantise it to INT8.

Two latency techniques are applied, in this order:

1. **ONNX Runtime conversion.** The whole pipeline (TF-IDF plus the
   one-vs-rest logistic regressions) becomes a single graph executed in C++,
   removing the Python and scipy overhead paid on every call.
2. **Dynamic INT8 quantisation.** The classifier weights drop from float32 to
   int8, shrinking the artifact and the memory traffic per inference.

Both steps are verified against the scikit-learn pipeline: an optimisation
that changes predictions is not an optimisation.

Two conversion obstacles are handled here, and neither is obvious:

* ``TfidfVectorizer(strip_accents="unicode")`` cannot be converted at all --
  skl2onnx supports ``strip_accents=None`` only. Accent folding therefore
  happens in :func:`medical_triage.data.normalize_abstract`, ahead of the
  pipeline, so both runtimes see identical text.
* ``quantize_dynamic`` only rewrites ``MatMul``/``Gemm``/``Conv``/``LSTM``.
  skl2onnx emits each one-vs-rest estimator as ``ai.onnx.ml.LinearClassifier``,
  which keeps its weights as *node attributes*, invisible to the quantiser --
  quantising the graph as converted is a silent no-op that returns a
  byte-identical file. :func:`fuse_ovr_classifiers` rewrites the whole
  one-vs-rest tail as a single ``MatMul + Add + Sigmoid`` first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from medical_triage.data import normalize_abstract

TARGET_OPSET = 15
DEFAULT_ONNX_PATH = Path("artifacts/model.onnx")
DEFAULT_INT8_PATH = Path("artifacts/model_int8.onnx")


def to_onnx_input(texts) -> np.ndarray:
    """ONNX expects a 2-D column of strings, not a flat list."""

    return np.array(list(texts), dtype=object).reshape(-1, 1)


def make_session(model_path: str | Path, *, threads: int = 1) -> ort.InferenceSession:
    """Build a session tuned for single-request latency.

    ``intra_op_num_threads=1`` is deliberate: at batch size 1 the internal
    parallelism only adds synchronisation overhead, and the API serves one
    abstract per request.
    """

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


def export_onnx(pipeline, output_path: str | Path = DEFAULT_ONNX_PATH) -> Path:
    """Convert the fitted scikit-learn pipeline into a single ONNX graph.

    ``zipmap=False`` keeps the probability output as a plain tensor. The
    default ``ZipMap`` wraps every row in a Python dictionary, which is pure
    overhead for a service that reads the probabilities positionally.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = convert_sklearn(
        pipeline,
        "medical-abstracts-classifier",
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=TARGET_OPSET,
        options={id(pipeline): {"zipmap": False}},
    )
    model.doc_string = "Medical Abstracts TC Corpus: TF-IDF + one-vs-rest logistic regression"
    _align_tokenizer_with_sklearn(model)
    if getattr(pipeline.named_steps["tfidf"], "sublinear_tf", False):
        _fix_sublinear_tf(model)
    fuse_ovr_classifiers(model)
    onnx.checker.check_model(model)
    output_path.write_bytes(model.SerializeToString())
    return output_path


def _align_tokenizer_with_sklearn(model: onnx.ModelProto) -> None:
    """Make the ONNX tokenizer drop one-character tokens, as scikit-learn does.

    skl2onnx emits ``com.microsoft.Tokenizer`` with ``mincharnum=1``, but
    scikit-learn's default ``token_pattern=r"(?u)\\b\\w\\w+\\b"`` requires at
    least two characters. The gap is invisible for unigrams -- the stray
    one-character tokens simply miss the vocabulary -- but it silently corrupts
    every bigram that spans one.

    For "vitamin a deficiency", scikit-learn produces the bigram
    ``vitamin deficiency`` while ONNX produces ``vitamin a`` and
    ``a deficiency``, neither of which is in the vocabulary. Measured on this
    corpus, 194 of 200 abstracts contain such a token, which pushed the
    probability gap to 0.12 and made 0.9% of labels disagree.
    """

    for node in model.graph.node:
        if node.op_type != "Tokenizer":
            continue
        for attribute in node.attribute:
            if attribute.name == "mincharnum":
                attribute.i = 2


def _fix_sublinear_tf(model: onnx.ModelProto) -> bool:
    """Correct the sublinear term-frequency formula emitted by skl2onnx.

    scikit-learn's ``sublinear_tf=True`` computes ``1 + log(tf)`` for non-zero
    counts and leaves zeros alone. skl2onnx emits ``Add(tf, 1) -> Log``, which
    is ``log(1 + tf)`` -- a different function: at ``tf = 1`` scikit-learn gives
    1.0 and the graph gives 0.693.

    The replacement keeps zeros at zero without a branch::

        sign(tf) * (1 + log(max(tf, 1)))

    ``max(tf, 1)`` protects ``log`` at ``tf = 0`` and is a no-op elsewhere,
    because term counts are integers; ``sign(tf)`` then zeroes the entries that
    were never present.
    """

    nodes = {node.output[0]: node for node in model.graph.node}
    log_node = next((n for n in model.graph.node if n.op_type == "Log"), None)
    if log_node is None:
        return False
    add_node = nodes.get(log_node.input[0])
    if add_node is None or add_node.op_type != "Add":
        return False

    counts = add_node.input[0]
    log_out = log_node.output[0]

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="tf_one"),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="tf_floor"),
        ]
    )
    replacement = [
        helper.make_node("Max", [counts, "tf_floor"], ["tf_clamped"]),
        helper.make_node("Log", ["tf_clamped"], ["tf_logged"]),
        helper.make_node("Add", ["tf_logged", "tf_one"], ["tf_sublinear"]),
        helper.make_node("Sign", [counts], ["tf_present"]),
        helper.make_node("Mul", ["tf_sublinear", "tf_present"], [log_out]),
    ]

    position = list(model.graph.node).index(add_node)
    model.graph.node.remove(add_node)
    model.graph.node.remove(log_node)
    for offset, new_node in enumerate(replacement):
        model.graph.node.insert(position + offset, new_node)
    return True


def fuse_ovr_classifiers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Collapse the five one-vs-rest heads into a single ``MatMul``.

    As converted, the tail of the graph is five ``LinearClassifier`` nodes,
    five ``Slice`` nodes that keep each positive-class column, and a ``Concat``.
    Every head multiplies the same feature vector by its own ``(n_features, 2)``
    matrix and then half of that work is sliced away.

    Stacking only the positive-class weights into one ``(n_features, 5)``
    matrix gives the same probabilities from one BLAS call instead of five,
    halves the weights kept in memory, and removes eleven nodes. It also
    exposes a single large ``MatMul`` to the quantiser rather than five small
    ones, so INT8 has something worth quantising.
    """

    heads = [node for node in model.graph.node if node.op_type == "LinearClassifier"]
    if not heads:
        raise RuntimeError("no LinearClassifier node found; graph already rewritten?")

    columns: list[np.ndarray] = []
    biases: list[float] = []
    features_in = heads[0].input[0]
    for node in heads:
        attributes = {attribute.name: attribute for attribute in node.attribute}
        if attributes["post_transform"].s != b"LOGISTIC":
            raise RuntimeError("expected a logistic link on every one-vs-rest head")
        if node.input[0] != features_in:
            raise RuntimeError("one-vs-rest heads do not share a single input")
        intercepts = np.array(attributes["intercepts"].floats, dtype=np.float32)
        coefficients = np.array(attributes["coefficients"].floats, dtype=np.float32)
        weights = coefficients.reshape(len(intercepts), -1)
        # Index 1 is the positive class: `classlabels_ints` is [0, 1] and the
        # Slice that followed kept column 1.
        columns.append(weights[1])
        biases.append(float(intercepts[1]))

    stacked = np.stack(columns, axis=1).astype(np.float32)
    bias = np.array(biases, dtype=np.float32)

    probabilities = _probabilities_output_name(model)
    position = list(model.graph.node).index(heads[0])

    # Everything between the heads and the probability tensor becomes dead:
    # the heads themselves, the Slice that kept each positive column, and the
    # Concat that stitched them back together.
    head_outputs = {name for node in heads for name in node.output}
    slices = [
        node
        for node in model.graph.node
        if node.op_type == "Slice" and node.input[0] in head_outputs
    ]
    slice_outputs = {name for node in slices for name in node.output}
    concats = [
        node
        for node in model.graph.node
        if node.op_type == "Concat" and set(node.input) <= slice_outputs
    ]
    if len(concats) != 1 or concats[0].output[0] != probabilities:
        raise RuntimeError("unexpected one-vs-rest tail; refusing to fuse blindly")

    for node in [*heads, *slices, *concats]:
        model.graph.node.remove(node)

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(stacked, name="ovr_weights"),
            numpy_helper.from_array(bias, name="ovr_bias"),
        ]
    )
    fused = [
        helper.make_node("MatMul", [features_in, "ovr_weights"], ["ovr_scores"]),
        helper.make_node("Add", ["ovr_scores", "ovr_bias"], ["ovr_logits"]),
        helper.make_node("Sigmoid", ["ovr_logits"], [probabilities]),
    ]
    for offset, node in enumerate(fused):
        model.graph.node.insert(position + offset, node)

    onnx.checker.check_model(model)
    return model


def _probabilities_output_name(model: onnx.ModelProto) -> str:
    for output in model.graph.output:
        if output.name != "label":
            return output.name
    raise RuntimeError("probability output not found")



def quantize(
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    int8_path: str | Path = DEFAULT_INT8_PATH,
) -> Path:
    """Rewrite the classifier nodes, then quantise the weights to INT8."""

    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
        # The tokenizer comes from the `com.microsoft` domain, so ONNX shape
        # inference cannot type anything downstream of it and the quantiser
        # refuses to touch our MatMul. Everything flowing through it is float32.
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    return int8_path


def predict_proba_onnx(session: ort.InferenceSession, texts) -> np.ndarray:
    """Run the ONNX graph over already-normalised abstracts."""

    feeds = {session.get_inputs()[0].name: to_onnx_input(texts)}
    outputs = session.run(None, feeds)
    # `probabilities` is the second output once ZipMap is disabled.
    probabilities = outputs[1] if len(outputs) > 1 else outputs[0]
    return np.asarray(probabilities, dtype=float)


def verify_parity(
    bundle: dict[str, Any],
    texts,
    *,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    int8_path: str | Path | None = DEFAULT_INT8_PATH,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare probabilities and thresholded labels against scikit-learn."""

    normalised = [normalize_abstract(text) for text in texts]
    reference = np.asarray(bundle["pipeline"].predict_proba(normalised), dtype=float)
    reference_labels = (reference >= threshold).astype(int)

    report: dict[str, Any] = {}
    candidates = [("onnx_fp32", Path(onnx_path))]
    if int8_path is not None and Path(int8_path).exists():
        candidates.append(("onnx_int8", Path(int8_path)))

    for tag, path in candidates:
        probabilities = predict_proba_onnx(make_session(path), normalised)
        labels = (probabilities >= threshold).astype(int)
        report[tag] = {
            "max_abs_probability_diff": float(np.abs(probabilities - reference).max()),
            "mean_abs_probability_diff": float(np.abs(probabilities - reference).mean()),
            "label_agreement": float((labels == reference_labels).mean()),
            "size_mb": round(path.stat().st_size / 1e6, 3),
        }
    return report


def _summarise(samples_ms: list[float], batch_size: int) -> dict[str, float]:
    ordered = sorted(samples_ms)
    count = len(ordered)
    mean = sum(ordered) / count
    return {
        "mean_ms": round(mean, 4),
        "p50_ms": round(ordered[int(count * 0.50)], 4),
        "p95_ms": round(ordered[int(count * 0.95)], 4),
        "p99_ms": round(ordered[min(int(count * 0.99), count - 1)], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
        "throughput_docs_per_second": round(batch_size / (mean / 1000), 1),
    }


def benchmark_latency(
    bundle: dict[str, Any],
    texts,
    *,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    int8_path: str | Path = DEFAULT_INT8_PATH,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    runs: int = 400,
    warmup: int = 40,
) -> dict[str, Any]:
    """Time scikit-learn against the ONNX variants on identical inputs.

    One timing per call, without an internal average, so the percentiles
    describe the real distribution. Batch size 1 is the API's regime: one
    abstract per HTTP request.
    """

    normalised = [normalize_abstract(text) for text in texts]
    pipeline = bundle["pipeline"]

    runners: dict[str, Any] = {"sklearn": pipeline.predict_proba}
    for tag, path in (("onnx_fp32", Path(onnx_path)), ("onnx_int8", Path(int8_path))):
        if path.exists():
            session = make_session(path)
            runners[tag] = lambda batch, s=session: predict_proba_onnx(s, batch)

    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        batches = [
            normalised[start : start + batch_size]
            for start in range(0, len(normalised), batch_size)
        ]
        batches = [batch for batch in batches if len(batch) == batch_size]
        if not batches:
            continue

        for tag, runner in runners.items():
            for batch in batches[:warmup]:
                runner(batch)
            timings = []
            for index in range(runs):
                batch = batches[index % len(batches)]
                start = time.perf_counter()
                runner(batch)
                timings.append((time.perf_counter() - start) * 1000)
            results.setdefault(tag, {})[f"batch_{batch_size}"] = _summarise(
                timings, batch_size
            )

    baseline = results.get("sklearn", {})
    for tag, per_batch in results.items():
        if tag == "sklearn":
            continue
        for key, stats in per_batch.items():
            reference = baseline.get(key, {}).get("mean_ms")
            if reference:
                stats["speedup_vs_sklearn"] = round(reference / stats["mean_ms"], 2)
    return results


def optimize_bundle(
    bundle: dict[str, Any],
    texts,
    *,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    int8_path: str | Path = DEFAULT_INT8_PATH,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full optimisation: export, quantise, and verify."""

    export_onnx(bundle["pipeline"], onnx_path)
    quantize(onnx_path, int8_path)
    report = verify_parity(bundle, texts, onnx_path=onnx_path, int8_path=int8_path)

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Export, quantise, verify, and benchmark from the command line."""

    import joblib

    from medical_triage.data import load_grouped_corpus

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--int8-path", default=str(DEFAULT_INT8_PATH))
    parser.add_argument("--report-path", default="artifacts/optimization.json")
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--runs", type=int, default=400)
    arguments = parser.parse_args(argv)

    bundle = joblib.load(arguments.model_path)
    corpus = load_grouped_corpus(arguments.dataset_dir, minimum_rows=2_000)
    texts = list(corpus.texts[: arguments.sample_size])

    print(f"amostras de verificacao: {len(texts)}")
    parity = optimize_bundle(
        bundle,
        texts,
        onnx_path=arguments.onnx_path,
        int8_path=arguments.int8_path,
    )
    print("\n--- paridade vs scikit-learn ---")
    for tag, stats in parity.items():
        print(
            f"  {tag:<10} concordancia={stats['label_agreement']:.5f}  "
            f"maxdiff={stats['max_abs_probability_diff']:.6f}  "
            f"{stats['size_mb']} MB"
        )

    latency = benchmark_latency(
        bundle,
        texts,
        onnx_path=arguments.onnx_path,
        int8_path=arguments.int8_path,
        runs=arguments.runs,
    )
    print("\n--- latencia ---")
    for batch_key in sorted(
        {key for stats in latency.values() for key in stats},
        key=lambda name: int(name.split("_")[1]),
    ):
        print(f"\n  {batch_key}")
        for tag, per_batch in latency.items():
            stats = per_batch.get(batch_key)
            if not stats:
                continue
            speedup = stats.get("speedup_vs_sklearn")
            suffix = f"  {speedup}x" if speedup else "  (baseline)"
            print(
                f"    {tag:<10} media={stats['mean_ms']:>8.3f}ms  "
                f"p95={stats['p95_ms']:>8.3f}ms{suffix}"
            )

    report_path = Path(arguments.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "sample_size": len(texts),
                "runs_per_measurement": arguments.runs,
                "parity": parity,
                "latency": latency,
                "sklearn_model_mb": round(
                    Path(arguments.model_path).stat().st_size / 1e6, 3
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nrelatorio salvo em {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
