from scripts.benchmark import percentile, summarize


def test_percentiles_and_summary_are_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 4.8
    assert summarize(values) == {
        "min_ms": 1.0,
        "mean_ms": 3.0,
        "p50_ms": 3.0,
        "p95_ms": 4.8,
        "p99_ms": 4.96,
        "max_ms": 5.0,
    }

