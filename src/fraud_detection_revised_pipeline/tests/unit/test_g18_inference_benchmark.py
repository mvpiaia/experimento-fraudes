"""Benchmark de inferencia (item G18 da revisao).

Protege a medicao que responde ao pedido de tempo de inferencia por 1.000
transacoes: a configuracao de referencia e escolhida de forma deterministica,
os postos do IC do p95 sao os da binomial, a latencia unitaria pontua uma
transacao por vez, e cada linha traz a proveniencia da maquina.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import benchmark_inference as benchmark  # noqa: E402
from test_anomaly_detection import TARGET, prepare_native_configuration  # noqa: E402
from test_g18_provenance_and_timing import _imbalanced_split  # noqa: E402


def test_p95_confidence_ranks_follow_the_binomial():
    # n = 1000: o IC do p95 vai do quantil ~93,6 ao ~96,5.
    assert benchmark.quantile_confidence_ranks(1000) == (936, 965)
    # n = 100 e largo demais: do quantil 90 ao 100.
    low, high = benchmark.quantile_confidence_ranks(100)
    assert (low, high) == (90, 100)


def test_latency_scores_one_transaction_at_a_time():
    seen_sizes = []

    def score(features):
        seen_sizes.append(len(features))
        return np.zeros(len(features))

    features = pd.DataFrame({"x": np.arange(50.0)})
    latency = benchmark.measure_latency(score, features, sample_size=20)
    # Aquecimento mais as 20 transacoes da amostra, todas com uma linha.
    assert seen_sizes == [1] * 21
    assert latency["latency_sample_size"] == 20
    assert latency["latency_p95_ci_low_ms"] <= latency["latency_p95_ms"] <= latency["latency_p95_ci_high_ms"]


def test_sample_never_exceeds_the_test_split():
    features = pd.DataFrame({"x": np.arange(5.0)})
    latency = benchmark.measure_latency(lambda frame: np.zeros(len(frame)), features, 1000)
    assert latency["latency_sample_size"] == 5


def test_batch_throughput_is_per_thousand_transactions():
    calls = []
    features = pd.DataFrame({"x": np.arange(2000.0)})
    value = benchmark.measure_batch(lambda frame: calls.append(len(frame)), features, repeats=3)
    assert calls == [2000] * 4  # aquecimento + 3 passadas
    assert value >= 0


def _manifest(rows):
    base = {
        "dataset_id": "ulb",
        "categorical_handling_mode": "native",
        "test_strategy": "stratified-holdout",
        "random_seed": "42",
        "undersampling_technique": "none",
        "categorical_encoding_strategy": "none",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_reference_configuration_prefers_median_without_resampling():
    manifest = _manifest(
        [
            {"configuration_md5": "a", "fill_na_strategy": "mean"},
            {"configuration_md5": "b", "fill_na_strategy": "median"},
            {"configuration_md5": "c", "fill_na_strategy": "median", "undersampling_technique": "rus"},
            {"configuration_md5": "d", "fill_na_strategy": "median", "random_seed": "7"},
        ]
    )
    reference = benchmark.select_reference_configuration(manifest, "ulb", benchmark.BLOCK_BOOSTING)
    assert reference["configuration_md5"] == "b"


def test_reference_configuration_falls_back_deterministically():
    manifest = _manifest(
        [
            {"configuration_md5": "z", "fill_na_strategy": "none"},
            {"configuration_md5": "y", "fill_na_strategy": "mean"},
        ]
    )
    reference = benchmark.select_reference_configuration(manifest, "ulb", benchmark.BLOCK_ANOMALY)
    assert reference["configuration_md5"] == "y"
    assert benchmark.select_reference_configuration(manifest, "ulb", benchmark.BLOCK_CLASSIC) is None


@pytest.mark.parametrize("algorithm", ["logistic-regression", "naive-bayes", "knn"])
def test_classic_scorer_returns_probabilities(algorithm):
    fold = {
        "train_data": _imbalanced_split(300, 1),
        "validation_data": _imbalanced_split(100, 2),
        "test_data": _imbalanced_split(100, 3),
    }
    scorer = benchmark.fit_scorer(algorithm, fold, "alvo", "ulb")
    scores = scorer.score(scorer.features.iloc[[0]])
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0
    assert scorer.fit_time > 0


@pytest.mark.parametrize("algorithm", ["isolation-forest", "xgboost", "devnet"])
def test_benchmark_writes_one_row_with_provenance(tmp_path, algorithm):
    configuration_md5, _command_line = prepare_native_configuration(
        tmp_path, np.random.default_rng(5)
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    manifest = manifest.assign(
        prediction_column_name=TARGET,
        test_strategy="stratified-holdout",
        random_seed="42",
        undersampling_technique="none",
        fill_na_strategy="none",
    )
    manifest.to_csv(manifest_path, sep="\t", index=False)
    args = benchmark.build_argument_parser().parse_args(
        [
            "--prepared-data-directory", str(tmp_path / "prepared"),
            "--prepared-dataset-manifest", str(manifest_path),
            "--results-directory", str(tmp_path / "results"),
            "--algorithm", algorithm,
            "--latency-sample-size", "30",
            "--batch-repeats", "2",
        ]
    )
    assert benchmark.run_benchmark(args) == 0
    rows = pd.read_csv(tmp_path / "results" / benchmark.RESULT_FILENAME, dtype=str)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["algorithm"] == algorithm
    assert row["configuration_md5"] == configuration_md5
    assert row["latency_sample_size"] == "30"
    assert float(row["inference_ms_per_1000"]) > 0
    assert row["execution_host"]
    assert row["environment_md5"]


def test_batch_sample_is_fixed_and_bounded():
    features = pd.DataFrame({"x": range(50_000)})
    first = benchmark.batch_sample(features, 10_000)
    second = benchmark.batch_sample(features, 10_000)
    assert len(first) == 10_000
    assert first.index.equals(second.index)
    assert benchmark.batch_sample(features, 0) is features
    assert len(benchmark.batch_sample(features.iloc[:500], 10_000)) == 500
