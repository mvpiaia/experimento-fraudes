"""Escores do teste por linha e bootstrap pareado sobre as transacoes (item G20)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import analise_hipoteses as analysis  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402


CLASS_VALUES = [0, 1]


def _scored_fold(targets, scores, fold_index=1):
    """Dobra pontuada como os executores a entregam a apply_decision_thresholds."""
    target = pd.Series(targets)
    probabilities = np.column_stack([1 - np.asarray(scores), np.asarray(scores)])
    split_scores = {
        name: pipeline._score_split(target, probabilities, [0, 1], CLASS_VALUES, name, "probability")
        for name in ("validation_data", "test_data")
    }
    return {
        "fold_index": fold_index,
        "class_values": CLASS_VALUES,
        "split_scores": split_scores,
        "split_md5": {"test_data": "md5-teste"},
        "partition_md5": {"test_data": "particao-teste"},
        "split_sizes": {"test_data": len(targets)},
    }


def _record(tmp_path, folds, algorithm="knn", **kwargs):
    pipeline.apply_decision_thresholds(folds, "fixed", 0.5)
    destination = tmp_path / f"resultados_{algorithm}.csv"
    pipeline.record_algorithm_result_to_csv(
        str(destination),
        "",
        {"algorithm": algorithm, "results": folds, "train_test_strategy": "temporal-holdout",
         "undersampling_technique": "none"},
        "none",
        "mean",
        1.0,
        configuration_md5="abc",
        database_type="ulb",
        **kwargs,
    )
    return destination


def test_recorder_writes_test_scores_companion(tmp_path):
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, 37)
    scores = rng.random(37)
    destination = _record(tmp_path, [_scored_fold(targets, scores)])

    path = Path(pipeline.test_scores_path(destination))
    assert path.name == "resultados_knn_test_scores.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["split"] == "test"
    assert record["fold_index"] == 1
    assert record["configuration_md5"] == "abc"
    assert record["test_partition_md5"] == "particao-teste"
    assert record["positives"] == int(targets.sum())

    decoded_targets, decoded_scores = pipeline.decode_test_scores(record)
    np.testing.assert_array_equal(decoded_targets, targets)
    np.testing.assert_allclose(decoded_scores, scores.astype(np.float32))
    # O script de analise le o mesmo formato sem importar o executor.
    analysis_targets, analysis_scores = analysis.decode_test_scores(record)
    np.testing.assert_array_equal(analysis_targets, decoded_targets)
    np.testing.assert_array_equal(analysis_scores, decoded_scores)


def test_recorder_can_skip_scores_and_ignores_results_without_predictions(tmp_path):
    rng = np.random.default_rng(1)
    destination = _record(
        tmp_path, [_scored_fold(rng.integers(0, 2, 10), rng.random(10))], record_test_scores=False
    )
    assert not Path(pipeline.test_scores_path(destination)).exists()

    # Os Transformers entregam so as metricas; os escores ficam no _details.jsonl.
    destination = tmp_path / "resultados_ft-transformer-mlp.csv"
    pipeline.record_algorithm_result_to_csv(
        str(destination), "",
        {"algorithm": "ft-transformer-mlp",
         "results": [{"split_results": {"test_data": {"metrics": {"f_beta_2": 0.5}}}}]},
        "", "", None, configuration_md5="def", database_type="ulb",
    )
    assert not Path(pipeline.test_scores_path(destination)).exists()


def test_paired_bootstrap_of_identical_predictions_is_zero():
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 2, 500).astype(np.uint8)
    predictions = rng.random(500) > 0.5
    point, draws = analysis.paired_transaction_bootstrap([(targets, predictions, predictions)], rng, 300)
    assert point == 0.0
    assert np.all(draws == 0.0)


def test_paired_bootstrap_matches_direct_f2_difference():
    targets = np.array([1] * 10 + [0] * 90, dtype=np.uint8)
    first = np.array([True] * 9 + [False] + [True] * 5 + [False] * 85)
    second = np.array([True] * 5 + [False] * 5 + [True] * 2 + [False] * 88)
    point, draws = analysis.paired_transaction_bootstrap(
        [(targets, first, second)], np.random.default_rng(3), 2000
    )
    expected = 5 * 9 / (5 * 9 + 4 * 1 + 5) - 5 * 5 / (5 * 5 + 4 * 5 + 2)
    assert point == pytest.approx(expected)
    assert np.percentile(draws, 2.5) < point < np.percentile(draws, 97.5)


# ------------------------------------------------- analise com escores reais


SIZE = 4000
POSITIVE_RATE = 0.05


def _execution(algorithm, source, hits, rng, targets):
    """Linhas do CSV e registro de escores coerentes entre si."""
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    predicted = np.zeros(SIZE, dtype=bool)
    predicted[positives[:hits]] = True
    predicted[rng.choice(negatives, 20, replace=False)] = True
    scores = np.where(predicted, 0.9, 0.1)
    tp = int(predicted[positives].sum())
    fp = int(predicted[negatives].sum())
    positive_total = len(positives)
    f2 = 5 * tp / (5 * tp + 4 * (positive_total - tp) + fp)
    base = {
        "configuration_md5": f"{algorithm}-{source}",
        "dataset_id": source,
        "database_type": source,
        "algorithm": algorithm,
        "train_validation_test_strategy": analysis.TEMPORAL,
        "undersampling_technique": "none",
        "random_seed": 42,
        "fold_index": 1,
        "classification_threshold": 0.5,
        "threshold_strategy": "fixed",
        "execution_status": "success",
        "evaluation_unit": "row",
        "precision": tp / (tp + fp),
        "recall": tp / positive_total,
        "false_positive_rate": fp / len(negatives),
        "auc_pr": 0.5,
        "roc_auc": 0.9,
        "test_df_size": SIZE,
    }
    rows = [
        {**base, "split": "validation_data", "f_beta_2": hits / positive_total},
        {**base, "split": "test_data", "f_beta_2": f2},
    ]
    encoded = pipeline.encode_test_scores(
        pd.DataFrame({"real_value": targets, "predicted_value": predicted.astype(int), "p": scores}),
        CLASS_VALUES,
    )
    record = {
        "record_type": "test_scores", "configuration_md5": base["configuration_md5"],
        "algorithm": algorithm, "regime": "", "fold_index": 1, "split": "test",
        "evaluation_unit": "row", "random_seed": 42, "threshold_strategy": "fixed",
        "undersampling_technique": "none", "test_partition_md5": f"particao-{source}", **encoded,
    }
    return rows, record


def _grid(leader_hits, other_hits, mismatched_targets=False):
    rng = np.random.default_rng(4)
    rows, store = [], analysis.TestScoreStore()
    for source in analysis.FRAUD_SOURCES:
        targets = (rng.random(SIZE) < POSITIVE_RATE).astype(np.uint8)
        for algorithm, hits in (("catboost", leader_hits), ("knn", other_hits), ("xgboost", other_hits - 5)):
            execution_targets = targets
            if mismatched_targets and algorithm == "knn":
                execution_targets = np.roll(targets, 1)
            execution_rows, record = _execution(algorithm, source, hits, rng, execution_targets)
            rows += execution_rows
            store.add(record, "test_scores")
    return analysis.normalize_results(pd.DataFrame(rows)), store


def test_analysis_prefers_paired_transaction_bootstrap():
    data, store = _grid(leader_hits=180, other_hits=120)
    _, summary = analysis.run_analysis(data, bootstrap_samples=500, scores=store)
    assert summary["uncertainty_units"] == ["transaction"]
    assert summary["test_score_records"] == 9
    detail = summary["H2"]["leader_vs_runner_up"][0]
    assert detail["units"] == SIZE
    assert detail["ci_low"] < detail["difference"] < detail["ci_high"]
    assert summary["H2"]["outcome"] == analysis.OUTCOME_CORROBORATED


def test_rows_that_differ_fall_back_to_unpaired_unit():
    data, store = _grid(leader_hits=180, other_hits=120, mismatched_targets=True)
    _, summary = analysis.run_analysis(data, bootstrap_samples=300, scores=store)
    assert summary["uncertainty_units"] == ["confusion"]


def test_store_reads_companion_files_and_transformer_details(tmp_path):
    targets = np.array([0, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.8, 0.3, 0.6])
    encoded = pipeline.encode_test_scores(
        pd.DataFrame({"real_value": targets, "predicted_value": targets, "p": scores}), CLASS_VALUES
    )
    (tmp_path / "resultados_knn_test_scores.jsonl").write_text(json.dumps({
        "configuration_md5": "a", "algorithm": "knn", "regime": "", "fold_index": 1,
        "split": "test", "random_seed": 7, **encoded,
    }) + "\n")
    (tmp_path / "resultados_ft-transformer-mlp_details.jsonl").write_text(
        json.dumps({
            "configuration_md5": "b", "algorithm": "ft-transformer-mlp", "regime": "", "fold_index": 1,
            "split": "validation", "random_seed": 7, "transaction_predictions": [{"target": 1.0, "probability": 0.2}],
        }) + "\n" + json.dumps({
            "configuration_md5": "b", "algorithm": "ft-transformer-mlp", "regime": "", "fold_index": 1,
            "split": "test", "random_seed": 7,
            "transaction_predictions": [
                {"prediction_index": index, "target": float(t), "probability": float(p)}
                for index, (t, p) in enumerate(zip(targets, scores))
            ],
        }) + "\n"
    )
    store = analysis.TestScoreStore.from_directory(tmp_path)
    assert len(store) == 2
    for md5, algorithm in (("a", "knn"), ("b", "ft-transformer-mlp")):
        execution = pd.Series({"configuration_md5": md5, "algorithm": algorithm, "regime": "",
                               "class_weight_intensity": "", "threshold_strategy": "",
                               "undersampling_technique": ""})
        entry = store.lookup(execution, 7, 1)
        loaded_targets, loaded_scores = store.arrays(entry)
        np.testing.assert_array_equal(loaded_targets, targets)
        np.testing.assert_allclose(loaded_scores, scores, rtol=1e-6)
