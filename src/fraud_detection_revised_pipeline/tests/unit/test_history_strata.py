"""Estratificacao pelo tamanho do historico em todos os modelos (item C3.19).

Protege o que torna a comparacao valida: o estrato de cada transacao e o
mesmo do indice de janelas do TabBERT, chega ao executor por um arquivo
lateral que nao pode faltar, cada estrato traz transacoes, fraudes e
prevalencia, e o validador recusa contagens diferentes entre algoritmos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import catboost_xgboost_specific as boosting  # noqa: E402
import data_partitioning  # noqa: E402
import experiment_datasets  # noqa: E402
import history_strata  # noqa: E402
import history_strata_analysis  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import prepare_lr_nb_knn_datasets as preparer  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402
import transformer_sequence_windows as windows  # noqa: E402
from lr_nb_knn_specific import PreparedDatasetStorageError  # noqa: E402
from prepared_dataset_storage import (  # noqa: E402
    load_prepared_data_splits,
    save_prepared_data_splits,
)

IEEE = experiment_datasets.DATASET_DEFINITIONS["ieee-cis-completa"]


def _ieee_like(holders: int = 30, records: int = 12, seed: int = 0) -> pd.DataFrame:
    """Base sintetica com as colunas de titular, ordem e alvo da IEEE-CIS."""
    rng = np.random.default_rng(seed)
    rows = holders * records
    return pd.DataFrame(
        {
            "TransactionID": np.arange(rows),
            "approximate_holder_id": np.repeat([f"h{index}" for index in range(holders)], records),
            "TransactionDT": rng.permutation(rows) * 10,
            "feature": rng.normal(size=rows),
            "isFraud": (rng.random(rows) < 0.2).astype(int),
        }
    )


def test_holder_with_twelve_records_gets_strata_zero_five_ten():
    fold = {
        "train_data": pd.DataFrame(
            {"approximate_holder_id": ["h"] * 12, "TransactionDT": range(12)}
        )
    }
    strata = windows.history_strata_by_split(fold, "ieee-cis-completa")["train_data"]
    assert strata.tolist() == [0] * 5 + [5] * 5 + [10] * 2


def test_partition_strata_match_the_tabbert_window_index():
    data = _ieee_like()
    tabbert = data_partitioning.build_partitions(
        data, None, IEEE, "temporal-holdout", 42, retained_columns=IEEE.sequence_keys
    )
    tabbert_fold = windows.assign_row_identity(tabbert)["test_info"][0]
    index = windows.build_sequence_window_index(tabbert_fold, "ieee-cis-completa")
    targets = index[index["is_target"]].set_index("row_id")["base_history_length"]

    native = data_partitioning.build_partitions(
        data, None, IEEE, "temporal-holdout", 42, history_window=(10, 5)
    )["test_info"][0]
    for split_name in data_partitioning.HISTORY_STRATA_SPLITS:
        expected = targets.reindex(tabbert_fold[split_name]["row_id"]).to_numpy()
        assert native["history_strata"][split_name].tolist() == expected.tolist()
        # Titular e ordem saem dos atributos, como sem o estrato.
        assert "approximate_holder_id" not in native[split_name].columns
        assert "TransactionDT" not in native[split_name].columns
    assert "train_data" not in native["history_strata"]


def test_strata_only_in_tabbert_datasets_protocol_and_comparable_modes():
    def window(dataset_id, test_strategy, mode):
        definition = experiment_datasets.DATASET_DEFINITIONS[dataset_id]
        args = argparse.Namespace(test_strategy=test_strategy, history_length=10, window_stride=5)
        return preparer._history_window(definition, args, mode)

    assert window("ieee-cis-completa", "temporal-holdout", "native") == (10, 5)
    assert window("ieee-cis-completa", "temporal-holdout", "transformer-indexed") == (10, 5)
    assert window("ieee-cis-completa", "stratified-k-fold", "native") is None
    assert window("ieee-cis-completa", "temporal-holdout", "encoded") is None
    assert window("ieee-cis-completa", "temporal-holdout", "sequence-tokenized") is None
    assert window("ulb", "temporal-holdout", "native") is None
    assert window("home-credit-completa", "stratified-holdout", "native") == (10, 5)


def _save_fold_with_strata(directory: Path, with_strata: bool = True) -> tuple[str, Path]:
    native = data_partitioning.build_partitions(
        _ieee_like(), None, IEEE, "temporal-holdout", 42,
        history_window=(10, 5) if with_strata else None,
    )
    configuration_md5 = "c" * 32
    created = save_prepared_data_splits(native, directory, configuration_md5)
    manifest = pd.DataFrame(
        [
            {
                "database_type": "ieee-cis-completa",
                "dataset_id": "ieee-cis-completa",
                "configuration_md5": configuration_md5,
                "dataset_md5": item.dataset_md5,
                "content_md5": item.content_md5,
                "fold_index": item.fold_index,
                "split": item.split_label,
                "test_strategy": "temporal-holdout",
                "categorical_handling_mode": "native",
            }
            for item in created
        ]
    )
    manifest_path = directory / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return configuration_md5, manifest_path


def test_side_file_is_written_staged_and_loaded(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    configuration_md5, manifest_path = _save_fold_with_strata(source)
    assert sorted(path.name.split(".", 1)[1] for path in source.glob("*.history.csv")) == [
        "history.csv",
        "history.csv",
    ]
    with pipeline.stage_prepared_dataset_files(
        source, manifest_path, configuration_md5, tmp_path / "staging"
    ) as staged:
        loaded = load_prepared_data_splits(
            staged, manifest_path, configuration_md5, "temporal-holdout"
        )
    fold = loaded["test_info"][0]
    for split_name in ("validation_data", "test_data"):
        assert len(fold["history_strata"][split_name]) == len(fold[split_name])


def test_missing_side_file_is_a_storage_failure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    configuration_md5, manifest_path = _save_fold_with_strata(source)
    next(source.glob("*.history.csv")).unlink()
    with pytest.raises(PreparedDatasetStorageError, match="History strata"):
        with pipeline.stage_prepared_dataset_files(
            source, manifest_path, configuration_md5, tmp_path / "staging"
        ):
            pass


def test_stratum_without_fraud_keeps_the_other_metrics():
    result = history_strata.stratum_metrics(
        targets=[0, 0, 0, 1, 0, 1],
        probabilities=[0.1, 0.2, 0.3, 0.9, 0.2, 0.8],
        strata=[0, 0, 0, 10, 10, 10],
        threshold=0.5,
    )
    assert result["0"]["transactions"] == 3
    assert result["0"]["positives"] == 0
    assert result["0"]["prevalence"] == 0.0
    assert np.isnan(result["0"]["roc_auc"])
    assert result["0"]["accuracy"] == 1.0
    assert result["10"]["positives"] == 2
    assert result["10"]["prevalence"] == pytest.approx(2 / 3)
    assert result["10"]["roc_auc"] == 1.0


def _record(algorithm: str, transactions: int, positives: int = 1) -> dict:
    return {
        "dataset_id": "sparkov", "test_strategy": "temporal-holdout", "random_seed": 42,
        "fold_index": 1, "split": "test", "algorithm": algorithm, "regime": "",
        "configuration_md5": algorithm,
        "history_strata": {
            "base_history_length": {
                "0": {"transactions": transactions, "positives": positives, "prevalence": 0.1,
                      "f1": 0.5, "f_beta_2": 0.5, "roc_auc": 0.9, "auc_pr": 0.4},
                "10": {"transactions": 90, "positives": 9, "prevalence": 0.1,
                       "f1": 0.7, "f_beta_2": 0.6, "roc_auc": 0.95, "auc_pr": 0.5},
            }
        },
    }


class _Report:
    def __init__(self):
        self.entries = []

    def add(self, check_id, scope, passed, detail):
        self.entries.append((check_id, scope, passed, detail))


def test_r6_flags_different_counts_between_algorithms():
    report = _Report()
    sanity.check_history_strata([_record("catboost", 10), _record("tabbert-mlp", 11)], report)
    failures = [entry for entry in report.entries if not entry[2]]
    assert len(failures) == 1
    assert "estrato 0" in failures[0][1]
    assert "catboost" in failures[0][3] and "tabbert-mlp" in failures[0][3]

    report = _Report()
    sanity.check_history_strata([_record("catboost", 10), _record("tabbert-mlp", 10)], report)
    assert all(entry[2] for entry in report.entries)


def test_boosting_records_strata_of_row_folds_only():
    predictions = pd.DataFrame(
        {"real_value": [0, 1, 0, 1], "predicted_value": [0, 1, 1, 1],
         "boosting_probability": [0.1, 0.9, 0.6, 0.7]}
    )
    row_fold = {
        "fold_index": 1, "class_values": [0, 1], "classification_threshold": 0.5,
        "threshold_strategy": "validation-f2",
        "history_strata": {"test_data": pd.Series([0, 0, 10, 10])},
        "split_results": {"test_data": {"predictions": predictions}},
    }
    client_fold = {**row_fold, "evaluation_unit": "client"}
    configuration = argparse.Namespace(
        configuration_md5="d" * 32, database_type="sparkov", test_strategy="temporal-holdout",
        random_seed=42, class_weight_intensity="none",
    )
    records = boosting.history_strata_records(
        configuration, {"algorithm": "catboost", "results": [row_fold, client_fold]}
    )
    assert len(records) == 1
    strata = records[0]["history_strata"]["base_history_length"]
    assert records[0]["split"] == "test"
    assert strata["0"]["transactions"] == 2 and strata["0"]["positives"] == 1
    assert strata["10"]["false_positive_rate"] == 1.0


def test_analysis_table_reports_prevalence_and_difference_to_full_history():
    rows = history_strata_analysis.strata_rows(
        [_record("catboost", 10), _record("tabbert-mlp", 10)]
    )
    summary = history_strata_analysis.summarize(rows)
    first = summary[(summary["algorithm"] == "catboost") & (summary["stratum"] == 0)].iloc[0]
    assert first["prevalence"] == 0.1
    assert first["f1_difference"] == pytest.approx(0.5 - 0.7)
    table = history_strata_analysis.latex_table(summary)
    assert "Prevalência" in table and "10{,}00\\%" in table
