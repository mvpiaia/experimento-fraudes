"""Identidade da preparacao antes/depois da materializacao dobra a dobra."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_datasets  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import prepare_lr_nb_knn_datasets as preparation  # noqa: E402
import preparation_report  # noqa: E402
import prepared_dataset_storage as storage  # noqa: E402


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "fold_sequential_identity.json"
PROTOCOLS = (
    experiment_datasets.STRATIFIED_HOLDOUT,
    experiment_datasets.STRATIFIED_K_FOLD,
    experiment_datasets.TEMPORAL_HOLDOUT,
)


def _definitions() -> dict[str, experiment_datasets.DatasetDefinition]:
    common = dict(
        dataset_name="Sintetica",
        legacy_aliases=(),
        source_files="memoria",
        prediction_column="label",
        identifier_columns_to_drop=("row_id",),
        temporal_order_column="event_time",
        group_column="client_id",
        evaluation_unit_column="client_id",
        protocols=PROTOCOLS,
        monetary_columns=("amount",),
    )
    return {
        "synthetic-full": experiment_datasets.DatasetDefinition(
            dataset_id="synthetic-full", **common
        ),
        "synthetic-ablation": experiment_datasets.DatasetDefinition(
            dataset_id="synthetic-ablation",
            ablated_columns=("optional",),
            **common,
        ),
        "synthetic-application": experiment_datasets.DatasetDefinition(
            dataset_id="synthetic-application",
            paired_dataset_id="synthetic-full",
            **common,
        ),
    }


def _raw_frames() -> dict[str, pd.DataFrame]:
    clients = np.arange(60)
    labels = (clients % 5 == 0).astype(int)
    repeated_clients = np.repeat(clients, 2)
    positions = np.arange(len(repeated_clients))
    full = pd.DataFrame(
        {
            "row_id": 10_000 + positions,
            "client_id": repeated_clients,
            "event_time": positions,
            "amount": 1.0 + positions % 17,
            "numeric": (positions % 11).astype(float),
            "optional": np.repeat(
                np.where(clients % 9 == 0, np.nan, clients % 7 + 0.5), 2
            ),
            "category": np.where(positions % 3 == 0, "a", "b"),
            "label": np.repeat(labels, 2),
        }
    )
    application = full.iloc[::2].reset_index(drop=True).copy()
    application["row_id"] += 100_000
    return {
        "synthetic-full": full,
        "synthetic-ablation": full.copy(),
        "synthetic-application": application,
    }


@pytest.fixture()
def synthetic_catalog(monkeypatch):
    definitions = _definitions()
    frames = _raw_frames()

    def get_definition(dataset_id):
        return definitions[dataset_id]

    def load_data(dataset_id):
        frame = frames[dataset_id].copy()
        frame.attrs["column_provenance"] = {
            "source_columns": list(frame.columns),
            "derived_columns": [],
            "constant_columns": [],
        }
        return frame, None, definitions[dataset_id]

    monkeypatch.setattr(preparation, "get_dataset_definition", get_definition)
    monkeypatch.setattr(preparation.pipeline, "load_data", load_data)
    monkeypatch.setattr(storage, "normalize_dataset_id", lambda dataset_id: dataset_id)
    return definitions


def _arguments(output: Path, **overrides) -> argparse.Namespace:
    values = dict(
        dataset="synthetic-full",
        categorical_encoding="woe",
        fill_na_strategy="mean",
        categorical_handling_mode="encoded",
        test_strategy=experiment_datasets.STRATIFIED_HOLDOUT,
        random_seed=42,
        prediction_column="label",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        output_directory=str(output),
        numeric_normalization="quantile",
        high_cardinality_strategy="minimum-frequency",
        high_cardinality_parameter=10,
        numeric_bins=10,
        window_length=15,
        history_length=10,
        window_stride=5,
        print_configuration_md5_only=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _manifest_lines(created) -> list[str]:
    lines = []
    for item in created:
        lines.append(
            "prepared_dataset="
            f"{item.dataset_md5}|{item.fold_index}|{item.split_label}|"
            f"{item.content_md5}|{Path(item.path).name}|{item.partition_md5}|"
            f"{item.feature_schema_md5}|{item.train_population_md5}|"
            f"{item.unit_partition_md5}|{item.train_unit_population_md5}"
        )
        if item.preparation_report_md5:
            lines.append(
                "preparation_report="
                f"{item.fold_index}|{item.dataset_md5}|"
                f"{item.preparation_report_md5}|"
                f"{item.missing_screening_threshold}|"
                f"{item.missing_screening_rule}|{item.screened_out_columns}"
            )
    return lines


def _snapshot_configuration(output: Path, **overrides) -> dict:
    configuration_md5, created = preparation.prepare_datasets(
        _arguments(output, **overrides)
    )
    report_hashes = {
        str(fold_index): storage.compute_file_content_md5(
            preparation_report.preparation_report_path(
                output, configuration_md5, fold_index
            )
        )
        for fold_index in sorted({item.fold_index for item in created})
    }
    manifest_payload = "\n".join(_manifest_lines(created)).encode("utf-8")
    return {
        "configuration_md5": configuration_md5,
        "content_md5": [item.content_md5 for item in created],
        "report_md5": report_hashes,
        "manifest_lines_md5": hashlib.md5(
            manifest_payload, usedforsecurity=False
        ).hexdigest(),
    }


def _current_snapshot(tmp_path: Path) -> dict:
    configurations = {
        "classic-holdout-mean": {},
        "classic-kfold-drop-rus": {
            "test_strategy": experiment_datasets.STRATIFIED_K_FOLD,
            "fill_na_strategy": pipeline.DROP_INCOMPLETE_ROWS_STRATEGY,
            "undersampling_technique": "random-undersampling",
        },
        "classic-temporal-rus-smote": {
            "test_strategy": experiment_datasets.TEMPORAL_HOLDOUT,
            "undersampling_technique": "rus+smote",
        },
        "native-holdout": {
            "categorical_handling_mode": "native",
            "categorical_encoding": None,
            "fill_na_strategy": "native",
        },
        "native-kfold-ablation": {
            "dataset": "synthetic-ablation",
            "categorical_handling_mode": "native",
            "categorical_encoding": None,
            "fill_na_strategy": "mean",
            "test_strategy": experiment_datasets.STRATIFIED_K_FOLD,
        },
        "paired-kfold-drop": {
            "dataset": "synthetic-application",
            "fill_na_strategy": pipeline.DROP_INCOMPLETE_ROWS_STRATEGY,
            "test_strategy": experiment_datasets.STRATIFIED_K_FOLD,
        },
    }
    return {
        name: _snapshot_configuration(tmp_path / name, **arguments)
        for name, arguments in configurations.items()
    }


def test_fold_sequential_preparation_is_bit_identical(
    tmp_path, synthetic_catalog, monkeypatch
):
    current = _current_snapshot(tmp_path)
    if not FIXTURE_PATH.exists():
        pytest.fail(
            "Frozen identity fixture is missing. Current snapshot:\n"
            + json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True)
        )
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert current == expected
