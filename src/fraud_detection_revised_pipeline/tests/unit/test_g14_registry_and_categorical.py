"""Item G14: codigos categoricos em todos os blocos e registro de colunas."""

from __future__ import annotations

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
import prepared_dataset_storage as storage  # noqa: E402
from anomaly_detection_specific import LabelFreeEncoder  # noqa: E402
from catboost_xgboost_specific import discover_categorical_columns  # noqa: E402
from transformer_feature_schema import resolve_feature_types  # noqa: E402

from test_g14_catalog_and_loading import (  # noqa: E402
    _write_ieee_files,
    _write_sparkov_files,
)


@pytest.fixture()
def raw_data_root(tmp_path, monkeypatch):
    _write_sparkov_files(tmp_path)
    _write_ieee_files(tmp_path)
    monkeypatch.setattr(pipeline, "CURRENT_PATH", "")
    monkeypatch.setattr(pipeline, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


def _prepare(monkeypatch, tmp_path, dataset, mode="native", strategy="stratified-holdout",
             encoding=None, fill_na="none", label="isFraud", extra=()):
    output = tmp_path / f"out-{dataset}-{mode}"
    arguments = [
        "prepare",
        "--dataset", dataset,
        "--test-strategy", strategy,
        "--prediction-column", label,
        "--fill-na-strategy", fill_na,
        "--categorical-handling-mode", mode,
        "--output-directory", str(output),
        "--auxiliary-data-is-test-data", "false",
    ]
    if encoding:
        arguments += ["--categorical-encoding", encoding]
    monkeypatch.setattr(sys, "argv", arguments + list(extra))
    md5, created = preparation.prepare_datasets(preparation.parse_arguments())
    return md5, created, output


# --- 4.2: codigos categoricos em cada bloco -------------------------------------


def test_transformer_schema_takes_the_codes_from_the_catalog():
    frame = pd.DataFrame(
        {
            "card1": ["1", "2"],
            "card2": ["3", "4"],
            "addr1": ["5", "6"],
            "TransactionAmt": [1.0, 2.0],
            "isFraud": [0, 1],
        }
    )
    categorical, numerical = resolve_feature_types(
        "ieee-cis-transacional", frame, "isFraud"
    )
    assert {"card1", "card2", "addr1"} <= set(categorical)
    assert "TransactionAmt" in numerical


def test_ablation_variant_inherits_identity_categoricals_and_drops_its_columns():
    frame = pd.DataFrame(
        {
            "card1": ["1", "2"],
            "card2": ["3", "4"],
            "id_01": [1, 2],
            "DeviceInfo": ["a", "b"],
            "TransactionAmt": [1.0, 2.0],
            "isFraud": [0, 1],
        }
    )
    categorical, numerical = resolve_feature_types(
        "ieee-cis-completa-sem-alta-cardinalidade", frame, "isFraud"
    )
    assert "id_01" in categorical  # herdado da ieee-cis-completa
    assert "card2" in categorical
    for removed in ("card1", "DeviceInfo"):
        assert removed not in categorical and removed not in numerical


def test_classic_block_encodes_card1_as_a_category(raw_data_root, monkeypatch):
    seen = {}
    original = pipeline._encode_supervised_data_splits

    def spy(split_information, strategy, target):
        train = split_information["test_info"][0]["train_data"]
        seen["categorical"] = list(pipeline._categorical_columns(train))
        return original(split_information, strategy, target)

    monkeypatch.setattr(pipeline, "_encode_supervised_data_splits", spy)
    _prepare(
        monkeypatch,
        raw_data_root,
        "ieee-cis-transacional",
        mode="encoded",
        encoding="smoothed-mean:k-fold",
        fill_na="mean",
    )
    assert {"card1", "card2", "addr1"} <= set(seen["categorical"])


def test_native_files_return_card1_as_text_to_boosting_and_anomaly(
    raw_data_root, monkeypatch
):
    md5, created, output = _prepare(monkeypatch, raw_data_root, "ieee-cis-transacional")
    manifest = pd.DataFrame(
        {
            "database_type": "ieee-cis-transacional",
            "configuration_md5": md5,
            "dataset_md5": [item.dataset_md5 for item in created],
            "content_md5": [item.content_md5 for item in created],
            "fold_index": [str(item.fold_index) for item in created],
            "split": [item.split_label for item in created],
            "categorical_handling_mode": "native",
            "feature_schema_md5": [item.feature_schema_md5 for item in created],
        }
    )
    manifest_path = output / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    loaded = storage.load_prepared_data_splits(
        output, manifest_path, md5, "stratified-holdout"
    )
    train = loaded["test_info"][0]["train_data"]
    assert pd.api.types.is_string_dtype(train["card1"])
    assert train["card1"].iloc[0].isdigit()
    features = train.drop(columns=["isFraud"])
    assert {"card1", "card2", "addr1"} <= set(discover_categorical_columns(features))
    # O detector de anomalia codifica por frequencia, nao como grandeza.
    encoder = LabelFreeEncoder.fit(features, discover_categorical_columns(features))
    encoded = encoder.transform(features)
    assert encoded["card1"].between(0.0, 1.0).all()


# --- 5: registro de colunas e identidade ----------------------------------------


def test_preprocessing_revision_is_part_of_the_configuration_identity(monkeypatch):
    arguments = dict(
        dataset="ulb",
        categorical_encoding=None,
        fill_na_strategy="none",
        test_strategy="stratified-holdout",
        prediction_column="Class",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
        random_seed=42,
    )
    current = storage.build_prepared_dataset_md5(**arguments)
    pre_g14 = hashlib.md5(
        "ulb|none|none|stratified-holdout|Class|none|none|False|42|encoded".encode(),
        usedforsecurity=False,
    ).hexdigest()
    assert current != pre_g14
    monkeypatch.setattr(storage, "PREPROCESSING_REVISION", "g99")
    assert storage.build_prepared_dataset_md5(**arguments) != current


def test_feature_schema_md5_hashes_the_ordered_columns():
    expected = hashlib.md5(b"a|b|c", usedforsecurity=False).hexdigest()
    assert storage.compute_feature_schema_md5(["a", "b", "c"]) == expected
    assert storage.compute_feature_schema_md5(["b", "a", "c"]) != expected


def test_registry_json_matches_the_csv_and_the_returned_hash(tmp_path):
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4], "label": [0, 1]})
    information = {
        "train_test_strategy": "stratified-holdout",
        "test_info": [{"train_data": frame, "validation_data": None, "test_data": frame}],
        "column_registry": {"source_columns": ["x", "y", "label"], "pca_applied": False},
    }
    created = storage.save_prepared_data_splits(information, tmp_path, "c" * 32)
    for item in created:
        payload = json.loads(
            storage.column_registry_path(tmp_path, item.dataset_md5).read_text("utf-8")
        )
        header = pd.read_csv(item.path, nrows=0).columns.tolist()
        assert payload["columns"] == header
        assert payload["feature_schema_md5"] == item.feature_schema_md5
        assert item.feature_schema_md5 == storage.compute_feature_schema_md5(header)
        assert payload["source_columns"] == ["x", "y", "label"]
        assert set(payload["dropped_columns"]) == {"identifier", "constant", "ablation"}


def test_sparkov_variant_registry_names_the_reason_of_each_removal(
    raw_data_root, monkeypatch
):
    _md5, created, output = _prepare(
        monkeypatch,
        raw_data_root,
        "sparkov-sem-titular",
        label="is_fraud",
        extra=("--auxiliary-data-is-test-data", "true"),
    )
    payload = json.loads(
        storage.column_registry_path(output, created[0].dataset_md5).read_text("utf-8")
    )
    assert payload["dropped_columns"]["ablation"] == [
        "zip", "lat", "long", "city", "city_pop", "job"
    ]
    assert "cc_num" in payload["dropped_columns"]["identifier"]
    assert "always_the_same" in payload["dropped_columns"]["constant"]
    assert "customer_merchant_km" in payload["derived_columns"]
    assert "customer_merchant_km" in payload["columns"]
    assert "cc_num" not in payload["columns"]
    assert payload["log_transformed_columns"] == []  # nativo: valor original
    assert "category" in payload["categorical_columns"]


def test_classic_registry_records_log_transform_and_pca(raw_data_root, monkeypatch):
    _md5, created, output = _prepare(
        monkeypatch,
        raw_data_root,
        "ieee-cis-transacional",
        mode="encoded",
        encoding="smoothed-mean:k-fold",
        fill_na="mean",
    )
    payload = json.loads(
        storage.column_registry_path(output, created[0].dataset_md5).read_text("utf-8")
    )
    assert payload["log_transformed_columns"] == ["TransactionAmt"]
    assert payload["pca_applied"] is True
    assert all(column.startswith("PC") or column == "isFraud" for column in payload["columns"])
    assert "M1" in payload["dropped_columns"]["constant"]


def _manifest_for(created, md5, output, database_type, schema_md5=None):
    frame = pd.DataFrame(
        {
            "database_type": database_type,
            "configuration_md5": md5,
            "dataset_md5": [item.dataset_md5 for item in created],
            "content_md5": [item.content_md5 for item in created],
            "fold_index": [str(item.fold_index) for item in created],
            "split": [item.split_label for item in created],
            "categorical_handling_mode": "native",
        }
    )
    if schema_md5 is not None:
        frame["feature_schema_md5"] = (
            schema_md5 if isinstance(schema_md5, list) else [schema_md5] * len(created)
        )
    path = output / "manifest.tsv"
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_divergent_schema_hash_fails_the_read(raw_data_root, monkeypatch):
    md5, created, output = _prepare(monkeypatch, raw_data_root, "ieee-cis-transacional")
    manifest = _manifest_for(created, md5, output, "ieee-cis-transacional", "f" * 32)
    with pytest.raises(ValueError, match="feature schema MD5"):
        storage.load_prepared_data_splits(output, manifest, md5, "stratified-holdout")


def test_staging_requires_the_registry_when_the_manifest_records_the_hash(
    raw_data_root, monkeypatch, tmp_path
):
    md5, created, output = _prepare(monkeypatch, raw_data_root, "ieee-cis-transacional")
    manifest = _manifest_for(
        created,
        md5,
        output,
        "ieee-cis-transacional",
        [item.feature_schema_md5 for item in created],
    )
    staging = tmp_path / "staging"
    with pipeline.stage_prepared_dataset_files(output, manifest, md5, staging) as staged:
        for item in created:
            assert storage.column_registry_path(staged, item.dataset_md5).is_file()

    storage.column_registry_path(output, created[0].dataset_md5).unlink()
    with pytest.raises(pipeline.PreparedDatasetStorageError, match="Column registry"):
        with pipeline.stage_prepared_dataset_files(output, manifest, md5, staging):
            pass


def test_manifest_without_the_hash_column_is_read_without_the_registry(
    raw_data_root, monkeypatch, tmp_path
):
    md5, created, output = _prepare(monkeypatch, raw_data_root, "ieee-cis-transacional")
    manifest = _manifest_for(created, md5, output, "ieee-cis-transacional")
    for item in created:
        storage.column_registry_path(output, item.dataset_md5).unlink()
    staging = tmp_path / "staging"
    with pipeline.stage_prepared_dataset_files(output, manifest, md5, staging):
        pass
    loaded = storage.load_prepared_data_splits(output, manifest, md5, "stratified-holdout")
    assert loaded["test_info"]


def test_old_configurations_are_not_reused(raw_data_root, monkeypatch):
    new_md5, _created, _output = _prepare(
        monkeypatch, raw_data_root, "ieee-cis-transacional"
    )
    monkeypatch.setattr(storage, "PREPROCESSING_REVISION", "g13")
    old_md5 = storage.build_prepared_dataset_md5(
        dataset="ieee-cis-transacional",
        categorical_encoding=None,
        fill_na_strategy="none",
        test_strategy="stratified-holdout",
        prediction_column="isFraud",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="native",
        random_seed=42,
    )
    assert old_md5 != new_md5
