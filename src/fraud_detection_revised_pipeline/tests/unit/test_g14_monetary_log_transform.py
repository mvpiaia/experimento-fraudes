"""Item G14: log1p dos valores monetarios antes da padronizacao."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import anomaly_detection_specific as anomaly  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import prepare_lr_nb_knn_datasets as preparation  # noqa: E402
from label_free_encoding import LabelFreeEncoder  # noqa: E402
from monetary_transform import (  # noqa: E402
    NegativeMonetaryValueError,
    log1p_columns,
    log1p_data_splits,
)
from prepared_dataset_storage import FIRST_FOLD_INDEX  # noqa: E402


TARGET = "Class"
ROW_COUNT = 300


# --- funcao comum -------------------------------------------------------------


def test_log1p_columns_transforms_only_the_declared_columns():
    frame = pd.DataFrame({"amt": [0.0, 9.0, 99.0], "other": [1.0, 2.0, 3.0]})
    result = log1p_columns(frame, ("amt",))
    assert list(result["amt"]) == pytest.approx(list(np.log1p([0.0, 9.0, 99.0])))
    assert list(result["other"]) == [1.0, 2.0, 3.0]
    assert list(frame["amt"]) == [0.0, 9.0, 99.0]


def test_log1p_ignores_absent_columns_and_none():
    frame = pd.DataFrame({"x": [1.0]})
    assert log1p_columns(frame, ("amt",)) is frame
    assert log1p_columns(None, ("amt",)) is None


def test_negative_monetary_value_names_dataset_and_column():
    frame = pd.DataFrame({"Amount": [1.0, -5.0]})
    with pytest.raises(NegativeMonetaryValueError, match="Amount.*ulb"):
        log1p_columns(frame, ("Amount",), "ulb")


def test_missing_monetary_values_stay_missing():
    result = log1p_columns(pd.DataFrame({"amt": [np.nan, 3.0]}), ("amt",))
    assert np.isnan(result["amt"].iloc[0])
    assert result["amt"].iloc[1] == pytest.approx(np.log(4.0))


def test_log1p_data_splits_covers_train_validation_and_test():
    fold = {
        "train_data": pd.DataFrame({"amt": [1.0]}),
        "validation_data": pd.DataFrame({"amt": [2.0]}),
        "test_data": None,
    }
    result = log1p_data_splits({"test_info": [fold]}, ("amt",))
    transformed = result["test_info"][0]
    assert transformed["train_data"]["amt"].iloc[0] == pytest.approx(np.log1p(1.0))
    assert transformed["validation_data"]["amt"].iloc[0] == pytest.approx(np.log1p(2.0))
    assert transformed["test_data"] is None
    assert fold["train_data"]["amt"].iloc[0] == 1.0


def test_log1p_data_splits_without_columns_returns_the_same_object():
    information = {"test_info": []}
    assert log1p_data_splits(information, ()) is information


# --- bloco classico e nativo ---------------------------------------------------


def _write_ulb(root: Path, amounts: np.ndarray) -> None:
    (root / "ref-18").mkdir(parents=True)
    rng = np.random.default_rng(3)
    pd.DataFrame(
        {
            "Time": np.arange(ROW_COUNT) * 60,
            "V1": rng.normal(size=ROW_COUNT),
            "V2": rng.normal(size=ROW_COUNT),
            "Amount": amounts,
            TARGET: [1 if value % 10 == 0 else 0 for value in range(ROW_COUNT)],
        }
    ).to_csv(root / "ref-18" / "creditcard.csv", index=False)


def _prepare(monkeypatch, tmp_path, mode: str) -> list[pd.DataFrame]:
    monkeypatch.setattr(pipeline, "CURRENT_PATH", "")
    monkeypatch.setattr(pipeline, "RELATIVE_DATA_PATH", str(tmp_path))
    output = tmp_path / f"out-{mode}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset", "ulb",
            "--test-strategy", "stratified-holdout",
            "--prediction-column", TARGET,
            "--fill-na-strategy", "none",
            "--categorical-handling-mode", mode,
            "--auxiliary-data-is-test-data", "false",
            "--output-directory", str(output),
        ],
    )
    _md5, created = preparation.prepare_datasets(preparation.parse_arguments())
    return [pd.read_csv(item.path) for item in created]


def test_classic_block_standardizes_the_log1p_of_the_amount(monkeypatch, tmp_path):
    amounts = 1.0 + np.arange(ROW_COUNT) ** 2 * 0.37
    _write_ulb(tmp_path, amounts)
    captured = {}
    original = pipeline.standardize_data_splits

    def spy(split_information, target_column):
        captured["train"] = split_information["test_info"][0]["train_data"].copy()
        return original(split_information, target_column)

    monkeypatch.setattr(pipeline, "standardize_data_splits", spy)
    _prepare(monkeypatch, tmp_path, "encoded")

    seen = captured["train"]["Amount"].to_numpy()
    assert set(np.round(seen, 9)) <= set(np.round(np.log1p(amounts), 9))
    assert not set(np.round(seen, 9)) & set(np.round(amounts, 9))


def test_native_block_keeps_the_original_amount(monkeypatch, tmp_path):
    amounts = 1.0 + np.arange(ROW_COUNT) ** 2 * 0.37
    _write_ulb(tmp_path, amounts)
    frames = _prepare(monkeypatch, tmp_path, "native")
    saved = np.concatenate([frame["Amount"].to_numpy() for frame in frames])
    assert set(np.round(saved, 9)) <= set(np.round(amounts, 9))


def test_negative_amount_stops_the_classic_preparation(monkeypatch, tmp_path):
    amounts = 1.0 + np.arange(ROW_COUNT, dtype=float)
    amounts[5] = -3.0
    _write_ulb(tmp_path, amounts)
    with pytest.raises(NegativeMonetaryValueError, match="Amount.*ulb"):
        _prepare(monkeypatch, tmp_path, "encoded")


# --- Isolation Forest e DevNet ---------------------------------------------------


def _fold_with_amount(rng) -> dict:
    def split(rows: int, anomalies: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "amount": rng.lognormal(3.0, 1.5, size=rows),
                "v1": rng.normal(size=rows),
                TARGET: [0] * (rows - anomalies) + [1] * anomalies,
            }
        )
        return frame

    return {
        "train_test_strategy": "stratified-holdout",
        "test_info": [
            {
                "train_data": split(300, 12),
                "validation_data": split(120, 6),
                "test_data": split(120, 6),
                "fold_index": FIRST_FOLD_INDEX,
            }
        ],
    }


def test_encoder_applies_log1p_before_imputation_and_standardization():
    train = pd.DataFrame({"amount": [0.0, 9.0, np.nan, 99.0]})
    encoder = LabelFreeEncoder.fit(
        train, [], impute_missing=True, standardize=True, monetary_columns=("amount",)
    )
    logged = np.log1p(np.array([0.0, 9.0, 99.0]))
    filled = np.array([logged[0], logged[1], np.median(logged), logged[2]])
    expected = (filled - filled.mean()) / filled.std()
    assert list(encoder.transform(train)["amount"]) == pytest.approx(list(expected))


def test_encoder_without_monetary_columns_is_unchanged():
    train = pd.DataFrame({"amount": [0.0, 9.0, 99.0]})
    encoded = LabelFreeEncoder.fit(train, []).transform(train)
    assert list(encoded["amount"]) == [0.0, 9.0, 99.0]


def test_isolation_forest_receives_the_log1p_amount(monkeypatch):
    rng = np.random.default_rng(5)
    information = _fold_with_amount(rng)
    raw_train = information["test_info"][0]["train_data"]["amount"].to_numpy()
    seen = {}
    original = anomaly.create_isolation_forest

    def recording_forest():
        model = original()
        fit = model.fit

        def recording_fit(matrix, *args, **kwargs):
            seen["matrix"] = matrix.copy()
            return fit(matrix, *args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(anomaly, "create_isolation_forest", recording_forest)
    anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST,
        information,
        TARGET,
        monetary_columns=("amount",),
    )
    assert seen["matrix"]["amount"].to_numpy() == pytest.approx(np.log1p(raw_train))


def test_isolation_forest_without_declaration_keeps_the_raw_amount(monkeypatch):
    rng = np.random.default_rng(5)
    information = _fold_with_amount(rng)
    raw_train = information["test_info"][0]["train_data"]["amount"].to_numpy()
    seen = {}
    original = anomaly.create_isolation_forest

    def recording_forest():
        model = original()
        fit = model.fit

        def recording_fit(matrix, *args, **kwargs):
            seen["matrix"] = matrix.copy()
            return fit(matrix, *args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(anomaly, "create_isolation_forest", recording_forest)
    anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST, information, TARGET
    )
    assert seen["matrix"]["amount"].to_numpy() == pytest.approx(raw_train)


def test_devnet_trains_on_the_standardized_log1p_amount(monkeypatch):
    rng = np.random.default_rng(6)
    information = _fold_with_amount(rng)
    raw_train = information["test_info"][0]["train_data"]["amount"].to_numpy()
    seen = {}
    original = anomaly.train_devnet

    def recording_train(matrix, labeled_indices, settings):
        seen["matrix"] = matrix.copy()
        return original(matrix, labeled_indices, settings)

    monkeypatch.setattr(anomaly, "train_devnet", recording_train)
    fast = anomaly.DevNetSettings(epochs=5, batches_per_epoch=10, batch_size=64)
    fold_data = anomaly.prepare_fold_data(information["test_info"][0], TARGET)
    anomaly.execute_devnet_fold(
        fold_data, label_fraction=0.5, settings=fast, monetary_columns=("amount",)
    )
    logged = np.log1p(raw_train)
    expected = (logged - logged.mean()) / logged.std()
    column = list(fold_data.train_features.columns).index("amount")
    assert seen["matrix"][:, column] == pytest.approx(expected)
