"""Item G17: motivo da falha em cada linha de resultado e particao vazia."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import execution_failures as failures  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402

from test_model_metadata import _record, _execution_result  # noqa: E402


def _failure_row(tmp_path, error):
    destination = tmp_path / "resultados_knn.csv"
    pipeline.record_algorithm_failure_to_csv(
        str(destination), "", "knn", "woe", "median", "stratified-holdout",
        configuration_md5="abc", database_type="eqm-ref-18", error=error,
    )
    return pd.read_csv(destination, dtype=str, keep_default_na=False).iloc[0]


def test_memory_error_is_classified_as_memory(tmp_path):
    row = _failure_row(tmp_path, MemoryError("Unable to allocate 12 GiB"))
    assert row["failure_category"] == failures.FAILURE_MEMORY
    assert row["failure_message"] == "Unable to allocate 12 GiB"


def test_native_library_memory_message_is_classified_as_memory():
    error = RuntimeError("std::bad_alloc while building histogram")
    assert failures.classify_failure(error) == failures.FAILURE_MEMORY


def test_unexpected_error_keeps_type_in_message(tmp_path):
    row = _failure_row(tmp_path, KeyError("x"))
    assert row["failure_category"] == failures.FAILURE_OTHER
    assert row["failure_message"].startswith("KeyError")


def test_convergence_and_storage_categories():
    assert failures.classify_failure(np.linalg.LinAlgError("Singular matrix")) == failures.FAILURE_CONVERGENCE
    assert failures.classify_failure(ValueError("Input contains NaN.")) == failures.FAILURE_CONVERGENCE
    assert (
        failures.classify_failure(pipeline.PreparedDatasetStorageError("HD offline"))
        == failures.FAILURE_STORAGE
    )


def test_message_is_first_line_and_truncated():
    error = MemoryError("a" * 500 + "\nsegunda linha")
    message = failures.describe_failure(error)
    assert len(message) == failures.FAILURE_MESSAGE_MAX_LENGTH
    assert "segunda" not in message


def test_success_rows_leave_failure_columns_empty(tmp_path):
    row = _record(tmp_path, _execution_result()).iloc[0]
    assert row["execution_status"] == "success"
    assert row["failure_category"] == ""
    assert row["failure_message"] == ""


def test_training_without_frauds_fails_before_fitting():
    train = pd.DataFrame({"amount": [1.0, 2.0], "is_fraud": [0, 0]})
    test = pd.DataFrame({"amount": [3.0], "is_fraud": [1]})
    with pytest.raises(failures.EmptyPartitionError) as raised:
        failures.check_fold_partitions(
            {"train_data": train, "validation_data": None, "test_data": test}, "is_fraud", 2
        )
    assert "train" in str(raised.value) and "classe unica" in str(raised.value)
    assert failures.classify_failure(raised.value) == failures.FAILURE_EMPTY_PARTITION


def test_empty_test_split_fails_before_fitting():
    train = pd.DataFrame({"amount": [1.0, 2.0], "is_fraud": [0, 1]})
    with pytest.raises(failures.EmptyPartitionError, match="test sem linhas"):
        failures.check_loaded_folds(
            {"test_info": [{"fold_index": 1, "train_data": train, "validation_data": None,
                            "test_data": train.iloc[0:0]}]},
            "is_fraud",
        )


def test_complete_fold_passes():
    frame = pd.DataFrame({"amount": [1.0, 2.0], "is_fraud": [0, 1]})
    failures.check_fold_partitions({"train": frame, "validation": frame, "test": frame}, "is_fraud")
