"""Metadados do modelo e orcamento de rotulos em cada linha de resultado.

Protege a proposta extend-result-schema-model-metadata: sem os hiperparametros
efetivos, dois resultados diferentes nao podem ser explicados depois; sem o
orcamento de rotulos, o eixo "disponibilidade de rotulos" do item G3 nao tem
onde ser lido.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from torch import nn

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import lr_nb_knn_specific as pipeline  # noqa: E402
import model_metadata  # noqa: E402
from prepared_dataset_storage import FIRST_FOLD_INDEX  # noqa: E402


def _fitted_logistic_regression():
    features = np.array([[0.0], [0.1], [0.9], [1.0]])
    target = np.array([0, 0, 1, 1])
    return LogisticRegression(max_iter=2000, random_state=42).fit(features, target)


def test_scikit_learn_hyperparameters_and_training_summary():
    model = _fitted_logistic_regression()

    hyperparameters = model_metadata.describe_model_hyperparameters(model)
    summary = model_metadata.summarize_training(model)

    assert hyperparameters["max_iter"] == 2000
    assert hyperparameters["random_state"] == 42
    assert summary["n_features_in_"] == 1
    assert "n_iter_" in summary


def test_isolation_forest_summary_records_effective_subsample():
    features = np.random.default_rng(0).normal(size=(50, 2))
    model = IsolationForest(n_estimators=10, max_samples=256, random_state=42).fit(features)

    summary = model_metadata.summarize_training(model)

    # max_samples=256 pedido, mas a base tem 50 linhas: o efetivo e 50.
    assert summary["max_samples_"] == 50
    assert summary["estimator_count"] == 10


class _CatBoostLike:
    """Imita o CatBoost: get_params() so com o que foi passado."""

    def get_params(self, deep=False):
        return {"random_seed": 42}

    def get_all_params(self):
        return {"random_seed": 42, "learning_rate": 0.0412, "depth": 6}


def test_catboost_resolved_defaults_are_preferred():
    hyperparameters = model_metadata.describe_model_hyperparameters(_CatBoostLike())
    assert hyperparameters["learning_rate"] == pytest.approx(0.0412)
    assert hyperparameters["depth"] == 6


class _BoosterLike:
    def save_config(self):
        return json.dumps(
            {
                "learner": {
                    "learner_train_param": {"objective": "binary:logistic"},
                    "gradient_booster": {
                        "tree_train_param": {"max_depth": "6", "eta": "0.3"},
                    },
                }
            }
        )


class _XGBoostLike:
    """Imita o XGBoost: get_params() devolve None nos padroes."""

    def get_params(self, deep=False):
        return {"max_depth": None, "n_jobs": 1, "unknown_default": None}

    def get_booster(self):
        return _BoosterLike()


def test_xgboost_none_parameters_are_filled_from_the_booster():
    hyperparameters = model_metadata.describe_model_hyperparameters(_XGBoostLike())
    assert hyperparameters["max_depth"] == "6"
    assert hyperparameters["n_jobs"] == 1
    # Sem valor efetivo conhecido, o None permanece: nada e inventado.
    assert hyperparameters["unknown_default"] is None


class _TinyModule(nn.Module):
    def __init__(self, input_size: int, hidden: int = 20, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden)


def test_torch_constructor_includes_defaults_and_explicit_values():
    arguments = model_metadata.describe_torch_constructor(_TinyModule, hidden=8)
    assert arguments == {"hidden": 8, "dropout": 0.1}
    assert model_metadata.count_trainable_parameters(_TinyModule(3, hidden=8)) == 3 * 8 + 8


def test_serialization_is_deterministic_and_json_valid():
    values = {"b": np.int64(3), "a": np.array([1.5, 2.0]), "c": float("nan"), "d": object}
    serialized = model_metadata.serialize_metadata(values)

    assert serialized == model_metadata.serialize_metadata(dict(reversed(values.items())))
    decoded = json.loads(serialized)
    assert list(decoded) == ["a", "b", "c", "d"]
    assert decoded["b"] == 3 and decoded["a"] == [1.5, 2.0] and decoded["c"] == "nan"


# ---------------------------------------------------------------------------
# Gravacao
# ---------------------------------------------------------------------------


def _execution_result(**fold_extra):
    metrics = {
        "f1": 0.5, "f_beta_2": 0.5, "accuracy": 0.9, "precision": 0.5,
        "recall": 0.5, "roc_auc": 0.8, "auc_pr": 0.4, "false_positive_rate": 0.1,
    }
    return {
        "algorithm": "logistic-regression",
        "undersampling_technique": None,
        "train_test_strategy": "stratified-holdout",
        "results": [
            {
                "fold_index": FIRST_FOLD_INDEX,
                "split_sizes": {"train_data": 4, "validation_data": 0, "test_data": 0},
                "split_results": {"test_data": {"metrics": metrics}},
                **fold_extra,
            }
        ],
    }


def _record(tmp_path, execution_result, algorithm="logistic-regression"):
    destination = tmp_path / f"resultados_{algorithm}.csv"
    pipeline.record_algorithm_result_to_csv(
        str(destination), "", execution_result, "none", "mean", 1.0,
        configuration_md5="abc", database_type="eqm-ref-18",
    )
    return pd.read_csv(destination, dtype=str, keep_default_na=False)


def test_supervised_row_records_minus_one_and_model_metadata(tmp_path):
    recorded = _record(
        tmp_path, _execution_result(model=_fitted_logistic_regression())
    )

    row = recorded.iloc[0]
    assert row["dataset_id"] == "ulb"
    assert row["dataset_name"] == "ULB"
    assert row["database_type"] == "ulb"
    assert row["label_budget"] == "-1"
    assert json.loads(row["model_hyperparameters"])["max_iter"] == 2000
    assert json.loads(row["training_summary"])["n_features_in_"] == 1


def test_explicit_budget_and_metadata_override_the_model(tmp_path):
    recorded = _record(
        tmp_path,
        _execution_result(
            label_budget=30,
            model_hyperparameters={"hidden_units": 20},
            training_summary={"epochs": 50},
        ),
    )

    row = recorded.iloc[0]
    assert row["label_budget"] == "30"
    assert json.loads(row["model_hyperparameters"]) == {"hidden_units": 20}
    assert json.loads(row["training_summary"]) == {"epochs": 50}


def test_failure_row_records_budget_and_empty_metadata(tmp_path):
    destination = tmp_path / "resultados_devnet.csv"
    pipeline.record_algorithm_failure_to_csv(
        str(destination), "", "devnet", "none", "none", "stratified-holdout",
        configuration_md5="abc", database_type="eqm-ref-18", label_budget=10,
    )
    row = pd.read_csv(destination, dtype=str, keep_default_na=False).iloc[0]
    assert row["dataset_id"] == "ulb"
    assert row["dataset_name"] == "ULB"
    assert row["database_type"] == "ulb"
    assert row["label_budget"] == "10"
    assert row["model_hyperparameters"] == "{}"
    assert row["training_summary"] == "{}"


def test_resume_key_distinguishes_label_budgets(tmp_path):
    _record(tmp_path, _execution_result(label_budget=10))

    def recorded(budget):
        return pipeline.is_experiment_already_recorded(
            tmp_path, "abc", "logistic-regression", label_budget=budget
        )

    assert recorded(10)
    assert not recorded(30)
    # Sem orcamento pedido, o comportamento anterior nao muda.
    assert pipeline.is_experiment_already_recorded(tmp_path, "abc", "logistic-regression")


def test_legacy_rows_without_label_budget_count_as_not_applicable(tmp_path):
    legacy = pd.DataFrame(
        [{"configuration_md5": "abc", "algorithm": "knn", "execution_status": "success"}]
    )
    legacy.to_csv(tmp_path / "resultados_knn.csv", index=False)

    assert pipeline.is_experiment_already_recorded(tmp_path, "abc", "knn", label_budget=-1)
    assert not pipeline.is_experiment_already_recorded(tmp_path, "abc", "knn", label_budget=0)


def test_legacy_csv_is_migrated_with_empty_metadata_columns(tmp_path):
    destination = tmp_path / "resultados_logistic-regression.csv"
    old_columns = [
        column
        for column in pipeline.RESULT_CSV_COLUMNS
        if column not in ("label_budget", "model_hyperparameters", "training_summary")
    ]
    pd.DataFrame([{column: "" for column in old_columns}]).to_csv(destination, index=False)

    recorded = _record(tmp_path, _execution_result(label_budget=-1))

    assert list(recorded.columns) == pipeline.RESULT_CSV_COLUMNS
    assert recorded.loc[0, "label_budget"] == ""
    assert recorded.loc[1, "label_budget"] == "-1"


def test_boosting_fold_carries_effective_hyperparameters():
    """O boosting nao guarda o modelo na dobra: os metadados saem extraidos."""
    pytest.importorskip("catboost")
    import catboost_xgboost_specific as boosting

    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"amount": rng.normal(size=60), "channel": rng.choice(["web", "pos"], size=60)}
    )
    target = pd.Series((features["amount"] > 0).astype(int))
    fold_data = boosting.FoldData(
        train_features=features,
        train_target=target,
        validation_features=None,
        validation_target=None,
        test_features=features,
        test_target=target,
        categorical_columns=["channel"],
        class_values=[0, 1],
        split_sizes={"train_data": 60, "validation_data": 0, "test_data": 60},
        fold_index=FIRST_FOLD_INDEX,
        split_md5={},
        partition_md5={},
    )

    fold_result = boosting.execute_catboost_fold(fold_data)

    # O learning_rate nao foi passado: e o padrao que o CatBoost resolveu.
    assert fold_result["model_hyperparameters"]["learning_rate"] > 0
    assert fold_result["training_summary"]["tree_count_"] > 0
