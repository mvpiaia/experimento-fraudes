"""Linha de base nao supervisionada (item G3 da revisao).

Protege tres garantias: nenhum rotulo entra nos atributos, o Isolation Forest
so roda sobre a prevalencia natural, e o escore cresce com a anomalia.
"""

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

import anomaly_detection_specific as anomaly  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
from label_free_encoding import LabelFreeEncoder  # noqa: E402
from prepared_dataset_storage import FIRST_FOLD_INDEX  # noqa: E402


TARGET = "Class"


# ---------------------------------------------------------------------------
# Codificacao sem rotulo
# ---------------------------------------------------------------------------


def test_categories_become_training_frequencies():
    train = pd.DataFrame({"channel": ["web", "web", "web", "pos"], "amount": [1.0, 2.0, 3.0, 4.0]})
    encoder = LabelFreeEncoder.fit(train, ["channel"])

    encoded = encoder.transform(train)

    assert list(encoded["channel"]) == pytest.approx([0.75, 0.75, 0.75, 0.25])
    assert list(encoded["amount"]) == [1.0, 2.0, 3.0, 4.0]


def test_unseen_category_is_zero_and_missing_is_its_own_category():
    train = pd.DataFrame({"channel": ["web", None, None, "pos"]})
    encoder = LabelFreeEncoder.fit(train, ["channel"])

    held_out = pd.DataFrame({"channel": ["atm", None]})
    encoded = encoder.transform(held_out)

    assert encoded.loc[0, "channel"] == 0.0
    assert encoded.loc[1, "channel"] == pytest.approx(0.5)


def test_encoder_never_receives_the_target():
    """O ajuste so aceita atributos: nao ha parametro onde o alvo caiba."""
    import inspect

    parameters = inspect.signature(LabelFreeEncoder.fit).parameters
    assert "target" not in parameters and "y" not in parameters


def test_imputation_and_standardization_use_training_statistics_only():
    train = pd.DataFrame({"amount": [1.0, np.nan, 3.0]})
    test = pd.DataFrame({"amount": [np.nan, 100.0]})
    encoder = LabelFreeEncoder.fit(train, [], impute_missing=True, standardize=True)

    encoded = encoder.transform(test)

    # Mediana do treino (2,0) preenche o teste; media e escala vem do treino.
    train_filled = np.array([1.0, 2.0, 3.0])
    expected = (np.array([2.0, 100.0]) - train_filled.mean()) / train_filled.std()
    assert list(encoded["amount"]) == pytest.approx(list(expected))


def test_missing_indicators_mark_training_columns_with_missing_values():
    train = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "age": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"amount": [np.nan, 5.0], "age": [np.nan, 40.0]})
    encoder = LabelFreeEncoder.fit(
        train, [], impute_missing=True, standardize=True, missing_indicators=True
    )

    encoded = encoder.transform(test)

    # So "amount" tem ausente no treino; "age" e imputada sem indicador.
    assert encoder.missing_indicator_columns == ("amount",)
    assert list(encoded.columns) == ["amount", "age", "amount__is_missing"]
    assert list(encoded["amount__is_missing"]) == [1.0, 0.0]
    assert not encoded.isna().any().any()


def test_without_missing_values_the_indicators_leave_the_input_unchanged():
    """Os bracos imputados na preparacao recebem a mesma entrada de antes."""
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "channel": ["a", None, "b"]})
    plain = LabelFreeEncoder.fit(train, ["channel"], impute_missing=True, standardize=True)
    marked = LabelFreeEncoder.fit(
        train, ["channel"], impute_missing=True, standardize=True, missing_indicators=True
    )
    pd.testing.assert_frame_equal(plain.transform(train), marked.transform(train))


def test_native_arm_differs_from_median_imputation_with_indicators():
    """Sem indicador, deixar o ausente ao DevNet equivalia ao braco median."""
    native = pd.DataFrame({"amount": [1.0, np.nan, 3.0, 4.0]})
    median = native.fillna(native["amount"].median())
    fit = lambda data: LabelFreeEncoder.fit(  # noqa: E731
        data, [], impute_missing=True, standardize=True, missing_indicators=True
    ).transform(data)
    plain = lambda data: LabelFreeEncoder.fit(  # noqa: E731
        data, [], impute_missing=True, standardize=True
    ).transform(data)

    pd.testing.assert_frame_equal(plain(native), plain(median))
    assert fit(native).shape[1] == fit(median).shape[1] + 1


def test_infinite_values_are_treated_as_missing():
    train = pd.DataFrame({"amount": [1.0, np.inf, 3.0]})
    encoded = LabelFreeEncoder.fit(train, []).transform(train)
    assert np.isnan(encoded.loc[1, "amount"])


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


def _split(rows: int, anomalies: int, rng) -> pd.DataFrame:
    normal = rng.normal(0.0, 1.0, size=(rows - anomalies, 2))
    outliers = rng.normal(8.0, 0.5, size=(anomalies, 2))
    frame = pd.DataFrame(np.vstack([normal, outliers]), columns=["v1", "v2"])
    frame["channel"] = ["web"] * (rows - anomalies) + ["rare"] * anomalies
    frame[TARGET] = [0] * (rows - anomalies) + [1] * anomalies
    return frame


def _test_data_information(rng) -> dict:
    return {
        "train_test_strategy": "stratified-holdout",
        "test_info": [
            {
                "train_data": _split(400, 8, rng),
                "validation_data": _split(200, 4, rng),
                "test_data": _split(200, 4, rng),
                "fold_index": FIRST_FOLD_INDEX,
            }
        ],
    }


def test_resampled_configuration_is_rejected():
    with pytest.raises(ValueError, match="without resampling"):
        anomaly.validate_not_resampled("random-undersampling")
    anomaly.validate_not_resampled("none")
    anomaly.validate_not_resampled(None)


def test_anomalies_receive_higher_scores_and_budget_zero():
    rng = np.random.default_rng(0)
    execution = anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST, _test_data_information(rng), TARGET
    )

    fold = execution["results"][0]
    assert fold["label_budget"] == 0
    assert fold["label_fraction"] == 0.0
    test_metrics = fold["split_results"]["test_data"]["metrics"]
    assert test_metrics["roc_auc"] > 0.95
    predictions = fold["split_results"]["test_data"]["predictions"]
    scores_by_class = predictions.groupby("real_value")[anomaly.SCORE_COLUMN].mean()
    assert scores_by_class[1] > scores_by_class[0]


def test_isolation_forest_ignores_training_labels():
    """Trocar os rotulos do treino nao muda nenhum escore."""
    rng = np.random.default_rng(1)
    information = _test_data_information(rng)
    shuffled = {
        **information,
        "test_info": [
            {
                **information["test_info"][0],
                "train_data": information["test_info"][0]["train_data"].assign(
                    **{TARGET: lambda frame: frame[TARGET].sample(frac=1, random_state=3).to_numpy()}
                ),
            }
        ],
    }

    def test_scores(data):
        execution = anomaly.execute_anomaly_algorithm(
            anomaly.ALGORITHM_ISOLATION_FOREST, data, TARGET
        )
        predictions = execution["results"][0]["split_results"]["test_data"]["predictions"]
        return predictions[anomaly.SCORE_COLUMN].to_numpy()

    np.testing.assert_allclose(test_scores(information), test_scores(shuffled))


def test_missing_numeric_values_reach_the_estimator():
    rng = np.random.default_rng(2)
    information = _test_data_information(rng)
    information["test_info"][0]["train_data"].loc[0:20, "v1"] = np.nan

    execution = anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST, information, TARGET
    )
    assert execution["results"][0]["split_results"]["test_data"] is not None


def test_recorded_row_has_budget_zero_and_fixed_hyperparameters(tmp_path):
    rng = np.random.default_rng(0)
    execution = anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST, _test_data_information(rng), TARGET
    )
    destination = tmp_path / "resultados_isolation-forest.csv"
    pipeline.record_algorithm_result_to_csv(
        str(destination), "", execution, "none", "none", 1.0,
        configuration_md5="abc", database_type="eqm-ref-18",
    )

    recorded = pd.read_csv(destination, dtype=str, keep_default_na=False)
    assert set(recorded["label_budget"]) == {"0"}
    assert set(recorded["split"]) == {"train", "validation", "test"}
    hyperparameters = json.loads(recorded.loc[0, "model_hyperparameters"])
    for name, value in anomaly.ISOLATION_FOREST_PARAMETERS.items():
        assert hyperparameters[name] == value
    assert pipeline.is_experiment_already_recorded(
        tmp_path, "abc", "isolation-forest", label_budget=0
    )
    # A retomada do executor identifica o braco pela fracao, 0 sem rotulos.
    assert pipeline.is_experiment_already_recorded(
        tmp_path, "abc", "isolation-forest", label_fraction=0.0
    )


# ---------------------------------------------------------------------------
# Execucao completa a partir de um manifesto preparado
# ---------------------------------------------------------------------------


def prepare_native_configuration(tmp_path, rng, undersampling_technique="none"):
    """Grava uma configuracao nativa como a preparacao faria e devolve os argumentos."""
    from prepared_dataset_storage import build_prepared_dataset_md5, save_prepared_data_splits

    arguments = {
        "database_type": "eqm-ref-18",
        "dataset_filename": "creditcard.csv",
        "prediction_column_name": TARGET,
        "fill_na_strategy": "none",
        "test_strategy": "stratified-holdout",
        "undersampling_technique": undersampling_technique,
    }
    configuration_md5 = build_prepared_dataset_md5(
        arguments["database_type"],
        None,
        None,
        arguments["test_strategy"],
        TARGET,
        None if undersampling_technique == "none" else undersampling_technique,
        None,
        True,
        "native",
        42,
    )
    prepared_directory = tmp_path / "prepared"
    created = save_prepared_data_splits(
        _test_data_information(rng), prepared_directory, configuration_md5
    )
    manifest = pd.DataFrame(
        [
            {
                "database_type": arguments["database_type"],
                "configuration_md5": configuration_md5,
                "dataset_md5": created_file.dataset_md5,
                "content_md5": created_file.content_md5,
                "fold_index": created_file.fold_index,
                "split": created_file.split_label,
                "categorical_handling_mode": "native",
                "categorical_encoding_strategy": "none",
            }
            for created_file in created
        ]
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    command_line = [
        "--algorithm", "isolation-forest",
        "--database-type", arguments["database_type"],
        "--dataset-filename", arguments["dataset_filename"],
        "--prediction-column-name", TARGET,
        "--categorical-encoding-strategy", "none",
        "--fill-na-strategy", arguments["fill_na_strategy"],
        "--undersampling-technique", undersampling_technique,
        "--undersampling-parameter-values", "none",
        "--test-strategy", arguments["test_strategy"],
        "--prepared-data-directory", str(prepared_directory),
        "--prepared-dataset-md5", configuration_md5,
        "--prepared-dataset-manifest", str(manifest_path),
        "--results-directory", str(tmp_path / "results"),
        "--local-staging-directory", str(tmp_path / "staging"),
    ]
    return configuration_md5, command_line


def test_run_configuration_records_and_then_skips(tmp_path, capsys):
    configuration_md5, command_line = prepare_native_configuration(
        tmp_path, np.random.default_rng(4)
    )
    parser = anomaly.build_argument_parser()

    assert anomaly.run_configuration(parser.parse_args(command_line)) == 0
    recorded = pd.read_csv(
        tmp_path / "results" / "resultados_isolation-forest.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert set(recorded["execution_status"]) == {"success"}
    assert set(recorded["configuration_md5"]) == {configuration_md5}

    assert anomaly.run_configuration(parser.parse_args(command_line)) == 0
    assert "Skipping experiment already recorded" in capsys.readouterr().out


def test_run_configuration_refuses_resampled_configuration(tmp_path):
    _, command_line = prepare_native_configuration(
        tmp_path, np.random.default_rng(5), undersampling_technique="random-undersampling"
    )
    with pytest.raises(ValueError, match="without resampling"):
        anomaly.run_configuration(anomaly.build_argument_parser().parse_args(command_line))
